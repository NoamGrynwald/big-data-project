#!/usr/bin/env python3
"""
Enhanced Real-Time Fraud Detection System
Processes transactions from Kafka and detects fraud using ensemble models with time-aware features
"""

import json
import os
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from kafka import KafkaConsumer, KafkaProducer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

class Config:
    """Centralized configuration"""
    # Model paths
    MODEL_DIR = Path("/app/models")
    RF_PATH = MODEL_DIR / "random_forest.joblib"
    ET_PATH = MODEL_DIR / "extra_trees.joblib"
    SVC_PATH = MODEL_DIR / "svc_rbf.joblib"
    
    # Meta model candidates (in order of preference)
    META_CANDIDATES = [
        "decision_tree_meta.joblib",
        "meta_model.joblib",
        "decision_tree.joblib",
        "dt_meta.joblib"
    ]
    
    # Kafka settings
    KAFKA_BOOTSTRAP_SERVERS = ['kafka:29092']
    INPUT_TOPIC = 'creditcard-data'
    OUTPUT_TOPIC = 'fraud-alerts'
    CONSUMER_GROUP = 'realtime-fraud-detector'
    CONSUMER_TIMEOUT = 5000
    CONNECTION_RETRIES = 60
    
    # Feature columns (must match training order)
    RAW_FEATURES = [
        "Amount", "Time", "V1", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
        "V17", "V18", "V19", "V2", "V20", "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28", "V3", "V4", "V5", "V6", "V7", "V8", "V9"
    ]
    
    # Time feature parameters
    TIME_NORMALIZATION_FACTOR = 172800.0  # 48 hours
    TIME_BUCKET_SIZE = 10000
    TIME_BUCKET_MAX = 17
    SECONDS_IN_DAY = 86400
    TIME_PROGRESSION_FACTOR = 50000
    TIME_PROGRESSION_CAP = 2.0
    
    # Reporting intervals
    PROGRESS_INTERVAL = 1000
    SUMMARY_INTERVAL = 10000
    HEARTBEAT_INTERVAL = 30  # seconds


# ================================================================
# PERFORMANCE METRICS
# ================================================================

class MetricsTracker:
    """Track model performance and statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = defaultdict(int)  # tp, fp, tn, fn
        self.transaction_count = 0
        self.fraud_alerts = 0
        self.total_fraud_amount = 0.0
        self.start_time = time.time()
        self.time_bucket_stats = defaultdict(lambda: defaultdict(int))
    
    def update_confusion_matrix(self, actual, predicted):
        """Update confusion matrix"""
        if actual == 1 and predicted == 1:
            self.confusion_matrix['tp'] += 1
        elif actual == 0 and predicted == 1:
            self.confusion_matrix['fp'] += 1
        elif actual == 1 and predicted == 0:
            self.confusion_matrix['fn'] += 1
        else:
            self.confusion_matrix['tn'] += 1
    
    def record_transaction(self, actual, predicted, amount, time_bucket):
        """Record a transaction"""
        self.transaction_count += 1
        self.update_confusion_matrix(actual, predicted)
        
        if predicted == 1:
            self.fraud_alerts += 1
            self.total_fraud_amount += amount
        
        # Track performance by time bucket
        self.time_bucket_stats[time_bucket]['total'] += 1
        if actual == 1:
            self.time_bucket_stats[time_bucket]['actual_fraud'] += 1
        if predicted == 1:
            self.time_bucket_stats[time_bucket]['predicted_fraud'] += 1
        if actual == 1 and predicted == 1:
            self.time_bucket_stats[time_bucket]['correct_fraud'] += 1
    
    def get_metrics(self):
        """Calculate current metrics"""
        cm = self.confusion_matrix
        tp, fp, tn, fn = cm['tp'], cm['fp'], cm['tn'], cm['fn']
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'transactions': self.transaction_count,
            'alerts': self.fraud_alerts,
            'fraud_amount': self.total_fraud_amount,
            'runtime': time.time() - self.start_time,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'confusion_matrix': dict(cm)
        }
    
    def print_summary(self):
        """Print comprehensive performance summary"""
        metrics = self.get_metrics()
        cm = metrics['confusion_matrix']
        
        print(f"\n{'=' * 80}")
        print("🎯 FRAUD DETECTION PERFORMANCE SUMMARY")
        print(f"{'=' * 80}")
        print(f"📊 Overall Statistics:")
        print(f"   Total Transactions: {metrics['transactions']:,}")
        print(f"   Fraud Alerts Sent:  {metrics['alerts']:,}")
        print(f"   Total Fraud Amount: ${metrics['fraud_amount']:,.2f}")
        print(f"   Runtime:           {metrics['runtime']/60:.1f} minutes")
        print()
        
        print("🎭 Confusion Matrix:")
        print(f"                    Predicted")
        print(f"              Normal    Fraud")
        print(f"Actual Normal  {cm.get('tn', 0):6d}   {cm.get('fp', 0):6d}")
        print(f"       Fraud   {cm.get('fn', 0):6d}   {cm.get('tp', 0):6d}")
        print()
        
        print("📈 Performance Metrics:")
        print(f"   Precision:  {metrics['precision']:.3f}")
        print(f"   Recall:     {metrics['recall']:.3f}")
        print(f"   Accuracy:   {metrics['accuracy']:.3f}")
        print(f"   F1-Score:   {metrics['f1_score']:.3f}")
        
        # Time bucket performance
        if self.time_bucket_stats:
            print(f"\n🕒 Performance by Time Bucket:")
            print(f"   Bucket | Transactions | Actual Fraud | Detected | Recall")
            print(f"   {'-'*6}|{'-'*13}|{'-'*13}|{'-'*9}|{'-'*7}")
            
            for bucket in sorted(self.time_bucket_stats.keys()):
                stats = self.time_bucket_stats[bucket]
                total = stats['total']
                actual = stats['actual_fraud']
                detected = stats['correct_fraud']
                recall = detected / actual if actual > 0 else 0
                
                print(f"   {bucket:6d}|{total:12,}|{actual:12,}|{detected:8,}|{recall:6.1%}")
        
        print(f"{'=' * 80}\n")


# ================================================================
# FEATURE ENGINEERING
# ================================================================

class FeatureEngineer:
    """Handle feature extraction and engineering"""
    
    def __init__(self):
        self.raw_features = Config.RAW_FEATURES
        self.expected_feature_count = None
        self.feature_info = self.load_feature_info()
        
    def load_feature_info(self):
        """Load feature information from training"""
        feature_info_path = Config.MODEL_DIR / "feature_info.json"
        if feature_info_path.exists():
            try:
                with open(feature_info_path, 'r') as f:
                    info = json.load(f)
                    self.expected_feature_count = len(info.get('columns', []))
                    print(f"📋 Loaded feature info: {self.expected_feature_count} expected features")
                    print(f"   Time features enabled: {info.get('has_time_features', False)}")
                    return info
            except Exception as e:
                print(f"⚠️ Could not load feature_info.json: {e}")
        
        print("⚠️ No feature_info.json found, using default configuration")
        self.expected_feature_count = 36  # 30 raw + 6 time features (FIXED!)
        return {}
    
    def extract_raw_features(self, transaction_data):
        """Extract raw features in correct order"""
        features = []
        for col in self.raw_features:
            val = transaction_data.get(col, 0.0)
            try:
                features.append(float(val))
            except (ValueError, TypeError):
                features.append(0.0)
        
        print(f"🔧 Extracted {len(features)} raw features") if len(features) != 30 else None
        return features
    
    def create_time_features(self, time_val):
        """Create time-based features matching training exactly"""
        time_val = float(time_val)
        
        # Normalization features (matching model creator exactly)
        time_normalized = time_val / Config.TIME_NORMALIZATION_FACTOR
        time_bucket = int(time_val // Config.TIME_BUCKET_SIZE)
        time_bucket_normalized = time_bucket / Config.TIME_BUCKET_MAX
        
        # Cyclical features for daily patterns
        time_of_day = time_val % Config.SECONDS_IN_DAY
        time_sin = np.sin(2 * np.pi * time_of_day / Config.SECONDS_IN_DAY)
        time_cos = np.cos(2 * np.pi * time_of_day / Config.SECONDS_IN_DAY)
        
        # Progression feature (capped at 2.0)
        time_progression = min(time_val / Config.TIME_PROGRESSION_FACTOR, Config.TIME_PROGRESSION_CAP)
        
        # FIXED: Return 6 features including time_bucket (was missing!)
        time_features = [
            time_normalized, 
            time_bucket, 
            time_bucket_normalized, 
            time_sin, 
            time_cos, 
            time_progression
        ]
        
        print(f"🕒 Created {len(time_features)} time features") if len(time_features) != 6 else None
        
        return time_features
    
    def extract_all_features(self, transaction_data):
        """Extract complete feature vector with validation"""
        # Get raw features
        raw_features = self.extract_raw_features(transaction_data)
        
        # Get time features
        time_val = transaction_data.get('Time', 0.0)
        time_features = self.create_time_features(time_val)
        
        # Combine all features
        all_features = raw_features + time_features
        
        # Validate feature count
        actual_count = len(all_features)
        if self.expected_feature_count and actual_count != self.expected_feature_count:
            print(f"⚠️ FEATURE MISMATCH: Expected {self.expected_feature_count}, got {actual_count}")
            print(f"   Raw features: {len(raw_features)} (expected 30)")
            print(f"   Time features: {len(time_features)} (expected 6)")  # Changed from 5 to 6
            print(f"   Raw feature names: {self.raw_features[:5]}...")
            
            # Try to fix common issues
            if actual_count < self.expected_feature_count:
                # Pad with zeros
                padding_needed = self.expected_feature_count - actual_count
                all_features.extend([0.0] * padding_needed)
                print(f"   🔧 Padded with {padding_needed} zeros")
            elif actual_count > self.expected_feature_count:
                # Truncate
                all_features = all_features[:self.expected_feature_count]
                print(f"   🔧 Truncated to {self.expected_feature_count} features")
        
        # Final validation
        final_count = len(all_features)
        if self.expected_feature_count and final_count != self.expected_feature_count:
            print(f"❌ STILL MISMATCHED: Expected {self.expected_feature_count}, final {final_count}")
        
        time_bucket = int(float(time_val) // Config.TIME_BUCKET_SIZE)
        return np.array(all_features).reshape(1, -1), time_bucket


# ================================================================
# FRAUD PREDICTOR
# ================================================================

class FraudPredictor:
    """Handle fraud prediction with ensemble models"""
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.feature_engineer = FeatureEngineer()
        self.load_models()
    
    def load_models(self):
        """Load all models with feature validation"""
        print("🔥 Loading fraud detection models...")
        
        # Load base models
        model_files = {
            'random_forest': Config.RF_PATH,
            'extra_trees': Config.ET_PATH, 
            'svc': Config.SVC_PATH
        }
        
        for name, path in model_files.items():
            try:
                model = load(path)
                self.models[name] = model
                
                # Check expected feature count for models that have it
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                    print(f"   ✅ Loaded {name} (expects {expected_features} features)")
                    
                    # Update expected feature count if not set
                    if not hasattr(self.feature_engineer, 'expected_feature_count') or self.feature_engineer.expected_feature_count is None:
                        self.feature_engineer.expected_feature_count = expected_features
                else:
                    print(f"   ✅ Loaded {name}")
                    
            except Exception as e:
                print(f"   ❌ Failed to load {name}: {e}")
                raise
        
        # Load meta model
        for candidate in Config.META_CANDIDATES:
            meta_path = Config.MODEL_DIR / candidate
            if meta_path.exists():
                try:
                    model = load(meta_path)
                    if hasattr(model, 'predict'):
                        self.meta_model = model
                        print(f"   ✅ Loaded meta model: {candidate}")
                        break
                except Exception as e:
                    print(f"   ⚠️ Failed to load {candidate}: {e}")
        
        if self.meta_model is None:
            print("   ⚠️ No meta model found, using soft voting")
        
        # Final feature count validation
        if hasattr(self.feature_engineer, 'expected_feature_count') and self.feature_engineer.expected_feature_count:
            print(f"   🎯 Target feature count: {self.feature_engineer.expected_feature_count}")
        
        print("✅ All models loaded successfully")
    
    def _get_probability(self, model, X):
        """Get probability of fraud (class 1) from model"""
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                # Handle different class ordering
                if hasattr(model, 'classes_'):
                    try:
                        fraud_idx = list(model.classes_).index(1)
                    except ValueError:
                        fraud_idx = 1 if proba.shape[1] > 1 else 0
                else:
                    fraud_idx = 1 if proba.shape[1] > 1 else 0
                return float(proba[0, fraud_idx])
            
            elif hasattr(model, 'decision_function'):
                # SVM decision function - convert to probability
                decision = float(model.decision_function(X)[0])
                return 1.0 / (1.0 + np.exp(-decision))
            
            else:
                # Fallback to binary prediction
                return float(model.predict(X)[0])
                
        except Exception as e:
            print(f"   ⚠️ Error getting probability: {e}")
            return 0.0
    
    def _get_adaptive_threshold(self, time_bucket, avg_probability):
        """Get time-adaptive threshold based on concept drift analysis"""
        # Time-based threshold adaptation
        if time_bucket < 5:
            base_threshold = 0.3  # Early periods - standard threshold
        elif time_bucket < 10:
            base_threshold = 0.25  # Middle periods - slightly lower
        else:
            base_threshold = 0.15  # Later periods - much lower for drift compensation
        
        return base_threshold
    
    def predict(self, transaction_data):
        """Predict fraud for a transaction with feature validation"""
        try:
            # Extract features with validation
            X, time_bucket = self.feature_engineer.extract_all_features(transaction_data)
            
            # Debug: Print feature info for first few transactions
            if not hasattr(self, '_debug_printed'):
                print(f"🔍 DEBUG: Feature shape: {X.shape}")
                print(f"   Expected: {self.feature_engineer.expected_feature_count} features")
                print(f"   Actual: {X.shape[1]} features")
                print(f"   First 5 features: {X[0, :5].tolist()}")
                print(f"   Last 5 features: {X[0, -5:].tolist()}")
                self._debug_printed = True
            
            # Get base model predictions
            probabilities = {}
            for name, model in self.models.items():
                prob = self._get_probability(model, X)
                probabilities[name] = prob
            
            # Calculate ensemble prediction
            rf_prob = probabilities.get('random_forest', 0.0)
            et_prob = probabilities.get('extra_trees', 0.0)
            svc_prob = probabilities.get('svc', 0.0)
            avg_prob = (rf_prob + et_prob + svc_prob) / 3.0
            
            # Get adaptive threshold
            threshold = self._get_adaptive_threshold(time_bucket, avg_prob)
            
            # Make final prediction
            if self.meta_model is not None:
                # Use meta model for final decision
                X_meta = np.array([rf_prob, et_prob, svc_prob]).reshape(1, -1)
                meta_prediction = int(self.meta_model.predict(X_meta)[0])
                
                # Combine meta prediction with threshold-based override
                if meta_prediction == 1 or avg_prob >= threshold:
                    prediction = 1
                else:
                    prediction = 0
            else:
                # Soft voting with adaptive threshold
                prediction = int(avg_prob >= threshold)
            
            # Return prediction and metadata
            return prediction, {
                'probabilities': probabilities,
                'average_probability': avg_prob,
                'threshold': threshold,
                'time_bucket': time_bucket,
                'meta_used': self.meta_model is not None,
                'feature_count': X.shape[1]
            }
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            # Print more debug info on error
            try:
                X, time_bucket = self.feature_engineer.extract_all_features(transaction_data)
                print(f"   🔍 Debug - Feature shape: {X.shape}")
                print(f"   🔍 Debug - Expected: {self.feature_engineer.expected_feature_count}")
            except Exception as e2:
                print(f"   🔍 Debug extraction also failed: {e2}")
            
            return 0, {
                'probabilities': {},
                'average_probability': 0.0,
                'threshold': 0.3,
                'time_bucket': 0,
                'error': str(e)
            }


# ================================================================
# KAFKA HANDLER
# ================================================================

class KafkaHandler:
    """Handle Kafka connections and messaging"""
    
    def __init__(self):
        self.consumer = None
        self.producer = None
        self.setup_connections()
    
    def setup_connections(self):
        """Setup Kafka consumer and producer"""
        print("🔌 Setting up Kafka connections...")
        
        for attempt in range(Config.CONNECTION_RETRIES):
            try:
                # Setup consumer
                self.consumer = KafkaConsumer(
                    Config.INPUT_TOPIC,
                    bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                    group_id=Config.CONSUMER_GROUP,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    consumer_timeout_ms=Config.CONSUMER_TIMEOUT,
                    max_poll_records=100,
                    fetch_min_bytes=1,
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=3000
                )
                
                # Setup producer
                self.producer = KafkaProducer(
                    bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                
                # Test connection
                self.consumer.poll(timeout_ms=1000)
                print("✅ Kafka connected successfully")
                return
                
            except Exception as e:
                print(f"   ⏳ Connection attempt {attempt + 1}/{Config.CONNECTION_RETRIES}: {e}")
                time.sleep(2)
        
        raise Exception("❌ Could not connect to Kafka after maximum retries")
    
    def send_alert(self, alert_data):
        """Send fraud alert"""
        try:
            self.producer.send(Config.OUTPUT_TOPIC, value=alert_data)
        except Exception as e:
            print(f"   ⚠️ Failed to send alert: {e}")
    
    def close(self):
        """Close connections"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.close()


# ================================================================
# MAIN FRAUD DETECTION SYSTEM
# ================================================================

class FraudDetectionSystem:
    """Main fraud detection system"""
    
    def __init__(self):
        self.predictor = FraudPredictor()
        self.kafka_handler = KafkaHandler()
        self.metrics = MetricsTracker()
        self.last_heartbeat = time.time()
    
    def parse_class_value(self, class_value):
        """Parse class value from various formats"""
        try:
            if isinstance(class_value, str):
                # Handle quoted strings like "'0'" or "'1'"
                return int(class_value.strip("'\""))
            return int(class_value)
        except (ValueError, TypeError):
            return 0
    
    def process_transaction(self, transaction_data):
        """Process a single transaction"""
        try:
            # Skip end markers
            if isinstance(transaction_data, dict) and "__END_OF_STREAM__" in str(transaction_data):
                print("🏁 End of stream marker received")
                return False
            
            # Make prediction
            prediction, metadata = self.predictor.predict(transaction_data)
            
            # Get transaction details
            actual_class = self.parse_class_value(transaction_data.get('Class', 0))
            amount = float(transaction_data.get('Amount', 0))
            time_val = float(transaction_data.get('Time', 0))
            time_bucket = metadata.get('time_bucket', 0)
            
            # Record metrics
            self.metrics.record_transaction(actual_class, prediction, amount, time_bucket)
            
            # Handle actual fraud cases
            if actual_class == 1:
                status = "✅ DETECTED" if prediction == 1 else "❌ MISSED"
                threshold = metadata.get('threshold', 0.3)
                avg_prob = metadata.get('average_probability', 0.0)
                print(f"💳 ACTUAL FRAUD: ${amount:.2f} | {status} | "
                      f"Time: {time_val:.0f} | Bucket: {time_bucket} | "
                      f"Threshold: {threshold:.2f} | Prob: {avg_prob:.3f}")
            
            # Send fraud alert if predicted
            if prediction == 1:
                alert_data = {
                    'timestamp': datetime.now().isoformat(),
                    'transaction_id': transaction_data.get('id', f"tx_{int(time.time())}"),
                    'amount': amount,
                    'actual_class': actual_class,
                    'predicted_fraud': True,
                    'metadata': metadata,
                    'time_value': time_val
                }
                
                self.kafka_handler.send_alert(alert_data)
                
                # Log alert
                alert_type = "TRUE POSITIVE" if actual_class == 1 else "FALSE POSITIVE"
                threshold = metadata.get('threshold', 0.3)
                avg_prob = metadata.get('average_probability', 0.0)
                alert_num = self.metrics.fraud_alerts
                
                print(f"🚨 FRAUD ALERT #{alert_num}: ${amount:.2f} | {alert_type} | "
                      f"Time: {time_val:.0f} | T: {threshold:.2f} | Prob: {avg_prob:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing transaction: {e}")
            return True  # Continue processing
    
    def run(self):
        """Main processing loop"""
        print("🚀 Starting Enhanced Fraud Detection System...")
        print(f"   Listening on topic: {Config.INPUT_TOPIC}")
        print(f"   Sending alerts to: {Config.OUTPUT_TOPIC}")
        print("⏳ Waiting for transactions...\n")
        
        message_count = 0
        last_message_time = time.time()
        
        try:
            while True:
                # Poll for messages
                message_batch = self.kafka_handler.consumer.poll(timeout_ms=Config.CONSUMER_TIMEOUT)
                
                if not message_batch:
                    # Handle no messages
                    current_time = time.time()
                    if current_time - self.last_heartbeat > Config.HEARTBEAT_INTERVAL:
                        print(f"💓 System running - Processed: {message_count:,} | "
                              f"Alerts: {self.metrics.fraud_alerts:,} | "
                              f"Runtime: {(current_time - self.metrics.start_time)/60:.1f}m")
                        self.last_heartbeat = current_time
                    continue
                
                # Process message batch
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        message_count += 1
                        
                        # Process transaction
                        should_continue = self.process_transaction(message.value)
                        if not should_continue:
                            print("🏁 Processing completed")
                            return
                        
                        # Progress reporting
                        if message_count % Config.PROGRESS_INTERVAL == 0:
                            print(f"📊 Processed {message_count:,} transactions "
                                  f"(Alerts: {self.metrics.fraud_alerts:,})")
                        
                        # Detailed summary
                        if message_count % Config.SUMMARY_INTERVAL == 0:
                            self.metrics.print_summary()
                        
                        last_message_time = time.time()
                        self.last_heartbeat = last_message_time
        
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
        except Exception as e:
            print(f"💥 Fatal error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🧹 Cleaning up...")
        self.metrics.print_summary()
        self.kafka_handler.close()
        print("✅ Shutdown complete")


# ================================================================
# MAIN ENTRY POINT
# ================================================================

def main():
    """Main entry point"""
    try:
        fraud_system = FraudDetectionSystem()
        fraud_system.run()
    except Exception as e:
        print(f"💥 System initialization failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()