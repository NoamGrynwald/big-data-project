#!/usr/bin/env python3

import json
import os
import time
from collections import defaultdict
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global metrics tracking
CONFUSION_MATRIX = defaultdict(int)  # (actual_class, predicted_class) -> count
ALERT_STATS = defaultdict(int)       # Track alerts sent
TOTAL_TRANSACTIONS = 0
FRAUD_ALERTS_SENT = 0
MESSAGE_COUNT = 0  # Track total messages for 10k intervals

def wait_for_kafka(bootstrap_servers, max_retries=30, retry_delay=2):
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                consumer_timeout_ms=1000
            )
            consumer.close()
            logger.info("Kafka is available.")
            return True
        except NoBrokersAvailable:
            if attempt < 5:
                logger.info(f"Waiting for Kafka... (attempt {attempt + 1})")
            time.sleep(retry_delay)
        except Exception as e:
            logger.info(f"Kafka connection attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
    return False

def create_consumer(bootstrap_servers, topic, group_id):
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        max_poll_records=500,
        fetch_min_bytes=4096
    )

def update_stats(actual_class, predicted_class, alerted):
    """Update confusion matrix and alert statistics"""
    global CONFUSION_MATRIX, ALERT_STATS, TOTAL_TRANSACTIONS, FRAUD_ALERTS_SENT
    
    # Convert to integers for consistency
    actual = int(actual_class) if actual_class not in ['?', 'N/A', 'unknown', None] else None
    predicted = int(predicted_class) if predicted_class not in ['?', 'N/A', 'unknown', None] else None
    
    if actual is not None and predicted is not None:
        CONFUSION_MATRIX[(actual, predicted)] += 1
        TOTAL_TRANSACTIONS += 1
    
    if alerted:
        FRAUD_ALERTS_SENT += 1
        if actual is not None:
            ALERT_STATS[(actual, 1)] += 1  # (actual_class, alerted=1)

def print_summary_table():
    """Print fraud detection summary table every 10k messages"""
    print("\n" + "="*60)
    print(f"📊 FRAUD DETECTION SUMMARY (Messages: {MESSAGE_COUNT:,})")
    print("="*60)
    
    if TOTAL_TRANSACTIONS == 0:
        print("⌛ No transactions processed with valid labels yet")
        print("="*60)
        return
    
    # Get confusion matrix values
    tp = CONFUSION_MATRIX.get((1, 1), 0)  # Fraud marked as fraud
    tn = CONFUSION_MATRIX.get((0, 0), 0)  # Not fraud marked as not fraud
    fp = CONFUSION_MATRIX.get((0, 1), 0)  # Not fraud marked as fraud
    fn = CONFUSION_MATRIX.get((1, 0), 0)  # Fraud marked as not fraud
    
    # Create summary table
    print("┌─────────────────────┬──────────────┬────────────────┐")
    print("│                     │ Marked Fraud │ Marked Not     │")
    print("│                     │              │ Fraud          │")
    print("├─────────────────────┼──────────────┼────────────────┤")
    print(f"│ Actual Fraud        │    {tp:6d}    │     {fn:6d}     │")
    print(f"│ Actual Not Fraud    │    {fp:6d}    │     {tn:6d}     │")
    print("└─────────────────────┴──────────────┴────────────────┘")
    
    # Calculate metrics
    total_actual_fraud = tp + fn
    total_actual_not_fraud = tn + fp
    total_marked_fraud = tp + fp
    total_marked_not_fraud = tn + fn
    
    accuracy = ((tp + tn) / TOTAL_TRANSACTIONS) if TOTAL_TRANSACTIONS > 0 else 0.0
    precision = (tp / total_marked_fraud) if total_marked_fraud > 0 else 0.0
    recall = (tp / total_actual_fraud) if total_actual_fraud > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("-" * 30)
    print(f"🎯 Accuracy:     {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"🎪 Precision:    {precision:.4f} ({precision*100:.1f}%)")
    print(f"🔍 Recall:       {recall:.4f} ({recall*100:.1f}%)")
    print(f"⚖️ F1-Score:     {f1_score:.4f} ({f1_score*100:.1f}%)")
    
    print(f"\n📊 SUMMARY COUNTS")
    print("-" * 30)
    print(f"💳 Total Transactions: {TOTAL_TRANSACTIONS:,}")
    print(f"🚨 Total Alerts Sent: {FRAUD_ALERTS_SENT:,}")
    print(f"✅ Correctly Detected Fraud: {tp:,}")
    print(f"❌ Missed Fraud: {fn:,}")
    print(f"⚠️ False Alarms: {fp:,}")
    
    if total_actual_fraud > 0:
        fraud_detection_rate = (tp / total_actual_fraud) * 100
        print(f"🎯 Fraud Detection Rate: {fraud_detection_rate:.1f}%")
    
    print("="*60)

def print_final_statistics():
    """Print comprehensive statistics at the end"""
    print("\n" + "="*80)
    print("🎯 FINAL FRAUD DETECTION STATISTICS")
    print("="*80)
    
    # Basic counts
    print(f"📊 Total Transactions Processed: {TOTAL_TRANSACTIONS}")
    print(f"🚨 Total Fraud Alerts Sent: {FRAUD_ALERTS_SENT}")
    
    if TOTAL_TRANSACTIONS == 0:
        print("⌛ No transactions processed with valid labels!")
        return
    
    # Confusion Matrix
    tp = CONFUSION_MATRIX.get((1, 1), 0)  # True Positives: Actual=1, Predicted=1
    tn = CONFUSION_MATRIX.get((0, 0), 0)  # True Negatives: Actual=0, Predicted=0
    fp = CONFUSION_MATRIX.get((0, 1), 0)  # False Positives: Actual=0, Predicted=1
    fn = CONFUSION_MATRIX.get((1, 0), 0)  # False Negatives: Actual=1, Predicted=0
    
    print(f"\n🎯 CONFUSION MATRIX (Prediction vs Actual)")
    print("="*50)
    print("                 │  Actual")
    print("    Predicted    │   0 (Not Fraud)  │   1 (Fraud)   │")
    print("─────────────────┼─────────────────┼───────────────┤")
    print(f"  0 (Not Fraud)  │      {tn:6d}     │     {fn:6d}     │")
    print(f"  1 (Fraud)      │      {fp:6d}     │     {tp:6d}     │")
    print("─────────────────┴─────────────────┴───────────────┘")
    
    # Calculate metrics
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    accuracy = ((tp + tn) / TOTAL_TRANSACTIONS) if TOTAL_TRANSACTIONS > 0 else 0.0
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("="*50)
    print(f"🎯 Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"🔍 Recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"🎛️  Specificity:  {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"⚖️  F1-Score:     {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    # Success rate specifically for fraud detection
    actual_frauds = tp + fn
    detected_frauds = tp
    fraud_success_rate = (detected_frauds / actual_frauds) if actual_frauds > 0 else 0.0
    
    print(f"\n🚨 FRAUD DETECTION SUCCESS RATE")
    print("="*50)
    print(f"💳 Total Fraudulent Transactions: {actual_frauds}")
    print(f"✅ Successfully Detected: {detected_frauds}")
    print(f"❌ Missed Frauds: {fn}")
    print(f"🎯 Fraud Success Rate: {fraud_success_rate:.4f} ({fraud_success_rate*100:.2f}%)")
    
    # Alert effectiveness
    alert_precision = (tp / FRAUD_ALERTS_SENT) if FRAUD_ALERTS_SENT > 0 else 0.0
    print(f"\n🚨 ALERT EFFECTIVENESS")
    print("="*50)
    print(f"🔔 Total Alerts Sent: {FRAUD_ALERTS_SENT}")
    print(f"✅ Correct Alerts (True Frauds): {tp}")
    print(f"❌ False Alerts (False Positives): {fp}")
    print(f"🎯 Alert Precision: {alert_precision:.4f} ({alert_precision*100:.2f}%)")
    
    # Detailed breakdown
    actual_non_frauds = tn + fp
    print(f"\n📊 DETAILED BREAKDOWN")
    print("="*50)
    print(f"💳 Legitimate Transactions: {actual_non_frauds}")
    print(f"   ✅ Correctly Identified: {tn} ({(tn/actual_non_frauds)*100:.1f}%)" if actual_non_frauds > 0 else "   ✅ Correctly Identified: 0")
    print(f"   ❌ Incorrectly Flagged: {fp} ({(fp/actual_non_frauds)*100:.1f}%)" if actual_non_frauds > 0 else "   ❌ Incorrectly Flagged: 0")
    print(f"🚨 Fraudulent Transactions: {actual_frauds}")
    print(f"   ✅ Correctly Detected: {tp} ({(tp/actual_frauds)*100:.1f}%)" if actual_frauds > 0 else "   ✅ Correctly Detected: 0")
    print(f"   ❌ Missed: {fn} ({(fn/actual_frauds)*100:.1f}%)" if actual_frauds > 0 else "   ❌ Missed: 0")
    
    print("\n" + "="*80)

def main():
    global MESSAGE_COUNT
    
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    topic = os.getenv('KAFKA_TOPIC', 'fraud-alerts')
    group_id = os.getenv('KAFKA_GROUP_ID', 'fraud-consumer-group')

    logger.info("🚀 Starting Enhanced Fraud Consumer...")
    logger.info(f"Bootstrap servers: {bootstrap_servers}")
    logger.info(f"Topic: {topic}")
    logger.info(f"Group ID: {group_id}")
    logger.info("📊 Will print summary table every 10,000 messages")

    if not wait_for_kafka(bootstrap_servers):
        logger.error("Could not connect to Kafka. Exiting.")
        return

    consumer = create_consumer(bootstrap_servers, topic, group_id)
    logger.info("✅ Successfully connected to Kafka and subscribed to fraud-alerts topic")

    try:
        for message in consumer:
            try:
                MESSAGE_COUNT += 1
                record = message.value
                amount = record.get("Amount", "N/A")
                cls = record.get("Class", "?")
                is_fraud = record.get("is_fraud", "N/A")
                timestamp = record.get("timestamp", "N/A")

                # Only show the fraud alert details (no change here)
                print(f"🚨 FRAUD ALERT:")
                print(f"   💰 Amount: ${amount}")
                print(f"   🏷️  Class: {cls}")
                print(f"   ⚖️ Fraud Prediction: {is_fraud}")
                print(f"   ⏰ Timestamp: {timestamp}")
                print("-" * 50)

                # Update statistics (this is an alert, so alerted=True)
                update_stats(cls, is_fraud, alerted=True)

                # Print summary table every 10k messages
                if MESSAGE_COUNT % 10000 == 0:
                    print_summary_table()

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Consumer error: {e}")
    finally:
        consumer.close()
        logger.info("Consumer closed.")
        
        # Print final statistics
        print_final_statistics()

if __name__ == "__main__":
    main()