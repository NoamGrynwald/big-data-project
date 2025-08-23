#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from joblib import dump
import numpy as np
import os

# ================================================================
# Config
# ================================================================
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "creditcard_json.json"

# -------- IMPROVED CONFIG --------
# Use more data and more reasonable class weights
DATA_FRACTION = 0.1
CLASS_WEIGHT_OPTIONS = [
    {0: 1.0, 1: 10000.0},
]

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# FIXED: Match the exact feature order from training (from fraud_detection_spark.py)
RAW_FEATURE_COLUMNS = [
    "Amount", "Time", "V1", "V10", "V11", "V12", "V13", "V14", "V15", "V16",
    "V17", "V18", "V19", "V2", "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28", "V3", "V4", "V5", "V6", "V7", "V8", "V9"
]


def add_time_features(df):
    """Add the same time features used in fraud detection script"""
    print("🕒 Adding enhanced time features...")

    # Ensure Time column exists and is numeric
    if 'Time' not in df.columns:
        print("⚠️ Warning: Time column not found, using zeros")
        df['Time'] = 0.0

    # Convert Time to float, handling any string formats
    df['Time'] = pd.to_numeric(df['Time'], errors='coerce').fillna(0.0)
    time_vals = df['Time'].astype(float)

    print(f"   Time column stats: min={time_vals.min():.1f}, max={time_vals.max():.1f}, mean={time_vals.mean():.1f}")

    # Time normalization and bucketing features (matching fraud detection script)
    df['time_normalized'] = time_vals / 172800.0  # Normalize to 48 hours (common dataset span)
    df['time_bucket'] = (time_vals // 10000).astype(int)  # Time bucket (0-17 for typical dataset)
    df['time_bucket_normalized'] = df['time_bucket'] / 17.0  # Normalize bucket

    # Cyclical time features (assuming seconds in a day pattern)
    seconds_in_day = 86400
    time_of_day = time_vals % seconds_in_day
    df['time_sin'] = np.sin(2 * np.pi * time_of_day / seconds_in_day)
    df['time_cos'] = np.cos(2 * np.pi * time_of_day / seconds_in_day)

    # Time progression feature (capped at 2.0)
    df['time_progression'] = np.minimum(time_vals / 50000, 2.0)

    print(f"✅ Added 6 time features:")  # Changed from 5 to 6
    print(f"   • time_normalized: {df['time_normalized'].min():.3f} - {df['time_normalized'].max():.3f}")
    print(f"   • time_bucket: {df['time_bucket'].min()} - {df['time_bucket'].max()} buckets")
    print(f"   • time_bucket_normalized: {df['time_bucket_normalized'].min():.3f} - {df['time_bucket_normalized'].max():.3f}")
    print(f"   • time_sin/cos: cyclical features for daily patterns")
    print(f"   • time_progression: {df['time_progression'].min():.3f} - {df['time_progression'].max():.3f}")

    return df


def ensure_feature_order(df):
    """Ensure features are in the exact order expected by fraud detection script"""
    print("📄 Ensuring correct feature order...")

    # Start with raw features in correct order
    ordered_features = []

    # Add raw features in exact order
    for col in RAW_FEATURE_COLUMNS:
        if col in df.columns:
            ordered_features.append(col)
        else:
            print(f"⚠️ Warning: Missing expected column {col}")

    # Add time features in the same order as fraud detection script
    # FIXED: Include time_bucket in the feature list (it was missing!)
    time_feature_order = [
        'time_normalized',
        'time_bucket',           # This was missing - causing the mismatch!
        'time_bucket_normalized',
        'time_sin',
        'time_cos',
        'time_progression'
    ]

    for col in time_feature_order:
        if col in df.columns:
            ordered_features.append(col)

    # Add any remaining columns (excluding Class)
    for col in df.columns:
        if col not in ordered_features and col != 'Class':
            ordered_features.append(col)

    print(f"✅ Feature order established: {len(ordered_features)} features")
    print(f"   First 10: {ordered_features[:10]}")
    print(f"   Last 10: {ordered_features[-10:]}")

    return df[ordered_features + ['Class'] if 'Class' in df.columns else ordered_features]


# ================================================================
# Enhanced data loading with better sampling
# ================================================================
def load_data_stratified(file_path, frac=0.3):
    """Load data with better sampling strategy - handles quoted class format"""
    print(f"Loading {frac * 100:.0f}% of dataset with stratified sampling...")

    # First pass: count classes to ensure good sampling
    class_counts = {0: 0, 1: 0}
    total_lines = 0

    with open(file_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # Handle quoted class format: "'0'" -> 0, "'1'" -> 1
                class_raw = record.get("Class", "'0'")
                class_val = int(str(class_raw).strip("'\""))
                class_counts[class_val] += 1
                total_lines += 1
            except Exception as e:
                continue

    print(f"Full dataset: {total_lines:,} transactions")
    print(f"Class distribution: Normal={class_counts[0]:,}, Fraud={class_counts[1]:,}")
    fraud_rate = class_counts[1] / total_lines
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate * 100:.2f}%)")

    # Calculate sampling targets
    fraud_target = max(int(class_counts[1] * frac), class_counts[1])  # Take all fraud if small dataset
    normal_target = int(class_counts[0] * frac)

    print(f"Sampling targets: {fraud_target:,} fraud, {normal_target:,} normal")

    # Second pass: stratified sampling
    records = []
    fraud_collected = 0
    normal_collected = 0

    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                # Handle quoted class format
                class_raw = record.get("Class", "'0'")
                class_val = int(str(class_raw).strip("'\""))

                if class_val == 1 and fraud_collected < fraud_target:
                    records.append(record)
                    fraud_collected += 1
                elif class_val == 0 and normal_collected < normal_target:
                    # Use deterministic sampling based on line number for reproducibility
                    if (line_num * 17 + 42) % int(1 / frac) == 0:  # Pseudo-random but deterministic
                        records.append(record)
                        normal_collected += 1

                # Stop when we have enough samples
                if fraud_collected >= fraud_target and normal_collected >= normal_target:
                    break

                # Progress indicator for large datasets
                if line_num % 50000 == 0:
                    print(f"  Processed {line_num:,} lines, collected {len(records):,} samples")

            except Exception as e:
                if line_num < 10:  # Only show first few errors
                    print(f"  Warning: Error parsing line {line_num}: {e}")
                continue

    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df):,} transactions ({fraud_collected:,} fraud, {normal_collected:,} normal)")

    # Verify class distribution in sampled data
    if len(df) > 0:
        df_fraud_count = sum(df['Class'].apply(lambda x: int(str(x).strip("'\"")) == 1))
        print(f"   Final fraud cases: {df_fraud_count:,} ({df_fraud_count / len(df) * 100:.1f}%)")

        # Show sample data structure
        print(f"   Sample record keys: {list(df.iloc[0].keys())}")
        print(f"   Time range: {df['Time'].min():.1f} - {df['Time'].max():.1f}")
        print(f"   Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")

    return df


def evaluate_model_with_cv(model, X, y, cv_folds=5):
    """Evaluate model with cross-validation"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Cross-validation scores
    roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
    recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')

    return {
        'roc_auc_mean': roc_scores.mean(),
        'roc_auc_std': roc_scores.std(),
        'precision_mean': precision_scores.mean(),
        'precision_std': precision_scores.std(),
        'recall_mean': recall_scores.mean(),
        'recall_std': recall_scores.std()
    }


def calculate_expected_results(y_true, y_pred, y_proba, total_dataset_size, fraud_rate):
    """Calculate what to expect during deployment"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Project to full dataset
    total_fraud_cases = int(total_dataset_size * fraud_rate)
    total_normal_cases = total_dataset_size - total_fraud_cases

    # Expected results on full dataset
    expected_tp = int(recall * total_fraud_cases)
    expected_fn = total_fraud_cases - expected_tp
    expected_fp = int(false_positive_rate * total_normal_cases)
    expected_tn = total_normal_cases - expected_fp

    expected_alerts = expected_tp + expected_fp

    return {
        'test_metrics': {
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'false_positive_rate': false_positive_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        },
        'expected_full_dataset': {
            'total_transactions': total_dataset_size,
            'total_fraud_cases': total_fraud_cases,
            'total_normal_cases': total_normal_cases,
            'expected_fraud_detected': expected_tp,
            'expected_fraud_missed': expected_fn,
            'expected_false_alarms': expected_fp,
            'expected_total_alerts': expected_alerts,
            'expected_alert_rate': expected_alerts / total_dataset_size,
            'fraud_detection_rate': expected_tp / total_fraud_cases if total_fraud_cases > 0 else 0
        }
    }


def print_expected_results(results):
    """Print formatted expected results"""
    test = results['test_metrics']
    expected = results['expected_full_dataset']

    print(f"\n{'=' * 80}")
    print("🎯 EXPECTED DEPLOYMENT RESULTS")
    print(f"{'=' * 80}")

    print(f"\n📊 TEST SET PERFORMANCE:")
    print(f"   Precision:     {test['precision']:.3f}")
    print(f"   Recall:        {test['recall']:.3f}")
    print(f"   Specificity:   {test['specificity']:.3f}")
    print(f"   False Pos Rate: {test['false_positive_rate']:.4f}")

    print(f"\n🔮 EXPECTED FULL DATASET PERFORMANCE:")
    print(f"   Total Transactions:     {expected['total_transactions']:,}")
    print(f"   Total Fraud Cases:      {expected['total_fraud_cases']:,}")
    print(
        f"   Expected Fraud Detected: {expected['expected_fraud_detected']:,} ({expected['fraud_detection_rate']:.1%})")
    print(f"   Expected Fraud Missed:   {expected['expected_fraud_missed']:,}")
    print(f"   Expected False Alarms:   {expected['expected_false_alarms']:,}")
    print(f"   Expected Total Alerts:   {expected['expected_total_alerts']:,}")
    print(
        f"   Alert Rate:             {expected['expected_alert_rate']:.4f} ({expected['expected_alert_rate'] * 100:.2f}%)")

    print(f"\n🚨 WHAT TO EXPECT DURING STREAMING:")
    print(f"   • You should see ~{expected['expected_fraud_detected']} actual fraud cases detected")
    print(f"   • You should see ~{expected['expected_fraud_missed']} actual fraud cases missed")
    print(f"   • You should see ~{expected['expected_false_alarms']} false positive alerts")
    print(f"   • Total alerts should be around {expected['expected_total_alerts']}")
    print(f"   • Alert frequency: ~{expected['expected_alert_rate'] * 1000:.1f} alerts per 1000 transactions")


# ================================================================
# Load and prepare data
# ================================================================
df = load_data_stratified(DATA_FILE, frac=DATA_FRACTION)

# Count total dataset size for projections
total_dataset_size = 0
total_fraud_count = 0

print(f"\n🔍 Counting full dataset for projections...")
with open(DATA_FILE, "r") as f:
    for line_num, line in enumerate(f, 1):
        try:
            record = json.loads(line.strip())
            class_raw = record.get("Class", "'0'")
            class_val = int(str(class_raw).strip("'\""))
            total_dataset_size += 1
            if class_val == 1:
                total_fraud_count += 1

            # Progress indicator
            if line_num % 100000 == 0:
                print(f"  Counted {line_num:,} lines...")
        except:
            continue

full_fraud_rate = total_fraud_count / total_dataset_size
print(f"✅ Full dataset: {total_dataset_size:,} transactions, {total_fraud_count:,} fraud ({full_fraud_rate:.4f} rate)")

# ================================================================
# ENHANCED FEATURE ENGINEERING
# ================================================================
print(f"\n{'=' * 80}")
print("🔧 ENHANCED FEATURE ENGINEERING")
print("=" * 80)

# Check data structure before processing
print(f"Loaded dataframe shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Class column sample: {df['Class'].head().tolist()}")

# Clean the Class column first
print("🧹 Cleaning Class column...")
df['Class'] = df['Class'].apply(lambda x: int(str(x).strip("'\"")))
print(f"Class values after cleaning: {sorted(df['Class'].unique())}")

# Add time features to match fraud detection script
df = add_time_features(df)

# Ensure proper feature order
df = ensure_feature_order(df)

# Features & labels with improved processing
X = df.drop(columns=["Class"])
y = df["Class"]  # Already cleaned above

print(f"\nFinal feature set:")
print(f"   Total features: {X.shape[1]}")
print(f"   Feature columns: {list(X.columns)}")
print(f"   Shape: {X.shape}")

print(f"\nFinal training class distribution:")
print(f"Normal (0): {sum(y == 0)} ({sum(y == 0) / len(y) * 100:.2f}%)")
print(f"Fraud (1):  {sum(y == 1)} ({sum(y == 1) / len(y) * 100:.2f}%)")

# Save enhanced feature info for deployment
feature_info = {
    'columns': list(X.columns),
    'dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
    'shape': X.shape,
    'fraud_rate': sum(y == 1) / len(y),
    'full_dataset_size': total_dataset_size,
    'full_fraud_rate': full_fraud_rate,
    'means': X.mean().to_dict(),
    'stds': X.std().to_dict(),
    'class_distribution': {0: int(sum(y == 0)), 1: int(sum(y == 1))},
    'has_time_features': True,
    'time_feature_names': [
        'time_normalized', 'time_bucket', 'time_bucket_normalized',  # Added time_bucket
        'time_sin', 'time_cos', 'time_progression'
    ],
    'raw_feature_columns': RAW_FEATURE_COLUMNS,
    'time_range': {
        'min': float(df['Time'].min()) if 'Time' in df.columns else 0.0,
        'max': float(df['Time'].max()) if 'Time' in df.columns else 0.0
    }
}

with open(OUTPUT_DIR / "feature_info.json", "w") as f:
    json.dump(feature_info, f, indent=2)
print("✅ Saved enhanced feature_info.json with time feature metadata")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain/Test split:")
print(f"   Training: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# ================================================================
# Test different class weights
# ================================================================
print("\n" + "=" * 80)
print("TESTING DIFFERENT CLASS WEIGHTS WITH TIME FEATURES")
print("=" * 80)

best_models = {}
best_cv_results = {}

for i, class_weight in enumerate(CLASS_WEIGHT_OPTIONS):
    print(f"\n--- Class Weight Option {i + 1}: {class_weight} ---")

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight=class_weight, random_state=42
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, class_weight=class_weight, random_state=42
        ),
        "svc_rbf": SVC(
            kernel="rbf", probability=True, class_weight=class_weight, random_state=42
        )
    }

    cv_results = {}
    for name, model in models.items():
        print(f"\nEvaluating {name} with time features...")
        cv_result = evaluate_model_with_cv(model, X_train, y_train)
        cv_results[name] = cv_result

        print(f"  ROC-AUC: {cv_result['roc_auc_mean']:.4f} (±{cv_result['roc_auc_std']:.4f})")
        print(f"  Precision: {cv_result['precision_mean']:.4f} (±{cv_result['precision_std']:.4f})")
        print(f"  Recall: {cv_result['recall_mean']:.4f} (±{cv_result['recall_std']:.4f})")

    # Store results for comparison
    best_cv_results[str(class_weight)] = cv_results
    best_models[str(class_weight)] = models

# ================================================================
# Select best class weight based on validation
# ================================================================
print(f"\n{'=' * 80}")
print("SELECTING BEST CLASS WEIGHT")
print("=" * 80)

best_class_weight = None
best_avg_score = 0

for weight_str, cv_results in best_cv_results.items():
    # Calculate average score across models (focusing on recall for fraud detection)
    avg_score = np.mean([
        cv_results[model]['recall_mean'] for model in cv_results.keys()
    ])
    print(f"Class weight {weight_str}: Average recall = {avg_score:.4f}")

    if avg_score > best_avg_score:
        best_avg_score = avg_score
        best_class_weight = weight_str

print(f"\n🎯 Best class weight: {best_class_weight} (avg recall: {best_avg_score:.4f})")

# ================================================================
# Train final models with best class weight
# ================================================================
selected_models = best_models[best_class_weight]
base_preds_train = []
base_preds_test = []

print(f"\n{'=' * 80}")
print(f"TRAINING FINAL MODELS WITH TIME FEATURES & CLASS_WEIGHT = {best_class_weight}")
print("=" * 80)

for name, model in selected_models.items():
    print(f"\nTraining final {name} with {X_train.shape[1]} features...")
    model.fit(X_train, y_train)

    # Save model
    dump(model, OUTPUT_DIR / f"{name}.joblib")
    print(f"✅ Saved {name}.joblib")

    # Store predictions for meta model
    train_pred = model.predict_proba(X_train)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    base_preds_train.append(train_pred)
    base_preds_test.append(test_pred)

    # Show test performance
    y_pred = model.predict(X_test)
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_pred):.4f}")

    # Show feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 features for {name}:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")

# ================================================================
# Train meta-model with different options
# ================================================================
X_train_meta = np.vstack(base_preds_train).T
X_test_meta = np.vstack(base_preds_test).T

print(f"\n{'=' * 80}")
print("TRAINING META-MODELS")
print("=" * 80)

# Try different meta-model configurations
meta_options = [
    ("dt_depth3", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)),
    ("dt_depth5", DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)),
    ("rf_meta", RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42))
]

best_meta = None
best_meta_name = None
best_meta_score = 0

for name, meta_model in meta_options:
    print(f"\nTesting meta-model: {name}")
    meta_model.fit(X_train_meta, y_train)

    y_pred_meta = meta_model.predict(X_test_meta)
    y_proba_meta = meta_model.predict_proba(X_test_meta)[:, 1]

    roc_score = roc_auc_score(y_test, y_proba_meta)
    prec, rec, _ = precision_recall_curve(y_test, y_proba_meta)
    pr_score = auc(rec, prec)

    print(f"  ROC-AUC: {roc_score:.4f}, PR-AUC: {pr_score:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, y_pred_meta, digits=4))

    if roc_score > best_meta_score:
        best_meta_score = roc_score
        best_meta = meta_model
        best_meta_name = name

print(f"\n🎯 Best meta-model: {best_meta_name} (ROC-AUC: {best_meta_score:.4f})")

# Save best meta model
dump(best_meta, OUTPUT_DIR / "decision_tree_meta.joblib")
print("✅ Saved decision_tree_meta.joblib")

# ================================================================
# Final evaluation and expected results
# ================================================================
y_pred_final = best_meta.predict(X_test_meta)
y_proba_final = best_meta.predict_proba(X_test_meta)[:, 1]

print(f"\n{'=' * 80}")
print("FINAL MODEL EVALUATION WITH TIME FEATURES")
print("=" * 80)
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred_final))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_final, digits=4))

roc_auc = roc_auc_score(y_test, y_proba_final)
prec, rec, _ = precision_recall_curve(y_test, y_proba_final)
pr_auc = auc(rec, prec)

print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

# ================================================================
# CALCULATE AND DISPLAY EXPECTED RESULTS
# ================================================================
expected_results = calculate_expected_results(
    y_test, y_pred_final, y_proba_final,
    total_dataset_size, full_fraud_rate
)

print_expected_results(expected_results)

# Save comprehensive model info including expected results
# Fix class weight parsing
try:
    if best_class_weight == "balanced":
        parsed_class_weight = "balanced"
    else:
        parsed_class_weight = eval(best_class_weight)
except:
    parsed_class_weight = best_class_weight

model_info = {
    'best_class_weight': parsed_class_weight,
    'best_meta_model': best_meta_name,
    'final_roc_auc': float(roc_auc),
    'final_pr_auc': float(pr_auc),
    'data_fraction_used': DATA_FRACTION,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'full_dataset_size': total_dataset_size,
    'full_fraud_rate': full_fraud_rate,
    'expected_results': expected_results,
    'has_time_features': True,
    'total_features': X.shape[1],
    'feature_names': list(X.columns),
    'time_features_added': [
        'time_normalized', 'time_bucket', 'time_bucket_normalized',  # Added time_bucket
        'time_sin', 'time_cos', 'time_progression'
    ]
}

with open(OUTPUT_DIR / "model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\n🎉 TRAINING COMPLETE!")
print(f"✅ Enhanced models with time features saved to: {OUTPUT_DIR.resolve()}")
print(f"📁 Files created:")
print(f"   • random_forest.joblib")
print(f"   • extra_trees.joblib")
print(f"   • svc_rbf.joblib")
print(f"   • decision_tree_meta.joblib")
print(f"   • feature_info.json (with time feature metadata)")
print(f"   • model_info.json (with performance expectations)")
print(f"\n🔥 Models now trained with {X.shape[1]} features including:")
print(f"   • {len(RAW_FEATURE_COLUMNS)} original features")
print(f"   • 6 enhanced time features")  # Changed from 5 to 6
print("=" * 80)