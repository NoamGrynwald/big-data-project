#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ONE meta-ensemble for fraud detection on ONLY the first 5% of creditcard_json.json (project root):
  Meta-learner: DecisionTreeClassifier
  Base models:  RandomForest, ExtraTrees, SVC(RBF)

Flow:
- Read first 5% of JSONL file (project root)
- Clean target (Class "'0'"/"'1'" -> 0/1)
- Split that 5% into train/test (stratified)
- Build out-of-fold (OOF) probabilities from base models for meta-train (no leakage)
- Fit DecisionTree meta-learner on OOF features
- Refit base models on FULL train; create meta-test features and evaluate on test
- Print detailed report (confusion matrix + rates, success table, classification report)
- Save the ENTIRE trained ensemble (base models + meta tree + metadata) as ONE .joblib file

Run (from project root):
    python scripts/train_meta_third_option.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, accuracy_score,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

# --------------------------- config ---------------------------
RANDOM_STATE = 42
PCT = 0.05  # Use FIRST 5% of the file total lines
DATA_FILE = "data/creditcard_json.json"  # at project root

# Write artifacts at project root (even when executed from scripts/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# --------------------------------------------------------------


# --------------------------- utils ----------------------------
def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_first_percent_jsonl(path: str, pct: float) -> pd.DataFrame:
    total = count_lines(path)
    nrows = max(1, int(total * pct))
    df = pd.read_json(path, lines=True, nrows=nrows)
    return df


def clean_target(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    df[target_col] = (
        df[target_col].astype(str).str.replace("'", "", regex=False).str.strip().astype(int)
    )
    return df


def compute_weights(y: np.ndarray) -> dict:
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def get_probs(est, X):
    """Return positive-class probabilities if available, else sigmoid(decision_function); else 0/1."""
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(est, "decision_function"):
        scores = est.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return est.predict(X).astype(float)


def success_table(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc, pr = None, None
    if y_prob is not None:
        try: roc = roc_auc_score(y_true, y_prob)
        except Exception: roc = None
        try: pr = average_precision_score(y_true, y_prob)
        except Exception: pr = None

    return pd.DataFrame([{
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc, 4) if roc is not None else None,
        "PR-AUC": round(pr, 4) if pr is not None else None,
    }])


def print_detailed_report(title: str, y_true, y_pred, y_prob=None):
    print("=" * 70)
    print(title)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0  # recall
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0  # specificity
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    print("Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame([[tn, fp], [fn, tp]], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))
    print(f"\nRates: TPR/Recall={tpr:.4f}, FPR={fpr:.4f}, TNR/Specificity={tnr:.4f}, FNR={fnr:.4f}")

    print("\nSuccess Table:")
    print(success_table(y_true, y_pred, y_prob).to_string(index=False))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print()
# --------------------------------------------------------------


# -------------- base model constructors (weighted) --------------
def make_rf(class_weights):
    return RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight=class_weights
    )

def make_extratrees(class_weights):
    return ExtraTreesClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_STATE, class_weight=class_weights
    )

def make_svc(class_weights):
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True,
            class_weight=class_weights, random_state=RANDOM_STATE
        ))
    ])
# --------------------------------------------------------------


# --------- OOF meta-feature builder (no leakage) ----------
def oof_meta_features(base_ctors, X_tr, y_tr, X_te, class_weights):
    """
    base_ctors: list of callables that accept (class_weights) and return an estimator
    Return:
      oof_train: (n_train, n_models)  out-of-fold prob features
      test_meta: (n_test,  n_models)  test prob features (fit base on full train)
      fitted:    list of fitted base models (on full train)
    """
    n_models = len(base_ctors)
    oof_train = np.zeros((X_tr.shape[0], n_models), dtype=float)
    test_meta = np.zeros((X_te.shape[0], n_models), dtype=float)
    fitted = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for m_idx, ctor in enumerate(base_ctors):
        # OOF predictions
        oof_col = np.zeros(X_tr.shape[0], dtype=float)
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            model = ctor(class_weights)
            model.fit(X_tr.iloc[tr_idx], y_tr[tr_idx])
            oof_col[va_idx] = get_probs(model, X_tr.iloc[va_idx])

        oof_train[:, m_idx] = oof_col

        # Fit on full train → meta-test features
        full_model = ctor(class_weights)
        full_model.fit(X_tr, y_tr)
        test_meta[:, m_idx] = get_probs(full_model, X_te)
        fitted.append(full_model)

    return oof_train, test_meta, fitted
# --------------------------------------------------------------


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    # Load ONLY the first 5% and prep
    df = load_first_percent_jsonl(DATA_PATH, PCT)
    df = clean_target(df, "Class")

    # Features / target
    y = df["Class"].values
    X = df.drop(columns=["Class"]).select_dtypes(include=[np.number]).copy()

    # Split that 5% into train/test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Class weighting: inverse frequency
    class_weights = compute_weights(y_tr)

    print("5% subset sizes:", {"train": X_tr.shape[0], "test": X_te.shape[0]})
    print("Train class distribution:", dict(pd.Series(y_tr).value_counts().sort_index()))
    print("Applied class weights:", class_weights, "\n")

    # Define the third option bases: RF + ExtraTrees + SVC(RBF)
    base_ctors = [make_rf, make_extratrees, make_svc]

    # Build OOF meta-train + meta-test features
    oof_train, test_meta, fitted_bases = oof_meta_features(
        base_ctors, X_tr, y_tr, X_te, class_weights
    )

    # Meta-learner: shallow, explainable tree
    meta = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
    )
    meta.fit(oof_train, y_tr)

    # Evaluate on test
    y_prob_meta = get_probs(meta, test_meta)
    y_pred_meta = (y_prob_meta >= 0.5).astype(int)
    print_detailed_report("DecisionTree on [RF, ExtraTrees, SVC(RBF)] — 5% TEST", y_te, y_pred_meta, y_prob_meta)

    # -------- Save ONE file with the whole ensemble --------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle = {
        "ensemble_name": "DT_on_RF_ExtraTrees_SVC_RBF",
        "base_models": [
            ("RandomForest", fitted_bases[0]),
            ("ExtraTrees",   fitted_bases[1]),
            ("SVC_RBF",      fitted_bases[2]),
        ],
        "meta_model": meta,
        "feature_order": list(X.columns),
        "class_weights": class_weights,
        "train_info": {
            "pct_used": PCT,
            "random_state": RANDOM_STATE,
            "timestamp": timestamp,
            "oof_folds": 5,
        },
        "notes": "Predict by stacking base model probabilities into meta decision tree."
    }
    out_path = os.path.join(MODELS_DIR, f"fraud_ensemble_rf_extratrees_svc_dt_{timestamp}.joblib")
    dump(bundle, out_path)
    print(f"Saved full ensemble as ONE file: {out_path}")

    # Also write a tiny metrics CSV for the test split
    metrics = success_table(y_te, y_pred_meta, y_prob_meta)
    metrics.insert(0, "Ensemble", "DT_on_RF_ExtraTrees_SVC_RBF")
    csv_path = os.path.join(REPORTS_DIR, f"meta_third_option_5pct_metrics_{timestamp}.csv")
    print(f"Saved metrics CSV: {csv_path}")

    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
