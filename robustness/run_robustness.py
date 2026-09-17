"""Repeat the dissertation's supervised adaptation experiment across seeds 1-30.

Run from the project root:
    python robustness/run_robustness.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_model as tm

rows = []
for seed in range(1, 31):
    df = tm.generate_data(n=6000, seed=seed)
    X, y = tm.prepare_features(df)
    initial = df["step"] <= tm.INITIAL_TRAIN_END
    reference = df["step"].between(tm.REFERENCE_START, tm.REFERENCE_END)
    drift = df["step"].between(tm.DRIFT_START, tm.DRIFT_END)
    final = df["step"].between(tm.FINAL_START, tm.FINAL_END)
    adaptation = df["step"] <= tm.DRIFT_END

    base_scaler = StandardScaler().fit(X.loc[initial])
    base_model = tm.build_fraud_model()
    base_model.fit(base_scaler.transform(X.loc[initial]), y.loc[initial])
    ref = tm.evaluate_classifier(base_model, base_scaler, X, y, reference)
    obs = tm.evaluate_classifier(base_model, base_scaler, X, y, drift)
    pre = tm.evaluate_classifier(base_model, base_scaler, X, y, final)
    drift_detected = (
        ref["recall"] - obs["recall"] >= tm.DRIFT_METRIC_DROP_THRESHOLD
        or ref["roc_auc"] - obs["roc_auc"] >= tm.DRIFT_METRIC_DROP_THRESHOLD
    )

    adapt_scaler = StandardScaler().fit(X.loc[adaptation])
    adapt_model = tm.build_fraud_model()
    adapt_model.fit(adapt_scaler.transform(X.loc[adaptation]), y.loc[adaptation])
    post = tm.evaluate_classifier(adapt_model, adapt_scaler, X, y, final)

    rows.append({
        "seed": seed,
        "drift_detected": drift_detected,
        "pre_recall": pre["recall"],
        "post_recall": post["recall"],
        "recall_gain": post["recall"] - pre["recall"],
        "pre_f1": pre["f1_score"],
        "post_f1": post["f1_score"],
        "f1_gain": post["f1_score"] - pre["f1_score"],
        "pre_auc": pre["roc_auc"],
        "post_auc": post["roc_auc"],
        "auc_gain": post["roc_auc"] - pre["roc_auc"],
    })

out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("robustness_30_seeds.csv"), index=False)
print(out[["recall_gain", "f1_gain", "auc_gain"]].mean())
print(f"Recall improved in {(out.recall_gain > 0).sum()}/30 runs")
print(f"Drift signal fired in {out.drift_detected.sum()}/30 runs")
