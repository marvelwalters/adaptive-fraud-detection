from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

# Chronological experiment design.
# 1-15: initial model training
# 16-20: stable reference/validation period
# 21-25: drift observation period
# 26-30: untouched final holdout period
INITIAL_TRAIN_END = 15
REFERENCE_START = 16
REFERENCE_END = 20
DRIFT_START = 21
DRIFT_END = 25
FINAL_START = 26
FINAL_END = 30
DRIFT_METRIC_DROP_THRESHOLD = 0.05


def generate_data(n=6000, seed=RANDOM_STATE):
    """Generate a reproducible PaySim-style synthetic transaction dataset.

    A controlled change in fraud behaviour is introduced after step 22. This is
    deliberate: it lets the prototype test whether a model trained on earlier
    behaviour degrades when fraud tactics evolve.
    """
    rng = np.random.default_rng(seed)
    tx_types = np.array(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
    tx_probs = np.array([0.28, 0.22, 0.30, 0.15, 0.05])

    step = rng.integers(1, 31, size=n)
    tx_type = rng.choice(tx_types, size=n, p=tx_probs)
    amount = np.clip(rng.lognormal(mean=8.0, sigma=1.05, size=n), 20, 250000)
    old_org = np.clip(rng.lognormal(mean=9.0, sigma=1.1, size=n), 0, 450000)
    drop = rng.beta(2, 6, size=n)
    new_org = np.clip(old_org - old_org * drop, 0, None)
    old_dest = np.clip(rng.lognormal(mean=9.4, sigma=1.0, size=n), 0, 700000)
    new_dest = old_dest + amount * rng.uniform(0.1, 1.0, size=n)

    risky_type = np.isin(tx_type, ["TRANSFER", "CASH_OUT"]).astype(int)
    high_amount = (amount > np.quantile(amount, 0.88)).astype(int)
    drained = ((old_org > 0) & (new_org < old_org * 0.05)).astype(int)
    late_period = (step > 22).astype(int)

    risk_score = (
        0.55 * risky_type
        + 0.45 * high_amount
        + 0.75 * drained
        + 0.20 * late_period
        + rng.normal(0, 0.12, size=n)
    )
    fraud = (risk_score > 1.02).astype(int)

    # Controlled evolving-fraud pattern: after step 22, some medium/high-value
    # TRANSFER and CASH_OUT transactions become fraudulent even though they do
    # not necessarily match the original high-value/account-draining pattern.
    medium_high_amount = (amount > np.quantile(amount, 0.70)).astype(int)
    evolving = (
        (step > 22)
        & np.isin(tx_type, ["TRANSFER", "CASH_OUT"])
        & (medium_high_amount == 1)
        & (rng.random(n) < 0.18)
    )
    fraud = np.where(evolving, 1, fraud)

    # Small label noise keeps the task imperfect, as in realistic classification.
    flip_to_normal = np.where((fraud == 1) & (rng.random(n) < 0.04))[0]
    fraud[flip_to_normal] = 0

    df = pd.DataFrame(
        {
            "step": step,
            "type": tx_type,
            "amount": amount.round(2),
            "oldbalanceOrg": old_org.round(2),
            "newbalanceOrig": new_org.round(2),
            "oldbalanceDest": old_dest.round(2),
            "newbalanceDest": new_dest.round(2),
            "isFraud": fraud,
        }
    )
    return df.sort_values("step").reset_index(drop=True)


def prepare_features(df):
    x = df.copy()
    x["balanceChangeOrig"] = x["oldbalanceOrg"] - x["newbalanceOrig"]
    x["balanceChangeDest"] = x["newbalanceDest"] - x["oldbalanceDest"]
    x["amountToOldBalanceRatio"] = x["amount"] / (x["oldbalanceOrg"] + 1)
    x["isRiskyType"] = x["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    x = pd.get_dummies(x, columns=["type"], drop_first=False)
    y = x["isFraud"].astype(int)
    X = x.drop(columns=["isFraud"])
    return X, y


def build_fraud_model():
    return RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate_classifier(model, scaler, X, y, mask):
    X_eval = X.loc[mask]
    y_eval = y.loc[mask]
    X_eval_s = scaler.transform(X_eval)
    pred = model.predict(X_eval_s)
    prob = model.predict_proba(X_eval_s)[:, 1]

    auc = None
    if y_eval.nunique() > 1:
        auc = float(roc_auc_score(y_eval, prob))

    return {
        "rows": int(len(y_eval)),
        "fraud_cases": int(y_eval.sum()),
        "fraud_rate": float(y_eval.mean()),
        "accuracy": float(accuracy_score(y_eval, pred)),
        "precision": float(precision_score(y_eval, pred, zero_division=0)),
        "recall": float(recall_score(y_eval, pred, zero_division=0)),
        "f1_score": float(f1_score(y_eval, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_eval, pred, labels=[0, 1]).tolist(),
    }


def anomaly_summary(model, scaler, X, y, mask):
    X_eval_s = scaler.transform(X.loc[mask])
    scores = model.decision_function(X_eval_s)
    flags = scores < 0
    y_eval = y.loc[mask].to_numpy()
    fraud_mask = y_eval == 1

    return {
        "anomaly_rate": float(flags.mean()),
        "mean_anomaly_score": float(scores.mean()),
        "fraud_flag_rate": float(flags[fraud_mask].mean()) if fraud_mask.any() else 0.0,
    }


def main():
    print("Generating synthetic PaySim-style dataset...")
    df = generate_data()
    df.to_csv(DATA / "synthetic_paysim_transactions.csv", index=False)

    X, y = prepare_features(df)
    features = list(X.columns)

    initial_train = df["step"] <= INITIAL_TRAIN_END
    reference = df["step"].between(REFERENCE_START, REFERENCE_END)
    drift_window = df["step"].between(DRIFT_START, DRIFT_END)
    final_holdout = df["step"].between(FINAL_START, FINAL_END)
    future = df["step"] >= DRIFT_START
    adaptation_train = df["step"] <= DRIFT_END

    # ------------------------------------------------------------------
    # Stage 1: initial supervised model trained only on historical data.
    # ------------------------------------------------------------------
    baseline_scaler = StandardScaler()
    baseline_scaler.fit(X.loc[initial_train])

    baseline_model = build_fraud_model()
    baseline_model.fit(baseline_scaler.transform(X.loc[initial_train]), y.loc[initial_train])

    # Isolation Forest is trained only on legitimate historical transactions.
    baseline_anomaly_model = IsolationForest(
        n_estimators=160,
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    baseline_normal = initial_train & (y == 0)
    baseline_anomaly_model.fit(baseline_scaler.transform(X.loc[baseline_normal]))

    reference_metrics = evaluate_classifier(baseline_model, baseline_scaler, X, y, reference)
    drift_metrics = evaluate_classifier(baseline_model, baseline_scaler, X, y, drift_window)
    final_pre_metrics = evaluate_classifier(baseline_model, baseline_scaler, X, y, final_holdout)
    future_pre_metrics = evaluate_classifier(baseline_model, baseline_scaler, X, y, future)

    recall_drop_at_drift = reference_metrics["recall"] - drift_metrics["recall"]
    auc_drop_at_drift = reference_metrics["roc_auc"] - drift_metrics["roc_auc"]
    recall_drop_final = reference_metrics["recall"] - final_pre_metrics["recall"]
    # A labelled performance monitor raises a drift signal when either recall or
    # ROC-AUC deteriorates by at least five percentage points from the stable reference.
    # This indicates changing model/data behaviour; it does not by itself establish
    # the underlying causal form of concept drift.
    drift_detected = (
        recall_drop_at_drift >= DRIFT_METRIC_DROP_THRESHOLD
        or auc_drop_at_drift >= DRIFT_METRIC_DROP_THRESHOLD
    )

    # ------------------------------------------------------------------
    # Stage 2: adaptation. Once degradation is observed in steps 21-25,
    # retrain using all labelled data available through step 25. Steps
    # 26-30 remain untouched until the final evaluation.
    # ------------------------------------------------------------------
    adaptive_scaler = StandardScaler()
    adaptive_scaler.fit(X.loc[adaptation_train])

    adaptive_model = build_fraud_model()
    adaptive_model.fit(adaptive_scaler.transform(X.loc[adaptation_train]), y.loc[adaptation_train])

    adaptive_anomaly_model = IsolationForest(
        n_estimators=160,
        contamination=0.05,
        random_state=RANDOM_STATE,
    )
    adaptive_normal = adaptation_train & (y == 0)
    adaptive_anomaly_model.fit(adaptive_scaler.transform(X.loc[adaptive_normal]))

    final_post_metrics = evaluate_classifier(adaptive_model, adaptive_scaler, X, y, final_holdout)

    drift_rows = []
    windows = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30)]
    for start, end in windows:
        mask = df["step"].between(start, end)
        perf = evaluate_classifier(baseline_model, baseline_scaler, X, y, mask)
        anomaly = anomaly_summary(baseline_anomaly_model, baseline_scaler, X, y, mask)
        drift_rows.append(
            {
                "window": f"{start}-{end}",
                "rows": perf["rows"],
                "fraud_rate": perf["fraud_rate"],
                "precision": perf["precision"],
                "recall": perf["recall"],
                "f1_score": perf["f1_score"],
                "roc_auc": perf["roc_auc"],
                "anomaly_rate": anomaly["anomaly_rate"],
                "mean_anomaly_score": anomaly["mean_anomaly_score"],
            }
        )

    recall_gain = final_post_metrics["recall"] - final_pre_metrics["recall"]
    f1_gain = final_post_metrics["f1_score"] - final_pre_metrics["f1_score"]
    auc_gain = (final_post_metrics["roc_auc"] or 0.0) - (final_pre_metrics["roc_auc"] or 0.0)

    metrics = {
        "experiment_design": {
            "initial_training": "steps 1-15",
            "stable_reference": "steps 16-20",
            "drift_observation": "steps 21-25",
            "final_holdout": "steps 26-30",
            "adaptation_training": "steps 1-25 after drift is observed",
            "drift_injection": "controlled evolving fraud behaviour begins after step 22",
            "drift_threshold": f"recall or ROC-AUC drop >= {DRIFT_METRIC_DROP_THRESHOLD:.0%}",
        },
        "total_rows": int(len(df)),
        "fraud_rate": float(df["isFraud"].mean()),
        "initial_train_rows": int(initial_train.sum()),
        "adaptation_train_rows": int(adaptation_train.sum()),
        "reference_metrics": reference_metrics,
        "drift_observation_metrics": drift_metrics,
        "pre_adaptation_future_metrics": future_pre_metrics,
        "pre_adaptation_final_metrics": final_pre_metrics,
        "post_adaptation_final_metrics": final_post_metrics,
        "drift_windows": drift_rows,
        "drift_detected": bool(drift_detected),
        "drift_status": "Drift signal detected — adaptation triggered" if drift_detected else "No major drift signal detected",
        "drift_reason": (
            f"Reference recall was {reference_metrics['recall']:.1%}. "
            f"It fell to {drift_metrics['recall']:.1%} in steps 21-25 "
            f"({recall_drop_at_drift:.1%} absolute drop). "
            f"ROC-AUC also fell from {reference_metrics['roc_auc']:.3f} "
            f"to {drift_metrics['roc_auc']:.3f} ({auc_drop_at_drift:.3f} drop)."
        ),
        "recall_drop_at_drift": float(recall_drop_at_drift),
        "roc_auc_drop_at_drift": float(auc_drop_at_drift),
        "recall_drop_final_pre_adaptation": float(recall_drop_final),
        "adaptation_gain": {
            "recall": float(recall_gain),
            "f1_score": float(f1_gain),
            "roc_auc": float(auc_gain),
        },
        # Backward-compatible headline keys now report the adapted model on
        # the untouched final holdout rather than a random split.
        "accuracy": final_post_metrics["accuracy"],
        "precision": final_post_metrics["precision"],
        "recall": final_post_metrics["recall"],
        "f1_score": final_post_metrics["f1_score"],
        "roc_auc": final_post_metrics["roc_auc"],
        "confusion_matrix": final_post_metrics["confusion_matrix"],
        "train_rows": int(adaptation_train.sum()),
        "test_rows": int(final_holdout.sum()),
    }

    # Save both model generations so the dashboard can demonstrate adaptation.
    joblib.dump(baseline_model, MODELS / "baseline_fraud_model.pkl")
    joblib.dump(baseline_scaler, MODELS / "baseline_scaler.pkl")
    joblib.dump(adaptive_model, MODELS / "fraud_model.pkl")
    joblib.dump(adaptive_anomaly_model, MODELS / "anomaly_model.pkl")
    joblib.dump(adaptive_scaler, MODELS / "scaler.pkl")
    (MODELS / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Done")
    print("Rows:", metrics["total_rows"])
    print("Overall fraud rate:", f"{metrics['fraud_rate']:.2%}")
    print("Reference recall (16-20):", f"{reference_metrics['recall']:.3f}")
    print("Drift-window recall (21-25):", f"{drift_metrics['recall']:.3f}")
    print("Final recall before adaptation (26-30):", f"{final_pre_metrics['recall']:.3f}")
    print("Final recall after adaptation (26-30):", f"{final_post_metrics['recall']:.3f}")
    print("Final F1 before adaptation:", f"{final_pre_metrics['f1_score']:.3f}")
    print("Final F1 after adaptation:", f"{final_post_metrics['f1_score']:.3f}")
    print("Drift:", metrics["drift_status"])


if __name__ == "__main__":
    main()
