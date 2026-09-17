from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "synthetic_paysim_transactions.csv"
MODEL_DIR = BASE / "models"
ARCHITECTURE_PATH = BASE / "assets" / "architecture.png"

st.set_page_config(page_title="Adaptive AI Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    max-width: 1200px;
}
.good {background:#DCFCE7;color:#166534;border:1px solid #16A34A;}
.warn {background:#FEF3C7;color:#92400E;border:1px solid #D97706;}
.bad {background:#FFE4E6;color:#BE123C;border:1px solid #E11D48;}
.action {font-size:1.35rem;font-weight:800;text-align:center;padding:14px;border-radius:16px;}
.status-card {padding:14px 16px;border-radius:14px;border:1px solid #D97706;background:#FEF3C7;color:#92400E;}
.status-card strong {font-size:1.05rem;}
.small-note {font-size:0.92rem;opacity:0.86;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    required = [
        DATA_PATH,
        MODEL_DIR / "fraud_model.pkl",
        MODEL_DIR / "anomaly_model.pkl",
        MODEL_DIR / "scaler.pkl",
        MODEL_DIR / "baseline_fraud_model.pkl",
        MODEL_DIR / "baseline_scaler.pkl",
        MODEL_DIR / "feature_columns.json",
        MODEL_DIR / "metrics.json",
    ]
    if not all(p.exists() for p in required):
        st.error("Missing or outdated model files. Run: python train_model.py")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    adaptive_model = joblib.load(MODEL_DIR / "fraud_model.pkl")
    anomaly_model = joblib.load(MODEL_DIR / "anomaly_model.pkl")
    adaptive_scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    baseline_model = joblib.load(MODEL_DIR / "baseline_fraud_model.pkl")
    baseline_scaler = joblib.load(MODEL_DIR / "baseline_scaler.pkl")
    features = json.loads((MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    return (
        df,
        adaptive_model,
        anomaly_model,
        adaptive_scaler,
        baseline_model,
        baseline_scaler,
        features,
        metrics,
    )


def prepare_input(row, feature_columns):
    data = pd.DataFrame([row])
    data["balanceChangeOrig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balanceChangeDest"] = data["newbalanceDest"] - data["oldbalanceDest"]
    data["amountToOldBalanceRatio"] = data["amount"] / (data["oldbalanceOrg"] + 1)
    data["isRiskyType"] = data["type"].isin(["TRANSFER", "CASH_OUT"]).astype(int)
    data = pd.get_dummies(data, columns=["type"], drop_first=False)
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    return data[feature_columns]


def explain(row, fraud_prob, anomaly_score):
    reasons = []
    if row["type"] in ["TRANSFER", "CASH_OUT"]:
        reasons.append("Risky transaction type: TRANSFER/CASH_OUT transactions are treated as higher-risk patterns.")
    if row["amount"] > 50000:
        reasons.append("High transaction amount compared with many normal transactions.")
    if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] < row["oldbalanceOrg"] * 0.05:
        reasons.append("Origin account appears almost fully drained after the transaction.")
    if row["amount"] / (row["oldbalanceOrg"] + 1) > 0.85:
        reasons.append("Transaction amount is large compared with the original account balance.")
    if anomaly_score < 0:
        reasons.append("Anomaly detector considers the transaction unusual compared with learned normal behaviour.")
    if fraud_prob > 0.70:
        reasons.append("The adapted supervised model produced a high fraud probability.")
    return reasons or ["No major risk factor was strongly triggered."]


def get_action(prob, anomaly_score, drift_status):
    if prob >= 0.70 or anomaly_score < -0.05:
        return "Manual Review", "bad"
    if prob >= 0.35 or anomaly_score < 0 or "Drift detected" in drift_status:
        return "Monitor", "warn"
    return "Approve", "good"


def confusion_fig(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    mat = np.array(cm)
    ax.imshow(mat)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def pct_point(value):
    return f"{value * 100:+.1f} pp"


def performance_through_time_fig(drift):
    """Plot chronological baseline performance without categorical re-sorting."""
    ordered = drift.copy().reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for metric, label in [("precision", "Precision"), ("recall", "Recall"), ("f1_score", "F1-score")]:
        ax.plot(x, ordered[metric], marker="o", linewidth=2, label=label)
    ax.axvline(3.5, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(3.55, 0.06, "Drift-observation period", rotation=90, va="bottom", fontsize=9)
    ax.set_xticks(x, ordered["window"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Chronological time window")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", ncol=3, frameon=False)
    fig.tight_layout()
    return fig


def anomaly_through_time_fig(drift):
    """Plot anomaly rate in the same explicit chronological order."""
    ordered = drift.copy().reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(x, ordered["anomaly_rate"], marker="o", linewidth=2)
    ax.axvline(3.5, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xticks(x, ordered["window"].tolist())
    ax.set_ylabel("Anomaly rate")
    ax.set_xlabel("Chronological time window")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def cm_counts(cm):
    mat = np.asarray(cm, dtype=int)
    tn, fp, fn, tp = mat.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


(
    df,
    fraud_model,
    anomaly_model,
    scaler,
    baseline_model,
    baseline_scaler,
    feature_columns,
    metrics,
) = load_artifacts()

st.sidebar.title("🛡️ Adaptive Fraud AI")
page = st.sidebar.radio(
    "Choose section",
    [
        "Project Overview",
        "Dataset Explorer",
        "Fraud Prediction Demo",
        "Anomaly & Drift Monitor",
        "Model Evaluation",
        "Development Files",
    ],
)
st.sidebar.caption("COMP702 adaptive fraud-detection prototype")

st.title("Adaptive AI Fraud Detection Dashboard")
st.markdown("**Machine learning + anomaly detection + chronological concept-drift monitoring for evolving fraud tactics.**")

if page == "Project Overview":
    st.subheader("Purpose")
    st.write(
        "This prototype combines supervised fraud classification, anomaly detection and "
        "performance-based drift monitoring. A chronological experiment is used so future "
        "transactions are not leaked into the initial model training."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Known fraud detection", "Random Forest")
    c2.metric("Emerging behaviour", "Isolation Forest")
    if metrics.get("drift_detected"):
        c3.metric("Drift signal", "Detected", delta="Adaptation triggered", delta_color="off")
    else:
        c3.metric("Drift signal", "Not detected", delta="No retraining trigger", delta_color="off")

    st.subheader("Chronological experiment")
    design = metrics.get("experiment_design", {})
    st.markdown(
        f"**Initial training:** {design.get('initial_training', 'steps 1-15')}  →  "
        f"**Reference:** {design.get('stable_reference', 'steps 16-20')}  →  "
        f"**Drift observation:** {design.get('drift_observation', 'steps 21-25')}  →  "
        f"**Final holdout:** {design.get('final_holdout', 'steps 26-30')}"
    )
    st.caption(
        "After a drift signal is observed, the model is retrained using labelled data available through step 25. "
        "Steps 26-30 remain untouched for the final before-vs-after evaluation."
    )

    pre = metrics.get("pre_adaptation_final_metrics", {})
    post = metrics.get("post_adaptation_final_metrics", {})
    if pre and post:
        st.info(
            f"**Final holdout result:** fraud recall improves from {pre['recall']:.1%} to {post['recall']:.1%} "
            f"after adaptation, while ROC-AUC improves from {pre['roc_auc']:.3f} to {post['roc_auc']:.3f}."
        )

    st.subheader("System architecture")
    if ARCHITECTURE_PATH.exists():
        st.image(str(ARCHITECTURE_PATH), use_container_width=True)
    st.success("Ethics: synthetic demonstration data only; no real customer information or human participants.")

elif page == "Dataset Explorer":
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Fraud cases", f"{int(df['isFraud'].sum()):,}")
    c3.metric("Fraud rate", f"{df['isFraud'].mean():.2%}")
    c4.metric("Time steps", f"{df['step'].min()}–{df['step'].max()}")

    st.dataframe(df.head(25), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.write("Fraud vs legitimate")
        st.bar_chart(df["isFraud"].map({0: "Legitimate", 1: "Fraud"}).value_counts())
    with right:
        st.write("Transaction type distribution")
        st.bar_chart(df["type"].value_counts())

    st.subheader("Fraud rate through time")
    fraud_time = df.groupby("step")["isFraud"].mean().rename("fraud_rate")
    st.line_chart(fraud_time)
    st.caption("The synthetic experiment intentionally introduces evolving fraud behaviour after step 22.")

elif page == "Fraud Prediction Demo":
    st.subheader("Transaction-level prediction")
    st.write(
        "The same transaction is scored by the original historical model and the adapted model. "
        "This makes the effect of retraining visible in the live prototype."
    )

    left, right = st.columns(2)
    with left:
        tx_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = st.number_input("Amount", min_value=1.0, max_value=250000.0, value=85000.0, step=500.0)
        step = st.slider("Time step", 1, 30, 28)
    with right:
        oldbalanceOrg = st.number_input("Old balance - origin account", min_value=0.0, value=90000.0, step=500.0)
        newbalanceOrig = st.number_input("New balance - origin account", min_value=0.0, value=0.0, step=500.0)
        oldbalanceDest = st.number_input("Old balance - destination account", min_value=0.0, value=15000.0, step=500.0)
        newbalanceDest = st.number_input("New balance - destination account", min_value=0.0, value=100000.0, step=500.0)

    row = {
        "step": step,
        "type": tx_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }
    X = prepare_input(row, feature_columns)

    baseline_prob = float(baseline_model.predict_proba(baseline_scaler.transform(X))[0, 1])
    Xs = scaler.transform(X)
    adapted_prob = float(fraud_model.predict_proba(Xs)[0, 1])
    anomaly_score = float(anomaly_model.decision_function(Xs)[0])
    action, cls = get_action(adapted_prob, anomaly_score, metrics.get("drift_status", ""))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original model probability", f"{baseline_prob:.1%}")
    c2.metric("Adapted model probability", f"{adapted_prob:.1%}", delta=f"{(adapted_prob-baseline_prob)*100:+.1f} pp")
    c3.metric("Anomaly score", f"{anomaly_score:.3f}")
    c4.metric("Adapted prediction", "Fraud" if adapted_prob >= 0.5 else "Legitimate")

    st.markdown(f'<div class="action {cls}">Recommended action: {action}</div>', unsafe_allow_html=True)
    st.subheader("Explanation")
    for reason in explain(row, adapted_prob, anomaly_score):
        st.write("- " + reason)

elif page == "Anomaly & Drift Monitor":
    st.subheader("Concept-drift monitoring")
    if metrics.get("drift_detected"):
        st.markdown(
            '<div class="status-card"><strong>⚠️ Drift signal detected</strong><br>'
            '<span class="small-note">Performance deterioration crossed the predefined threshold, so model adaptation was triggered.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No material drift signal detected against the stable reference period.")
    st.write(metrics.get("drift_reason", ""))

    drift = pd.DataFrame(metrics.get("drift_windows", []))
    if not drift.empty:
        display_cols = ["window", "rows", "fraud_rate", "precision", "recall", "f1_score", "roc_auc", "anomaly_rate"]
        st.dataframe(
            drift[display_cols].style.format(
                {
                    "fraud_rate": "{:.2%}",
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1_score": "{:.2%}",
                    "roc_auc": "{:.3f}",
                    "anomaly_rate": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        left, right = st.columns(2)
        with left:
            st.write("Baseline model performance through time")
            st.pyplot(performance_through_time_fig(drift), use_container_width=True)
        with right:
            st.write("Unsupervised anomaly rate through time")
            st.pyplot(anomaly_through_time_fig(drift), use_container_width=True)

    st.caption(
        "The monitoring rule treats a material fall in labelled fraud recall or ROC-AUC relative to the stable reference period "
        "as a **drift signal**. This is evidence of changing model/data behaviour, not proof of the underlying cause. "
        "The anomaly rate is a complementary unsupervised indicator."
    )

elif page == "Model Evaluation":
    st.subheader("Final untouched holdout: before vs after adaptation")
    pre = metrics["pre_adaptation_final_metrics"]
    post = metrics["post_adaptation_final_metrics"]
    gain = metrics.get("adaptation_gain", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud recall", f"{post['recall']:.2%}", delta=pct_point(gain.get("recall", 0.0)))
    c2.metric("F1-score", f"{post['f1_score']:.2%}", delta=pct_point(gain.get("f1_score", 0.0)))
    c3.metric("ROC-AUC", f"{post['roc_auc']:.3f}", delta=f"{gain.get('roc_auc', 0.0):+.3f}")

    comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
            "Before adaptation": [pre["accuracy"], pre["precision"], pre["recall"], pre["f1_score"], pre["roc_auc"]],
            "After adaptation": [post["accuracy"], post["precision"], post["recall"], post["f1_score"], post["roc_auc"]],
        }
    ).set_index("Metric")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)

    before_counts = cm_counts(pre["confusion_matrix"])
    after_counts = cm_counts(post["confusion_matrix"])
    st.subheader("Operational interpretation")
    o1, o2, o3 = st.columns(3)
    o1.metric(
        "Fraud cases detected",
        f"{after_counts['tp']} / {post['fraud_cases']}",
        delta=f"{after_counts['tp'] - before_counts['tp']:+d} cases",
    )
    o2.metric(
        "Fraud cases missed",
        f"{after_counts['fn']}",
        delta=f"{after_counts['fn'] - before_counts['fn']:+d} cases",
        delta_color="inverse",
    )
    o3.metric(
        "False alarms",
        f"{after_counts['fp']}",
        delta=f"{after_counts['fp'] - before_counts['fp']:+d} cases",
        delta_color="inverse",
    )
    st.caption(
        "The adapted model is evaluated on exactly the same untouched final period. The comparison therefore makes the "
        "benefit and cost of adaptation visible: more fraud is detected, potentially at the expense of additional false positives."
    )

    left, right = st.columns(2)
    with left:
        st.pyplot(confusion_fig(pre["confusion_matrix"], "Before adaptation: steps 26-30"))
    with right:
        st.pyplot(confusion_fig(post["confusion_matrix"], "After adaptation: steps 26-30"))

    st.success(
        f"On the same untouched final period, recall improves from {pre['recall']:.1%} to {post['recall']:.1%} "
        f"and ROC-AUC improves from {pre['roc_auc']:.3f} to {post['roc_auc']:.3f}. "
        f"The adapted model detects {after_counts['tp'] - before_counts['tp']} additional fraud cases while producing "
        f"{after_counts['fp'] - before_counts['fp']} additional false positives."
    )
    st.caption(
        "Accuracy is reported but is not treated as the main metric because fraud is imbalanced. "
        "Recall, F1-score and ROC-AUC are more informative for the project evaluation."
    )

elif page == "Development Files":
    st.subheader("Development environment")
    st.code(
        """adaptive-fraud-detection-main/
├── app.py                         # Streamlit dashboard
├── train_model.py                 # chronological ML, anomaly detection and adaptation experiment
├── requirements.txt               # required packages
├── data/                          # synthetic PaySim-style transaction data
├── models/
│   ├── baseline_fraud_model.pkl   # original model trained on steps 1-15
│   ├── baseline_scaler.pkl
│   ├── fraud_model.pkl            # adapted model trained through step 25
│   ├── anomaly_model.pkl
│   ├── scaler.pkl
│   └── metrics.json
└── assets/                        # architecture diagram""",
        language="text",
    )
    st.write(
        "The experiment preserves time order. The original model is evaluated on future windows, "
        "a drift signal is raised when monitored performance deteriorates beyond the predefined threshold, "
        "and retraining is tested on a final holdout that was not used during adaptation."
    )
