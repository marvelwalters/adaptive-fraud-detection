from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE/'data'/'synthetic_paysim_transactions.csv'
MODEL_DIR = BASE/'models'
ARCHITECTURE_PATH = BASE/'assets'/'architecture.png'

st.set_page_config(page_title='Adaptive AI Fraud Detection', page_icon='🛡️', layout='wide')

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    max-width: 1200px;
}

.big-title {
    font-size: 2.2rem;
    font-weight: 900;
    color: #0F172A;
    line-height: 1.15;
    text-align: center;
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}

.subtle {
    font-size: 1.05rem;
    color: #475569;
    text-align: center;
    font-weight: 600;
    margin-bottom: 1.3rem;
}
.good {background:#DCFCE7;color:#166534;border:1px solid #16A34A;}
.warn {background:#FEF3C7;color:#92400E;border:1px solid #D97706;}
.bad {background:#FFE4E6;color:#BE123C;border:1px solid #E11D48;}
.action {font-size:1.35rem;font-weight:800;text-align:center;padding:14px;border-radius:16px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    if not DATA_PATH.exists() or not (MODEL_DIR/'fraud_model.pkl').exists():
        st.error('Missing model files. Run: python train_model.py')
        st.stop()
    df = pd.read_csv(DATA_PATH)
    fraud_model = joblib.load(MODEL_DIR/'fraud_model.pkl')
    anomaly_model = joblib.load(MODEL_DIR/'anomaly_model.pkl')
    scaler = joblib.load(MODEL_DIR/'scaler.pkl')
    features = json.loads((MODEL_DIR/'feature_columns.json').read_text(encoding='utf-8'))
    metrics = json.loads((MODEL_DIR/'metrics.json').read_text(encoding='utf-8'))
    return df, fraud_model, anomaly_model, scaler, features, metrics

def prepare_input(row, feature_columns):
    data = pd.DataFrame([row])
    data['balanceChangeOrig'] = data['oldbalanceOrg'] - data['newbalanceOrig']
    data['balanceChangeDest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['amountToOldBalanceRatio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
    data['isRiskyType'] = data['type'].isin(['TRANSFER','CASH_OUT']).astype(int)
    data = pd.get_dummies(data, columns=['type'], drop_first=False)
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0
    return data[feature_columns]

def explain(row, fraud_prob, anomaly_score):
    reasons = []
    if row['type'] in ['TRANSFER','CASH_OUT']:
        reasons.append('Risky transaction type: TRANSFER/CASH_OUT transactions are treated as higher-risk patterns.')
    if row['amount'] > 50000:
        reasons.append('High transaction amount compared with many normal transactions.')
    if row['oldbalanceOrg'] > 0 and row['newbalanceOrig'] < row['oldbalanceOrg'] * 0.05:
        reasons.append('Origin account appears almost fully drained after the transaction.')
    if row['amount'] / (row['oldbalanceOrg'] + 1) > 0.85:
        reasons.append('Transaction amount is large compared with the original account balance.')
    if anomaly_score < 0:
        reasons.append('Anomaly detector considers the transaction unusual compared with normal behaviour.')
    if fraud_prob > 0.70:
        reasons.append('Supervised model produced a high fraud probability.')
    return reasons or ['No major risk factor was strongly triggered.']

def get_action(prob, anomaly_score, drift_status):
    if prob >= 0.70 or anomaly_score < -0.05:
        return 'Manual Review', 'bad'
    if prob >= 0.35 or anomaly_score < 0 or 'Possible' in drift_status:
        return 'Monitor', 'warn'
    return 'Approve', 'good'

def confusion_fig(cm):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    mat = np.array(cm)
    ax.imshow(mat)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1], ['Legitimate', 'Fraud'])
    ax.set_yticks([0, 1], ['Legitimate', 'Fraud'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig

df, fraud_model, anomaly_model, scaler, feature_columns, metrics = load_artifacts()

st.sidebar.title('🛡️ Adaptive Fraud AI')
page = st.sidebar.radio('Choose section', ['Project Overview','Dataset Explorer','Fraud Prediction Demo','Anomaly & Drift Monitor','Model Evaluation','Development Files'])
st.sidebar.caption('COMP702 CA2 video demo prototype')

st.title("Adaptive AI Fraud Detection Dashboard")

st.markdown(
    "**Machine learning + anomaly detection + concept drift monitoring for evolving fraud tactics.**"
)

if page == 'Project Overview':
    st.subheader('Purpose')
    st.write('This software prototype demonstrates a fraud detection framework that combines supervised ML, anomaly detection and concept drift monitoring. It is designed as a decision-support tool rather than a production banking system.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Known fraud detection', 'Supervised ML')
    c2.metric('Emerging behaviour', 'Anomaly detection')
    c3.metric('Adaptation signal', metrics.get('drift_status', 'Checked'))
    st.subheader('System architecture')
    if ARCHITECTURE_PATH.exists():
        st.image(str(ARCHITECTURE_PATH), use_container_width=True)
    st.success('Ethics: synthetic demonstration data, no real personal or customer information, no human participants.')

elif page == 'Dataset Explorer':
    st.subheader('Dataset overview')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Rows', f"{len(df):,}")
    c2.metric('Fraud cases', f"{int(df['isFraud'].sum()):,}")
    c3.metric('Fraud rate', f"{df['isFraud'].mean():.2%}")
    c4.metric('Time steps', f"{df['step'].min()}–{df['step'].max()}")
    st.dataframe(df.head(25), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.write('Fraud vs legitimate')
        st.bar_chart(df['isFraud'].map({0:'Legitimate',1:'Fraud'}).value_counts())
    with right:
        st.write('Transaction type distribution')
        st.bar_chart(df['type'].value_counts())

elif page == 'Fraud Prediction Demo':
    st.subheader('Transaction-level prediction')
    st.write('Use the default example for a high-risk transaction, then adjust values to show how the model response changes.')
    left, right = st.columns(2)
    with left:
        tx_type = st.selectbox('Transaction type', ['TRANSFER','CASH_OUT','PAYMENT','CASH_IN','DEBIT'])
        amount = st.number_input('Amount', min_value=1.0, max_value=250000.0, value=85000.0, step=500.0)
        step = st.slider('Time step', 1, 30, 25)
    with right:
        oldbalanceOrg = st.number_input('Old balance - origin account', min_value=0.0, value=90000.0, step=500.0)
        newbalanceOrig = st.number_input('New balance - origin account', min_value=0.0, value=0.0, step=500.0)
        oldbalanceDest = st.number_input('Old balance - destination account', min_value=0.0, value=15000.0, step=500.0)
        newbalanceDest = st.number_input('New balance - destination account', min_value=0.0, value=100000.0, step=500.0)
    row = {'step': step, 'type': tx_type, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg, 'newbalanceOrig': newbalanceOrig, 'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest}
    X = prepare_input(row, feature_columns)
    Xs = scaler.transform(X)
    prob = float(fraud_model.predict_proba(Xs)[0,1])
    anomaly_score = float(anomaly_model.decision_function(Xs)[0])
    action, cls = get_action(prob, anomaly_score, metrics.get('drift_status',''))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Fraud probability', f'{prob:.1%}')
    c2.metric('Model prediction', 'Fraud' if prob >= 0.5 else 'Legitimate')
    c3.metric('Anomaly score', f'{anomaly_score:.3f}')
    c4.metric('Drift status', metrics.get('drift_status','Unknown'))
    st.markdown(f'<div class="action {cls}">Recommended action: {action}</div>', unsafe_allow_html=True)
    st.subheader('Explanation')
    for r in explain(row, prob, anomaly_score):
        st.write('- ' + r)

elif page == 'Anomaly & Drift Monitor':
    st.subheader('Anomaly detection')
    sample = df.sample(300, random_state=7)
    Xs = pd.concat([prepare_input(r.to_dict(), feature_columns) for _, r in sample.iterrows()], ignore_index=True)
    scores = anomaly_model.decision_function(scaler.transform(Xs))
    st.line_chart(pd.DataFrame({'anomaly_score': scores}))
    st.subheader('Concept drift monitor')
    st.info(metrics.get('drift_status', 'Unknown'))
    st.write(metrics.get('drift_reason', ''))
    drift = pd.DataFrame(metrics.get('drift_windows', []))
    if not drift.empty:
        st.dataframe(drift, use_container_width=True)
        st.line_chart(drift.set_index('window')[['precision','recall','f1_score']])

elif page == 'Model Evaluation':
    st.subheader('Model evaluation metrics')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Accuracy', f"{metrics['accuracy']:.2%}")
    c2.metric('Precision', f"{metrics['precision']:.2%}")
    c3.metric('Recall', f"{metrics['recall']:.2%}")
    c4.metric('F1-score', f"{metrics['f1_score']:.2%}")
    c5.metric('ROC-AUC', f"{metrics['roc_auc']:.2%}")
    st.pyplot(confusion_fig(metrics['confusion_matrix']))
    st.write('These metrics will support the final dissertation evaluation, especially because fraud data is imbalanced.')

elif page == 'Development Files':
    st.subheader('Development environment to show in the video')
    st.code('''adaptive_fraud_detection_mvp/
├── app.py                 # Streamlit dashboard
├── train_model.py         # data generation, ML training, anomaly detection, drift metrics
├── requirements.txt       # required packages
├── data/                  # synthetic transaction data
├── models/                # saved ML models and metrics
└── assets/                # architecture diagram''', language='text')
    st.write('The important code files are `train_model.py` and `app.py`. The model is trained first, saved with joblib, and then loaded into the dashboard for live transaction prediction.')
