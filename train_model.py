from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
MODELS = BASE / 'models'
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

def generate_data(n=6000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    tx_types = np.array(['CASH_OUT','TRANSFER','PAYMENT','CASH_IN','DEBIT'])
    tx_probs = np.array([0.28,0.22,0.30,0.15,0.05])
    step = rng.integers(1, 31, size=n)
    tx_type = rng.choice(tx_types, size=n, p=tx_probs)
    amount = np.clip(rng.lognormal(mean=8.0, sigma=1.05, size=n), 20, 250000)
    old_org = np.clip(rng.lognormal(mean=9.0, sigma=1.1, size=n), 0, 450000)
    drop = rng.beta(2, 6, size=n)
    new_org = np.clip(old_org - old_org * drop, 0, None)
    old_dest = np.clip(rng.lognormal(mean=9.4, sigma=1.0, size=n), 0, 700000)
    new_dest = old_dest + amount * rng.uniform(0.1, 1.0, size=n)
    risky_type = np.isin(tx_type, ['TRANSFER','CASH_OUT']).astype(int)
    high_amount = (amount > np.quantile(amount, 0.88)).astype(int)
    drained = ((old_org > 0) & (new_org < old_org * 0.05)).astype(int)
    late_period = (step > 22).astype(int)
    # Synthetic fraud logic: clear enough for a prototype, but still imperfect.
    # Fraud risk increases for risky transaction types, high amounts and account-draining behaviour.
    risk_score = (
        0.55*risky_type
        + 0.45*high_amount
        + 0.75*drained
        + 0.20*late_period
        + rng.normal(0, 0.12, size=n)
    )
    fraud = (risk_score > 1.02).astype(int)

    # Add evolving late-period fraud behaviour for the drift demo.
    medium_high_amount = (amount > np.quantile(amount, 0.70)).astype(int)
    evolving = (step > 22) & np.isin(tx_type, ['TRANSFER','CASH_OUT']) & (medium_high_amount == 1) & (rng.random(n) < 0.18)
    fraud = np.where(evolving, 1, fraud)

    # Small noise: some cases are difficult, as real fraud detection is never perfect.
    flip_to_normal = np.where((fraud == 1) & (rng.random(n) < 0.04))[0]
    fraud[flip_to_normal] = 0
    return pd.DataFrame({
        'step': step,
        'type': tx_type,
        'amount': amount.round(2),
        'oldbalanceOrg': old_org.round(2),
        'newbalanceOrig': new_org.round(2),
        'oldbalanceDest': old_dest.round(2),
        'newbalanceDest': new_dest.round(2),
        'isFraud': fraud
    })

def prepare_features(df):
    x = df.copy()
    x['balanceChangeOrig'] = x['oldbalanceOrg'] - x['newbalanceOrig']
    x['balanceChangeDest'] = x['newbalanceDest'] - x['oldbalanceDest']
    x['amountToOldBalanceRatio'] = x['amount'] / (x['oldbalanceOrg'] + 1)
    x['isRiskyType'] = x['type'].isin(['TRANSFER','CASH_OUT']).astype(int)
    x = pd.get_dummies(x, columns=['type'], drop_first=False)
    y = x['isFraud'].astype(int)
    X = x.drop(columns=['isFraud'])
    return X, y

def main():
    print('Generating synthetic PaySim-style dataset...')
    df = generate_data()
    df.to_csv(DATA/'synthetic_paysim_transactions.csv', index=False)
    X, y = prepare_features(df)
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print('Training supervised fraud model...')
    fraud_model = RandomForestClassifier(n_estimators=160, max_depth=10, min_samples_leaf=3, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    fraud_model.fit(X_train_s, y_train)
    print('Training anomaly detector...')
    normal_train = X_train_s[y_train == 0]
    anomaly_model = IsolationForest(n_estimators=120, contamination=0.08, random_state=RANDOM_STATE)
    anomaly_model.fit(normal_train)
    pred = fraud_model.predict(X_test_s)
    prob = fraud_model.predict_proba(X_test_s)[:,1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, prob)),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
        'total_rows': int(len(df)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'fraud_rate': float(df['isFraud'].mean())
    }
    drift_rows = []
    for start in range(1, 31, 5):
        end = start + 4
        m = (df['step'] >= start) & (df['step'] <= end)
        if m.sum() == 0:
            continue
        Xw, yw = X.loc[m], y.loc[m]
        pw = fraud_model.predict(scaler.transform(Xw))
        drift_rows.append({
            'window': f'{start}-{end}',
            'rows': int(m.sum()),
            'fraud_rate': float(yw.mean()),
            'mean_amount': float(df.loc[m, 'amount'].mean()),
            'precision': float(precision_score(yw, pw, zero_division=0)),
            'recall': float(recall_score(yw, pw, zero_division=0)),
            'f1_score': float(f1_score(yw, pw, zero_division=0))
        })
    metrics['drift_windows'] = drift_rows
    if len(drift_rows) >= 2:
        recall_change = drift_rows[0]['recall'] - drift_rows[-1]['recall']
        amount_shift = abs(drift_rows[-1]['mean_amount'] - drift_rows[0]['mean_amount']) / max(drift_rows[0]['mean_amount'], 1)
        metrics['drift_status'] = 'Possible drift detected' if recall_change > 0.15 or amount_shift > 0.25 else 'No major drift detected'
        metrics['drift_reason'] = f'Recall change: {recall_change:.2f}; mean transaction amount shift: {amount_shift:.2%}.'
    else:
        metrics['drift_status'] = 'Not enough windows'
        metrics['drift_reason'] = 'More time windows are needed.'
    joblib.dump(fraud_model, MODELS/'fraud_model.pkl')
    joblib.dump(anomaly_model, MODELS/'anomaly_model.pkl')
    joblib.dump(scaler, MODELS/'scaler.pkl')
    (MODELS/'feature_columns.json').write_text(json.dumps(features, indent=2), encoding='utf-8')
    (MODELS/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Done')
    print('Rows:', metrics['total_rows'])
    print('Fraud rate:', f"{metrics['fraud_rate']:.2%}")
    print('Precision:', f"{metrics['precision']:.3f}")
    print('Recall:', f"{metrics['recall']:.3f}")
    print('F1:', f"{metrics['f1_score']:.3f}")
    print('ROC-AUC:', f"{metrics['roc_auc']:.3f}")
    print('Drift:', metrics['drift_status'])

if __name__ == '__main__':
    main()
