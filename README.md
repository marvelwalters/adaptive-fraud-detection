# Adaptive AI Fraud Detection MVP

Working Streamlit prototype for the COMP702 project.

## Project title
An Adaptive AI Framework for Detecting Evolving Fraud Tactics Using Machine Learning, Anomaly Detection and Concept Drift Monitoring

## What it demonstrates
- Synthetic PaySim-style transaction data
- Supervised fraud prediction with Random Forest
- Anomaly detection using Isolation Forest
- Chronological performance-based concept-drift monitoring using an explicit drift signal
- A controlled evolving-fraud pattern after step 22
- Model adaptation/retraining after the drift signal is observed
- Before-vs-after evaluation on an untouched future holdout
- Streamlit decision-support dashboard

## Chronological experiment
The prototype deliberately avoids a random train/test split for the drift experiment:

1. **Steps 1-15:** initial model training
2. **Steps 16-20:** stable reference/validation period
3. **Steps 21-25:** drift observation period
4. **Adaptation:** retrain using labelled data available through step 25
5. **Steps 26-30:** untouched final holdout used to compare the original and adapted models

A reduction of at least 5 percentage points in either fraud recall or ROC-AUC relative to the stable reference period is used as a labelled **drift signal** and retraining trigger. This is evidence that model/data behaviour has changed, rather than proof of the underlying causal form of concept drift. The dashboard also displays Isolation Forest anomaly rates as a complementary unsupervised signal.

## Run instructions

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

On Windows, you can also double-click `START_DASHBOARD_WINDOWS.bat` after training the models.

## Academic note
This is a prototype demonstration using synthetic data. The evolving fraud pattern is intentionally injected so that adaptation can be evaluated under controlled conditions. It does not use real customer data and is not a production banking system.
