# Adaptive AI Fraud Detection MVP

Working Streamlit prototype for the COMP702 CA2 project video.

## Project title
An Adaptive AI Framework for Detecting Evolving Fraud Tactics Using Machine Learning, Anomaly Detection and Concept Drift Monitoring

## What it demonstrates
- Synthetic PaySim-style transaction data
- Supervised machine learning fraud prediction
- Anomaly detection using Isolation Forest
- Simple concept drift monitoring through rolling time windows
- Model evaluation metrics
- Streamlit dashboard for the video demo

## Run instructions

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

On Windows, you can also double-click `START_DASHBOARD_WINDOWS.bat`.

## Academic note
This is a prototype demonstration using synthetic data. It does not use real customer data and is not a production banking system.
