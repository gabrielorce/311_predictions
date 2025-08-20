from prometheus_client import start_http_server, Gauge
import time
import pandas as pd
from monitor_drift import psi

# Create Prometheus metrics
psi_metric = Gauge('psi_complaint_type', 'PSI for Complaint Type Feature')
roc_auc_metric = Gauge('roc_auc_metric', 'ROC AUC Score for XGBoost Model')

def update_metrics():
    # Load test/train data to simulate metrics
    train = pd.read_csv('train.csv')
    test = pd.read_csv('new_batch.csv')

    # Example metrics
    psi_val = psi(train['Complaint Type'], test['Complaint Type'])
    auc_val = 0.86  # Placeholder or load from log

    psi_metric.set(psi_val)
    roc_auc_metric.set(auc_val)

if __name__ == '__main__':
    start_http_server(9100)
    while True:
        update_metrics()
        time.sleep(60)