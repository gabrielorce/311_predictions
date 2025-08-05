import os
import ssl
import urllib.request
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# pip instal evidently

year=2023
# ================================
# Configuration
# ================================
DATA_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv$where=created_date%20between%20'{year}-01-01T00:00:00'%20and%20'{year}-12-31T23:59:59'"
DATA_FILE = "311_Service_Requests_2023.csv"
EXPERIMENT_NAME = "NYC311_XGBoost"
MODEL_OUTPUT = "xgb_model.json"
REPORT_OUTPUT = "evidently_report.html"

# Fix SSL issue on Windows
ssl._create_default_https_context = ssl._create_unverified_context

# ================================
# Step 1: Download the Data
# ================================
def download_data():
    if not os.path.exists(DATA_FILE):
        print("Downloading data...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        except Exception as e:
            raise RuntimeError(f"Error downloading or processing data: {e}")
    print("Data downloaded.")

# ================================
# Step 2: Preprocess Data
# ================================
def preprocess():
    df = pd.read_csv(DATA_FILE, parse_dates=['created_date', 'closed_date'], low_memory=False)

    features = ['complaint_type', 'borough', 'agency', 'incident_zip', 'created_date', 'closed_date']
    target = "resolved_within_7days"

    # Basic feature engineering
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["created_hour"] = df["created_date"].dt.hour
    df["created_weekday"] = df["created_date"].dt.dayofweek
    
    # Target: resolved_within_7days
    df = df.dropna(subset=['created_date', 'closed_date'])
    df['resolution_time'] = (df['closed_date'] - df['created_date']).dt.days
    df['resolved_within_7days'] = (df['resolution_time'] <= 7).astype(int)

#    df = df.dropna(subset=features)
    df = df[features + ['resolution_time'] + ['resolved_within_7days']].dropna()

    # Drop original date to prevent leakage
#    df = df.drop(columns=['created_date'])

    # Encode categoricals
    for col in ['complaint_type', 'borough', 'agency', 'incident_zip']: # 'Zip Code']:
        df[col] = df[col].astype('category').cat.codes

    processed_features = ['complaint_type', 'borough', 'agency', 'incident_zip']
    X = df[processed_features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Step 3: Train & Track with MLflow
# ================================
def train_with_mlflow(X_train, X_test, y_train, y_test):
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.xgboost.autolog()

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)

        mlflow.log_metric("rmse", rmse)
        model.save_model(MODEL_OUTPUT)
        mlflow.log_artifact(MODEL_OUTPUT)

        print(f"Model RMSE: {rmse:.2f}")

    return model, X_train, X_test, y_train, y_test

# ================================
# Step 4: Generate Evidently Report
# ================================
def generate_evidently_report(X_train, X_test, y_train, y_test):
    train_data = X_train.copy()
    train_data["target"] = y_train
    test_data = X_test.copy()
    test_data["target"] = y_test

    report = Report(metrics=[ DataDriftPreset()])
    my_report=report.run(reference_data=train_data, current_data=test_data)
    my_report.save_html(REPORT_OUTPUT)
    print(f"Evidently report saved to {REPORT_OUTPUT}")

# ================================
# Main
# ================================
if __name__ == "__main__":
#    download_data()
    X_train, X_test, y_train, y_test = preprocess()
    model, X_train, X_test, y_train, y_test = train_with_mlflow(X_train, X_test, y_train, y_test)
    generate_evidently_report(X_train, X_test, y_train, y_test)
