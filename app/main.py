from fastapi import FastAPI, Request
import xgboost as xgb
import pandas as pd
import uvicorn
import joblib

# Load saved encoders
encoders = joblib.load("encoders.pkl")

def apply_encoding(df, encoders):
    df_copy = df.copy()
    for col, categories in encoders.items():
        # Convert to categorical with the original categories
        cat_col = pd.Categorical(df_copy[col], categories=categories)
        # Convert to integer codes
        df_copy[col] = cat_col.codes.astype(int)
    return df_copy

app = FastAPI()
model = xgb.Booster()
model.load_model("xgb_model.json")

@app.post("/predict")
async def predict(data: dict):
    # Step 1: convert dict to DataFrame
    df = pd.DataFrame([data])  # single row

    # Step 2: encode categorical columns
    df_encoded = apply_encoding(df, encoders)

    # Step 3: convert to DMatrix
    dmatrix = xgb.DMatrix(df_encoded)

    # Step 4: predict
    prediction = model.predict(dmatrix)[0]
    return {"prediction": float(prediction)}