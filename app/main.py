from fastapi import FastAPI, Request
import xgboost as xgb
import pandas as pd
import uvicorn
import joblib

app = FastAPI()
model = xgb.Booster()
model.load_model("xgb_model.json")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    dmatrix = xgb.DMatrix(df)
    prediction = model.predict(dmatrix)[0]
    return {"prediction": float(prediction)}