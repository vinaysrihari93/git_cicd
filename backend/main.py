from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Linear Regression API")

import os
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://git-cicd-1-ngtw.onrender.com",  # ← your Streamlit URL
        "http://localhost:8501",                 # for local testing
        "*"                                      # temporary wildcard for quick test (remove later)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL_PATH = r"D:\mlops_sample_cicid\backend\models\model.pkl" 
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


class InputData(BaseModel):
    area: float
    bedrooms: int


@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.area, data.bedrooms]])
    prediction = model.predict(X)[0]
    return {"predicted_price": float(prediction)}