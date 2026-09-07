"""
FastAPI service for the Parkinson's voice screening model.

Run with: uvicorn main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Parkinson's Voice Screening API",
    description="Screening aid based on acoustic voice features. NOT a diagnostic tool — "
                 "flags elevated risk worth discussing with a doctor.",
    version="1.0.0",
)

# Load model artifacts once at startup, not per-request.
model = joblib.load("parkinsons_voice_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_ORDER = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]


class VoiceFeatures(BaseModel):
    """Input schema: the 5 acoustic features the model was trained on."""
    mdvp_fo_hz: float = Field(..., description="Average vocal fundamental frequency (Hz)", example=154.2)
    mdvp_jitter_pct: float = Field(..., description="MDVP jitter (%), frequency variation", example=0.005)
    mdvp_shimmer: float = Field(..., description="MDVP shimmer, amplitude variation", example=0.03)
    hnr: float = Field(..., description="Harmonics-to-noise ratio", example=21.5)
    ppe: float = Field(..., description="Pitch period entropy", example=0.2)


class PredictionResponse(BaseModel):
    prediction: str
    probability_parkinsons: float
    disclaimer: str = (
        "This is a screening aid, not a medical diagnosis. "
        "Please consult a qualified healthcare professional for any health concerns."
    )


@app.get("/")
def root():
    return {"message": "Parkinson's Voice Screening API is running. See /docs for usage."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: VoiceFeatures):
    try:
        X = pd.DataFrame([[
            features.mdvp_fo_hz,
            features.mdvp_jitter_pct,
            features.mdvp_shimmer,
            features.hnr,
            features.ppe,
        ]], columns=FEATURE_ORDER)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]  # probability of class 1 (Parkinson's)

        return PredictionResponse(
            prediction="Elevated risk indicators detected" if prediction == 1 else "No elevated risk indicators detected",
            probability_parkinsons=round(float(probability), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")