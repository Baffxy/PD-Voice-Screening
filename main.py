"""
FastAPI service for the Parkinson's voice screening model.

Run with: uvicorn main:app --reload
Then visit http://127.0.0.1:8000/docs for interactive API documentation.
"""

import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from extract_features import extract_features
from db_logger import init_db, log_prediction, get_stats

app = FastAPI(
    title="Parkinson's Voice Screening API",
    description="Screening aid based on acoustic voice features. NOT a diagnostic tool — "
                 "flags elevated risk worth discussing with a doctor.",
    version="1.2.0",
)

# Load model artifacts once at startup, not per-request.
model = joblib.load("parkinsons_voice_model.pkl")
scaler = joblib.load("scaler.pkl")

# Create the predictions table if it doesn't exist yet. Safe to call every startup.
init_db()

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
    extracted_features: dict | None = None
    recording_caveat: str | None = None
    disclaimer: str = (
        "This is a screening aid, not a medical diagnosis. "
        "Please consult a qualified healthcare professional for any health concerns."
    )


def run_prediction(feature_values: dict, endpoint: str) -> PredictionResponse:
    """Shared prediction logic used by both the manual-input and audio endpoints."""
    try:
        X = pd.DataFrame([[feature_values[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        prediction_label = "Elevated risk indicators detected" if prediction == 1 else "No elevated risk indicators detected"
        probability_rounded = round(float(probability), 4)

        log_prediction(endpoint, feature_values, prediction_label, probability_rounded)

        return PredictionResponse(
            prediction=prediction_label,
            probability_parkinsons=probability_rounded,
            extracted_features=feature_values,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    return {"message": "Parkinson's Voice Screening API is running. See /docs for usage."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    """Basic monitoring: aggregate counts and average predicted probability
    over every prediction logged so far, across both endpoints."""
    return get_stats()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: VoiceFeatures):
    feature_values = {
        "MDVP:Fo(Hz)": features.mdvp_fo_hz,
        "MDVP:Jitter(%)": features.mdvp_jitter_pct,
        "MDVP:Shimmer": features.mdvp_shimmer,
        "HNR": features.hnr,
        "PPE": features.ppe,
    }
    result = run_prediction(feature_values, endpoint="predict")
    result.extracted_features = None  # not relevant for manual entry — user already knows these
    return result


@app.post("/predict-audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        feature_values = extract_features(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)

    result = run_prediction(feature_values, endpoint="predict-audio")
    result.recording_caveat = (
        "This prediction depends on clean audio input. A malfunctioning or very low-quality "
        "microphone can distort jitter/shimmer/HNR measurements enough to produce a misleading "
        "result — verify your recording equipment is working normally before relying on this "
        "result, and treat it as a demonstration of the pipeline rather than a guaranteed-accurate "
        "individual assessment."
    )
    return result