"""
Simple Streamlit frontend for the Parkinson's Voice Screening API.
Talks to the FastAPI backend running separately (locally or in Docker) on port 8000.

Run with: streamlit run frontend.py
"""

import os

import requests
import streamlit as st

# Uses a deployed API URL if provided (via Streamlit secrets or an env var),
# falling back to localhost for local development.
try:
    API_BASE = st.secrets["API_BASE_URL"]
except (KeyError, FileNotFoundError):
    API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API_URL = f"{API_BASE}/predict"

st.set_page_config(page_title="Parkinson's Voice Screening", page_icon="🎙️", layout="centered")

st.title("🎙️ Parkinson's Voice Screening")
st.caption(
    "A screening aid based on acoustic voice features. "
    "**This is not a diagnostic tool** — it flags patterns worth discussing with a doctor, nothing more."
)

st.markdown("---")

st.subheader("Enter voice acoustic measurements")
st.caption(
    "These values normally come from analyzing a recorded voice sample. "
    "For this demo, enter them manually — try the example values below, or your own."
)

col1, col2 = st.columns(2)

with col1:
    fo = st.number_input(
        "Fundamental frequency — MDVP:Fo(Hz)",
        min_value=0.0, value=154.2, step=1.0,
        help="Average vocal fundamental frequency",
    )
    jitter = st.number_input(
        "Jitter — MDVP:Jitter(%)",
        min_value=0.0, value=0.005, step=0.001, format="%.4f",
        help="Frequency variation between cycles",
    )
    shimmer = st.number_input(
        "Shimmer — MDVP:Shimmer",
        min_value=0.0, value=0.03, step=0.001, format="%.4f",
        help="Amplitude variation between cycles",
    )

with col2:
    hnr = st.number_input(
        "Harmonics-to-noise ratio — HNR",
        min_value=0.0, value=21.5, step=0.1,
        help="Ratio of harmonic sound to noise in the voice",
    )
    ppe = st.number_input(
        "Pitch period entropy — PPE",
        min_value=0.0, value=0.2, step=0.01, format="%.3f",
        help="A measure of pitch variation irregularity",
    )

st.markdown("---")

if st.button("Run Screening", type="primary", use_container_width=True):
    payload = {
        "mdvp_fo_hz": fo,
        "mdvp_jitter_pct": jitter,
        "mdvp_shimmer": shimmer,
        "hnr": hnr,
        "ppe": ppe,
    }

    try:
        with st.spinner("Analyzing..."):
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()

        probability = result["probability_parkinsons"]
        prediction = result["prediction"]

        if "Elevated" in prediction:
            st.warning(f"**{prediction}**")
        else:
            st.success(f"**{prediction}**")

        st.metric("Model confidence (probability of elevated risk)", f"{probability:.1%}")
        st.progress(probability)

        st.info(result["disclaimer"])

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the prediction API. Make sure the FastAPI backend is running "
            "(locally with `uvicorn main:app --reload`, or via `docker run -p 8000:8000 pd-voice-api`)."
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")

with st.expander("How does this work?"):
    st.markdown(
        """
        This tool uses a Random Forest model trained on the UCI Parkinson's voice dataset,
        with class imbalance explicitly handled (`class_weight='balanced'`) and validated
        using stratified 5-fold cross-validation.

        **The 5 acoustic features used** were chosen based on cross-validated performance
        comparison against using all 22 available features — not arbitrarily.

        In a full deployment, these feature values would be automatically extracted from
        a short recorded voice sample rather than entered manually.
        """
    )