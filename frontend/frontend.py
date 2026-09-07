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

st.set_page_config(page_title="Parkinson's Voice Screening", page_icon="🎙️", layout="centered")

st.title("🎙️ Parkinson's Voice Screening")
st.caption(
    "A screening aid based on acoustic voice features. "
    "**This is not a diagnostic tool** — it flags patterns worth discussing with a doctor, nothing more."
)

st.markdown("---")


def show_result(result):
    probability = result["probability_parkinsons"]
    prediction = result["prediction"]

    if "Elevated" in prediction:
        st.warning(f"**{prediction}**")
    else:
        st.success(f"**{prediction}**")

    st.metric("Model confidence (probability of elevated risk)", f"{probability:.1%}")
    st.progress(probability)

    if result.get("extracted_features"):
        with st.expander("Extracted acoustic features"):
            st.json(result["extracted_features"])

    st.info(result["disclaimer"])
    if result.get("recording_caveat"):
        st.caption(result["recording_caveat"])


tab1, tab2 = st.tabs(["🎤 Record or Upload Voice", "🔢 Manual Entry (Advanced)"])

with tab1:
    st.subheader("Record or upload a voice sample")
    st.caption(
        "Say a sustained **\"ahhh\"** for 3–5 seconds at a steady pitch — "
        "not normal talking. Works best in a quiet room."
    )

    audio_value = st.audio_input("Record your voice")
    uploaded_wav = st.file_uploader("...or upload a .wav file instead", type=["wav"])

    audio_bytes = None
    filename = "recording.wav"
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
    elif uploaded_wav is not None:
        audio_bytes = uploaded_wav.getvalue()
        filename = uploaded_wav.name

    if audio_bytes and st.button("Run Screening", type="primary", use_container_width=True, key="audio_run"):
        try:
            with st.spinner("Analyzing voice sample..."):
                response = requests.post(
                    f"{API_BASE}/predict-audio",
                    files={"file": (filename, audio_bytes, "audio/wav")},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
            show_result(result)
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the prediction API. It may be waking up from idle "
                "(free-tier hosting sleeps after inactivity) — try again in ~30 seconds."
            )
        except requests.exceptions.HTTPError:
            detail = response.json().get("detail", "Unknown error")
            st.error(f"Couldn't process that recording: {detail}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

with tab2:
    st.subheader("Enter voice acoustic measurements manually")
    st.caption(
        "For advanced users who already have these values (e.g. from prior acoustic analysis)."
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

    if st.button("Run Screening", type="primary", use_container_width=True, key="manual_run"):
        payload = {
            "mdvp_fo_hz": fo,
            "mdvp_jitter_pct": jitter,
            "mdvp_shimmer": shimmer,
            "hnr": hnr,
            "ppe": ppe,
        }
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
            show_result(result)
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the prediction API. It may be waking up from idle "
                "(free-tier hosting sleeps after inactivity) — try again in ~30 seconds."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.markdown("---")

with st.expander("How does this work?"):
    st.markdown(
        """
        This tool uses a Random Forest model trained on the UCI Parkinson's voice dataset,
        with class imbalance explicitly handled (`class_weight='balanced'`) and validated
        using stratified 5-fold cross-validation.

        **The 5 acoustic features used** were chosen based on cross-validated performance
        comparison against using all 22 available features — not arbitrarily.

        When you record or upload audio, these features are extracted automatically using
        `parselmouth` (a Python wrapper around Praat, the standard tool for acoustic voice
        analysis). One feature, Pitch Period Entropy (PPE), has no standard Praat equivalent
        and is approximated rather than computed exactly as in the original research paper.
        """
    )