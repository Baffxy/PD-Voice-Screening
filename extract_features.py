"""
Extracts the 5 acoustic features the model was trained on, directly from a
raw voice recording (.wav), using parselmouth (a Python wrapper around Praat).

Note on PPE: unlike the other 4 features, Pitch Period Entropy has no built-in
Praat function — it's a specific statistic from the original research paper
(Little et al., 2007). The implementation below is a reasonable APPROXIMATION
(entropy of the normalized log pitch-period distribution), not a byte-for-byte
reproduction of the original paper's exact computation. This is documented
here and should be documented in the project README too — approximated is an
honest engineering statement, pretending exactness would not be.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_features(audio_path: str) -> dict:
    """
    Extract MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, HNR, and PPE
    from a voice recording (ideally a few seconds of a sustained vowel, e.g. "ahhh").
    """
    sound = parselmouth.Sound(audio_path)

    # Pitch floor/ceiling of 75-500 Hz covers typical human speaking voice range.
    pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=500)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    # --- MDVP:Fo(Hz) — average fundamental frequency ---
    fo = call(pitch, "Get mean", 0, 0, "Hertz")

    # --- MDVP:Jitter(%) — cycle-to-cycle frequency variation ---
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # --- MDVP:Shimmer — cycle-to-cycle amplitude variation ---
    shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # --- HNR — harmonics-to-noise ratio ---
    harmonicity = sound.to_harmonicity()
    hnr = call(harmonicity, "Get mean", 0, 0)

    # --- PPE — approximated ---
    ppe = _approximate_ppe(pitch)

    if any(np.isnan(v) for v in [fo, jitter, shimmer, hnr, ppe]):
        raise ValueError(
            "Could not extract one or more features — the recording may be too short, "
            "too quiet, or not contain a clear sustained voiced sound."
        )

    return {
        "MDVP:Fo(Hz)": round(fo, 3),
        "MDVP:Jitter(%)": round(jitter, 5),
        "MDVP:Shimmer": round(shimmer, 5),
        "HNR": round(hnr, 3),
        "PPE": round(ppe, 4),
    }


def _approximate_ppe(pitch) -> float:
    """
    Approximate Pitch Period Entropy: entropy of the distribution of
    log-transformed, normalized pitch periods across voiced frames.
    Higher values indicate more irregular pitch control (associated with PD).
    """
    pitch_values = pitch.selected_array["frequency"]
    pitch_values = pitch_values[pitch_values > 0]  # drop unvoiced frames

    if len(pitch_values) < 10:
        return float("nan")

    periods = 1.0 / pitch_values
    log_periods = np.log(periods)

    # Normalize (zero mean) as in the original PPE formulation's intent —
    # entropy of the DEVIATION pattern, not the absolute pitch level.
    normalized = log_periods - np.mean(log_periods)

    # Histogram-based entropy estimate.
    hist, _ = np.histogram(normalized, bins=30, density=True)
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    entropy = -np.sum(probs * np.log(probs))

    # Rescale roughly into the range PPE values occupy in the training data (~0.05-0.7).
    return entropy / 10


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_features.py <path_to_wav_file>")
        sys.exit(1)

    features = extract_features(sys.argv[1])
    print("Extracted features:")
    for k, v in features.items():
        print(f"  {k}: {v}")