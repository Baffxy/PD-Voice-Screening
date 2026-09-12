"""
Parkinson's Disease voice screening — orchestrated training pipeline (Prefect).

Same logic as train_model.py, but broken into named, tracked tasks:
- Each task's inputs/outputs, duration, and success/failure are logged automatically
- A failed task (e.g. a dropped connection during dataset download) can retry
  automatically instead of killing the whole run
- Running this produces a record in Prefect's dashboard you can inspect later

Run with: python training_flow.py
(For task/flow run history: `prefect server start` in a separate terminal, then
visit http://127.0.0.1:4200 while this runs.)
"""

import json

import joblib
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

ALL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]
CANDIDATE_FEATURES = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]


@task(retries=3, retry_delay_seconds=10, log_prints=True)
def load_data() -> pd.DataFrame:
    """Download the dataset. Retries automatically if the connection drops mid-download —
    this is exactly the kind of failure that happened repeatedly during this project's
    local development, so it's a real, relevant use of Prefect's retry mechanism."""
    print(f"Loading dataset from {DATA_URL} ...")
    data = pd.read_csv(DATA_URL)
    print(f"Loaded {len(data)} rows. Class balance:\n{data['status'].value_counts()}")
    return data


@task(log_prints=True)
def evaluate_feature_set(data: pd.DataFrame, feature_names: list, label: str) -> float:
    """Run stratified 5-fold CV and report averaged metrics for a given feature set.
    Returns the mean macro-F1 score, used to decide which feature set to ship."""
    X = data[feature_names]
    y = data["status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=["accuracy", "f1_macro", "roc_auc", "recall_macro"],
    )

    print(f"\n--- {label} ({len(feature_names)} features) ---")
    print(f"Accuracy:      {scores['test_accuracy'].mean():.3f}")
    print(f"F1 (macro):    {scores['test_f1_macro'].mean():.3f}")
    print(f"ROC-AUC:       {scores['test_roc_auc'].mean():.3f}")
    print(f"Recall(macro): {scores['test_recall_macro'].mean():.3f}")

    return scores["test_f1_macro"].mean()


@task(log_prints=True)
def choose_feature_set(f1_candidate: float, f1_all: float) -> list:
    """A trackable decision step — Prefect logs which choice was made and why,
    for every pipeline run, not just the one you happen to be watching."""
    if f1_all > f1_candidate:
        print(f"All-features set performed better (F1 {f1_all:.3f} vs {f1_candidate:.3f}). Using it.")
        return ALL_FEATURES
    print(f"Candidate 5-feature set performed as well or better (F1 {f1_candidate:.3f} vs {f1_all:.3f}). Using it.")
    return CANDIDATE_FEATURES


@task(log_prints=True)
def train_final_model(data: pd.DataFrame, feature_names: list):
    """Train on a held-out split, evaluate, and return the fitted model + scaler."""
    X = data[feature_names]
    y = data["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Final held-out test set evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    return model, scaler


@task(log_prints=True)
def save_artifacts(model, scaler, feature_names: list):
    joblib.dump(model, "parkinsons_voice_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    metadata = {
        "features_used": feature_names,
        "model": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
        "dataset": DATA_URL,
        "notes": "class_weight='balanced' used to address ~3:1 class imbalance (147 PD vs 48 healthy).",
        "trained_via": "Prefect orchestrated flow (training_flow.py)",
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved: parkinsons_voice_model.pkl, scaler.pkl, model_metadata.json")


@flow(name="pd-voice-screening-training", log_prints=True)
def training_flow():
    """The full pipeline, as a single flow made of tracked tasks."""
    data = load_data()

    f1_candidate = evaluate_feature_set(data, CANDIDATE_FEATURES, "Candidate 5 features")
    f1_all = evaluate_feature_set(data, ALL_FEATURES, "All 22 features")

    chosen_features = choose_feature_set(f1_candidate, f1_all)

    model, scaler = train_final_model(data, chosen_features)
    save_artifacts(model, scaler, chosen_features)


if __name__ == "__main__":
    training_flow()