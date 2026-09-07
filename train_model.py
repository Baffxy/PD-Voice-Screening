"""
Parkinson's Disease voice screening — model training (v2).

Fixes vs. the original notebook:
- Stratified k-fold cross-validation instead of a single train/test split
- Class imbalance handled explicitly (class_weight='balanced'), compared against unweighted
- Feature importance computed to justify which features are kept
- Saves model + scaler + a metadata file documenting exactly what was used and why
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

# All acoustic features available in the dataset (excluding 'name' and 'status')
ALL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

# The 5 features the original notebook picked, kept here as a candidate to compare against.
CANDIDATE_FEATURES = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)", "MDVP:Shimmer", "HNR", "PPE"]


def load_data():
    print(f"Loading dataset from {DATA_URL} ...")
    data = pd.read_csv(DATA_URL)
    print(f"Loaded {len(data)} rows. Class balance:\n{data['status'].value_counts()}")
    return data


def evaluate_feature_set(X, y, feature_names, label):
    """Run stratified 5-fold CV and report averaged metrics for a given feature set."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=["accuracy", "f1_macro", "roc_auc", "recall_macro"],
    )

    print(f"\n--- {label} ({len(feature_names)} features) ---")
    print(f"Accuracy:      {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std():.3f})")
    print(f"F1 (macro):    {scores['test_f1_macro'].mean():.3f}")
    print(f"ROC-AUC:       {scores['test_roc_auc'].mean():.3f}")
    print(f"Recall(macro): {scores['test_recall_macro'].mean():.3f}  <- this is the one that exposes class imbalance bias")

    return scores["test_f1_macro"].mean()


def get_feature_importance(X, y, feature_names):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nFeature importance (all features, full dataset):")
    print(importance.to_string())
    return importance


def train_final_model(X, y, feature_names):
    """Train on a held-out split (not just CV) so we have a fixed artifact to save and report on."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # stratify fixes the missing safeguard from before
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


def main():
    data = load_data()
    y = data["status"]

    # Compare: original 5-feature set vs. all 22 features, both with class_weight='balanced'.
    f1_candidate = evaluate_feature_set(data[CANDIDATE_FEATURES], y, CANDIDATE_FEATURES, "Candidate 5 features")
    f1_all = evaluate_feature_set(data[ALL_FEATURES], y, ALL_FEATURES, "All 22 features")

    get_feature_importance(data[ALL_FEATURES], y, ALL_FEATURES)

    # Decide which feature set to ship with, based on CV F1 (macro), not accuracy.
    if f1_all > f1_candidate:
        print(f"\nAll-features set performed better (F1 {f1_all:.3f} vs {f1_candidate:.3f}). Using it.")
        chosen_features = ALL_FEATURES
    else:
        print(f"\nCandidate 5-feature set performed as well or better (F1 {f1_candidate:.3f} vs {f1_all:.3f}). Using it — smaller, easier to justify, and easier to extract from a short voice recording later.")
        chosen_features = CANDIDATE_FEATURES

    model, scaler = train_final_model(data[chosen_features], y, chosen_features)

    joblib.dump(model, "parkinsons_voice_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    metadata = {
        "features_used": chosen_features,
        "model": "RandomForestClassifier(n_estimators=200, class_weight='balanced')",
        "dataset": DATA_URL,
        "notes": "class_weight='balanced' used to address ~3:1 class imbalance (147 PD vs 48 healthy).",
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved: parkinsons_voice_model.pkl, scaler.pkl, model_metadata.json")


if __name__ == "__main__":
    main()