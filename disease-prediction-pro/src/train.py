"""
Train and compare multiple classifiers for disease prediction from symptoms.

Dataset: 4,920 records x 132 binary symptom features -> 41 disease classes
(source: public "Disease Prediction Using Machine Learning" dataset, derived
from the Columbia University DiseaseSymptomKB).

Pipeline:
1. Load and clean data (strip whitespace from column names / labels — the
   raw CSV has trailing spaces on some symptom and disease names).
2. Train/test split, stratified by disease.
3. Compare several standard classifiers with stratified k-fold cross-validation.
4. Refit the best-performing model on the full training set.
5. Evaluate on the held-out test set (accuracy, precision/recall/F1, confusion matrix).
6. Save the model, label encoder, feature column order, and a metrics report.

Run:
    python src/train.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Training.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Clean up whitespace artifacts present in the raw dataset
    df.columns = [c.strip() for c in df.columns]
    df["prognosis"] = df["prognosis"].str.strip()
    # Drop the stray unnamed index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def build_candidate_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    feature_cols = [c for c in df.columns if c != "prognosis"]
    X = df[feature_cols]
    y_raw = df["prognosis"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset: {len(df)} records, {len(feature_cols)} symptoms, "
          f"{len(label_encoder.classes_)} diseases")
    print(f"Train: {len(X_train)}  Test: {len(X_test)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = build_candidate_models()

    print(f"{'Model':<20}{'CV Accuracy (mean ± std)':<30}{'Time (s)'}")
    print("-" * 65)

    cv_results = {}
    for name, model in candidates.items():
        start = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - start
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        print(f"{name:<20}{scores.mean():.4f} ± {scores.std():.4f}{'':<8}{elapsed:.1f}")

    best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
    print(f"\nBest model by cross-validation: {best_name}")

    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"Held-out test accuracy: {test_acc:.4f}")

    # Refit on ALL data (train+test) for the deployed model, now that we've
    # honestly measured generalization on the held-out split above.
    best_model.fit(X, y)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "held_out_test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_records": len(df),
        "n_features": len(feature_cols),
        "n_classes": len(label_encoder.classes_),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, encoder, feature list, and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
