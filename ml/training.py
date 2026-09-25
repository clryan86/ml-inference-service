from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .features import FEATURES


def generate_training_data(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vibration = np.clip(rng.normal(4.0, 1.4, rows), 0.2, 12)
    temperature = np.clip(rng.normal(72, 11, rows), 25, 140)
    pressure = np.clip(rng.normal(6.2, 1.0, rows), 1, 12)
    rpm = np.clip(rng.normal(1800, 420, rows), 400, 3500)
    operating_hours = rng.uniform(50, 22000, rows)
    days_since_maintenance = rng.uniform(0, 365, rows)

    logit = (
        -8.2
        + 0.55 * vibration
        + 0.035 * (temperature - 60)
        + 0.000055 * operating_hours
        + 0.0065 * days_since_maintenance
        + 0.00022 * np.maximum(rpm - 2000, 0)
        + 0.16 * np.abs(pressure - 6.2)
    )
    probability = 1 / (1 + np.exp(-logit))
    failure = rng.binomial(1, np.clip(probability, 0.01, 0.95))

    return pd.DataFrame({
        "vibration_mm_s": vibration.round(3),
        "temperature_c": temperature.round(3),
        "pressure_bar": pressure.round(3),
        "rpm": rpm.round(1),
        "operating_hours": operating_hours.round(1),
        "days_since_maintenance": days_since_maintenance.round(1),
        "failure_within_30d": failure,
    })


def train_model(artifact_dir: Path, version: str = "1.0.0", rows: int = 5000, seed: int = 42) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    df = generate_training_data(rows=rows, seed=seed)
    X = df[FEATURES]
    y = df["failure_within_30d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=9,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "test_rows": int(len(X_test)),
        "training_rows": int(len(X_train)),
    }
    baseline = {
        feature: {
            "mean": round(float(X_train[feature].mean()), 6),
            "std": round(float(X_train[feature].std(ddof=0)), 6),
            "min": round(float(X_train[feature].min()), 6),
            "max": round(float(X_train[feature].max()), 6),
        }
        for feature in FEATURES
    }
    importance = sorted(
        ({"feature": feature, "importance": round(float(value), 6)} for feature, value in zip(FEATURES, model.feature_importances_)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "target": "failure_within_30d",
        "features": FEATURES,
        "metrics": metrics,
        "baseline": baseline,
        "feature_importance": importance,
        "threshold": 0.5,
        "seed": seed,
    }
    joblib.dump(model, artifact_dir / "model.joblib")
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
