from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import FEATURES
from .training import train_model


class ModelManager:
    def __init__(self, artifact_dir: Path, version: str = "1.0.0"):
        self.artifact_dir = Path(artifact_dir)
        self.version = version
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.load_or_train()

    def load_or_train(self) -> None:
        model_path = self.artifact_dir / "model.joblib"
        metadata_path = self.artifact_dir / "metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            train_model(self.artifact_dir, version=self.version)
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        frame = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)
        probability = float(self.model.predict_proba(frame)[0, 1])
        threshold = float(self.metadata.get("threshold", 0.5))
        prediction = int(probability >= threshold)
        risk = "high" if probability >= 0.7 else "medium" if probability >= 0.35 else "low"
        return {
            "prediction": prediction,
            "failure_probability": round(probability, 6),
            "risk_band": risk,
            "model_version": self.metadata["version"],
            "threshold": threshold,
        }
