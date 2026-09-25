from pathlib import Path

from ml.manager import ModelManager
from ml.training import train_model


def test_training_pipeline_produces_model_and_metadata(tmp_path: Path):
    metadata = train_model(tmp_path, version="test-1", rows=1200, seed=7)
    assert (tmp_path / "model.joblib").exists()
    assert metadata["version"] == "test-1"
    assert 0.5 <= metadata["metrics"]["roc_auc"] <= 1.0
    assert len(metadata["feature_importance"]) == 6


def test_model_manager_predicts_probability(tmp_path: Path):
    train_model(tmp_path, version="test-2", rows=1000, seed=11)
    manager = ModelManager(tmp_path, version="test-2")
    result = manager.predict({
        "vibration_mm_s": 8.2,
        "temperature_c": 105,
        "pressure_bar": 8.1,
        "rpm": 2600,
        "operating_hours": 18000,
        "days_since_maintenance": 250,
    })
    assert 0 <= result["failure_probability"] <= 1
    assert result["risk_band"] in {"low", "medium", "high"}
