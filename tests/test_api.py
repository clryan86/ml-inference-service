from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def client(tmp_path: Path) -> TestClient:
    settings = Settings(artifact_dir=tmp_path / "artifacts", database_path=tmp_path / "inference.db", model_version="test")
    return TestClient(create_app(settings))


def payload():
    return {
        "vibration_mm_s": 7.5,
        "temperature_c": 98,
        "pressure_bar": 7.8,
        "rpm": 2450,
        "operating_hours": 16000,
        "days_since_maintenance": 220,
    }


def test_health_model_predict_metrics_and_drift(tmp_path):
    c = client(tmp_path)
    assert c.get("/healthz").status_code == 200
    model = c.get("/api/model")
    assert model.status_code == 200
    assert "roc_auc" in model.json()["metrics"]

    prediction = c.post("/api/predict", json=payload())
    assert prediction.status_code == 200
    assert prediction.json()["prediction_id"] == 1
    assert 0 <= prediction.json()["failure_probability"] <= 1

    batch = c.post("/api/predict/batch", json={"items": [payload(), payload()]})
    assert batch.status_code == 200
    assert len(batch.json()) == 2

    metrics = c.get("/api/metrics").json()
    assert metrics["prediction_count"] == 3

    drift = c.get("/api/drift").json()
    assert drift["sample_size"] == 3
    assert drift["overall_status"] in {"stable", "watch", "alert"}


def test_schema_rejects_impossible_sensor_value(tmp_path):
    c = client(tmp_path)
    invalid = payload()
    invalid["rpm"] = 99999
    assert c.post("/api/predict", json=invalid).status_code == 422
