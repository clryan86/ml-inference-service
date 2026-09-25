from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ml.drift import drift_report
from ml.manager import ModelManager

from .config import Settings
from .schemas import BatchRequest, EquipmentFeatures, PredictionResponse
from .store import PredictionStore

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = ModelManager(settings.artifact_dir, settings.model_version)
    store = PredictionStore(settings.database_path)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Train, version, serve, and monitor a predictive-maintenance classification model.",
    )
    app.state.manager = manager
    app.state.store = store
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "service": settings.app_name, "model_version": manager.metadata["version"]}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        recent = store.recent(12)
        metrics = store.metrics()
        drift = drift_report([item["features"] for item in store.recent(100)], manager.metadata["baseline"])
        return TEMPLATES.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "app_name": settings.app_name,
                "model": manager.metadata,
                "recent": recent,
                "metrics": metrics,
                "drift": drift,
            },
        )

    @app.get("/api/model")
    def model_card():
        return manager.metadata

    @app.post("/api/predict", response_model=PredictionResponse)
    def predict(features: EquipmentFeatures):
        payload = features.model_dump()
        result = manager.predict(payload)
        prediction_id = store.record(payload, result)
        return {"prediction_id": prediction_id, **result}

    @app.post("/api/predict/batch", response_model=list[PredictionResponse])
    def predict_batch(request: BatchRequest):
        outputs = []
        for item in request.items:
            payload = item.model_dump()
            result = manager.predict(payload)
            prediction_id = store.record(payload, result)
            outputs.append({"prediction_id": prediction_id, **result})
        return outputs

    @app.get("/api/metrics")
    def inference_metrics():
        return {**store.metrics(), "model_version": manager.metadata["version"]}

    @app.get("/api/drift")
    def drift(limit: int = 200):
        limit = max(1, min(limit, 2000))
        rows = store.recent(limit)
        return drift_report([item["features"] for item in rows], manager.metadata["baseline"])

    @app.get("/api/predictions")
    def predictions(limit: int = 50):
        return store.recent(max(1, min(limit, 500)))

    return app


app = create_app()
