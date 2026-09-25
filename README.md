# ML Inference Service

[![CI](https://github.com/clryan86/ml-inference-service/actions/workflows/ci.yml/badge.svg)](https://github.com/clryan86/ml-inference-service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-model%20training-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-inference-009688)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![License](https://img.shields.io/badge/license-MIT-green)

A complete machine-learning engineering portfolio project covering **synthetic training data, model training, evaluation, versioned artifacts, online/batch inference, inference logging, feature drift monitoring, model metadata, tests, CI, and containerized deployment**.

The service predicts whether an industrial asset is at elevated risk of failure within 30 days from six operating/sensor features. The business scenario is synthetic; the engineering pipeline is real and reproducible.

## ML lifecycle demonstrated

1. Generate deterministic training data.
2. Split train/test sets with stratification.
3. Train a Random Forest classifier.
4. Evaluate accuracy, precision, recall, F1, and ROC AUC.
5. Store a versioned model artifact plus model metadata.
6. Load the artifact behind a typed FastAPI service.
7. Support single and batch prediction.
8. Persist inference traffic.
9. Compare recent feature distributions with the training baseline.
10. Surface model performance, feature importance, traffic, and drift on an operations dashboard.

## Input features

- vibration
- temperature
- pressure
- RPM
- operating hours
- days since maintenance

## Quick start

```bash
python -m venv .venv
```

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements-dev.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

The service will also train a baseline artifact automatically if none exists.

Open the dashboard at `http://127.0.0.1:8000`, OpenAPI at `/docs`, the model card at `/api/model`, and drift monitoring at `/api/drift`.

## Example prediction

```json
{
  "vibration_mm_s": 7.5,
  "temperature_c": 98,
  "pressure_bar": 7.8,
  "rpm": 2450,
  "operating_hours": 16000,
  "days_since_maintenance": 220
}
```

## Monitoring

Each inference is stored with its feature values, probability, risk band, model version, and timestamp. `/api/drift` compares recent feature means with the training baseline in standard-deviation units and labels each feature as stable, watch, or alert.

This is intentionally a lightweight drift indicator rather than pretending to be a full production monitoring stack. The design leaves a clear upgrade path to PSI/KS tests, Evidently, OpenTelemetry, or a dedicated model-monitoring platform.

## Tests

```bash
pytest
ruff check .
```

## Docker

```bash
docker compose up --build
```

## Roadmap

- model registry with multiple active versions
- shadow/canary inference
- calibration curves
- feature schema versioning
- request latency histograms
- PSI / KS drift tests
- actual vs predicted outcome feedback
- retraining triggers
- MLflow integration
- Prometheus/OpenTelemetry monitoring

## Author

**Christopher Ryan**  
Python • Machine Learning • Backend • Automation

## License

MIT
