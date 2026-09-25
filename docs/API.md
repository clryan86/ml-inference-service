# API Guide

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/healthz` | Service/model health |
| GET | `/api/model` | Model card and training metrics |
| POST | `/api/predict` | Single inference |
| POST | `/api/predict/batch` | Batch inference |
| GET | `/api/predictions` | Recent prediction log |
| GET | `/api/metrics` | Inference metrics |
| GET | `/api/drift` | Feature drift report |

Interactive OpenAPI documentation is generated at `/docs`.
