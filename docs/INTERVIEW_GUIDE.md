# Interview Guide

## 30-second explanation

> ML Inference Service demonstrates the whole path from training to production-style serving. It creates deterministic predictive-maintenance data, trains and evaluates a Random Forest, writes a versioned artifact and model metadata, serves typed single and batch predictions through FastAPI, logs inference traffic, and compares recent feature distributions with the training baseline for drift.

## Why generate synthetic data?

The point of the project is ML engineering rather than claiming a proprietary real-world dataset. Synthetic data makes the pipeline reproducible, legally clean, and immediately runnable by reviewers.

## Why not commit the model binary?

Generated artifacts are build outputs. Keeping training code as the source of truth proves reproducibility and avoids filling Git history with opaque binaries.

## Why Random Forest?

It handles nonlinear relationships, needs little preprocessing, provides feature importance, has fast CPU inference, and is easy to reproduce. The serving architecture is model-agnostic enough to replace it later.

## How is drift measured?

Recent inference means are compared with training means and normalized by training standard deviations. It is a lightweight signal, not a claim of complete statistical monitoring.

## Production improvements

- MLflow/model registry
- explicit artifact checksum and signed releases
- calibrated probabilities
- Prometheus latency/error metrics
- actual-outcome feedback
- automated retraining criteria
- shadow/canary deployments
- feature store
- robust PSI/KS drift tests

## Resume bullets

- Built a reproducible scikit-learn training and inference pipeline with versioned artifacts, evaluation metrics, model metadata, feature importance, and baseline statistics.
- Served validated single/batch predictions through FastAPI and persisted inference traffic for operations metrics and feature-drift monitoring.
- Added automated model/API tests, Docker deployment, GitHub Actions CI, and production-oriented architecture/model documentation.
