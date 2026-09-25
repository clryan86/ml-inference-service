from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from ml.training import train_model

settings = Settings.from_env()
metadata = train_model(settings.artifact_dir, version=settings.model_version)
print(f"trained {metadata['algorithm']} version {metadata['version']} roc_auc={metadata['metrics']['roc_auc']}")
