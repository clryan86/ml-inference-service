from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifact_dir: Path
    database_path: Path
    model_version: str = "1.0.0"
    app_name: str = "ML Inference Service"

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(__file__).resolve().parents[1]
        return cls(
            artifact_dir=Path(os.getenv("ARTIFACT_DIR", root / "artifacts")),
            database_path=Path(os.getenv("DATABASE_PATH", root / "data" / "inference.db")),
            model_version=os.getenv("MODEL_VERSION", "1.0.0"),
        )
