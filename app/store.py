from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    probability REAL NOT NULL,
    risk_band TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class PredictionStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record(self, features: dict[str, float], result: dict[str, Any]) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT INTO predictions(model_version,features_json,prediction,probability,risk_band,created_at) VALUES(?,?,?,?,?,?)",
                (
                    result["model_version"],
                    json.dumps(features),
                    result["prediction"],
                    result["failure_probability"],
                    result["risk_band"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["features"] = json.loads(item.pop("features_json"))
            items.append(item)
        return items

    def metrics(self) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) total,
                          COALESCE(AVG(probability),0) avg_probability,
                          SUM(CASE WHEN risk_band='high' THEN 1 ELSE 0 END) high_risk
                   FROM predictions"""
            ).fetchone()
        return {
            "prediction_count": int(row["total"] or 0),
            "average_failure_probability": round(float(row["avg_probability"] or 0), 4),
            "high_risk_predictions": int(row["high_risk"] or 0),
        }
