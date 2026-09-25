from __future__ import annotations

from typing import Any

import pandas as pd

from .features import FEATURES


def drift_report(rows: list[dict[str, float]], baseline: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"sample_size": 0, "overall_status": "insufficient_data", "features": []}
    df = pd.DataFrame(rows)
    findings = []
    max_shift = 0.0
    for feature in FEATURES:
        if feature not in df.columns:
            continue
        reference = baseline[feature]
        std = float(reference["std"])
        mean = float(df[feature].mean())
        shift = abs(mean - float(reference["mean"])) / std if std > 0 else 0.0
        max_shift = max(max_shift, shift)
        status = "alert" if shift >= 1.5 else "watch" if shift >= 0.75 else "stable"
        findings.append({
            "feature": feature,
            "current_mean": round(mean, 5),
            "baseline_mean": reference["mean"],
            "standardized_mean_shift": round(shift, 4),
            "status": status,
        })
    overall = "alert" if max_shift >= 1.5 else "watch" if max_shift >= 0.75 else "stable"
    return {"sample_size": len(rows), "overall_status": overall, "features": findings}
