from __future__ import annotations

from pydantic import BaseModel, Field


class EquipmentFeatures(BaseModel):
    vibration_mm_s: float = Field(ge=0, le=20)
    temperature_c: float = Field(ge=-20, le=200)
    pressure_bar: float = Field(ge=0, le=30)
    rpm: float = Field(ge=0, le=6000)
    operating_hours: float = Field(ge=0, le=100000)
    days_since_maintenance: float = Field(ge=0, le=2000)


class PredictionResponse(BaseModel):
    prediction_id: int
    prediction: int
    failure_probability: float
    risk_band: str
    model_version: str
    threshold: float


class BatchRequest(BaseModel):
    items: list[EquipmentFeatures] = Field(min_length=1, max_length=500)
