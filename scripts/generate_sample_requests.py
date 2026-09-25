import json

samples = [
    {"vibration_mm_s": 2.8, "temperature_c": 62, "pressure_bar": 6.1, "rpm": 1550, "operating_hours": 1800, "days_since_maintenance": 15},
    {"vibration_mm_s": 6.3, "temperature_c": 88, "pressure_bar": 7.2, "rpm": 2150, "operating_hours": 11000, "days_since_maintenance": 135},
    {"vibration_mm_s": 9.2, "temperature_c": 118, "pressure_bar": 9.1, "rpm": 2900, "operating_hours": 20500, "days_since_maintenance": 320},
]
print(json.dumps({"items": samples}, indent=2))
