#!/usr/bin/env python3
"""JSON stdin/stdout bridge between Express and the existing Python ML services."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dashboard() -> dict[str, Any]:
    features_path = PROJECT_ROOT / "data/processed/household_energy_features.csv"
    artifacts = PROJECT_ROOT / "ml/artifacts"
    result: dict[str, Any] = {
        "project": "HomeSolar",
        "data": {"available": features_path.is_file(), "rows": None, "household_count": 0, "households": [], "latest_timestamp_utc": None},
        "models": {},
        "update_history": _read_json(artifacts / "model_update_history.json") or [],
    }
    if features_path.is_file():
        meters: set[str] = set()
        total = 0
        max_ts = None
        for chunk in pd.read_csv(features_path, usecols=["meter", "timestamp_utc"], chunksize=250_000, dtype={"meter": "string"}, low_memory=False):
            total += len(chunk)
            meters.update(chunk["meter"].dropna().astype(str).unique().tolist())
            ts = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed").dropna()
            if not ts.empty:
                part = ts.max()
                max_ts = part if max_ts is None else max(max_ts, part)
        result["data"] = {
            "available": True,
            "rows": total,
            "household_count": len(meters),
            "households": sorted(meters),
            "latest_timestamp_utc": max_ts.isoformat() if max_ts is not None else None,
            "warning": "Historical feature data is not a live reading feed. Upload newer meter and weather CSVs to update it.",
        }

    for target, artifact_name in (("next_24h", "household_model.pkl"), ("next_30d", "household_model_30d.pkl")):
        model_path = artifacts / artifact_name
        train_report = _read_json(artifacts / f"training_report_{target}.json") or {}
        eval_report = _read_json(artifacts / f"evaluation_{target}.json") or {}
        result["models"][target] = {
            "available": model_path.is_file(),
            "artifact": str(model_path.relative_to(PROJECT_ROOT)) if model_path.is_file() else None,
            "selected_model": train_report.get("selected_model"),
            "validation_metrics": (train_report.get("validation_metrics_by_candidate") or {}).get(train_report.get("selected_model")),
            "test_metrics": eval_report if eval_report else None,
        }
    return result


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    from backend.app.services.prediction_service import predict_household_from_latest_data

    meter = str(payload.get("meter", "")).strip()
    target = str(payload.get("target", "next_24h"))
    if target not in {"next_24h", "next_30d"}:
        raise ValueError("target must be next_24h or next_30d")
    return predict_household_from_latest_data(meter=meter, target=target)


def _solar_bill(payload: dict[str, Any]) -> dict[str, Any]:
    from backend.app.services.prediction_service import predict_household_from_latest_data
    from backend.app.services.solar_service import estimate_solar_from_peak_sun_hours
    from backend.app.services.bill_service import estimate_bill_after_solar

    meter = str(payload.get("meter", "")).strip()
    if not meter:
        raise ValueError("meter is required")
    panel_capacity = float(payload.get("panel_capacity_kw", 0))
    connected_load = float(payload.get("connected_load_kw", 0))
    peak_sun_hours_per_day = float(payload.get("peak_sun_hours", 0))
    billing_days = float(payload.get("billing_days", 30))
    # The input field is daily peak-sun-hours; the 30-day consumption forecast
    # requires period-total PSH for the same billing period.
    peak_sun_hours = peak_sun_hours_per_day * billing_days
    category_id = str(payload.get("category_id", "LMV1_URBAN_DOMESTIC"))
    pr = payload.get("performance_ratio")
    tariff_path = PROJECT_ROOT / "data/raw/tariff/uppcl_tariff.json"

    forecast = predict_household_from_latest_data(meter, target="next_30d")
    solar = estimate_solar_from_peak_sun_hours(
        consumption_kwh=forecast["predicted_future_energy_kwh"],
        panel_capacity_kw=panel_capacity,
        peak_sun_hours=peak_sun_hours,
        performance_ratio=None if pr in (None, "") else float(pr),
    )
    bill = estimate_bill_after_solar(
        solar_estimate=solar,
        category_id=category_id,
        connected_load_kw=connected_load,
        billing_days=billing_days,
        tariff_path=tariff_path,
    )
    return {
        "meter": meter,
        "electricity_forecast": forecast,
        "solar": solar,
        "bill": bill,
        "daily_peak_sun_hours_input": peak_sun_hours_per_day,
        "period_peak_sun_hours_used": peak_sun_hours,
        "weather_assumption": "Daily peak sun hours were supplied in the request and multiplied by billing_days. Use a forecast-derived location/season value; this demo does not fetch future weather automatically.",
    }


def _run_update(payload: dict[str, Any]) -> dict[str, Any]:
    from ml.online_update import run_update

    return run_update(
        new_electricity=payload["new_electricity"],
        new_weather=payload["new_weather"],
        base_electricity=payload.get("base_electricity"),
        base_weather=payload.get("base_weather"),
        location_map=payload.get("location_map"),
        min_improvement_pct=float(payload.get("min_improvement_pct", 0.5)),
    )


def main() -> int:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("Worker input must be a JSON object")
        if command == "dashboard":
            result = _dashboard()
        elif command == "predict":
            result = _predict(payload)
        elif command == "solar-bill":
            result = _solar_bill(payload)
        elif command == "update":
            result = _run_update(payload)
        elif command == "history":
            result = _read_json(PROJECT_ROOT / "ml/artifacts/model_update_history.json") or []
        else:
            raise ValueError(f"Unknown Python worker command: {command!r}")
        print(json.dumps({"ok": True, "result": result}, default=str, allow_nan=False, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, default=str, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
