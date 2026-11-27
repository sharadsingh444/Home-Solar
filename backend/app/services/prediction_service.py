"""Prediction and HomeSolar service orchestration.

This module connects the trained ML artifacts with the existing solar and bill
services. It does not train models. Train them first with:
    python ml/train.py --target next_24h
    python ml/train.py --target next_30d

For a forecast, supply either:
  * a current feature dictionary from the application's database/API, or
  * use predict_household_from_latest_data() for the latest row in the locally
    generated data/processed/household_energy_features.csv.

The CSV fallback is useful for a local demo. The user's current CEEW dataset
ends in 2021, so it must not be presented as a real-time 2026 household state.
For a live application, update the feature row from newly ingested meter data.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

# When a weighted-ensemble model was trained by running `python ml/train.py`,
# joblib may refer to its class as the top-level module `ensemble`. Register the
# package module before loading the saved model so FastAPI can unpickle it too.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
try:  # works with the Home-Solar project opened from its root directory
    from ml import ensemble as _homesolar_ensemble  # type: ignore

    sys.modules.setdefault("ensemble", _homesolar_ensemble)
except ImportError:
    # A non-ensemble sklearn/LightGBM pipeline can still be loaded without it.
    _homesolar_ensemble = None

try:
    from .solar_service import estimate_solar_from_weather
    from .bill_service import estimate_bill_after_solar
except ImportError:  # supports importing as app.services.* when cwd is backend/
    from app.services.solar_service import estimate_solar_from_weather
    from app.services.bill_service import estimate_bill_after_solar

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
DEFAULT_FEATURE_DATASET = PROJECT_ROOT / "data" / "processed" / "household_energy_features.csv"
MODEL_FILES = {
    "next_24h": "household_model.pkl",
    "next_30d": "household_model_30d.pkl",
}
TARGET_COLUMNS = {
    "next_24h": "target_next_24h_kWh",
    "next_30d": "target_next_30d_kWh",
}


class PredictionServiceError(RuntimeError):
    """Raised when a prediction cannot be produced from the configured model."""


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _model_path(target: str, model_path: str | Path | None = None) -> Path:
    if target not in MODEL_FILES:
        raise ValueError("target must be 'next_24h' or 'next_30d'.")
    return _resolve_project_path(model_path) if model_path else DEFAULT_ARTIFACT_DIR / MODEL_FILES[target]


@lru_cache(maxsize=4)
def _load_artifact_cached(path_string: str, modified_ns: int) -> dict[str, Any]:
    """Cache a model artifact; file modification time invalidates the cache."""
    del modified_ns  # included in cache key deliberately
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Train it with python ml/train.py first."
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise PredictionServiceError(f"Invalid model artifact format: {path}")
    for key in ("feature_columns", "target_key", "target_column"):
        if key not in artifact:
            raise PredictionServiceError(f"Model artifact is missing required key {key!r}: {path}")
    return artifact


def load_model_artifact(target: str = "next_24h", model_path: str | Path | None = None) -> dict[str, Any]:
    """Load the requested trained artifact, reloading automatically after retraining."""
    path = _model_path(target, model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run python ml/train.py --target {target}."
        )
    return _load_artifact_cached(str(path), path.stat().st_mtime_ns)


def predict_household_energy(
    features: Mapping[str, Any],
    target: str = "next_24h",
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Predict future electricity energy from one leakage-safe feature record.

    `features` must represent information available at the forecast origin.
    Missing numeric inputs are handled by the fitted sklearn preprocessing
    pipeline, but the service reports which values were absent/null.
    """
    if not isinstance(features, Mapping):
        raise TypeError("features must be a mapping/dictionary.")
    meter = str(features.get("meter", "")).strip()
    if not meter:
        raise ValueError("A non-empty household 'meter' ID is required.")

    artifact = load_model_artifact(target, model_path)
    if artifact.get("target_key") != target:
        raise PredictionServiceError(
            f"Requested {target}, but the selected artifact predicts {artifact.get('target_key')!r}."
        )

    feature_columns = list(artifact["feature_columns"])
    missing_fields = [name for name in feature_columns if name not in features]
    null_fields = [
        name for name in feature_columns
        if name in features and (features[name] is None or (isinstance(features[name], float) and np.isnan(features[name])))
    ]
    row = {name: features.get(name, np.nan) for name in feature_columns}
    frame = pd.DataFrame([row], columns=feature_columns)

    if "meter" in frame.columns:
        frame["meter"] = frame["meter"].astype("string")
    for column in artifact.get("numeric_features", []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    prediction = float(np.asarray(artifact["model"].predict(frame), dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(prediction):
        raise PredictionServiceError("The trained model returned a non-finite prediction.")
    prediction = max(0.0, prediction)

    return {
        "meter": meter,
        "target": artifact["target_column"],
        "horizon": target,
        "predicted_future_energy_kwh": round(prediction, 6),
        "unit": "kWh",
        "model_name": artifact.get("model_name", "saved_model"),
        "feature_timestamp_utc": features.get("timestamp_utc"),
        "missing_input_fields": missing_fields,
        "null_input_fields": null_fields,
        "note": (
            "Forecast from supplied current/past features only. Do not pass future observed meter or weather values."
        ),
    }


def get_latest_household_features(
    meter: str,
    features_path: str | Path | None = None,
    target: str = "next_24h",
    chunksize: int = 200_000,
) -> dict[str, Any]:
    """Read the latest saved feature row for one household, using bounded memory.

    This is a local-demo fallback. For live use, retrieve the latest meter and
    rolling-history features from the database and pass them to
    predict_household_energy().
    """
    meter = str(meter).strip()
    if not meter:
        raise ValueError("meter must be a non-empty household ID.")
    path = _resolve_project_path(features_path or DEFAULT_FEATURE_DATASET)
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. Run python ml/features.py first."
        )
    artifact = load_model_artifact(target)
    feature_columns = list(artifact["feature_columns"])
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    meter_col = next((c for c in columns if c.lower() in {"meter", "meter_id", "household_id"}), None)
    time_col = next((c for c in ("timestamp_utc", "timestamp_local", "timestamp") if c in columns), None)
    if meter_col is None:
        raise PredictionServiceError("Feature CSV does not contain a meter identifier column.")
    if time_col is None:
        raise PredictionServiceError("Feature CSV does not contain a timestamp column.")

    usecols = list(dict.fromkeys([meter_col, time_col] + [c for c in feature_columns if c in columns]))
    latest_row: dict[str, Any] | None = None
    latest_time: pd.Timestamp | None = None
    found_rows = 0

    for chunk in pd.read_csv(path, usecols=usecols, dtype={meter_col: "string"}, chunksize=chunksize, low_memory=False):
        chunk[meter_col] = chunk[meter_col].astype("string").str.strip()
        selected = chunk.loc[chunk[meter_col] == meter].copy()
        if selected.empty:
            continue
        parsed = pd.to_datetime(selected[time_col], errors="coerce", utc=True, format="mixed")
        selected = selected.loc[parsed.notna()].copy()
        parsed = parsed.loc[parsed.notna()]
        if selected.empty:
            continue
        idx = parsed.idxmax()
        row_time = parsed.loc[idx]
        if latest_time is None or row_time > latest_time:
            latest_time = row_time
            source = selected.loc[idx]
            latest_row = {name: source.get(name, np.nan) for name in feature_columns}
            latest_row["meter"] = meter
            latest_row["timestamp_utc"] = row_time.isoformat()
        found_rows += len(selected)

    if latest_row is None or latest_time is None:
        raise KeyError(f"No feature rows found for household {meter!r} in {path}.")

    now_utc = pd.Timestamp.now(tz="UTC")
    age_days = max(0.0, (now_utc - latest_time).total_seconds() / 86400.0)
    latest_row["_feature_timestamp_utc"] = latest_time.isoformat()
    latest_row["_feature_data_age_days"] = round(age_days, 2)
    latest_row["_stale_data_warning"] = age_days > 7.0
    latest_row["_rows_found_for_meter"] = found_rows
    return latest_row


def predict_household_from_latest_data(
    meter: str,
    target: str = "next_24h",
    features_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function: predict using the latest row in the feature CSV."""
    features = get_latest_household_features(meter, features_path, target)
    metadata = {
        "feature_timestamp_utc": features.pop("_feature_timestamp_utc"),
        "feature_data_age_days": features.pop("_feature_data_age_days"),
        "stale_data_warning": features.pop("_stale_data_warning"),
        "rows_found_for_meter": features.pop("_rows_found_for_meter"),
    }
    result = predict_household_energy(features, target)
    result.update(metadata)
    if metadata["stale_data_warning"]:
        result["warning"] = (
            "The latest available feature row is more than 7 days old. The included CEEW data ends in 2021; "
            "this is a historical demonstration prediction, not a current household forecast."
        )
    return result


def _as_weather_frame(hourly_weather: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(hourly_weather, pd.DataFrame):
        weather = hourly_weather.copy()
    else:
        weather = pd.DataFrame(list(hourly_weather))
    if weather.empty:
        raise ValueError("hourly_weather must contain hourly forecast observations.")
    if "shortwave_radiation" not in weather.columns:
        raise ValueError("hourly_weather must contain 'shortwave_radiation' in W/m².")
    return weather


def estimate_household_solar_and_bill(
    consumption_forecast_kwh: float,
    panel_capacity_kw: float,
    hourly_weather: pd.DataFrame | Sequence[Mapping[str, Any]],
    category_id: str,
    connected_load_kw: float,
    weather_location_id: str | None = None,
    billing_days: int = 30,
    performance_ratio: float | None = None,
    tariff_path: str | Path | None = None,
) -> dict[str, Any]:
    """Connect solar_service.py to bill_service.py for one billing period.

    The weather input must represent the same future billing period as the
    consumption forecast. For a 30-day bill, supply 720 distinct hourly
    forecast records for the selected location. Historical 2021 weather rows
    are not a substitute for a current future-weather forecast.
    """
    if billing_days <= 0:
        raise ValueError("billing_days must be greater than zero.")
    weather = _as_weather_frame(hourly_weather)
    location_col = "location_id" if "location_id" in weather.columns else ("meter" if "meter" in weather.columns else None)
    if location_col:
        locations = weather[location_col].dropna().astype(str).unique().tolist()
        if weather_location_id is not None:
            weather = weather.loc[weather[location_col].astype(str) == str(weather_location_id)].copy()
            if weather.empty:
                raise ValueError(f"No hourly weather records for location {weather_location_id!r}.")
        elif len(locations) > 1:
            raise ValueError("hourly_weather contains multiple locations; provide weather_location_id.")

    time_col = next((c for c in ("timestamp_utc", "weather_timestamp_utc", "date_utc", "date_ist", "timestamp_local", "timestamp", "time") if c in weather.columns), None)
    if time_col is None:
        raise ValueError("hourly_weather needs a timestamp column such as timestamp_utc, date_utc or date_ist.")
    parsed = pd.to_datetime(weather[time_col], errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        raise ValueError("hourly_weather contains missing or invalid timestamps.")
    weather = weather.assign(__parsed_time=parsed).sort_values("__parsed_time")
    if weather["__parsed_time"].duplicated().any():
        raise ValueError("hourly_weather contains duplicate timestamps for the selected location.")

    expected_hours = int(billing_days * 24)
    if len(weather) != expected_hours:
        raise ValueError(
            f"Expected exactly {expected_hours} hourly weather rows for {billing_days} billing days; "
            f"received {len(weather)}. Do not calculate a monthly solar offset from partial weather coverage."
        )
    hour_gaps = weather["__parsed_time"].diff().dropna()
    if not hour_gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("hourly_weather must have exactly one observation per consecutive hour.")
    weather = weather.drop(columns=["__parsed_time"])

    solar_result = estimate_solar_from_weather(
        consumption_kwh=float(consumption_forecast_kwh),
        panel_capacity_kw=float(panel_capacity_kw),
        hourly_weather=weather,
        performance_ratio=performance_ratio,
        irradiance_column="shortwave_radiation",
        interval_hours=1.0,
        irradiance_unit="W/m2",
        location_id=weather_location_id,
    )
    bill_result = estimate_bill_after_solar(
        solar_estimate=solar_result,
        category_id=category_id,
        connected_load_kw=connected_load_kw,
        billing_days=float(billing_days),
        tariff_path=tariff_path,
    )
    return {
        "solar": solar_result,
        "bill": bill_result,
        "period_days": billing_days,
        "weather_rows_used": expected_hours,
        "warning": (
            "Solar production is estimated from forecast irradiance and panel capacity. "
            "The result does not model intraday load matching, battery storage or export compensation."
        ),
    }


def build_household_solar_bill_summary(
    meter: str,
    panel_capacity_kw: float,
    category_id: str,
    connected_load_kw: float,
    hourly_weather: pd.DataFrame | Sequence[Mapping[str, Any]],
    weather_location_id: str | None = None,
    features: Mapping[str, Any] | None = None,
    billing_days: int = 30,
    performance_ratio: float | None = None,
    tariff_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate 24h/30d electricity predictions, solar estimate and bill estimate.

    Pass `features` from the latest household state when available. If omitted,
    the function uses the latest row in the local feature CSV (historical-demo
    mode, clearly marked as stale when older than seven days).
    """
    meter = str(meter).strip()
    if not meter:
        raise ValueError("meter is required.")
    if panel_capacity_kw <= 0:
        raise ValueError("panel_capacity_kw must be greater than zero.")
    if connected_load_kw <= 0:
        raise ValueError("connected_load_kw must be greater than zero.")

    if features is None:
        feature_row = get_latest_household_features(meter, target="next_24h")
        source_metadata = {
            "feature_timestamp_utc": feature_row.pop("_feature_timestamp_utc"),
            "feature_data_age_days": feature_row.pop("_feature_data_age_days"),
            "stale_data_warning": feature_row.pop("_stale_data_warning"),
        }
    else:
        feature_row = dict(features)
        if str(feature_row.get("meter", "")).strip() != meter:
            raise ValueError("features['meter'] must match the requested meter.")
        source_metadata = {
            "feature_timestamp_utc": feature_row.get("timestamp_utc"),
            "feature_data_age_days": None,
            "stale_data_warning": False,
        }

    prediction_24h = predict_household_energy(feature_row, target="next_24h")
    prediction_30d = predict_household_energy(feature_row, target="next_30d")
    solar_bill = estimate_household_solar_and_bill(
        consumption_forecast_kwh=prediction_30d["predicted_future_energy_kwh"],
        panel_capacity_kw=panel_capacity_kw,
        hourly_weather=hourly_weather,
        category_id=category_id,
        connected_load_kw=connected_load_kw,
        weather_location_id=weather_location_id,
        billing_days=billing_days,
        performance_ratio=performance_ratio,
        tariff_path=tariff_path,
    )
    return {
        "meter": meter,
        "feature_data": source_metadata,
        "electricity_predictions": {
            "next_24h": prediction_24h,
            "next_30d": prediction_30d,
        },
        "solar": solar_bill["solar"],
        "bill": solar_bill["bill"],
        "assumptions": [
            "The 30-day solar and bill estimate requires one future hourly irradiance record per hour for the same 30-day period.",
            "The supplied sequential BR-to-M mapping is an assumed location assignment, not verified household geolocation.",
            "Tariff estimate includes only charges present in uppcl_tariff.json; verify tariff effective date and category before real billing use.",
        ],
    }


__all__ = [
    "PredictionServiceError",
    "load_model_artifact",
    "predict_household_energy",
    "get_latest_household_features",
    "predict_household_from_latest_data",
    "estimate_household_solar_and_bill",
    "build_household_solar_bill_summary",
]
