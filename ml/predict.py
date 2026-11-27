#!/usr/bin/env python3
"""Generate a household electricity forecast from one row of current features.

Input JSON must contain `meter` and may contain the model feature fields. Missing
numeric weather/history values are passed to the saved pipeline's imputer.
The input timestamp must represent the forecast origin; values from after that
origin must not be supplied as features.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import sys

# Weighted ensemble artifacts created by running `python ml/train.py` may
# reference the helper module as top-level `ensemble`. Register a compatible
# alias when this file is imported as `ml.evaluate` / `ml.predict` by FastAPI.
try:
    from . import ensemble as _homesolar_ensemble  # type: ignore
    sys.modules.setdefault("ensemble", _homesolar_ensemble)
except ImportError:
    try:
        import ensemble as _homesolar_ensemble  # type: ignore
    except ImportError:
        _homesolar_ensemble = None
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml/artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future household electricity energy from one feature JSON record.")
    parser.add_argument("--input-json", type=Path, required=True, help="JSON file containing one feature record or a {features: {...}} object.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--target", choices=["next_24h", "next_30d"], default="next_24h")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def predict_from_features(features: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    model_file = resolve_path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_file}")
    artifact = joblib.load(model_file)
    if "meter" not in features or not str(features["meter"]).strip():
        raise ValueError("A non-empty `meter` identifier is required.")
    feature_columns = artifact["feature_columns"]
    missing = [col for col in feature_columns if col not in features]
    # Missing numeric observations are allowed because the training pipeline
    # imputes them; a missing meter identifier is not.
    row = {col: features.get(col, None) for col in feature_columns}
    frame = pd.DataFrame([row], columns=feature_columns)
    frame["meter"] = frame["meter"].astype("string")
    for col in artifact.get("numeric_features", []):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    prediction = float(artifact["model"].predict(frame)[0])
    return {
        "meter": str(features["meter"]),
        "target": artifact["target_column"],
        "horizon": artifact["target_key"],
        "predicted_future_energy_kwh": max(0.0, prediction),
        "unit": "kWh",
        "model_name": artifact.get("model_name"),
        "missing_input_fields_imputed": missing,
        "note": "Forecast uses only the supplied current/past features. It is an estimate, not an electricity bill or solar-generation measurement.",
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    try:
        input_path = resolve_path(args.input_json)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input JSON not found: {input_path}")
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        features = payload.get("features", payload) if isinstance(payload, dict) else None
        if not isinstance(features, dict):
            raise ValueError("Input JSON must be an object containing feature names and values.")
        default_model = DEFAULT_ARTIFACT_DIR / ("household_model.pkl" if args.target == "next_24h" else "household_model_30d.pkl")
        model_path = resolve_path(args.model) if args.model else default_model
        result = predict_from_features(features, model_path)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        logging.exception("Prediction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
