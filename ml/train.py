#!/usr/bin/env python3
"""Train robust LightGBM and ensemble models for HomeSolar forecasts.

Models are compared on a chronological validation period drawn only from the
ETL training partition. A LightGBM + HistGradientBoosting weighted blend is
kept only when it improves validation MAE over both individual models.
The latest chronological ETL test partition is reserved for evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:  # supports both `python ml/train.py` and `python -m ml.train`
    from .features import FEATURE_COLUMNS, NUMERIC_FEATURES, HORIZON_CONFIG
    from .ensemble import WeightedBlendRegressor
except ImportError:
    from features import FEATURE_COLUMNS, NUMERIC_FEATURES, HORIZON_CONFIG
    from ensemble import WeightedBlendRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/household_energy_features.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml/artifacts"
CHUNK_SIZE = 150_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM and ensemble HomeSolar electricity forecasting models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target", choices=["next_24h", "next_30d"], default="next_24h")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--max-train-rows", type=int, default=350_000)
    parser.add_argument("--max-validation-rows", type=int, default=100_000)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--lgbm-estimators", type=int, default=900)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def target_column(target_key: str) -> str:
    return HORIZON_CONFIG["24h" if target_key == "next_24h" else "30d"]["target"]


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0.0)
    abs_err = np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.divide(2.0 * abs_err, denominator, out=np.zeros_like(abs_err), where=denominator > 1e-9)
    wape_denominator = float(np.abs(y_true).sum())
    return {
        "mae_kwh": float(mean_absolute_error(y_true, y_pred)),
        "rmse_kwh": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "wape_percent": float(abs_err.sum() / wape_denominator * 100.0) if wape_denominator > 1e-9 else float("nan"),
        "smape_percent": float(smape.mean() * 100.0),
        "rows": int(len(y_true)),
    }


def scan_train_range(path: Path, y_col: str, chunksize: int) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    min_time = max_time = None
    count = 0
    for chunk in pd.read_csv(path, usecols=["timestamp_utc", "split", y_col], chunksize=chunksize, low_memory=False):
        target = pd.to_numeric(chunk[y_col], errors="coerce")
        mask = chunk["split"].astype("string").str.lower().eq("train") & target.notna()
        if not mask.any():
            continue
        ts = pd.to_datetime(chunk.loc[mask, "timestamp_utc"], errors="coerce", utc=True, format="mixed").dropna()
        count += len(ts)
        if len(ts):
            lo, hi = ts.min(), ts.max()
            min_time = lo if min_time is None else min(min_time, lo)
            max_time = hi if max_time is None else max(max_time, hi)
    if not count or min_time is None or max_time is None or max_time <= min_time:
        raise ValueError(f"No usable training rows found for target {y_col}. Run ml/features.py and check target coverage.")
    return min_time, max_time, count


def sample_chronological_rows(
    path: Path,
    y_col: str,
    cutoff: pd.Timestamp,
    max_train_rows: int,
    max_validation_rows: int,
    chunksize: int,
    random_state: int,
    horizon_duration: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Sample early-fit and later-validation rows with a horizon-sized embargo."""
    train_count = val_count = 0
    for chunk in pd.read_csv(path, usecols=["timestamp_utc", "split", y_col], chunksize=chunksize, low_memory=False):
        ts = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed")
        target = pd.to_numeric(chunk[y_col], errors="coerce")
        valid = chunk["split"].astype("string").str.lower().eq("train") & target.notna() & ts.notna()
        train_count += int((valid & ts.le(cutoff - horizon_duration)).sum())
        val_count += int((valid & ts.gt(cutoff)).sum())
    if train_count == 0 or val_count == 0:
        raise ValueError("Training/validation split is empty. Reduce --validation-fraction or inspect target coverage.")

    train_fraction = min(1.0, max_train_rows / train_count)
    val_fraction = min(1.0, max_validation_rows / val_count)
    required = list(dict.fromkeys(FEATURE_COLUMNS + [y_col, "split", "timestamp_utc"]))
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)

    for chunk in pd.read_csv(path, usecols=required, dtype={"meter": "string"}, chunksize=chunksize, low_memory=False):
        chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed")
        y = pd.to_numeric(chunk[y_col], errors="coerce")
        valid = chunk["split"].astype("string").str.lower().eq("train") & y.notna() & chunk["timestamp_utc"].notna()
        early = chunk.loc[valid & chunk["timestamp_utc"].le(cutoff - horizon_duration)].copy()
        late = chunk.loc[valid & chunk["timestamp_utc"].gt(cutoff)].copy()
        if len(early) and train_fraction < 1.0:
            keep = max(1, int(round(len(early) * train_fraction)))
            early = early.sample(n=min(keep, len(early)), random_state=int(rng.integers(0, 2**31 - 1)))
        if len(late) and val_fraction < 1.0:
            keep = max(1, int(round(len(late) * val_fraction)))
            late = late.sample(n=min(keep, len(late)), random_state=int(rng.integers(0, 2**31 - 1)))
        if len(early):
            train_parts.append(early)
        if len(late):
            val_parts.append(late)

    train_sample = pd.concat(train_parts, ignore_index=True)
    val_sample = pd.concat(val_parts, ignore_index=True)
    if len(train_sample) > max_train_rows:
        train_sample = train_sample.sample(n=max_train_rows, random_state=random_state)
    if len(val_sample) > max_validation_rows:
        val_sample = val_sample.sample(n=max_validation_rows, random_state=random_state + 1)
    train_sample = train_sample.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    val_sample = val_sample.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    return train_sample, val_sample, {
        "available_early_train_rows": train_count,
        "available_late_validation_rows": val_count,
        "sampled_early_train_rows": len(train_sample),
        "sampled_late_validation_rows": len(val_sample),
    }


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, ["meter"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(regressor: Any) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("regressor", regressor)])


def build_candidates(random_state: int, lgbm_estimators: int, learning_rate: float) -> dict[str, Pipeline]:
    common = {
        "n_estimators": lgbm_estimators,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.15,
        "reg_lambda": 5.0,
        "max_bin": 127,
        "n_jobs": -1,
        "random_state": random_state,
        "verbosity": -1,
        "force_col_wise": True,
    }
    return {
        "lightgbm_l1": make_pipeline(lgb.LGBMRegressor(objective="regression_l1", **common)),
        "lightgbm_huber": make_pipeline(lgb.LGBMRegressor(objective="huber", alpha=0.90, **common)),
        "hist_gradient_boosting_mae": make_pipeline(HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=60,
            l2_regularization=3.0,
            early_stopping=False,
            random_state=random_state,
        )),
    }


def feature_importance_rows(model: Pipeline, weight: float = 1.0) -> list[dict[str, Any]]:
    """Extract tree split/gain importance from a fitted LightGBM pipeline."""
    regressor = model.named_steps.get("regressor")
    if not hasattr(regressor, "feature_importances_"):
        return []
    preprocessor = model.named_steps["preprocess"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(regressor.feature_importances_))]
    values = np.asarray(regressor.feature_importances_, dtype=np.float64) * weight
    return [{"feature": str(name), "weighted_importance": float(value)} for name, value in zip(names, values)]


def run(args: argparse.Namespace) -> int:
    input_path = resolve_path(args.input)
    artifact_dir = resolve_path(args.artifact_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Feature dataset not found: {input_path}. Run ml/features.py first.")
    if args.max_train_rows < 100 or args.max_validation_rows < 50:
        raise ValueError("Training/validation sample limits are too small.")
    if not 0.05 <= args.validation_fraction <= 0.4:
        raise ValueError("--validation-fraction must be between 0.05 and 0.40.")
    if args.lgbm_estimators < 100 or args.lgbm_learning_rate <= 0:
        raise ValueError("Use at least 100 LightGBM estimators and a positive learning rate.")

    y_col = target_column(args.target)
    min_time, max_time, _ = scan_train_range(input_path, y_col, args.chunksize)
    cutoff = min_time + (max_time - min_time) * (1.0 - args.validation_fraction)
    horizon_duration = HORIZON_CONFIG["24h" if args.target == "next_24h" else "30d"]["duration"]
    logging.info("Target: %s", y_col)
    logging.info("Training-period validation cutoff: %s", cutoff)
    logging.info("LightGBM version: %s", lgb.__version__)

    train_df, val_df, sample_counts = sample_chronological_rows(
        input_path, y_col, cutoff, args.max_train_rows, args.max_validation_rows,
        args.chunksize, args.random_state, horizon_duration,
    )
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[y_col].to_numpy(dtype=np.float64)
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df[y_col].to_numpy(dtype=np.float64)

    candidate_models = build_candidates(args.random_state, args.lgbm_estimators, args.lgbm_learning_rate)
    fitted: dict[str, Pipeline] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    scores: dict[str, dict[str, float]] = {}

    # A constant baseline remains in the comparison to show whether models
    # learn more than the average target.
    baseline_prediction = np.full(len(y_val), float(np.mean(y_train)), dtype=np.float64)
    scores["mean_baseline"] = metric_summary(y_val, baseline_prediction)
    logging.info("mean_baseline validation: MAE=%.4f kWh, RMSE=%.4f kWh, WAPE=%.2f%%",
                 scores["mean_baseline"]["mae_kwh"], scores["mean_baseline"]["rmse_kwh"], scores["mean_baseline"]["wape_percent"])

    for name, model in candidate_models.items():
        logging.info("Training candidate: %s", name)
        model.fit(X_train, y_train)
        pred = np.maximum(np.asarray(model.predict(X_val), dtype=np.float64), 0.0)
        fitted[name] = model
        validation_predictions[name] = pred
        scores[name] = metric_summary(y_val, pred)
        logging.info("%s validation: MAE=%.4f kWh, RMSE=%.4f kWh, WAPE=%.2f%%",
                     name, scores[name]["mae_kwh"], scores[name]["rmse_kwh"], scores[name]["wape_percent"])

    # Blend the best LightGBM objective with HGB. Search a small, transparent
    # validation-only grid; keep the blend only if it improves validation MAE.
    best_lgbm_name = min(
        [name for name in fitted if name.startswith("lightgbm_")],
        key=lambda name: scores[name]["mae_kwh"],
    )
    hgb_name = "hist_gradient_boosting_mae"
    lgbm_pred = validation_predictions[best_lgbm_name]
    hgb_pred = validation_predictions[hgb_name]
    best_blend = None
    for lgbm_weight in np.linspace(0.0, 1.0, 21):
        blended = lgbm_weight * lgbm_pred + (1.0 - lgbm_weight) * hgb_pred
        blend_score = metric_summary(y_val, blended)
        if best_blend is None or blend_score["mae_kwh"] < best_blend["score"]["mae_kwh"]:
            best_blend = {
                "lgbm_weight": float(lgbm_weight),
                "hgb_weight": float(1.0 - lgbm_weight),
                "prediction": blended,
                "score": blend_score,
            }
    assert best_blend is not None
    scores["weighted_ensemble_lgbm_hgb"] = best_blend["score"]
    logging.info("weighted_ensemble_lgbm_hgb validation: MAE=%.4f kWh, RMSE=%.4f kWh, WAPE=%.2f%% (weights LightGBM=%.2f, HGB=%.2f)",
                 best_blend["score"]["mae_kwh"], best_blend["score"]["rmse_kwh"], best_blend["score"]["wape_percent"],
                 best_blend["lgbm_weight"], best_blend["hgb_weight"])

    # Compare actual single models and the blend; baseline is diagnostic only.
    model_score_names = list(fitted.keys()) + ["weighted_ensemble_lgbm_hgb"]
    best_name = min(model_score_names, key=lambda name: scores[name]["mae_kwh"])
    logging.info("Selected model by chronological validation MAE: %s", best_name)

    # Refit only the selected model(s) on the combined pre-test training period.
    refit_df = pd.concat([train_df, val_df], ignore_index=True)
    X_refit = refit_df[FEATURE_COLUMNS]
    y_refit = refit_df[y_col].to_numpy(dtype=np.float64)
    if best_name == "weighted_ensemble_lgbm_hgb":
        selected_lgbm = build_candidates(args.random_state, args.lgbm_estimators, args.lgbm_learning_rate)[best_lgbm_name]
        selected_hgb = build_candidates(args.random_state, args.lgbm_estimators, args.lgbm_learning_rate)[hgb_name]
        selected_lgbm.fit(X_refit, y_refit)
        selected_hgb.fit(X_refit, y_refit)
        best_model: Any = WeightedBlendRegressor(
            {best_lgbm_name: selected_lgbm, hgb_name: selected_hgb},
            {best_lgbm_name: best_blend["lgbm_weight"], hgb_name: best_blend["hgb_weight"]},
        )
        ensemble_weights = best_model.weights
        importance = feature_importance_rows(selected_lgbm, best_model.weights[best_lgbm_name])
    else:
        best_model = build_candidates(args.random_state, args.lgbm_estimators, args.lgbm_learning_rate)[best_name]
        best_model.fit(X_refit, y_refit)
        ensemble_weights = None
        importance = feature_importance_rows(best_model)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = "household_model.pkl" if args.target == "next_24h" else "household_model_30d.pkl"
    artifact_path = artifact_dir / artifact_name
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "target_key": args.target,
        "target_column": y_col,
        "target_unit": "kWh over the forecast horizon",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": ["meter"],
        "validation_metrics": scores,
        "selected_validation_metrics": scores[best_name],
        "ensemble_weights": ensemble_weights,
        "training_config": {
            "input_file": str(input_path),
            "training_sample_rows": int(len(refit_df)),
            "early_fit_sample_rows": int(len(train_df)),
            "chronological_validation_sample_rows": int(len(val_df)),
            "validation_cutoff_utc": cutoff.isoformat(),
            "horizon_embargo": str(horizon_duration),
            "available_rows": sample_counts,
            "random_state": args.random_state,
            "lightgbm_version": lgb.__version__,
            "lightgbm_estimators": args.lgbm_estimators,
            "lightgbm_learning_rate": args.lgbm_learning_rate,
            "history_features": [
                "past_24h_sum_kWh", "past_7d_sum_kWh", "past_30d_sum_kWh",
                "past_24h_mean_interval_kWh", "past_7d_mean_interval_kWh", "past_30d_mean_interval_kWh",
            ],
        },
        "notes": [
            "Primary candidates are LightGBM L1, LightGBM Huber, and HistGradientBoosting absolute-error regression.",
            "A weighted LightGBM + HistGradientBoosting blend is selected only when it improves chronological validation MAE.",
            "Only the ETL split=train partition is used for training/validation; split=test is reserved for evaluate.py.",
            "History features use only readings before the forecast timestamp.",
            "The model uses weather observations available at the forecast timestamp, not future observed weather. For operational future-weather inputs, connect a weather forecast provider.",
        ],
    }
    joblib.dump(artifact, artifact_path)
    report_path = artifact_dir / f"training_report_{args.target}.json"
    report = {
        "model_artifact": str(artifact_path),
        "selected_model": best_name,
        "target": y_col,
        "validation_metrics_by_candidate": scores,
        "ensemble_weights": ensemble_weights,
        "training_config": artifact["training_config"],
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if importance:
        importance_df = pd.DataFrame(importance)
        importance_df = importance_df.groupby("feature", as_index=False)["weighted_importance"].sum()
        importance_df = importance_df.sort_values("weighted_importance", ascending=False)
        importance_df.to_csv(artifact_dir / f"feature_importance_{args.target}.csv", index=False)

    logging.info("Saved model: %s", artifact_path)
    logging.info("Saved training report: %s", report_path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        logging.exception("Model training failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
