#!/usr/bin/env python3
"""Append new HomeSolar meter/weather batches, rerun ETL, and promote only better models.

This is the self-development workflow used by the Express upload endpoint.
It does not overwrite raw source files. It merges the original CEEW/Open-Meteo
files with every CSV archived under data/raw/incoming/, runs the existing ETL,
regenerates features, trains candidate LightGBM/ensemble models, and compares
incumbent and candidate models on the same held-out rows from newly arrived
chronological test data.

A candidate is promoted per horizon only when it beats the incumbent MAE by
at least --min-improvement-pct on that common new-data comparison window.
Otherwise the currently deployed artifact remains active.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "household_energy_features.csv"
RAW_ELECTRICITY = PROJECT_ROOT / "data" / "raw" / "electricity" / "CEEW - Smart meter data Bareilly 2021.csv"
RAW_WEATHER = PROJECT_ROOT / "data" / "raw" / "weather" / "bareilly_36_meters_weather_2021.csv"
LOCATION_MAP = PROJECT_ROOT / "data" / "raw" / "household_locations.csv"
UPDATE_HISTORY = ARTIFACT_DIR / "model_update_history.json"
TARGETS = {
    "next_24h": "target_next_24h_kWh",
    "next_30d": "target_next_30d_kWh",
}


def _python() -> str:
    configured = os.getenv("HOMESOLAR_PYTHON") or os.getenv("PYTHON_EXECUTABLE")
    if configured:
        return str(Path(configured).expanduser())
    if os.name == "nt":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.is_file() else sys.executable)


def _resolve(value: str | Path | None, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default.resolve()
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _run(command: list[str], label: str, timeout_seconds: int = 3600) -> None:
    logging.info("UPDATE STAGE: %s", label)
    logging.info("COMMAND: %s", subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.stdout:
        for line in completed.stdout.splitlines()[-35:]:
            logging.info("%s", line)
    if completed.stderr:
        for line in completed.stderr.splitlines()[-35:]:
            logging.info("%s", line)
    if completed.returncode != 0:
        detail = "\n".join((completed.stdout or "").splitlines()[-30:] + (completed.stderr or "").splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {completed.returncode}). Last output:\n{detail[-12000:]}")


def _csv_header(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def merge_csv_sources(sources: Iterable[Path], output: Path, chunksize: int = 250_000) -> dict[str, Any]:
    """Stream multiple same-schema CSVs into a staged file without editing sources."""
    sources = [p.resolve() for p in sources if p is not None]
    if not sources:
        raise ValueError("No source CSVs were supplied for merging.")
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"CSV source does not exist: {path}")
    expected_header = _csv_header(sources[0])
    for path in sources[1:]:
        current_header = _csv_header(path)
        if current_header != expected_header:
            raise ValueError(
                f"CSV schema mismatch in {path.name}.\nExpected: {expected_header}\nReceived: {current_header}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    rows_written = 0
    first = True
    for source in sources:
        for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False, encoding="utf-8-sig"):
            chunk.to_csv(
                output,
                mode="w" if first else "a",
                header=first,
                index=False,
                encoding="utf-8-sig" if first else "utf-8",
            )
            rows_written += len(chunk)
            first = False
    if first:
        raise ValueError(f"No rows were found in source files: {sources}")
    return {
        "sources": [str(p) for p in sources],
        "columns": expected_header,
        "rows_before_ETL_deduplication": rows_written,
        "merged_file": str(output),
    }


def _incoming_sources(folder: Path, uploaded: Path, base: Path) -> list[Path]:
    """Include all previously archived incoming batches plus the current file."""
    if not base.is_file():
        raise FileNotFoundError(f"Base historical CSV not found: {base}")
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(p.resolve() for p in folder.glob("*.csv") if p.is_file())
    uploaded = uploaded.resolve()
    if uploaded not in files:
        files.append(uploaded)
    # Keep each incoming file only once. Base file remains first.
    unique: list[Path] = [base.resolve()]
    seen = {str(base.resolve()).casefold()}
    for item in files:
        key = str(item).casefold()
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _latest_feature_timestamp(path: Path, chunksize: int = 200_000) -> pd.Timestamp:
    if not path.is_file():
        raise FileNotFoundError(
            f"Previous feature dataset not found: {path}. Train the baseline models before using online updates."
        )
    max_ts: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=["timestamp_utc"], chunksize=chunksize, low_memory=False):
        parsed = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed").dropna()
        if not parsed.empty:
            part_max = parsed.max()
            max_ts = part_max if max_ts is None else max(max_ts, part_max)
    if max_ts is None:
        raise ValueError("The existing feature dataset contains no valid timestamp_utc values.")
    return max_ts


def _scan_new_train_rows(path: Path, baseline_max: pd.Timestamp, target_col: str) -> int:
    count = 0
    cols = ["timestamp_utc", "split", target_col]
    for chunk in pd.read_csv(path, usecols=cols, chunksize=200_000, low_memory=False):
        ts = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed")
        split_train = chunk["split"].astype("string").str.lower().eq("train")
        labels = pd.to_numeric(chunk[target_col], errors="coerce").notna()
        count += int((split_train & labels & ts.gt(baseline_max)).sum())
    return count


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Incumbent model artifact missing: {path}")
    # WeightedBlendRegressor may have been serialized under the top-level name
    # `ensemble` when the trainer was invoked as `python ml/train.py`.
    ml_dir = str(PROJECT_ROOT / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    try:
        import ensemble  # noqa: F401
    except ImportError:
        pass
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unsupported model artifact: {path}")
    return artifact


def _metric_accumulator() -> dict[str, float]:
    return {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_abs_y": 0.0, "sum_smape": 0.0}


def _update_metrics(acc: dict[str, float], y: np.ndarray, pred: np.ndarray) -> None:
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 0.0)
    y = np.asarray(y, dtype=np.float64)
    error = y - pred
    acc["n"] += int(len(y))
    acc["sum_abs"] += float(np.abs(error).sum())
    acc["sum_sq"] += float(np.square(error).sum())
    acc["sum_abs_y"] += float(np.abs(y).sum())
    denom = np.abs(y) + np.abs(pred)
    smape = np.divide(2.0 * np.abs(error), denom, out=np.zeros_like(error), where=denom > 1e-9)
    acc["sum_smape"] += float(smape.sum())


def _finish_metrics(acc: dict[str, float]) -> dict[str, Any] | None:
    n = int(acc["n"])
    if n == 0:
        return None
    return {
        "rows": n,
        "mae_kwh": acc["sum_abs"] / n,
        "rmse_kwh": float(np.sqrt(acc["sum_sq"] / n)),
        "wape_percent": acc["sum_abs"] / acc["sum_abs_y"] * 100.0 if acc["sum_abs_y"] > 1e-12 else None,
        "smape_percent": acc["sum_smape"] / n * 100.0,
    }


def compare_incumbent_candidate(
    feature_path: Path,
    incumbent_path: Path,
    candidate_path: Path,
    target_col: str,
    baseline_max: pd.Timestamp,
    chunksize: int = 100_000,
) -> dict[str, Any]:
    """Compare both models on identical newly arrived held-out rows only."""
    incumbent = _load_artifact(incumbent_path)
    candidate = _load_artifact(candidate_path)
    incumbent_features = list(incumbent.get("feature_columns", []))
    candidate_features = list(candidate.get("feature_columns", []))
    if incumbent_features != candidate_features:
        raise ValueError(
            "The deployed and candidate models use different input-feature schemas. "
            "Cannot make a fair same-row comparison. Retrain a baseline with the current ml/features.py first."
        )
    feature_cols = candidate_features
    required = list(dict.fromkeys(["timestamp_utc", "split", target_col] + feature_cols))
    acc_old, acc_new = _metric_accumulator(), _metric_accumulator()
    min_test_ts: pd.Timestamp | None = None
    max_test_ts: pd.Timestamp | None = None

    for chunk in pd.read_csv(feature_path, usecols=required, dtype={"meter": "string"}, chunksize=chunksize, low_memory=False):
        ts = pd.to_datetime(chunk["timestamp_utc"], errors="coerce", utc=True, format="mixed")
        split_test = chunk["split"].astype("string").str.lower().eq("test")
        y = pd.to_numeric(chunk[target_col], errors="coerce")
        mask = split_test & ts.gt(baseline_max) & y.notna() & ts.notna()
        if not mask.any():
            continue
        sample = chunk.loc[mask].copy()
        sample_ts = ts.loc[mask]
        min_part, max_part = sample_ts.min(), sample_ts.max()
        min_test_ts = min_part if min_test_ts is None else min(min_test_ts, min_part)
        max_test_ts = max_part if max_test_ts is None else max(max_test_ts, max_part)
        X = sample[feature_cols].copy()
        if "meter" in X.columns:
            X["meter"] = X["meter"].astype("string")
        y_np = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=np.float64)
        pred_old = incumbent["model"].predict(X)
        pred_new = candidate["model"].predict(X)
        _update_metrics(acc_old, y_np, pred_old)
        _update_metrics(acc_new, y_np, pred_new)

    old_metrics = _finish_metrics(acc_old)
    new_metrics = _finish_metrics(acc_new)
    if old_metrics is None or new_metrics is None:
        raise ValueError(
            f"No valid newly arrived held-out labels for {target_col}. "
            "A 24-hour target needs at least 24 hours of future data; a 30-day target needs at least 30 days, "
            "with the configured coverage threshold. More new data may be required."
        )
    return {
        "comparison_window_start_utc": min_test_ts.isoformat() if min_test_ts is not None else None,
        "comparison_window_end_utc": max_test_ts.isoformat() if max_test_ts is not None else None,
        "baseline": old_metrics,
        "candidate": new_metrics,
    }


def _append_history(record: dict[str, Any]) -> None:
    UPDATE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if UPDATE_HISTORY.is_file():
        try:
            history = json.loads(UPDATE_HISTORY.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(record)
    temp = UPDATE_HISTORY.with_suffix(".json.tmp")
    temp.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    temp.replace(UPDATE_HISTORY)


def run_update(
    new_electricity: str | Path,
    new_weather: str | Path,
    base_electricity: str | Path | None = None,
    base_weather: str | Path | None = None,
    location_map: str | Path | None = None,
    min_improvement_pct: float = 0.5,
) -> dict[str, Any]:
    """Run the full append -> ETL -> features -> candidate -> compare -> promote workflow."""
    if not 0 <= float(min_improvement_pct) < 100:
        raise ValueError("min_improvement_pct must be in [0, 100).")
    base_electricity_path = _resolve(base_electricity or os.getenv("HOMESOLAR_BASE_ELECTRICITY"), RAW_ELECTRICITY)
    base_weather_path = _resolve(base_weather or os.getenv("HOMESOLAR_BASE_WEATHER"), RAW_WEATHER)
    location_map_path = _resolve(location_map or os.getenv("HOMESOLAR_LOCATION_MAP"), LOCATION_MAP)
    uploaded_electricity_path = _resolve(new_electricity, PROJECT_ROOT / "data/raw/incoming/electricity/missing.csv")
    uploaded_weather_path = _resolve(new_weather, PROJECT_ROOT / "data/raw/incoming/weather/missing.csv")

    if not uploaded_electricity_path.is_file() or not uploaded_weather_path.is_file():
        raise FileNotFoundError("Both new electricity and new hourly weather CSV uploads are required.")
    if not location_map_path.is_file():
        raise FileNotFoundError(f"Household-to-weather mapping file not found: {location_map_path}")
    if not FEATURES_CSV.is_file():
        raise FileNotFoundError("Baseline features file not found. Run ml/features.py and train the baseline models first.")

    baseline_max = _latest_feature_timestamp(FEATURES_CSV)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    stage_dir = PROJECT_ROOT / "data" / "interim" / "update_staging" / run_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    incoming_electricity_dir = PROJECT_ROOT / "data" / "raw" / "incoming" / "electricity"
    incoming_weather_dir = PROJECT_ROOT / "data" / "raw" / "incoming" / "weather"
    electricity_sources = _incoming_sources(incoming_electricity_dir, uploaded_electricity_path, base_electricity_path)
    weather_sources = _incoming_sources(incoming_weather_dir, uploaded_weather_path, base_weather_path)
    merged_electricity = stage_dir / "electricity_all_batches.csv"
    merged_weather = stage_dir / "weather_all_batches.csv"
    merged_electricity_report = merge_csv_sources(electricity_sources, merged_electricity)
    merged_weather_report = merge_csv_sources(weather_sources, merged_weather)

    logging.info("Previous feature-data maximum timestamp: %s", baseline_max)
    _run([
        _python(), str(PROJECT_ROOT / "etl" / "run_pipeline.py"),
        "--electricity-input", str(merged_electricity),
        "--weather-input", str(merged_weather),
        "--location-map", str(location_map_path),
    ], "Run existing six-step ETL pipeline", timeout_seconds=3600)
    _run([_python(), str(PROJECT_ROOT / "ml" / "features.py")], "Regenerate ML targets and features", timeout_seconds=3600)

    feature_path = FEATURES_CSV
    new_training_counts: dict[str, int] = {}
    per_target: dict[str, Any] = {}
    candidate_dir = ARTIFACT_DIR / "candidates" / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    version_dir = ARTIFACT_DIR / "model_versions" / run_id
    version_dir.mkdir(parents=True, exist_ok=True)

    for target_key, target_col in TARGETS.items():
        incumbent_path = ARTIFACT_DIR / ("household_model.pkl" if target_key == "next_24h" else "household_model_30d.pkl")
        candidate_path = candidate_dir / incumbent_path.name
        try:
            if not incumbent_path.is_file():
                per_target[target_key] = {"status": "skipped", "reason": f"No deployed model at {incumbent_path}"}
                continue

            new_train_rows = _scan_new_train_rows(feature_path, baseline_max, target_col)
            new_training_counts[target_key] = new_train_rows
            if new_train_rows < 100:
                per_target[target_key] = {
                    "status": "skipped",
                    "reason": f"Only {new_train_rows} valid training rows are newer than the previous dataset; at least 100 are required.",
                    "new_training_rows": new_train_rows,
                }
                continue

            _run([
                _python(), str(PROJECT_ROOT / "ml" / "train.py"),
                "--target", target_key,
                "--artifact-dir", str(candidate_dir),
            ], f"Train candidate {target_key}", timeout_seconds=7200)
            if not candidate_path.is_file():
                per_target[target_key] = {"status": "skipped", "reason": "Candidate training did not create a model artifact."}
                continue

            comparison = compare_incumbent_candidate(
                feature_path, incumbent_path, candidate_path, target_col, baseline_max
            )
            base_mae = float(comparison["baseline"]["mae_kwh"])
            candidate_mae = float(comparison["candidate"]["mae_kwh"])
            improvement = ((base_mae - candidate_mae) / base_mae * 100.0) if base_mae > 1e-12 else 0.0
            promoted = improvement >= float(min_improvement_pct) and candidate_mae < base_mae

            record_for_target = {
                "status": "promoted" if promoted else "kept_incumbent",
                "new_training_rows": new_train_rows,
                "min_improvement_required_pct": float(min_improvement_pct),
                "improvement_pct": round(improvement, 4),
                "comparison": comparison,
            }
            if promoted:
                shutil.copy2(incumbent_path, version_dir / f"previous_{incumbent_path.name}")
                shutil.copy2(candidate_path, incumbent_path)
                candidate_report = candidate_dir / f"training_report_{target_key}.json"
                if candidate_report.is_file():
                    shutil.copy2(candidate_report, ARTIFACT_DIR / candidate_report.name)
                record_for_target["promoted_model_path"] = str(incumbent_path)
            else:
                record_for_target["deployed_model_unchanged"] = str(incumbent_path)
            per_target[target_key] = record_for_target

            # Refresh the dashboard's current-model test metrics using the same
            # newly arrived held-out rows. This avoids displaying stale metrics
            # from the original pre-update test period after a model promotion.
            deployed_metrics = comparison["candidate"] if promoted else comparison["baseline"]
            active_artifact = _load_artifact(incumbent_path)
            evaluation_record = {
                "model_file": str(incumbent_path),
                "model_name": active_artifact.get("model_name"),
                "target": target_col,
                "evaluation_split": "newly arrived held-out test rows after previous dataset end",
                "rows_evaluated": deployed_metrics["rows"],
                "mae_kwh": deployed_metrics["mae_kwh"],
                "rmse_kwh": deployed_metrics["rmse_kwh"],
                "wape_percent": deployed_metrics["wape_percent"],
                "smape_percent": deployed_metrics["smape_percent"],
                "comparison_window_start_utc": comparison["comparison_window_start_utc"],
                "comparison_window_end_utc": comparison["comparison_window_end_utc"],
                "warning": "Metrics are measured on the newly arrived held-out rows used for this update decision.",
            }
            evaluation_path = ARTIFACT_DIR / f"evaluation_{target_key}.json"
            evaluation_path.write_text(json.dumps(evaluation_record, indent=2, default=str), encoding="utf-8")

            comparison_path = ARTIFACT_DIR / f"online_update_comparison_{target_key}.json"
            comparison_path.write_text(json.dumps(record_for_target, indent=2, default=str), encoding="utf-8")
        except Exception as target_error:
            logging.exception("Update comparison/training failed for %s", target_key)
            per_target[target_key] = {"status": "error", "reason": str(target_error), "new_training_rows": new_training_counts.get(target_key, 0)}

    statuses = [v.get("status") for v in per_target.values()]
    overall = (
        "promoted" if "promoted" in statuses
        else "no_improvement" if "kept_incumbent" in statuses
        else "partial_error" if "error" in statuses
        else "insufficient_new_data"
    )
    latest_new_ts = _latest_feature_timestamp(feature_path)
    result = {
        "run_id": run_id,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "baseline_max_timestamp_utc": baseline_max.isoformat(),
        "latest_dataset_timestamp_utc": latest_new_ts.isoformat(),
        "min_improvement_required_pct": float(min_improvement_pct),
        "merged_inputs": {
            "electricity": merged_electricity_report,
            "weather": merged_weather_report,
        },
        "new_training_rows": new_training_counts,
        "targets": per_target,
        "model_versions_directory": str(version_dir),
        "note": (
            "A candidate was evaluated on the same newly arrived chronological test rows as the incumbent. "
            "Only a candidate that improves MAE by the configured threshold is promoted. "
            "If no target is promoted, the existing deployed model files remain unchanged."
        ),
    }
    _append_history(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun HomeSolar ETL and retrain/promote models after new data arrives.")
    parser.add_argument("--new-electricity", required=True, type=Path)
    parser.add_argument("--new-weather", required=True, type=Path)
    parser.add_argument("--base-electricity", type=Path, default=None)
    parser.add_argument("--base-weather", type=Path, default=None)
    parser.add_argument("--location-map", type=Path, default=None)
    parser.add_argument("--min-improvement-pct", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    try:
        result = run_update(
            new_electricity=args.new_electricity,
            new_weather=args.new_weather,
            base_electricity=args.base_electricity,
            base_weather=args.base_weather,
            location_map=args.location_map,
            min_improvement_pct=args.min_improvement_pct,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as exc:
        logging.exception("Online model update failed: %s", exc)
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
