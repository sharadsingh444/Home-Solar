#!/usr/bin/env python3
"""Evaluate a saved HomeSolar model on the untouched chronological test split."""
from __future__ import annotations

import argparse
import json
import logging
import math
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = PROJECT_ROOT / "data/processed/household_energy_features.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml/artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved HomeSolar electricity model on split=test rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--target", choices=["next_24h", "next_30d"], default="next_24h")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--plot-sample-rows", type=int, default=20_000)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    try:
        input_path = resolve_path(args.input)
        artifact_dir = resolve_path(args.artifact_dir)
        default_model = artifact_dir / ("household_model.pkl" if args.target == "next_24h" else "household_model_30d.pkl")
        model_path = resolve_path(args.model) if args.model else default_model
        if not input_path.is_file():
            raise FileNotFoundError(f"Feature dataset not found: {input_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}. Run ml/train.py first.")
        artifact: dict[str, Any] = joblib.load(model_path)
        model = artifact["model"]
        feature_columns = artifact["feature_columns"]
        y_col = artifact["target_column"]
        if artifact.get("target_key") != args.target:
            raise ValueError(f"Artifact predicts {artifact.get('target_key')}, but --target={args.target} was requested.")

        usecols = list(dict.fromkeys(feature_columns + [y_col, "split", "timestamp_utc"]))
        n = 0
        sum_abs = 0.0
        sum_sq = 0.0
        sum_abs_actual = 0.0
        smape_sum = 0.0
        mape_sum = 0.0
        mape_n = 0
        sample_frames: list[pd.DataFrame] = []
        # Keep household-level diagnostics so an overall score cannot hide
        # poor performance on a subset of meters with shorter/longer histories.
        per_meter: dict[str, dict[str, Any]] = {}
        absolute_error_chunks: list[np.ndarray] = []
        rng = np.random.default_rng(42)

        for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=args.chunksize, low_memory=False):
            mask = chunk["split"].astype("string").str.lower().eq("test")
            chunk = chunk.loc[mask].copy()
            if chunk.empty:
                continue
            chunk[y_col] = pd.to_numeric(chunk[y_col], errors="coerce")
            chunk = chunk.loc[chunk[y_col].notna()].copy()
            if chunk.empty:
                continue
            X = chunk[feature_columns].copy()
            if "meter" in X.columns:
                X["meter"] = X["meter"].astype("string")
            y_true = chunk[y_col].to_numpy(dtype=np.float64)
            y_pred = np.maximum(np.asarray(model.predict(X), dtype=np.float64), 0.0)
            err = y_true - y_pred
            abs_err = np.abs(err)
            absolute_error_chunks.append(abs_err.copy())
            n += len(y_true)
            sum_abs += float(abs_err.sum())
            sum_sq += float(np.square(err).sum())
            sum_abs_actual += float(np.abs(y_true).sum())
            denom = np.abs(y_true) + np.abs(y_pred)
            smape_sum += float(np.divide(2.0 * abs_err, denom, out=np.zeros_like(abs_err), where=denom > 1e-9).sum())

            if "meter" in chunk.columns:
                meter_values = chunk["meter"].astype("string").fillna("UNKNOWN").astype(str).to_numpy()
                for meter_name in np.unique(meter_values):
                    selected = meter_values == meter_name
                    item = per_meter.setdefault(meter_name, {
                        "rows": 0, "sum_abs_error": 0.0, "sum_squared_error": 0.0,
                        "sum_abs_actual": 0.0,
                        "absolute_errors": [],
                    })
                    item["rows"] += int(selected.sum())
                    item["sum_abs_error"] += float(abs_err[selected].sum())
                    item["sum_squared_error"] += float(np.square(err[selected]).sum())
                    item["sum_abs_actual"] += float(np.abs(y_true[selected]).sum())
                    item["absolute_errors"].append(abs_err[selected].copy())

            nonzero = np.abs(y_true) > 1e-6
            if nonzero.any():
                mape_sum += float((abs_err[nonzero] / np.abs(y_true[nonzero])).sum())
                mape_n += int(nonzero.sum())

            if args.plot_sample_rows > 0:
                keep_n = min(args.plot_sample_rows, max(1, int(len(chunk) * min(1.0, args.plot_sample_rows / max(n, 1)))))
                sampled_idx = rng.choice(len(chunk), size=keep_n, replace=False) if keep_n < len(chunk) else np.arange(len(chunk))
                sample_frames.append(pd.DataFrame({
                    "timestamp_utc": chunk.iloc[sampled_idx]["timestamp_utc"].astype(str).to_numpy(),
                    "actual_kwh": y_true[sampled_idx],
                    "predicted_kwh": y_pred[sampled_idx],
                }))

        if n == 0:
            raise ValueError(f"No valid split=test targets found for {y_col}. Check future-window coverage and test dates.")
        per_meter_metrics = {}
        for meter_name, item in sorted(per_meter.items()):
            meter_rows = max(int(item["rows"]), 1)
            meter_abs = np.concatenate(item["absolute_errors"]) if item["absolute_errors"] else np.array([], dtype=float)
            per_meter_metrics[meter_name] = {
                "rows": int(item["rows"]),
                "mae_kwh": float(item["sum_abs_error"] / meter_rows),
                "rmse_kwh": float(math.sqrt(item["sum_squared_error"] / meter_rows)),
                "wape_percent": float(item["sum_abs_error"] / item["sum_abs_actual"] * 100.0)
                    if item["sum_abs_actual"] > 1e-9 else None,
                "p95_absolute_error_kwh": float(np.quantile(meter_abs, 0.95)) if meter_abs.size else None,
                "max_absolute_error_kwh": float(meter_abs.max()) if meter_abs.size else None,
            }

        all_abs = np.concatenate(absolute_error_chunks) if absolute_error_chunks else np.array([], dtype=float)
        metrics = {
            "model_file": str(model_path),
            "model_name": artifact.get("model_name"),
            "target": y_col,
            "evaluation_split": "test (latest chronological period)",
            "rows_evaluated": n,
            "mae_kwh": sum_abs / n,
            "rmse_kwh": math.sqrt(sum_sq / n),
            "median_absolute_error_kwh": float(np.quantile(all_abs, 0.50)),
            "p95_absolute_error_kwh": float(np.quantile(all_abs, 0.95)),
            "p99_absolute_error_kwh": float(np.quantile(all_abs, 0.99)),
            "max_absolute_error_kwh": float(all_abs.max()),
            "wape_percent": sum_abs / sum_abs_actual * 100.0 if sum_abs_actual > 1e-9 else None,
            "smape_percent": smape_sum / n * 100.0,
            "mape_percent_nonzero_actual_only": mape_sum / mape_n * 100.0 if mape_n else None,
            "mape_rows_used": mape_n,
            "per_meter_metrics": per_meter_metrics,
            "macro_average_household_mae_kwh": float(np.mean([v["mae_kwh"] for v in per_meter_metrics.values()])) if per_meter_metrics else None,
            "warning": "MAPE excludes rows with actual target <= 1e-6 kWh; WAPE and sMAPE are preferred when targets may be near zero. Forecast origins overlap (3-minute spacing), so evaluation rows are not independent daily/monthly forecasts.",
        }
        artifact_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = artifact_dir / f"evaluation_{args.target}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if sample_frames:
            plot_df = pd.concat(sample_frames, ignore_index=True)
            if len(plot_df) > args.plot_sample_rows:
                plot_df = plot_df.sample(n=args.plot_sample_rows, random_state=42)
            plot_dir = artifact_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(plot_df["actual_kwh"], plot_df["predicted_kwh"], alpha=0.25, s=8)
            lo = float(min(plot_df["actual_kwh"].min(), plot_df["predicted_kwh"].min()))
            hi = float(max(plot_df["actual_kwh"].max(), plot_df["predicted_kwh"].max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
            ax.set_xlabel("Actual future electricity energy (kWh)")
            ax.set_ylabel("Predicted future electricity energy (kWh)")
            ax.set_title(f"Actual vs predicted - {args.target}")
            fig.tight_layout()
            scatter_path = plot_dir / f"actual_vs_predicted_{args.target}.png"
            fig.savefig(scatter_path, dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 5))
            plot_df = plot_df.copy()
            plot_df["timestamp_utc"] = pd.to_datetime(plot_df["timestamp_utc"], utc=True, errors="coerce", format="mixed")
            plot_df = plot_df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").head(1000)
            ax.plot(plot_df["timestamp_utc"], plot_df["actual_kwh"], label="Actual", linewidth=1)
            ax.plot(plot_df["timestamp_utc"], plot_df["predicted_kwh"], label="Predicted", linewidth=1)
            ax.set_xlabel("Timestamp (UTC)")
            ax.set_ylabel("Future electricity energy (kWh)")
            ax.set_title(f"Sample predictions over time - {args.target}")
            ax.legend()
            fig.tight_layout()
            time_path = plot_dir / f"prediction_timeline_{args.target}.png"
            fig.savefig(time_path, dpi=160)
            plt.close(fig)
            metrics["plots"] = [str(scatter_path), str(time_path)]
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        logging.info("Evaluation rows: %s", f"{n:,}")
        logging.info("MAE: %.5f kWh | RMSE: %.5f kWh | WAPE: %s%% | sMAPE: %.2f%%",
                     metrics["mae_kwh"], metrics["rmse_kwh"],
                     f"{metrics['wape_percent']:.2f}" if metrics["wape_percent"] is not None else "N/A",
                     metrics["smape_percent"])
        logging.info("Metrics saved: %s", metrics_path)
        return 0
    except Exception as exc:
        logging.exception("Model evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
