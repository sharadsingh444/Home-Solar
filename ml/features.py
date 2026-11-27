#!/usr/bin/env python3
"""Create leakage-safe future-consumption targets and ML features for HomeSolar.

Input:
    data/processed/household_energy_dataset.csv

Output:
    data/processed/household_energy_features.csv
    data/processed/household_energy_features_profile.json

The CEEW t_kWh field is treated as energy consumed during each meter interval.
Targets sum future interval energy over (forecast_time, forecast_time + horizon].
Each target is calculated within the same household AND the same chronological
split so a training target cannot consume testing-period readings.

Example:
    python ml/features.py
    python ml/features.py --horizons 24h 30d --minimum-coverage 0.95
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/processed/household_energy_dataset.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/household_energy_features.csv"
DEFAULT_PROFILE = PROJECT_ROOT / "data/processed/household_energy_features_profile.json"

# These are inputs known at the forecast timestamp. Future target columns are
# deliberately excluded. `meter` is handled as a categorical feature in train.py.
NUMERIC_FEATURES = [
    "t_kWh",
    "avg_voltage_v",
    "avg_current_a",
    "frequency_hz",
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "dew_point_2m",
    "shortwave_radiation",
    "direct_radiation",
    "global_tilted_irradiance",
    "weather_time_offset_minutes",
    "weather_available",
    "lag_1_reading_kWh",
    "lag_3_reading_kWh",
    "lag_10_reading_kWh",
    "rolling_1h_mean_kWh",
    "rolling_1h_sum_kWh",
    "rolling_1h_std_kWh",
    "rolling_1h_count",
    "minutes_since_previous_reading",
    # Longer historical summaries are particularly useful for 24-hour and
    # 30-day consumption forecasts. Each window is closed on the left so the
    # current reading is never included in its own historical features.
    "past_24h_sum_kWh",
    "past_24h_mean_interval_kWh",
    "past_24h_count",
    "past_7d_sum_kWh",
    "past_7d_mean_interval_kWh",
    "past_7d_count",
    "past_30d_sum_kWh",
    "past_30d_mean_interval_kWh",
    "past_30d_count",
    "hour",
    "day",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]
CATEGORICAL_FEATURES = ["meter"]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

HORIZON_CONFIG = {
    "24h": {
        "duration": pd.Timedelta(hours=24),
        "target": "target_next_24h_kWh",
        "coverage": "target_next_24h_coverage",
        "label": "Total future interval electricity energy over the next 24 hours (kWh)",
    },
    "30d": {
        "duration": pd.Timedelta(days=30),
        "target": "target_next_30d_kWh",
        "coverage": "target_next_30d_coverage",
        "label": "Total future interval electricity energy over the next 30 days (kWh)",
    },
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 24-hour and 30-day future electricity targets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--horizons", nargs="+", choices=sorted(HORIZON_CONFIG), default=["24h", "30d"])
    parser.add_argument("--interval-minutes", type=int, default=3,
                         help="Expected CEEW meter interval in minutes. Default: 3.")
    parser.add_argument("--minimum-coverage", type=float, default=0.95,
                         help="Minimum fraction of expected future interval rows required for a label.")
    parser.add_argument("--boundary-tolerance-intervals", type=float, default=2.0,
                         help="Allow the first/last future record to be this many intervals from window boundaries.")
    parser.add_argument("--chunksize", type=int, default=150_000)
    return parser.parse_args()


def _compute_forward_target(
    frame: pd.DataFrame,
    timestamp_ns: np.ndarray,
    energy: np.ndarray,
    horizon: pd.Timedelta,
    interval_minutes: int,
    minimum_coverage: float,
    boundary_tolerance_intervals: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return target sums and coverage ratios for the future (t, t+horizon] window."""
    n = len(frame)
    expected = int(round(horizon / pd.Timedelta(minutes=interval_minutes)))
    if expected <= 0:
        raise ValueError("The forecast horizon must contain at least one meter interval.")

    # For origin row i, use rows i+1 through the last row at or before t_i+h.
    future_ends = np.searchsorted(timestamp_ns, timestamp_ns + horizon.value, side="right")
    starts = np.arange(n, dtype=np.int64) + 1
    counts = np.maximum(future_ends - starts, 0)

    safe_energy = np.where(np.isfinite(energy), energy, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(safe_energy, dtype=np.float64)))
    sums = prefix[future_ends] - prefix[starts]
    coverage = counts.astype(np.float64) / float(expected)

    valid = (counts >= math.ceil(expected * minimum_coverage)) & (counts > 0)
    # Prevent a dense cluster of early readings from being mistaken for a complete
    # window when the tail of the future period is actually absent (or vice versa).
    tol_ns = int(boundary_tolerance_intervals * interval_minutes * 60 * 1e9)
    first_index = np.minimum(starts, n - 1)
    last_index = np.maximum(future_ends - 1, 0)
    has_future = counts > 0
    first_gap = timestamp_ns[first_index] - timestamp_ns
    last_gap = timestamp_ns + horizon.value - timestamp_ns[last_index]
    valid &= has_future & (first_gap <= tol_ns) & (last_gap <= tol_ns)

    sums[~valid] = np.nan
    return sums, coverage, int(valid.sum())


def add_past_consumption_features(
    frame: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    energy: np.ndarray,
) -> pd.DataFrame:
    """Add leakage-safe trailing energy summaries for each household.

    Windows are based on elapsed time rather than row counts, so missing or
    irregular 3-minute observations do not silently shift the feature horizon.
    ``closed='left'`` excludes the current observation; only information
    available strictly before the forecast origin is used.
    """
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Past-window features require unique, increasing timestamps per meter.")

    energy_series = pd.Series(energy, index=timestamps, dtype="float64")
    windows = {
        "24h": "24h",
        "7d": "7D",
        "30d": "30D",
    }
    for label, window in windows.items():
        rolling = energy_series.rolling(window=window, closed="left", min_periods=1)
        frame[f"past_{label}_sum_kWh"] = rolling.sum().to_numpy(dtype=np.float64)
        frame[f"past_{label}_mean_interval_kWh"] = rolling.mean().to_numpy(dtype=np.float64)
        frame[f"past_{label}_count"] = rolling.count().fillna(0).to_numpy(dtype=np.float32)
    return frame


def process_meter_group(
    meter_frame: pd.DataFrame,
    timestamp_col: str,
    split_col: str,
    energy_col: str,
    horizons: list[str],
    interval_minutes: int,
    minimum_coverage: float,
    boundary_tolerance_intervals: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Calculate targets independently for each meter and split."""
    out = meter_frame.sort_values("__timestamp_utc", kind="mergesort").copy()
    if out["__timestamp_utc"].duplicated().any():
        examples = out.loc[out["__timestamp_utc"].duplicated(keep=False), "__timestamp_utc"].head(5).tolist()
        raise ValueError(f"Duplicate timestamps found for meter {out['meter'].iloc[0]}: {examples}")

    energy = pd.to_numeric(out[energy_col], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(energy).any() or not np.isfinite(energy).all():
        raise ValueError(f"Missing or non-finite {energy_col} found for meter {out['meter'].iloc[0]}.")
    if (energy < 0).any():
        raise ValueError(f"Negative {energy_col} found for meter {out['meter'].iloc[0]}.")

    # Trailing consumption profiles strengthen the model beyond very short
    # lags. They use the full past history for a meter, including observations
    # before the train/test split, which would genuinely be known at forecast
    # time. They never include current or future energy.
    out = add_past_consumption_features(
        out,
        pd.DatetimeIndex(out["__timestamp_utc"]),
        energy,
    )

    for horizon_key in horizons:
        config = HORIZON_CONFIG[horizon_key]
        out[config["target"]] = np.nan
        out[config["coverage"]] = np.nan

    valid_counts = {h: 0 for h in horizons}
    for split_value, indexes in out.groupby(split_col, sort=False).groups.items():
        if pd.isna(split_value):
            continue
        block = out.loc[indexes]
        # Pandas can preserve parsed datetimes at second or microsecond resolution
        # (rather than nanoseconds).  horizon.value is always nanoseconds, so
        # normalize explicitly before using NumPy integer arithmetic/searchsorted.
        # Without this conversion, adding a nanosecond horizon to second-resolution
        # epoch values makes every search appear to extend beyond the group.
        ts_ns = pd.DatetimeIndex(block["__timestamp_utc"]).as_unit("ns").asi8
        if len(ts_ns) > 1 and np.any(np.diff(ts_ns) <= 0):
            raise ValueError(
                f"Timestamps for meter {block['meter'].iloc[0]} and split "
                f"{split_value!r} must be strictly increasing before target generation."
            )
        block_energy = pd.to_numeric(block[energy_col], errors="coerce").to_numpy(dtype=np.float64)
        for horizon_key in horizons:
            cfg = HORIZON_CONFIG[horizon_key]
            target, coverage, valid_count = _compute_forward_target(
                block,
                ts_ns,
                block_energy,
                cfg["duration"],
                interval_minutes,
                minimum_coverage,
                boundary_tolerance_intervals,
            )
            out.loc[indexes, cfg["target"]] = target
            out.loc[indexes, cfg["coverage"]] = coverage
            valid_counts[horizon_key] += valid_count

    out = out.drop(columns=["__timestamp_utc"])
    return out, valid_counts


def run(args: argparse.Namespace) -> int:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    profile_path = resolve_path(args.profile)
    if not input_path.is_file():
        raise FileNotFoundError(f"Processed ETL dataset not found: {input_path}")
    if args.interval_minutes <= 0 or args.chunksize <= 0:
        raise ValueError("--interval-minutes and --chunksize must be positive.")
    if not 0 < args.minimum_coverage <= 1:
        raise ValueError("--minimum-coverage must be in (0, 1].")
    if args.boundary_tolerance_intervals < 0:
        raise ValueError("--boundary-tolerance-intervals cannot be negative.")

    header = pd.read_csv(input_path, nrows=0)
    columns = list(header.columns)
    meter_col = next((c for c in columns if c.lower() in {"meter", "meter_id", "household_id"}), None)
    timestamp_col = next((c for c in columns if c.lower() == "timestamp_utc"), None)
    if timestamp_col is None:
        timestamp_col = next((c for c in columns if c.lower() in {"timestamp", "timestamp_local", "datetime"}), None)
    energy_col = next((c for c in columns if c.lower() in {"t_kwh", "energy_kwh", "consumption_kwh"}), None)
    split_col = next((c for c in columns if c.lower() == "split"), None)
    if not all([meter_col, timestamp_col, energy_col, split_col]):
        raise ValueError(
            "Input must contain meter, timestamp/timestamp_utc, t_kWh and split columns. "
            f"Columns found: {columns}"
        )

    optional_keep = [
        "timestamp", "timestamp_local", "timestamp_utc", "location_id", "weather_match",
        "weather_time_offset_minutes", *FEATURE_COLUMNS,
    ]
    usecols = list(dict.fromkeys([meter_col, timestamp_col, energy_col, split_col] + [c for c in optional_keep if c in columns]))
    # Keep the raw weather-match flag temporarily to create a numeric availability feature.
    usecols = list(dict.fromkeys(usecols + [c for c in columns if c.lower() == "weather_match"]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    generated_in_this_script = {
        "weather_available",
        "lag_1_reading_kWh", "lag_3_reading_kWh", "lag_10_reading_kWh",
        "rolling_1h_mean_kWh", "rolling_1h_sum_kWh", "rolling_1h_std_kWh",
        "rolling_1h_count", "minutes_since_previous_reading",
        "past_24h_sum_kWh", "past_24h_mean_interval_kWh", "past_24h_count",
        "past_7d_sum_kWh", "past_7d_mean_interval_kWh", "past_7d_count",
        "past_30d_sum_kWh", "past_30d_mean_interval_kWh", "past_30d_count",
    }
    missing_input_features = [
        col for col in NUMERIC_FEATURES
        if col not in columns and col not in generated_in_this_script
    ]
    if missing_input_features:
        logging.warning(
            "These optional model input columns are absent and will be written as missing values: %s",
            missing_input_features,
        )
    stats: dict[str, Any] = {
        "input_file": str(input_path),
        "input_feature_columns_missing": missing_input_features,
        "output_file": str(output_path),
        "rows_written": 0,
        "meters_processed": 0,
        "split_isolation": True,
        "interval_minutes_assumed": args.interval_minutes,
        "minimum_future_window_coverage": args.minimum_coverage,
        "horizons": {},
        "features": FEATURE_COLUMNS,
        "target_definitions": {},
        "notes": [
            "t_kWh is interpreted as interval energy based on the user's diagnostic comparison with V*I*3-minute energy.",
            "Future targets sum rows in (t, t+horizon], separately by meter and by train/test split.",
            "Rows that do not meet future-window coverage and boundary checks receive a missing target.",
            "Internal integer timestamps are explicitly normalized to nanoseconds before horizon arithmetic.",
            "The train/test boundary is not crossed when building target labels.",
            "Past 24-hour, 7-day, and 30-day consumption summaries are generated from trailing windows; current and future readings are excluded.",
        ],
    }
    for h in args.horizons:
        cfg = HORIZON_CONFIG[h]
        stats["horizons"][h] = {"target": cfg["target"], "valid_rows": 0, "invalid_rows": 0}
        stats["target_definitions"][cfg["target"]] = cfg["label"]

    pending: pd.DataFrame | None = None
    last_completed_meter: str | None = None
    first_write = True

    def emit_meter_group(group: pd.DataFrame) -> None:
        nonlocal first_write, last_completed_meter
        meter_value = str(group["meter"].iloc[0])
        if last_completed_meter is not None and meter_value <= last_completed_meter:
            raise ValueError(
                "Input must be sorted globally by meter and timestamp, as produced by the ETL pipeline. "
                f"Meter ordering returned to {meter_value} after {last_completed_meter}."
            )
        processed, valid = process_meter_group(
            group,
            timestamp_col,
            split_col,
            energy_col,
            args.horizons,
            args.interval_minutes,
            args.minimum_coverage,
            args.boundary_tolerance_intervals,
        )
        for h in args.horizons:
            target_col = HORIZON_CONFIG[h]["target"]
            count = valid[h]
            stats["horizons"][h]["valid_rows"] += count
            stats["horizons"][h]["invalid_rows"] += int(processed[target_col].isna().sum())
        # Canonical model identifier is always named meter.
        if meter_col != "meter":
            processed = processed.rename(columns={meter_col: "meter"})
        if timestamp_col != "timestamp_utc":
            processed["timestamp_utc"] = pd.to_datetime(processed[timestamp_col], utc=True, errors="coerce", format="mixed")
        if "timestamp" not in processed.columns:
            processed["timestamp"] = processed["timestamp_utc"].dt.tz_convert("Asia/Kolkata").astype(str)
        if "timestamp_local" not in processed.columns:
            processed["timestamp_local"] = processed["timestamp"].astype(str)
        if "weather_match" in processed.columns:
            raw_match = processed["weather_match"].astype("string").str.strip().str.lower()
            processed["weather_available"] = raw_match.isin(["true", "1", "yes", "y"]).astype("int8")
            processed = processed.drop(columns=["weather_match"])
        elif "weather_available" not in processed.columns:
            processed["weather_available"] = 0
        for col in NUMERIC_FEATURES:
            if col not in processed.columns:
                processed[col] = np.nan
            processed[col] = pd.to_numeric(processed[col], errors="coerce")
        # Keep only model inputs, row identity, split, target labels and their coverage.
        keep = ["meter", "timestamp", "timestamp_utc", "timestamp_local", "split"] + FEATURE_COLUMNS
        for h in args.horizons:
            cfg = HORIZON_CONFIG[h]
            keep.extend([cfg["target"], cfg["coverage"]])
        keep = list(dict.fromkeys(c for c in keep if c in processed.columns))
        processed = processed.loc[:, keep]
        processed.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            encoding="utf-8-sig" if first_write else "utf-8",
        )
        first_write = False
        stats["rows_written"] += len(processed)
        stats["meters_processed"] += 1
        last_completed_meter = meter_value
        logging.info("Created future targets for %s (%s rows).", meter_value, f"{len(processed):,}")

    for chunk in pd.read_csv(input_path, usecols=usecols, dtype={meter_col: "string"}, chunksize=args.chunksize, low_memory=False):
        chunk[meter_col] = chunk[meter_col].astype("string").str.strip()
        chunk["__timestamp_utc"] = pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=True, format="mixed")
        chunk[split_col] = chunk[split_col].astype("string").str.lower().str.strip()
        bad = chunk[meter_col].isna() | chunk[meter_col].eq("") | chunk["__timestamp_utc"].isna() | chunk[split_col].isna()
        if bad.any():
            raise ValueError(f"Found {int(bad.sum())} rows missing meter, timestamp or split. Fix upstream ETL first.")
        chunk = chunk.sort_values([meter_col, "__timestamp_utc"], kind="mergesort")
        if pending is not None and not pending.empty:
            chunk = pd.concat([pending, chunk], ignore_index=True)
        last_meter = str(chunk[meter_col].iloc[-1])
        completed = chunk.loc[chunk[meter_col].astype(str) != last_meter]
        pending = chunk.loc[chunk[meter_col].astype(str) == last_meter].copy()
        for meter_value, meter_group in completed.groupby(meter_col, sort=False):
            meter_group = meter_group.drop(columns=["__timestamp_utc"] if "__timestamp_utc" in meter_group.columns else [])
            meter_group["__timestamp_utc"] = pd.to_datetime(meter_group[timestamp_col], utc=True, errors="coerce", format="mixed")
            emit_meter_group(meter_group)

    if pending is not None and not pending.empty:
        pending = pending.sort_values([meter_col, "__timestamp_utc"], kind="mergesort")
        emit_meter_group(pending)

    if first_write:
        raise ValueError("No rows were written. Check the processed input dataset.")
    for h in args.horizons:
        target_col = HORIZON_CONFIG[h]["target"]
        stats["horizons"][h]["valid_percent"] = round(
            100.0 * stats["horizons"][h]["valid_rows"] / max(stats["rows_written"], 1), 3
        )
        stats["horizons"][h]["invalid_rows"] = stats["rows_written"] - stats["horizons"][h]["valid_rows"]
    profile_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    logging.info("Feature dataset saved: %s (%s rows).", output_path, f"{stats['rows_written']:,}")
    for h in args.horizons:
        logging.info("%s valid targets: %s rows.", h, f"{stats['horizons'][h]['valid_rows']:,}")
    logging.info("Feature profile saved: %s", profile_path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    try:
        return run(args)
    except Exception as exc:
        logging.exception("Feature engineering failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
