#!/usr/bin/env python3
"""Build a machine-learning-ready household energy dataset from aligned data.

Default input:
    data/interim/aligned/household_electricity_weather.csv

Default outputs:
    data/processed/household_energy_dataset.csv
    data/processed/household_energy_train.csv
    data/processed/household_energy_test.csv
    data/processed/household_energy_dataset_profile.json

The aligned data is expected to be sorted by meter and timestamp, as produced
by align_weather_meter.py. Data is processed in chunks and historical features
are calculated using only earlier readings from the same meter (no future-data
leakage in the lag/rolling features).

The chronological split is based on timestamp: the earliest (1 - test_fraction)
portion of the full time span is training data, and the latest portion is test
data. The default test fraction is 20%.

`t_kWh` is retained as the observed interval-energy column / target candidate.
This script does not invent a next-24-hour or next-month target; those targets
must be defined explicitly once the meter reading interval and forecast target
aggregation are confirmed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import deque
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/aligned/household_electricity_weather.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed"
DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_CHUNK_SIZE = 100_000
LAG_STEPS = (1, 3, 10)
ROLLING_WINDOW = pd.Timedelta(hours=1)


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create time features, past-consumption features and chronological train/test datasets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.20,
        help="Fraction of the chronological time span reserved for testing. Default: 0.20.",
    )
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser


def find_time_column(columns: list[str]) -> str | None:
    lookup = {"".join(c.lower().split()): c for c in columns}
    for candidate in ("timestamp_utc", "timestamp", "timestamp_local", "datetime"):
        key = "".join(candidate.split())
        if key in lookup:
            return lookup[key]
    return None


def scan_time_range(input_path: Path, timestamp_col: str, chunksize: int) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    minimum: pd.Timestamp | None = None
    maximum: pd.Timestamp | None = None
    rows = 0
    invalid = 0
    for chunk in pd.read_csv(input_path, usecols=[timestamp_col], chunksize=chunksize, low_memory=False):
        parsed = pd.to_datetime(chunk[timestamp_col], errors="coerce", utc=True, format="mixed")
        rows += len(chunk)
        invalid += int(parsed.isna().sum())
        if parsed.notna().any():
            chunk_min = parsed.min()
            chunk_max = parsed.max()
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
    if minimum is None or maximum is None:
        raise ValueError("No valid timestamps found in the aligned dataset.")
    if maximum <= minimum:
        raise ValueError(
            "The dataset must span more than one timestamp to create a chronological train/test split."
        )
    return minimum, maximum, rows, invalid


def write_chunk(frame: pd.DataFrame, path: Path, first_write: bool) -> None:
    frame.to_csv(
        path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
        encoding="utf-8-sig" if first_write else "utf-8",
    )


def run(args: argparse.Namespace) -> int:
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Aligned electricity-weather dataset not found: {input_path}")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be greater than zero.")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1 (exclusive).")

    header = pd.read_csv(input_path, nrows=0)
    columns = list(header.columns)
    meter_col = next((c for c in columns if c.lower() in {"meter", "meter_id", "household_id"}), None)
    time_col = find_time_column(columns)
    energy_col = next((c for c in columns if c.lower() in {"t_kwh", "energy_kwh", "consumption_kwh"}), None)
    if not meter_col or not time_col or not energy_col:
        raise ValueError(
            "Aligned input must contain meter, timestamp_utc (or timestamp), and t_kWh columns. "
            f"Columns found: {columns}"
        )

    logging.info("Scanning timestamps to determine chronological split...")
    min_time, max_time, input_rows, invalid_time_rows = scan_time_range(
        input_path, time_col, args.chunksize
    )
    split_cutoff = min_time + (max_time - min_time) * (1.0 - args.test_fraction)
    logging.info("Time range: %s to %s", min_time, max_time)
    logging.info("Training rows are at or before: %s", split_cutoff)

    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "household_energy_dataset.csv"
    train_path = output_dir / "household_energy_train.csv"
    test_path = output_dir / "household_energy_test.csv"
    profile_path = output_dir / "household_energy_dataset_profile.json"
    for path in (full_path, train_path, test_path):
        if path.exists():
            path.unlink()

    counters: dict[str, Any] = {
        "input_rows": input_rows,
        "invalid_timestamp_rows_dropped": 0,
        "output_rows": 0,
        "training_rows": 0,
        "testing_rows": 0,
        "chunks_processed": 0,
        "missing_values": {},
        "weather_unmatched_rows": 0,
        "first_meter": None,
        "last_meter": None,
    }
    first_full = first_train = first_test = True

    # Only the current meter's history is needed because align_weather_meter.py
    # writes records in meter-then-time order. This keeps memory use low.
    current_meter: str | None = None
    last_timestamp: pd.Timestamp | None = None
    last_meter_seen: str | None = None
    lag_history: deque[float] = deque(maxlen=max(LAG_STEPS))
    rolling_history: deque[tuple[pd.Timestamp, float]] = deque()
    previous_timestamp: pd.Timestamp | None = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, dtype={meter_col: "string"}, chunksize=args.chunksize, low_memory=False),
        start=1,
    ):
        counters["chunks_processed"] += 1
        chunk[meter_col] = chunk[meter_col].astype("string").str.strip()
        timestamps = pd.to_datetime(chunk[time_col], errors="coerce", utc=True, format="mixed")
        invalid = timestamps.isna() | chunk[meter_col].isna() | chunk[meter_col].eq("")
        counters["invalid_timestamp_rows_dropped"] += int(invalid.sum())
        chunk = chunk.loc[~invalid].copy()
        timestamps = timestamps.loc[~invalid]
        if chunk.empty:
            continue

        # Preserve the required streaming order; sort within this chunk and
        # verify that consecutive chunks continue in the same global order.
        chunk["__parsed_time"] = timestamps
        chunk = chunk.sort_values([meter_col, "__parsed_time"], kind="mergesort")
        timestamps = chunk["__parsed_time"]

        feature_values: dict[str, list[Any]] = {
            f"lag_{lag}_reading_kWh": [] for lag in LAG_STEPS
        }
        feature_values.update({
            "rolling_1h_mean_kWh": [],
            "rolling_1h_sum_kWh": [],
            "rolling_1h_std_kWh": [],
            "rolling_1h_count": [],
            "minutes_since_previous_reading": [],
        })

        for meter_value, timestamp_value, energy_value in zip(
            chunk[meter_col].astype(str),
            timestamps,
            pd.to_numeric(chunk[energy_col], errors="coerce"),
        ):
            timestamp_value = pd.Timestamp(timestamp_value)
            energy = float(energy_value) if pd.notna(energy_value) else math.nan
            if not math.isfinite(energy) or energy < 0:
                raise ValueError(
                    f"Invalid electricity target found for meter {meter_value} at {timestamp_value}. "
                    "Re-run clean_meter_data.py before building features."
                )

            if current_meter != meter_value:
                if last_meter_seen is not None and meter_value < last_meter_seen:
                    raise ValueError(
                        "Input is not globally sorted by meter. Re-run align_weather_meter.py "
                        "to create the expected meter/timestamp ordering."
                    )
                current_meter = meter_value
                last_meter_seen = meter_value
                lag_history.clear()
                rolling_history.clear()
                previous_timestamp = None
                last_timestamp = None

            if last_timestamp is not None and timestamp_value <= last_timestamp:
                raise ValueError(
                    f"Timestamps for meter {meter_value} are not strictly increasing. "
                    "Check duplicate readings and input ordering."
                )

            # Remove observations strictly older than one hour. Only previous
            # readings are considered; the current target is appended afterwards.
            window_start = timestamp_value - ROLLING_WINDOW
            while rolling_history and rolling_history[0][0] < window_start:
                rolling_history.popleft()

            for lag in LAG_STEPS:
                feature_values[f"lag_{lag}_reading_kWh"].append(
                    lag_history[-lag] if len(lag_history) >= lag else np.nan
                )

            prior_values = [value for _, value in rolling_history]
            feature_values["rolling_1h_mean_kWh"].append(
                float(np.mean(prior_values)) if prior_values else np.nan
            )
            feature_values["rolling_1h_sum_kWh"].append(
                float(np.sum(prior_values)) if prior_values else np.nan
            )
            feature_values["rolling_1h_std_kWh"].append(
                float(np.std(prior_values, ddof=0)) if len(prior_values) >= 2 else np.nan
            )
            feature_values["rolling_1h_count"].append(len(prior_values))
            feature_values["minutes_since_previous_reading"].append(
                (timestamp_value - previous_timestamp).total_seconds() / 60.0
                if previous_timestamp is not None else np.nan
            )

            lag_history.append(energy)
            rolling_history.append((timestamp_value, energy))
            previous_timestamp = timestamp_value
            last_timestamp = timestamp_value

        for name, values in feature_values.items():
            chunk[name] = values

        chunk["timestamp_utc"] = chunk["__parsed_time"]
        chunk["timestamp_local"] = chunk["__parsed_time"].dt.tz_convert(args.timezone)
        local_time = chunk["timestamp_local"]
        chunk["hour"] = local_time.dt.hour.astype("int8")
        chunk["day"] = local_time.dt.day.astype("int8")
        chunk["day_of_week"] = local_time.dt.dayofweek.astype("int8")
        chunk["month"] = local_time.dt.month.astype("int8")
        chunk["day_of_year"] = local_time.dt.dayofyear.astype("int16")
        chunk["is_weekend"] = local_time.dt.dayofweek.isin([5, 6]).astype("int8")
        chunk["hour_sin"] = np.sin(2.0 * np.pi * chunk["hour"] / 24.0)
        chunk["hour_cos"] = np.cos(2.0 * np.pi * chunk["hour"] / 24.0)
        chunk["day_of_year_sin"] = np.sin(2.0 * np.pi * chunk["day_of_year"] / 366.0)
        chunk["day_of_year_cos"] = np.cos(2.0 * np.pi * chunk["day_of_year"] / 366.0)
        chunk["split"] = np.where(chunk["__parsed_time"] <= split_cutoff, "train", "test")
        chunk = chunk.drop(columns=["__parsed_time"])

        # Remove temporary columns that are not useful model inputs.
        if "weather_timestamp_utc" not in chunk.columns and "weather_timestamp" in chunk.columns:
            pass

        weather_match_col = next((c for c in chunk.columns if c.lower() == "weather_match"), None)
        if weather_match_col:
            matched = chunk[weather_match_col].astype("string").str.lower().isin(["true", "1", "yes"])
            counters["weather_unmatched_rows"] += int((~matched).sum())

        counters["output_rows"] += len(chunk)
        counters["training_rows"] += int((chunk["split"] == "train").sum())
        counters["testing_rows"] += int((chunk["split"] == "test").sum())
        counters["first_meter"] = counters["first_meter"] or str(chunk[meter_col].iloc[0])
        counters["last_meter"] = str(chunk[meter_col].iloc[-1])
        for column in chunk.columns:
            missing_count = int(chunk[column].isna().sum())
            counters["missing_values"][column] = counters["missing_values"].get(column, 0) + missing_count

        write_chunk(chunk, full_path, first_full)
        first_full = False
        train_chunk = chunk.loc[chunk["split"] == "train"]
        test_chunk = chunk.loc[chunk["split"] == "test"]
        if not train_chunk.empty:
            write_chunk(train_chunk, train_path, first_train)
            first_train = False
        if not test_chunk.empty:
            write_chunk(test_chunk, test_path, first_test)
            first_test = False

        logging.info(
            "Built features for chunk %d: %s rows (train=%s, test=%s)",
            chunk_number,
            f"{len(chunk):,}",
            f"{len(train_chunk):,}",
            f"{len(test_chunk):,}",
        )

    if first_full:
        raise ValueError("No rows were written. Check the aligned input data.")
    if first_train or first_test:
        raise ValueError(
            "The chronological split produced an empty train or test dataset. "
            "Check the date range or adjust --test-fraction."
        )

    profile = {
        "input_file": str(input_path),
        "outputs": {
            "full_dataset": str(full_path),
            "training_dataset": str(train_path),
            "testing_dataset": str(test_path),
        },
        "timezone_for_calendar_features": args.timezone,
        "rows": {
            "input_rows": input_rows,
            "invalid_timestamp_rows_dropped": counters["invalid_timestamp_rows_dropped"],
            "output_rows": counters["output_rows"],
            "training_rows": counters["training_rows"],
            "testing_rows": counters["testing_rows"],
            "weather_unmatched_rows": counters["weather_unmatched_rows"],
            "chunks_processed": counters["chunks_processed"],
        },
        "chronological_split": {
            "minimum_timestamp_utc": min_time.isoformat(),
            "maximum_timestamp_utc": max_time.isoformat(),
            "training_cutoff_utc_inclusive": split_cutoff.isoformat(),
            "test_fraction_of_time_span": args.test_fraction,
            "rule": "train if timestamp_utc <= cutoff; test if timestamp_utc > cutoff",
        },
        "features_created": [
            "hour", "day", "day_of_week", "month", "day_of_year", "is_weekend",
            "hour_sin", "hour_cos", "day_of_year_sin", "day_of_year_cos",
            "lag_1_reading_kWh", "lag_3_reading_kWh", "lag_10_reading_kWh",
            "rolling_1h_mean_kWh", "rolling_1h_sum_kWh", "rolling_1h_std_kWh",
            "rolling_1h_count", "minutes_since_previous_reading",
        ],
        "missing_values_by_column": counters["missing_values"],
        "target_note": (
            "t_kWh is retained as the observed electricity reading. The future forecast horizon "
            "(for example, next 24 hours or next month) is not generated here because its "
            "aggregation definition must be explicitly selected."
        ),
        "history_feature_rule": "Only readings strictly earlier than the current row are used.",
    }
    profile_path.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")

    logging.info("Final dataset: %s (%s rows)", full_path, f"{counters['output_rows']:,}")
    logging.info("Training dataset: %s (%s rows)", train_path, f"{counters['training_rows']:,}")
    logging.info("Testing dataset: %s (%s rows)", test_path, f"{counters['testing_rows']:,}")
    logging.info("Dataset profile: %s", profile_path)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = make_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        logging.exception("Dataset building failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
