#!/usr/bin/env python3
"""Clean CEEW electricity readings and Bareilly weather data.

Run from the Home-Solar project root:
    python etl/clean_meter_data.py

By default the script looks for:
    Electricity: data/interim/electricity/selected_households.csv
                 then data/raw/electricity/CEEW - Smart meter data Bareilly 2020.csv
    Weather:    data/interim/weather/bareilly_weather_hourly.csv
                 then a single CSV in data/raw/weather/

It never edits the input files. Cleaned outputs and JSON quality profiles are
written under data/interim/.

Electricity assumptions:
- `meter` identifies a household/meter.
- `x_Timestamp` is the reading time.
- `t_kWh` is an energy reading in kWh, based on the source column name.
- Negative or non-numeric energy readings are removed. Zero readings are kept.
- Missing optional voltage/current/frequency values remain blank (NaN); they are
  not filled with zero because zero would imply a real measurement.
- Duplicate meter + timestamp records keep the last occurrence in the input.

Weather assumptions:
- Naive timestamps are interpreted in Asia/Kolkata by default.
- Weather rows with invalid timestamps are removed; duplicate timestamp/location
  records keep the last occurrence.
- Missing weather features remain blank. Clearly invalid humidity/cloud/solar
  values are changed to missing rather than guessed or imputed.

Only pandas and NumPy are required beyond Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = "Asia/Kolkata"

ELECTRICITY_CANDIDATES = [
    Path("data/interim/electricity/selected_households.csv"),
    Path("data/raw/electricity/CEEW - Smart meter data Bareilly 2020.csv"),
]
WEATHER_CANDIDATES = [
    Path("data/interim/weather/bareilly_weather_hourly.csv"),
    Path("data/raw/weather/bareilly_weather_2020.csv"),
]

ELECTRICITY_FIELDS = {
    "meter": ["meter", "meter_id", "household_id", "consumer_id", "house_id"],
    "timestamp": ["x_Timestamp", "timestamp", "datetime", "date_time", "time", "reading_time"],
    "energy": ["t_kWh", "energy_kwh", "consumption_kwh", "kwh", "electricity_kwh"],
    "voltage": ["z_Avg Voltage (Volt)", "avg_voltage", "average_voltage", "voltage", "voltage_v"],
    "current": ["z_Avg Current (Amp)", "avg_current", "average_current", "current", "current_a"],
    "frequency": ["y_Freq (Hz)", "frequency", "freq", "frequency_hz"],
}

WEATHER_FIELDS = {
    "temperature_2m": ["temperature_2m", "2m_temperature", "temperature", "temp", "air_temperature"],
    "relative_humidity_2m": ["relative_humidity_2m", "relative_humidity", "humidity", "rh", "relative_humidity_percent"],
    "dew_point_2m": ["dew_point_2m", "dewpoint_2m", "dew_point", "dewpoint"],
    "cloud_cover": ["cloud_cover", "cloudcover", "cloud_cover_percent", "total_cloud_cover"],
    "shortwave_radiation": ["shortwave_radiation", "surface_solar_radiation_downwards", "ghi", "global_horizontal_irradiance", "solar_radiation"],
    "direct_radiation": ["direct_radiation", "direct_solar_radiation"],
    "diffuse_radiation": ["diffuse_radiation", "diffuse_solar_radiation", "dhi"],
    "direct_normal_irradiance": ["direct_normal_irradiance", "dni"],
    "global_tilted_irradiance": ["global_tilted_irradiance", "gti"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long"],
    "location_id": ["location_id", "location", "site_id", "grid_id"],
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def norm_name(value: object) -> str:
    """Normalize a column name for flexible matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def find_column(columns: Iterable[object], aliases: list[str]) -> str | None:
    lookup = {norm_name(col): str(col) for col in columns}
    for alias in aliases:
        match = lookup.get(norm_name(alias))
        if match is not None:
            return match
    return None


def discover_input(
    explicit: str | None,
    candidates: list[Path],
    fallback_dir: Path,
    description: str,
) -> Path | None:
    if explicit:
        path = resolve_project_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{description} input file does not exist: {path}")
        return path

    for candidate in candidates:
        path = resolve_project_path(candidate)
        if path.is_file():
            return path

    search_dir = resolve_project_path(fallback_dir)
    csv_files = sorted(search_dir.glob("*.csv")) if search_dir.is_dir() else []
    if len(csv_files) == 1:
        return csv_files[0]
    if len(csv_files) > 1:
        names = "\n  ".join(str(item.relative_to(PROJECT_ROOT)) for item in csv_files)
        raise FileNotFoundError(
            f"More than one {description} CSV was found in {search_dir}. "
            f"Pass --{description.lower()}-input to choose one:\n  {names}"
        )
    return None


def parse_timestamp_series(values: pd.Series, source_timezone: str) -> pd.Series:
    """Parse mixed timestamps and return timezone-aware local timestamps."""
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")

    # Fast path for the typical case: all values are timezone-naive.
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(source_timezone)
    if pd.api.types.is_datetime64_dtype(parsed.dtype):
        return parsed.dt.tz_localize(
            source_timezone, ambiguous="NaT", nonexistent="NaT"
        )

    # Fallback for mixed naive/offset-aware values.
    def convert_one(value: object):
        if pd.isna(value):
            return pd.NaT
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize(
                    source_timezone, ambiguous="NaT", nonexistent="NaT"
                )
            else:
                stamp = stamp.tz_convert(source_timezone)
            return stamp
        except (TypeError, ValueError, OverflowError):
            return pd.NaT

    return pd.Series(
        pd.array(parsed.map(convert_one), dtype=f"datetime64[ns, {source_timezone}]"),
        index=values.index,
    )


def timestamps_to_iso(values: pd.Series) -> pd.Series:
    """Format timezone-aware timestamps as ISO-8601 strings with UTC offset."""
    return (
        values.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        .str.replace(r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", regex=True)
    )


def numeric_values(series: pd.Series) -> pd.Series:
    """Convert common numeric strings to numbers; unparsable values become NaN."""
    cleaned = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def get_electricity_columns(columns: Iterable[object]) -> dict[str, str | None]:
    return {
        key: find_column(columns, aliases)
        for key, aliases in ELECTRICITY_FIELDS.items()
    }


def clean_electricity(
    input_path: Path,
    output_dir: Path,
    source_timezone: str,
    chunksize: int,
) -> dict[str, object]:
    """Clean a large electricity CSV using chunked input and SQLite deduplication."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "clean_electricity.csv"
    profile_path = output_dir / "electricity_cleaning_profile.json"
    staging_path = output_dir / ".electricity_cleaning_staging.sqlite"

    logging.info("Cleaning electricity data: %s", input_path)
    counters = {
        "input_rows": 0,
        "missing_meter_or_timestamp_rows": 0,
        "missing_or_non_numeric_energy_rows": 0,
        "negative_energy_rows": 0,
        "valid_rows_before_deduplication": 0,
        "optional_voltage_missing": 0,
        "optional_current_missing": 0,
        "optional_frequency_missing": 0,
        "chunks_processed": 0,
    }
    min_energy: float | None = None
    max_energy: float | None = None

    # Remove a stale staging file from an interrupted prior run.
    if staging_path.exists():
        staging_path.unlink()

    connection = sqlite3.connect(staging_path)
    try:
        connection.execute(
            """CREATE TABLE readings (
                meter TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                t_kWh REAL NOT NULL,
                avg_voltage_v REAL,
                avg_current_a REAL,
                frequency_hz REAL,
                PRIMARY KEY (meter, timestamp)
            )"""
        )

        # Read the header first so dtype=str can preserve meter IDs such as BR02/0012.
        header = pd.read_csv(input_path, nrows=0)
        columns = get_electricity_columns(header.columns)
        required = ["meter", "timestamp", "energy"]
        missing_required = [name for name in required if not columns.get(name)]
        if missing_required:
            raise ValueError(
                "Could not identify required electricity columns: "
                + ", ".join(missing_required)
                + f". Available columns: {list(header.columns)}"
            )

        logging.info(
            "Electricity columns detected: meter=%s, timestamp=%s, energy=%s",
            columns["meter"], columns["timestamp"], columns["energy"],
        )
        logging.info(
            "Optional columns: voltage=%s, current=%s, frequency=%s",
            columns["voltage"], columns["current"], columns["frequency"],
        )

        chunk_iter = pd.read_csv(
            input_path,
            dtype="string",
            chunksize=chunksize,
            low_memory=False,
        )

        for chunk_number, chunk in enumerate(chunk_iter, start=1):
            counters["chunks_processed"] += 1
            counters["input_rows"] += len(chunk)

            meter = chunk[columns["meter"]].astype("string").str.strip()
            meter = meter.mask(meter.eq(""))
            timestamps = parse_timestamp_series(chunk[columns["timestamp"]], source_timezone)
            energy = numeric_values(chunk[columns["energy"]])

            missing_key = meter.isna() | timestamps.isna()
            counters["missing_meter_or_timestamp_rows"] += int(missing_key.sum())

            missing_energy = energy.isna()
            counters["missing_or_non_numeric_energy_rows"] += int((~missing_key & missing_energy).sum())
            negative_energy = energy.lt(0).fillna(False)
            counters["negative_energy_rows"] += int((~missing_key & negative_energy).sum())

            valid = (~missing_key) & (~missing_energy) & (~negative_energy)
            if not valid.any():
                logging.info("Electricity chunk %d: no valid rows", chunk_number)
                continue

            valid_meter = meter.loc[valid].astype(str)
            valid_timestamp = timestamps.loc[valid]
            valid_energy = energy.loc[valid]
            local_frame = pd.DataFrame(
                {
                    "meter": valid_meter.to_numpy(),
                    "timestamp": timestamps_to_iso(valid_timestamp).to_numpy(),
                    "t_kWh": valid_energy.to_numpy(dtype=float),
                }
            )

            for source_name, target_name, counter_name in [
                ("voltage", "avg_voltage_v", "optional_voltage_missing"),
                ("current", "avg_current_a", "optional_current_missing"),
                ("frequency", "frequency_hz", "optional_frequency_missing"),
            ]:
                source_col = columns[source_name]
                if source_col:
                    values = numeric_values(chunk.loc[valid, source_col])
                    local_frame[target_name] = values.to_numpy(dtype=float, na_value=np.nan)
                    counters[counter_name] += int(values.isna().sum())
                else:
                    local_frame[target_name] = np.nan

            counters["valid_rows_before_deduplication"] += len(local_frame)
            chunk_min = float(valid_energy.min())
            chunk_max = float(valid_energy.max())
            min_energy = chunk_min if min_energy is None else min(min_energy, chunk_min)
            max_energy = chunk_max if max_energy is None else max(max_energy, chunk_max)

            # Remove duplicates within this chunk; INSERT OR REPLACE also handles
            # duplicate meter/timestamp keys that occur in different chunks.
            local_frame = local_frame.drop_duplicates(
                subset=["meter", "timestamp"], keep="last"
            )
            records = []
            for row in local_frame.itertuples(index=False, name=None):
                values = tuple(None if pd.isna(value) else value for value in row)
                records.append(values)
            connection.executemany(
                "INSERT OR REPLACE INTO readings "
                "(meter, timestamp, t_kWh, avg_voltage_v, avg_current_a, frequency_hz) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                records,
            )
            connection.commit()
            if chunk_number == 1 or chunk_number % 5 == 0:
                logging.info("Processed electricity chunk %d (%s rows so far)", chunk_number, f"{counters['input_rows']:,}")

        unique_rows = int(connection.execute("SELECT COUNT(*) FROM readings").fetchone()[0])
        duplicates_removed = counters["valid_rows_before_deduplication"] - unique_rows

        with output_csv.open("w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "meter", "timestamp", "t_kWh", "avg_voltage_v", "avg_current_a", "frequency_hz"
            ])
            cursor = connection.execute(
                "SELECT meter, timestamp, t_kWh, avg_voltage_v, avg_current_a, frequency_hz "
                "FROM readings ORDER BY meter, timestamp"
            )
            while True:
                rows = cursor.fetchmany(100_000)
                if not rows:
                    break
                writer.writerows(rows)
    finally:
        connection.close()
        if staging_path.exists():
            staging_path.unlink()

    profile: dict[str, object] = {
        "source_file": str(input_path),
        "output_file": str(output_csv),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_used_for_naive_timestamps": source_timezone,
        "rows": {
            **counters,
            "duplicate_meter_timestamp_rows_removed": duplicates_removed,
            "output_rows": unique_rows,
        },
        "electricity_unit_check": {
            "source_column": columns["energy"],
            "assumed_unit": "kWh",
            "reason": "The source field is named t_kWh; verify this against the dataset documentation.",
            "minimum_clean_value": min_energy,
            "maximum_clean_value": max_energy,
            "negative_values_removed": counters["negative_energy_rows"],
            "zero_values_retained": "Yes; zero may be a valid interval reading.",
            "automatic_unit_conversion": False,
        },
        "missing_value_policy": (
            "Rows without meter, timestamp, or valid non-negative t_kWh were removed. "
            "Missing optional voltage/current/frequency values were kept as blank values."
        ),
        "duplicate_policy": "One row per meter and timestamp; last occurrence in source order is kept.",
    }
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Electricity output: %s (%s rows)", output_csv, f"{unique_rows:,}")
    logging.info("Electricity profile: %s", profile_path)
    return profile


def detect_weather_columns(columns: Iterable[object]) -> dict[str, str | None]:
    result = {key: find_column(columns, aliases) for key, aliases in WEATHER_FIELDS.items()}
    if not result.get("timestamp"):
        result["timestamp"] = find_column(
            columns,
            ["timestamp_local", "timestamp", "time", "datetime", "date_time", "x_Timestamp", "timestamp_utc", "time_utc"],
        )
    return result


def clean_weather(
    input_path: Path,
    output_dir: Path,
    source_timezone: str,
) -> dict[str, object]:
    """Clean an hourly (or other interval) weather CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "clean_weather.csv"
    profile_path = output_dir / "weather_cleaning_profile.json"

    logging.info("Cleaning weather data: %s", input_path)
    frame = pd.read_csv(input_path, dtype="string", low_memory=False)
    detected = detect_weather_columns(frame.columns)
    timestamp_col = detected.get("timestamp")
    if not timestamp_col:
        raise ValueError(
            "Could not identify a weather timestamp column. Expected a column such as "
            f"timestamp, time, or datetime. Available columns: {list(frame.columns)}"
        )

    # Rename recognized fields to stable, documented names and retain any
    # additional source columns so useful weather variables are not discarded.
    rename_map: dict[str, str] = {timestamp_col: "timestamp"}
    for standard_name, source_name in detected.items():
        if source_name and standard_name != "timestamp":
            rename_map[source_name] = standard_name
    frame = frame.rename(columns=rename_map)

    parsed_time = parse_timestamp_series(frame["timestamp"], source_timezone)
    invalid_timestamp_count = int(parsed_time.isna().sum())
    frame["timestamp"] = timestamps_to_iso(parsed_time)
    frame = frame.loc[parsed_time.notna()].copy()

    # Convert known weather features and coordinates to numeric values.
    numeric_feature_names = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
        "shortwave_radiation", "direct_radiation", "diffuse_radiation",
        "direct_normal_irradiance", "global_tilted_irradiance", "latitude", "longitude",
    ]
    numeric_missing: dict[str, int] = {}
    for name in numeric_feature_names:
        if name in frame.columns:
            values = numeric_values(frame[name])
            numeric_missing[name] = int(values.isna().sum())
            frame[name] = values

    # Keep physically impossible sensor values as missing rather than fabricating values.
    invalid_ranges: dict[str, int] = {}
    for name in ["relative_humidity_2m", "cloud_cover"]:
        if name in frame.columns:
            invalid = frame[name].notna() & ~frame[name].between(0, 100)
            invalid_ranges[name] = int(invalid.sum())
            frame.loc[invalid, name] = np.nan
    for name in ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance"]:
        if name in frame.columns:
            invalid = frame[name].notna() & frame[name].lt(0)
            invalid_ranges[name] = int(invalid.sum())
            frame.loc[invalid, name] = np.nan

    # Deduplicate by timestamp and the best available location key.
    location_keys = [key for key in ["location_id", "latitude", "longitude"] if key in frame.columns]
    dedup_keys = ["timestamp"] + location_keys
    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=dedup_keys, keep="last")
    duplicates_removed = before_dedup - len(frame)

    # Put standard columns first, while retaining any additional source fields.
    preferred = [
        "timestamp", "location_id", "latitude", "longitude", "temperature_2m",
        "relative_humidity_2m", "dew_point_2m", "cloud_cover", "shortwave_radiation",
        "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
        "global_tilted_irradiance",
    ]
    ordered = [name for name in preferred if name in frame.columns]
    ordered += [name for name in frame.columns if name not in ordered]
    frame = frame[ordered].sort_values(dedup_keys).reset_index(drop=True)
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")

    recognized_weather_features = [
        name for name in [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance", "global_tilted_irradiance",
        ] if name in frame.columns
    ]
    profile: dict[str, object] = {
        "source_file": str(input_path),
        "output_file": str(output_csv),
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_used_for_naive_timestamps": source_timezone,
        "input_rows": int(len(pd.read_csv(input_path, usecols=[timestamp_col]))),
        "invalid_timestamp_rows_removed": invalid_timestamp_count,
        "duplicate_rows_removed": duplicates_removed,
        "output_rows": int(len(frame)),
        "deduplication_keys": dedup_keys,
        "recognized_weather_features": recognized_weather_features,
        "missing_values_after_numeric_conversion": numeric_missing,
        "physically_invalid_values_changed_to_missing": invalid_ranges,
        "missing_value_policy": "Missing weather features are retained as blank values; no imputation is performed.",
    }
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Weather output: %s (%s rows)", output_csv, f"{len(frame):,}")
    logging.info("Weather profile: %s", profile_path)
    if not recognized_weather_features:
        logging.warning(
            "No standard weather variables were recognized. The timestamp and other source columns were retained; review column names."
        )
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean the CEEW electricity dataset and Bareilly weather dataset."
    )
    parser.add_argument(
        "--electricity-input",
        help="Electricity CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--weather-input",
        help="Weather CSV path (absolute or relative to project root).",
    )
    parser.add_argument(
        "--timezone",
        default=LOCAL_TIMEZONE,
        help=f"Timezone used for naive source timestamps (default: {LOCAL_TIMEZONE}).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Number of electricity rows processed per chunk (default: 200000).",
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Clean electricity only; skip the weather dataset.",
    )
    return parser


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()
    if args.chunksize <= 0:
        logging.error("--chunksize must be greater than zero")
        return 2

    try:
        electricity_input = discover_input(
            args.electricity_input,
            ELECTRICITY_CANDIDATES,
            Path("data/raw/electricity"),
            "Electricity",
        )
        if electricity_input is None:
            raise FileNotFoundError(
                "Electricity CSV was not found. Put selected_households.csv in "
                "data/interim/electricity/ or pass --electricity-input."
            )

        clean_electricity(
            electricity_input,
            resolve_project_path("data/interim/electricity"),
            args.timezone,
            args.chunksize,
        )

        if not args.skip_weather:
            weather_input = discover_input(
                args.weather_input,
                WEATHER_CANDIDATES,
                Path("data/raw/weather"),
                "Weather",
            )
            if weather_input is None:
                logging.warning(
                    "Weather CSV not found; electricity cleaning is complete. "
                    "Run again with --weather-input <path> to clean weather data, "
                    "or use --skip-weather to explicitly skip it."
                )
            else:
                clean_weather(
                    weather_input,
                    resolve_project_path("data/interim/weather"),
                    args.timezone,
                )

        logging.info("Cleaning pipeline finished.")
        return 0
    except Exception as exc:
        logging.exception("Cleaning failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
