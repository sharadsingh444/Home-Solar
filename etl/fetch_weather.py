#!/usr/bin/env python3
"""Standardize an already-downloaded, multi-meter Bareilly weather CSV.

This script does not call Open-Meteo or download data. It reads the user's
existing weather CSV, preserves its original columns and writes a standardized
copy under data/interim/weather/.

The supplied Bareilly weather format is supported directly:
    meter, lat, lon, date_ist, date_utc,
    shortwave_radiation, direct_radiation, global_tilted_irradiance,
    temperature_2m, relative_humidity_2m, dew_point_2m, ...

`date_ist` is preferred over `date_utc` because it is already local time.
The per-meter `meter` field becomes `location_id`, and lat/lon become the
canonical latitude/longitude fields. This preserves the 36 separate locations
for the later electricity-weather alignment step.

Run from C:\\Home-Solar:
    python etl\\fetch_weather.py --input-file "data/raw/weather/bareilly_36_meters_weather_2021.csv"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "weather"
DEFAULT_INPUT_FILE = RAW_WEATHER_DIR / "bareilly_36_meters_weather_2021.csv"
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Prefer date_ist when both date_ist and date_utc exist.
TIME_COLUMN_CANDIDATES = [
    "date_ist",
    "timestamp_local",
    "time_local",
    "timestamp",
    "datetime",
    "date_time",
    "time",
    "date_utc",
    "timestamp_utc",
    "time_utc",
    "datetime_utc",
    "date",
]

COLUMN_ALIASES: dict[str, list[str]] = {
    "temperature_2m": ["temperature_2m", "temperature", "temp", "temp_c", "t2m"],
    "relative_humidity_2m": [
        "relative_humidity_2m", "relative_humidity", "humidity", "rh"
    ],
    "dew_point_2m": ["dew_point_2m", "dewpoint_2m", "dew_point", "dewpoint"],
    "cloud_cover": ["cloud_cover", "cloudcover", "total_cloud_cover"],
    "shortwave_radiation": [
        "shortwave_radiation", "ghi", "global_horizontal_irradiance",
        "solar_radiation", "surface_solar_radiation_downwards",
    ],
    "direct_radiation": ["direct_radiation", "direct_solar_radiation"],
    "diffuse_radiation": ["diffuse_radiation", "dhi", "diffuse_solar_radiation"],
    "direct_normal_irradiance": ["direct_normal_irradiance", "dni"],
    "global_tilted_irradiance": ["global_tilted_irradiance", "gti"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long"],
    "location_id": ["location_id", "site_id", "grid_id", "location"],
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalise_name(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {normalise_name(column): column for column in columns}
    for candidate in candidates:
        actual = lookup.get(normalise_name(candidate))
        if actual is not None:
            return actual
    return None


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path.resolve()
    return path.resolve()


def parse_and_convert_timestamps(
    values: pd.Series,
    source_column: str,
    input_timezone: str,
    output_timezone: str,
) -> tuple[pd.Series, pd.Series]:
    """Return timezone-aware local and UTC timestamp series.

    Naive `date_ist` values are interpreted in Asia/Kolkata. If a UTC-named
    source column is selected, naive values are interpreted as UTC. Values
    that already contain an offset are converted without relocalizing them.
    """
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    source_is_utc = "utc" in normalise_name(source_column)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        aware = parsed
    elif pd.api.types.is_datetime64_dtype(parsed.dtype):
        source_tz = "UTC" if source_is_utc else input_timezone
        aware = parsed.dt.tz_localize(
            source_tz,
            ambiguous="NaT",
            nonexistent="NaT",
        )
    else:
        # Fallback for mixed naive and offset-aware timestamps.
        source_tz = "UTC" if source_is_utc else input_timezone

        def convert_one(value: object):
            if pd.isna(value):
                return pd.NaT
            try:
                stamp = pd.Timestamp(value)
                if stamp.tzinfo is None:
                    return stamp.tz_localize(
                        source_tz, ambiguous="NaT", nonexistent="NaT"
                    ).tz_convert(output_timezone)
                return stamp.tz_convert(output_timezone)
            except (TypeError, ValueError, OverflowError):
                return pd.NaT

        local = pd.Series(
            pd.array(
                parsed.map(convert_one),
                dtype=f"datetime64[ns, {output_timezone}]",
            ),
            index=values.index,
        )
        utc = local.dt.tz_convert("UTC")
        return local, utc

    local = aware.dt.tz_convert(output_timezone)
    utc = local.dt.tz_convert("UTC")
    return local, utc


def to_iso(series: pd.Series) -> pd.Series:
    """Serialize timezone-aware datetimes with an explicit UTC offset."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%S%z").str.replace(
        r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", regex=True
    )


def prepare_weather(
    input_file: Path,
    output_dir: Path,
    output_name: str,
    input_timezone: str,
    output_timezone: str,
    fallback_latitude: float | None = None,
    fallback_longitude: float | None = None,
) -> Path:
    input_file = resolve_path(input_file)
    output_dir = resolve_path(output_dir)

    if not input_file.is_file():
        raise FileNotFoundError(f"Weather input CSV does not exist: {input_file}")
    if input_file.suffix.lower() != ".csv":
        raise ValueError("Weather input must be a .csv file.")

    logging.info("Reading weather CSV: %s", input_file)
    frame = pd.read_csv(input_file, low_memory=False)
    if frame.empty:
        raise ValueError("The weather CSV contains no data rows.")

    logging.info("Input rows: %s", f"{len(frame):,}")
    logging.info("Input columns: %s", list(frame.columns))

    source_time_col = find_column(list(frame.columns), TIME_COLUMN_CANDIDATES)
    if source_time_col is None:
        raise ValueError(
            "Could not identify the timestamp column. Your file has columns: "
            f"{list(frame.columns)}. This script supports date_ist and date_utc."
        )
    logging.info("Timestamp column selected: %s", source_time_col)

    local_time, utc_time = parse_and_convert_timestamps(
        frame[source_time_col], source_time_col, input_timezone, output_timezone
    )
    frame["timestamp_local"] = to_iso(local_time)
    frame["timestamp_utc"] = to_iso(utc_time)

    # Keep the original source columns while adding canonical weather columns.
    original_columns = list(frame.columns)
    lookup = {normalise_name(column): column for column in frame.columns}
    recognized: list[str] = []
    for canonical, aliases in COLUMN_ALIASES.items():
        source = None
        for alias in aliases:
            source = lookup.get(normalise_name(alias))
            if source is not None:
                break
        if source is not None:
            if canonical not in frame.columns:
                if canonical in {"latitude", "longitude"}:
                    frame[canonical] = pd.to_numeric(frame[source], errors="coerce")
                elif canonical == "location_id":
                    frame[canonical] = frame[source].astype("string").str.strip()
                else:
                    frame[canonical] = pd.to_numeric(frame[source], errors="coerce")
            recognized.append(canonical)

    # The supplied CSV uses meter as the location key (M01, M02, ...).
    meter_col = find_column(list(frame.columns), ["meter", "meter_id", "household_id"])
    if meter_col:
        frame[meter_col] = frame[meter_col].astype("string").str.strip()
        if "location_id" not in frame.columns:
            frame["location_id"] = frame[meter_col]
        else:
            # The meter is the per-household weather-location identifier in this file.
            frame["location_id"] = frame[meter_col]
        logging.info("Using '%s' as the weather location identifier.", meter_col)
    elif "location_id" not in frame.columns:
        lat_col = find_column(list(frame.columns), ["latitude", "lat"])
        lon_col = find_column(list(frame.columns), ["longitude", "lon", "long"])
        if lat_col and lon_col:
            lat = pd.to_numeric(frame[lat_col], errors="coerce")
            lon = pd.to_numeric(frame[lon_col], errors="coerce")
            frame["location_id"] = [
                f"coord_{a:.5f}_{b:.5f}" if pd.notna(a) and pd.notna(b) else pd.NA
                for a, b in zip(lat, lon)
            ]
        else:
            frame["location_id"] = "bareilly_default"

    # If the file has no coordinates, allow explicit fallback values; otherwise
    # do not overwrite the actual per-meter coordinates with one city coordinate.
    if "latitude" not in frame.columns and fallback_latitude is not None:
        frame["latitude"] = fallback_latitude
    if "longitude" not in frame.columns and fallback_longitude is not None:
        frame["longitude"] = fallback_longitude

    # Normalize numeric fields and log data quality, but retain missing values.
    numeric_columns = [
        "latitude", "longitude", "temperature_2m", "relative_humidity_2m",
        "dew_point_2m", "cloud_cover", "shortwave_radiation", "direct_radiation",
        "diffuse_radiation", "direct_normal_irradiance", "global_tilted_irradiance",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_timestamp_count = int(frame["timestamp_local"].isna().sum())
    if invalid_timestamp_count == len(frame):
        raise ValueError(
            f"No valid timestamps were parsed from '{source_time_col}'. Check date_ist/date_utc formatting."
        )
    if invalid_timestamp_count:
        logging.warning("Rows with invalid timestamps: %s", f"{invalid_timestamp_count:,}")

    # Preserve all rows, but sort for easier inspection and downstream processing.
    frame = frame.loc[frame["timestamp_local"].notna()].copy()
    frame = frame.sort_values(
        ["location_id", "timestamp_utc"], kind="mergesort"
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")

    weather_variables = [
        name for name in [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
            "shortwave_radiation", "direct_radiation", "diffuse_radiation",
            "direct_normal_irradiance", "global_tilted_irradiance",
        ] if name in frame.columns
    ]
    location_ids = sorted(frame["location_id"].dropna().astype(str).unique().tolist())
    profile = {
        "source_file": str(input_file),
        "output_file": str(output_path),
        "input_rows": int(len(pd.read_csv(input_file, usecols=[source_time_col]))),
        "output_rows": int(len(frame)),
        "timestamp_source_column": source_time_col,
        "input_timezone_for_naive_values": input_timezone,
        "output_timezone": output_timezone,
        "location_key_source": meter_col if meter_col else "location_id/coordinates",
        "number_of_locations": len(location_ids),
        "location_ids_sample": location_ids[:50],
        "weather_variables_found": weather_variables,
        "weather_variables_not_found": [
            name for name in [
                "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
                "shortwave_radiation", "direct_radiation", "global_tilted_irradiance",
            ] if name not in frame.columns
        ],
        "invalid_timestamp_rows_removed": invalid_timestamp_count,
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        },
        "first_timestamp_local": str(local_time.min()) if local_time.notna().any() else None,
        "last_timestamp_local": str(local_time.max()) if local_time.notna().any() else None,
        "note": "Original input CSV is unchanged. date_ist is preferred over date_utc when both exist.",
    }
    profile_path = output_dir / "weather_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    logging.info("Rows written: %s", f"{len(frame):,}")
    logging.info("Weather locations found: %s", len(location_ids))
    logging.info("Weather variables found: %s", weather_variables)
    logging.info("Standardized weather CSV: %s", output_path)
    logging.info("Weather profile: %s", profile_path)
    logging.info("Original input CSV was not modified.")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standardize an existing Bareilly weather CSV; no API download is performed."
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--input-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--standardized-name", default="bareilly_weather_hourly.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latitude", type=float, default=None,
                        help="Optional fallback latitude, used only if the CSV has no latitude/lat column.")
    parser.add_argument("--longitude", type=float, default=None,
                        help="Optional fallback longitude, used only if the CSV has no longitude/lon column.")
    return parser


def main() -> int:
    setup_logging()
    args = build_parser().parse_args()
    try:
        output = prepare_weather(
            input_file=args.input_file,
            output_dir=args.output_dir,
            output_name=args.standardized_name,
            input_timezone=args.input_timezone,
            output_timezone=args.timezone,
            fallback_latitude=args.latitude,
            fallback_longitude=args.longitude,
        )
        logging.info("Weather preparation completed successfully: %s", output)
        return 0
    except Exception as exc:
        logging.exception("Weather preparation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
