"""Prepare an already-downloaded Bareilly weather CSV for the Home-Solar project.

This script does NOT call Open-Meteo or download anything.
It reads an existing CSV from data/raw/weather/ (or a path supplied by the user),
keeps the original file unchanged, and writes a standardized copy to
 data/interim/weather/.

Run from the Home-Solar project root:
    python etl/fetch_weather.py --input-file "data/raw/weather/your_weather_file.csv"

If your file is named bareilly_weather_2020.csv and is already in
 data/raw/weather/, this is enough:
    python etl/fetch_weather.py

Required package: pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths and defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
INTERIM_WEATHER_DIR = PROJECT_ROOT / "data" / "interim" / "weather"

DEFAULT_INPUT_FILE = RAW_WEATHER_DIR / "bareilly_weather_2020.csv"
DEFAULT_TIMEZONE = "Asia/Kolkata"

# Replace these with the actual project/site coordinates if known.
# They are metadata only; this script does not fetch weather using them.
DEFAULT_LATITUDE = 28.3670
DEFAULT_LONGITUDE = 79.4304

# Canonical names added to the output when matching source columns are found.
# The original input columns are retained.
COLUMN_ALIASES = {
    "temperature_2m": ["temperature_2m", "temperature", "temp", "temp_c", "temperature_c", "t2m"],
    "relative_humidity_2m": ["relative_humidity_2m", "relative_humidity", "humidity", "rh", "relative_humidity_2m_percent"],
    "dew_point_2m": ["dew_point_2m", "dewpoint_2m", "dew_point", "dewpoint"],
    "cloud_cover": ["cloud_cover", "cloudcover", "cloud_cover_total", "total_cloud_cover"],
    "shortwave_radiation": ["shortwave_radiation", "ghi", "global_horizontal_irradiance", "solar_radiation", "surface_solar_radiation_downwards"],
    "direct_radiation": ["direct_radiation", "direct_solar_radiation"],
    "diffuse_radiation": ["diffuse_radiation", "dhi", "diffuse_solar_radiation"],
    "direct_normal_irradiance": ["direct_normal_irradiance", "dni"],
    "global_tilted_irradiance": ["global_tilted_irradiance", "gti"],
}

TIME_COLUMN_CANDIDATES = [
    "time", "timestamp", "datetime", "date_time", "date", "x_timestamp",
    "timestamp_utc", "time_utc", "datetime_utc",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_output_directories() -> None:
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_WEATHER_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_name(value: Any) -> str:
    """Normalize a column name for case-insensitive matching."""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def find_time_column(columns: list[str]) -> str | None:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in TIME_COLUMN_CANDIDATES:
        match = normalized.get(normalize_column_name(candidate))
        if match:
            return match

    # Fallback for less common timestamp column names.
    for column in columns:
        name = normalize_column_name(column)
        if "timestamp" in name or name in {"time", "datetime"}:
            return column
    return None
