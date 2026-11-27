#!/usr/bin/env python3
"""Align cleaned CEEW electricity readings with hourly weather observations.

Default inputs:
    data/interim/electricity/clean_electricity.csv
    data/interim/weather/clean_weather.csv

Default output:
    data/interim/aligned/household_electricity_weather.csv

Electricity data can be much more frequent than weather data (for example,
3-minute meter readings and hourly weather). By default, each electricity row
is matched to the most recent weather observation at or before its timestamp
(backward as-of join, 60-minute tolerance). This avoids using a future weather
observation. Unmatched electricity rows are retained with blank weather fields.

If the weather file contains multiple locations, provide a household mapping CSV
with two columns: meter,location_id. Alternatively, use --location-id to apply
one selected weather location to every household.

Run from the Home-Solar project root:
    python etl/align_weather_meter.py
    python etl/align_weather_meter.py --location-map data/raw/household_locations.csv
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
DEFAULT_ELECTRICITY = PROJECT_ROOT / "data/interim/electricity/clean_electricity.csv"
DEFAULT_WEATHER = PROJECT_ROOT / "data/interim/weather/clean_weather.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/aligned/household_electricity_weather.csv"
DEFAULT_TIMEZONE = "Asia/Kolkata"
DEFAULT_CHUNK_SIZE = 150_000


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def normalise_name(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {normalise_name(column): column for column in columns}
    for candidate in candidates:
        match = lookup.get(normalise_name(candidate))
        if match is not None:
            return match
    return None


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match cleaned household electricity readings to weather records by time and location."
    )
    parser.add_argument("--electricity-input", type=Path, default=DEFAULT_ELECTRICITY)
    parser.add_argument("--weather-input", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--location-map",
        type=Path,
        default=None,
        help="Optional CSV with meter and location_id columns, for multiple weather locations.",
    )
    parser.add_argument(
        "--location-id",
        default=None,
        help="Use this one weather location for every household (optional).",
    )
    parser.add_argument(
        "--tolerance-minutes",
        type=float,
        default=60.0,
        help="Maximum allowed time difference for a weather match. Default: 60 minutes.",
    )
    parser.add_argument(
        "--direction",
        choices=["backward", "nearest", "forward"],
        default="backward",
        help="As-of matching direction. Default backward avoids future weather values.",
    )
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser


def load_weather(path: Path, timezone: str, selected_location: str | None) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned weather CSV not found: {path}")

    weather = pd.read_csv(path, low_memory=False)
    if weather.empty:
        raise ValueError(f"Weather CSV is empty: {path}")

    timestamp_col = find_column(
        list(weather.columns),
        ["timestamp", "timestamp_local", "time", "datetime", "timestamp_utc"],
    )
    if timestamp_col is None:
        raise ValueError(
            "Could not find a timestamp column in the weather file. "
            f"Columns found: {list(weather.columns)}"
        )

    # Use timestamp_utc when available only if the primary timestamp field is
    # absent. The clean_meter_data.py output has a normalized `timestamp`.
    weather["__weather_time_utc"] = pd.to_datetime(
        weather[timestamp_col], errors="coerce", utc=True, format="mixed"
    )
    invalid_time_count = int(weather["__weather_time_utc"].isna().sum())
    weather = weather.loc[weather["__weather_time_utc"].notna()].copy()
    if weather.empty:
        raise ValueError("No valid weather timestamps remain after parsing.")

    location_col = find_column(list(weather.columns), ["location_id", "site_id", "grid_id"])
    # The weather file also has a `meter` column (M01-M36), while the
    # electricity file has its own `meter` column (BR02, BR04, ...). Keeping
    # both under the same name makes pandas rename them to meter_x/meter_y
    # during merge_asof, which breaks the final sort by the electricity meter.
    # Preserve the weather-side identifier under a distinct name.
    weather_meter_col = find_column(list(weather.columns), ["meter", "meter_id", "household_id"])
    if weather_meter_col and weather_meter_col != location_col:
        weather.rename(columns={weather_meter_col: "weather_meter_id"}, inplace=True)

    latitude_col = find_column(list(weather.columns), ["latitude", "lat"])
    longitude_col = find_column(list(weather.columns), ["longitude", "lon", "long"])

    if location_col:
        ids = weather[location_col].astype("string").str.strip()
        ids = ids.mask(ids.isna() | ids.eq(""))
        weather["location_id"] = ids
    elif latitude_col and longitude_col:
        lat = pd.to_numeric(weather[latitude_col], errors="coerce")
        lon = pd.to_numeric(weather[longitude_col], errors="coerce")
        if lat.isna().any() or lon.isna().any():
            raise ValueError("Weather latitude/longitude contain missing or invalid values.")
        weather["location_id"] = [
            f"coord_{a:.5f}_{b:.5f}" for a, b in zip(lat, lon)
        ]
    else:
        weather["location_id"] = "bareilly_default"

    if weather["location_id"].isna().any():
        raise ValueError("Some weather rows have no usable location_id.")

    all_location_ids = sorted(weather["location_id"].dropna().astype(str).unique().tolist())
    if selected_location:
        weather = weather.loc[weather["location_id"].astype(str).eq(selected_location)].copy()
        if weather.empty:
            raise ValueError(
                f"Selected --location-id '{selected_location}' was not found. "
                f"Available locations: {all_location_ids}"
            )

    # Keep every weather variable except source timestamps; avoid name clashes
    # with the electricity timestamp column.
    excluded = {timestamp_col, "__weather_time_utc"}
    for column in list(weather.columns):
        if normalise_name(column) in {"timestamputc", "timestampgmt", "weather timestamp"}:
            excluded.add(column)
    weather_fields = [c for c in weather.columns if c not in excluded and c != "location_id"]
    keep = ["location_id", "__weather_time_utc"] + weather_fields
    weather = weather[keep].copy()
    weather = weather.rename(columns={"__weather_time_utc": "weather_timestamp_utc"})
    weather = weather.drop_duplicates(
        subset=["location_id", "weather_timestamp_utc"], keep="last"
    )
    weather["location_id"] = weather["location_id"].astype("string")
    weather = weather.sort_values(
        ["weather_timestamp_utc", "location_id"], kind="mergesort"
    ).reset_index(drop=True)
    location_ids = sorted(weather["location_id"].astype(str).unique().tolist())

    logging.info("Weather rows loaded: %s", f"{len(weather):,}")
    logging.info("Weather locations: %s", location_ids)
    if invalid_time_count:
        logging.warning("Weather rows with invalid timestamps ignored: %s", f"{invalid_time_count:,}")
    return weather, location_ids


def load_location_map(path: Path, available_locations: list[str]) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Household location mapping file not found: {path}")
    mapping = pd.read_csv(path, dtype="string")
    meter_col = find_column(list(mapping.columns), ["meter", "meter_id", "household_id"])
    location_col = find_column(list(mapping.columns), ["location_id", "site_id", "grid_id"])
    if not meter_col or not location_col:
        raise ValueError(
            "Location mapping CSV must contain columns named 'meter' and 'location_id'. "
            f"Found: {list(mapping.columns)}"
        )
    mapping = mapping[[meter_col, location_col]].copy()
    mapping[meter_col] = mapping[meter_col].astype("string").str.strip()
    mapping[location_col] = mapping[location_col].astype("string").str.strip()
    if mapping[meter_col].duplicated().any():
        duplicates = mapping.loc[mapping[meter_col].duplicated(), meter_col].head(10).tolist()
        raise ValueError(f"Duplicate meter IDs in location map: {duplicates}")
    result = dict(zip(mapping[meter_col].astype(str), mapping[location_col].astype(str)))
    invalid = sorted(set(result.values()) - set(available_locations))
    if invalid:
        raise ValueError(
            f"Location map refers to weather location IDs that do not exist: {invalid}. "
            f"Available IDs: {available_locations}"
        )
    return result


def run(args: argparse.Namespace) -> int:
    electricity_path = resolve_path(args.electricity_input)
    weather_path = resolve_path(args.weather_input)
    output_path = resolve_path(args.output)

    if not electricity_path.is_file():
        raise FileNotFoundError(f"Cleaned electricity CSV not found: {electricity_path}")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be greater than zero.")
    if args.tolerance_minutes < 0:
        raise ValueError("--tolerance-minutes cannot be negative.")

    weather, available_locations = load_weather(weather_path, args.timezone, args.location_id)
    if args.location_id:
        location_map = None
        default_location = args.location_id
    elif args.location_map:
        location_map = load_location_map(resolve_path(args.location_map), available_locations)
        default_location = None
    elif len(available_locations) == 1:
        location_map = None
        default_location = available_locations[0]
    else:
        raise ValueError(
            "The weather file contains multiple locations. Supply --location-map "
            "with meter,location_id columns, or use --location-id to apply one "
            f"location to all households. Available locations: {available_locations}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path = output_path.with_name(output_path.stem + "_profile.json")
    if output_path.exists():
        output_path.unlink()

    # Weather is normally only ~8,760 rows per location/year, so it can be held
    # in memory while the much larger electricity CSV is streamed in chunks.
    counters = {
        "electricity_rows": 0,
        "valid_electricity_rows": 0,
        "output_rows": 0,
        "matched_rows": 0,
        "unmatched_rows": 0,
        "rows_missing_timestamp": 0,
        "rows_missing_household": 0,
        "rows_excluded_unmapped_meter": 0,
        "chunks_processed": 0,
    }
    first_output_chunk = True
    seen_meters: set[str] = set()
    excluded_unmapped_meters: set[str] = set()
    last_sort_key: tuple[str, pd.Timestamp] | None = None
    time_tolerance = pd.Timedelta(minutes=args.tolerance_minutes)

    header = pd.read_csv(electricity_path, nrows=0)
    meter_col = find_column(list(header.columns), ["meter", "meter_id", "household_id"])
    timestamp_col = find_column(list(header.columns), ["timestamp", "timestamp_local", "datetime"])
    if not meter_col or not timestamp_col:
        raise ValueError(
            "Cleaned electricity file must contain household/meter and timestamp columns. "
            f"Found: {list(header.columns)}"
        )

    logging.info("Electricity input: %s", electricity_path)
    logging.info("Weather input: %s", weather_path)
    logging.info("Matching direction: %s; tolerance: %s minutes", args.direction, args.tolerance_minutes)

    for chunk_number, chunk in enumerate(
        pd.read_csv(electricity_path, dtype={meter_col: "string"}, chunksize=args.chunksize, low_memory=False),
        start=1,
    ):
        counters["chunks_processed"] += 1
        counters["electricity_rows"] += len(chunk)
        chunk[meter_col] = chunk[meter_col].astype("string").str.strip()
        chunk["__electricity_time_utc"] = pd.to_datetime(
            chunk[timestamp_col], errors="coerce", utc=True, format="mixed"
        )

        invalid_meter = chunk[meter_col].isna() | chunk[meter_col].eq("")
        invalid_time = chunk["__electricity_time_utc"].isna()
        counters["rows_missing_household"] += int(invalid_meter.sum())
        counters["rows_missing_timestamp"] += int(invalid_time.sum())
        chunk = chunk.loc[~(invalid_meter | invalid_time)].copy()
        if chunk.empty:
            continue
        counters["valid_electricity_rows"] += len(chunk)

        if location_map is not None:
            mapped_locations = chunk[meter_col].astype(str).map(location_map)
            unmapped_mask = mapped_locations.isna()
            if unmapped_mask.any():
                unmapped_ids = (
                    chunk.loc[unmapped_mask, meter_col]
                    .dropna().astype(str).drop_duplicates().tolist()
                )
                excluded_unmapped_meters.update(unmapped_ids)
                excluded_count = int(unmapped_mask.sum())
                counters["rows_excluded_unmapped_meter"] += excluded_count
                logging.warning(
                    "Excluding %s electricity row(s) from unmapped meter(s): %s",
                    f"{excluded_count:,}",
                    unmapped_ids,
                )
                # The supplied location map defines the households included in
                # this analysis. Unmapped meters (for example BR52 when the map
                # intentionally contains 36 households) are excluded rather
                # than causing the entire ETL pipeline to fail.
                chunk = chunk.loc[~unmapped_mask].copy()
                mapped_locations = mapped_locations.loc[~unmapped_mask]
            if chunk.empty:
                continue
            chunk["location_id"] = mapped_locations.astype("string")
        else:
            chunk["location_id"] = default_location
        chunk["location_id"] = chunk["location_id"].astype("string")

        seen_meters.update(chunk[meter_col].dropna().astype(str).unique().tolist())

        # This validates the ordering needed by the downstream streaming feature
        # builder. clean_meter_data.py writes rows sorted by meter and timestamp.
        chunk = chunk.sort_values([meter_col, "__electricity_time_utc"], kind="mergesort")
        for meter_value, ts_value in zip(
            chunk[meter_col].astype(str), chunk["__electricity_time_utc"]
        ):
            key = (meter_value, ts_value)
            if last_sort_key is not None and key < last_sort_key:
                raise ValueError(
                    "The cleaned electricity CSV is not globally sorted by meter and timestamp. "
                    "Re-run clean_meter_data.py, which writes this ordering, before alignment."
                )
            last_sort_key = key

        left = chunk.sort_values(
            ["__electricity_time_utc", "location_id"], kind="mergesort"
        )
        right = weather.sort_values(
            ["weather_timestamp_utc", "location_id"], kind="mergesort"
        )
        aligned = pd.merge_asof(
            left,
            right,
            left_on="__electricity_time_utc",
            right_on="weather_timestamp_utc",
            by="location_id",
            direction=args.direction,
            tolerance=time_tolerance,
            allow_exact_matches=True,
        )

        aligned["weather_match"] = aligned["weather_timestamp_utc"].notna()
        offset_minutes = (
            aligned["__electricity_time_utc"] - aligned["weather_timestamp_utc"]
        ).dt.total_seconds() / 60.0
        aligned["weather_time_offset_minutes"] = offset_minutes
        aligned["weather_time_difference_minutes"] = offset_minutes.abs()
        aligned["weather_is_future"] = offset_minutes.lt(0).fillna(False)
        aligned["timestamp_utc"] = aligned["__electricity_time_utc"]
        aligned = aligned.drop(columns=["__electricity_time_utc"])
        aligned = aligned.sort_values([meter_col, "timestamp_utc"], kind="mergesort")

        matched_count = int(aligned["weather_match"].sum())
        counters["matched_rows"] += matched_count
        counters["unmatched_rows"] += len(aligned) - matched_count
        counters["output_rows"] += len(aligned)

        aligned.to_csv(
            output_path,
            mode="w" if first_output_chunk else "a",
            header=first_output_chunk,
            index=False,
            encoding="utf-8-sig" if first_output_chunk else "utf-8",
        )
        first_output_chunk = False
        logging.info(
            "Aligned chunk %d: %s electricity rows; %s weather matches",
            chunk_number,
            f"{len(aligned):,}",
            f"{matched_count:,}",
        )

    if first_output_chunk:
        raise ValueError("No valid electricity rows were found to align.")

    profile = {
        "electricity_input": str(electricity_path),
        "weather_input": str(weather_path),
        "output_file": str(output_path),
        "timezone": args.timezone,
        "matching": {
            "method": "pandas merge_asof by location_id",
            "direction": args.direction,
            "tolerance_minutes": args.tolerance_minutes,
            "future_weather_allowed": args.direction in {"nearest", "forward"},
            "unmatched_rows_retained": True,
        },
        "locations": available_locations,
        "included_electricity_meters": sorted(seen_meters),
        "excluded_unmapped_electricity_meters": sorted(excluded_unmapped_meters),
        "rows": counters,
        "match_percent": round(
            100.0 * counters["matched_rows"] / counters["output_rows"], 3
        ) if counters["output_rows"] else 0.0,
        "timestamp_check": (
            "weather_time_offset_minutes = electricity_timestamp - matched_weather_timestamp; "
            "positive means weather observation is earlier, negative means it is in the future."
        ),
    }
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    logging.info("Aligned dataset saved: %s", output_path)
    logging.info("Alignment profile saved: %s", profile_path)
    logging.info(
        "Matched %s / %s mapped rows (%.2f%%); unmatched weather rows retained; "
        "excluded %s rows from unmapped meters %s.",
        f"{counters['matched_rows']:,}",
        f"{counters['output_rows']:,}",
        profile["match_percent"],
        f"{counters['rows_excluded_unmapped_meter']:,}",
        sorted(excluded_unmapped_meters),
    )
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
        logging.exception("Electricity-weather alignment failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
