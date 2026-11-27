#!/usr/bin/env python3
"""Run the complete Home-Solar ETL workflow in the required order.

The electricity and weather CSVs are expected to have already been downloaded
manually and placed in data/raw/. The first two stages prepare those local files;
they do not fetch from CEEW or Open-Meteo over the internet.

Run from the Home-Solar project root:
    python etl/run_pipeline.py

When more than one source CSV exists in a raw folder, specify it explicitly:
    python etl/run_pipeline.py --electricity-input "data/raw/electricity/meter.csv" \
        --weather-input "data/raw/weather/weather_2020.csv"

For multiple weather locations, also provide a meter-to-location mapping CSV:
    python etl/run_pipeline.py --location-map data/raw/household_locations.csv
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ETL_DIR = PROJECT_ROOT / "etl"
RAW_ELECTRICITY_DIR = PROJECT_ROOT / "data/raw/electricity"
RAW_WEATHER_DIR = PROJECT_ROOT / "data/raw/weather"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def discover_csv(explicit: Path | None, folder: Path, label: str, preferred_name: str | None = None) -> Path:
    if explicit is not None:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{label} CSV not found: {path}")
        return path

    preferred = folder / preferred_name if preferred_name else None
    if preferred and preferred.is_file():
        return preferred

    files = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(
            f"No {label} CSV found in {folder}. Place the file there or pass the appropriate input option."
        )
    names = "\n  ".join(str(path.relative_to(PROJECT_ROOT)) for path in files)
    raise ValueError(
        f"More than one {label} CSV exists in {folder}. Choose one with the command-line option:\n  {names}"
    )


def run_step(label: str, script: Path, arguments: list[str]) -> None:
    if not script.is_file():
        raise FileNotFoundError(f"Required ETL script is missing: {script}")
    command = [sys.executable, str(script), *arguments]
    logging.info("\n%s", "=" * 76)
    logging.info("STEP: %s", label)
    logging.info("COMMAND: %s", " ".join(f'"{part}"' if " " in part else part for part in command))
    logging.info("%s", "=" * 76)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step '{label}' failed with exit code {completed.returncode}. "
            "The pipeline has stopped; fix this step before continuing."
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the complete Home-Solar ETL workflow sequentially."
    )
    parser.add_argument("--electricity-input", type=Path, default=None)
    parser.add_argument("--weather-input", type=Path, default=None)
    parser.add_argument("--min-readings", type=int, default=10_000)
    parser.add_argument("--min-coverage-days", type=float, default=0.0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--location-map",
        type=Path,
        default=None,
        help="CSV with meter,location_id columns if weather data has multiple locations.",
    )
    parser.add_argument(
        "--location-id",
        default=None,
        help="Apply one weather location to all households if weather has multiple locations.",
    )
    parser.add_argument("--match-tolerance-minutes", type=float, default=60.0)
    parser.add_argument("--match-direction", choices=["backward", "nearest", "forward"], default="backward")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--timezone", default="Asia/Kolkata", help="Local timezone used for timestamps/features.")
    parser.add_argument(
        "--weather-input-timezone",
        default="Asia/Kolkata",
        help="Timezone used to interpret timezone-naive weather timestamps.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = make_parser().parse_args()

    try:
        electricity_source = discover_csv(
            args.electricity_input,
            RAW_ELECTRICITY_DIR,
            "electricity",
            preferred_name="CEEW - Smart meter data Bareilly 2020.csv",
        )
        weather_source = discover_csv(
            args.weather_input,
            RAW_WEATHER_DIR,
            "weather",
            preferred_name="bareilly_weather_2020.csv",
        )
        if args.min_readings < 1:
            raise ValueError("--min-readings must be at least 1.")
        if args.min_coverage_days < 0:
            raise ValueError("--min-coverage-days cannot be negative.")
        if args.chunksize < 1:
            raise ValueError("--chunksize must be greater than zero.")
        if args.match_tolerance_minutes < 0:
            raise ValueError("--match-tolerance-minutes cannot be negative.")
        if not 0.0 < args.test_fraction < 1.0:
            raise ValueError("--test-fraction must be between 0 and 1 (exclusive).")

        selected_households = PROJECT_ROOT / "data/interim/electricity/selected_households.csv"
        standardized_weather = PROJECT_ROOT / "data/interim/weather/bareilly_weather_hourly.csv"
        clean_electricity = PROJECT_ROOT / "data/interim/electricity/clean_electricity.csv"
        clean_weather = PROJECT_ROOT / "data/interim/weather/clean_weather.csv"
        aligned_data = PROJECT_ROOT / "data/interim/aligned/household_electricity_weather.csv"
        final_data = PROJECT_ROOT / "data/processed/household_energy_dataset.csv"

        # Step 1: The source file is already downloaded by the user. This step
        # profiles it and exports rows for households meeting the threshold.
        run_step(
            "1/6 — Inspect CEEW electricity data and select households",
            ETL_DIR / "download_ceew.py",
            [
                "--input", str(electricity_source),
                "--output-dir", str(PROJECT_ROOT / "data/interim/electricity"),
                "--min-readings", str(args.min_readings),
                "--min-coverage-days", str(args.min_coverage_days),
                "--chunksize", str(args.chunksize),
            ],
        )
        if not selected_households.is_file():
            raise FileNotFoundError(f"Household selection output was not created: {selected_households}")

        # Step 2: Prepare the existing local weather CSV; fetch_weather.py does
        # not make an API request.
        run_step(
            "2/6 — Standardize the existing Bareilly weather CSV",
            ETL_DIR / "fetch_weather.py",
            [
                "--input-file", str(weather_source),
                "--input-timezone", args.weather_input_timezone,
                "--timezone", args.timezone,
                "--standardized-name", "bareilly_weather_hourly.csv",
            ],
        )
        if not standardized_weather.is_file():
            raise FileNotFoundError(f"Standardized weather output was not created: {standardized_weather}")

        # Step 3: Validate the tariff JSON and load tariff categories into its
        # SQLite staging database.
        run_step(
            "3/6 — Validate and load UPPCL tariff configuration",
            ETL_DIR / "load_tariff.py",
            [],
        )

        # Step 4: Clean both electricity and weather datasets.
        run_step(
            "4/6 — Clean electricity and weather datasets",
            ETL_DIR / "clean_meter_data.py",
            [
                "--electricity-input", str(selected_households),
                "--weather-input", str(standardized_weather),
                "--timezone", args.timezone,
                "--chunksize", str(args.chunksize),
            ],
        )
        if not clean_electricity.is_file() or not clean_weather.is_file():
            raise FileNotFoundError("Expected clean electricity/weather outputs were not created.")

        # Step 5: Align interval electricity data to available weather times.
        align_args = [
            "--electricity-input", str(clean_electricity),
            "--weather-input", str(clean_weather),
            "--tolerance-minutes", str(args.match_tolerance_minutes),
            "--direction", args.match_direction,
            "--timezone", args.timezone,
        ]
        if args.location_map:
            align_args.extend(["--location-map", str(resolve_path(args.location_map))])
        if args.location_id:
            align_args.extend(["--location-id", args.location_id])
        run_step(
            "5/6 — Align electricity readings with weather observations",
            ETL_DIR / "align_weather_meter.py",
            align_args,
        )
        if not aligned_data.is_file():
            raise FileNotFoundError(f"Aligned output was not created: {aligned_data}")

        # Step 6: Create calendar/history features and chronological splits.
        run_step(
            "6/6 — Build final ML dataset and chronological train/test splits",
            ETL_DIR / "build_dataset.py",
            [
                "--input", str(aligned_data),
                "--output-dir", str(PROJECT_ROOT / "data/processed"),
                "--test-fraction", str(args.test_fraction),
                "--timezone", args.timezone,
            ],
        )
        if not final_data.is_file():
            raise FileNotFoundError(f"Final dataset was not created: {final_data}")

        logging.info("\n%s", "=" * 76)
        logging.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("Final dataset: %s", final_data)
        logging.info("Training data: %s", PROJECT_ROOT / "data/processed/household_energy_train.csv")
        logging.info("Testing data : %s", PROJECT_ROOT / "data/processed/household_energy_test.csv")
        logging.info("%s", "=" * 76)
        return 0

    except Exception as exc:
        logging.exception("ETL pipeline stopped: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
