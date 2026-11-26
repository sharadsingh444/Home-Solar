"""Profile the CEEW Bareilly smart-meter CSV and select households with enough data.

This script never edits the downloaded source CSV. It reads the source in chunks,
creates profiling reports under data/interim/electricity/, and (by default)
creates a separate CSV containing all source rows for households that pass the
minimum usable-reading threshold.

Expected CEEW columns (automatically detected by default):
    meter         -> household / meter ID
    x_Timestamp   -> reading timestamp
    t_kWh         -> electricity reading

If your CSV uses different column names, pass --household-col, --timestamp-col,
and --reading-col when running the script.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project paths and defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(
    r"C:\Users\ashut\OneDrive\Desktop\homsolar\data\raw\electricity\CEEW - Smart meter data Bareilly 2021.csv"
    )
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "electricity"
DEFAULT_CHUNK_SIZE = 200_000

# Starting screening threshold, not a claim about the minimum data needed by
# every model. Change this from the command line with --min-readings.
DEFAULT_MIN_READINGS = 10_000

COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "household": (
        "meter",
        "meter_id",
        "household_id",
        "household",
        "consumer_id",
        "consumerid",
        "hh_id",
        "customer_id",
    ),
    "timestamp": (
        "x_Timestamp",
        "timestamp",
        "datetime",
        "date_time",
        "dateandtime",
        "reading_timestamp",
        "time_stamp",
    ),
    "reading": (
        "t_kWh",
        "kwh",
        "electricity_kwh",
        "energy_kwh",
        "consumption_kwh",
        "active_energy_kwh",
        "reading_kwh",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalise_column_name(name: str) -> str:
    """Make column-name matching insensitive to spaces, punctuation and case."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def resolve_column(
    headers: list[str], role: str, override: str | None
) -> str:
    """Find a requested column, first using an explicit override, then aliases."""
    normalised_headers = {normalise_column_name(h): h for h in headers}

    if override:
        if override in headers:
            return override
        wanted = normalise_column_name(override)
        if wanted in normalised_headers:
            return normalised_headers[wanted]
        raise ValueError(
            f"Column '{override}' supplied for {role!r} was not found. "
            f"Available columns: {headers}"
        )

    for candidate in COLUMN_CANDIDATES[role]:
        key = normalise_column_name(candidate)
        if key in normalised_headers:
            return normalised_headers[key]

    aliases = ", ".join(COLUMN_CANDIDATES[role])
    raise ValueError(
        f"Could not automatically identify the {role} column. "
        f"Expected one of: {aliases}. Available columns: {headers}. "
        f"Run again with the matching --{role}-col option."
    )


def timestamp_to_text(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat(sep=" ")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the CEEW Bareilly smart-meter CSV and create a separate "
            "dataset for households with sufficient usable readings."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the original CEEW CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reports and the selected-household CSV.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows read per chunk; lower this if your computer has limited RAM.",
    )
    parser.add_argument(
        "--min-readings",
        type=int,
        default=DEFAULT_MIN_READINGS,
        help=(
            "Minimum usable readings required for a household to be selected. "
            f"Default: {DEFAULT_MIN_READINGS}."
        ),
    )
    parser.add_argument(
        "--min-coverage-days",
        type=float,
        default=0.0,
        help=(
            "Optional minimum time span between a household's first and last "
            "valid timestamp. Default 0 disables this extra filter."
        ),
    )
    parser.add_argument(
        "--household-col",
        default=None,
        help="Explicit household/meter ID column name (optional).",
    )
    parser.add_argument(
        "--timestamp-col",
        default=None,
        help="Explicit timestamp column name (optional).",
    )
    parser.add_argument(
        "--reading-col",
        default=None,
        help="Explicit electricity-reading column name (optional).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV text encoding. Default: utf-8-sig.",
    )
    parser.add_argument(
        "--separator",
        default=",",
        help="CSV delimiter. Default: comma.",
    )
    parser.add_argument(
        "--dayfirst",
        action="store_true",
        help="Interpret ambiguous dates as day/month/year when parsing timestamps.",
    )
    parser.add_argument(
        "--no-export-selected",
        action="store_true",
        help="Create reports only; do not write selected_households.csv.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N chunks. Default: 5.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main data profiling and household selection
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    input_path: Path = args.input.expanduser()
    output_dir: Path = args.output_dir.expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found:\n{input_path}\n\n"
            "Check the path or pass the correct path using --input."
        )
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be greater than zero.")
    if args.min_readings < 1:
        raise ValueError("--min-readings must be at least 1.")
    if args.min_coverage_days < 0:
        raise ValueError("--min-coverage-days cannot be negative.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")

    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Source CSV: %s", input_path)
    logging.info("Output directory: %s", output_dir)
    logging.info("Reading the CSV header to identify columns...")

    # Read only the header first. This allows meter IDs to be read as strings
    # during the full scan, preserving IDs such as '0012'.
    header_frame = pd.read_csv(
        input_path,
        nrows=0,
        encoding=args.encoding,
        sep=args.separator,
    )
    headers = [str(column) for column in header_frame.columns]
    if not headers:
        raise ValueError("The input CSV does not contain any columns.")

    household_col = resolve_column(headers, "household", args.household_col)
    timestamp_col = resolve_column(headers, "timestamp", args.timestamp_col)
    reading_col = resolve_column(headers, "reading", args.reading_col)

    logging.info("Detected household ID column: %s", household_col)
    logging.info("Detected timestamp column: %s", timestamp_col)
    logging.info("Detected electricity-reading column: %s", reading_col)
    logging.info("Columns in source file: %s", headers)

    # Per-household running totals. Only one small record per household is kept
    # in memory; the full multi-million-row dataset is never loaded at once.
    household_stats: dict[str, dict[str, Any]] = {}
    column_non_null = {column: 0 for column in headers}
    detected_dtypes: dict[str, str] | None = None

    total_rows = 0
    rows_missing_household_id = 0
    rows_with_invalid_timestamp = 0
    rows_with_invalid_reading = 0
    usable_rows = 0
    chunk_count = 0

    logging.info("Starting chunked scan (chunk size: %s rows)...", args.chunksize)

    reader = pd.read_csv(
        input_path,
        chunksize=args.chunksize,
        encoding=args.encoding,
        sep=args.separator,
        low_memory=False,
        dtype={household_col: "string"},
    )

    for chunk in reader:
        chunk_count += 1
        if detected_dtypes is None:
            detected_dtypes = {
                str(column): str(dtype) for column, dtype in chunk.dtypes.items()
            }

        chunk_rows = len(chunk)
        total_rows += chunk_rows
        for column in headers:
            column_non_null[column] += int(chunk[column].notna().sum())

        # Meter IDs are kept as strings so leading zeroes are not lost.
        household_ids = chunk[household_col].astype("string").str.strip()
        id_valid = household_ids.notna() & household_ids.ne("")
        rows_missing_household_id += int((~id_valid).sum())

        timestamps = pd.to_datetime(
            chunk[timestamp_col], errors="coerce", dayfirst=args.dayfirst
        )
        timestamp_valid = timestamps.notna()
        rows_with_invalid_timestamp += int((~timestamp_valid).sum())

        readings = pd.to_numeric(chunk[reading_col], errors="coerce")
        # NaN and +/- infinity are not usable readings. Negative readings are
        # retained for now because their meaning depends on the meter schema.
        reading_valid = readings.notna() & np.isfinite(
            readings.fillna(0).to_numpy(dtype="float64")
        )
        rows_with_invalid_reading += int((~reading_valid).sum())

        usable = id_valid & timestamp_valid & reading_valid
        usable_rows += int(usable.sum())

        # Aggregate only rows with a non-empty household ID. Invalid timestamps
        # and readings remain visible in the per-household quality counts.
        if id_valid.any():
            work = pd.DataFrame(
                {
                    "household_id": household_ids[id_valid].astype(str).to_numpy(),
                    "total_rows": np.ones(int(id_valid.sum()), dtype=np.int64),
                    "valid_timestamps": timestamp_valid[id_valid].to_numpy(
                        dtype=np.int64
                    ),
                    "valid_readings": reading_valid[id_valid].to_numpy(
                        dtype=np.int64
                    ),
                    "usable_readings": usable[id_valid].to_numpy(dtype=np.int64),
                    "timestamp_for_range": timestamps[id_valid].to_numpy(),
                }
            )
            grouped = work.groupby("household_id", sort=False).agg(
                total_rows=("total_rows", "sum"),
                valid_timestamps=("valid_timestamps", "sum"),
                valid_readings=("valid_readings", "sum"),
                usable_readings=("usable_readings", "sum"),
                first_timestamp=("timestamp_for_range", "min"),
                last_timestamp=("timestamp_for_range", "max"),
            )

            for household_id, row in grouped.iterrows():
                current = household_stats.setdefault(
                    str(household_id),
                    {
                        "total_rows": 0,
                        "valid_timestamps": 0,
                        "valid_readings": 0,
                        "usable_readings": 0,
                        "first_timestamp": pd.NaT,
                        "last_timestamp": pd.NaT,
                    },
                )
                for count_name in (
                    "total_rows",
                    "valid_timestamps",
                    "valid_readings",
                    "usable_readings",
                ):
                    current[count_name] += int(row[count_name])

                first_value = row["first_timestamp"]
                last_value = row["last_timestamp"]
                if pd.notna(first_value) and (
                    pd.isna(current["first_timestamp"])
                    or first_value < current["first_timestamp"]
                ):
                    current["first_timestamp"] = first_value
                if pd.notna(last_value) and (
                    pd.isna(current["last_timestamp"])
                    or last_value > current["last_timestamp"]
                ):
                    current["last_timestamp"] = last_value

        if chunk_count % args.log_every == 0:
            logging.info(
                "Scanned %s rows across %s chunks; found %s household IDs so far.",
                f"{total_rows:,}",
                chunk_count,
                f"{len(household_stats):,}",
            )

    if detected_dtypes is None:
        raise ValueError("The CSV contains a header but no data rows.")

    # Build one row per household and calculate its timestamp span.
    summary_records: list[dict[str, Any]] = []
    for household_id, stats in household_stats.items():
        first_ts = stats["first_timestamp"]
        last_ts = stats["last_timestamp"]
        if pd.notna(first_ts) and pd.notna(last_ts):
            coverage_days = (last_ts - first_ts).total_seconds() / 86_400
        else:
            coverage_days = 0.0

        summary_records.append(
            {
                "household_id": household_id,
                "total_rows": stats["total_rows"],
                "valid_timestamps": stats["valid_timestamps"],
                "valid_readings": stats["valid_readings"],
                "usable_readings": stats["usable_readings"],
                "first_timestamp": timestamp_to_text(first_ts),
                "last_timestamp": timestamp_to_text(last_ts),
                "coverage_days": round(float(coverage_days), 3),
            }
        )

    household_summary = pd.DataFrame(
        summary_records,
        columns=[
            "household_id",
            "total_rows",
            "valid_timestamps",
            "valid_readings",
            "usable_readings",
            "first_timestamp",
            "last_timestamp",
            "coverage_days",
        ],
    )
    if not household_summary.empty:
        household_summary = household_summary.sort_values(
            ["usable_readings", "household_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    eligible_mask = pd.Series(dtype=bool)
    if not household_summary.empty:
        eligible_mask = household_summary["usable_readings"].ge(args.min_readings)
        if args.min_coverage_days > 0:
            eligible_mask &= household_summary["coverage_days"].ge(
                args.min_coverage_days
            )
        household_summary["selected"] = eligible_mask
    else:
        household_summary["selected"] = pd.Series(dtype=bool)

    selected_households = set(
        household_summary.loc[household_summary["selected"], "household_id"].astype(str)
        if not household_summary.empty
        else []
    )

    # Column-level null report based on the entire scan.
    column_profile = pd.DataFrame(
        [
            {
                "column": column,
                "dtype_first_chunk": detected_dtypes.get(column, "unknown"),
                "non_null_count": column_non_null[column],
                "missing_count": total_rows - column_non_null[column],
                "missing_percent": round(
                    100 * (total_rows - column_non_null[column]) / total_rows, 3
                )
                if total_rows
                else 0.0,
            }
            for column in headers
        ]
    )

    # Persist reports. These are all new files; the downloaded CSV is untouched.
    household_report_path = output_dir / "household_reading_summary.csv"
    column_report_path = output_dir / "column_profile.csv"
    profile_json_path = output_dir / "dataset_profile.json"
    selected_csv_path = output_dir / "selected_households.csv"

    household_summary.to_csv(household_report_path, index=False)
    column_profile.to_csv(column_report_path, index=False)

    selected_rows_exported: int | None = None
    if not args.no_export_selected:
        logging.info(
            "Writing rows for %s selected households to %s...",
            f"{len(selected_households):,}",
            selected_csv_path,
        )
        selected_rows_exported = 0
        first_output_chunk = True

        # Second pass: export every original row belonging to an eligible
        # household. Rows with invalid fields are retained for later cleaning.
        export_reader = pd.read_csv(
            input_path,
            chunksize=args.chunksize,
            encoding=args.encoding,
            sep=args.separator,
            low_memory=False,
            dtype={household_col: "string"},
        )
        with selected_csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
            for chunk in export_reader:
                ids = chunk[household_col].astype("string").str.strip()
                chosen = chunk.loc[ids.isin(selected_households)]
                if not chosen.empty:
                    chosen.to_csv(
                        output_file,
                        index=False,
                        header=first_output_chunk,
                    )
                    selected_rows_exported += len(chosen)
                    first_output_chunk = False

            # Ensure a header-only file exists when no household passes.
            if first_output_chunk:
                pd.DataFrame(columns=headers).to_csv(
                    output_file, index=False, header=True
                )

    input_size_bytes = input_path.stat().st_size
    profile = {
        "input_file": str(input_path.resolve()),
        "input_size_bytes": input_size_bytes,
        "rows_scanned": total_rows,
        "column_count": len(headers),
        "columns": headers,
        "detected_columns": {
            "household_id": household_col,
            "timestamp": timestamp_col,
            "electricity_reading": reading_col,
        },
        "column_dtypes_from_first_chunk": detected_dtypes,
        "household_count_with_valid_id": len(household_stats),
        "selected_household_count": len(selected_households),
        "minimum_usable_readings": args.min_readings,
        "minimum_timestamp_coverage_days": args.min_coverage_days,
        "rows_missing_household_id": rows_missing_household_id,
        "rows_with_invalid_timestamp": rows_with_invalid_timestamp,
        "rows_with_invalid_reading": rows_with_invalid_reading,
        "usable_rows": usable_rows,
        "chunksize": args.chunksize,
        "chunk_count": chunk_count,
        "selected_rows_exported": selected_rows_exported,
        "outputs": {
            "household_summary": str(household_report_path.resolve()),
            "column_profile": str(column_report_path.resolve()),
            "dataset_profile": str(profile_json_path.resolve()),
            "selected_households_csv": (
                str(selected_csv_path.resolve()) if not args.no_export_selected else None
            ),
        },
        "original_file_modified": False,
    }

    with profile_json_path.open("w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2, ensure_ascii=False)

    # Console summary.
    print("\n========== CEEW DATASET PROFILE ==========")
    print(f"Source file                 : {input_path}")
    print(f"Rows scanned                : {total_rows:,}")
    print(f"Columns                     : {len(headers)}")
    print(f"Households with valid ID    : {len(household_stats):,}")
    print(f"Rows with usable fields     : {usable_rows:,}")
    print(f"Selected households         : {len(selected_households):,}")
    print(f"Minimum usable readings     : {args.min_readings:,}")
    if args.min_coverage_days > 0:
        print(f"Minimum coverage span       : {args.min_coverage_days:g} days")
    print("\nDetected columns:")
    print(f"  Household ID : {household_col}")
    print(f"  Timestamp    : {timestamp_col}")
    print(f"  Reading      : {reading_col}")
    print("\nReports saved:")
    print(f"  {household_report_path}")
    print(f"  {column_report_path}")
    print(f"  {profile_json_path}")
    if not args.no_export_selected:
        print(f"  {selected_csv_path}")
        print(f"  Rows exported: {selected_rows_exported:,}")
    print("\nOriginal downloaded CSV was not modified.")
    print("==========================================\n")

    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = make_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        logging.error("Processing failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
