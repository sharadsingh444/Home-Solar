"""Diagnose whether CEEW t_kWh is interval energy or a cumulative register.

Run from the Home-Solar project root:
    python etl/diagnose_energy_units.py

The script reads only the required columns and a limited number of rows from
one meter (default: first meter in the file). It does not modify any data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

DEFAULT_INPUT = Path(
    r"data/raw/electricity/CEEW - Smart meter data Bareilly 2021.csv"
)
COLUMN_MAP = {
    "timestamp": "x_Timestamp",
    "energy": "t_kWh",
    "voltage": "z_Avg Voltage (Volt)",
    "current": "z_Avg Current (Amp)",
    "meter": "meter",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether t_kWh behaves like interval kWh or a cumulative meter register."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--meter", type=str, default=None,
                        help="Meter to inspect, e.g. BR02. Defaults to the first meter in the file.")
    parser.add_argument("--rows", type=int, default=300,
                        help="Maximum rows to inspect for the selected meter (default: 300).")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        # Read in chunks so the multi-million-row source is not loaded into RAM.
        parts: list[pd.DataFrame] = []
        chosen_meter = args.meter
        collected = 0
        for chunk in pd.read_csv(
            args.input,
            usecols=list(COLUMN_MAP.values()),
            chunksize=100_000,
            low_memory=False,
        ):
            chunk = chunk.rename(columns={v: k for k, v in COLUMN_MAP.items()})
            chunk["meter"] = chunk["meter"].astype("string").str.strip()
            if chosen_meter is None:
                non_null = chunk["meter"].dropna()
                if not non_null.empty:
                    chosen_meter = str(non_null.iloc[0])
            if chosen_meter is not None:
                subset = chunk.loc[chunk["meter"] == chosen_meter].head(args.rows - collected)
                if not subset.empty:
                    parts.append(subset.copy())
                    collected += len(subset)
            if collected >= args.rows:
                break

        if not parts:
            print(f"ERROR: No rows found for meter {chosen_meter!r}.", file=sys.stderr)
            return 1

        df = pd.concat(parts, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in ["energy", "voltage", "current"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp", "energy"]).sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

        if len(df) < 3:
            print("ERROR: Fewer than three valid readings were found; cannot assess the pattern.")
            return 1

        # If t_kWh is interval energy, it should roughly track P(kW) * elapsed hours.
        df["interval_minutes"] = df["timestamp"].diff().dt.total_seconds() / 60.0
        df["estimated_energy_from_VI_kWh"] = (
            (df["voltage"] * df["current"] / 1000.0)
            * (df["interval_minutes"] / 60.0)
        )
        valid = df.loc[
            df["interval_minutes"].between(1, 10)
            & df["estimated_energy_from_VI_kWh"].gt(0)
            & df["energy"].notna()
        ].copy()

        energy_diff = df["energy"].diff().dropna()
        monotonic_fraction = float((energy_diff >= 0).mean()) if not energy_diff.empty else float("nan")

        print("\n========== CEEW t_kWh UNIT DIAGNOSTIC ==========")
        print(f"Input file: {args.input.resolve()}")
        print(f"Meter inspected: {chosen_meter}")
        print(f"Valid readings inspected: {len(df)}")
        print(f"Median timestamp gap (minutes): {df['interval_minutes'].median():.3f}")
        print(f"t_kWh min / median / max: {df['energy'].min():.6f} / "
              f"{df['energy'].median():.6f} / {df['energy'].max():.6f}")
        print(f"Fraction of consecutive t_kWh values that do not decrease: {monotonic_fraction:.3f}")

        if not valid.empty:
            ratio = valid["energy"] / valid["estimated_energy_from_VI_kWh"]
            corr = valid[["energy", "estimated_energy_from_VI_kWh"]].corr().iloc[0, 1]
            print(f"Rows with 1–10 minute gap and valid V/I: {len(valid)}")
            print(f"Correlation: t_kWh vs V*I*time estimate: {corr:.4f}")
            print(f"Median t_kWh / V*I*time estimate: {ratio.median():.4f}")
            print("\nFirst 12 readings (actual vs approximate energy from V and I):")
            print(
                df[["meter", "timestamp", "energy", "voltage", "current",
                    "interval_minutes", "estimated_energy_from_VI_kWh"]]
                .head(12).to_string(index=False)
            )
        else:
            print("Not enough valid short-interval voltage/current rows for the power check.")

        print("\nInterpretation guide:")
        print("- Interval energy: t_kWh varies up/down and tracks V*I*elapsed-time energy.")
        print("- Cumulative register: t_kWh is mostly non-decreasing and its changes, not its raw level, track interval energy.")
        print("- V*I is only an approximate cross-check (power factor and meter measurement details are not included).")
        return 0

    except Exception as exc:  # show useful details in a local diagnostic
        print(f"ERROR while inspecting the CSV: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
