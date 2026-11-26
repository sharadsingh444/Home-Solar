"""Validate the UPPCL tariff JSON and load it into a SQLAlchemy table.

Default input:
    data/raw/tariff/uppcl_tariff.json

Default database:
    data/interim/tariff/uppcl_tariff.db

You can point this script at the same database used by the backend by setting
DATABASE_URL, or by passing --database-url. Example:
    python etl/load_tariff.py --database-url "sqlite:///data/homesolar.db"

This script stores the tariff configuration. It does NOT calculate electricity
bills. Bill calculations belong in backend/app/services/bill_service.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Column, DateTime, JSON, MetaData, String, Table, create_engine, select, update
from sqlalchemy.engine import Engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARIFF_FILE = PROJECT_ROOT / "data" / "raw" / "tariff" / "uppcl_tariff.json"
DEFAULT_DATABASE_FILE = PROJECT_ROOT / "data" / "interim" / "tariff" / "uppcl_tariff.db"

metadata = MetaData()
tariff_configs = Table(
    "tariff_configs",
    metadata,
    Column("category_id", String(100), primary_key=True),
    Column("provider", String(50), nullable=False),
    Column("tariff_category_code", String(30), nullable=False),
    Column("category_name", String(150), nullable=False),
    Column("location", String(60), nullable=False),
    Column("effective_from", String(30), nullable=True),
    Column("is_complete", String(5), nullable=False),
    Column("config_json", JSON, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate uppcl_tariff.json and load tariff categories into a SQLAlchemy database."
    )
    parser.add_argument(
        "--tariff-file",
        type=Path,
        default=DEFAULT_TARIFF_FILE,
        help=f"Tariff JSON path (default: {DEFAULT_TARIFF_FILE})",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL"),
        help="SQLAlchemy database URL. Defaults to a local SQLite database under data/interim/tariff/.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate and summarize the JSON without writing to a database.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Tariff file not found: {path}\n"
            "Create data/raw/tariff/uppcl_tariff.json or pass --tariff-file with the correct path."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("The top level of the tariff JSON must be an object.")
    return data


def validate_tariff(data: dict[str, Any]) -> list[str]:
    """Validate required fields and return non-fatal warnings."""
    required_top_level = {"provider", "currency", "billing_unit", "categories"}
    missing_top_level = required_top_level - data.keys()
    if missing_top_level:
        raise ValueError(f"Missing top-level fields: {', '.join(sorted(missing_top_level))}")

    if not isinstance(data["categories"], list) or not data["categories"]:
        raise ValueError("'categories' must be a non-empty list.")

    warnings: list[str] = []
    seen_ids: set[str] = set()
    required_category_fields = {
        "category_id",
        "tariff_category_code",
        "name",
        "consumer_type",
        "location",
        "energy_charge_type",
        "slabs",
        "fixed_charge",
        "other_charges",
        "data_status",
    }

    for index, category in enumerate(data["categories"], start=1):
        if not isinstance(category, dict):
            raise ValueError(f"Category #{index} must be a JSON object.")

        missing = required_category_fields - category.keys()
        if missing:
            raise ValueError(
                f"Category #{index} is missing fields: {', '.join(sorted(missing))}"
            )

        category_id = category["category_id"]
        if not isinstance(category_id, str) or not category_id.strip():
            raise ValueError(f"Category #{index} has an empty or invalid category_id.")
        if category_id in seen_ids:
            raise ValueError(f"Duplicate category_id: {category_id}")
        seen_ids.add(category_id)

        slabs = category["slabs"]
        if not isinstance(slabs, list) or not slabs:
            raise ValueError(f"{category_id}: 'slabs' must be a non-empty list.")

        previous_to: int | None = None
        for slab_index, slab in enumerate(slabs, start=1):
            for key in ("from_units_exclusive", "to_units_inclusive", "rate_per_kwh"):
                if key not in slab:
                    raise ValueError(f"{category_id}, slab #{slab_index}: missing '{key}'.")

            lower = slab["from_units_exclusive"]
            upper = slab["to_units_inclusive"]
            rate = slab["rate_per_kwh"]

            if not isinstance(lower, (int, float)) or lower < 0:
                raise ValueError(f"{category_id}, slab #{slab_index}: invalid lower unit bound.")
            if upper is not None and (
                not isinstance(upper, (int, float)) or upper <= lower
            ):
                raise ValueError(f"{category_id}, slab #{slab_index}: invalid upper unit bound.")
            if not isinstance(rate, (int, float)) or rate < 0:
                raise ValueError(f"{category_id}, slab #{slab_index}: rate must be a non-negative number.")

            if previous_to is None and slab_index > 1:
                raise ValueError(
                    f"{category_id}: a slab follows an open-ended slab. Check the slab order."
                )
            if previous_to is not None and lower < previous_to:
                raise ValueError(
                    f"{category_id}: overlapping or unordered slabs near slab #{slab_index}."
                )
            previous_to = upper

        fixed = category["fixed_charge"]
        if not isinstance(fixed, dict) or "amount" not in fixed or "unit" not in fixed:
            raise ValueError(f"{category_id}: fixed_charge must contain amount and unit.")
        if not isinstance(fixed["amount"], (int, float)) or fixed["amount"] < 0:
            raise ValueError(f"{category_id}: fixed charge must be a non-negative number.")

        status = category["data_status"]
        if not isinstance(status, dict) or "complete" not in status:
            raise ValueError(f"{category_id}: data_status.complete is required.")
        if status["complete"] is False:
            ranges = status.get("missing_ranges", [])
            warnings.append(
                f"{category_id} is INCOMPLETE. Missing rate range(s): {ranges}. "
                "Do not use it for a final bill calculation until completed."
            )

    if not data.get("effective_from"):
        warnings.append(
            "No effective_from date is set. Add the effective date from the applicable official tariff order."
        )

    source_status = data.get("source", {}).get("verification_status")
    if source_status != "verified":
        warnings.append(
            "Tariff source is not marked as officially verified. Confirm rates and conditions against the applicable UPPCL tariff order before production use."
        )

    return warnings


def make_engine(database_url: str | None) -> Engine:
    if database_url:
        url = database_url
    else:
        DEFAULT_DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{DEFAULT_DATABASE_FILE.as_posix()}"

    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def load_into_database(data: dict[str, Any], engine: Engine) -> int:
    metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    provider = str(data["provider"])
    effective_from = data.get("effective_from")

    inserted_or_updated = 0
    with engine.begin() as connection:
        for category in data["categories"]:
            category_id = category["category_id"]
            record = {
                "category_id": category_id,
                "provider": provider,
                "tariff_category_code": category["tariff_category_code"],
                "category_name": category["name"],
                "location": category["location"],
                "effective_from": effective_from,
                "is_complete": str(bool(category["data_status"].get("complete"))).lower(),
                "config_json": category,
                "updated_at": now,
            }

            existing = connection.execute(
                select(tariff_configs.c.category_id).where(
                    tariff_configs.c.category_id == category_id
                )
            ).first()

            if existing:
                connection.execute(
                    update(tariff_configs)
                    .where(tariff_configs.c.category_id == category_id)
                    .values(**record)
                )
            else:
                connection.execute(tariff_configs.insert().values(**record))
            inserted_or_updated += 1

    return inserted_or_updated


def print_summary(data: dict[str, Any], warnings: list[str]) -> None:
    print("\nUPPCL tariff configuration")
    print("-" * 60)
    print(f"Provider       : {data['provider']}")
    print(f"Currency       : {data['currency']}")
    print(f"Effective from : {data.get('effective_from') or 'NOT SET'}")
    print(f"Categories     : {len(data['categories'])}")

    for category in data["categories"]:
        status = "complete" if category["data_status"].get("complete") else "INCOMPLETE"
        print(
            f"  - {category['category_id']}: {category['name']} "
            f"[{status}; {len(category['slabs'])} rate band(s)]"
        )

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("\nValidation passed with no warnings.")


def main() -> int:
    args = parse_args()
    try:
        tariff_data = load_json(args.tariff_file)
        warnings = validate_tariff(tariff_data)
        print_summary(tariff_data, warnings)

        if args.check_only:
            print("\nCheck-only mode: no database changes made.")
            return 0

        engine = make_engine(args.database_url)
        count = load_into_database(tariff_data, engine)
        print(f"\nLoaded {count} tariff category record(s) into table 'tariff_configs'.")
        if args.database_url:
            print("Database: URL supplied by --database-url or DATABASE_URL.")
        else:
            print(f"Database: {DEFAULT_DATABASE_FILE}")
        engine.dispose()
        return 0

    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # Display actionable database/driver errors.
        print(f"ERROR while loading tariff: {exc}", file=sys.stderr)
        print(
            "Check that SQLAlchemy and the database driver are installed, and that the database URL is correct.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
