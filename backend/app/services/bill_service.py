#!/usr/bin/env python3
"""UPPCL bill estimation service for HomeSolar.

Tariff rates are read from data/raw/tariff/uppcl_tariff.json. No slab or fixed
charge rates are hard-coded in this service.

The service estimates the bill for a billing period (normally approximately
one month). It can consume the grid-import output returned by solar_service.py.
It does NOT model net-metering export credits, taxes, subsidies, rebates, or
charges that are absent from the supplied tariff JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARIFF_PATH = PROJECT_ROOT / "data" / "raw" / "tariff" / "uppcl_tariff.json"


class TariffConfigurationError(ValueError):
    """Raised when a tariff configuration is missing or invalid."""


class IncompleteTariffError(ValueError):
    """Raised when the requested consumption falls in an unspecified tariff band."""


def load_tariff_config(tariff_path: str | Path | None = None) -> dict[str, Any]:
    """Load tariff JSON from the project configuration file."""
    path = Path(tariff_path) if tariff_path is not None else DEFAULT_TARIFF_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"UPPCL tariff JSON not found: {path}. Expected data/raw/tariff/uppcl_tariff.json."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TariffConfigurationError(f"Invalid tariff JSON at {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
        raise TariffConfigurationError("Tariff JSON must contain a top-level categories array.")
    return data


def _get_category(tariff_data: Mapping[str, Any], category_id: str) -> dict[str, Any]:
    for category in tariff_data.get("categories", []):
        if category.get("category_id") == category_id:
            return category
    available = [c.get("category_id") for c in tariff_data.get("categories", [])]
    raise KeyError(f"Unknown tariff category {category_id!r}. Available categories: {available}")


def _validate_nonnegative(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if number < 0:
        raise ValueError(f"{name} cannot be negative.")
    return number


def calculate_energy_charge(consumption_kwh: float, category: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate marginal slab charges and fail on missing tariff bands.

    Slab bounds in the JSON are treated as cumulative kWh thresholds. For
    example, 0-150 and 150-300 means the first 150 units use slab 1, the next
    150 use slab 2, and consumption above 300 uses slab 3.
    """
    units = _validate_nonnegative("consumption_kwh", consumption_kwh)
    eligibility_limit = category.get("eligibility_limit_units")
    if eligibility_limit is not None and units > float(eligibility_limit) + 1e-9:
        raise IncompleteTariffError(
            f"{category['category_id']} applies only up to {eligibility_limit} kWh; "
            f"requested period consumption is {units:.3f} kWh. Check eligibility and tariff rules."
        )

    slabs = category.get("slabs") or []
    if not slabs:
        raise TariffConfigurationError(f"{category['category_id']} has no energy slabs.")
    slabs = sorted(slabs, key=lambda x: float(x["from_units_exclusive"]))

    charge = 0.0
    breakdown: list[dict[str, float | None]] = []
    covered_until = 0.0
    eps = 1e-9

    for slab in slabs:
        lower = float(slab["from_units_exclusive"])
        upper_raw = slab.get("to_units_inclusive")
        upper = float(upper_raw) if upper_raw is not None else None
        rate = float(slab["rate_per_kwh"])

        # Detect missing portions in the tariff schedule only when requested
        # consumption actually reaches that portion.
        if lower > covered_until + eps and units > covered_until + eps:
            gap_end = min(lower, units)
            raise IncompleteTariffError(
                f"Tariff {category['category_id']} does not specify a rate for "
                f"consumption above {covered_until:g} and up to {gap_end:g} kWh. "
                "Update uppcl_tariff.json from the applicable tariff order."
            )

        band_start = max(lower, covered_until)
        band_end = units if upper is None else min(units, upper)
        band_units = max(0.0, band_end - band_start)
        if band_units > eps:
            band_charge = band_units * rate
            charge += band_charge
            breakdown.append({
                "from_units_exclusive": lower,
                "to_units_inclusive": upper,
                "units_charged": round(band_units, 6),
                "rate_per_kwh": rate,
                "amount_inr": round(band_charge, 2),
            })

        if upper is None:
            covered_until = float("inf")
            break
        covered_until = max(covered_until, upper)
        if units <= covered_until + eps:
            break

    if units > covered_until + eps:
        raise IncompleteTariffError(
            f"Tariff {category['category_id']} has no rate for consumption above "
            f"{covered_until:g} kWh. Update uppcl_tariff.json."
        )

    return {
        "energy_charge_inr": round(charge, 2),
        "slab_breakdown": breakdown,
    }


def estimate_electricity_bill(
    grid_energy_kwh: float,
    category_id: str,
    connected_load_kw: float,
    billing_days: float = 30.0,
    tariff_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate bill on grid-import energy for one billing period.

    The slab thresholds are monthly thresholds from the supplied tariff JSON.
    Use this for a roughly monthly energy total (for example, a 30-day forecast).
    `billing_days` only prorates the fixed charge against a 30-day month; it does
    not rescale the energy slab thresholds.
    """
    grid_units = _validate_nonnegative("grid_energy_kwh", grid_energy_kwh)
    load_kw = _validate_nonnegative("connected_load_kw", connected_load_kw)
    days = _validate_nonnegative("billing_days", billing_days)
    if days <= 0:
        raise ValueError("billing_days must be greater than zero.")

    tariff_data = load_tariff_config(tariff_path)
    category = _get_category(tariff_data, category_id)

    # Enforce connected-load eligibility for categories that define it.
    load_condition = category.get("connected_load_condition_kw")
    if isinstance(load_condition, dict):
        operator = load_condition.get("operator")
        threshold = float(load_condition.get("value"))
        if operator == "less_than_or_equal_to" and load_kw > threshold:
            raise ValueError(
                f"{category_id} requires connected load <= {threshold:g} kW; "
                f"received {load_kw:g} kW."
            )
        if operator == "greater_than" and load_kw <= threshold:
            raise ValueError(
                f"{category_id} requires connected load > {threshold:g} kW; "
                f"received {load_kw:g} kW."
            )
        if operator not in {"less_than_or_equal_to", "greater_than"}:
            raise TariffConfigurationError(
                f"Unsupported connected-load condition {operator!r} for {category_id}."
            )

    fixed = category.get("fixed_charge") or {}
    fixed_amount = _validate_nonnegative("fixed charge amount", fixed.get("amount", 0))
    fixed_unit = str(fixed.get("unit", ""))
    if fixed_unit != "INR_per_kW_per_month":
        raise TariffConfigurationError(
            f"Unsupported fixed-charge unit {fixed_unit!r} for {category_id}."
        )

    energy_result = calculate_energy_charge(grid_units, category)
    fixed_charge = fixed_amount * load_kw * (days / 30.0)
    other_charges = 0.0
    warnings: list[str] = []

    if not category.get("data_status", {}).get("complete", False):
        warnings.append(
            "This tariff category is marked incomplete. This estimate is valid only "
            "if the requested consumption falls entirely within supplied rate bands."
        )
    if not tariff_data.get("effective_from"):
        warnings.append("Tariff effective date is not set in uppcl_tariff.json.")
    source_status = tariff_data.get("source", {}).get("verification_status")
    if source_status != "verified":
        warnings.append("Tariff rates have not been marked as officially verified.")
    warnings.append(
        "Other duties, taxes, subsidies, rebates, meter charges and export credits "
        "are excluded because they are not specified in the supplied tariff JSON."
    )
    if not 28 <= days <= 31:
        warnings.append(
            "The supplied tariff slab thresholds are monthly. Applying them to a "
            "billing period far from 28-31 days is an approximation."
        )

    total = energy_result["energy_charge_inr"] + fixed_charge + other_charges
    return {
        "provider": tariff_data.get("provider", "UPPCL"),
        "currency": tariff_data.get("currency", "INR"),
        "category_id": category_id,
        "category_name": category.get("name"),
        "effective_from": tariff_data.get("effective_from"),
        "billing_period_days": days,
        "billing_energy_basis": "grid_import_kWh",
        "grid_energy_kwh": round(grid_units, 6),
        "connected_load_kw": load_kw,
        "energy_charge_inr": energy_result["energy_charge_inr"],
        "slab_breakdown": energy_result["slab_breakdown"],
        "fixed_charge_inr": round(fixed_charge, 2),
        "other_charges_inr": round(other_charges, 2),
        "estimated_total_bill_inr": round(total, 2),
        "tariff_source_verification_status": source_status,
        "warnings": warnings,
        "calculation_note": (
            "Estimated tariff bill on grid-import energy. The period-level solar "
            "offset assumes solar generation can offset consumption within the "
            "same period; it does not model intraday timing, batteries, or net-metering credits."
        ),
    }


def estimate_bill_after_solar(
    solar_estimate: Mapping[str, Any],
    category_id: str,
    connected_load_kw: float,
    billing_days: float = 30.0,
    tariff_path: str | Path | None = None,
) -> dict[str, Any]:
    """Calculate the tariff estimate using `estimated_grid_electricity_kwh`.

    `solar_estimate` is the dictionary returned by solar_service.py.
    """
    if "estimated_grid_electricity_kwh" not in solar_estimate:
        raise ValueError(
            "solar_estimate must contain 'estimated_grid_electricity_kwh'. "
            "Pass the result returned by solar_service.estimate_solar_from_weather()."
        )
    result = estimate_electricity_bill(
        grid_energy_kwh=solar_estimate["estimated_grid_electricity_kwh"],
        category_id=category_id,
        connected_load_kw=connected_load_kw,
        billing_days=billing_days,
        tariff_path=tariff_path,
    )
    result["gross_consumption_kwh"] = solar_estimate.get("consumption_kwh")
    result["estimated_solar_generation_kwh"] = solar_estimate.get("estimated_solar_generation_kwh")
    result["estimated_solar_surplus_kwh"] = solar_estimate.get("estimated_solar_surplus_kwh")
    result["bill_after_solar_estimation"] = True
    return result


__all__ = [
    "TariffConfigurationError",
    "IncompleteTariffError",
    "load_tariff_config",
    "calculate_energy_charge",
    "estimate_electricity_bill",
    "estimate_bill_after_solar",
]
