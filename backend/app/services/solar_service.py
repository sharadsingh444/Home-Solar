"""Rule-based solar generation and grid-energy estimation for HomeSolar.

The project proposal specifies estimated solar production from sunlight and
panel details. There are no measured solar-panel output readings in the CEEW
household dataset, so this service returns an estimate rather than a trained
solar-output prediction.

Expected hourly irradiance units: W/m^2 averaged over each interval.
For hourly data, sum(W/m^2 * 1 hour) / 1000 = kWh/m^2, numerically equivalent
to peak-sun-hours for the period. Estimated AC energy is:
    panel_capacity_kW * peak_sun_hours * performance_ratio
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SolarEstimateConfig:
    """Central solar-system assumptions; override via args or environment."""
    performance_ratio: float = float(os.getenv("HOMESOLAR_SOLAR_PERFORMANCE_RATIO", "0.80"))
    default_irradiance_column: str = "shortwave_radiation"
    irradiance_unit: str = "W/m2"
    interval_hours: float = 1.0

    def validate(self) -> None:
        if not 0 < self.performance_ratio <= 1:
            raise ValueError("performance_ratio must be greater than 0 and at most 1.")
        if self.interval_hours <= 0:
            raise ValueError("interval_hours must be greater than zero.")
        if self.irradiance_unit not in {"W/m2", "kW/m2"}:
            raise ValueError("irradiance_unit must be 'W/m2' or 'kW/m2'.")


DEFAULT_SOLAR_CONFIG = SolarEstimateConfig()


def estimate_solar_from_peak_sun_hours(
    consumption_kwh: float,
    panel_capacity_kw: float,
    peak_sun_hours: float,
    performance_ratio: float | None = None,
) -> dict[str, Any]:
    """Estimate solar generation and grid requirement for the same time period."""
    pr = DEFAULT_SOLAR_CONFIG.performance_ratio if performance_ratio is None else float(performance_ratio)
    _validate_inputs(consumption_kwh, panel_capacity_kw, pr)
    if peak_sun_hours < 0:
        raise ValueError("peak_sun_hours cannot be negative.")

    production = float(panel_capacity_kw) * float(peak_sun_hours) * pr
    consumption = float(consumption_kwh)
    solar_used = min(consumption, production)
    grid_import = max(0.0, consumption - production)
    surplus = max(0.0, production - consumption)
    return {
        "consumption_kwh": consumption,
        "panel_capacity_kw": float(panel_capacity_kw),
        "peak_sun_hours": float(peak_sun_hours),
        "performance_ratio": pr,
        "estimated_solar_generation_kwh": production,
        "estimated_solar_used_kwh": solar_used,
        "estimated_grid_electricity_kwh": grid_import,
        "estimated_solar_surplus_kwh": surplus,
        "assumption": "Period-level energy balance; does not model intraday load timing, battery storage, export compensation, shading, or panel degradation.",
    }


def estimate_solar_from_weather(
    consumption_kwh: float,
    panel_capacity_kw: float,
    hourly_weather: pd.DataFrame,
    performance_ratio: float | None = None,
    irradiance_column: str | None = None,
    interval_hours: float | None = None,
    irradiance_unit: str | None = None,
    location_id: str | None = None,
) -> dict[str, Any]:
    """Integrate hourly irradiance observations and estimate solar electricity.

    Uses shortwave_radiation (GHI) by default. You may pass
    irradiance_column="global_tilted_irradiance" when GTI was generated for the
    actual panel tilt and azimuth. Open-Meteo defaults GTI to 0-degree tilt
    (horizontal), so do not assume an unconfigured GTI represents a rooftop panel.
    """
    if not isinstance(hourly_weather, pd.DataFrame) or hourly_weather.empty:
        raise ValueError("hourly_weather must be a non-empty pandas DataFrame.")
    cfg = DEFAULT_SOLAR_CONFIG
    weather = hourly_weather.copy()
    # Weather input may contain all 36 Bareilly grid locations. Never sum
    # radiation across multiple locations: require the caller to select one.
    location_col = "location_id" if "location_id" in weather.columns else ("meter" if "meter" in weather.columns else None)
    selected_location = None
    if location_col is not None:
        locations = weather[location_col].dropna().astype(str).unique().tolist()
        if location_id is not None:
            weather = weather.loc[weather[location_col].astype(str) == str(location_id)].copy()
            if weather.empty:
                raise ValueError(f"No weather rows found for location_id={location_id!r}.")
            selected_location = str(location_id)
        elif len(locations) > 1:
            raise ValueError(
                f"Weather data contains multiple locations ({len(locations)}). "
                "Pass location_id to select exactly one location before estimating solar generation."
            )
        elif locations:
            selected_location = locations[0]
    col = irradiance_column or cfg.default_irradiance_column
    if col not in weather.columns:
        if irradiance_column is None and "shortwave_radiation" in hourly_weather.columns:
            col = "shortwave_radiation"
        else:
            raise ValueError(f"Irradiance column '{col}' is missing from weather data.")
    hours = cfg.interval_hours if interval_hours is None else float(interval_hours)
    unit = cfg.irradiance_unit if irradiance_unit is None else irradiance_unit
    if hours <= 0:
        raise ValueError("interval_hours must be greater than zero.")
    if unit not in {"W/m2", "kW/m2"}:
        raise ValueError("irradiance_unit must be 'W/m2' or 'kW/m2'.")

    irradiance = pd.to_numeric(weather[col], errors="coerce").dropna()
    if irradiance.empty:
        raise ValueError(f"No valid irradiance values found in '{col}'.")
    if (irradiance < 0).any():
        raise ValueError(f"Negative irradiance values found in '{col}'. Clean the weather input first.")
    # kWh/m^2 over the requested window. For hourly mean W/m^2, multiply by
    # interval hours and divide by 1000; for kW/m^2, do not divide by 1000.
    scale = 1.0 / 1000.0 if unit == "W/m2" else 1.0
    irradiation_kwh_m2 = float((irradiance * hours * scale).sum())
    result = estimate_solar_from_peak_sun_hours(
        consumption_kwh=consumption_kwh,
        panel_capacity_kw=panel_capacity_kw,
        peak_sun_hours=irradiation_kwh_m2,
        performance_ratio=performance_ratio,
    )
    result.update({
        "irradiance_column": col,
        "irradiance_unit": unit,
        "interval_hours": hours,
        "irradiance_observations_used": int(len(irradiance)),
        "estimated_irradiation_kwh_per_m2": irradiation_kwh_m2,
        "weather_location_id": selected_location,
        "note": "Solar energy is estimated from irradiance and panel capacity; it is not measured PV output.",
    })
    return result


def _validate_inputs(consumption_kwh: float, panel_capacity_kw: float, performance_ratio: float) -> None:
    if consumption_kwh < 0:
        raise ValueError("consumption_kwh cannot be negative.")
    if panel_capacity_kw < 0:
        raise ValueError("panel_capacity_kw cannot be negative.")
    if not 0 < performance_ratio <= 1:
        raise ValueError("performance_ratio must be greater than 0 and at most 1.")


__all__ = [
    "SolarEstimateConfig",
    "DEFAULT_SOLAR_CONFIG",
    "estimate_solar_from_peak_sun_hours",
    "estimate_solar_from_weather",
]
