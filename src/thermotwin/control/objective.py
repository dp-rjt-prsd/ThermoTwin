"""Pump energy approximations used to account for cooling effort.

The configurable cubic flow relationship is an MVP approximation, not a
manufacturer-specific pump curve.
"""

from __future__ import annotations

import math
from numbers import Real


def _require_nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def pump_power_w(
    coolant_mass_flow_kg_s: float,
    pump_power_coefficient_w_s3_kg3: float,
) -> float:
    """Estimate pump power as ``coefficient * mass_flow**3`` in watts.

    The coefficient is configurable; this cubic relationship is an MVP
    approximation and does not represent a manufacturer-specific pump curve.
    """
    flow = _require_nonnegative_finite(
        "coolant_mass_flow_kg_s", coolant_mass_flow_kg_s
    )
    coefficient = _require_nonnegative_finite(
        "pump_power_coefficient_w_s3_kg3", pump_power_coefficient_w_s3_kg3
    )
    try:
        power = coefficient * flow**3
    except OverflowError as error:
        raise ValueError("pump power result must be finite") from error
    if not math.isfinite(power):
        raise ValueError("pump power result must be finite")
    return power


def pump_energy_j(
    coolant_mass_flow_kg_s: float,
    duration_s: float,
    pump_power_coefficient_w_s3_kg3: float,
) -> float:
    """Estimate pump energy in joules over a constant-flow interval.

    Energy is pump power multiplied by the interval duration. Power follows a
    configurable MVP cubic approximation, not a manufacturer pump curve.
    """
    duration = _require_nonnegative_finite("duration_s", duration_s)
    power = pump_power_w(
        coolant_mass_flow_kg_s, pump_power_coefficient_w_s3_kg3
    )
    energy = power * duration
    if not math.isfinite(energy):
        raise ValueError("pump energy result must be finite")
    return energy
