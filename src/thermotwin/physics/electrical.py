"""Pure electrical state updates using SI units."""

from __future__ import annotations

import math
from numbers import Real


def _finite_scalar(name: str, value: object) -> float:
    """Convert a real scalar to float, rejecting booleans and nonfinite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def soc_after_coulomb_counting(
    soc_fraction: float,
    current_a: float,
    cell_capacity_ah: float,
    dt_s: float,
) -> float:
    """Return SOC after one signed-current coulomb-counting time step.

    Positive current discharges the cell; negative current charges it. The
    resulting fraction is bounded to the physically valid inclusive range.
    """
    soc = _finite_scalar("soc_fraction", soc_fraction)
    current = _finite_scalar("current_a", current_a)
    capacity = _finite_scalar("cell_capacity_ah", cell_capacity_ah)
    dt = _finite_scalar("dt_s", dt_s)

    if not 0.0 <= soc <= 1.0:
        raise ValueError("soc_fraction must be in [0, 1]")
    if capacity <= 0.0:
        raise ValueError("cell_capacity_ah must be positive")
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")

    soc_next = soc - current * dt / (3600.0 * capacity)
    return min(1.0, max(0.0, soc_next))
