"""Pure current profiles for synthetic battery charging scenarios."""

from __future__ import annotations

import math
from numbers import Real

from thermotwin.core.types import CurrentProfile


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def fast_charging_profile(
    max_charge_current_magnitude_a: float,
    total_duration_s: float,
    taper_start_fraction: float = 0.7,
    final_current_fraction: float = 0.2,
) -> CurrentProfile:
    """Create a constant-current charge profile with a linear current taper.

    Charging current follows the project's negative-current convention. The
    profile is zero at and after ``total_duration_s``.
    """
    maximum_current_a = _finite_real(
        "max_charge_current_magnitude_a", max_charge_current_magnitude_a
    )
    duration_s = _finite_real("total_duration_s", total_duration_s)
    taper_start = _finite_real("taper_start_fraction", taper_start_fraction)
    final_fraction = _finite_real("final_current_fraction", final_current_fraction)

    if maximum_current_a <= 0.0:
        raise ValueError("max_charge_current_magnitude_a must be positive")
    if duration_s <= 0.0:
        raise ValueError("total_duration_s must be positive")
    if not 0.0 < taper_start < 1.0:
        raise ValueError("taper_start_fraction must be strictly between 0 and 1")
    if not 0.0 <= final_fraction <= 1.0:
        raise ValueError("final_current_fraction must be in [0, 1]")

    taper_start_s = duration_s * taper_start

    def current_at(time_s: float) -> float:
        elapsed_s = _finite_real("time_s", time_s)
        if elapsed_s < 0.0:
            raise ValueError("time_s must be nonnegative")
        if elapsed_s >= duration_s:
            return 0.0
        if elapsed_s < taper_start_s:
            return -maximum_current_a

        taper_progress = (elapsed_s - taper_start_s) / (duration_s - taper_start_s)
        current_fraction = 1.0 - (1.0 - final_fraction) * taper_progress
        return -maximum_current_a * current_fraction

    return current_at
