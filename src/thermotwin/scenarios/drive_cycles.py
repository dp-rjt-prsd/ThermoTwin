"""Pure current-profile factories for synthetic battery driving scenarios."""

from __future__ import annotations

import math
from numbers import Real

from thermotwin.core.types import CurrentProfile


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _positive_duration(name: str, value: object) -> float:
    result = _finite_number(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_peak_current(value: object) -> float:
    result = _finite_number("peak_discharge_current_a", value)
    if result <= 0.0:
        raise ValueError("peak_discharge_current_a must be positive")
    return result


def _query_time(time_s: object) -> float:
    result = _finite_number("time_s", time_s)
    if result < 0.0:
        raise ValueError("time_s must be nonnegative")
    return result


def constant_current_profile(
    current_a: float, total_duration_s: float
) -> CurrentProfile:
    """Return a constant current until total_duration_s, then zero."""
    current = _finite_number("current_a", current_a)
    duration = _positive_duration("total_duration_s", total_duration_s)

    def profile(time_s: float) -> float:
        time = _query_time(time_s)
        return current if time < duration else 0.0

    return profile


def pulse_current_profile(
    base_current_a: float,
    pulse_current_a: float,
    pulse_start_s: float,
    pulse_duration_s: float,
    total_duration_s: float,
) -> CurrentProfile:
    """Return a base current with one half-open pulse interval."""
    base_current = _finite_number("base_current_a", base_current_a)
    pulse_current = _finite_number("pulse_current_a", pulse_current_a)
    pulse_start = _finite_number("pulse_start_s", pulse_start_s)
    pulse_duration = _positive_duration("pulse_duration_s", pulse_duration_s)
    total_duration = _positive_duration("total_duration_s", total_duration_s)
    if pulse_start < 0.0:
        raise ValueError("pulse_start_s must be nonnegative")
    if pulse_start + pulse_duration > total_duration:
        raise ValueError("pulse must end at or before total_duration_s")

    pulse_end = pulse_start + pulse_duration

    def profile(time_s: float) -> float:
        time = _query_time(time_s)
        if time >= total_duration:
            return 0.0
        if pulse_start <= time < pulse_end:
            return pulse_current
        return base_current

    return profile


def stop_and_go_profile(
    peak_discharge_current_a: float,
    cycle_duration_s: float,
    total_duration_s: float,
) -> CurrentProfile:
    """Return repeated idle, acceleration, cruise, and braking stages."""
    peak_current = _positive_peak_current(peak_discharge_current_a)
    cycle_duration = _positive_duration("cycle_duration_s", cycle_duration_s)
    total_duration = _positive_duration("total_duration_s", total_duration_s)
    if cycle_duration > total_duration:
        raise ValueError("cycle_duration_s must not exceed total_duration_s")

    stage_currents = (0.0, peak_current, 0.4 * peak_current, -0.25 * peak_current)

    def profile(time_s: float) -> float:
        time = _query_time(time_s)
        if time >= total_duration:
            return 0.0
        cycle_fraction = (time % cycle_duration) / cycle_duration
        stage = min(int(cycle_fraction * 4.0), 3)
        return stage_currents[stage]

    return profile


def aggressive_drive_profile(
    peak_discharge_current_a: float,
    cycle_duration_s: float,
    total_duration_s: float,
) -> CurrentProfile:
    """Return repeated high-load stages followed by regenerative braking."""
    peak_current = _positive_peak_current(peak_discharge_current_a)
    cycle_duration = _positive_duration("cycle_duration_s", cycle_duration_s)
    total_duration = _positive_duration("total_duration_s", total_duration_s)
    if cycle_duration > total_duration:
        raise ValueError("cycle_duration_s must not exceed total_duration_s")

    def profile(time_s: float) -> float:
        time = _query_time(time_s)
        if time >= total_duration:
            return 0.0
        cycle_time = time % cycle_duration
        fraction = cycle_time / cycle_duration
        if fraction < 0.35:
            return peak_current
        if fraction < 0.60:
            return 0.75 * peak_current
        if fraction < 0.80:
            return 0.9 * peak_current
        return -0.35 * peak_current

    return profile
