"""Immutable aggregate metrics for a sequence of post-step battery states."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np

from thermotwin.core.types import BatteryState


def _require_sequence(name: str, value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ExperimentMetrics:
    """Aggregate experiment measurements, expressed in SI units."""

    peak_temperature_k: float
    mean_temperature_k: float
    maximum_temperature_spread_k: float
    pump_energy_j: float
    thermal_violation_count: int
    time_above_thermal_limit_s: float


def calculate_experiment_metrics(
    states: Sequence[BatteryState],
    pump_powers_w: Sequence[float],
    time_step_s: float,
    thermal_limit_k: float,
) -> ExperimentMetrics:
    """Calculate deterministic metrics from post-step state observations."""
    state_values = _require_sequence("states", states)
    pump_power_values = _require_sequence("pump_powers_w", pump_powers_w)
    if not state_values:
        raise ValueError("states must be nonempty")
    if len(pump_power_values) != len(state_values):
        raise ValueError("pump_powers_w must contain one value per state")

    time_step = _finite_real("time_step_s", time_step_s)
    if time_step <= 0.0:
        raise ValueError("time_step_s must be positive")
    thermal_limit = _finite_real("thermal_limit_k", thermal_limit_k)
    if thermal_limit <= 0.0:
        raise ValueError("thermal_limit_k must be positive")

    powers: list[float] = []
    for power in pump_power_values:
        normalized_power = _finite_real("pump power", power)
        if normalized_power < 0.0:
            raise ValueError("pump power must be nonnegative")
        powers.append(normalized_power)

    expected_cell_count: int | None = None
    temperatures: list[float] = []
    peak_temperature = -math.inf
    maximum_spread = 0.0
    thermal_violation_count = 0
    for state in state_values:
        if not isinstance(state, BatteryState):
            raise ValueError("states must contain BatteryState values")
        cell_temperatures = state.cell_temperatures_k
        if cell_temperatures.ndim != 1 or cell_temperatures.size == 0:
            raise ValueError("each state must have a positive cell count")
        if expected_cell_count is None:
            expected_cell_count = cell_temperatures.size
        elif cell_temperatures.size != expected_cell_count:
            raise ValueError("states must have the same cell count")
        if not np.all(np.isfinite(cell_temperatures)) or np.any(
            cell_temperatures <= 0.0
        ):
            raise ValueError("cell temperatures must be positive and finite")

        state_minimum = float(np.min(cell_temperatures))
        state_maximum = float(np.max(cell_temperatures))
        peak_temperature = max(peak_temperature, state_maximum)
        maximum_spread = max(maximum_spread, state_maximum - state_minimum)
        if state_maximum > thermal_limit:
            thermal_violation_count += 1
        temperatures.extend(float(value) for value in cell_temperatures)

    temperature_scale = max(temperatures)
    try:
        mean_temperature = temperature_scale * (
            math.fsum(value / temperature_scale for value in temperatures)
            / len(temperatures)
        )
        pump_energy = math.fsum(power * time_step for power in powers)
        time_above_limit = thermal_violation_count * time_step
    except (OverflowError, ValueError) as error:
        raise ValueError("calculated metrics must be finite") from error
    if not all(
        math.isfinite(value)
        for value in (
            peak_temperature,
            mean_temperature,
            maximum_spread,
            pump_energy,
            time_above_limit,
        )
    ):
        raise ValueError("calculated metrics must be finite")

    return ExperimentMetrics(
        peak_temperature_k=peak_temperature,
        mean_temperature_k=mean_temperature,
        maximum_temperature_spread_k=maximum_spread,
        pump_energy_j=pump_energy,
        thermal_violation_count=thermal_violation_count,
        time_above_thermal_limit_s=time_above_limit,
    )
