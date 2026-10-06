from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from thermotwin.core import BatteryState
from thermotwin.diagnostics.metrics import (
    ExperimentMetrics,
    calculate_experiment_metrics,
)


def make_state(temperatures_k: list[float]) -> BatteryState:
    cell_count = len(temperatures_k)
    return BatteryState(
        time_s=1.0,
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=300.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=0.0,
        cell_temperatures_k=np.array(temperatures_k),
        cell_resistances_ohm=np.full(cell_count, 0.01),
    )


def test_metrics_calculate_analytical_values_from_post_step_states() -> None:
    states = [make_state([300.0, 302.0]), make_state([304.0, 308.0])]

    metrics = calculate_experiment_metrics(states, [2.0, 3.0], 5.0, 306.0)

    assert metrics == ExperimentMetrics(
        peak_temperature_k=308.0,
        mean_temperature_k=303.5,
        maximum_temperature_spread_k=4.0,
        pump_energy_j=25.0,
        thermal_violation_count=1,
        time_above_thermal_limit_s=5.0,
    )


def test_exact_limit_is_not_a_violation() -> None:
    metrics = calculate_experiment_metrics(
        [make_state([306.0, 305.0])], [0.0], 2.0, 306.0
    )

    assert metrics.thermal_violation_count == 0
    assert metrics.time_above_thermal_limit_s == 0.0


def test_zero_violations_and_pump_power_are_supported() -> None:
    metrics = calculate_experiment_metrics(
        [make_state([290.0]), make_state([295.0])], [0.0, 0.0], 3.0, 300.0
    )

    assert metrics.pump_energy_j == 0.0
    assert metrics.thermal_violation_count == 0
    assert metrics.time_above_thermal_limit_s == 0.0


def test_metrics_are_frozen_and_slotted() -> None:
    metrics = calculate_experiment_metrics([make_state([300.0])], [0.0], 1.0, 310.0)

    with pytest.raises(FrozenInstanceError):
        metrics.peak_temperature_k = 301.0  # type: ignore[misc]
    assert not hasattr(metrics, "__dict__")


def test_calculation_does_not_mutate_battery_state_arrays() -> None:
    state = make_state([300.0, 305.0])
    original_temperatures = state.cell_temperatures_k.copy()
    original_resistances = state.cell_resistances_ohm.copy()

    calculate_experiment_metrics([state], [1.0], 1.0, 310.0)

    np.testing.assert_array_equal(state.cell_temperatures_k, original_temperatures)
    np.testing.assert_array_equal(state.cell_resistances_ohm, original_resistances)


def test_metrics_reject_empty_or_non_sequence_states() -> None:
    with pytest.raises(ValueError):
        calculate_experiment_metrics([], [], 1.0, 310.0)
    with pytest.raises(ValueError):
        calculate_experiment_metrics(iter([make_state([300.0])]), [0.0], 1.0, 310.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("power", [True, "1.0", None, complex(1.0, 0.0)])
def test_metrics_reject_non_real_pump_power(power: Any) -> None:
    with pytest.raises(ValueError):
        calculate_experiment_metrics([make_state([300.0])], [power], 1.0, 310.0)


@pytest.mark.parametrize("power", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_metrics_reject_negative_or_nonfinite_pump_power(power: float) -> None:
    with pytest.raises(ValueError):
        calculate_experiment_metrics([make_state([300.0])], [power], 1.0, 310.0)


def test_metrics_require_one_power_per_state() -> None:
    state = make_state([300.0])

    with pytest.raises(ValueError):
        calculate_experiment_metrics([state, state], [1.0], 1.0, 310.0)
    with pytest.raises(ValueError):
        calculate_experiment_metrics([state], [1.0, 2.0], 1.0, 310.0)


@pytest.mark.parametrize("time_step_s", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_metrics_reject_invalid_fixed_time_step(time_step_s: Any) -> None:
    with pytest.raises(ValueError):
        calculate_experiment_metrics(
            [make_state([300.0])], [0.0], time_step_s, 310.0
        )


@pytest.mark.parametrize(
    "thermal_limit_k", [True, 0.0, -1.0, float("nan"), float("inf")]
)
def test_metrics_reject_invalid_thermal_limit(thermal_limit_k: Any) -> None:
    with pytest.raises(ValueError):
        calculate_experiment_metrics(
            [make_state([300.0])], [0.0], 1.0, thermal_limit_k
        )


def test_metrics_require_consistent_positive_cell_count() -> None:
    with pytest.raises(ValueError, match="same cell count"):
        calculate_experiment_metrics(
            [make_state([300.0]), make_state([300.0, 301.0])],
            [0.0, 0.0],
            1.0,
            310.0,
        )
