"""Tests for the immutable model-agnostic forecasting contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from thermotwin.core.types import BatteryState, CoolingAction
from thermotwin.models.forecaster import (
    ForecastConditions,
    ForecastRollout,
    ThermalForecaster,
)


def _state(
    time_s: float,
    *,
    flow: float = 0.1,
    cell_count: int = 12,
    temperatures: np.ndarray | None = None,
) -> BatteryState:
    if temperatures is None:
        temperatures = np.full(cell_count, 300.0)
    return BatteryState(
        time_s=time_s,
        current_a=-30.0,
        soc_fraction=0.7,
        ambient_temperature_k=295.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=np.full(cell_count, 0.01),
    )


def _rollout() -> ForecastRollout:
    initial = _state(0.0)
    actions = (CoolingAction(0.2), CoolingAction(0.3))
    states = (_state(1.0, flow=0.2), _state(2.0, flow=0.3))
    return ForecastRollout(initial, actions, states)


def test_valid_twelve_cell_conditions_and_rollout_contracts() -> None:
    initial = _state(0.0)
    conditions = ForecastConditions.from_state(initial)
    result = _rollout()

    assert conditions == ForecastConditions(-30.0, 0.7, 295.0, 290.0)
    assert result.time_step_s == 1.0
    assert result.step_count == 2
    assert result.cell_count == 12
    assert result.cell_temperatures_k.shape == (2, 12)
    np.testing.assert_array_equal(
        result.cell_temperatures_k, np.full((2, 12), 300.0)
    )


def test_forecast_conditions_validate_scalar_boundaries() -> None:
    for current in (float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            ForecastConditions(current, 0.5, 295.0, 290.0)  # type: ignore[arg-type]
    for soc in (-0.01, 1.01, float("nan"), True):
        with pytest.raises(ValueError):
            ForecastConditions(0.0, soc, 295.0, 290.0)
    for temperature in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            ForecastConditions(0.0, 0.5, temperature, 290.0)
        with pytest.raises(ValueError):
            ForecastConditions(0.0, 0.5, 295.0, temperature)


def test_conditions_are_frozen_and_from_state_copies_values() -> None:
    source = _state(0.0)
    conditions = ForecastConditions.from_state(source)

    assert conditions.current_a == source.current_a
    with pytest.raises(FrozenInstanceError):
        conditions.current_a = 0.0  # type: ignore[misc]


def test_battery_state_takes_defensive_ownership_of_cell_arrays() -> None:
    temperatures = np.full(12, 300.0)
    resistances = np.full(12, 0.01)
    state = BatteryState(
        time_s=0.0,
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=295.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=0.0,
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=resistances,
    )

    temperatures[0] = 999.0
    resistances[0] = 9.0
    assert state.cell_temperatures_k[0] == 300.0
    assert state.cell_resistances_ohm[0] == 0.01
    with pytest.raises(ValueError):
        state.cell_temperatures_k.setflags(write=True)


def test_rollout_validates_container_types_and_matching_lengths() -> None:
    initial = _state(0.0)
    with pytest.raises(ValueError):
        ForecastRollout(initial, (), ())
    with pytest.raises(ValueError):
        ForecastRollout(
            initial,
            [CoolingAction(0.1)],  # type: ignore[arg-type]
            (_state(1.0, flow=0.1),),
        )
    with pytest.raises(ValueError):
        ForecastRollout(initial, (CoolingAction(0.1),), ())
    with pytest.raises(ValueError):
        ForecastRollout(
            initial,
            (CoolingAction(0.1),),
            (_state(1.0, flow=0.1, cell_count=11),),
        )


def test_rollout_validates_action_flow_and_post_step_times() -> None:
    initial = _state(0.0)
    action = CoolingAction(0.2)
    with pytest.raises(ValueError, match="flow"):
        ForecastRollout(initial, (action,), (_state(1.0, flow=0.3),))
    with pytest.raises(ValueError, match="time"):
        ForecastRollout(initial, (action,), (_state(0.0, flow=0.2),))
    with pytest.raises(ValueError, match="time"):
        ForecastRollout(
            initial,
            (action, action),
            (_state(1.0, flow=0.2), _state(2.1, flow=0.2)),
        )


def test_temperature_matrix_is_defensive_read_only_float64() -> None:
    result = _rollout()
    matrix = result.cell_temperatures_k

    assert matrix.dtype == np.float64
    assert not matrix.flags.writeable
    with pytest.raises(ValueError):
        matrix[0, 0] = 0.0
    with pytest.raises(ValueError):
        matrix.setflags(write=True)


def test_runtime_protocol_accepts_minimal_forecaster() -> None:
    class FakeForecaster:
        time_step_s = 1.0
        cell_count = 12

        def predict_next(
            self, state: BatteryState, action: CoolingAction
        ) -> BatteryState:
            return _state(state.time_s + 1.0, flow=action.coolant_mass_flow_kg_s)

        def rollout(
            self,
            initial_state: BatteryState,
            actions: tuple[CoolingAction, ...],
            future_conditions: tuple[ForecastConditions, ...] | None = None,
        ) -> ForecastRollout:
            del future_conditions
            states = tuple(
                _state(
                    initial_state.time_s + (index + 1),
                    flow=action.coolant_mass_flow_kg_s,
                )
                for index, action in enumerate(actions)
            )
            return ForecastRollout(initial_state, actions, states)

    assert isinstance(FakeForecaster(), ThermalForecaster)
