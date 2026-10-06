"""Behavioral tests for model-agnostic predictive coolant selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from thermotwin.control.base import Controller
from thermotwin.control.predictive import PredictiveController
from thermotwin.control.predictive_types import (
    PredictiveCandidateEvaluation,
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.models.forecaster import (
    ForecastRollout,
    ThermalForecaster,
)


def _state(
    time_s: float = 0.0,
    *,
    flow: float = 0.1,
    temperatures: tuple[float, ...] = (300.0, 301.0),
) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=20.0,
        soc_fraction=0.6,
        ambient_temperature_k=295.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=np.asarray(temperatures, dtype=np.float64),
        cell_resistances_ohm=np.full(len(temperatures), 0.01),
    )


def _copy_state(
    state: BatteryState,
    *,
    current_a: float | None = None,
) -> BatteryState:
    return BatteryState(
        time_s=state.time_s,
        current_a=state.current_a if current_a is None else current_a,
        soc_fraction=state.soc_fraction,
        ambient_temperature_k=state.ambient_temperature_k,
        coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
        coolant_mass_flow_kg_s=state.coolant_mass_flow_kg_s,
        cell_temperatures_k=state.cell_temperatures_k,
        cell_resistances_ohm=state.cell_resistances_ohm,
    )


def _config(
    *,
    horizon_steps: int = 3,
    candidate_count: int = 3,
    minimum_flow: float = 0.1,
    maximum_flow: float = 0.5,
    peak_weight: float = 0.0,
) -> PredictiveControllerConfig:
    return PredictiveControllerConfig(
        horizon_steps=horizon_steps,
        candidate_count=candidate_count,
        minimum_coolant_mass_flow_kg_s=minimum_flow,
        maximum_coolant_mass_flow_kg_s=maximum_flow,
        thermal_safety_limit_k=320.0,
        pump_power_coefficient_w_s3_kg3=0.0,
        objective_weights=PredictiveObjectiveWeights(
            peak_temperature_weight=peak_weight,
            temperature_spread_weight=0.0,
            pump_energy_weight=0.0,
            safety_penalty_weight=0.0,
        ),
    )


class FakeForecaster:
    """Forecast fake exposing only the current state and candidate actions."""

    time_step_s = 1.0
    cell_count = 2

    def __init__(
        self,
        temperature_for_flow: Callable[[float], float] | None = None,
        *,
        copy_initial_state: bool = False,
    ) -> None:
        self.temperature_for_flow = temperature_for_flow or (lambda flow: 310.0)
        self.copy_initial_state = copy_initial_state
        self.calls: list[tuple[BatteryState, tuple[CoolingAction, ...]]] = []

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        return _state(
            state.time_s + self.time_step_s,
            flow=action.coolant_mass_flow_kg_s,
        )

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
    ) -> ForecastRollout:
        self.calls.append((initial_state, actions))
        states = tuple(
            _state(
                initial_state.time_s + (index + 1) * self.time_step_s,
                flow=action.coolant_mass_flow_kg_s,
                temperatures=(
                    self.temperature_for_flow(action.coolant_mass_flow_kg_s),
                    self.temperature_for_flow(action.coolant_mass_flow_kg_s) - 1.0,
                ),
            )
            for index, action in enumerate(actions)
        )
        rollout_initial_state = (
            _copy_state(initial_state) if self.copy_initial_state else initial_state
        )
        return ForecastRollout(rollout_initial_state, actions, states)


def test_candidate_grid_horizon_and_first_action_are_exact() -> None:
    forecaster = FakeForecaster()
    config = _config(
        horizon_steps=4,
        candidate_count=5,
        minimum_flow=0.15,
        maximum_flow=0.55,
    )
    controller = _controller(forecaster, config)
    state = _state()

    action = controller.act(state)

    expected = np.linspace(0.15, 0.55, 5)
    assert len(forecaster.calls) == 5
    assert all(call[0] is state for call in forecaster.calls)
    assert [
        call[1][0].coolant_mass_flow_kg_s for call in forecaster.calls
    ] == pytest.approx(expected)
    assert all(len(actions) == 4 for _, actions in forecaster.calls)
    assert all(
        len({item.coolant_mass_flow_kg_s for item in actions}) == 1
        for _, actions in forecaster.calls
    )
    assert action == CoolingAction(float(expected[0]))
    assert (
        config.minimum_coolant_mass_flow_kg_s
        <= action.coolant_mass_flow_kg_s
        <= config.maximum_coolant_mass_flow_kg_s
    )


def test_minimum_total_cost_candidate_is_selected() -> None:
    forecaster = FakeForecaster(
        lambda flow: {0.1: 315.0, 0.3: 305.0, 0.5: 310.0}[round(flow, 1)]
    )
    controller = _controller(
        forecaster,
        _config(peak_weight=1.0),
    )

    selected = controller.act(_state())

    assert selected.coolant_mass_flow_kg_s == pytest.approx(0.3)
    selected_evaluation = controller.selected_evaluation
    assert selected_evaluation is not None
    assert selected_evaluation.action == selected
    assert selected_evaluation.total_cost == pytest.approx(305.0)


def test_equal_cost_candidates_choose_lower_flow_deterministically() -> None:
    controller = _controller(FakeForecaster(), _config(candidate_count=4))

    selected = controller.act(_state())

    assert selected.coolant_mass_flow_kg_s == pytest.approx(0.1)
    evaluations = controller.last_candidate_evaluations
    assert evaluations is not None
    assert controller.selected_evaluation is evaluations[0]


def test_diagnostics_are_unavailable_before_act() -> None:
    controller = _controller(FakeForecaster(), _config())

    assert controller.last_candidate_evaluations is None
    assert controller.selected_evaluation is None


def test_diagnostics_are_immutable_and_cleared_by_reset() -> None:
    controller = _controller(FakeForecaster(), _config())

    controller.act(_state())
    evaluations = controller.last_candidate_evaluations
    assert evaluations is not None
    assert len(evaluations) == 3
    assert all(isinstance(item, PredictiveCandidateEvaluation) for item in evaluations)
    assert controller.selected_evaluation is not None

    controller.reset()

    assert controller.last_candidate_evaluations is None
    assert controller.selected_evaluation is None


def test_controller_satisfies_shared_controller_protocol() -> None:
    controller = _controller(FakeForecaster(), _config())

    assert isinstance(controller, Controller)
    assert isinstance(FakeForecaster(), ThermalForecaster)


@pytest.mark.parametrize("state", [None, object()])
def test_act_rejects_non_battery_state(state: object) -> None:
    controller = _controller(FakeForecaster(), _config())

    with pytest.raises(TypeError, match="BatteryState"):
        controller.act(cast(BatteryState, state))


def test_constructor_rejects_non_forecaster_and_wrong_config_types() -> None:
    with pytest.raises(TypeError, match="ThermalForecaster"):
        _controller(object(), _config())
    with pytest.raises(TypeError, match="PredictiveControllerConfig"):
        _controller(FakeForecaster(), object())


@pytest.mark.parametrize("cell_count", [0, -1, 2.5, True])
def test_constructor_rejects_invalid_forecaster_cell_count(cell_count: object) -> None:
    forecaster = FakeForecaster()
    forecaster.cell_count = cell_count  # type: ignore[assignment]

    with pytest.raises((TypeError, ValueError), match="cell_count"):
        _controller(forecaster, _config())


def test_constructor_requires_forecaster_and_state_cell_counts_to_match() -> None:
    controller = _controller(FakeForecaster(), _config())

    with pytest.raises(ValueError, match="cell count"):
        controller.act(_state(temperatures=(300.0, 301.0, 302.0)))


def test_rollout_accepts_a_defensive_copy_of_the_requested_state() -> None:
    controller = _controller(
        FakeForecaster(copy_initial_state=True),
        _config(),
    )

    action = controller.act(_state())

    assert action.coolant_mass_flow_kg_s == pytest.approx(0.1)


def test_rollout_rejects_an_initial_state_with_different_scalar_values() -> None:
    class MismatchedInitialStateForecaster(FakeForecaster):
        def rollout(
            self,
            initial_state: BatteryState,
            actions: tuple[CoolingAction, ...],
        ) -> ForecastRollout:
            result = super().rollout(initial_state, actions)
            mismatched = _copy_state(
                initial_state,
                current_a=initial_state.current_a + 1,
            )
            return ForecastRollout(mismatched, result.actions, result.states)

    controller = _controller(MismatchedInitialStateForecaster(), _config())

    with pytest.raises(ValueError, match="initial state"):
        controller.act(_state())


def test_non_rollout_forecaster_result_is_rejected() -> None:
    class InvalidResultForecaster(FakeForecaster):
        def rollout(
            self,
            initial_state: BatteryState,
            actions: tuple[CoolingAction, ...],
        ) -> ForecastRollout:
            del initial_state, actions
            return cast(ForecastRollout, object())

    controller = _controller(InvalidResultForecaster(), _config())

    with pytest.raises(TypeError, match="ForecastRollout"):
        controller.act(_state())


def test_inconsistent_rollout_for_candidate_is_rejected() -> None:
    class InconsistentForecaster(FakeForecaster):
        def rollout(
            self,
            initial_state: BatteryState,
            actions: tuple[CoolingAction, ...],
        ) -> ForecastRollout:
            wrong_actions = tuple(CoolingAction(0.0) for _ in actions)
            states = tuple(
                _state(index + 1, flow=0.0) for index in range(len(actions))
            )
            return ForecastRollout(initial_state, wrong_actions, states)

    controller = _controller(InconsistentForecaster(), _config())

    with pytest.raises(ValueError, match="candidate actions"):
        controller.act(_state())


def test_forecaster_exception_propagates_unchanged() -> None:
    error = RuntimeError("forecast unavailable")

    class RaisingForecaster(FakeForecaster):
        def rollout(
            self,
            initial_state: BatteryState,
            actions: tuple[CoolingAction, ...],
        ) -> ForecastRollout:
            del initial_state, actions
            raise error

    controller = _controller(RaisingForecaster(), _config())

    with pytest.raises(RuntimeError) as raised:
        controller.act(_state())

    assert raised.value is error


def _controller(
    forecaster: object,
    config: object,
) -> PredictiveController:
    return PredictiveController(
        cast(ThermalForecaster, forecaster),
        cast(PredictiveControllerConfig, config),
    )
