"""Behavioral tests for deterministic closed-loop experiment execution."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from thermotwin.control.base import Controller
from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.diagnostics.metrics import calculate_experiment_metrics
from thermotwin.experiments.types import ExperimentScenario
from thermotwin.physics.environment import BatteryEnvironment


def make_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=2,
            cell_mass_kg=0.8,
            cell_specific_heat_capacity_j_kg_k=900.0,
            nominal_cell_resistance_ohm=0.015,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.5,
            ambient_thermal_resistance_k_w=8.0,
            cell_cooling_surface_area_m2=0.02,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.0,
            maximum_coolant_mass_flow_kg_s=0.1,
            base_heat_transfer_coefficient_w_m2_k=5.0,
            flow_heat_transfer_gain=500.0,
            flow_exponent=0.8,
        ),
        simulation=SimulationConfig(
            default_time_step_s=0.5,
            initial_soc_fraction=0.7,
            initial_cell_temperature_k=302.0,
            ambient_temperature_k=300.0,
            coolant_inlet_temperature_k=295.0,
            initial_current_a=0.0,
        ),
    )


def make_scenario(
    *, current_profile: Callable[[float], float] | None = None
) -> ExperimentScenario:
    return ExperimentScenario(
        name="runner-test",
        environment_config=make_environment_config(),
        current_profile=current_profile or (lambda time_s: 10.0 + time_s),
        duration_s=1.5,
        time_step_s=0.5,
        thermal_limit_k=303.0,
        pump_power_coefficient_w_s3_kg3=200.0,
        metadata={"source": "test", "version": 2},
    )


class InstrumentedController:
    def __init__(self, flow: float = 0.04) -> None:
        self.flow = flow
        self.reset_calls = 0
        self.states: list[BatteryState] = []

    def reset(self) -> None:
        self.reset_calls += 1
        self.states.clear()

    def act(self, state: BatteryState) -> CoolingAction:
        self.states.append(state)
        return CoolingAction(self.flow)


class MissingResetController:
    def act(self, state: BatteryState) -> CoolingAction:
        del state
        return CoolingAction(0.0)


def test_runner_executes_exact_steps_and_records_post_step_metrics() -> None:
    from thermotwin.experiments.runner import run_experiment

    scenario = make_scenario()
    controller = InstrumentedController()

    result = run_experiment(scenario, controller, seed=12)

    assert isinstance(controller, Controller)
    assert controller.reset_calls == 1
    assert len(controller.states) == scenario.step_count == 3
    assert [state.time_s for state in controller.states] == [0.0, 0.5, 1.0]
    assert [record.state.time_s for record in result.records] == [0.5, 1.0, 1.5]
    assert all(
        before is not after.state
        for before, after in zip(controller.states, result.records, strict=True)
    )
    assert [record.pump_power_w for record in result.records] == pytest.approx(
        [200.0 * 0.04**3] * 3
    )
    assert result.metrics == calculate_experiment_metrics(
        [record.state for record in result.records],
        [record.pump_power_w for record in result.records],
        scenario.time_step_s,
        scenario.thermal_limit_k,
    )
    assert result.scenario is scenario
    assert result.controller_name == type(controller).__name__
    assert result.seed == 12
    assert isinstance(result.records, tuple)
    assert dict(result.metadata) == dict(scenario.metadata)


@pytest.mark.parametrize(
    ("scenario", "controller", "seed"),
    [
        (object(), InstrumentedController(), 0),
        (make_scenario(), MissingResetController(), 0),
        (make_scenario(), InstrumentedController(), -1),
        (make_scenario(), InstrumentedController(), True),
        (make_scenario(), InstrumentedController(), 1.5),
    ],
)
def test_invalid_inputs_fail_before_controller_reset_or_environment_step(
    monkeypatch: pytest.MonkeyPatch,
    scenario: object,
    controller: Any,
    seed: object,
) -> None:
    from thermotwin.experiments.runner import run_experiment

    reset_calls: list[object] = []
    step_calls: list[object] = []
    original_reset = BatteryEnvironment.reset
    original_step = BatteryEnvironment.step

    def observe_reset(
        self: BatteryEnvironment, config: EnvironmentConfig
    ) -> BatteryState:
        reset_calls.append(config)
        return original_reset(self, config)

    def observe_step(
        self: BatteryEnvironment, action: CoolingAction, dt: float
    ) -> BatteryState:
        step_calls.append((action, dt))
        return original_step(self, action, dt)

    monkeypatch.setattr(BatteryEnvironment, "reset", observe_reset)
    monkeypatch.setattr(BatteryEnvironment, "step", observe_step)

    with pytest.raises((TypeError, ValueError)):
        run_experiment(scenario, controller, seed)  # type: ignore[arg-type]

    assert getattr(controller, "reset_calls", 0) == 0
    assert reset_calls == []
    assert step_calls == []


def test_runner_rejects_non_cooling_action() -> None:
    from thermotwin.experiments.runner import run_experiment

    class BadController(InstrumentedController):
        def act(self, state: BatteryState) -> CoolingAction:
            self.states.append(state)
            return object()  # type: ignore[return-value]

    controller = BadController()
    with pytest.raises(TypeError, match="CoolingAction"):
        run_experiment(make_scenario(), controller, seed=0)
    assert controller.reset_calls == 1
    assert len(controller.states) == 1


def test_repeated_runs_are_deterministic_and_do_not_mutate_global_rngs() -> None:
    from thermotwin.experiments.runner import run_experiment

    scenario = make_scenario()
    py_state = random.getstate()
    np_state = repr(np.random.get_state())

    first = run_experiment(scenario, InstrumentedController(), seed=91)
    second = run_experiment(scenario, InstrumentedController(), seed=91)

    assert random.getstate() == py_state
    assert repr(np.random.get_state()) == np_state
    assert [record.state.time_s for record in first.records] == [
        record.state.time_s for record in second.records
    ]
    np.testing.assert_array_equal(
        [record.state.cell_temperatures_k for record in first.records],
        [record.state.cell_temperatures_k for record in second.records],
    )
    assert first.metrics == second.metrics
