"""Deterministic closed-loop execution for immutable experiment scenarios."""

from __future__ import annotations

from numbers import Integral

from thermotwin.control.base import Controller
from thermotwin.control.objective import pump_power_w
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.diagnostics.metrics import calculate_experiment_metrics
from thermotwin.experiments.types import (
    ExperimentRecord,
    ExperimentResult,
    ExperimentScenario,
)
from thermotwin.physics.environment import BatteryEnvironment


def _normalize_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def run_experiment(
    scenario: ExperimentScenario,
    controller: Controller,
    seed: int,
) -> ExperimentResult:
    """Run one controller on a fresh authoritative thermal environment.

    The seed is retained as experiment provenance. Scenario factories own any
    stochastic setup and seeding; executing a fixed scenario does not modify
    process-global random state.
    """
    if type(scenario) is not ExperimentScenario:
        raise ValueError("scenario must be exactly an ExperimentScenario")
    if not isinstance(controller, Controller):
        raise ValueError("controller must implement the Controller protocol")
    normalized_seed = _normalize_seed(seed)
    environment = BatteryEnvironment(scenario.current_profile)
    controller.reset()
    current_state = environment.reset(scenario.environment_config)

    records: list[ExperimentRecord] = []
    states: list[BatteryState] = []
    pump_powers_w: list[float] = []
    for _ in range(scenario.step_count):
        action = controller.act(current_state)
        if not isinstance(action, CoolingAction):
            raise TypeError("controller.act must return a CoolingAction")

        current_state = environment.step(action, scenario.time_step_s)
        power = pump_power_w(
            action.coolant_mass_flow_kg_s,
            scenario.pump_power_coefficient_w_s3_kg3,
        )
        records.append(ExperimentRecord(state=current_state, pump_power_w=power))
        states.append(current_state)
        pump_powers_w.append(power)

    metrics = calculate_experiment_metrics(
        states,
        pump_powers_w,
        scenario.time_step_s,
        scenario.thermal_limit_k,
    )
    return ExperimentResult(
        scenario=scenario,
        controller_name=type(controller).__name__,
        seed=normalized_seed,
        records=tuple(records),
        metrics=metrics,
        metadata=scenario.metadata,
    )
