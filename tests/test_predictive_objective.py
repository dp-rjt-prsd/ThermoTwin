from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from thermotwin.control.objective import evaluate_predictive_candidate
from thermotwin.control.predictive_types import (
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.models.forecaster import ForecastRollout


def _config(
    *,
    horizon_steps: int = 2,
    peak: float = 0.0,
    spread: float = 0.0,
    pump: float = 0.0,
    safety: float = 0.0,
    limit: float = 310.0,
    pump_coefficient: float = 0.0,
) -> PredictiveControllerConfig:
    return PredictiveControllerConfig(
        horizon_steps=horizon_steps,
        candidate_count=2,
        minimum_coolant_mass_flow_kg_s=0.0,
        maximum_coolant_mass_flow_kg_s=1.0,
        thermal_safety_limit_k=limit,
        pump_power_coefficient_w_s3_kg3=pump_coefficient,
        objective_weights=PredictiveObjectiveWeights(peak, spread, pump, safety),
    )


def _state(time_s: float, temperatures: tuple[float, ...], flow: float) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=1.0,
        soc_fraction=0.5,
        ambient_temperature_k=300.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=np.array(temperatures),
        cell_resistances_ohm=np.ones(len(temperatures)),
    )


def _rollout(
    temperatures: tuple[tuple[float, ...], ...] = ((301.0, 303.0), (304.0, 302.0)),
    flows: tuple[float, ...] = (0.1, 0.2),
    *,
    time_step_s: float = 2.0,
) -> ForecastRollout:
    initial = _state(0.0, (300.0, 300.0), 0.0)
    actions = tuple(CoolingAction(flow) for flow in flows)
    states = tuple(
        _state((index + 1) * time_step_s, values, flow)
        for index, (values, flow) in enumerate(zip(temperatures, flows, strict=True))
    )
    return ForecastRollout(initial, actions, states)


def test_peak_component_uses_maximum_cell_temperature_across_horizon() -> None:
    result = evaluate_predictive_candidate(_rollout(), _config(peak=3.0))

    assert result.peak_temperature_cost == pytest.approx(3.0 * 304.0)
    assert result.total_cost == result.peak_temperature_cost


def test_spread_component_uses_worst_per_state_spread() -> None:
    rollout = _rollout(((301.0, 308.0), (320.0, 310.0)))

    result = evaluate_predictive_candidate(rollout, _config(spread=2.5))

    assert result.temperature_spread_cost == pytest.approx(25.0)
    assert result.total_cost == result.temperature_spread_cost


def test_pump_component_sums_energy_for_every_action_and_step() -> None:
    result = evaluate_predictive_candidate(
        _rollout(flows=(0.1, 0.2), time_step_s=2.0),
        _config(pump=4.0, pump_coefficient=100.0),
    )

    expected_energy = (100.0 * 0.1**3 * 2.0) + (100.0 * 0.2**3 * 2.0)
    assert result.pump_energy_cost == pytest.approx(expected_energy * 4.0)
    assert result.total_cost == result.pump_energy_cost


def test_safety_component_sums_squared_exceedances_over_steps_and_cells() -> None:
    rollout = _rollout(((312.0, 308.0), (315.0, 311.0)))

    result = evaluate_predictive_candidate(
        rollout, _config(safety=0.5, limit=310.0)
    )

    assert result.safety_penalty_cost == pytest.approx(0.5 * (2.0**2 + 5.0**2 + 1.0**2))
    assert result.total_cost == result.safety_penalty_cost


def test_safety_cost_is_zero_when_no_predicted_cell_exceeds_limit() -> None:
    result = evaluate_predictive_candidate(
        _rollout(((309.0, 310.0), (300.0, 305.0))), _config(safety=9.0)
    )

    assert result.safety_penalty_cost == 0.0


def test_total_is_sum_of_weighted_components_and_retains_first_action() -> None:
    rollout = _rollout()
    result = evaluate_predictive_candidate(
        rollout,
        _config(peak=1.0, spread=2.0, pump=3.0, safety=4.0, limit=302.0,
                pump_coefficient=100.0),
    )

    assert result.action is rollout.actions[0]
    assert result.total_cost == pytest.approx(
        result.peak_temperature_cost
        + result.temperature_spread_cost
        + result.pump_energy_cost
        + result.safety_penalty_cost
    )


def test_higher_weights_do_not_reduce_their_weighted_component() -> None:
    rollout = _rollout(((312.0, 302.0), (304.0, 302.0)))

    lower = evaluate_predictive_candidate(
        rollout, _config(peak=1.0, spread=1.0, pump=1.0, safety=1.0, limit=310.0,
                         pump_coefficient=10.0)
    )
    higher = evaluate_predictive_candidate(
        rollout, _config(peak=2.0, spread=2.0, pump=2.0, safety=2.0, limit=310.0,
                         pump_coefficient=10.0)
    )

    assert higher.peak_temperature_cost >= lower.peak_temperature_cost
    assert higher.temperature_spread_cost >= lower.temperature_spread_cost
    assert higher.pump_energy_cost >= lower.pump_energy_cost
    assert higher.safety_penalty_cost >= lower.safety_penalty_cost


@pytest.mark.parametrize("rollout,config", [(None, _config()), (_rollout(), None)])
def test_evaluator_rejects_wrong_contract_types(
    rollout: object, config: object
) -> None:
    with pytest.raises(TypeError):
        evaluate_predictive_candidate(
            cast(ForecastRollout, rollout), cast(PredictiveControllerConfig, config)
        )


def test_evaluator_rejects_rollout_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="horizon"):
        evaluate_predictive_candidate(_rollout(), _config(horizon_steps=3))


def test_evaluator_rejects_overflow_in_weighted_cost() -> None:
    rollout = _rollout(((1e200, 300.0), (300.0, 300.0)))

    with pytest.raises(ValueError, match="finite"):
        evaluate_predictive_candidate(rollout, _config(peak=1e200))


def test_evaluator_rejects_overflow_while_summing_pump_energy() -> None:
    rollout = _rollout(flows=(1e100, 1e100), time_step_s=1e100)

    with pytest.raises(ValueError, match="finite"):
        evaluate_predictive_candidate(
            rollout, _config(pump=1.0, pump_coefficient=1e100)
        )
