"""Pump energy approximations used to account for cooling effort.

The configurable cubic flow relationship is an MVP approximation, not a
manufacturer-specific pump curve.
"""

from __future__ import annotations

import math
from numbers import Real

from thermotwin.control.predictive_types import (
    PredictiveCandidateEvaluation,
    PredictiveControllerConfig,
)
from thermotwin.models.forecaster import ForecastRollout


def evaluate_predictive_candidate(
    rollout: ForecastRollout,
    config: PredictiveControllerConfig,
) -> PredictiveCandidateEvaluation:
    """Calculate weighted horizon costs for one candidate rollout.

    Peak temperature and spread use the worst predicted state in the rollout.
    Pump energy and safety exceedance accumulate over every predicted step.
    This function only scores the rollout; it does not select or optimize actions.
    """
    if type(rollout) is not ForecastRollout:
        raise TypeError("rollout must be a ForecastRollout")
    if type(config) is not PredictiveControllerConfig:
        raise TypeError("config must be a PredictiveControllerConfig")
    if rollout.step_count != config.horizon_steps:
        raise ValueError("rollout step count must match configured horizon_steps")

    weights = config.objective_weights
    temperatures = rollout.cell_temperatures_k
    peak_temperature = float(temperatures.max())
    maximum_spread = max(
        float(state.cell_temperatures_k.max() - state.cell_temperatures_k.min())
        for state in rollout.states
    )

    try:
        pump_energy = math.fsum(
            pump_energy_j(
                action.coolant_mass_flow_kg_s,
                rollout.time_step_s,
                config.pump_power_coefficient_w_s3_kg3,
            )
            for action in rollout.actions
        )
        safety_penalty = math.fsum(
            max(float(temperature) - config.thermal_safety_limit_k, 0.0) ** 2
            for state in rollout.states
            for temperature in state.cell_temperatures_k
        )
        peak_excess = max(peak_temperature - weights.peak_temperature_reference_k, 0.0)
        peak_cost = peak_excess * weights.peak_temperature_weight
        spread_cost = maximum_spread * weights.temperature_spread_weight
        pump_cost = pump_energy * weights.pump_energy_weight
        safety_cost = safety_penalty * weights.safety_penalty_weight
        total_cost = math.fsum((peak_cost, spread_cost, pump_cost, safety_cost))
    except (OverflowError, ValueError) as error:
        raise ValueError("predictive objective costs must be finite") from error

    costs = (peak_cost, spread_cost, pump_cost, safety_cost, total_cost)
    if any(not math.isfinite(cost) or cost < 0.0 for cost in costs):
        raise ValueError("predictive objective costs must be finite and nonnegative")

    return PredictiveCandidateEvaluation(
        action=rollout.actions[0],
        total_cost=total_cost,
        peak_temperature_cost=peak_cost,
        temperature_spread_cost=spread_cost,
        pump_energy_cost=pump_cost,
        safety_penalty_cost=safety_cost,
    )


def _require_nonnegative_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def pump_power_w(
    coolant_mass_flow_kg_s: float,
    pump_power_coefficient_w_s3_kg3: float,
) -> float:
    """Estimate pump power as ``coefficient * mass_flow**3`` in watts.

    The coefficient is configurable; this cubic relationship is an MVP
    approximation and does not represent a manufacturer-specific pump curve.
    """
    flow = _require_nonnegative_finite(
        "coolant_mass_flow_kg_s", coolant_mass_flow_kg_s
    )
    coefficient = _require_nonnegative_finite(
        "pump_power_coefficient_w_s3_kg3", pump_power_coefficient_w_s3_kg3
    )
    try:
        power = coefficient * flow**3
    except OverflowError as error:
        raise ValueError("pump power result must be finite") from error
    if not math.isfinite(power):
        raise ValueError("pump power result must be finite")
    return power


def pump_energy_j(
    coolant_mass_flow_kg_s: float,
    duration_s: float,
    pump_power_coefficient_w_s3_kg3: float,
) -> float:
    """Estimate pump energy in joules over a constant-flow interval.

    Energy is pump power multiplied by the interval duration. Power follows a
    configurable MVP cubic approximation, not a manufacturer pump curve.
    """
    duration = _require_nonnegative_finite("duration_s", duration_s)
    power = pump_power_w(
        coolant_mass_flow_kg_s, pump_power_coefficient_w_s3_kg3
    )
    energy = power * duration
    if not math.isfinite(energy):
        raise ValueError("pump energy result must be finite")
    return energy
