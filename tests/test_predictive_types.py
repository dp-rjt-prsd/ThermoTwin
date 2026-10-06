from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from thermotwin.control.predictive_types import (
    PredictiveCandidateEvaluation,
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.core import CoolingAction


@pytest.fixture
def valid_weights() -> PredictiveObjectiveWeights:
    return PredictiveObjectiveWeights(
        peak_temperature_weight=1.0,
        temperature_spread_weight=2.0,
        pump_energy_weight=0.001,
        safety_penalty_weight=100.0,
    )


@pytest.fixture
def valid_config(
    valid_weights: PredictiveObjectiveWeights,
) -> PredictiveControllerConfig:
    return PredictiveControllerConfig(
        horizon_steps=12,
        candidate_count=5,
        minimum_coolant_mass_flow_kg_s=0.0,
        maximum_coolant_mass_flow_kg_s=0.1,
        thermal_safety_limit_k=313.15,
        pump_power_coefficient_w_s3_kg3=100_000.0,
        objective_weights=valid_weights,
    )


def test_objective_weights_retain_valid_values_and_are_immutable() -> None:
    weights = PredictiveObjectiveWeights(1, 2.5, 0.0, 100.0)

    assert weights.peak_temperature_weight == 1.0
    assert weights.temperature_spread_weight == 2.5
    assert weights.pump_energy_weight == 0.0
    assert weights.safety_penalty_weight == 100.0
    with pytest.raises(FrozenInstanceError):
        weights.pump_energy_weight = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "peak_temperature_weight",
        "temperature_spread_weight",
        "pump_energy_weight",
        "safety_penalty_weight",
    ],
)
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_objective_weights_reject_negative_or_nonfinite_values(
    field_name: str, invalid: float
) -> None:
    values = {
        "peak_temperature_weight": 1.0,
        "temperature_spread_weight": 1.0,
        "pump_energy_weight": 1.0,
        "safety_penalty_weight": 1.0,
    }
    values[field_name] = invalid

    with pytest.raises(ValueError):
        PredictiveObjectiveWeights(**values)


@pytest.mark.parametrize("invalid", [True, "1", None, complex(1.0, 0.0)])
def test_objective_weights_reject_non_real_values(invalid: object) -> None:
    with pytest.raises(ValueError):
        PredictiveObjectiveWeights(invalid, 1.0, 1.0, 1.0)  # type: ignore[arg-type]


def test_predictive_config_retains_values_and_is_immutable(
    valid_config: PredictiveControllerConfig,
    valid_weights: PredictiveObjectiveWeights,
) -> None:
    assert valid_config.horizon_steps == 12
    assert valid_config.candidate_count == 5
    assert valid_config.minimum_coolant_mass_flow_kg_s == 0.0
    assert valid_config.maximum_coolant_mass_flow_kg_s == 0.1
    assert valid_config.thermal_safety_limit_k == 313.15
    assert valid_config.pump_power_coefficient_w_s3_kg3 == 100_000.0
    assert valid_config.objective_weights is valid_weights
    with pytest.raises(FrozenInstanceError):
        valid_config.horizon_steps = 6  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["horizon_steps", "candidate_count"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_predictive_config_rejects_invalid_counts(
    valid_config: PredictiveControllerConfig, field_name: str, invalid: object
) -> None:
    with pytest.raises(ValueError):
        PredictiveControllerConfig(
            horizon_steps=(
                cast(int, invalid)
                if field_name == "horizon_steps"
                else valid_config.horizon_steps
            ),
            candidate_count=(
                cast(int, invalid)
                if field_name == "candidate_count"
                else valid_config.candidate_count
            ),
            minimum_coolant_mass_flow_kg_s=(
                valid_config.minimum_coolant_mass_flow_kg_s
            ),
            maximum_coolant_mass_flow_kg_s=(
                valid_config.maximum_coolant_mass_flow_kg_s
            ),
            thermal_safety_limit_k=valid_config.thermal_safety_limit_k,
            pump_power_coefficient_w_s3_kg3=(
                valid_config.pump_power_coefficient_w_s3_kg3
            ),
            objective_weights=valid_config.objective_weights,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("minimum_coolant_mass_flow_kg_s", -0.01),
        ("minimum_coolant_mass_flow_kg_s", float("nan")),
        ("minimum_coolant_mass_flow_kg_s", float("inf")),
        ("maximum_coolant_mass_flow_kg_s", -0.01),
        ("maximum_coolant_mass_flow_kg_s", float("nan")),
        ("maximum_coolant_mass_flow_kg_s", float("inf")),
        ("thermal_safety_limit_k", 0.0),
        ("thermal_safety_limit_k", -1.0),
        ("thermal_safety_limit_k", float("nan")),
        ("thermal_safety_limit_k", float("inf")),
        ("pump_power_coefficient_w_s3_kg3", -1.0),
        ("pump_power_coefficient_w_s3_kg3", float("nan")),
        ("pump_power_coefficient_w_s3_kg3", float("inf")),
    ],
)
def test_predictive_config_rejects_illegal_or_nonfinite_physical_values(
    valid_config: PredictiveControllerConfig, field_name: str, invalid: object
) -> None:
    with pytest.raises(ValueError):
        PredictiveControllerConfig(
            horizon_steps=valid_config.horizon_steps,
            candidate_count=valid_config.candidate_count,
            minimum_coolant_mass_flow_kg_s=(
                cast(float, invalid)
                if field_name == "minimum_coolant_mass_flow_kg_s"
                else valid_config.minimum_coolant_mass_flow_kg_s
            ),
            maximum_coolant_mass_flow_kg_s=(
                cast(float, invalid)
                if field_name == "maximum_coolant_mass_flow_kg_s"
                else valid_config.maximum_coolant_mass_flow_kg_s
            ),
            thermal_safety_limit_k=(
                cast(float, invalid)
                if field_name == "thermal_safety_limit_k"
                else valid_config.thermal_safety_limit_k
            ),
            pump_power_coefficient_w_s3_kg3=(
                cast(float, invalid)
                if field_name == "pump_power_coefficient_w_s3_kg3"
                else valid_config.pump_power_coefficient_w_s3_kg3
            ),
            objective_weights=valid_config.objective_weights,
        )


def test_predictive_config_allows_equal_nonnegative_flow_bounds(
    valid_config: PredictiveControllerConfig,
) -> None:
    config = PredictiveControllerConfig(
        horizon_steps=valid_config.horizon_steps,
        candidate_count=valid_config.candidate_count,
        minimum_coolant_mass_flow_kg_s=0.02,
        maximum_coolant_mass_flow_kg_s=0.02,
        thermal_safety_limit_k=valid_config.thermal_safety_limit_k,
        pump_power_coefficient_w_s3_kg3=(
            valid_config.pump_power_coefficient_w_s3_kg3
        ),
        objective_weights=valid_config.objective_weights,
    )

    assert config.minimum_coolant_mass_flow_kg_s == 0.02
    assert config.maximum_coolant_mass_flow_kg_s == 0.02


def test_predictive_config_accepts_inclusive_minimum_boundaries() -> None:
    config = PredictiveControllerConfig(
        horizon_steps=1,
        candidate_count=2,
        minimum_coolant_mass_flow_kg_s=0.0,
        maximum_coolant_mass_flow_kg_s=0.0,
        thermal_safety_limit_k=0.1,
        pump_power_coefficient_w_s3_kg3=0.0,
        objective_weights=PredictiveObjectiveWeights(0.0, 0.0, 0.0, 0.0),
    )

    assert config.horizon_steps == 1
    assert config.candidate_count == 2
    assert config.pump_power_coefficient_w_s3_kg3 == 0.0


def test_predictive_config_rejects_reversed_flow_bounds(
    valid_config: PredictiveControllerConfig,
) -> None:
    with pytest.raises(ValueError):
        PredictiveControllerConfig(
            horizon_steps=valid_config.horizon_steps,
            candidate_count=valid_config.candidate_count,
            minimum_coolant_mass_flow_kg_s=0.1,
            maximum_coolant_mass_flow_kg_s=0.0,
            thermal_safety_limit_k=valid_config.thermal_safety_limit_k,
            pump_power_coefficient_w_s3_kg3=(
                valid_config.pump_power_coefficient_w_s3_kg3
            ),
            objective_weights=valid_config.objective_weights,
        )


def test_predictive_config_requires_typed_objective_weights(
    valid_config: PredictiveControllerConfig,
) -> None:
    with pytest.raises(ValueError):
        PredictiveControllerConfig(
            horizon_steps=valid_config.horizon_steps,
            candidate_count=valid_config.candidate_count,
            minimum_coolant_mass_flow_kg_s=(
                valid_config.minimum_coolant_mass_flow_kg_s
            ),
            maximum_coolant_mass_flow_kg_s=(
                valid_config.maximum_coolant_mass_flow_kg_s
            ),
            thermal_safety_limit_k=valid_config.thermal_safety_limit_k,
            pump_power_coefficient_w_s3_kg3=(
                valid_config.pump_power_coefficient_w_s3_kg3
            ),
            objective_weights=cast(PredictiveObjectiveWeights, None),
        )


def test_candidate_evaluation_retains_components_and_is_immutable() -> None:
    action = CoolingAction(coolant_mass_flow_kg_s=0.04)
    evaluation = PredictiveCandidateEvaluation(
        action=action,
        total_cost=42.5,
        peak_temperature_cost=31.0,
        temperature_spread_cost=3.5,
        pump_energy_cost=2.0,
        safety_penalty_cost=6.0,
    )

    assert evaluation.action is action
    assert evaluation.total_cost == 42.5
    assert evaluation.peak_temperature_cost == 31.0
    assert evaluation.temperature_spread_cost == 3.5
    assert evaluation.pump_energy_cost == 2.0
    assert evaluation.safety_penalty_cost == 6.0
    with pytest.raises(FrozenInstanceError):
        evaluation.total_cost = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field_name",
    [
        "total_cost",
        "peak_temperature_cost",
        "temperature_spread_cost",
        "pump_energy_cost",
        "safety_penalty_cost",
    ],
)
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_candidate_evaluation_rejects_negative_or_nonfinite_costs(
    field_name: str, invalid: float
) -> None:
    with pytest.raises(ValueError):
        PredictiveCandidateEvaluation(
            action=CoolingAction(coolant_mass_flow_kg_s=0.04),
            total_cost=invalid if field_name == "total_cost" else 1.0,
            peak_temperature_cost=(
                invalid if field_name == "peak_temperature_cost" else 1.0
            ),
            temperature_spread_cost=(
                invalid if field_name == "temperature_spread_cost" else 1.0
            ),
            pump_energy_cost=(
                invalid if field_name == "pump_energy_cost" else 1.0
            ),
            safety_penalty_cost=(
                invalid if field_name == "safety_penalty_cost" else 1.0
            ),
        )


def test_candidate_evaluation_requires_cooling_action() -> None:
    with pytest.raises(ValueError):
        PredictiveCandidateEvaluation(
            action=None,  # type: ignore[arg-type]
            total_cost=1.0,
            peak_temperature_cost=1.0,
            temperature_spread_cost=0.0,
            pump_energy_cost=0.0,
            safety_penalty_cost=0.0,
        )
