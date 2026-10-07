"""Immutable configuration and result contracts for predictive control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

from thermotwin.core import CoolingAction


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_real(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_real(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _positive_integer(name: str, value: object, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        if minimum == 1:
            requirement = "a positive integer"
        else:
            requirement = f"an integer greater than or equal to {minimum}"
        raise ValueError(f"{name} must be {requirement}")
    return int(value)


@dataclass(frozen=True, slots=True)
class PredictiveObjectiveWeights:
    """Nonnegative weights for the predictive objective's component costs."""

    peak_temperature_weight: float
    temperature_spread_weight: float
    pump_energy_weight: float
    safety_penalty_weight: float
    # Peak temperature is only penalised above this reference (kelvin). The
    # default of zero keeps the original absolute peak-temperature cost.
    peak_temperature_reference_k: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "peak_temperature_weight",
            "temperature_spread_weight",
            "pump_energy_weight",
            "safety_penalty_weight",
            "peak_temperature_reference_k",
        ):
            object.__setattr__(
                self, name, _nonnegative_real(name, getattr(self, name))
            )


@dataclass(frozen=True, slots=True)
class PredictiveControllerConfig:
    """Validated receding-horizon settings, using SI units for physical values."""

    horizon_steps: int
    candidate_count: int
    minimum_coolant_mass_flow_kg_s: float
    maximum_coolant_mass_flow_kg_s: float
    thermal_safety_limit_k: float
    pump_power_coefficient_w_s3_kg3: float
    objective_weights: PredictiveObjectiveWeights

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon_steps",
            _positive_integer("horizon_steps", self.horizon_steps),
        )
        object.__setattr__(
            self,
            "candidate_count",
            _positive_integer("candidate_count", self.candidate_count, minimum=2),
        )
        minimum = _nonnegative_real(
            "minimum_coolant_mass_flow_kg_s", self.minimum_coolant_mass_flow_kg_s
        )
        maximum = _nonnegative_real(
            "maximum_coolant_mass_flow_kg_s", self.maximum_coolant_mass_flow_kg_s
        )
        if minimum > maximum:
            raise ValueError(
                "minimum_coolant_mass_flow_kg_s must not exceed "
                "maximum_coolant_mass_flow_kg_s"
            )
        object.__setattr__(self, "minimum_coolant_mass_flow_kg_s", minimum)
        object.__setattr__(self, "maximum_coolant_mass_flow_kg_s", maximum)
        object.__setattr__(
            self,
            "thermal_safety_limit_k",
            _positive_real("thermal_safety_limit_k", self.thermal_safety_limit_k),
        )
        object.__setattr__(
            self,
            "pump_power_coefficient_w_s3_kg3",
            _nonnegative_real(
                "pump_power_coefficient_w_s3_kg3",
                self.pump_power_coefficient_w_s3_kg3,
            ),
        )
        if not isinstance(self.objective_weights, PredictiveObjectiveWeights):
            raise ValueError("objective_weights must be PredictiveObjectiveWeights")


@dataclass(frozen=True, slots=True)
class PredictiveCandidateEvaluation:
    """Cost accounting for one candidate action; performs no optimization."""

    action: CoolingAction
    total_cost: float
    peak_temperature_cost: float
    temperature_spread_cost: float
    pump_energy_cost: float
    safety_penalty_cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.action, CoolingAction):
            raise ValueError("action must be a CoolingAction")
        for name in (
            "total_cost",
            "peak_temperature_cost",
            "temperature_spread_cost",
            "pump_energy_cost",
            "safety_penalty_cost",
        ):
            object.__setattr__(
                self, name, _nonnegative_real(name, getattr(self, name))
            )


__all__ = [
    "PredictiveCandidateEvaluation",
    "PredictiveControllerConfig",
    "PredictiveObjectiveWeights",
]
