"""Fair, leakage-resistant controller construction for Phase 12 benchmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Final

from thermotwin.control.base import Controller
from thermotwin.control.pid import PIDController
from thermotwin.control.predictive import PredictiveController
from thermotwin.control.predictive_types import (
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.control.threshold import ThresholdController
from thermotwin.core import CoolingConfig
from thermotwin.experiments.benchmark_types import (
    VARIANT_IDS,
    ForecastAssumptions,
    ModelVariant,
    VariantId,
)
from thermotwin.models.forecaster import ThermalForecaster


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _finite_tuple(
    name: str, values: object, *, length: int, positive: bool = False
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain exactly {length} finite real numbers")
    if not isinstance(values, Sequence):
        raise ValueError(f"{name} must contain exactly {length} finite real numbers")
    if len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} finite real numbers")
    result = tuple(
        _finite_real(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _nonnegative(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
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
class BenchmarkControllerParameters:
    """Immutable controller settings shared by every benchmark scenario."""

    threshold_temperature_boundaries_k: tuple[float, float, float]
    threshold_intermediate_flow_fractions: tuple[float, float]
    pid_target_temperature_k: float
    pid_kp_kg_s_k: float
    pid_ki_kg_s_k_s: float
    pid_kd_kg_s_k_per_s: float
    predictive_horizon_steps: int
    predictive_candidate_count: int
    thermal_safety_limit_k: float
    pump_power_coefficient_w_s3_kg3: float
    objective_weights: PredictiveObjectiveWeights

    def __post_init__(self) -> None:
        boundaries = _finite_tuple(
            "threshold_temperature_boundaries_k",
            self.threshold_temperature_boundaries_k,
            length=3,
            positive=True,
        )
        if any(
            left >= right
            for left, right in zip(boundaries, boundaries[1:], strict=False)
        ):
            raise ValueError(
                "threshold_temperature_boundaries_k must be strictly ascending"
            )
        fractions = _finite_tuple(
            "threshold_intermediate_flow_fractions",
            self.threshold_intermediate_flow_fractions,
            length=2,
        )
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError(
                "threshold_intermediate_flow_fractions values must be in [0, 1]"
            )
        if fractions[0] > fractions[1]:
            raise ValueError(
                "threshold_intermediate_flow_fractions must be nondecreasing"
            )
        object.__setattr__(self, "threshold_temperature_boundaries_k", boundaries)
        object.__setattr__(self, "threshold_intermediate_flow_fractions", fractions)

        target = _finite_real(
            "pid_target_temperature_k", self.pid_target_temperature_k
        )
        if target <= 0.0:
            raise ValueError("pid_target_temperature_k must be positive")
        object.__setattr__(self, "pid_target_temperature_k", target)
        for name in ("pid_kp_kg_s_k", "pid_ki_kg_s_k_s", "pid_kd_kg_s_k_per_s"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))

        object.__setattr__(
            self,
            "predictive_horizon_steps",
            _positive_integer(
                "predictive_horizon_steps", self.predictive_horizon_steps
            ),
        )
        object.__setattr__(
            self,
            "predictive_candidate_count",
            _positive_integer(
                "predictive_candidate_count", self.predictive_candidate_count, minimum=2
            ),
        )
        limit = _finite_real("thermal_safety_limit_k", self.thermal_safety_limit_k)
        if limit <= 0.0:
            raise ValueError("thermal_safety_limit_k must be positive")
        object.__setattr__(self, "thermal_safety_limit_k", limit)
        object.__setattr__(
            self,
            "pump_power_coefficient_w_s3_kg3",
            _nonnegative(
                "pump_power_coefficient_w_s3_kg3",
                self.pump_power_coefficient_w_s3_kg3,
            ),
        )
        if type(self.objective_weights) is not PredictiveObjectiveWeights:
            raise ValueError("objective_weights must be PredictiveObjectiveWeights")


BENCHMARK_FORECAST_ASSUMPTIONS: Final[ForecastAssumptions] = ForecastAssumptions(
    current_assumption="current_value_persistence",
    soc_assumption="current_value_persistence",
    ambient_temperature_assumption="current_value_persistence",
    coolant_inlet_temperature_assumption="current_value_persistence",
    action_handling="constant_candidate_sequence",
    uses_future_simulator_truth=False,
)


def _validated_forecaster(
    name: str, value: object
) -> tuple[ThermalForecaster, float]:
    if not isinstance(value, ThermalForecaster):
        raise ValueError(f"forecasters[{name!r}] must satisfy ThermalForecaster")
    if not callable(getattr(value, "predict_next", None)) or not callable(
        getattr(value, "rollout", None)
    ):
        raise ValueError(f"forecasters[{name!r}] methods must be callable")
    cell_count = getattr(value, "cell_count", None)
    if isinstance(cell_count, bool) or not isinstance(cell_count, Integral):
        raise ValueError(f"forecasters[{name!r}].cell_count must be 12")
    if int(cell_count) != 12:
        raise ValueError(f"forecasters[{name!r}].cell_count must be 12")
    time_step = _finite_real(
        f"forecasters[{name!r}].time_step_s", getattr(value, "time_step_s", None)
    )
    if time_step <= 0.0:
        raise ValueError(f"forecasters[{name!r}].time_step_s must be positive")
    return value, time_step


def build_benchmark_controllers(
    cooling_config: CoolingConfig,
    parameters: BenchmarkControllerParameters,
    forecasters: Mapping[ModelVariant, ThermalForecaster],
) -> Mapping[VariantId, Controller]:
    """Create the four fresh controllers using only scenario cooling limits."""
    if type(cooling_config) is not CoolingConfig:
        raise ValueError("cooling_config must be a CoolingConfig")
    if type(parameters) is not BenchmarkControllerParameters:
        raise ValueError("parameters must be BenchmarkControllerParameters")
    if not isinstance(forecasters, Mapping):
        raise ValueError("forecasters must map data_only and physics_informed")
    expected = {"data_only", "physics_informed"}
    received = set(forecasters)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected, key=repr)
        raise ValueError(f"forecasters keys mismatch; missing={missing}, extra={extra}")

    data_forecaster, data_step = _validated_forecaster(
        "data_only", forecasters["data_only"]
    )
    physics_forecaster, physics_step = _validated_forecaster(
        "physics_informed", forecasters["physics_informed"]
    )
    if not math.isclose(data_step, physics_step, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("forecasters must have the same time_step_s")

    minimum = cooling_config.minimum_coolant_mass_flow_kg_s
    maximum = cooling_config.maximum_coolant_mass_flow_kg_s
    span = maximum - minimum
    intermediate_flows = tuple(
        minimum + fraction * span
        for fraction in parameters.threshold_intermediate_flow_fractions
    )
    predictive_config = PredictiveControllerConfig(
        horizon_steps=parameters.predictive_horizon_steps,
        candidate_count=parameters.predictive_candidate_count,
        minimum_coolant_mass_flow_kg_s=minimum,
        maximum_coolant_mass_flow_kg_s=maximum,
        thermal_safety_limit_k=parameters.thermal_safety_limit_k,
        pump_power_coefficient_w_s3_kg3=parameters.pump_power_coefficient_w_s3_kg3,
        objective_weights=parameters.objective_weights,
    )
    controllers: dict[VariantId, Controller] = {
        "threshold": ThresholdController(
            cooling_config,
            parameters.threshold_temperature_boundaries_k,
            intermediate_flows,
        ),
        "pid": PIDController(
            cooling_config,
            parameters.pid_target_temperature_k,
            parameters.pid_kp_kg_s_k,
            parameters.pid_ki_kg_s_k_s,
            parameters.pid_kd_kg_s_k_per_s,
        ),
        "predictive_data_only": PredictiveController(
            data_forecaster, predictive_config
        ),
        "predictive_physics_informed": PredictiveController(
            physics_forecaster, predictive_config
        ),
    }
    return MappingProxyType({variant: controllers[variant] for variant in VARIANT_IDS})


__all__ = [
    "BENCHMARK_FORECAST_ASSUMPTIONS",
    "BenchmarkControllerParameters",
    "build_benchmark_controllers",
]
