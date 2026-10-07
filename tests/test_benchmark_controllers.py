from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from thermotwin.control.base import Controller
from thermotwin.control.pid import PIDController
from thermotwin.control.predictive import PredictiveController
from thermotwin.control.predictive_types import PredictiveObjectiveWeights
from thermotwin.control.threshold import ThresholdController
from thermotwin.core import BatteryState, CoolingAction, CoolingConfig
from thermotwin.experiments.benchmark_controllers import (
    BENCHMARK_FORECAST_ASSUMPTIONS,
    BenchmarkControllerParameters,
    build_benchmark_controllers,
)
from thermotwin.experiments.benchmark_types import (
    VARIANT_IDS,
    ForecastAssumptions,
    ModelVariant,
    VariantId,
)
from thermotwin.models.forecaster import ForecastRollout, ThermalForecaster


def _parameters(**overrides: object) -> BenchmarkControllerParameters:
    values: dict[str, object] = {
        "threshold_temperature_boundaries_k": (303.15, 308.15, 313.15),
        "threshold_intermediate_flow_fractions": (0.25, 0.6),
        "pid_target_temperature_k": 308.15,
        "pid_kp_kg_s_k": 0.005,
        "pid_ki_kg_s_k_s": 0.0001,
        "pid_kd_kg_s_k_per_s": 0.0,
        "predictive_horizon_steps": 3,
        "predictive_candidate_count": 4,
        "thermal_safety_limit_k": 313.15,
        "pump_power_coefficient_w_s3_kg3": 100000.0,
        "objective_weights": PredictiveObjectiveWeights(1.0, 1.0, 0.0001, 100.0),
    }
    values.update(overrides)
    return BenchmarkControllerParameters(**values)  # type: ignore[arg-type]


def _cooling(maximum: float = 0.1, minimum: float = 0.0) -> CoolingConfig:
    return CoolingConfig(minimum, maximum, 100.0, 1.0, 0.5)


def _state(*, time_s: float = 0.0) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=20.0,
        soc_fraction=0.55,
        ambient_temperature_k=303.0,
        coolant_inlet_temperature_k=295.0,
        coolant_mass_flow_kg_s=0.01,
        cell_temperatures_k=np.full(12, 310.0),
        cell_resistances_ohm=np.linspace(0.004, 0.008, 12),
    )


class RecordingForecaster:
    def __init__(self, *, cell_count: int = 12, time_step_s: float = 1.0) -> None:
        self._cell_count = cell_count
        self._time_step_s = time_step_s
        self.calls: list[tuple[BatteryState, tuple[CoolingAction, ...], object]] = []

    @property
    def cell_count(self) -> int:
        return self._cell_count

    @property
    def time_step_s(self) -> float:
        return self._time_step_s

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        return self._next(state, action)

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: object = None,
    ) -> ForecastRollout:
        self.calls.append((initial_state, actions, future_conditions))
        states: list[BatteryState] = []
        current = initial_state
        for _action in actions:
            current = self._next(current, _action)
            states.append(current)
        return ForecastRollout(initial_state, actions, tuple(states))

    def _next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        return BatteryState(
            time_s=state.time_s + self.time_step_s,
            current_a=state.current_a,
            soc_fraction=state.soc_fraction,
            ambient_temperature_k=state.ambient_temperature_k,
            coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=action.coolant_mass_flow_kg_s,
            cell_temperatures_k=state.cell_temperatures_k,
            cell_resistances_ohm=state.cell_resistances_ohm,
        )


def _forecasters(
    data: RecordingForecaster | None = None,
    physics: RecordingForecaster | None = None,
) -> dict[ModelVariant, ThermalForecaster]:
    return {
        "data_only": data or RecordingForecaster(),
        "physics_informed": physics or RecordingForecaster(),
    }


def test_factory_returns_exact_order_fresh_immutable_controller_mapping() -> None:
    first = build_benchmark_controllers(_cooling(), _parameters(), _forecasters())
    second = build_benchmark_controllers(_cooling(), _parameters(), _forecasters())

    assert isinstance(first, MappingProxyType)
    assert tuple(first) == VARIANT_IDS
    assert tuple(first.values()) == (
        first["threshold"],
        first["pid"],
        first["predictive_data_only"],
        first["predictive_physics_informed"],
    )
    assert isinstance(first["threshold"], ThresholdController)
    assert isinstance(first["pid"], PIDController)
    assert isinstance(first["predictive_data_only"], PredictiveController)
    assert isinstance(first["predictive_physics_informed"], PredictiveController)
    assert all(first[key] is not second[key] for key in VARIANT_IDS)
    with pytest.raises(TypeError):
        cast(dict[str, Controller], first)["pid"] = first["threshold"]


def test_parameters_copy_mutable_sequences_and_are_frozen() -> None:
    boundaries = [303.15, 308.15, 313.15]
    fractions = [0.25, 0.75]
    parameters = _parameters(
        threshold_temperature_boundaries_k=boundaries,
        threshold_intermediate_flow_fractions=fractions,
    )
    boundaries[0] = 200.0
    fractions[0] = 1.0
    assert parameters.threshold_temperature_boundaries_k == (303.15, 308.15, 313.15)
    assert parameters.threshold_intermediate_flow_fractions == (0.25, 0.75)
    with pytest.raises(FrozenInstanceError):
        parameters.pid_target_temperature_k = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold_temperature_boundaries_k", (303.0, 303.0, 314.0), "ascending"),
        ("threshold_temperature_boundaries_k", (303.0, True, 314.0), "finite real"),
        ("threshold_intermediate_flow_fractions", (-0.1, 0.5), "[0, 1]"),
        ("threshold_intermediate_flow_fractions", (0.8, 0.2), "nondecreasing"),
        ("threshold_intermediate_flow_fractions", (0.2, float("nan")), "finite"),
        ("pid_target_temperature_k", True, "finite real"),
        ("pid_kp_kg_s_k", -1.0, "nonnegative"),
        ("predictive_horizon_steps", True, "positive integer"),
        ("predictive_candidate_count", 1, "greater than or equal to 2"),
        ("thermal_safety_limit_k", float("inf"), "finite"),
        ("pump_power_coefficient_w_s3_kg3", -1.0, "nonnegative"),
        ("objective_weights", object(), "PredictiveObjectiveWeights"),
    ],
)
def test_parameter_validation_is_field_specific(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _parameters(**{field: value})


def test_forecast_assumptions_disclose_persistence_without_truth() -> None:
    assert BENCHMARK_FORECAST_ASSUMPTIONS == ForecastAssumptions(
        "current_value_persistence",
        "current_value_persistence",
        "current_value_persistence",
        "current_value_persistence",
        "constant_candidate_sequence",
        False,
    )


def test_predictive_variants_use_identical_observed_state_and_constant_actions(
) -> None:
    data = RecordingForecaster()
    physics = RecordingForecaster()
    forecasters = _forecasters(data, physics)
    controllers = build_benchmark_controllers(_cooling(), _parameters(), forecasters)
    # Resistance values may reflect a fault, but no fault identity or schedule is
    # supplied to the factory or either predictive controller.
    observed = _state()

    controllers["predictive_data_only"].act(observed)
    controllers["predictive_physics_informed"].act(observed)

    assert len(data.calls) == len(physics.calls) == 4
    assert all(call[0] is observed for call in data.calls + physics.calls)
    assert all(call[2] is None for call in data.calls + physics.calls)
    data_sequences = [call[1] for call in data.calls]
    physics_sequences = [call[1] for call in physics.calls]
    assert data_sequences == physics_sequences
    for actions in data_sequences:
        assert len(actions) == 3
        assert len({action.coolant_mass_flow_kg_s for action in actions}) == 1


def test_reactive_controllers_use_current_state_and_all_actions_are_bounded() -> None:
    cooling = _cooling(maximum=0.025, minimum=0.005)
    controllers = build_benchmark_controllers(
        cooling,
        _parameters(threshold_intermediate_flow_fractions=(0.5, 1.0)),
        _forecasters(),
    )
    state = _state()

    reactive_variants: tuple[VariantId, ...] = ("threshold", "pid")
    predictive_variants: tuple[VariantId, ...] = (
        "predictive_data_only",
        "predictive_physics_informed",
    )
    for variant in reactive_variants:
        controller = controllers[variant]
        controller.reset()
        action = controller.act(state)
        assert cooling.minimum_coolant_mass_flow_kg_s <= action.coolant_mass_flow_kg_s
        assert action.coolant_mass_flow_kg_s <= cooling.maximum_coolant_mass_flow_kg_s

    for variant in predictive_variants:
        action = controllers[variant].act(state)
        assert cooling.minimum_coolant_mass_flow_kg_s <= action.coolant_mass_flow_kg_s
        assert action.coolant_mass_flow_kg_s <= cooling.maximum_coolant_mass_flow_kg_s

    assert controllers["threshold"].act(state).coolant_mass_flow_kg_s == 0.025


@pytest.mark.parametrize(
    "forecasters",
    [
        {},
        {"data_only": RecordingForecaster()},
        {
            "data_only": RecordingForecaster(),
            "physics_informed": RecordingForecaster(),
            "extra": RecordingForecaster(),
        },
    ],
)
def test_forecasters_require_exact_model_variant_keys(
    forecasters: Mapping[ModelVariant, ThermalForecaster],
) -> None:
    with pytest.raises(ValueError, match="forecasters keys mismatch"):
        build_benchmark_controllers(_cooling(), _parameters(), forecasters)


@pytest.mark.parametrize(
    ("data_count", "physics_count", "data_step", "physics_step", "message"),
    [
        (11, 12, 1.0, 1.0, "data_only.*cell_count must be 12"),
        (12, 13, 1.0, 1.0, "physics_informed.*cell_count must be 12"),
        (12, 12, 1.0, 2.0, "same time_step_s"),
        (12, 12, float("nan"), 1.0, "time_step_s must be finite"),
    ],
)
def test_forecaster_shape_and_timestep_are_validated_before_build(
    data_count: int,
    physics_count: int,
    data_step: float,
    physics_step: float,
    message: str,
) -> None:
    forecasters = _forecasters(
        RecordingForecaster(cell_count=data_count, time_step_s=data_step),
        RecordingForecaster(cell_count=physics_count, time_step_s=physics_step),
    )
    with pytest.raises(ValueError, match=message):
        build_benchmark_controllers(_cooling(), _parameters(), forecasters)


def test_runtime_protocol_impostor_is_rejected() -> None:
    class Impostor:
        cell_count = 12
        time_step_s = 1.0

        def predict_next(
            self, state: BatteryState, action: CoolingAction
        ) -> BatteryState:
            return state

    impostor: object = Impostor()
    with pytest.raises(ValueError, match="satisfy ThermalForecaster"):
        build_benchmark_controllers(
            _cooling(),
            _parameters(),
            cast(
                Mapping[ModelVariant, ThermalForecaster],
                {
                    "data_only": cast(ThermalForecaster, impostor),
                    "physics_informed": RecordingForecaster(),
                },
            ),
        )


def test_threshold_fractions_are_affine_over_nonzero_cooling_interval() -> None:
    cooling = _cooling(maximum=0.1, minimum=0.02)
    controller = build_benchmark_controllers(cooling, _parameters(), _forecasters())[
        "threshold"
    ]
    low = _state()
    low = BatteryState(
        time_s=low.time_s,
        current_a=low.current_a,
        soc_fraction=low.soc_fraction,
        ambient_temperature_k=low.ambient_temperature_k,
        coolant_inlet_temperature_k=low.coolant_inlet_temperature_k,
        coolant_mass_flow_kg_s=low.coolant_mass_flow_kg_s,
        cell_temperatures_k=np.full(12, 304.0),
        cell_resistances_ohm=low.cell_resistances_ohm,
    )
    assert controller.act(low).coolant_mass_flow_kg_s == pytest.approx(0.04)
