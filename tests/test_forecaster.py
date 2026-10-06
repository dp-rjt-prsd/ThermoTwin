"""Tests for the ThermalMLP-backed thermal forecasting adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import torch

from thermotwin.core.types import BatteryState, CoolingAction
from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import MLPConfig
from thermotwin.models.forecaster import (
    ForecastConditions,
    ForecastRollout,
    MLPThermalForecaster,
    ThermalForecaster,
)
from thermotwin.models.mlp import ThermalMLP


def _normalization(cell_count: int = 2) -> NormalizationStats:
    return NormalizationStats(
        feature_names(cell_count),
        target_names(cell_count),
        np.arange(cell_count + 5, dtype=np.float64),
        np.full(cell_count + 5, 2.0),
        np.arange(cell_count, dtype=np.float64) + 10.0,
        np.full(cell_count, 3.0),
    )


def _model(cell_count: int = 2, *, dtype: torch.dtype = torch.float64) -> ThermalMLP:
    model = ThermalMLP(
        MLPConfig(input_dim=cell_count + 5, output_dim=cell_count, hidden_widths=(4,))
    ).to(dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def _state(
    temperatures: tuple[float, ...] = (300.0, 301.0),
    *,
    time: float = 5.0,
    current: float = -10.0,
    soc: float = 0.6,
    ambient: float = 295.0,
    coolant: float = 290.0,
    flow: float = 0.2,
) -> BatteryState:
    return BatteryState(
        time_s=time,
        current_a=current,
        soc_fraction=soc,
        ambient_temperature_k=ambient,
        coolant_inlet_temperature_k=coolant,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=np.asarray(temperatures),
        cell_resistances_ohm=np.full(len(temperatures), 0.01),
    )


def _forecaster(model: ThermalMLP | None = None) -> MLPThermalForecaster:
    return MLPThermalForecaster(model or _model(), _normalization(), 2.0, 0.1, 0.8)


def test_predict_next_denormalizes_delta_and_propagates_physical_state() -> None:
    model = _model()
    output_layer = cast(torch.nn.Linear, model.network[-1])
    assert output_layer.bias is not None
    with torch.no_grad():
        output_layer.bias.copy_(torch.tensor([1.0, -2.0], dtype=torch.float64))
    forecaster = _forecaster(model)
    state = _state()

    result = forecaster.predict_next(state, CoolingAction(0.5))

    np.testing.assert_allclose(result.cell_temperatures_k, [313.0, 306.0])
    assert result.time_s == 7.0
    assert result.coolant_mass_flow_kg_s == 0.5
    assert result.current_a == state.current_a
    assert result.soc_fraction == state.soc_fraction
    assert result.ambient_temperature_k == state.ambient_temperature_k
    assert result.coolant_inlet_temperature_k == state.coolant_inlet_temperature_k
    np.testing.assert_array_equal(
        result.cell_resistances_ohm, state.cell_resistances_ohm
    )


def test_model_runs_in_eval_inference_mode_and_restores_each_training_mode() -> None:
    model = _model()
    model.train()
    model.network[1].eval()
    modes = tuple(module.training for module in model.modules())
    observed: list[tuple[bool, bool]] = []
    original_forward = cast(Callable[[torch.Tensor], torch.Tensor], model.forward)

    def inspect(inputs: torch.Tensor) -> torch.Tensor:
        observed.append((model.training, torch.is_inference_mode_enabled()))
        return original_forward(inputs)

    model.forward = inspect  # type: ignore[method-assign]
    _forecaster(model).predict_next(_state(), CoolingAction(0.5))

    assert observed == [(False, True)]
    assert tuple(module.training for module in model.modules()) == modes


def test_input_state_arrays_and_model_weights_are_not_mutated() -> None:
    model = _model()
    state = _state()
    temps_before = state.cell_temperatures_k.copy()
    resistance_before = state.cell_resistances_ohm.copy()
    weights_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )

    _forecaster(model).predict_next(state, CoolingAction(0.5))

    np.testing.assert_array_equal(state.cell_temperatures_k, temps_before)
    np.testing.assert_array_equal(state.cell_resistances_ohm, resistance_before)
    assert all(
        torch.equal(before, after)
        for before, after in zip(weights_before, model.parameters(), strict=True)
    )


def test_constant_predicted_delta_accumulates_autoregressively() -> None:
    model = _model()
    output_layer = cast(torch.nn.Linear, model.network[-1])
    assert output_layer.bias is not None
    with torch.no_grad():
        output_layer.bias.fill_(1.0)
    forecaster = _forecaster(model)
    initial = _state()

    result = forecaster.rollout(
        initial,
        (CoolingAction(0.1), CoolingAction(0.4), CoolingAction(0.8)),
    )

    assert isinstance(result, ForecastRollout)
    np.testing.assert_allclose(
        result.cell_temperatures_k, [[313, 315], [326, 329], [339, 343]]
    )
    assert result.step_count == 3


def test_rollout_conditions_apply_after_each_action_and_persist_by_default() -> None:
    forecaster = _forecaster()
    initial = _state()
    actions = (CoolingAction(0.3), CoolingAction(0.4))
    conditions = (
        ForecastConditions(20.0, 0.5, 301.0, 281.0),
        ForecastConditions(21.0, 0.4, 302.0, 282.0),
    )

    conditioned = forecaster.rollout(initial, actions, conditions)
    persisted = forecaster.rollout(initial, actions)

    first, second = conditioned.states
    assert (first.current_a, first.soc_fraction, first.ambient_temperature_k) == (
        20,
        0.5,
        301,
    )
    assert (second.current_a, second.soc_fraction, second.ambient_temperature_k) == (
        21,
        0.4,
        302,
    )
    assert second.coolant_inlet_temperature_k == 282
    assert persisted.states[0].current_a == initial.current_a
    assert persisted.states[1].ambient_temperature_k == initial.ambient_temperature_k
    assert (
        persisted.states[1].coolant_inlet_temperature_k
        == initial.coolant_inlet_temperature_k
    )


def test_condition_zero_is_an_input_to_the_second_autoregressive_step() -> None:
    model = _model()
    observed_inputs: list[torch.Tensor] = []
    original_forward = cast(Callable[[torch.Tensor], torch.Tensor], model.forward)

    def capture(inputs: torch.Tensor) -> torch.Tensor:
        observed_inputs.append(inputs.detach().clone())
        return original_forward(inputs)

    model.forward = capture  # type: ignore[method-assign]
    first_condition = ForecastConditions(25.0, 0.45, 303.0, 283.0)
    _forecaster(model).rollout(
        _state(),
        (CoolingAction(0.3), CoolingAction(0.4)),
        (first_condition, ForecastConditions(26.0, 0.4, 304.0, 284.0)),
    )

    stats = _normalization()
    assert len(observed_inputs) == 2
    second_features = observed_inputs[1][0]
    for index, value in (
        (0, first_condition.current_a),
        (1, first_condition.soc_fraction),
        (4, first_condition.ambient_temperature_k),
        (5, first_condition.coolant_inlet_temperature_k),
    ):
        assert second_features[index].item() == pytest.approx(
            (value - stats.feature_mean[index]) / stats.feature_scale[index]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_step_s": 0.0},
        {"time_step_s": float("nan")},
        {"minimum_coolant_flow_kg_s": -0.1},
        {"maximum_coolant_flow_kg_s": float("inf")},
        {"minimum_coolant_flow_kg_s": 0.9, "maximum_coolant_flow_kg_s": 0.1},
    ],
)
def test_constructor_rejects_invalid_time_step_and_flow_limits(
    kwargs: dict[str, float],
) -> None:
    values: dict[str, object] = {
        "model": _model(),
        "normalization": _normalization(),
        "time_step_s": 1.0,
        "minimum_coolant_flow_kg_s": 0.1,
        "maximum_coolant_flow_kg_s": 0.8,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        MLPThermalForecaster(**values)  # type: ignore[arg-type]


def test_constructor_checks_model_normalization_schema_and_dtype() -> None:
    with pytest.raises(TypeError):
        MLPThermalForecaster(object(), _normalization(), 1.0, 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        MLPThermalForecaster(_model(), object(), 1.0, 0.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_dim"):
        MLPThermalForecaster(
            ThermalMLP(MLPConfig(input_dim=8, output_dim=2, hidden_widths=(4,))),
            _normalization(),
            1.0,
            0.0,
            1.0,
        )
    wrong_names = NormalizationStats(
        ("wrong", *_normalization().feature_names[1:]),
        target_names(2),
        np.zeros(7),
        np.ones(7),
        np.zeros(2),
        np.ones(2),
    )
    with pytest.raises(ValueError, match="feature_names"):
        MLPThermalForecaster(_model(), wrong_names, 1.0, 0.0, 1.0)
    wrong_targets = NormalizationStats(
        feature_names(2),
        ("wrong", target_names(2)[1]),
        np.zeros(7),
        np.ones(7),
        np.zeros(2),
        np.ones(2),
    )
    with pytest.raises(ValueError, match="target_names"):
        MLPThermalForecaster(_model(), wrong_targets, 1.0, 0.0, 1.0)
    malformed_dimensions = _normalization()
    object.__setattr__(malformed_dimensions, "feature_mean", np.zeros(6))
    with pytest.raises(ValueError, match="dimensions"):
        MLPThermalForecaster(_model(), malformed_dimensions, 1.0, 0.0, 1.0)
    mixed_dtype_model = _model(dtype=torch.float32)
    mixed_dtype_model.network[-1].double()
    with pytest.raises(ValueError, match="one floating dtype"):
        MLPThermalForecaster(mixed_dtype_model, _normalization(), 1.0, 0.0, 1.0)


def test_predict_next_validates_state_action_and_model_output() -> None:
    forecaster = _forecaster()
    with pytest.raises(TypeError):
        forecaster.predict_next(object(), CoolingAction(0.2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cell count"):
        forecaster.predict_next(_state((300.0,)), CoolingAction(0.2))
    with pytest.raises(ValueError, match="bounds"):
        forecaster.predict_next(_state(), CoolingAction(0.9))

    model = _model()
    model.forward = lambda inputs: torch.zeros((1, 3), dtype=inputs.dtype)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="output"):
        _forecaster(model).predict_next(_state(), CoolingAction(0.2))

    model = _model()
    model.forward = lambda inputs: torch.full((1, 2), float("nan"), dtype=inputs.dtype)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="finite"):
        _forecaster(model).predict_next(_state(), CoolingAction(0.2))

    model = _model()
    model.forward = lambda inputs: torch.zeros((1, 2), dtype=torch.float32)  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="device and dtype"):
        _forecaster(model).predict_next(_state(), CoolingAction(0.2))


@pytest.mark.parametrize(
    ("actions", "conditions"),
    [
        ((), None),
        ([CoolingAction(0.2)], None),
        ((CoolingAction(0.2),), ()),
        ((CoolingAction(0.2),), (object(),)),
    ],
)
def test_rollout_requires_valid_immutable_tuples(
    actions: object, conditions: object
) -> None:
    with pytest.raises(ValueError):
        _forecaster().rollout(_state(), actions, conditions)  # type: ignore[arg-type]


def test_rollout_requires_matching_condition_length_and_valid_action_types() -> None:
    forecaster = _forecaster()
    with pytest.raises(ValueError, match="length"):
        forecaster.rollout(
            _state(), (CoolingAction(0.2),), (ForecastConditions(0, 0.5, 295, 290),) * 2
        )
    with pytest.raises(ValueError, match="CoolingAction"):
        forecaster.rollout(_state(), (object(),))  # type: ignore[arg-type]


def test_adapter_satisfies_runtime_thermal_forecaster_protocol() -> None:
    assert isinstance(_forecaster(), ThermalForecaster)
