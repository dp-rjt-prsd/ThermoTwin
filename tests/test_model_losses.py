"""Tests for differentiable physics-informed thermal losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import MLPConfig, PhysicsLossConfig
from thermotwin.models.losses import combined_thermal_loss, physics_residual_w
from thermotwin.models.mlp import ThermalMLP


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        ("feature",),
        ("delta_cell_temperature_0_k", "delta_cell_temperature_1_k"),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0, -2.0]),
        np.array([2.0, 4.0]),
    )


def _equilibrium_inputs() -> tuple[Tensor, Tensor]:
    # Physical deltas are 0.2 K and -0.1 K for this heat balance.
    physical_deltas = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    normalized = (physical_deltas - torch.tensor([1.0, -2.0])) / torch.tensor(
        [2.0, 4.0]
    )
    contexts = torch.tensor([[2.0, 10.0, 1.0, -0.5]], dtype=torch.float32)
    return normalized, contexts


def test_physics_residual_is_zero_for_euler_heat_balance() -> None:
    predictions, contexts = _equilibrium_inputs()

    residual = physics_residual_w(predictions, contexts, _normalization())

    torch.testing.assert_close(
        residual, torch.zeros_like(residual), atol=2e-7, rtol=0
    )


def test_incorrect_temperature_derivative_increases_residual_and_loss() -> None:
    predictions, contexts = _equilibrium_inputs()
    targets = predictions.clone()
    config = PhysicsLossConfig(lambda_physics=2.0, residual_scale_w=1.0)
    correct = combined_thermal_loss(
        predictions, targets, contexts, _normalization(), config
    )
    incorrect = combined_thermal_loss(
        predictions + 1.0, targets, contexts, _normalization(), config
    )

    assert incorrect.physics_mse > correct.physics_mse
    assert incorrect.total > correct.total


def test_combined_loss_uses_exact_weight_and_supports_zero_physics_weight() -> None:
    predictions, contexts = _equilibrium_inputs()
    predictions = predictions + 0.25
    targets = predictions + 1.0
    config = PhysicsLossConfig(lambda_physics=3.5, residual_scale_w=2.0)

    result = combined_thermal_loss(
        predictions, targets, contexts, _normalization(), config
    )

    assert result.physics_mse > 0.0
    torch.testing.assert_close(
        result.total, result.data_mse + 3.5 * result.physics_mse
    )
    disabled = combined_thermal_loss(
        predictions,
        targets,
        contexts,
        _normalization(),
        PhysicsLossConfig(lambda_physics=0.0, residual_scale_w=2.0),
    )
    torch.testing.assert_close(disabled.total, disabled.data_mse)


def test_residual_denormalizes_prediction_using_kelvin_delta_stats() -> None:
    predictions = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    contexts = torch.tensor([[4.0, 8.0, 3.0, -1.0]], dtype=torch.float64)
    normalization = _normalization()

    residual = physics_residual_w(predictions, contexts, normalization)

    physical_delta = predictions * torch.tensor([2.0, 4.0]) + torch.tensor(
        [1.0, -2.0]
    )
    expected = 8.0 * physical_delta / 4.0 - torch.tensor([3.0, -1.0])
    torch.testing.assert_close(residual, expected)
    assert residual.dtype == predictions.dtype
    assert residual.device == predictions.device


@pytest.mark.parametrize(
    ("predictions", "contexts", "normalization", "error"),
    [
        (
            np.zeros((1, 2)),
            torch.ones((1, 4)),
            _normalization(),
            TypeError,
        ),
        (
            torch.ones((2,)),
            torch.ones((1, 4)),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((0, 2)),
            torch.ones((0, 4)),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 3)),
            torch.ones((1, 5)),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 2), dtype=torch.int64),
            torch.ones((1, 4)),
            _normalization(),
            TypeError,
        ),
        (
            torch.tensor([[float("nan"), 0.0]]),
            torch.ones((1, 4)),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[1.0, 1.0, float("inf"), 0.0]]),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            object(),
            TypeError,
        ),
        (
            torch.ones((1, 2)),
            torch.tensor([[1.0, -1.0, 0.0, 0.0]]),
            _normalization(),
            ValueError,
        ),
        (
            torch.ones((1, 2)),
            torch.ones((1, 3)),
            NormalizationStats(
                ("feature",),
                ("cell_0", "cell_1"),
                np.array([0.0]),
                np.array([1.0]),
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
            ),
            ValueError,
        ),
    ],
)
def test_physics_residual_rejects_malformed_inputs(
    predictions: object,
    contexts: object,
    normalization: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        physics_residual_w(predictions, contexts, normalization)


def test_combined_loss_rejects_nonfinite_targets_and_mismatched_batch() -> None:
    predictions, contexts = _equilibrium_inputs()
    with pytest.raises(ValueError, match="finite"):
        combined_thermal_loss(
            predictions,
            torch.tensor([[float("inf"), 0.0]]),
            contexts,
            _normalization(),
            PhysicsLossConfig(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="matching"):
        combined_thermal_loss(
            predictions,
            predictions.expand(2, -1),
            contexts,
            _normalization(),
            PhysicsLossConfig(1.0, 1.0),
        )


def test_thermal_mlp_combined_loss_backpropagates_finite_nonzero_gradients() -> None:
    torch.manual_seed(7)
    model = ThermalMLP(MLPConfig(input_dim=3, output_dim=2, hidden_widths=(5,)))
    inputs = torch.tensor(
        [[0.1, 0.2, -0.3], [0.4, -0.2, 0.5]], dtype=torch.float32
    )
    targets = torch.tensor([[0.5, -0.2], [-0.1, 0.3]], dtype=torch.float32)
    contexts = torch.tensor(
        [[1.0, 10.0, 2.0, -1.0], [2.0, 12.0, 1.0, 0.5]], dtype=torch.float32
    )
    loss = combined_thermal_loss(
        model(inputs),
        targets,
        contexts,
        _normalization(),
        PhysicsLossConfig(lambda_physics=0.5, residual_scale_w=3.0),
    )

    loss.total.backward()  # type: ignore[no-untyped-call]

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
        if gradient is not None
    )
