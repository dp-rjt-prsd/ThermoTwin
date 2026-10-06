"""Focused contract tests for the data-driven thermal delta MLP."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import pytest
import torch
from torch import nn

from thermotwin.models.config import MLPConfig
from thermotwin.models.mlp import ThermalMLP


@pytest.fixture
def baseline_config() -> MLPConfig:
    return MLPConfig(input_dim=17, hidden_widths=(64, 32), output_dim=12)


def test_baseline_architecture_and_forward_contract(baseline_config: MLPConfig) -> None:
    model = ThermalMLP(baseline_config)
    layers = list(model.network)

    assert [type(layer) for layer in layers] == [
        nn.Linear,
        nn.GELU,
        nn.Linear,
        nn.GELU,
        nn.Linear,
    ]
    linear_dimensions = [
        (layer.in_features, layer.out_features)
        for layer in layers
        if isinstance(layer, nn.Linear)
    ]
    assert linear_dimensions == [
        (17, 64),
        (64, 32),
        (32, 12),
    ]
    inputs = torch.ones((5, 17), dtype=torch.float32)
    output = model(inputs)
    assert output.shape == (5, 12)
    assert output.dtype == torch.float32


def test_construction_is_deterministic_under_manual_seed(
    baseline_config: MLPConfig,
) -> None:
    torch.manual_seed(73)
    first = ThermalMLP(baseline_config)
    torch.manual_seed(73)
    second = ThermalMLP(baseline_config)

    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        (torch.ones(17), ValueError, "rank 2"),
        (torch.ones((2, 16)), ValueError, "input_dim 17"),
        (torch.empty((0, 17)), ValueError, "nonempty"),
        (torch.ones((2, 17), dtype=torch.int64), TypeError, "floating"),
        (torch.ones((2, 17), dtype=torch.bool), TypeError, "floating"),
        (torch.full((2, 17), float("nan")), ValueError, "finite"),
        (torch.full((2, 17), float("inf")), ValueError, "finite"),
    ],
)
def test_forward_rejects_invalid_inputs(
    baseline_config: MLPConfig,
    inputs: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ThermalMLP(baseline_config)(inputs)


def test_forward_propagates_gradients_to_every_parameter(
    baseline_config: MLPConfig,
) -> None:
    model = ThermalMLP(baseline_config)
    model(torch.randn(4, 17)).square().mean().backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_architecture_metadata_is_immutable_and_round_trips(
    baseline_config: MLPConfig,
) -> None:
    metadata = ThermalMLP(baseline_config).architecture_metadata
    encoded = json.dumps(asdict(metadata))
    restored = MLPConfig(**json.loads(encoded))

    assert restored == baseline_config
    with pytest.raises(FrozenInstanceError):
        metadata.input_dim = 9  # type: ignore[misc]
