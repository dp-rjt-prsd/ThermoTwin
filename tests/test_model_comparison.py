"""Tests for physical consistency and accuracy comparison of thermal models."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.physics_dataset import PhysicsTransitionDataset
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.comparison import (
    ModelComparisonMetrics,
    evaluate_physics_model,
)


class _ConstantModel(nn.Module):
    def __init__(self, outputs: tuple[float, float]) -> None:
        super().__init__()
        self.output = nn.Parameter(torch.tensor(outputs, dtype=torch.float32))
        self.saw_eval = False
        self.saw_inference_mode = False

    def forward(self, inputs: Tensor) -> Tensor:
        self.saw_eval = not self.training
        self.saw_inference_mode = torch.is_inference_mode_enabled()
        return self.output.expand(inputs.shape[0], -1)


def _inputs(
    *,
    contexts: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> tuple[PhysicsTransitionDataset, NormalizationStats]:
    row_count = 5
    normalization = NormalizationStats(
        feature_names(2),
        target_names(2),
        np.zeros(7),
        np.ones(7),
        np.array([10.0, 20.0]),
        np.array([2.0, 4.0]),
    )
    target_values = (
        np.array(
            [[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [-1.0, 0.25], [0.25, -1.0]],
            dtype=np.float32,
        )
        if targets is None
        else targets
    )
    physics_values = (
        np.array(
            [
                [2.0, 4.0, 20.0, 40.0],
                [2.0, 4.0, 21.0, 39.0],
                [2.0, 4.0, 22.0, 38.0],
                [2.0, 4.0, 23.0, 37.0],
                [2.0, 4.0, 24.0, 36.0],
            ],
            dtype=np.float32,
        )
        if contexts is None
        else contexts
    )
    dataset = PhysicsTransitionDataset(
        np.arange(row_count * 7, dtype=np.float32).reshape(row_count, 7),
        target_values,
        physics_values,
        ("trajectory",),
    )
    return dataset, normalization


def test_evaluation_aggregates_unequal_batches_with_kelvin_and_watt_metrics() -> None:
    dataset, normalization = _inputs()
    model = _ConstantModel((0.5, -0.25))
    original_features = dataset.features.copy()
    original_targets = dataset.targets.copy()
    original_contexts = dataset.physics_contexts.copy()
    original_parameter = model.output.detach().clone()

    result = evaluate_physics_model(model, dataset, normalization, batch_size=2)

    physical_prediction = np.tile(np.array([[11.0, 19.0]]), (len(dataset), 1))
    physical_target = normalization.inverse_transform_targets(
        dataset.targets.astype(np.float64)
    )
    errors = physical_prediction - physical_target
    residuals = np.array(
        [[2.0, -2.0], [1.0, -1.0], [0.0, 0.0], [-1.0, 1.0], [-2.0, 2.0]]
    )

    assert result.regression.mae_kelvin == pytest.approx(np.abs(errors).mean())
    assert result.regression.rmse_kelvin == pytest.approx(
        np.sqrt(np.square(errors).mean())
    )
    assert result.mean_absolute_physics_residual_w == pytest.approx(
        np.abs(residuals).sum() / residuals.size
    )
    assert result.root_mean_square_physics_residual_w == pytest.approx(
        np.sqrt(np.square(residuals).sum() / residuals.size)
    )
    assert result.maximum_absolute_physics_residual_w == 2.0
    assert model.saw_eval and model.saw_inference_mode
    assert model.training
    assert torch.equal(model.output, original_parameter)
    np.testing.assert_array_equal(dataset.features, original_features)
    np.testing.assert_array_equal(dataset.targets, original_targets)
    np.testing.assert_array_equal(dataset.physics_contexts, original_contexts)
    assert json.loads(json.dumps(result.to_dict()))["regression"]["mae_kelvin"] == (
        result.regression.mae_kelvin
    )


def test_metrics_record_is_frozen_and_json_friendly() -> None:
    dataset, normalization = _inputs()
    result = evaluate_physics_model(
        _ConstantModel((0.5, -0.25)), dataset, normalization, batch_size=3
    )

    assert isinstance(result, ModelComparisonMetrics)
    with pytest.raises((AttributeError, TypeError)):
        result.mean_absolute_physics_residual_w = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_evaluation_rejects_invalid_batch_size(batch_size: object) -> None:
    dataset, normalization = _inputs()
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_physics_model(
            _ConstantModel((0.0, 0.0)),
            dataset,
            normalization,
            batch_size,  # type: ignore[arg-type]
        )


def test_evaluation_rejects_wrong_physics_target_schema() -> None:
    dataset, normalization = _inputs()
    invalid = NormalizationStats(
        normalization.feature_names,
        ("wrong_0", "wrong_1"),
        normalization.feature_mean,
        normalization.feature_scale,
        normalization.target_mean,
        normalization.target_scale,
    )

    with pytest.raises(ValueError, match="target_names"):
        evaluate_physics_model(_ConstantModel((0.0, 0.0)), dataset, invalid, 2)


class _BadOutputModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.full((inputs.shape[0], 3), float("nan"))


class _NonfiniteOutputModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.full((inputs.shape[0], 2), float("nan"))


def test_evaluation_rejects_nonfinite_or_wrong_width_model_output() -> None:
    dataset, normalization = _inputs()
    with pytest.raises(ValueError, match="shape"):
        evaluate_physics_model(_BadOutputModel(), dataset, normalization, 2)
    with pytest.raises(ValueError, match="finite"):
        evaluate_physics_model(_NonfiniteOutputModel(), dataset, normalization, 2)


class _MalformedDataset(PhysicsTransitionDataset):
    malformed_sample: object

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        del index
        return cast(tuple[Tensor, Tensor, Tensor], self.malformed_sample)


def test_evaluation_rejects_nontriple_dataset_sample() -> None:
    dataset, normalization = _inputs()
    malformed = _MalformedDataset(
        dataset.features,
        dataset.targets,
        dataset.physics_contexts,
        dataset.trajectory_ids,
    )
    object.__setattr__(malformed, "malformed_sample", (torch.zeros(7), torch.zeros(2)))

    with pytest.raises(ValueError, match="three tensors"):
        evaluate_physics_model(_ConstantModel((0.0, 0.0)), malformed, normalization, 2)


@pytest.mark.parametrize(
    ("sample", "error", "message"),
    [
        ((torch.zeros(1, 7), torch.zeros(2), torch.zeros(4)), ValueError, "rank 1"),
        ((torch.zeros(7), torch.zeros(3), torch.zeros(4)), ValueError, "shape"),
        (
            (torch.zeros(7), torch.zeros(2, dtype=torch.int64), torch.zeros(4)),
            TypeError,
            "floating",
        ),
        (
            (torch.full((7,), float("nan")), torch.zeros(2), torch.zeros(4)),
            ValueError,
            "finite",
        ),
    ],
)
def test_evaluation_rejects_malformed_tensor_samples(
    sample: object, error: type[Exception], message: str
) -> None:
    dataset, normalization = _inputs()
    malformed = _MalformedDataset(
        dataset.features,
        dataset.targets,
        dataset.physics_contexts,
        dataset.trajectory_ids,
    )
    object.__setattr__(malformed, "malformed_sample", sample)

    with pytest.raises(error, match=message):
        evaluate_physics_model(_ConstantModel((0.0, 0.0)), malformed, normalization, 2)
