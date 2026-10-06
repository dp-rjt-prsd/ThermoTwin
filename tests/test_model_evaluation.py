"""Tests for physical-unit evaluation of thermal surrogate models."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from thermotwin.data.model_dataset import ThermalTransitionDataset
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.evaluation import (
    RegressionMetrics,
    compute_regression_metrics,
    evaluate_model,
)


def test_metrics_calculate_aggregate_and_per_cell_errors_exactly() -> None:
    predicted = np.array([[2.0, 1.0], [4.0, 3.0]])
    observed = np.array([[1.0, 3.0], [1.0, 3.0]])

    result = compute_regression_metrics(predicted, observed)

    assert result.mae_kelvin == 1.5
    assert result.rmse_kelvin == pytest.approx(np.sqrt(3.5))
    assert result.per_cell_mae_kelvin == (2.0, 1.0)
    assert result.per_cell_rmse_kelvin == (np.sqrt(5.0), np.sqrt(2.0))
    assert result.maximum_absolute_error_kelvin == 3.0
    assert result.worst_cell_index == 0
    assert json.loads(json.dumps(result.to_dict()))["worst_cell_index"] == 0


def test_worst_cell_ties_choose_the_lowest_cell_index() -> None:
    result = compute_regression_metrics(
        np.array([[1.0, -1.0]]), np.zeros((1, 2))
    )

    assert result.worst_cell_index == 0


@pytest.mark.parametrize(
    ("predicted", "observed", "exception"),
    [
        (np.zeros(2), np.zeros(2), ValueError),
        (np.zeros((0, 2)), np.zeros((0, 2)), ValueError),
        (np.zeros((2, 2)), np.zeros((2, 1)), ValueError),
        (np.zeros((1, 2), dtype=object), np.zeros((1, 2)), ValueError),
        (np.array([[np.nan]]), np.zeros((1, 1)), ValueError),
        (np.zeros((1, 1)), np.array([[np.inf]]), ValueError),
        ([[1.0]], [[1.0]], TypeError),
    ],
)
def test_metrics_reject_untrusted_or_incompatible_arrays(
    predicted: object, observed: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        compute_regression_metrics(predicted, observed)


def test_metrics_record_rejects_invalid_values_and_inconsistent_cells() -> None:
    with pytest.raises(ValueError):
        RegressionMetrics(-1.0, 0.0, (0.0,), (0.0,), 0.0, 0)
    with pytest.raises(ValueError):
        RegressionMetrics(0.0, 0.0, (0.0,), (0.0, 0.0), 0.0, 0)
    with pytest.raises(ValueError):
        RegressionMetrics(0.0, 0.0, (0.0, float("nan")), (0.0, 0.0), 0.0, 0)
    with pytest.raises(ValueError):
        RegressionMetrics(0.0, 0.0, (0.0, 0.0), (0.0, 2.0), 0.0, 0)


class _ConstantModel(nn.Module):
    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = nn.Parameter(output)
        self.saw_grad_enabled = True

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.saw_grad_enabled = torch.is_grad_enabled()
        return self.output.expand(inputs.shape[0], -1)


def _evaluation_inputs() -> tuple[
    ThermalTransitionDataset, NormalizationStats
]:
    normalization = NormalizationStats(
        ("feature",),
        ("cell_0", "cell_1"),
        np.array([0.0]),
        np.array([1.0]),
        np.array([10.0, -2.0]),
        np.array([2.0, 3.0]),
    )
    dataset = ThermalTransitionDataset(
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        np.array([[0.0, 0.0], [0.5, 1.0], [-0.5, -1.0]], dtype=np.float32),
        ("trajectory",),
    )
    return dataset, normalization


def test_evaluation_denormalizes_targets_before_calculating_metrics() -> None:
    dataset, normalization = _evaluation_inputs()
    model = _ConstantModel(torch.tensor([0.5, -0.5]))

    result = evaluate_model(model, dataset, normalization, batch_size=2, device="cpu")

    expected = normalization.inverse_transform_targets(
        dataset.targets.astype(np.float64)
    )
    predicted = np.tile(np.array([[11.0, -3.5]]), (len(dataset), 1))
    expected_errors = compute_regression_metrics(predicted, expected)
    assert result == expected_errors
    assert result.mae_kelvin == pytest.approx(1.75)


def test_evaluation_restores_training_mode_and_disables_gradients() -> None:
    dataset, normalization = _evaluation_inputs()
    model = _ConstantModel(torch.zeros(2))
    model.train()

    evaluate_model(
        model, dataset, normalization, batch_size=2, device=torch.device("cpu")
    )

    assert model.training
    assert not model.saw_grad_enabled
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_preserves_eval_mode() -> None:
    dataset, normalization = _evaluation_inputs()
    model = _ConstantModel(torch.zeros(2))
    model.eval()

    evaluate_model(model, dataset, normalization, batch_size=2, device="cpu")

    assert not model.training


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_evaluation_rejects_invalid_batch_size(batch_size: object) -> None:
    dataset, normalization = _evaluation_inputs()
    with pytest.raises(ValueError):
        evaluate_model(
            _ConstantModel(torch.zeros(2)),
            dataset,
            normalization,
            batch_size,  # type: ignore[arg-type]
            "cpu",
        )


@pytest.mark.parametrize("device", ["mps", "cuda:invalid", "not-a-device"])
def test_evaluation_rejects_invalid_or_unsupported_device(device: str) -> None:
    dataset, normalization = _evaluation_inputs()
    with pytest.raises((ValueError, TypeError)):
        evaluate_model(
            _ConstantModel(torch.zeros(2)), dataset, normalization, 2, device
        )


def test_evaluation_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    dataset, normalization = _evaluation_inputs()
    with pytest.raises(ValueError, match="CUDA is not available"):
        evaluate_model(
            _ConstantModel(torch.zeros(2)), dataset, normalization, 2, "cuda"
        )


def test_evaluation_rejects_dataset_model_dimension_mismatches() -> None:
    dataset, normalization = _evaluation_inputs()
    with pytest.raises(ValueError, match="feature"):
        evaluate_model(
            _ConstantModel(torch.zeros(2)),
            dataset,
            NormalizationStats(
                ("a", "b"),
                ("x", "y"),
                np.zeros(2),
                np.ones(2),
                np.zeros(2),
                np.ones(2),
            ),
            2,
            "cpu",
        )
    with pytest.raises(ValueError, match="output"):
        evaluate_model(_ConstantModel(torch.zeros(3)), dataset, normalization, 2, "cpu")
