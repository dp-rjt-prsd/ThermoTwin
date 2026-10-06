"""Physical-unit evaluation for one-step thermal delta models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from thermotwin.data.model_dataset import ThermalTransitionDataset
from thermotwin.data.preprocessing import NormalizationStats

FloatArray = NDArray[np.float64]


def _metric_array(value: object, label: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if value.ndim != 2:
        raise ValueError(f"{label} must have rank 2")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{label} must have a nonempty shape")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain valid real values") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Immutable, JSON-compatible regression metrics in kelvin."""

    mae_kelvin: float
    rmse_kelvin: float
    per_cell_mae_kelvin: tuple[float, ...]
    per_cell_rmse_kelvin: tuple[float, ...]
    maximum_absolute_error_kelvin: float
    worst_cell_index: int

    def __post_init__(self) -> None:
        scalar_names = (
            "mae_kelvin",
            "rmse_kelvin",
            "maximum_absolute_error_kelvin",
        )
        for name in scalar_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative real number")
            normalized = float(value)
            if not isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
            object.__setattr__(self, name, normalized)

        for name in ("per_cell_mae_kelvin", "per_cell_rmse_kelvin"):
            values = getattr(self, name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{name} must be a nonempty tuple")
            normalized_values: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{name} values must be finite and nonnegative")
                normalized = float(value)
                if not isfinite(normalized) or normalized < 0.0:
                    raise ValueError(f"{name} values must be finite and nonnegative")
                normalized_values.append(normalized)
            object.__setattr__(self, name, tuple(normalized_values))

        if len(self.per_cell_mae_kelvin) != len(self.per_cell_rmse_kelvin):
            raise ValueError("per-cell metric tuples must have matching lengths")
        if (
            isinstance(self.worst_cell_index, bool)
            or type(self.worst_cell_index) is not int
            or not 0 <= self.worst_cell_index < len(self.per_cell_rmse_kelvin)
        ):
            raise ValueError("worst_cell_index must identify a per-cell metric")
        expected_worst = max(
            range(len(self.per_cell_rmse_kelvin)),
            key=lambda index: (self.per_cell_rmse_kelvin[index], -index),
        )
        if self.worst_cell_index != expected_worst:
            raise ValueError("worst_cell_index must identify the highest-RMSE cell")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready scalars and arrays without exposing mutable state."""
        return {
            "mae_kelvin": self.mae_kelvin,
            "rmse_kelvin": self.rmse_kelvin,
            "per_cell_mae_kelvin": list(self.per_cell_mae_kelvin),
            "per_cell_rmse_kelvin": list(self.per_cell_rmse_kelvin),
            "maximum_absolute_error_kelvin": self.maximum_absolute_error_kelvin,
            "worst_cell_index": self.worst_cell_index,
        }


def compute_regression_metrics(
    predicted: object, observed: object
) -> RegressionMetrics:
    """Calculate exact aggregate and per-cell errors for physical-unit arrays."""
    prediction = _metric_array(predicted, "predicted")
    target = _metric_array(observed, "observed")
    if prediction.shape != target.shape:
        raise ValueError("predicted and observed arrays must have identical shapes")

    errors = prediction - target
    absolute_errors = np.abs(errors)
    squared_errors = np.square(errors)
    per_cell_mae = absolute_errors.mean(axis=0, dtype=np.float64)
    per_cell_rmse = np.sqrt(squared_errors.mean(axis=0, dtype=np.float64))
    return RegressionMetrics(
        mae_kelvin=float(absolute_errors.mean(dtype=np.float64)),
        rmse_kelvin=float(sqrt(float(squared_errors.mean(dtype=np.float64)))),
        per_cell_mae_kelvin=tuple(float(value) for value in per_cell_mae),
        per_cell_rmse_kelvin=tuple(float(value) for value in per_cell_rmse),
        maximum_absolute_error_kelvin=float(absolute_errors.max()),
        worst_cell_index=int(np.argmax(per_cell_rmse)),
    )


def _evaluation_device(value: object) -> torch.device:
    if not isinstance(value, (str, torch.device)):
        raise TypeError("device must be 'cpu', 'cuda', or a torch.device")
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("device must identify CPU or CUDA") from error
    if device.type == "cpu":
        if str(device) != "cpu":
            raise ValueError("CPU device must not include an index")
        return torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        if (
            device.index is not None
            and not 0 <= device.index < torch.cuda.device_count()
        ):
            raise ValueError(f"CUDA device index {device.index} is unavailable")
        return device
    raise ValueError("device must be CPU or an available CUDA device")


def _model_device(model: nn.Module) -> torch.device:
    tensor: Tensor | None = next(iter(model.parameters()), None)
    if tensor is None:
        tensor = next(iter(model.buffers()), None)
    return torch.device("cpu") if tensor is None else tensor.device


def evaluate_model(
    model: nn.Module,
    dataset: ThermalTransitionDataset,
    normalization: NormalizationStats,
    batch_size: int,
    device: str | torch.device,
) -> RegressionMetrics:
    """Evaluate normalized model deltas after train-statistics denormalization."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(dataset, ThermalTransitionDataset):
        raise TypeError("dataset must be a ThermalTransitionDataset")
    if not isinstance(normalization, NormalizationStats):
        raise TypeError("normalization must be NormalizationStats")
    if isinstance(batch_size, bool) or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if dataset.feature_count != len(normalization.feature_names):
        raise ValueError("dataset feature dimension does not match normalization")
    if dataset.target_count != len(normalization.target_names):
        raise ValueError("dataset target dimension does not match normalization")
    selected_device = _evaluation_device(device)
    original_device = _model_device(model)
    prior_training_mode = model.training
    predictions: list[NDArray[np.float64]] = []
    observations: list[NDArray[np.float64]] = []

    try:
        model.to(selected_device)
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.inference_mode():
            for features, targets in loader:
                features = features.to(selected_device)
                targets = targets.to(selected_device)
                predicted_targets = model(features)
                if (
                    not isinstance(predicted_targets, Tensor)
                    or predicted_targets.ndim != 2
                ):
                    raise ValueError("model output must be a rank-2 tensor")
                if predicted_targets.shape != targets.shape:
                    raise ValueError("model output shape must match dataset targets")
                if predicted_targets.shape[1] != len(normalization.target_names):
                    raise ValueError(
                        "model output dimension does not match normalization"
                    )
                if not torch.isfinite(predicted_targets).all().item():
                    raise ValueError("model output must contain only finite values")
                predictions.append(
                    cast(NDArray[np.float64], predicted_targets.cpu().numpy())
                )
                observations.append(cast(NDArray[np.float64], targets.cpu().numpy()))
    finally:
        model.to(original_device)
        model.train(prior_training_mode)

    normalized_predictions = np.concatenate(predictions, axis=0).astype(
        np.float64, copy=False
    )
    normalized_observations = np.concatenate(observations, axis=0).astype(
        np.float64, copy=False
    )
    physical_predictions = normalization.inverse_transform_targets(
        normalized_predictions
    )
    physical_observations = normalization.inverse_transform_targets(
        normalized_observations
    )
    return compute_regression_metrics(physical_predictions, physical_observations)
