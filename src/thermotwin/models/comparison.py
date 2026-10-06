"""Read-only evaluation of model accuracy and physical consistency."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from numbers import Real
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.physics_dataset import PhysicsTransitionDataset
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.evaluation import RegressionMetrics, compute_regression_metrics
from thermotwin.models.losses import physics_residual_w


@dataclass(frozen=True, slots=True)
class ModelComparisonMetrics:
    """Immutable, JSON-friendly accuracy and physics-consistency metrics."""

    regression: RegressionMetrics
    mean_absolute_physics_residual_w: float
    root_mean_square_physics_residual_w: float
    maximum_absolute_physics_residual_w: float

    def __post_init__(self) -> None:
        if type(self.regression) is not RegressionMetrics:
            raise ValueError("regression must be RegressionMetrics")
        for name in (
            "mean_absolute_physics_residual_w",
            "root_mean_square_physics_residual_w",
            "maximum_absolute_physics_residual_w",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative real number")
            normalized = float(value)
            if not isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
            object.__setattr__(self, name, normalized)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable metrics without mutable internal state."""
        return {
            "regression": self.regression.to_dict(),
            "mean_absolute_physics_residual_w": self.mean_absolute_physics_residual_w,
            "root_mean_square_physics_residual_w": (
                self.root_mean_square_physics_residual_w
            ),
            "maximum_absolute_physics_residual_w": (
                self.maximum_absolute_physics_residual_w
            ),
        }


def _model_device(model: nn.Module) -> torch.device:
    tensors = (*model.parameters(), *model.buffers())
    if not tensors:
        return torch.device("cpu")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("model parameters and buffers must share one device")
    return device


def _validate_setup(
    model: object,
    dataset: object,
    normalization: object,
    batch_size: object,
) -> tuple[nn.Module, PhysicsTransitionDataset, NormalizationStats]:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(dataset, PhysicsTransitionDataset):
        raise TypeError("dataset must be a PhysicsTransitionDataset")
    if type(normalization) is not NormalizationStats:
        raise TypeError("normalization must be a NormalizationStats")
    if isinstance(batch_size, bool) or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(dataset) == 0:
        raise ValueError("dataset must be nonempty")
    if dataset.feature_count != len(normalization.feature_names):
        raise ValueError("dataset feature dimension does not match normalization")
    if dataset.target_count != dataset.cell_count:
        raise ValueError("dataset target and physics cell counts must match")
    if normalization.feature_names != feature_names(dataset.cell_count):
        raise ValueError("normalization feature_names do not match dataset schema")
    if normalization.target_names != target_names(dataset.cell_count):
        raise ValueError("normalization target_names do not match dataset schema")
    if (
        normalization.target_mean.shape != (dataset.cell_count,)
        or normalization.target_scale.shape != (dataset.cell_count,)
    ):
        raise ValueError("normalization target statistics do not match dataset")
    return model, dataset, normalization


def _validate_sample(
    sample: object, dataset: PhysicsTransitionDataset
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(sample, (tuple, list)) or len(sample) != 3:
        raise ValueError("each dataset sample must contain three tensors")
    features, targets, contexts = sample
    values = (features, targets, contexts)
    names = ("features", "targets", "physics_contexts")
    for value, name in zip(values, names, strict=True):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} sample must have rank 1")
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    expected_shapes = (
        (dataset.feature_count,),
        (dataset.target_count,),
        (dataset.target_count + 2,),
    )
    for value, name, shape in zip(values, names, expected_shapes, strict=True):
        if value.shape != shape:
            raise ValueError(f"{name} sample must have shape {shape}")
    if len({value.dtype for value in values}) != 1:
        raise ValueError("features, targets, and physics_contexts must share a dtype")
    return cast(tuple[Tensor, Tensor, Tensor], values)


def evaluate_physics_model(
    model: nn.Module,
    dataset: PhysicsTransitionDataset,
    normalization: NormalizationStats,
    batch_size: int = 256,
) -> ModelComparisonMetrics:
    """Evaluate normalized delta predictions and watt residuals without mutation."""
    model, dataset, normalization = _validate_setup(
        model, dataset, normalization, batch_size
    )
    device = _model_device(model)
    module_modes = tuple((module, module.training) for module in model.modules())
    normalized_predictions: list[NDArray[np.float64]] = []
    normalized_targets: list[NDArray[np.float64]] = []
    residual_abs_sum = 0.0
    residual_square_sum = 0.0
    residual_count = 0
    maximum_residual = 0.0

    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                rows = [
                    _validate_sample(dataset[index], dataset)
                    for index in range(start, min(start + batch_size, len(dataset)))
                ]
                features = torch.stack([row[0] for row in rows], dim=0)
                targets = torch.stack([row[1] for row in rows], dim=0)
                contexts = torch.stack([row[2] for row in rows], dim=0)
                features = features.to(device=device).clone()
                targets = targets.to(device=device)
                contexts = contexts.to(device=device)
                predictions = model(features)
                if not isinstance(predictions, Tensor) or predictions.ndim != 2:
                    raise ValueError("model output must be a rank-2 tensor")
                if predictions.shape != targets.shape:
                    raise ValueError("model output shape must match dataset targets")
                if predictions.device != device:
                    raise ValueError("model output must remain on the model device")
                if predictions.dtype != targets.dtype:
                    raise ValueError("model output dtype must match dataset targets")
                if not torch.isfinite(predictions).all().item():
                    raise ValueError("model output must contain only finite values")

                residuals = physics_residual_w(predictions, contexts, normalization)
                residual_array = cast(
                    NDArray[np.float64],
                    residuals.detach().cpu().numpy().astype(np.float64, copy=False),
                )
                absolute = np.abs(residual_array)
                residual_abs_sum += float(absolute.sum(dtype=np.float64))
                residual_square_sum += float(
                    np.square(residual_array).sum(dtype=np.float64)
                )
                residual_count += int(residual_array.size)
                maximum_residual = max(maximum_residual, float(absolute.max()))
                normalized_predictions.append(
                    cast(
                        NDArray[np.float64],
                        predictions.detach().cpu().numpy().astype(
                            np.float64, copy=True
                        ),
                    )
                )
                normalized_targets.append(
                    cast(
                        NDArray[np.float64],
                        targets.detach().cpu().numpy().astype(np.float64, copy=True),
                    )
                )
    finally:
        for module, training in module_modes:
            module.training = training

    prediction_array = np.concatenate(normalized_predictions, axis=0)
    target_array = np.concatenate(normalized_targets, axis=0)
    physical_predictions = normalization.inverse_transform_targets(prediction_array)
    physical_targets = normalization.inverse_transform_targets(target_array)
    regression = compute_regression_metrics(physical_predictions, physical_targets)
    return ModelComparisonMetrics(
        regression=regression,
        mean_absolute_physics_residual_w=residual_abs_sum / residual_count,
        root_mean_square_physics_residual_w=sqrt(residual_square_sum / residual_count),
        maximum_absolute_physics_residual_w=maximum_residual,
    )
