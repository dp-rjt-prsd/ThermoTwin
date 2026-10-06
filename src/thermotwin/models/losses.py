"""Differentiable data and physics losses for thermal surrogate models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import PhysicsLossConfig


@dataclass(frozen=True, slots=True)
class LossComponents:
    """Scalar components of the normalized thermal training objective."""

    total: Tensor
    data_mse: Tensor
    physics_mse: Tensor


def _validate_matrix(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have rank 2 [batch, cells]")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} batch and cell dimensions must be nonempty")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_normalization(
    normalization: object, cell_count: int
) -> NormalizationStats:
    if type(normalization) is not NormalizationStats:
        raise TypeError("normalization must be a NormalizationStats")
    expected_names = tuple(
        f"delta_cell_temperature_{index}_k" for index in range(cell_count)
    )
    if normalization.target_names != expected_names:
        raise ValueError(
            "normalization target_names must match the ordered cell-temperature "
            "delta schema"
        )
    if normalization.target_mean.shape != (cell_count,) or (
        normalization.target_scale.shape != (cell_count,)
    ):
        raise ValueError("normalization target statistics must match the cell count")
    return normalization


def _validate_contexts(
    physics_contexts: object,
    predictions: Tensor,
) -> Tensor:
    contexts = _validate_matrix(physics_contexts, "physics_contexts")
    if contexts.shape[0] != predictions.shape[0]:
        raise ValueError("physics_contexts and predictions must have matching rows")
    if contexts.shape[1] != predictions.shape[1] + 2:
        raise ValueError(
            "physics_contexts width must equal two scalars plus prediction cells"
        )
    if contexts.device != predictions.device:
        raise ValueError("physics_contexts and predictions must be on the same device")
    if contexts.dtype != predictions.dtype:
        raise ValueError("physics_contexts and predictions must have the same dtype")
    if torch.any(contexts[:, 0] <= 0.0).item():
        raise ValueError("physics_contexts dt_s values must be positive")
    if torch.any(contexts[:, 1] <= 0.0).item():
        raise ValueError(
            "physics_contexts cell_thermal_capacity_j_k values must be positive"
        )
    return contexts


def _require_finite_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not torch.isfinite(value).item():
        raise ValueError(f"{name} must be finite")


def physics_residual_w(
    predicted_normalized_deltas: object,
    physics_contexts: object,
    normalization: object,
) -> Tensor:
    """Return per-cell watt residuals for normalized one-step delta predictions.

    Context columns follow ``[dt_s, cell_thermal_capacity_j_k, net_heat_w...]``.
    Predictions are denormalized with the training target statistics using tensor
    operations, preserving autograd through the predicted deltas.
    """
    predictions = _validate_matrix(
        predicted_normalized_deltas, "predicted_normalized_deltas"
    )
    stats = _validate_normalization(normalization, predictions.shape[1])
    contexts = _validate_contexts(physics_contexts, predictions)

    target_mean = torch.tensor(
        stats.target_mean, dtype=predictions.dtype, device=predictions.device
    )
    target_scale = torch.tensor(
        stats.target_scale, dtype=predictions.dtype, device=predictions.device
    )
    dt_s = contexts[:, 0:1]
    capacity_j_k = contexts[:, 1:2]
    net_heat_w = contexts[:, 2:]
    predicted_delta_k = predictions * target_scale + target_mean
    residual_w = capacity_j_k * predicted_delta_k / dt_s - net_heat_w
    if not torch.isfinite(residual_w).all().item():
        raise ValueError("physics residual must contain only finite values")
    return residual_w


def combined_thermal_loss(
    predictions: object,
    targets: object,
    physics_contexts: object,
    normalization: object,
    config: object,
) -> LossComponents:
    """Combine normalized-delta MSE with scaled watt-residual MSE."""
    predicted = _validate_matrix(predictions, "predictions")
    observed = _validate_matrix(targets, "targets")
    stats = _validate_normalization(normalization, predicted.shape[1])
    if type(config) is not PhysicsLossConfig:
        raise TypeError("config must be a PhysicsLossConfig")
    if observed.shape != predicted.shape:
        raise ValueError("predictions and targets must have matching shapes")
    if observed.device != predicted.device:
        raise ValueError("predictions and targets must be on the same device")
    if observed.dtype != predicted.dtype:
        raise ValueError("predictions and targets must have the same dtype")

    residual_w = physics_residual_w(predicted, physics_contexts, stats)
    residual_scale_w = torch.as_tensor(
        config.residual_scale_w, dtype=predicted.dtype, device=predicted.device
    )
    data_mse = torch.mean(torch.square(predicted - observed))
    physics_mse = torch.mean(torch.square(residual_w / residual_scale_w))
    total = data_mse + config.lambda_physics * physics_mse
    _require_finite_scalar(data_mse, "data_mse")
    _require_finite_scalar(physics_mse, "physics_mse")
    _require_finite_scalar(total, "total")
    return LossComponents(total=total, data_mse=data_mse, physics_mse=physics_mse)
