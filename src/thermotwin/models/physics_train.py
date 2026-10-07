"""Deterministic training for data-only and physics-informed thermal MLPs."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sized
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import MLPConfig, PhysicsLossConfig, TrainerConfig
from thermotwin.models.losses import combined_thermal_loss, physics_residual_w
from thermotwin.models.mlp import ThermalMLP

_Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class PhysicsEpochMetrics:
    """Sample-weighted loss components for one completed training epoch."""

    epoch: int
    train_total_loss: float
    train_data_mse: float
    train_physics_mse: float
    validation_total_loss: float
    validation_data_mse: float
    validation_physics_mse: float


@dataclass(frozen=True, slots=True)
class PhysicsTrainingResult:
    """Immutable, JSON-friendly summary of a physics-aware training run."""

    history: tuple[PhysicsEpochMetrics, ...]
    best_epoch: int
    best_validation_total_loss: float
    stopped_early: bool
    device: str
    checkpoint_path: str


def _resolve_device(policy: str) -> torch.device:
    if policy == "cpu":
        return torch.device("cpu")
    if policy == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if policy == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    raise ValueError("device policy must be 'auto', 'cpu', or 'cuda'")


def _validate_values(
    values: tuple[object, ...], name: str, widths: tuple[int, int, int]
) -> None:
    if len(values) != 3 or any(not isinstance(value, Tensor) for value in values):
        raise TypeError(f"{name} samples must be (features, targets, contexts) tensors")
    tensors = cast(tuple[Tensor, Tensor, Tensor], values)
    features, targets, contexts = tensors
    if features.ndim != 1 or features.shape[0] != widths[0]:
        raise ValueError(f"{name} feature width must match input_dim {widths[0]}")
    if targets.ndim != 1 or targets.shape[0] != widths[1]:
        raise ValueError(f"{name} target width must match output_dim {widths[1]}")
    if contexts.ndim != 1 or contexts.shape[0] != widths[2]:
        raise ValueError(
            f"{name} physics_contexts width must be {widths[2]} "
            "(two scalars plus output cells)"
        )
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError(f"{name} features, targets, and contexts must be float32")
    if len({value.device for value in tensors}) != 1:
        raise ValueError(f"{name} features, targets, and contexts must share a device")
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError(f"{name} tensors must contain only finite values")


def _validate_dataset(
    dataset: Dataset[_Batch], name: str, widths: tuple[int, int, int]
) -> None:
    if not isinstance(dataset, Sized):
        raise TypeError(f"{name} dataset must have a length")
    if len(dataset) <= 0:
        raise ValueError(f"{name} dataset must be nonempty")
    sample = dataset[0]
    if not isinstance(sample, (tuple, list)):
        raise TypeError(f"{name} dataset samples must contain three tensors")
    _validate_values(tuple(sample), f"{name} dataset", widths)


def _validate_batch(batch: object, name: str, widths: tuple[int, int, int]) -> _Batch:
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise TypeError(f"{name} batches must be (features, targets, contexts)")
    if any(not isinstance(value, Tensor) for value in batch):
        raise TypeError(f"{name} batches must contain tensors")
    tensors = cast(tuple[Tensor, Tensor, Tensor], tuple(batch))
    features, targets, contexts = tensors
    if features.ndim != 2 or features.shape[1] != widths[0]:
        raise ValueError(f"{name} batch feature width must match input_dim {widths[0]}")
    count = int(features.shape[0])
    if count <= 0:
        raise ValueError(f"{name} batches must be nonempty")
    if targets.shape != (count, widths[1]):
        raise ValueError(f"{name} batch target width must match output_dim {widths[1]}")
    if contexts.shape != (count, widths[2]):
        raise ValueError(f"{name} batch physics_contexts width must be {widths[2]}")
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError(f"{name} batch tensors must have dtype torch.float32")
    if len({value.device for value in tensors}) != 1:
        raise ValueError(f"{name} batch tensors must share a device")
    if any(not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError(f"{name} batch must contain only finite values")
    return features, targets, contexts


def _validate_normalization(
    normalization: object, output_dim: int
) -> NormalizationStats:
    if type(normalization) is not NormalizationStats:
        raise TypeError("normalization must be a NormalizationStats")
    expected_targets = tuple(
        f"delta_cell_temperature_{index}_k" for index in range(output_dim)
    )
    if normalization.target_names != expected_targets:
        raise ValueError(
            "normalization target_names must match the ordered cell-temperature "
            "delta schema"
        )
    if normalization.target_mean.shape != (output_dim,) or (
        normalization.target_scale.shape != (output_dim,)
    ):
        raise ValueError("normalization target statistics must match output_dim")
    return normalization


def _mean_components(
    model: ThermalMLP,
    batches: DataLoader[_Batch],
    device: torch.device,
    widths: tuple[int, int, int],
    normalization: NormalizationStats,
    loss_config: PhysicsLossConfig,
    *,
    optimizer: Adam | None = None,
    collocation: Iterator[_Batch] | None = None,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    totals = [0.0, 0.0, 0.0]
    sample_count = 0
    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for raw_batch in batches:
            features, targets, contexts = _validate_batch(
                raw_batch, "training" if is_training else "validation", widths
            )
            features = features.to(device)
            targets = targets.to(device)
            contexts = contexts.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            components = combined_thermal_loss(
                predictions, targets, contexts, normalization, loss_config
            )
            total = components.total
            if collocation is not None and loss_config.lambda_physics > 0.0:
                # Unlabeled rows: only the energy-balance residual applies.
                colloc_features, _, colloc_contexts = next(collocation)
                residual = physics_residual_w(
                    model(colloc_features.to(device)),
                    colloc_contexts.to(device),
                    normalization,
                )
                collocation_mse = torch.mean(
                    torch.square(residual / loss_config.residual_scale_w)
                )
                total = total + loss_config.lambda_physics * collocation_mse
            for component in (
                total,
                components.data_mse,
                components.physics_mse,
            ):
                if not torch.isfinite(component).item():
                    raise ValueError("training produced a nonfinite loss component")
            if optimizer is not None:
                total.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            count = int(features.shape[0])
            for index, component in enumerate(
                (total, components.data_mse, components.physics_mse)
            ):
                totals[index] += float(component.detach().item()) * count
            sample_count += count
    if sample_count == 0:
        raise ValueError("dataset must be nonempty")
    means = (
        totals[0] / sample_count,
        totals[1] / sample_count,
        totals[2] / sample_count,
    )
    if not np.isfinite(means).all():
        raise ValueError("training produced a nonfinite mean loss component")
    return means


def train_physics_aware_mlp(
    model_config: MLPConfig,
    trainer_config: TrainerConfig,
    physics_loss_config: PhysicsLossConfig,
    normalization: NormalizationStats,
    train_dataset: Dataset[_Batch],
    validation_dataset: Dataset[_Batch],
    checkpoint_path: str | Path,
    collocation_dataset: Dataset[_Batch] | None = None,
) -> PhysicsTrainingResult:
    """Train one MLP with the shared data plus weighted physics objective.

    Setting ``lambda_physics=0`` provides the data-only comparison while
    retaining this exact model, trainer, and combined-loss execution path.
    Best-state selection uses validation total loss and the same strict
    ``min_improvement`` and consecutive ``patience`` policy as the legacy
    trainer.
    """
    if not isinstance(model_config, MLPConfig):
        raise TypeError("model_config must be an MLPConfig")
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError("trainer_config must be a TrainerConfig")
    if type(physics_loss_config) is not PhysicsLossConfig:
        raise TypeError("physics_loss_config must be a PhysicsLossConfig")
    stats = _validate_normalization(normalization, model_config.output_dim)

    path = Path(checkpoint_path)
    if path.suffix != ".pt":
        raise ValueError("checkpoint path must use the .pt suffix")
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise FileExistsError(
            f"checkpoint parent path is not a directory: {path.parent}"
        )

    widths = (
        model_config.input_dim,
        model_config.output_dim,
        model_config.output_dim + 2,
    )
    _validate_dataset(train_dataset, "training", widths)
    _validate_dataset(validation_dataset, "validation", widths)
    device = _resolve_device(trainer_config.device_policy)

    random.seed(trainer_config.seed)
    np.random.seed(trainer_config.seed % (2**32))
    torch_seed = trainer_config.seed % (2**64)
    torch.manual_seed(torch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(torch_seed)

    model = ThermalMLP(model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
    )
    train_generator = torch.Generator(device="cpu").manual_seed(torch_seed)
    train_batches = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )
    validation_batches = DataLoader(
        validation_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    collocation_stream: Iterator[_Batch] | None = None
    if collocation_dataset is not None:
        _validate_dataset(collocation_dataset, "collocation", widths)
        collocation_batches = DataLoader(
            collocation_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(torch_seed + 1),
            num_workers=0,
        )

        def _cycle() -> Iterator[_Batch]:
            while True:
                yield from collocation_batches

        collocation_stream = _cycle()

    best_loss = float("inf")
    best_epoch = 0
    best_weights: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[PhysicsEpochMetrics] = []
    stopped_early = False
    for epoch in range(1, trainer_config.max_epochs + 1):
        train_total, train_data, train_physics = _mean_components(
            model,
            train_batches,
            device,
            widths,
            stats,
            physics_loss_config,
            optimizer=optimizer,
            collocation=collocation_stream,
        )
        validation_total, validation_data, validation_physics = _mean_components(
            model,
            validation_batches,
            device,
            widths,
            stats,
            physics_loss_config,
        )
        history.append(
            PhysicsEpochMetrics(
                epoch,
                train_total,
                train_data,
                train_physics,
                validation_total,
                validation_data,
                validation_physics,
            )
        )

        if validation_total < best_loss - trainer_config.min_improvement:
            best_loss = validation_total
            best_epoch = epoch
            best_weights = deepcopy(
                {
                    name: tensor.detach().to(device="cpu").clone()
                    for name, tensor in model.state_dict().items()
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= trainer_config.patience:
                stopped_early = epoch < trainer_config.max_epochs
                break

    if best_weights is None:
        raise RuntimeError("training completed without a finite validation loss")
    model.load_state_dict(best_weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    checkpoint_created = False
    try:
        with path.open("xb") as checkpoint_file:
            checkpoint_created = True
            torch.save(best_weights, checkpoint_file)
    except BaseException:
        if checkpoint_created and path.exists():
            path.unlink()
        raise

    return PhysicsTrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_total_loss=best_loss,
        stopped_early=stopped_early,
        device=str(device),
        checkpoint_path=str(path),
    )
