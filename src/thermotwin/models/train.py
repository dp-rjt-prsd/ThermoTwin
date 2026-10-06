"""Deterministic supervised training for the data-driven thermal MLP."""

from __future__ import annotations

import random
from collections.abc import Sized
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from thermotwin.models.config import MLPConfig, TrainerConfig
from thermotwin.models.mlp import ThermalMLP

_Batch: TypeAlias = tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Sample-weighted mean squared errors for one completed epoch."""

    epoch: int
    train_mse: float
    validation_mse: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Immutable, JSON-friendly summary of one thermal MLP training run."""

    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_mse: float
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


def _validate_dataset(
    dataset: Dataset[_Batch], name: str, input_dim: int, output_dim: int
) -> None:
    if not isinstance(dataset, Sized):
        raise TypeError(f"{name} dataset must have a length")
    sample_count = len(dataset)
    if sample_count <= 0:
        raise ValueError(f"{name} dataset must be nonempty")

    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise TypeError(f"{name} dataset samples must be (features, targets) pairs")
    features, targets = sample
    if not isinstance(features, Tensor) or not isinstance(targets, Tensor):
        raise TypeError(f"{name} dataset features and targets must be tensors")
    if features.ndim != 1 or features.shape[0] != input_dim:
        raise ValueError(
            f"{name} dataset feature width must match input_dim {input_dim}"
        )
    if targets.ndim != 1 or targets.shape[0] != output_dim:
        raise ValueError(
            f"{name} dataset target width must match output_dim {output_dim}"
        )
    if not features.is_floating_point() or not targets.is_floating_point():
        raise TypeError(f"{name} dataset features and targets must be floating tensors")
    if (
        not torch.isfinite(features).all().item()
        or not torch.isfinite(targets).all().item()
    ):
        raise ValueError(f"{name} dataset samples must contain only finite values")


def _validate_batch(batch: _Batch, name: str, input_dim: int, output_dim: int) -> None:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(f"{name} batches must be (features, targets) pairs")
    features, targets = batch
    if not isinstance(features, Tensor) or not isinstance(targets, Tensor):
        raise TypeError(f"{name} batches must contain tensors")
    if features.ndim != 2 or features.shape[1] != input_dim:
        raise ValueError(f"{name} batch feature width must match input_dim {input_dim}")
    if targets.ndim != 2 or targets.shape != (features.shape[0], output_dim):
        raise ValueError(
            f"{name} batch target width must match output_dim {output_dim}"
        )
    if not features.is_floating_point() or not targets.is_floating_point():
        raise TypeError(f"{name} batch features and targets must be floating tensors")
    if (
        not torch.isfinite(features).all().item()
        or not torch.isfinite(targets).all().item()
    ):
        raise ValueError(f"{name} batch must contain only finite values")


def _mean_loss(
    model: ThermalMLP,
    batches: DataLoader[_Batch],
    loss_function: MSELoss,
    device: torch.device,
    input_dim: int,
    output_dim: int,
    *,
    optimizer: Adam | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    grad_context = torch.enable_grad() if is_training else torch.inference_mode()
    with grad_context:
        for batch in batches:
            _validate_batch(
                batch,
                "training" if is_training else "validation",
                input_dim,
                output_dim,
            )
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = loss_function(predictions, targets)
            if not torch.isfinite(loss).item():
                raise ValueError("training produced a nonfinite MSE loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(features.shape[0])
            total_loss += float(loss.detach().item()) * count
            sample_count += count
    if sample_count == 0:
        raise ValueError("dataset must be nonempty")
    mean_loss = total_loss / sample_count
    if not np.isfinite(mean_loss):
        raise ValueError("training produced a nonfinite mean MSE loss")
    return mean_loss


def train_thermal_mlp(
    model_config: MLPConfig,
    trainer_config: TrainerConfig,
    train_dataset: Dataset[_Batch],
    validation_dataset: Dataset[_Batch],
    checkpoint_path: str | Path,
) -> TrainingResult:
    """Fit normalized targets and save the state with the best validation MSE.

    An epoch improves the best state only when its validation MSE is strictly
    less than the best MSE by more than ``min_improvement``. ``patience`` counts
    consecutive epochs without that improvement.
    """
    if not isinstance(model_config, MLPConfig):
        raise TypeError("model_config must be an MLPConfig")
    if not isinstance(trainer_config, TrainerConfig):
        raise TypeError("trainer_config must be a TrainerConfig")
    path = Path(checkpoint_path)
    if path.suffix != ".pt":
        raise ValueError("checkpoint path must use the .pt suffix")
    if path.exists():
        raise FileExistsError(f"checkpoint path already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise FileExistsError(
            f"checkpoint parent path is not a directory: {path.parent}"
        )

    _validate_dataset(
        train_dataset, "training", model_config.input_dim, model_config.output_dim
    )
    _validate_dataset(
        validation_dataset,
        "validation",
        model_config.input_dim,
        model_config.output_dim,
    )
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
    loss_function = MSELoss(reduction="mean")
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

    best_loss = float("inf")
    best_epoch = 0
    best_weights: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []
    stopped_early = False
    for epoch in range(1, trainer_config.max_epochs + 1):
        train_mse = _mean_loss(
            model,
            train_batches,
            loss_function,
            device,
            model_config.input_dim,
            model_config.output_dim,
            optimizer=optimizer,
        )
        validation_mse = _mean_loss(
            model,
            validation_batches,
            loss_function,
            device,
            model_config.input_dim,
            model_config.output_dim,
        )
        history.append(EpochMetrics(epoch, train_mse, validation_mse))

        if validation_mse < best_loss - trainer_config.min_improvement:
            best_loss = validation_mse
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

    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_mse=best_loss,
        stopped_early=stopped_early,
        device=str(device),
        checkpoint_path=str(path),
    )
