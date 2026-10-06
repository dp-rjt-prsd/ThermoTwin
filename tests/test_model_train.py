from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from thermotwin.data.model_dataset import ThermalTransitionDataset
from thermotwin.models.config import MLPConfig, TrainerConfig
from thermotwin.models.mlp import ThermalMLP
from thermotwin.models.train import EpochMetrics, TrainingResult, train_thermal_mlp


def _dataset(
    features: np.ndarray, targets: np.ndarray, name: str
) -> ThermalTransitionDataset:
    return ThermalTransitionDataset(
        features.astype(np.float32),
        targets.astype(np.float32),
        (name,),
    )


def _inputs() -> tuple[ThermalTransitionDataset, ThermalTransitionDataset]:
    rng = np.random.default_rng(21)
    train_x = rng.normal(size=(24, 3)).astype(np.float32)
    train_y = (train_x[:, :2] * 0.4 - train_x[:, 2:] * 0.2).astype(np.float32)
    validation_x = rng.normal(size=(11, 3)).astype(np.float32)
    validation_y = (validation_x[:, :2] * 0.4 - validation_x[:, 2:] * 0.2).astype(
        np.float32
    )
    return _dataset(train_x, train_y, "train"), _dataset(
        validation_x, validation_y, "validation"
    )


def _configs() -> tuple[MLPConfig, TrainerConfig]:
    return (
        MLPConfig(input_dim=3, output_dim=2, hidden_widths=(8,)),
        TrainerConfig(
            seed=13,
            batch_size=7,
            learning_rate=0.025,
            max_epochs=30,
            patience=8,
            min_improvement=0.0,
            weight_decay=0.0,
            device_policy="cpu",
        ),
    )


def test_training_is_deterministic_and_saves_reloadable_best_weights(
    tmp_path: Path,
) -> None:
    train, validation = _inputs()
    model_config, trainer_config = _configs()
    first_path = tmp_path / "one" / "best.pt"
    second_path = tmp_path / "two" / "best.pt"

    first = train_thermal_mlp(
        model_config, trainer_config, train, validation, first_path
    )
    second = train_thermal_mlp(
        model_config, trainer_config, train, validation, second_path
    )

    assert isinstance(first, TrainingResult)
    assert first.device == "cpu"
    assert len(first.history) == len(second.history)
    assert all(isinstance(epoch, EpochMetrics) for epoch in first.history)
    assert first.history == second.history
    assert (
        first.best_epoch
        == min(first.history, key=lambda epoch: epoch.validation_mse).epoch
    )
    assert first.best_validation_mse == min(
        epoch.validation_mse for epoch in first.history
    )
    assert first.checkpoint_path == str(first_path)

    state_dict = torch.load(first_path, weights_only=True, map_location="cpu")
    assert isinstance(state_dict, dict)
    assert state_dict and all(
        isinstance(tensor, torch.Tensor) for tensor in state_dict.values()
    )
    restored = ThermalMLP(model_config)
    restored.load_state_dict(state_dict)
    restored.eval()
    validation_features = torch.from_numpy(validation.features.copy())
    validation_targets = torch.from_numpy(validation.targets.copy())
    with torch.inference_mode():
        restored_mse = torch.nn.functional.mse_loss(
            restored(validation_features), validation_targets
        ).item()
    assert restored_mse == pytest.approx(first.best_validation_mse, abs=1e-8)
    second_state = torch.load(second_path, weights_only=True, map_location="cpu")
    assert all(
        torch.equal(value, second_state[key]) for key, value in state_dict.items()
    )


def test_checkpoint_refuses_to_overwrite_existing_path(tmp_path: Path) -> None:
    train, validation = _inputs()
    model_config, trainer_config = _configs()
    checkpoint = tmp_path / "exists.pt"
    checkpoint.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="checkpoint"):
        train_thermal_mlp(model_config, trainer_config, train, validation, checkpoint)

    assert checkpoint.read_bytes() == b"keep me"


def test_training_rejects_empty_and_dimension_incompatible_datasets(
    tmp_path: Path,
) -> None:
    train, validation = _inputs()
    model_config, trainer_config = _configs()

    class EmptyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            raise IndexError(index)

    empty = EmptyDataset()

    with pytest.raises(ValueError, match="nonempty"):
        train_thermal_mlp(
            model_config, trainer_config, empty, validation, tmp_path / "empty.pt"
        )

    wrong_width = _dataset(np.ones((4, 4)), np.ones((4, 2)), "wrong")
    with pytest.raises(ValueError, match="input_dim"):
        train_thermal_mlp(
            model_config, trainer_config, wrong_width, validation, tmp_path / "bad.pt"
        )


def test_training_uses_early_stopping_and_best_validation_state(
    tmp_path: Path,
) -> None:
    train, validation = _inputs()
    model_config, trainer_config = _configs()
    trainer_config = TrainerConfig(
        seed=trainer_config.seed,
        batch_size=trainer_config.batch_size,
        learning_rate=trainer_config.learning_rate,
        max_epochs=trainer_config.max_epochs,
        patience=2,
        min_improvement=1.0e6,
        weight_decay=trainer_config.weight_decay,
        device_policy=trainer_config.device_policy,
    )
    result = train_thermal_mlp(
        model_config, trainer_config, train, validation, tmp_path / "stopped.pt"
    )

    assert result.stopped_early
    assert len(result.history) == 3
    assert result.best_epoch == 1


def test_explicit_cuda_policy_fails_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation = _inputs()
    model_config, trainer_config = _configs()
    trainer_config = TrainerConfig(
        seed=trainer_config.seed,
        batch_size=trainer_config.batch_size,
        learning_rate=trainer_config.learning_rate,
        max_epochs=trainer_config.max_epochs,
        patience=trainer_config.patience,
        min_improvement=trainer_config.min_improvement,
        weight_decay=trainer_config.weight_decay,
        device_policy="cuda",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        train_thermal_mlp(
            model_config, trainer_config, train, validation, tmp_path / "cuda.pt"
        )
