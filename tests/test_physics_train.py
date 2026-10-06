from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from thermotwin.data.physics_dataset import PhysicsTransitionDataset
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import MLPConfig, PhysicsLossConfig, TrainerConfig
from thermotwin.models.mlp import ThermalMLP
from thermotwin.models.physics_train import (
    PhysicsEpochMetrics,
    PhysicsTrainingResult,
    train_physics_aware_mlp,
)


def _normalization() -> NormalizationStats:
    return NormalizationStats(
        feature_names=("current", "soc", "temperature"),
        target_names=("delta_cell_temperature_0_k", "delta_cell_temperature_1_k"),
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        target_mean=np.zeros(2),
        target_scale=np.ones(2),
    )


def _datasets() -> tuple[PhysicsTransitionDataset, PhysicsTransitionDataset]:
    rng = np.random.default_rng(27)
    train_x = rng.normal(size=(18, 3)).astype(np.float32)
    train_y = (0.3 * train_x[:, :2]).astype(np.float32)
    validation_x = rng.normal(size=(9, 3)).astype(np.float32)
    validation_y = (0.3 * validation_x[:, :2]).astype(np.float32)
    train_context = np.column_stack(
        (np.full(18, 2.0), np.full(18, 10.0), np.full((18, 2), 0.4))
    ).astype(np.float32)
    validation_context = np.column_stack(
        (np.full(9, 2.0), np.full(9, 10.0), np.full((9, 2), 0.4))
    ).astype(np.float32)
    return (
        PhysicsTransitionDataset(train_x, train_y, train_context, ("train",)),
        PhysicsTransitionDataset(
            validation_x, validation_y, validation_context, ("validation",)
        ),
    )


def _configs(*, min_improvement: float = 0.0) -> tuple[MLPConfig, TrainerConfig]:
    return (
        MLPConfig(input_dim=3, output_dim=2, hidden_widths=(6,)),
        TrainerConfig(
            seed=19,
            batch_size=5,
            learning_rate=0.02,
            max_epochs=15,
            patience=5,
            min_improvement=min_improvement,
            weight_decay=0.0,
            device_policy="cpu",
        ),
    )


def _run(path: Path, weight: float = 0.0) -> PhysicsTrainingResult:
    model_config, trainer_config = _configs()
    train, validation = _datasets()
    return train_physics_aware_mlp(
        model_config,
        trainer_config,
        PhysicsLossConfig(weight, residual_scale_w=2.0),
        _normalization(),
        train,
        validation,
        path,
    )


def test_training_is_deterministic_and_restores_best_weights(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "best.pt"
    second_path = tmp_path / "second" / "best.pt"
    model_config, trainer_config = _configs()
    train, validation = _datasets()
    before = tuple(
        array.copy()
        for dataset in (train, validation)
        for array in (dataset.features, dataset.targets, dataset.physics_contexts)
    )
    first = train_physics_aware_mlp(
        model_config,
        trainer_config,
        PhysicsLossConfig(0.0, 2.0),
        _normalization(),
        train,
        validation,
        first_path,
    )
    second = train_physics_aware_mlp(
        model_config,
        trainer_config,
        PhysicsLossConfig(0.0, 2.0),
        _normalization(),
        train,
        validation,
        second_path,
    )
    after = tuple(
        array
        for dataset in (train, validation)
        for array in (dataset.features, dataset.targets, dataset.physics_contexts)
    )
    assert all(
        np.array_equal(original, current)
        for original, current in zip(before, after, strict=True)
    )

    assert isinstance(first, PhysicsTrainingResult)
    assert first.history == second.history
    assert first.device == "cpu"
    assert first.checkpoint_path == str(first_path)
    assert all(isinstance(item, PhysicsEpochMetrics) for item in first.history)
    assert first.best_epoch == min(
        first.history, key=lambda item: item.validation_total_loss
    ).epoch
    assert first.best_validation_total_loss == min(
        item.validation_total_loss for item in first.history
    )
    assert all(
        np.isfinite(value)
        for item in first.history
        for value in (
            item.train_total_loss,
            item.train_data_mse,
            item.train_physics_mse,
            item.validation_total_loss,
            item.validation_data_mse,
            item.validation_physics_mse,
        )
    )
    state = torch.load(first_path, weights_only=True, map_location="cpu")
    second_state = torch.load(second_path, weights_only=True, map_location="cpu")
    assert state and all(isinstance(value, torch.Tensor) for value in state.values())
    assert all(torch.equal(value, second_state[key]) for key, value in state.items())
    model_config, _ = _configs()
    restored = ThermalMLP(model_config)
    restored.load_state_dict(state)
    _, validation = _datasets()
    restored.eval()
    with torch.inference_mode():
        prediction = restored(torch.from_numpy(validation.features.copy()))
        mse = torch.mean(
            torch.square(prediction - torch.from_numpy(validation.targets.copy()))
        )
    assert mse.item() == pytest.approx(
        first.history[first.best_epoch - 1].validation_data_mse
    )


def test_combined_loss_keeps_data_only_and_weighted_variants_on_same_path(
    tmp_path: Path,
) -> None:
    data_only = _run(tmp_path / "data.pt", weight=0.0)
    physics = _run(tmp_path / "physics.pt", weight=4.0)

    assert all(
        item.train_total_loss == pytest.approx(item.train_data_mse, abs=1e-8)
        and item.validation_total_loss
        == pytest.approx(item.validation_data_mse, abs=1e-8)
        for item in data_only.history
    )
    assert any(
        item.train_total_loss > item.train_data_mse
        and item.validation_total_loss > item.validation_data_mse
        for item in physics.history
    )


def test_early_stopping_and_existing_checkpoint_protection(tmp_path: Path) -> None:
    model_config, trainer_config = _configs(min_improvement=1.0e6)
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
    train, validation = _datasets()
    result = train_physics_aware_mlp(
        model_config,
        trainer_config,
        PhysicsLossConfig(0.0, 2.0),
        _normalization(),
        train,
        validation,
        tmp_path / "early.pt",
    )
    assert result.stopped_early
    assert len(result.history) == 3
    assert result.best_epoch == 1

    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"keep me")
    with pytest.raises(FileExistsError, match="checkpoint"):
        train_physics_aware_mlp(
            model_config,
            trainer_config,
            PhysicsLossConfig(0.0, 2.0),
            _normalization(),
            train,
            validation,
            existing,
        )
    assert existing.read_bytes() == b"keep me"


def test_bad_physics_context_shape_and_target_schema_are_rejected(
    tmp_path: Path,
) -> None:
    model_config, trainer_config = _configs()
    train, validation = _datasets()

    class WrongContextDataset(
        Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        def __len__(self) -> int:
            return len(train)

        def __getitem__(
            self, index: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            features, targets, _ = train[index]
            return features, targets, torch.ones(3, dtype=torch.float32)

    with pytest.raises(ValueError, match="physics_contexts width"):
        train_physics_aware_mlp(
            model_config,
            trainer_config,
            PhysicsLossConfig(0.0, 2.0),
            _normalization(),
            WrongContextDataset(),
            validation,
            tmp_path / "bad-context.pt",
        )

    bad_normalization = NormalizationStats(
        feature_names=("current", "soc", "temperature"),
        target_names=("wrong_0", "wrong_1"),
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        target_mean=np.zeros(2),
        target_scale=np.ones(2),
    )
    with pytest.raises(ValueError, match="target_names"):
        train_physics_aware_mlp(
            model_config,
            trainer_config,
            PhysicsLossConfig(0.0, 2.0),
            bad_normalization,
            train,
            validation,
            tmp_path / "bad-normalization.pt",
        )
