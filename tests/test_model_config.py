from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from thermotwin.models import MLPConfig, TrainerConfig


def test_mlp_config_is_immutable_and_accepts_baseline_dimensions() -> None:
    config = MLPConfig(17, 12, (64, 32))

    assert (config.input_dim, config.output_dim) == (17, 12)
    assert config.hidden_widths == (64, 32)
    assert config.activation == "gelu"
    with pytest.raises(FrozenInstanceError):
        config.input_dim = 18  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "12"])
@pytest.mark.parametrize("field", ["input_dim", "output_dim"])
def test_mlp_config_rejects_invalid_dimensions(field: str, value: Any) -> None:
    values: dict[str, Any] = {
        "input_dim": 17,
        "output_dim": 12,
        "hidden_widths": (64, 32),
    }
    values[field] = value
    with pytest.raises(ValueError):
        MLPConfig(**values)


@pytest.mark.parametrize("widths", [(), [], (0, 32), (64, True), (64, 2.5), (-1,)])
def test_mlp_config_rejects_invalid_hidden_widths(widths: Any) -> None:
    with pytest.raises(ValueError):
        MLPConfig(17, 12, widths)


def test_mlp_config_rejects_unsupported_activation() -> None:
    with pytest.raises(ValueError, match="activation"):
        MLPConfig(17, 12, (64, 32), activation="relu")  # type: ignore[arg-type]


def test_trainer_config_is_immutable_and_has_explicit_defaults() -> None:
    config = TrainerConfig(
        seed=7,
        batch_size=64,
        learning_rate=0.001,
        max_epochs=100,
        patience=10,
        min_improvement=0.0001,
        weight_decay=0.0,
        device_policy="cpu",
    )

    assert config.seed == 7
    assert config.device_policy == "cpu"
    with pytest.raises(FrozenInstanceError):
        config.seed = 8  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_trainer_config_rejects_invalid_seed(value: Any) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(value, 64, 0.001, 100, 10, 0.0001, 0.0, "cpu")


@pytest.mark.parametrize("field", ["batch_size", "max_epochs", "patience"])
@pytest.mark.parametrize("value", [True, -1, 0, 1.5, "1"])
def test_trainer_config_rejects_invalid_integer_fields(field: str, value: Any) -> None:
    values: dict[str, Any] = {
        "seed": 7, "batch_size": 64, "learning_rate": 0.001,
        "max_epochs": 100, "patience": 10, "min_improvement": 0.0001,
        "weight_decay": 0.0, "device_policy": "cpu",
    }
    values[field] = value
    with pytest.raises(ValueError):
        TrainerConfig(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [("learning_rate", 0), ("learning_rate", float("nan")),
     ("learning_rate", float("inf")), ("min_improvement", -1),
     ("min_improvement", float("nan")), ("weight_decay", -1),
     ("weight_decay", float("inf")), ("min_improvement", True)],
)
def test_trainer_config_rejects_invalid_numeric_hyperparameters(
    field: str, value: Any
) -> None:
    values: dict[str, Any] = {
        "seed": 7, "batch_size": 64, "learning_rate": 0.001,
        "max_epochs": 100, "patience": 10, "min_improvement": 0.0001,
        "weight_decay": 0.0, "device_policy": "cpu",
    }
    values[field] = value
    with pytest.raises(ValueError):
        TrainerConfig(**values)


@pytest.mark.parametrize("device", ["", "gpu", "cuda:-1", "CPU", 1, None])
def test_trainer_config_rejects_invalid_device_policy(device: Any) -> None:
    with pytest.raises(ValueError, match="device_policy"):
        TrainerConfig(7, 64, 0.001, 100, 10, 0.0001, 0.0, device)


def test_training_yaml_defines_baseline_and_smoke_settings() -> None:
    raw = yaml.safe_load(Path("configs/training.yaml").read_text(encoding="utf-8"))
    model = MLPConfig(**raw["model"])
    normal = TrainerConfig(**raw["training"])
    smoke = TrainerConfig(**raw["smoke_test"])

    assert (model.input_dim, model.output_dim, model.hidden_widths) == (
        17,
        12,
        (64, 32),
    )
    assert model.activation == "gelu"
    assert normal.seed == smoke.seed
    assert normal.device_policy == "auto"
    assert smoke.device_policy == "cpu"
    assert smoke.max_epochs < normal.max_epochs
    assert smoke.batch_size < normal.batch_size


def test_model_config_contracts_do_not_import_torch() -> None:
    source = Path("src/thermotwin/models/config.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules.update(
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert not any(
        module == "torch" or module.startswith("torch.") for module in modules
    )
