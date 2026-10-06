"""Train and evaluate the data-driven thermal surrogate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from thermotwin.data.model_dataset import load_model_datasets
from thermotwin.models.config import MLPConfig, TrainerConfig
from thermotwin.models.evaluation import evaluate_model
from thermotwin.models.mlp import ThermalMLP
from thermotwin.models.train import TrainingResult, train_thermal_mlp

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "training.yaml"
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "processed" / "dataset-42"
_ARTIFACT_SCHEMA_VERSION = "1.0"
_ARTIFACT_NAMES = (
    "best_model.pt",
    "model_config.json",
    "trainer_config.json",
    "normalization.json",
    "training_result.json",
    "test_metrics.json",
    "provenance.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            content = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid {path.name}: {error}") from error
    if type(content) is not dict:
        raise ValueError(f"{path.name} must contain a JSON object")
    return cast(dict[str, Any], content)


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def _load_training_configs(smoke_test: bool) -> tuple[MLPConfig, TrainerConfig]:
    try:
        with _CONFIG_PATH.open(encoding="utf-8") as stream:
            configuration = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not load {_CONFIG_PATH}: {error}") from error
    if type(configuration) is not dict:
        raise ValueError("configs/training.yaml must contain a YAML mapping")

    try:
        model_values = dict(configuration["model"])
        model_values["hidden_widths"] = tuple(model_values["hidden_widths"])
        training_key = "smoke_test" if smoke_test else "training"
        trainer_values = dict(configuration[training_key])
        model_config = MLPConfig(**model_values)
        trainer_config = TrainerConfig(**trainer_values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid model or trainer configuration: {error}") from error
    return model_config, trainer_config


def _generate_smoke_dataset(destination: Path) -> None:
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "generate_dataset.py"),
        "--smoke-test",
        "--output-directory",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            (error.stderr or "").strip() or "generator exited unsuccessfully"
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise RuntimeError(f"smoke dataset generation failed: {detail}") from error


def _package_versions() -> dict[str, str]:
    names = {
        "thermotwin": "thermotwin",
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "PyYAML": "PyYAML",
    }
    versions: dict[str, str] = {}
    for key, distribution in names.items():
        try:
            versions[key] = version(distribution)
        except PackageNotFoundError:
            versions[key] = "0.1.0" if key == "thermotwin" else "unknown"
    return versions


def _new_output_directory(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"output directory already exists: {path}") from error
    return path.resolve()


def _train_and_save(
    dataset_directory: Path,
    output_directory: Path,
    *,
    smoke_test: bool,
) -> dict[str, object]:
    model_config, trainer_config = _load_training_configs(smoke_test)
    manifest_path = dataset_directory / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest_cell_count = manifest.get("cell_count")
    if type(manifest_cell_count) is int:
        if model_config.input_dim != manifest_cell_count + 5:
            raise ValueError(
                "model input dimension mismatch: configured input_dim "
                f"{model_config.input_dim}, manifest cell_count + 5 is "
                f"{manifest_cell_count + 5}"
            )
        if model_config.output_dim != manifest_cell_count:
            raise ValueError(
                "model output dimension mismatch: configured output_dim "
                f"{model_config.output_dim}, manifest cell_count is "
                f"{manifest_cell_count}"
            )
    datasets = load_model_datasets(dataset_directory)
    expected_input_dim = datasets.cell_count + 5
    if model_config.input_dim != expected_input_dim:
        raise ValueError(
            "model input dimension mismatch: configured input_dim "
            f"{model_config.input_dim}, dataset cell_count + 5 is {expected_input_dim}"
        )
    if model_config.output_dim != datasets.cell_count:
        raise ValueError(
            "model output dimension mismatch: configured output_dim "
            f"{model_config.output_dim}, dataset cell_count is {datasets.cell_count}"
        )

    resolved_output = _new_output_directory(output_directory)
    checkpoint = resolved_output / "best_model.pt"
    result = train_thermal_mlp(
        model_config,
        trainer_config,
        datasets.train,
        datasets.validation,
        checkpoint,
    )

    model = ThermalMLP(model_config)
    state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("saved checkpoint must contain only a tensor state_dict")
    model.load_state_dict(state_dict)
    metrics = evaluate_model(
        model,
        datasets.test,
        datasets.normalization,
        batch_size=trainer_config.batch_size,
        device=result.device,
    )

    _json_write(resolved_output / "model_config.json", asdict(model_config))
    _json_write(resolved_output / "trainer_config.json", asdict(trainer_config))
    _json_write(
        resolved_output / "normalization.json", datasets.normalization.to_dict()
    )
    _json_write(
        resolved_output / "training_result.json", _training_result_dict(result)
    )
    _json_write(resolved_output / "test_metrics.json", metrics.to_dict())

    provenance = {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "source_dataset_directory": str(dataset_directory.resolve()),
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest": manifest,
        "seed": trainer_config.seed,
        "package_versions": _package_versions(),
        "config_source": "configs/training.yaml",
        "configuration_source_path": str(_CONFIG_PATH.resolve()),
    }
    _json_write(resolved_output / "provenance.json", provenance)

    summary: dict[str, object] = {
        "output_directory": str(resolved_output),
        "dataset_directory": str(dataset_directory.resolve()),
        "best_epoch": result.best_epoch,
        "best_validation_mse": result.best_validation_mse,
        "test_mae_kelvin": metrics.mae_kelvin,
        "test_rmse_kelvin": metrics.rmse_kelvin,
        "worst_cell_index": metrics.worst_cell_index,
        "artifacts": list(_ARTIFACT_NAMES),
    }
    return summary


def _training_result_dict(result: TrainingResult) -> dict[str, object]:
    return {
        "history": [asdict(epoch) for epoch in result.history],
        "best_epoch": result.best_epoch,
        "best_validation_mse": result.best_validation_mse,
        "stopped_early": result.stopped_early,
        "device": result.device,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--dataset-directory",
        type=Path,
        help=(
            "directory containing train/validation/test splits and manifest; "
            "default: data/processed/dataset-42"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="new directory in which to save model and evaluation artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train the configured model, evaluate test data, and print JSON summary."""
    args = _parser().parse_args(argv)
    generated_dataset: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.dataset_directory is not None:
            dataset_directory = args.dataset_directory
        elif args.smoke_test:
            generated_dataset = tempfile.TemporaryDirectory(
                prefix="thermotwin-training-dataset-"
            )
            dataset_directory = Path(generated_dataset.name) / "dataset"
            _generate_smoke_dataset(dataset_directory)
        else:
            dataset_directory = _DEFAULT_DATASET
            if not dataset_directory.is_dir():
                raise FileNotFoundError(
                    "default dataset is missing at "
                    f"{dataset_directory}; provide --dataset-directory or run "
                    "scripts/generate_dataset.py first"
                )

        if args.output_directory is not None:
            output_directory = args.output_directory
        elif args.smoke_test:
            output_directory = (
                Path(tempfile.mkdtemp(prefix="thermotwin-training-parent-"))
                / "run"
            )
        else:
            output_directory = _PROJECT_ROOT / "results" / "model-training-42"

        summary = _train_and_save(
            dataset_directory,
            output_directory,
            smoke_test=args.smoke_test,
        )
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"train_model: error: {error}", file=sys.stderr)
        return 2
    finally:
        if generated_dataset is not None:
            generated_dataset.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
