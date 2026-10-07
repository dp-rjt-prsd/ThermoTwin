"""Train and compare data-only and physics-informed thermal MLPs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import torch
import yaml

from thermotwin.data.physics_dataset import (
    build_collocation_dataset,
    build_physics_datasets,
)
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.data.transition_artifacts import load_transition_artifacts
from thermotwin.models.comparison import evaluate_physics_model
from thermotwin.models.config import MLPConfig, PhysicsLossConfig, TrainerConfig
from thermotwin.models.mlp import ThermalMLP
from thermotwin.models.physics_train import (
    PhysicsTrainingResult,
    train_physics_aware_mlp,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "training.yaml"
_DEFAULT_ID_DATASET = _PROJECT_ROOT / "data" / "processed" / "dataset-42"
_DEFAULT_OOD_DATASET = _PROJECT_ROOT / "data" / "processed" / "dataset-43"
_ARTIFACT_SCHEMA_VERSION = "1.0"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not load {path}: {error}") from error
    if type(raw) is not dict:
        raise ValueError(
            f"{path.relative_to(_PROJECT_ROOT)} must contain a YAML mapping"
        )
    return cast(dict[str, Any], raw)


def _mapping(raw: object, name: str, expected: set[str]) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != expected:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")
    return cast(dict[str, Any], raw)


def _load_training_configs(
    smoke_test: bool,
) -> tuple[MLPConfig, TrainerConfig, PhysicsLossConfig, PhysicsLossConfig]:
    configuration = _read_yaml(_CONFIG_PATH)
    top = _mapping(
        configuration,
        "configs/training.yaml",
        {"model", "training", "smoke_test", "physics_loss"},
    )
    model_values = _mapping(
        top["model"],
        "model",
        {"input_dim", "output_dim", "hidden_widths", "activation"},
    )
    model_values["hidden_widths"] = tuple(model_values["hidden_widths"])
    model_config = MLPConfig(**model_values)
    trainer_key = "smoke_test" if smoke_test else "training"
    trainer_values = _mapping(
        top[trainer_key],
        trainer_key,
        {
            "seed",
            "batch_size",
            "learning_rate",
            "max_epochs",
            "patience",
            "min_improvement",
            "weight_decay",
            "device_policy",
        },
    )
    trainer_config = TrainerConfig(**trainer_values)
    physics_root = _mapping(
        top["physics_loss"], "physics_loss", {"standard", "smoke_test"}
    )
    physics_key = "smoke_test" if smoke_test else "standard"
    physics_values = physics_root[physics_key]
    required = {"lambda_physics", "residual_scale_w"}
    optional = {"collocation_samples_per_row", "collocation_max_current_a"}
    if type(physics_values) is not dict or not (
        required <= set(physics_values) <= required | optional
    ):
        raise ValueError(
            f"physics_loss.{physics_key} must contain "
            f"{', '.join(sorted(required))} and optionally "
            f"{', '.join(sorted(optional))}"
        )
    configured = PhysicsLossConfig(**physics_values)
    if configured.lambda_physics <= 0.0:
        raise ValueError("configured physics-informed lambda_physics must be positive")
    # The data-only baseline uses no physics term and no collocation rows.
    data_only = replace(configured, lambda_physics=0.0, collocation_samples_per_row=0)
    return model_config, trainer_config, data_only, configured


def _generate_dataset(destination: Path, *, ood: bool, seed: int) -> None:
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "generate_dataset.py"),
        "--smoke-test",
        "--seed",
        str(seed),
        "--output-directory",
        str(destination),
    ]
    if ood:
        command.append("--ood")
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


def _new_output_directory(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"output directory already exists: {path}") from error
    return path.resolve()


def _write_json(path: Path, value: object) -> None:
    try:
        rendered = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"could not serialize {path.name}: {error}") from error
    path.write_text(rendered, encoding="utf-8", newline="")


def _training_result_dict(result: PhysicsTrainingResult) -> dict[str, object]:
    return {
        "history": [asdict(epoch) for epoch in result.history],
        "best_epoch": result.best_epoch,
        "best_validation_total_loss": result.best_validation_total_loss,
        "stopped_early": result.stopped_early,
        "device": result.device,
    }


def _package_versions() -> dict[str, str]:
    distributions = {
        "thermotwin": "thermotwin",
        "numpy": "numpy",
        "pandas": "pandas",
        "torch": "torch",
        "PyYAML": "PyYAML",
    }
    results: dict[str, str] = {}
    for name, distribution in distributions.items():
        try:
            results[name] = version(distribution)
        except PackageNotFoundError:
            results[name] = "0.1.0" if name == "thermotwin" else "unknown"
    return results


def _manifest(directory: Path) -> tuple[dict[str, Any], Path]:
    path = directory / "manifest.json"
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read dataset manifest {path}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"dataset manifest must be a JSON object: {path}")
    return cast(dict[str, Any], value), path


def _dataset_seed(directory: Path) -> int:
    path = directory / "generation_config.json"
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read dataset generation config {path}: {error}"
        ) from error
    if type(value) is not dict or type(value.get("seed")) is not int:
        raise ValueError(
            f"dataset generation config must contain an integer seed: {path}"
        )
    return cast(int, value["seed"])


def _validate_dimensions(model: MLPConfig, datasets: Any, label: str) -> None:
    for split_name in ("train", "validation", "test"):
        dataset = getattr(datasets, split_name)
        if model.input_dim != dataset.feature_count:
            raise ValueError(
                f"model input dimension mismatch for {label} {split_name}: "
                f"configured {model.input_dim}, dataset has {dataset.feature_count}"
            )
        if model.output_dim != dataset.target_count:
            raise ValueError(
                f"model output dimension mismatch for {label} {split_name}: "
                f"configured {model.output_dim}, dataset has {dataset.target_count}"
            )
        if model.output_dim != dataset.cell_count:
            raise ValueError(
                f"model output dimension mismatch for {label} {split_name} "
                "physics cells"
            )


def _evaluate(
    checkpoint: Path,
    model_config: MLPConfig,
    dataset: Any,
    normalization: NormalizationStats,
    batch_size: int,
) -> dict[str, object]:
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError(
            f"checkpoint must contain only a tensor state_dict: {checkpoint}"
        )
    model = ThermalMLP(model_config)
    model.load_state_dict(state, strict=True)
    return evaluate_physics_model(model, dataset, normalization, batch_size).to_dict()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="generate deterministic temporary ID and OOD smoke datasets",
    )
    parser.add_argument("--id-dataset-directory", type=Path)
    parser.add_argument("--ood-dataset-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generated: tempfile.TemporaryDirectory[str] | None = None
    try:
        model_config, trainer_config, data_loss_config, physics_loss_config = (
            _load_training_configs(args.smoke_test)
        )
        if args.smoke_test:
            generated = tempfile.TemporaryDirectory(
                prefix="thermotwin-comparison-data-"
            )
            parent = Path(generated.name)
            id_directory = parent / "id"
            ood_directory = parent / "ood"
            _generate_dataset(id_directory, ood=False, seed=1729)
            _generate_dataset(ood_directory, ood=True, seed=1730)
        else:
            id_directory = args.id_dataset_directory or _DEFAULT_ID_DATASET
            ood_directory = args.ood_dataset_directory or _DEFAULT_OOD_DATASET
            for label, directory in (
                ("in-distribution", id_directory),
                ("out-of-distribution", ood_directory),
            ):
                if not directory.is_dir():
                    raise FileNotFoundError(
                        f"{label} dataset is missing at {directory}; provide "
                        f"--{'id' if label == 'in-distribution' else 'ood'}-"
                        "dataset-directory "
                        "or generate the dataset first"
                    )

        # Fit normalization from ID, then reuse the exact object for OOD.
        id_artifacts = load_transition_artifacts(id_directory)
        normalization = id_artifacts.model_datasets.normalization
        ood_artifacts = load_transition_artifacts(ood_directory, normalization)
        id_datasets = build_physics_datasets(id_artifacts)
        ood_datasets = build_physics_datasets(ood_artifacts)
        _validate_dimensions(model_config, id_datasets, "in-distribution")
        _validate_dimensions(model_config, ood_datasets, "out-of-distribution")
        if ood_artifacts.model_datasets.normalization is not normalization:
            raise ValueError(
                "OOD data did not retain the exact ID normalization object"
            )

        if args.output_directory is not None:
            output_path = args.output_directory
        elif args.smoke_test:
            output_path = (
                Path(tempfile.mkdtemp(prefix="thermotwin-comparison-output-")) / "run"
            )
        else:
            output_path = _PROJECT_ROOT / "results" / "model-comparison-1729"
        output = _new_output_directory(output_path)
        runs: dict[str, PhysicsTrainingResult] = {}
        for label, loss_config in (
            ("data_only", data_loss_config),
            ("physics_informed", physics_loss_config),
        ):
            collocation = (
                build_collocation_dataset(
                    id_artifacts,
                    loss_config.collocation_samples_per_row,
                    seed=trainer_config.seed,
                    max_current_a=loss_config.collocation_max_current_a,
                )
                if loss_config.collocation_samples_per_row > 0
                else None
            )
            runs[label] = train_physics_aware_mlp(
                model_config,
                trainer_config,
                loss_config,
                normalization,
                id_datasets.train,
                id_datasets.validation,
                output / f"{label}.pt",
                collocation,
            )

        metrics = {
            label: {
                "in_distribution": _evaluate(
                    output / f"{label}.pt",
                    model_config,
                    id_datasets.test,
                    normalization,
                    trainer_config.batch_size,
                ),
                "out_of_distribution": _evaluate(
                    output / f"{label}.pt",
                    model_config,
                    ood_datasets.test,
                    normalization,
                    trainer_config.batch_size,
                ),
            }
            for label in runs
        }
        id_manifest, id_manifest_path = _manifest(id_directory)
        ood_manifest, ood_manifest_path = _manifest(ood_directory)
        id_seed = _dataset_seed(id_directory)
        ood_seed = _dataset_seed(ood_directory)
        _write_json(output / "model_config.json", asdict(model_config))
        _write_json(output / "trainer_config.json", asdict(trainer_config))
        _write_json(
            output / "data_only_physics_loss_config.json", asdict(data_loss_config)
        )
        _write_json(
            output / "physics_informed_physics_loss_config.json",
            asdict(physics_loss_config),
        )
        _write_json(output / "normalization.json", normalization.to_dict())
        for label, result in runs.items():
            _write_json(
                output / f"{label}_training_result.json", _training_result_dict(result)
            )
        _write_json(
            output / "id_comparison_metrics.json",
            {label: values["in_distribution"] for label, values in metrics.items()},
        )
        _write_json(
            output / "ood_comparison_metrics.json",
            {label: values["out_of_distribution"] for label, values in metrics.items()},
        )
        provenance = {
            "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
            "datasets": {
                "in_distribution": {
                    "directory": str(id_directory.resolve()),
                    "manifest_path": str(id_manifest_path.resolve()),
                    "manifest": id_manifest,
                    "distribution": "in_distribution",
                },
                "out_of_distribution": {
                    "directory": str(ood_directory.resolve()),
                    "manifest_path": str(ood_manifest_path.resolve()),
                    "manifest": ood_manifest,
                    "distribution": "out_of_distribution",
                    "normalization_source": "in_distribution",
                },
            },
            "seed": trainer_config.seed,
            "dataset_seeds": {
                "in_distribution": id_seed,
                "out_of_distribution": ood_seed,
            },
            "package_versions": _package_versions(),
            "config_source": "configs/training.yaml",
            "configuration_source_path": str(_CONFIG_PATH.resolve()),
            "normalization": {
                "artifact": "normalization.json",
                "fit_dataset": "in_distribution",
                "reused_for": ["in_distribution", "out_of_distribution"],
            },
            "distribution_labels": ["in_distribution", "out_of_distribution"],
            "smoke_test": args.smoke_test,
        }
        _write_json(output / "provenance.json", provenance)
        summary = {
            "output_directory": str(output),
            "metrics": {
                label: {
                    distribution: {
                        "mae_kelvin": cast(dict[str, Any], metric)["regression"][
                            "mae_kelvin"
                        ],
                        "rmse_kelvin": cast(dict[str, Any], metric)["regression"][
                            "rmse_kelvin"
                        ],
                        "mean_absolute_physics_residual_w": metric[
                            "mean_absolute_physics_residual_w"
                        ],
                    }
                    for distribution, metric in values.items()
                }
                for label, values in metrics.items()
            },
        }
        print(json.dumps(summary, allow_nan=False, sort_keys=True))
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        print(f"compare_models: error: {error}", file=sys.stderr)
        return 2
    finally:
        if generated is not None:
            generated.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
