"""Evaluate both Phase 7 thermal twins on held-out exact-horizon forecasts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import torch

from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.data.transition_artifacts import load_transition_artifacts
from thermotwin.models.config import MLPConfig
from thermotwin.models.forecast_evaluation import (
    ForecastEvaluation,
    evaluate_forecast_horizons,
)
from thermotwin.models.forecaster import MLPThermalForecaster
from thermotwin.models.mlp import ThermalMLP

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_COMPARISON = _PROJECT_ROOT / "results" / "model-comparison-1729"
_DEFAULT_DATASET = _PROJECT_ROOT / "data" / "processed" / "dataset-42"
_HORIZONS_S = (30.0, 60.0, 120.0)
_ARTIFACT_SCHEMA_VERSION = "1.0"


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON artifact contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON artifact contains nonfinite constant {value}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not read valid {label} at {path}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _load_model_config(path: Path) -> MLPConfig:
    raw = _read_json(path, "model_config.json")
    expected = {field.name for field in fields(MLPConfig)}
    if set(raw) != expected:
        raise ValueError("model_config.json fields do not match the MLPConfig schema")
    if type(raw["hidden_widths"]) is not list:
        raise ValueError("model_config.json hidden_widths must be an array")
    try:
        return MLPConfig(**{**raw, "hidden_widths": tuple(raw["hidden_widths"])})
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"model_config.json is invalid: {error}") from error


def _load_normalization(path: Path) -> NormalizationStats:
    try:
        return NormalizationStats.from_dict(_read_json(path, "normalization.json"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"normalization.json is invalid: {error}") from error


def _checkpoint(path: Path, config: MLPConfig) -> ThermalMLP:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            f"could not load tensor-only checkpoint {path}: {error}"
        ) from error
    if (
        not isinstance(value, dict)
        or not value
        or not all(
            type(key) is str and isinstance(tensor, torch.Tensor)
            for key, tensor in value.items()
        )
    ):
        raise ValueError(f"checkpoint must contain only a tensor state_dict: {path}")
    model = ThermalMLP(config)
    try:
        model.load_state_dict(value, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"checkpoint state_dict does not match model schema {path}: {error}"
        ) from error
    return model


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("thermotwin", "numpy", "pandas", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unknown"
    return result


def _run_command(command: list[str], label: str) -> None:
    try:
        completed = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            (error.stderr or "").strip()
            or f"command exited with status {error.returncode}"
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise RuntimeError(f"{label} failed: {detail}") from error
    if not completed.stdout.strip():
        raise RuntimeError(f"{label} completed without a summary")


def _smoke_sources(root: Path) -> tuple[Path, Path]:
    comparison = root / "comparison"
    dataset = root / "dataset"
    _run_command(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "compare_models.py"),
            "--smoke-test",
            "--output-directory",
            str(comparison),
        ],
        "Phase 7 smoke comparison",
    )
    _run_command(
        [
            sys.executable,
            str(_PROJECT_ROOT / "scripts" / "generate_dataset.py"),
            "--trajectory-count",
            "3",
            "--seed",
            "1729",
            "--output-directory",
            str(dataset),
        ],
        "full-duration smoke evaluation dataset generation",
    )
    return comparison, dataset


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


def _copy_artifact(source: Path, output: Path, relative: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"required source artifact is missing: {source}")
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--comparison-directory", type=Path)
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        if args.output_directory is not None:
            output_argument = args.output_directory
        elif args.smoke_test:
            temporary_output_parent = Path(
                tempfile.mkdtemp(prefix="thermotwin-forecast-output-")
            )
            output_argument = temporary_output_parent / "evaluation"
        else:
            output_argument = _PROJECT_ROOT / "results" / "forecast-evaluation"
        if output_argument.exists():
            raise FileExistsError(f"output directory already exists: {output_argument}")

        if args.smoke_test:
            temporary = tempfile.TemporaryDirectory(prefix="thermotwin-forecast-smoke-")
            comparison, dataset = _smoke_sources(Path(temporary.name))
        else:
            comparison = args.comparison_directory or _DEFAULT_COMPARISON
            dataset = args.dataset_directory or _DEFAULT_DATASET
            if not comparison.is_dir():
                raise FileNotFoundError(
                    f"Phase 7 comparison directory is missing at {comparison}; "
                    "provide --comparison-directory or run "
                    "scripts/compare_models.py first"
                )
            if not dataset.is_dir():
                raise FileNotFoundError(
                    f"evaluation dataset directory is missing at {dataset}; "
                    "provide --dataset-directory or generate a dataset first"
                )

        comparison = comparison.resolve()
        dataset = dataset.resolve()
        model_config_path = comparison / "model_config.json"
        normalization_path = comparison / "normalization.json"
        checkpoint_paths = {
            "data_only": comparison / "data_only.pt",
            "physics_informed": comparison / "physics_informed.pt",
        }
        model_config = _load_model_config(model_config_path)
        normalization = _load_normalization(normalization_path)
        models = {
            label: _checkpoint(path, model_config)
            for label, path in checkpoint_paths.items()
        }
        artifacts = load_transition_artifacts(dataset, normalization)
        test_records = artifacts.test
        cooling = artifacts.base_environment_config.cooling
        timestep = float(test_records[0].time_step_s)
        if not test_records:
            raise ValueError("evaluation dataset test split contains no transitions")
        if model_config.output_dim != artifacts.model_datasets.cell_count:
            raise ValueError(
                "checkpoint model cell count does not match evaluation dataset"
            )
        records_by_trajectory: dict[str, int] = {}
        for record in test_records:
            records_by_trajectory[record.trajectory_id] = (
                records_by_trajectory.get(record.trajectory_id, 0) + 1
            )
        required_steps = int(_HORIZONS_S[-1] / timestep)
        if any(count < required_steps for count in records_by_trajectory.values()):
            short = sorted(
                (trajectory, count)
                for trajectory, count in records_by_trajectory.items()
                if count < required_steps
            )
            raise ValueError(
                "every test trajectory must contain at least 120 s "
                f"({required_steps} records at {timestep:g} s timestep); "
                f"short trajectories: {short}"
            )

        evaluations: dict[str, ForecastEvaluation] = {}
        for label, model in models.items():
            forecaster = MLPThermalForecaster(
                model,
                normalization,
                time_step_s=timestep,
                minimum_coolant_flow_kg_s=cooling.minimum_coolant_mass_flow_kg_s,
                maximum_coolant_flow_kg_s=cooling.maximum_coolant_mass_flow_kg_s,
            )
            evaluations[label] = evaluate_forecast_horizons(
                forecaster, test_records, _HORIZONS_S
            )

        output = _new_output_directory(output_argument)
        copied_sources = {
            "data_only_checkpoint": (
                checkpoint_paths["data_only"],
                "checkpoints/data_only.pt",
            ),
            "physics_informed_checkpoint": (
                checkpoint_paths["physics_informed"],
                "checkpoints/physics_informed.pt",
            ),
            "model_config": (model_config_path, "model_config.json"),
            "normalization": (normalization_path, "normalization.json"),
            "evaluation_test_csv": (dataset / "test.csv", "evaluated_dataset/test.csv"),
            "evaluation_manifest": (
                dataset / "manifest.json",
                "evaluated_dataset/manifest.json",
            ),
            "evaluation_generation_config": (
                dataset / "generation_config.json",
                "evaluated_dataset/generation_config.json",
            ),
        }
        copied_artifacts: dict[str, dict[str, str]] = {}
        for key, (source, relative) in copied_sources.items():
            _copy_artifact(source, output, relative)
            copied_artifacts[key] = {
                "source": str(source.resolve()),
                "copy": relative,
            }

        forecast_counts: dict[str, int] = {}
        summary_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
        for label, evaluation in evaluations.items():
            result = evaluation.to_dict()
            _write_json(output / f"{label}_forecast_metrics.json", result)
            model_summary: dict[str, dict[str, float | int]] = {}
            for horizon in evaluation.horizons:
                key = str(horizon.horizon_s)
                forecast_counts.setdefault(key, horizon.forecast_count)
                model_summary[key] = {
                    "mae_kelvin": horizon.metrics.mae_kelvin,
                    "rmse_kelvin": horizon.metrics.rmse_kelvin,
                    "max_absolute_cell_error_kelvin": (
                        horizon.metrics.maximum_absolute_error_kelvin
                    ),
                    "forecast_count": horizon.forecast_count,
                }
            summary_metrics[label] = model_summary

        generation_config_path = dataset / "generation_config.json"
        generation_config = _read_json(generation_config_path, "generation_config.json")
        source_config_files = generation_config.get("source_config_files")
        if type(source_config_files) is not list or any(
            type(item) is not str for item in source_config_files
        ):
            raise ValueError(
                "generation_config.json source_config_files must be a string array"
            )
        distribution_label = (
            "out_of_distribution"
            if "configs/dataset_ood.yaml" in source_config_files
            else "in_distribution"
        )
        comparison_provenance_path = comparison / "provenance.json"
        comparison_provenance = (
            _read_json(comparison_provenance_path, "comparison provenance")
            if comparison_provenance_path.is_file()
            else {}
        )
        provenance = {
            "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
            "seeds": {
                "training": comparison_provenance.get("seed"),
                "evaluation_dataset": generation_config.get("seed"),
            },
            "package_versions": _package_versions(),
            "source_paths": {
                "comparison_directory": str(comparison),
                "dataset_directory": str(dataset),
                "model_config": str(model_config_path),
                "normalization": str(normalization_path),
                "checkpoints": {
                    label: str(path) for label, path in checkpoint_paths.items()
                },
            },
            "copied_artifacts": copied_artifacts,
            "horizons_s": list(_HORIZONS_S),
            "time_step_s": timestep,
            "forecast_counts": forecast_counts,
            "normalization_source": str(normalization_path),
            "distribution_label": distribution_label,
            "split_label": "test",
            "smoke_test": args.smoke_test,
        }
        _write_json(output / "provenance.json", provenance)
        print(
            json.dumps(
                {"output_directory": str(output), "metrics": summary_metrics},
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        print(f"evaluate_forecasts: error: {error}", file=sys.stderr)
        return 2
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
