"""Integration checks for the data-driven training command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_model.py"
DATASET_SCRIPT = ROOT / "scripts" / "generate_dataset.py"
ARTIFACTS = {
    "best_model.pt",
    "model_config.json",
    "trainer_config.json",
    "normalization.json",
    "training_result.json",
    "test_metrics.json",
    "provenance.json",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    return environment


def _run(
    arguments: list[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), *arguments],
        cwd=ROOT,
        env=_environment(),
        check=check,
        capture_output=True,
        text=True,
    )


def _generate_dataset(directory: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(DATASET_SCRIPT),
            "--smoke-test",
            "--seed",
            "19",
            "--output-directory",
            str(directory),
        ],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )


def _summary(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_explicit_dataset_training_persists_evaluable_artifacts(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "dataset"
    output_directory = tmp_path / "training"
    _generate_dataset(dataset_directory)

    summary = _summary(
        _run(
            [
                "--smoke-test",
                "--dataset-directory",
                str(dataset_directory),
                "--output-directory",
                str(output_directory),
            ]
        )
    )

    assert set(path.name for path in output_directory.iterdir()) == ARTIFACTS
    assert summary["output_directory"] == str(output_directory.resolve())
    assert summary["dataset_directory"] == str(dataset_directory.resolve())
    assert set(summary["artifacts"]) == ARTIFACTS
    assert summary["test_mae_kelvin"] >= 0.0
    assert summary["test_rmse_kelvin"] >= 0.0
    assert 0 <= summary["worst_cell_index"] < 12

    provenance = json.loads(
        (output_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_dataset_directory"] == str(dataset_directory.resolve())
    assert provenance["source_manifest"]["cell_count"] == 12
    assert provenance["seed"] == 1729
    assert provenance["artifact_schema_version"] == "1.0"
    assert provenance["config_source"] == "configs/training.yaml"
    assert provenance["package_versions"]["torch"]

    metrics = json.loads(
        (output_directory / "test_metrics.json").read_text(encoding="utf-8")
    )
    assert summary["test_mae_kelvin"] == metrics["mae_kelvin"]
    assert summary["test_rmse_kelvin"] == metrics["rmse_kelvin"]
    assert summary["worst_cell_index"] == metrics["worst_cell_index"]
    weights = torch.load(
        output_directory / "best_model.pt", weights_only=True, map_location="cpu"
    )
    assert isinstance(weights, dict)
    assert weights
    assert all(isinstance(tensor, torch.Tensor) for tensor in weights.values())


def test_standalone_smoke_generates_dataset_and_uses_os_temp_output() -> None:
    completed = _run(["--smoke-test"])
    summary = _summary(completed)
    output_directory = Path(summary["output_directory"])
    assert output_directory.is_dir()
    assert output_directory.resolve().is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    )
    assert Path(summary["dataset_directory"]).resolve().is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    )
    provenance = json.loads(
        (output_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["source_manifest"]["cell_count"] == 12
    assert provenance["source_manifest"]["splits"]


def test_training_output_refuses_overwrite(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    output_directory = tmp_path / "training"
    _generate_dataset(dataset_directory)
    arguments = [
        "--smoke-test",
        "--dataset-directory",
        str(dataset_directory),
        "--output-directory",
        str(output_directory),
    ]
    _run(arguments)

    completed = _run(arguments, check=False)
    assert completed.returncode != 0
    assert "already exists" in completed.stderr.lower()
    assert set(path.name for path in output_directory.iterdir()) == ARTIFACTS


def test_normal_mode_reports_missing_default_dataset_clearly() -> None:
    completed = _run([], check=False)
    assert completed.returncode != 0
    assert "default dataset is missing" in completed.stderr.lower()
    assert "--dataset-directory" in completed.stderr


def test_dataset_model_dimension_mismatch_fails_clearly(tmp_path: Path) -> None:
    dataset_directory = tmp_path / "dataset"
    _generate_dataset(dataset_directory)
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cell_count"] = 11
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    completed = _run(
        [
            "--smoke-test",
            "--dataset-directory",
            str(dataset_directory),
            "--output-directory",
            str(tmp_path / "training"),
        ],
        check=False,
    )
    assert completed.returncode != 0
    assert "dimension" in completed.stderr.lower()


def test_repeated_smoke_training_is_semantically_deterministic(
    tmp_path: Path,
) -> None:
    dataset_directory = tmp_path / "dataset"
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    _generate_dataset(dataset_directory)
    common_arguments = [
        "--smoke-test",
        "--dataset-directory",
        str(dataset_directory),
    ]

    first_summary = _summary(
        _run([*common_arguments, "--output-directory", str(first_directory)])
    )
    second_summary = _summary(
        _run([*common_arguments, "--output-directory", str(second_directory)])
    )

    first_result = json.loads(
        (first_directory / "training_result.json").read_text(encoding="utf-8")
    )
    second_result = json.loads(
        (second_directory / "training_result.json").read_text(encoding="utf-8")
    )
    assert first_result == second_result
    assert json.loads(
        (first_directory / "test_metrics.json").read_text(encoding="utf-8")
    ) == json.loads(
        (second_directory / "test_metrics.json").read_text(encoding="utf-8")
    )
    first_weights = torch.load(
        first_directory / "best_model.pt", weights_only=True, map_location="cpu"
    )
    second_weights = torch.load(
        second_directory / "best_model.pt", weights_only=True, map_location="cpu"
    )
    assert first_weights.keys() == second_weights.keys()
    assert all(
        torch.equal(first_weights[name], second_weights[name]) for name in first_weights
    )
    assert first_summary["best_epoch"] == second_summary["best_epoch"]
