"""Subprocess coverage for the reproducible model-comparison CLI."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_models.py"
ARTIFACTS = {
    "data_only.pt",
    "physics_informed.pt",
    "model_config.json",
    "trainer_config.json",
    "data_only_physics_loss_config.json",
    "physics_informed_physics_loss_config.json",
    "normalization.json",
    "data_only_training_result.json",
    "physics_informed_training_result.json",
    "id_comparison_metrics.json",
    "ood_comparison_metrics.json",
    "provenance.json",
}


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    return environment


def _run(output: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--smoke-test",
            "--output-directory",
            str(output),
        ],
        cwd=ROOT,
        env=_environment(),
        check=check,
        capture_output=True,
        text=True,
    )


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _assert_finite_metrics(metrics: dict[str, Any]) -> None:
    assert set(metrics) == {"data_only", "physics_informed"}
    for values in metrics.values():
        regression = values["regression"]
        for name in ("mae_kelvin", "rmse_kelvin"):
            assert math.isfinite(regression[name])
            assert regression[name] >= 0.0
        residual = values["mean_absolute_physics_residual_w"]
        assert math.isfinite(residual)
        assert residual >= 0.0


def _assert_finite_summary_leaf(summary: dict[str, Any]) -> None:
    assert set(summary) == {
        "mae_kelvin",
        "rmse_kelvin",
        "mean_absolute_physics_residual_w",
    }
    for value in summary.values():
        assert math.isfinite(value)
        assert value >= 0.0


def test_smoke_comparison_persists_complete_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "run-one"
    second = tmp_path / "run-two"
    first_summary = json.loads(_run(first).stdout)
    second_summary = json.loads(_run(second).stdout)

    assert set(path.name for path in first.iterdir()) == ARTIFACTS
    assert set(path.name for path in second.iterdir()) == ARTIFACTS
    assert first_summary["output_directory"] == str(first.resolve())
    assert second_summary["output_directory"] == str(second.resolve())
    assert first_summary["metrics"] == second_summary["metrics"]

    data_config = _json(first / "data_only_physics_loss_config.json")
    informed_config = _json(first / "physics_informed_physics_loss_config.json")
    assert data_config["lambda_physics"] == 0.0
    assert informed_config["lambda_physics"] > 0.0
    assert {
        key: value for key, value in data_config.items() if key != "lambda_physics"
    } == {
        key: value for key, value in informed_config.items() if key != "lambda_physics"
    }

    model_config = _json(first / "model_config.json")
    data_state = torch.load(
        first / "data_only.pt", weights_only=True, map_location="cpu"
    )
    informed_state = torch.load(
        first / "physics_informed.pt", weights_only=True, map_location="cpu"
    )
    assert data_state and set(data_state) == set(informed_state)
    assert all(isinstance(tensor, torch.Tensor) for tensor in data_state.values())
    assert all(isinstance(tensor, torch.Tensor) for tensor in informed_state.values())
    # Strict load into two fresh, identically configured model instances.
    from thermotwin.models.config import MLPConfig
    from thermotwin.models.mlp import ThermalMLP

    architecture = MLPConfig(
        **{**model_config, "hidden_widths": tuple(model_config["hidden_widths"])}
    )
    ThermalMLP(architecture).load_state_dict(data_state, strict=True)
    ThermalMLP(architecture).load_state_dict(informed_state, strict=True)

    normalization = _json(first / "normalization.json")
    provenance = _json(first / "provenance.json")
    assert provenance["normalization"] == {
        "artifact": "normalization.json",
        "fit_dataset": "in_distribution",
        "reused_for": ["in_distribution", "out_of_distribution"],
    }
    assert (
        provenance["datasets"]["out_of_distribution"]["normalization_source"]
        == "in_distribution"
    )
    assert normalization["feature_names"]
    assert provenance["smoke_test"] is True
    assert provenance["datasets"]["in_distribution"]["manifest"]["cell_count"] == 12
    assert provenance["datasets"]["out_of_distribution"]["manifest"]["cell_count"] == 12
    assert provenance["package_versions"]["torch"]

    for filename in ("id_comparison_metrics.json", "ood_comparison_metrics.json"):
        _assert_finite_metrics(_json(first / filename))
    for distribution in ("in_distribution", "out_of_distribution"):
        _assert_finite_summary_leaf(
            first_summary["metrics"]["data_only"][distribution]
        )
        _assert_finite_summary_leaf(
            first_summary["metrics"]["physics_informed"][distribution]
        )

    for label in ("data_only", "physics_informed"):
        result = _json(first / f"{label}_training_result.json")
        assert result["history"]
        assert all(
            math.isfinite(value)
            for epoch in result["history"]
            for value in epoch.values()
            if isinstance(value, (int, float))
        )


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    completed = _run(output, check=False)

    assert completed.returncode == 2
    assert completed.stderr.startswith(
        "compare_models: error: output directory already exists:"
    )
    assert marker.read_text(encoding="utf-8") == "keep"
    assert set(path.name for path in output.iterdir()) == {"keep.txt"}


def test_normal_mode_reports_missing_persistent_datasets(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--id-dataset-directory",
            str(tmp_path / "missing-id"),
            "--ood-dataset-directory",
            str(tmp_path / "missing-ood"),
        ],
        cwd=ROOT,
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stderr.startswith(
        "compare_models: error: in-distribution dataset is missing at"
    )
