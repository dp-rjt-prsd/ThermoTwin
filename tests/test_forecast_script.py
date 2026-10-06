"""Subprocess coverage for reproducible held-out forecast evaluation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_forecasts.py"
COMPARISON_SCRIPT = ROOT / "scripts" / "compare_models.py"
DATASET_SCRIPT = ROOT / "scripts" / "generate_dataset.py"
HORIZONS = ("30.0", "60.0", "120.0")
ARTIFACTS = {
    "checkpoints/data_only.pt",
    "checkpoints/physics_informed.pt",
    "model_config.json",
    "normalization.json",
    "evaluated_dataset/test.csv",
    "evaluated_dataset/manifest.json",
    "evaluated_dataset/generation_config.json",
    "data_only_forecast_metrics.json",
    "physics_informed_forecast_metrics.json",
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
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=_environment(),
        check=check,
        capture_output=True,
        text=True,
    )


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _all_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


@pytest.fixture(scope="module")
def source_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    parent = tmp_path_factory.mktemp("forecast-sources")
    comparison = parent / "comparison"
    dataset = parent / "dataset"
    subprocess.run(
        [
            sys.executable,
            str(COMPARISON_SCRIPT),
            "--smoke-test",
            "--output-directory",
            str(comparison),
        ],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(DATASET_SCRIPT),
            "--trajectory-count",
            "3",
            "--seed",
            "481",
            "--output-directory",
            str(dataset),
        ],
        cwd=ROOT,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return comparison, dataset


def _assert_forecast_result(output: Path) -> None:
    assert _all_files(output) == ARTIFACTS
    for label in ("data_only", "physics_informed"):
        result = _json(output / f"{label}_forecast_metrics.json")
        assert set(result["horizons"]) == set(HORIZONS)
        for horizon in HORIZONS:
            measured = result["horizons"][horizon]
            assert measured["horizon_s"] == float(horizon)
            assert measured["step_count"] == int(float(horizon))
            assert measured["forecast_count"] > 0
            for metric in (
                "mae_kelvin",
                "rmse_kelvin",
                "maximum_absolute_error_kelvin",
            ):
                assert math.isfinite(measured["metrics"][metric])
                assert measured["metrics"][metric] >= 0.0

    data_metrics = _json(output / "data_only_forecast_metrics.json")
    physics_metrics = _json(output / "physics_informed_forecast_metrics.json")
    assert {data_metrics["horizons"][key]["forecast_count"] for key in HORIZONS} == {
        physics_metrics["horizons"][key]["forecast_count"] for key in HORIZONS
    }
    assert all(data_metrics["horizons"][key]["forecast_count"] > 0 for key in HORIZONS)


def test_normal_forecasts_copy_exact_inputs_and_record_provenance(
    tmp_path: Path, source_artifacts: tuple[Path, Path]
) -> None:
    comparison, dataset = source_artifacts
    output = tmp_path / "evaluation"
    completed = _run(
        [
            "--comparison-directory",
            str(comparison),
            "--dataset-directory",
            str(dataset),
            "--output-directory",
            str(output),
        ]
    )
    summary = json.loads(completed.stdout)
    assert summary["output_directory"] == str(output.resolve())
    assert set(summary["metrics"]) == {"data_only", "physics_informed"}
    _assert_forecast_result(output)

    for relative, source in (
        ("checkpoints/data_only.pt", comparison / "data_only.pt"),
        ("checkpoints/physics_informed.pt", comparison / "physics_informed.pt"),
        ("model_config.json", comparison / "model_config.json"),
        ("normalization.json", comparison / "normalization.json"),
        ("evaluated_dataset/test.csv", dataset / "test.csv"),
        ("evaluated_dataset/manifest.json", dataset / "manifest.json"),
        (
            "evaluated_dataset/generation_config.json",
            dataset / "generation_config.json",
        ),
    ):
        assert (output / relative).read_bytes() == source.read_bytes()

    config = _json(output / "model_config.json")
    from thermotwin.models.config import MLPConfig
    from thermotwin.models.mlp import ThermalMLP

    model_config = MLPConfig(
        **{**config, "hidden_widths": tuple(config["hidden_widths"])}
    )
    first_state = torch.load(
        output / "checkpoints/data_only.pt", weights_only=True, map_location="cpu"
    )
    second_state = torch.load(
        output / "checkpoints/physics_informed.pt",
        weights_only=True,
        map_location="cpu",
    )
    assert set(first_state) == set(second_state)
    ThermalMLP(model_config).load_state_dict(first_state, strict=True)
    ThermalMLP(model_config).load_state_dict(second_state, strict=True)

    provenance = _json(output / "provenance.json")
    assert provenance["normalization_source"] == str(
        (comparison / "normalization.json").resolve()
    )
    assert provenance["distribution_label"] == "in_distribution"
    assert provenance["smoke_test"] is False
    assert provenance["horizons_s"] == [30.0, 60.0, 120.0]
    assert provenance["time_step_s"] == 1.0
    assert provenance["forecast_counts"] == {key: 1 for key in HORIZONS}
    assert provenance["copied_artifacts"]["data_only_checkpoint"] == {
        "source": str((comparison / "data_only.pt").resolve()),
        "copy": "checkpoints/data_only.pt",
    }


def test_smoke_is_full_duration_and_deterministic(tmp_path: Path) -> None:
    first, second = tmp_path / "smoke-one", tmp_path / "smoke-two"
    first_summary = json.loads(
        _run(["--smoke-test", "--output-directory", str(first)]).stdout
    )
    second_summary = json.loads(
        _run(["--smoke-test", "--output-directory", str(second)]).stdout
    )
    _assert_forecast_result(first)
    _assert_forecast_result(second)
    for label in ("data_only", "physics_informed"):
        assert _json(first / f"{label}_forecast_metrics.json") == _json(
            second / f"{label}_forecast_metrics.json"
        )
    assert first_summary["metrics"] == second_summary["metrics"]
    first_provenance = _json(first / "provenance.json")
    assert first_provenance["smoke_test"] is True
    assert first_provenance["forecast_counts"] == {key: 1 for key in HORIZONS}
    assert (
        _json(first / "evaluated_dataset/generation_config.json")["trajectory_count"]
        == 3
    )
    assert (
        _json(first / "evaluated_dataset/generation_config.json")["sampling_config"][
            "duration_s"
        ]
        == 120.0
    )
    assert (
        len(
            _json(first / "evaluated_dataset/manifest.json")["splits"]["test"][
                "trajectory_ids"
            ]
        )
        == 1
    )


def test_smoke_without_output_uses_os_temporary_directory() -> None:
    completed = _run(["--smoke-test"])
    output = Path(json.loads(completed.stdout)["output_directory"]).resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    repository_root = ROOT.resolve()

    assert output.is_relative_to(temporary_root)
    assert not output.is_relative_to(repository_root)
    assert output.is_dir()
    assert _all_files(output) == ARTIFACTS


def test_missing_inputs_and_existing_output_return_two_without_overwrite(
    tmp_path: Path, source_artifacts: tuple[Path, Path]
) -> None:
    missing = _run(["--comparison-directory", str(tmp_path / "absent")], check=False)
    assert missing.returncode == 2
    assert missing.stderr.startswith("evaluate_forecasts: error: ")

    missing_dataset = _run(
        [
            "--comparison-directory",
            str(source_artifacts[0]),
            "--dataset-directory",
            str(tmp_path / "absent-dataset"),
        ],
        check=False,
    )
    assert missing_dataset.returncode == 2
    assert missing_dataset.stderr.startswith("evaluate_forecasts: error: ")

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    occupied = _run(["--smoke-test", "--output-directory", str(output)], check=False)
    assert occupied.returncode == 2
    assert occupied.stderr.startswith(
        "evaluate_forecasts: error: output directory already exists:"
    )
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert _all_files(output) == {"keep.txt"}
