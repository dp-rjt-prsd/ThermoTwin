"""Integration checks for the reproducible baseline comparison command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from thermotwin.diagnostics import ExperimentMetrics, calculate_experiment_metrics
from thermotwin.experiments import (
    ExperimentRecord,
    ExperimentResult,
    ExperimentScenario,
    MetadataValue,
    run_experiment,
    save_experiment_result,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_baseline.py"


def _run_baseline(output_directory: Path, seed: int = 42) -> dict[str, Any]:
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_path, environment.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--smoke-test",
            "--output-directory",
            str(output_directory),
            "--seed",
            str(seed),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_public_package_exports_are_explicit() -> None:
    import thermotwin.diagnostics as diagnostics
    import thermotwin.experiments as experiments

    assert diagnostics.__all__ == [
        "ExperimentMetrics",
        "calculate_experiment_metrics",
    ]
    assert experiments.__all__ == [
        "ExperimentRecord",
        "ExperimentResult",
        "ExperimentScenario",
        "MetadataValue",
        "run_experiment",
        "save_experiment_result",
    ]
    assert diagnostics.ExperimentMetrics is ExperimentMetrics
    assert diagnostics.calculate_experiment_metrics is calculate_experiment_metrics
    assert experiments.ExperimentRecord is ExperimentRecord
    assert experiments.ExperimentResult is ExperimentResult
    assert experiments.ExperimentScenario is ExperimentScenario
    assert experiments.MetadataValue is MetadataValue
    assert experiments.run_experiment is run_experiment
    assert experiments.save_experiment_result is save_experiment_result


def test_smoke_comparison_saves_fair_provenance_and_measured_metrics(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "baseline"
    summary = _run_baseline(output_directory, seed=13)

    assert set(summary) == {"runs"}
    assert set(summary["runs"]) == {"PIDController", "ThresholdController"}
    run_directories = {
        name: Path(run["artifact_directory"])
        for name, run in summary["runs"].items()
    }
    assert run_directories["PIDController"] != run_directories["ThresholdController"]

    provenances = {
        name: json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        for name, directory in run_directories.items()
    }
    for directory in run_directories.values():
        assert {path.name for path in directory.iterdir()} == {
            "metrics.json",
            "provenance.json",
            "timeseries.csv",
        }
    threshold_provenance = dict(provenances["ThresholdController"])
    pid_provenance = dict(provenances["PIDController"])
    threshold_controller = threshold_provenance.pop("controller_class_name")
    pid_controller = pid_provenance.pop("controller_class_name")
    assert threshold_controller == "ThresholdController"
    assert pid_controller == "PIDController"
    assert threshold_provenance == pid_provenance

    provenance = provenances["ThresholdController"]
    assert provenance["scenario"] == {
        "duration_s": 20.0,
        "name": "baseline-stop-and-go",
        "pump_power_coefficient_w_s3_kg3": 100000.0,
        "thermal_limit_k": 313.15,
        "time_step_s": 1.0,
    }
    assert provenance["seed"] == 13
    assert provenance["record_count"] == 20
    assert provenance["scenario_metadata"] == {
        "current_profile": "stop_and_go",
        "cycle_duration_s": 20.0,
        "peak_discharge_current_a": 120.0,
        "total_duration_s": 20.0,
    }
    assert provenance["environment_config"]["pack"] == {
        "rows": 3,
        "columns": 4,
        "cell_mass_kg": 0.8,
        "cell_specific_heat_capacity_j_kg_k": 900.0,
        "nominal_cell_resistance_ohm": 0.015,
        "cell_capacity_ah": 50.0,
        "neighbor_thermal_resistance_k_w": 1.5,
        "ambient_thermal_resistance_k_w": 8.0,
        "cell_cooling_surface_area_m2": 0.02,
    }
    assert provenance["environment_config"]["simulation"] == {
        "default_time_step_s": 1.0,
        "initial_soc_fraction": 0.8,
        "initial_cell_temperature_k": 298.15,
        "ambient_temperature_k": 298.15,
        "coolant_inlet_temperature_k": 293.15,
        "initial_current_a": 0.0,
        "initial_cell_temperatures_k": None,
    }

    for name, directory in run_directories.items():
        stored_metrics = json.loads(
            (directory / "metrics.json").read_text(encoding="utf-8")
        )
        assert summary["runs"][name]["metrics"] == stored_metrics

    repeated_directory = tmp_path / "baseline-repeat"
    repeated = _run_baseline(repeated_directory, seed=13)
    for name, directory in run_directories.items():
        repeated_artifacts = Path(repeated["runs"][name]["artifact_directory"])
        assert (directory / "timeseries.csv").read_bytes() == (
            repeated_artifacts / "timeseries.csv"
        ).read_bytes()
        assert (directory / "metrics.json").read_bytes() == (
            repeated_artifacts / "metrics.json"
        ).read_bytes()


def test_smoke_mode_without_output_uses_a_temporary_location() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(ROOT / "src"), environment.get("PYTHONPATH", ""))
        if part
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--smoke-test"],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    assert all(
        Path(run["artifact_directory"]).is_dir()
        for run in summary["runs"].values()
    )
    assert all(
        json.loads(
            (Path(run["artifact_directory"]) / "provenance.json").read_text(
                encoding="utf-8"
            )
        )["seed"]
        == 42
        for run in summary["runs"].values()
    )
    assert all(
        Path(run["artifact_directory"])
        .resolve()
        .is_relative_to(Path(tempfile.gettempdir()).resolve())
        for run in summary["runs"].values()
    )
