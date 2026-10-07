"""Integration checks for reproducible dataset generation artifacts."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_dataset.py"
ARTIFACTS = {
    "train.csv",
    "validation.csv",
    "test.csv",
    "manifest.json",
    "generation_config.json",
}


def _run_generator(
    output_directory: Path | None, seed: int = 42, *, ood: bool = False
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    command = [sys.executable, str(SCRIPT), "--smoke-test", "--seed", str(seed)]
    if ood:
        command.append("--ood")
    if output_directory is not None:
        command.extend(("--output-directory", str(output_directory)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, Any], json.loads(completed.stdout))


def test_public_data_exports_are_explicit() -> None:
    import thermotwin.data as data
    from thermotwin.data.dataset import (
        DatasetSplits,
        save_dataset_splits,
        split_by_trajectory,
    )
    from thermotwin.data.generate import generate_dataset, generate_trajectory
    from thermotwin.data.model_dataset import (
        ModelDatasets,
        ThermalTransitionDataset,
        feature_names,
        load_model_datasets,
        target_names,
    )
    from thermotwin.data.physics_dataset import (
        PhysicsDatasets,
        PhysicsTransitionDataset,
        build_physics_datasets,
        physics_context,
        physics_context_names,
    )
    from thermotwin.data.preprocessing import NormalizationStats, fit_normalization
    from thermotwin.data.sampling import sample_dataset_scenario
    from thermotwin.data.transition_artifacts import (
        TransitionArtifacts,
        load_transition_artifacts,
    )
    from thermotwin.data.types import (
        CoolantFlowProfile,
        DatasetMetadataValue,
        DatasetSamplingConfig,
        DatasetScenario,
        TransitionRecord,
    )

    expected = [
        "CoolantFlowProfile",
        "DatasetMetadataValue",
        "DatasetSamplingConfig",
        "DatasetScenario",
        "TransitionRecord",
        "DatasetSplits",
        "NormalizationStats",
        "fit_normalization",
        "feature_names",
        "target_names",
        "ThermalTransitionDataset",
        "ModelDatasets",
        "load_model_datasets",
        "sample_dataset_scenario",
        "generate_trajectory",
        "generate_dataset",
        "split_by_trajectory",
        "save_dataset_splits",
        "TransitionArtifacts",
        "load_transition_artifacts",
        "physics_context_names",
        "physics_context",
        "PhysicsTransitionDataset",
        "PhysicsDatasets",
        "build_physics_datasets",
    ]
    assert data.__all__ == expected
    for name, value in zip(
        expected,
        (
            CoolantFlowProfile,
            DatasetMetadataValue,
            DatasetSamplingConfig,
            DatasetScenario,
            TransitionRecord,
            DatasetSplits,
            NormalizationStats,
            fit_normalization,
            feature_names,
            target_names,
            ThermalTransitionDataset,
            ModelDatasets,
            load_model_datasets,
            sample_dataset_scenario,
            generate_trajectory,
            generate_dataset,
            split_by_trajectory,
            save_dataset_splits,
            TransitionArtifacts,
            load_transition_artifacts,
            physics_context_names,
            physics_context,
            PhysicsTransitionDataset,
            PhysicsDatasets,
            build_physics_datasets,
        ),
        strict=True,
    ):
        assert getattr(data, name) is value


def test_smoke_generation_persists_consistent_provenance_and_deterministic_files(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "dataset-first"
    summary = _run_generator(first_directory, seed=13)

    assert set(path.name for path in first_directory.iterdir()) == ARTIFACTS
    assert summary["output_directory"] == str(first_directory.resolve())
    assert summary["seed"] == 13
    assert summary["trajectory_count"] == 6
    assert summary["total_row_count"] == 60
    assert summary["distribution"] == "in_distribution"
    assert summary["artifacts"] == sorted(ARTIFACTS)

    manifest = json.loads(
        (first_directory / "manifest.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (first_directory / "generation_config.json").read_text(encoding="utf-8")
    )
    assert manifest["seed"] == 13
    assert manifest["cell_count"] == 12
    assert manifest["splits"] == summary["splits"]
    expected_rows = int(
        provenance["trajectory_count"]
        * provenance["sampling_config"]["duration_s"]
        / provenance["sampling_config"]["time_step_s"]
    )
    assert summary["total_row_count"] == expected_rows
    assert (
        sum(part["row_count"] for part in manifest["splits"].values())
        == expected_rows
    )
    assert sum(part["trajectory_count"] for part in manifest["splits"].values()) == 6
    split_ids = [
        set(part["trajectory_ids"]) for part in manifest["splits"].values()
    ]
    assert not (
        split_ids[0] & split_ids[1]
        or split_ids[0] & split_ids[2]
        or split_ids[1] & split_ids[2]
    )

    total_rows = 0
    for split_name in ("train", "validation", "test"):
        with (first_directory / f"{split_name}.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        total_rows += len(rows)
        assert len(rows) == summary["splits"][split_name]["row_count"]
        assert all(
            f"state_cell_temperature_{index}_k" in rows[0] for index in range(12)
        )
        assert all(
            f"next_state_cell_temperature_{index}_k" in rows[0]
            for index in range(12)
        )
        assert all(
            f"state_cell_resistance_{index}_ohm" in rows[0]
            for index in range(12)
        )
        assert all(
            f"next_state_cell_resistance_{index}_ohm" in rows[0]
            for index in range(12)
        )
    assert total_rows == summary["total_row_count"]

    assert provenance["schema_version"] == "1.0"
    assert provenance["seed"] == 13
    assert provenance["trajectory_count"] == 6
    assert provenance["source_config_files"] == [
        "configs/pack.yaml",
        "configs/simulation.yaml",
        "configs/dataset.yaml",
    ]
    assert "distribution" not in provenance
    base_config = provenance["base_environment_config"]
    assert set(base_config) == {
        "pack",
        "cooling",
        "simulation",
        "resistance_fault",
    }
    assert base_config["resistance_fault"] is None
    assert set(base_config["pack"]) == {
        "rows",
        "columns",
        "cell_mass_kg",
        "cell_specific_heat_capacity_j_kg_k",
        "nominal_cell_resistance_ohm",
        "cell_capacity_ah",
        "neighbor_thermal_resistance_k_w",
        "ambient_thermal_resistance_k_w",
        "cell_cooling_surface_area_m2",
    }
    assert set(base_config["cooling"]) == {
        "minimum_coolant_mass_flow_kg_s",
        "maximum_coolant_mass_flow_kg_s",
        "base_heat_transfer_coefficient_w_m2_k",
        "flow_heat_transfer_gain",
        "flow_exponent",
    }
    assert set(base_config["simulation"]) == {
        "default_time_step_s",
        "initial_soc_fraction",
        "initial_cell_temperature_k",
        "ambient_temperature_k",
        "coolant_inlet_temperature_k",
        "initial_current_a",
        "initial_cell_temperatures_k",
        "initial_cell_resistances_ohm",
    }
    assert base_config["pack"]["rows"] == 3
    assert base_config["simulation"]["initial_cell_temperatures_k"] is None
    assert set(provenance["sampling_config"]) == {
        "duration_s",
        "time_step_s",
        "initial_soc_fraction_range",
        "initial_cell_temperature_k_range",
        "ambient_temperature_k_range",
        "coolant_inlet_temperature_k_range",
        "peak_current_a_range",
        "coolant_mass_flow_kg_s_range",
        "cell_resistance_scale_range",
        "neighbor_thermal_resistance_scale_range",
        "cooling_effectiveness_scale_range",
        "allowed_profile_names",
        "coolant_flow_segment_s",
    }
    assert provenance["sampling_config"]["coolant_flow_segment_s"] == 10.0
    assert provenance["sampling_config"]["duration_s"] == 10.0
    assert provenance["sampling_config"]["time_step_s"] == 1.0
    assert provenance["sampling_config"]["allowed_profile_names"] == [
        "constant",
        "pulse",
        "stop_and_go",
        "aggressive",
        "fast_charging",
    ]

    second_directory = tmp_path / "dataset-second"
    _run_generator(second_directory, seed=13)
    for filename in ARTIFACTS:
        assert (first_directory / filename).read_bytes() == (
            second_directory / filename
        ).read_bytes()


def test_ood_smoke_generation_uses_shifted_ranges_and_is_deterministic(
    tmp_path: Path,
) -> None:
    first_directory = tmp_path / "ood-first"
    summary = _run_generator(first_directory, seed=13, ood=True)
    provenance = json.loads(
        (first_directory / "generation_config.json").read_text(encoding="utf-8")
    )

    assert summary["distribution"] == "out_of_distribution"
    assert summary["trajectory_count"] == 6
    assert summary["total_row_count"] == 60
    assert provenance["sampling_config"]["duration_s"] == 10.0
    assert provenance["source_config_files"] == [
        "configs/pack.yaml",
        "configs/simulation.yaml",
        "configs/dataset_ood.yaml",
    ]
    assert "distribution" not in provenance

    normal = yaml.safe_load(
        (ROOT / "configs" / "dataset.yaml").read_text(encoding="utf-8")
    )["sampling"]
    ood = yaml.safe_load(
        (ROOT / "configs" / "dataset_ood.yaml").read_text(encoding="utf-8")
    )["sampling"]
    pack = yaml.safe_load(
        (ROOT / "configs" / "pack.yaml").read_text(encoding="utf-8")
    )
    shifted_fields = (
        "initial_soc_fraction_range",
        "initial_cell_temperature_k_range",
        "ambient_temperature_k_range",
        "coolant_inlet_temperature_k_range",
        "peak_current_a_range",
        "cell_resistance_scale_range",
        "neighbor_thermal_resistance_scale_range",
        "cooling_effectiveness_scale_range",
    )
    for field in shifted_fields:
        assert ood[field] == provenance["sampling_config"][field]
        assert ood[field][1] < normal[field][0] or ood[field][0] > normal[field][1]
    assert ood["coolant_mass_flow_kg_s_range"] == provenance["sampling_config"][
        "coolant_mass_flow_kg_s_range"
    ]
    assert ood["coolant_mass_flow_kg_s_range"][0] >= 0.075
    assert ood["coolant_mass_flow_kg_s_range"][1] <= pack["cooling"][
        "maximum_coolant_mass_flow_kg_s"
    ]

    second_directory = tmp_path / "ood-second"
    _run_generator(second_directory, seed=13, ood=True)
    for filename in ARTIFACTS:
        assert (first_directory / filename).read_bytes() == (
            second_directory / filename
        ).read_bytes()


def test_smoke_generation_without_output_directory_uses_os_temp() -> None:
    repository_data = ROOT / "data"
    existing_entries = (
        set(path.name for path in repository_data.iterdir())
        if repository_data.exists()
        else set()
    )

    summary = _run_generator(None)

    output_directory = Path(summary["output_directory"])
    assert output_directory.is_dir()
    assert output_directory.resolve().is_relative_to(
        Path(tempfile.gettempdir()).resolve()
    )
    assert not output_directory.resolve().is_relative_to(ROOT.resolve())
    assert set(path.name for path in output_directory.iterdir()) == ARTIFACTS
    assert (
        set(path.name for path in repository_data.iterdir())
        if repository_data.exists()
        else set()
    ) == existing_entries


def test_existing_output_directory_is_reported_without_traceback(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "existing-dataset"
    output_directory.mkdir()
    marker = output_directory / "keep.txt"
    marker.write_text("preserve this file", encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--smoke-test",
            "--output-directory",
            str(output_directory),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("generate_dataset: error: ")
    assert len(completed.stderr.splitlines()) == 1
    assert "Traceback" not in completed.stderr
    assert marker.read_text(encoding="utf-8") == "preserve this file"
    assert set(path.name for path in output_directory.iterdir()) == {"keep.txt"}


def test_missing_config_is_reported_without_traceback(tmp_path: Path) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    for filename in ("pack.yaml", "simulation.yaml"):
        (config_directory / filename).write_bytes(
            (ROOT / "configs" / filename).read_bytes()
        )

    harness = (
        "import importlib.util, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('dataset_script', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module._PROJECT_ROOT = pathlib.Path(sys.argv[2]); "
        "sys.exit(module.main(['--smoke-test', '--output-directory', sys.argv[3]]))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(SCRIPT),
            str(tmp_path),
            str(tmp_path / "dataset"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("generate_dataset: error: ")
    assert len(completed.stderr.splitlines()) == 1
    assert "dataset.yaml" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "dataset").exists()


def test_invalid_yaml_is_reported_without_traceback(tmp_path: Path) -> None:
    config_directory = tmp_path / "configs"
    config_directory.mkdir()
    (config_directory / "pack.yaml").write_text("pack: [", encoding="utf-8")

    harness = (
        "import importlib.util, pathlib, sys; "
        "spec = importlib.util.spec_from_file_location('dataset_script', sys.argv[1]); "
        "module = importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module._PROJECT_ROOT = pathlib.Path(sys.argv[2]); "
        "sys.exit(module.main(['--smoke-test', '--output-directory', sys.argv[3]]))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            harness,
            str(SCRIPT),
            str(tmp_path),
            str(tmp_path / "dataset"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("generate_dataset: error: ")
    assert len(completed.stderr.splitlines()) == 1
    assert "Traceback" not in completed.stderr
    assert not (tmp_path / "dataset").exists()


def test_unexpected_type_error_from_generation_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("dataset_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def broken_generator(args: Any) -> int:
        raise TypeError("unexpected programming defect")

    monkeypatch.setattr(module, "_generate", broken_generator)

    with pytest.raises(TypeError, match="unexpected programming defect"):
        module.main(["--smoke-test"])
