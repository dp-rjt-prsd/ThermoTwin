"""Integration checks for reproducible dataset generation artifacts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_dataset.py"
ARTIFACTS = {
    "train.csv",
    "validation.csv",
    "test.csv",
    "manifest.json",
    "generation_config.json",
}


def _run_generator(output_directory: Path | None, seed: int = 42) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if part
    )
    command = [sys.executable, str(SCRIPT), "--smoke-test", "--seed", str(seed)]
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
    from thermotwin.data.sampling import sample_dataset_scenario
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
        "sample_dataset_scenario",
        "generate_trajectory",
        "generate_dataset",
        "split_by_trajectory",
        "save_dataset_splits",
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
            sample_dataset_scenario,
            generate_trajectory,
            generate_dataset,
            split_by_trajectory,
            save_dataset_splits,
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
    base_config = provenance["base_environment_config"]
    assert set(base_config) == {"pack", "cooling", "simulation"}
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
    }
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
