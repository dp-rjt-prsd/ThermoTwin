from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_model_dataset import _artifacts
from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.data.transition_artifacts import (
    TransitionArtifacts,
    load_transition_artifacts,
)
from thermotwin.data.types import DatasetSamplingConfig, TransitionRecord


def _generation_config(directory: Path) -> None:
    config = EnvironmentConfig(
        PackConfig(1, 2, 1.0, 1000.0, 0.01, 2.0, 5.0, 10.0, 0.02),
        CoolingConfig(0.0, 0.1, 100.0, 2.0, 1.0),
        SimulationConfig(1.0, 0.8, 300.0, 305.0, 290.0, 0.0),
    )
    sampling = DatasetSamplingConfig(
        10.0,
        1.0,
        (0.5, 0.9),
        (295.0, 305.0),
        (300.0, 310.0),
        (285.0, 295.0),
        (1.0, 100.0),
        (0.0, 0.1),
        (0.8, 1.2),
        (0.8, 1.2),
        (0.8, 1.2),
        ("constant",),
    )
    (directory / "generation_config.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "seed": 3,
                "trajectory_count": 4,
                "base_environment_config": asdict(config),
                "sampling_config": asdict(sampling),
                "source_config_files": [],
            }
        ),
        encoding="utf-8",
    )


def _artifacts_with_provenance(directory: Path) -> Path:
    artifact_dir = _artifacts(directory)
    _generation_config(artifact_dir)
    return artifact_dir


def test_transition_loader_reconstructs_csv_order_and_alignment(tmp_path: Path) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")

    artifacts = load_transition_artifacts(artifact_dir)

    assert isinstance(artifacts, TransitionArtifacts)
    assert type(artifacts.base_environment_config) is EnvironmentConfig
    assert artifacts.base_environment_config.pack.cell_count == 2
    assert artifacts.base_environment_config.resistance_fault is None
    for split_name in ("train", "validation", "test"):
        records = getattr(artifacts, split_name)
        frame = pd.read_csv(artifact_dir / f"{split_name}.csv")
        dataset = getattr(artifacts.model_datasets, split_name)
        assert len(records) == len(dataset) == len(frame)
        assert all(type(record) is TransitionRecord for record in records)
        assert [record.trajectory_id for record in records] == frame[
            "trajectory_id"
        ].tolist()
        assert [record.step_index for record in records] == frame["step_index"].tolist()
        for row_index, record in enumerate(records):
            for scalar in (
                "time_s",
                "current_a",
                "soc_fraction",
                "ambient_temperature_k",
                "coolant_inlet_temperature_k",
                "coolant_mass_flow_kg_s",
            ):
                assert (
                    getattr(record.state, scalar)
                    == frame.at[row_index, f"state_{scalar}"]
                )
                assert (
                    getattr(record.next_state, scalar)
                    == frame.at[row_index, f"next_state_{scalar}"]
                )
            assert record.time_step_s == frame.at[row_index, "time_step_s"]
            assert (
                record.action.coolant_mass_flow_kg_s
                == frame.at[row_index, "action_coolant_mass_flow_kg_s"]
            )
            for cell in range(2):
                assert (
                    record.state.cell_temperatures_k[cell]
                    == frame.at[row_index, f"state_cell_temperature_{cell}_k"]
                )
                assert (
                    record.state.cell_resistances_ohm[cell]
                    == frame.at[row_index, f"state_cell_resistance_{cell}_ohm"]
                )
                assert (
                    record.next_state.cell_temperatures_k[cell]
                    == frame.at[row_index, f"next_state_cell_temperature_{cell}_k"]
                )
                assert (
                    record.next_state.cell_resistances_ohm[cell]
                    == frame.at[row_index, f"next_state_cell_resistance_{cell}_ohm"]
                )
        features = np.column_stack(
            (
                frame["state_current_a"],
                frame["state_soc_fraction"],
                frame[[f"state_cell_temperature_{i}_k" for i in range(2)]],
                frame["state_ambient_temperature_k"],
                frame["state_coolant_inlet_temperature_k"],
                frame["action_coolant_mass_flow_kg_s"],
            )
        )
        deltas = (
            frame[[f"next_state_cell_temperature_{i}_k" for i in range(2)]].to_numpy()
            - frame[[f"state_cell_temperature_{i}_k" for i in range(2)]].to_numpy()
        )
        np.testing.assert_allclose(
            dataset.features,
            artifacts.model_datasets.normalization.transform_features(features),
        )
        np.testing.assert_allclose(
            dataset.targets,
            artifacts.model_datasets.normalization.transform_targets(deltas),
        )
    assert artifacts.train[0].metadata["step"] == 0


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"schema_version":"2.0"}', "schema_version"),
        ('{"schema_version":"1.0"}', "invalid keys"),
        (
            '{"schema_version":"1.0","schema_version":"1.0"}',
            "duplicate",
        ),
        ('{"schema_version":"1.0","x":NaN}', "nonfinite"),
    ],
)
def test_transition_loader_rejects_malformed_generation_json(
    tmp_path: Path, contents: str, message: str
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")
    (artifact_dir / "generation_config.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_transition_artifacts(artifact_dir)


def test_transition_loader_rejects_invalid_base_environment_config(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    path = artifact_dir / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["base_environment_config"]["pack"]["rows"] = True
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="base_environment_config"):
        load_transition_artifacts(artifact_dir)


def test_transition_loader_accepts_legacy_fault_free_environment_config(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    path = artifact_dir / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    del config["base_environment_config"]["resistance_fault"]
    path.write_text(json.dumps(config), encoding="utf-8")

    artifacts = load_transition_artifacts(artifact_dir)

    assert artifacts.base_environment_config.resistance_fault is None


def test_transition_loader_reconstructs_resistance_fault_provenance(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    path = artifact_dir / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["base_environment_config"]["resistance_fault"] = {
        "cell_index": 1,
        "start_time_s": 4.0,
        "resistance_multiplier": 2.5,
    }
    path.write_text(json.dumps(config), encoding="utf-8")

    artifacts = load_transition_artifacts(artifact_dir)

    fault = artifacts.base_environment_config.resistance_fault
    assert type(fault) is ResistanceFaultConfig
    assert fault.cell_index == 1
    assert fault.start_time_s == 4.0
    assert fault.resistance_multiplier == 2.5


@pytest.mark.parametrize(
    "fault",
    [
        {"cell_index": 1, "start_time_s": 4.0},
        {
            "cell_index": 1,
            "start_time_s": 4.0,
            "resistance_multiplier": 2.5,
            "unknown": True,
        },
        {"cell_index": 1, "start_time_s": 4.0, "resistance_multiplier": 1.0},
        {"cell_index": True, "start_time_s": 4.0, "resistance_multiplier": 2.5},
    ],
)
def test_transition_loader_rejects_malformed_resistance_fault(
    tmp_path: Path, fault: object
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    path = artifact_dir / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["base_environment_config"]["resistance_fault"] = fault
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="base_environment_config"):
        load_transition_artifacts(artifact_dir)


def test_transition_loader_rejects_unknown_environment_config_keys(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    path = artifact_dir / "generation_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["base_environment_config"]["unexpected"] = None
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="base_environment_config"):
        load_transition_artifacts(artifact_dir)


def test_transition_loader_rejects_nonscalar_or_duplicate_metadata(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    frame = pd.read_csv(
        artifact_dir / "train.csv", dtype="string", keep_default_na=False
    )
    frame.loc[0, "metadata_json"] = '{"step":0,"extra":[1]}'
    frame.to_csv(artifact_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="metadata"):
        load_transition_artifacts(artifact_dir)

    frame.loc[0, "metadata_json"] = '{"step":0,"step":1}'
    frame.to_csv(artifact_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_transition_artifacts(artifact_dir)


def test_transition_loader_rejects_out_of_order_trajectory_rows(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts_with_provenance(tmp_path / "data")
    frame = pd.read_csv(artifact_dir / "train.csv")
    # Add a second transition by duplicating and then reorder the two steps.
    second = frame.iloc[0].copy()
    second["step_index"] = 1
    second["time_step_s"] = 1.0
    second["state_time_s"] = 1.0
    second["next_state_time_s"] = 2.0
    frame = pd.concat([frame, second.to_frame().T], ignore_index=True)
    frame.loc[0, "step_index"] = 1
    frame.loc[len(frame) - 1, "step_index"] = 0
    frame.to_csv(artifact_dir / "train.csv", index=False)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="order"):
        load_transition_artifacts(artifact_dir)
