from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from thermotwin.core import BatteryState, CoolingAction
from thermotwin.data.dataset import DatasetSplits, save_dataset_splits
from thermotwin.data.model_dataset import (
    ThermalTransitionDataset,
    feature_names,
    load_model_datasets,
    target_names,
)
from thermotwin.data.types import TransitionRecord


def _record(trajectory: str, step: int, offset: float = 0.0) -> TransitionRecord:
    temperatures = np.array([300.0 + offset + step, 302.0 + offset + step])
    state = BatteryState(
        time_s=float(step),
        current_a=10.0 + offset,
        soc_fraction=0.8,
        ambient_temperature_k=305.0 + offset,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=0.01,
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=np.array([0.01, 0.011]),
    )
    action = CoolingAction(0.02 + offset)
    next_state = BatteryState(
        time_s=float(step + 1),
        current_a=999.0,
        soc_fraction=0.1,
        ambient_temperature_k=999.0,
        coolant_inlet_temperature_k=999.0,
        coolant_mass_flow_kg_s=action.coolant_mass_flow_kg_s,
        cell_temperatures_k=temperatures + np.array([0.5, -0.25]),
        cell_resistances_ohm=np.array([0.5, 0.5]),
    )
    return TransitionRecord(
        trajectory,
        step,
        1.0,
        state,
        action,
        next_state,
        {"step": step},
    )


def _artifacts(
    directory: Path, offsets: tuple[float, ...] = (0, 2, 4, 6, 8, 10)
) -> Path:
    train = (_record("train-a", 0), _record("train-b", 0, 2))
    validation = (_record("validation-a", 0, offsets[0]),)
    test = (_record("test-a", 0, offsets[1]),)
    return save_dataset_splits(
        DatasetSplits(train, validation, test, 3, 0.2, 0.2), directory
    )


def test_ordered_names_only_contain_current_state_and_controlled_action() -> None:
    assert feature_names(2) == (
        "state_current_a",
        "state_soc_fraction",
        "state_cell_temperature_0_k",
        "state_cell_temperature_1_k",
        "state_ambient_temperature_k",
        "state_coolant_inlet_temperature_k",
        "action_coolant_mass_flow_kg_s",
    )
    assert target_names(2) == (
        "delta_cell_temperature_0_k",
        "delta_cell_temperature_1_k",
    )


def test_loader_builds_delta_targets_normalizes_and_keeps_trajectories(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")

    bundle = load_model_datasets(artifact_dir)

    assert bundle.cell_count == 2
    assert bundle.train.trajectory_ids == ("train-a", "train-b")
    assert bundle.validation.trajectory_ids == ("validation-a",)
    assert bundle.test.trajectory_ids == ("test-a",)
    x, y = bundle.train[0]
    assert x.dtype == torch.float32 and x.shape == (7,)
    assert y.dtype == torch.float32 and y.shape == (2,)
    restored = bundle.normalization.inverse_transform_targets(y.numpy()[None, :])
    np.testing.assert_allclose(restored, np.array([[0.5, -0.25]], dtype=np.float64))
    assert bundle.train.features.dtype == np.float32
    assert bundle.train.targets.dtype == np.float32


def test_validation_and_test_shift_does_not_change_train_fitted_stats(
    tmp_path: Path,
) -> None:
    first = load_model_datasets(_artifacts(tmp_path / "first", (0, 2)))
    shifted = load_model_datasets(_artifacts(tmp_path / "shifted", (5000, 8000)))

    np.testing.assert_array_equal(
        first.normalization.feature_mean, shifted.normalization.feature_mean
    )
    np.testing.assert_array_equal(
        first.normalization.feature_scale, shifted.normalization.feature_scale
    )
    np.testing.assert_array_equal(
        first.normalization.target_mean, shifted.normalization.target_mean
    )
    np.testing.assert_array_equal(
        first.normalization.target_scale, shifted.normalization.target_scale
    )


def test_dataset_defensively_owns_arrays_and_is_map_style() -> None:
    values = np.array([[1.0, 2.0]])
    target = np.array([[3.0]])
    dataset = ThermalTransitionDataset(values, target, ("run-a",))
    values[0, 0] = 100.0
    target[0, 0] = 200.0

    features, targets = dataset[0]

    assert len(dataset) == 1
    assert dataset.feature_count == 2
    assert dataset.target_count == 1
    assert features.tolist() == [1.0, 2.0]
    assert targets.tolist() == [3.0]
    with pytest.raises((AttributeError, TypeError)):
        dataset.trajectory_ids += ("run-b",)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"cell_count": True}, "cell_count"),
        ({"cell_count": "2"}, "cell_count"),
        ({"cell_count": 0}, "cell_count"),
    ],
)
def test_loader_rejects_ambiguous_or_invalid_manifest_values(
    tmp_path: Path, manifest_update: dict[str, Any], message: str
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=message):
        load_model_datasets(artifact_dir)


def test_loader_rejects_row_count_membership_and_schema_mismatches(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["train"]["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="row_count"):
        load_model_datasets(artifact_dir)

    manifest["splits"]["train"]["row_count"] -= 1
    manifest["splits"]["train"]["trajectory_ids"] = ["wrong"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="trajectory"):
        load_model_datasets(artifact_dir)

    manifest["splits"]["train"]["trajectory_ids"] = ["train-a", "train-b"]
    manifest["splits"]["train"]["trajectory_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    frame = pd.read_csv(artifact_dir / "test.csv")
    frame = frame.drop(columns=["state_current_a"])
    frame.to_csv(artifact_dir / "test.csv", index=False)
    with pytest.raises(ValueError, match="columns"):
        load_model_datasets(artifact_dir)


@pytest.mark.parametrize(
    "bad_value", ["not-a-number", True, float("inf"), float("nan")]
)
def test_loader_rejects_nonfinite_or_nonnumeric_csv_values(
    tmp_path: Path, bad_value: Any
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")
    frame = pd.read_csv(artifact_dir / "train.csv")
    frame["state_current_a"] = frame["state_current_a"].astype(object)
    frame.loc[0, "state_current_a"] = bad_value
    frame.to_csv(artifact_dir / "train.csv", index=False)

    with pytest.raises(ValueError, match="finite numeric"):
        load_model_datasets(artifact_dir)


def test_loader_rejects_overlapping_ids_missing_files_and_bad_paths(
    tmp_path: Path,
) -> None:
    artifact_dir = _artifacts(tmp_path / "data")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["splits"]["validation"]["trajectory_ids"] = ["train-a"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disjoint"):
        load_model_datasets(artifact_dir)

    manifest["splits"]["validation"]["trajectory_ids"] = ["validation-a"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_dir / "validation.csv").unlink()
    with pytest.raises(ValueError, match="validation.csv"):
        load_model_datasets(artifact_dir)
    with pytest.raises((TypeError, ValueError)):
        load_model_datasets(object())  # type: ignore[arg-type]


def test_dataset_rejects_incompatible_nonfinite_arrays_and_id_count() -> None:
    with pytest.raises(ValueError, match="finite"):
        ThermalTransitionDataset(np.array([[np.inf]]), np.array([[1.0]]), ("a",))
    with pytest.raises(ValueError, match="matching"):
        ThermalTransitionDataset(np.array([[1.0]]), np.array([[1.0], [2.0]]), ("a",))
    with pytest.raises(ValueError, match="trajectory"):
        ThermalTransitionDataset(np.array([[1.0]]), np.array([[1.0]]), ())
