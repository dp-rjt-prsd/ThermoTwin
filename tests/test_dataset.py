from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thermotwin.core import BatteryState, CoolingAction
from thermotwin.data.dataset import (
    DatasetSplits,
    save_dataset_splits,
    split_by_trajectory,
)
from thermotwin.data.types import TransitionRecord


def make_record(
    trajectory_id: str, step_index: int, cell_count: int = 2
) -> TransitionRecord:
    flow = 0.01 + step_index * 0.001
    state = BatteryState(
        time_s=float(step_index),
        current_a=10.0 + step_index,
        soc_fraction=0.8 - step_index * 0.01,
        ambient_temperature_k=305.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=np.array(
            [300.0 + cell + step_index for cell in range(cell_count)]
        ),
        cell_resistances_ohm=np.array(
            [0.01 + cell * 0.001 for cell in range(cell_count)]
        ),
    )
    action = CoolingAction(flow + 0.001)
    next_state = BatteryState(
        time_s=float(step_index + 1),
        current_a=state.current_a,
        soc_fraction=state.soc_fraction,
        ambient_temperature_k=state.ambient_temperature_k,
        coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
        coolant_mass_flow_kg_s=action.coolant_mass_flow_kg_s,
        cell_temperatures_k=state.cell_temperatures_k + 0.5,
        cell_resistances_ohm=state.cell_resistances_ohm,
    )
    return TransitionRecord(
        trajectory_id,
        step_index,
        1.0,
        state,
        action,
        next_state,
        {"profile": "pulse", "step": step_index, "valid": True},
    )


def records_for_trajectories(
    identifiers: tuple[str, ...] = (
        "run-a",
        "run-b",
        "run-c",
        "run-d",
        "run-e",
        "run-f",
    ),
    *,
    steps: int = 2,
    cell_count: int = 2,
) -> tuple[TransitionRecord, ...]:
    return tuple(
        make_record(identifier, step, cell_count)
        for identifier in identifiers
        for step in range(steps)
    )


def test_split_keeps_whole_trajectories_and_original_record_order() -> None:
    records = records_for_trajectories()

    splits = split_by_trajectory(records, 0.2, 0.2, seed=41)

    memberships = [
        {record.trajectory_id for record in split}
        for split in (splits.train, splits.validation, splits.test)
    ]
    assert not memberships[0] & memberships[1]
    assert not memberships[0] & memberships[2]
    assert not memberships[1] & memberships[2]
    assert set.union(*memberships) == {record.trajectory_id for record in records}
    for split in (splits.train, splits.validation, splits.test):
        assert split == tuple(
            record
            for record in records
            if record.trajectory_id in {item.trajectory_id for item in split}
        )


def test_seeded_membership_is_deterministic_but_changes_with_seed() -> None:
    records = records_for_trajectories(tuple(f"run-{index}" for index in range(12)))

    first = split_by_trajectory(records, 0.25, 0.25, 772)
    repeated = split_by_trajectory(records, 0.25, 0.25, 772)
    changed = split_by_trajectory(records, 0.25, 0.25, 773)

    def membership(result: DatasetSplits) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(sorted({record.trajectory_id for record in part}))
            for part in (result.train, result.validation, result.test)
        )

    assert membership(first) == membership(repeated)
    assert membership(first) != membership(changed)


def test_split_does_not_change_global_random_state() -> None:
    records = records_for_trajectories()
    random.seed(123)
    expected = random.getstate()

    split_by_trajectory(records, 0.2, 0.2, 8)

    assert random.getstate() == expected


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction", "seed"),
    [
        (0.0, 0.2, 1),
        (1.0, 0.1, 1),
        (float("nan"), 0.1, 1),
        (0.1, float("inf"), 1),
        (0.6, 0.4, 1),
        (0.1, 0.1, -1),
        (0.1, 0.1, True),
        (0.1, 0.1, 1.0),
    ],
)
def test_split_rejects_invalid_fractions_or_seed(
    validation_fraction: float, test_fraction: float, seed: Any
) -> None:
    with pytest.raises((TypeError, ValueError)):
        split_by_trajectory(
            records_for_trajectories(), validation_fraction, test_fraction, seed
        )


def test_split_requires_three_trajectories_and_validates_all_records() -> None:
    with pytest.raises(ValueError, match="three"):
        split_by_trajectory(records_for_trajectories(("run-a", "run-b")), 0.2, 0.2, 1)

    noncontiguous = (make_record("run-a", 0), make_record("run-a", 2))
    invalid = records_for_trajectories(("run-b", "run-c")) + noncontiguous
    with pytest.raises(ValueError, match="contiguous"):
        split_by_trajectory(invalid, 0.2, 0.2, 1)

    with pytest.raises((TypeError, ValueError)):
        split_by_trajectory([object()], 0.2, 0.2, 1)  # type: ignore[list-item]


def test_dataset_splits_contract_rejects_mutable_or_leaking_splits() -> None:
    record = make_record("run-a", 0)
    mutable_validation: Any = [make_record("run-b", 0)]
    with pytest.raises(ValueError):
        DatasetSplits(
            (record,),
            mutable_validation,
            (make_record("run-c", 0),),
            1,
            0.2,
            0.2,
        )
    with pytest.raises(ValueError, match="disjoint"):
        DatasetSplits(
            (record,),
            (make_record("run-a", 0),),
            (make_record("run-c", 0),),
            1,
            0.2,
            0.2,
        )


def test_save_flattens_all_twelve_cells_and_writes_accurate_manifest(
    tmp_path: Path,
) -> None:
    splits = split_by_trajectory(
        records_for_trajectories(cell_count=12), 0.2, 0.2, seed=31
    )

    output = save_dataset_splits(splits, tmp_path / "nested" / "dataset")

    assert output == tmp_path / "nested" / "dataset"
    with (output / "train.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    first_record = splits.train[0]
    first_row = rows[0]
    assert first_row["trajectory_id"] == first_record.trajectory_id
    assert int(first_row["step_index"]) == first_record.step_index
    assert float(first_row["time_step_s"]) == 1.0
    assert float(first_row["state_current_a"]) == 10.0
    assert float(first_row["state_soc_fraction"]) == 0.8
    assert float(first_row["state_ambient_temperature_k"]) == 305.0
    assert float(first_row["state_coolant_inlet_temperature_k"]) == 290.0
    assert float(first_row["state_coolant_mass_flow_kg_s"]) == pytest.approx(0.01)
    assert float(first_row["state_cell_temperature_11_k"]) == 311.0
    assert float(first_row["state_cell_resistance_11_ohm"]) == pytest.approx(0.021)
    assert float(first_row["action_coolant_mass_flow_kg_s"]) == pytest.approx(
        first_record.action.coolant_mass_flow_kg_s
    )
    assert float(first_row["action_coolant_mass_flow_kg_s"]) == pytest.approx(0.011)
    assert float(first_row["next_state_time_s"]) == 1.0
    assert float(first_row["next_state_current_a"]) == 10.0
    assert float(first_row["next_state_soc_fraction"]) == 0.8
    assert float(first_row["next_state_ambient_temperature_k"]) == 305.0
    assert float(first_row["next_state_coolant_inlet_temperature_k"]) == 290.0
    assert float(first_row["next_state_coolant_mass_flow_kg_s"]) == pytest.approx(0.011)
    assert float(first_row["next_state_cell_temperature_11_k"]) == 311.5
    assert float(first_row["next_state_cell_resistance_11_ohm"]) == pytest.approx(0.021)
    assert all(f"state_cell_temperature_{index}_k" in first_row for index in range(12))
    assert all(f"state_cell_resistance_{index}_ohm" in first_row for index in range(12))
    assert all(
        f"next_state_cell_temperature_{index}_k" in first_row for index in range(12)
    )
    assert all(
        f"next_state_cell_resistance_{index}_ohm" in first_row for index in range(12)
    )
    assert json.loads(first_row["metadata_json"]) == {
        "profile": "pulse",
        "step": 0,
        "valid": True,
    }

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["seed"] == 31
    assert manifest["validation_fraction"] == 0.2
    assert manifest["test_fraction"] == 0.2
    assert manifest["cell_count"] == 12
    assert manifest["columns"] == list(rows[0])
    for name, part in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        assert manifest["splits"][name] == {
            "row_count": len(part),
            "trajectory_count": len({record.trajectory_id for record in part}),
            "trajectory_ids": sorted({record.trajectory_id for record in part}),
        }


def test_saved_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    splits = split_by_trajectory(records_for_trajectories(), 0.2, 0.2, seed=19)

    first = save_dataset_splits(splits, tmp_path / "first")
    second = save_dataset_splits(splits, tmp_path / "second")

    for filename in ("train.csv", "validation.csv", "test.csv", "manifest.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_save_rejects_wrong_type_and_existing_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        save_dataset_splits(object(), tmp_path / "not-created")  # type: ignore[arg-type]
    assert not (tmp_path / "not-created").exists()

    destination = tmp_path / "dataset"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    splits = split_by_trajectory(records_for_trajectories(), 0.2, 0.2, seed=2)
    with pytest.raises(FileExistsError):
        save_dataset_splits(splits, destination)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in destination.iterdir()) == ["keep.txt"]
