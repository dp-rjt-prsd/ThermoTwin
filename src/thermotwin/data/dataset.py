"""Trajectory-level dataset splitting and deterministic CSV persistence."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

from thermotwin.core import BatteryState
from thermotwin.data.types import TransitionRecord

_SCHEMA_VERSION = "1.0"
_STATE_SCALAR_FIELDS = (
    ("time_s", "time_s"),
    ("current_a", "current_a"),
    ("soc_fraction", "soc_fraction"),
    ("ambient_temperature_k", "ambient_temperature_k"),
    ("coolant_inlet_temperature_k", "coolant_inlet_temperature_k"),
    ("coolant_mass_flow_kg_s", "coolant_mass_flow_kg_s"),
)


def _normalize_fraction(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"{name} must be a finite fraction strictly between 0 and 1"
        ) from error
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        raise ValueError(f"{name} must be a finite fraction strictly between 0 and 1")
    return normalized


def _normalize_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _validate_record_collection(
    records: object,
) -> tuple[tuple[TransitionRecord, ...], dict[str, tuple[int, ...]], int]:
    if not isinstance(records, tuple):
        raise ValueError("each split must be an immutable tuple")
    if not records:
        raise ValueError("each split must be nonempty")
    if any(type(record) is not TransitionRecord for record in records):
        raise ValueError("split entries must be TransitionRecord instances")

    indices_by_trajectory: dict[str, list[int]] = {}
    cell_count: int | None = None
    for record in records:
        indices_by_trajectory.setdefault(record.trajectory_id, []).append(
            record.step_index
        )
        count = int(record.state.cell_temperatures_k.size)
        if cell_count is None:
            cell_count = count
        elif count != cell_count:
            raise ValueError("all transition records must use the same cell count")

    normalized_indices: dict[str, tuple[int, ...]] = {}
    for trajectory_id, indices in indices_by_trajectory.items():
        ordered = tuple(sorted(indices))
        if ordered != tuple(range(len(ordered))):
            raise ValueError(
                f"trajectory {trajectory_id!r} must have contiguous zero-based steps"
            )
        normalized_indices[trajectory_id] = ordered
    assert cell_count is not None
    return records, normalized_indices, cell_count


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Immutable, validated sets of whole-trajectory transitions."""

    train: tuple[TransitionRecord, ...]
    validation: tuple[TransitionRecord, ...]
    test: tuple[TransitionRecord, ...]
    seed: int
    validation_fraction: float
    test_fraction: float

    def __post_init__(self) -> None:
        seed = _normalize_seed(self.seed)
        validation_fraction = _normalize_fraction(
            "validation_fraction", self.validation_fraction
        )
        test_fraction = _normalize_fraction("test_fraction", self.test_fraction)
        if validation_fraction + test_fraction >= 1.0:
            raise ValueError(
                "validation_fraction and test_fraction must sum to less than 1"
            )

        split_trajectories: list[set[str]] = []
        cell_counts: set[int] = set()
        for name in ("train", "validation", "test"):
            records, indices, cell_count = _validate_record_collection(
                getattr(self, name)
            )
            object.__setattr__(self, name, records)
            split_trajectories.append(set(indices))
            cell_counts.add(cell_count)

        if len(cell_counts) != 1:
            raise ValueError("all splits must use the same cell count")
        if (
            split_trajectories[0] & split_trajectories[1]
            or split_trajectories[0] & split_trajectories[2]
            or split_trajectories[1] & split_trajectories[2]
        ):
            raise ValueError("trajectory IDs must be disjoint between splits")

        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "validation_fraction", validation_fraction)
        object.__setattr__(self, "test_fraction", test_fraction)

    @property
    def cell_count(self) -> int:
        """Number of cells represented by every transition in these splits."""
        return int(self.train[0].state.cell_temperatures_k.size)


def _allocated_counts(
    trajectory_count: int, validation_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    """Use largest remainder allocation while reserving one ID per split."""
    weights = (
        1.0 - validation_fraction - test_fraction,
        validation_fraction,
        test_fraction,
    )
    extra_count = trajectory_count - 3
    quotas = tuple(extra_count * weight for weight in weights)
    counts = [1 + math.floor(quota) for quota in quotas]
    remaining = trajectory_count - sum(counts)
    order = sorted(
        range(3),
        key=lambda index: (-(quotas[index] - math.floor(quotas[index])), index),
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def split_by_trajectory(
    records: Sequence[TransitionRecord],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplits:
    """Assign complete trajectories deterministically to three dataset splits."""
    validation_fraction = _normalize_fraction(
        "validation_fraction", validation_fraction
    )
    test_fraction = _normalize_fraction("test_fraction", test_fraction)
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction and test_fraction must sum to less than 1"
        )
    seed = _normalize_seed(seed)
    if not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of TransitionRecord values")

    validated_records = tuple(records)
    if not validated_records:
        raise ValueError("records must be nonempty")
    if any(type(record) is not TransitionRecord for record in validated_records):
        raise ValueError("records must contain only TransitionRecord instances")
    _, records_by_trajectory, _ = _validate_as_record_sequence(validated_records)
    trajectory_ids = sorted(records_by_trajectory)
    if len(trajectory_ids) < 3:
        raise ValueError("at least three unique trajectories are required")

    shuffled_ids = trajectory_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)
    train_count, validation_count, _ = _allocated_counts(
        len(shuffled_ids), validation_fraction, test_fraction
    )
    train_ids = set(shuffled_ids[:train_count])
    validation_ids = set(shuffled_ids[train_count : train_count + validation_count])
    test_ids = set(shuffled_ids[train_count + validation_count :])

    return DatasetSplits(
        train=tuple(
            record for record in validated_records if record.trajectory_id in train_ids
        ),
        validation=tuple(
            record
            for record in validated_records
            if record.trajectory_id in validation_ids
        ),
        test=tuple(
            record for record in validated_records if record.trajectory_id in test_ids
        ),
        seed=seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def _validate_as_record_sequence(
    records: tuple[TransitionRecord, ...],
) -> tuple[tuple[TransitionRecord, ...], dict[str, tuple[int, ...]], int]:
    """Validate sequence records without requiring each trajectory to be split yet."""
    by_trajectory: dict[str, list[int]] = {}
    cell_count: int | None = None
    for record in records:
        by_trajectory.setdefault(record.trajectory_id, []).append(record.step_index)
        count = int(record.state.cell_temperatures_k.size)
        if cell_count is None:
            cell_count = count
        elif count != cell_count:
            raise ValueError("all transition records must use the same cell count")
    normalized: dict[str, tuple[int, ...]] = {}
    for trajectory_id, indices in by_trajectory.items():
        ordered = tuple(sorted(indices))
        if ordered != tuple(range(len(ordered))):
            raise ValueError(
                f"trajectory {trajectory_id!r} must have contiguous zero-based steps"
            )
        normalized[trajectory_id] = ordered
    assert cell_count is not None
    return records, normalized, cell_count


def _column_names(cell_count: int) -> list[str]:
    fields = ["trajectory_id", "step_index", "time_step_s"]
    for prefix in ("state", "action", "next_state"):
        if prefix == "action":
            fields.append("action_coolant_mass_flow_kg_s")
            continue
        fields.extend(f"{prefix}_{name}" for _, name in _STATE_SCALAR_FIELDS)
        fields.extend(
            f"{prefix}_cell_temperature_{index}_k" for index in range(cell_count)
        )
        fields.extend(
            f"{prefix}_cell_resistance_{index}_ohm" for index in range(cell_count)
        )
    fields.append("metadata_json")
    return fields


def _state_values(
    state: BatteryState, prefix: str, cell_count: int
) -> dict[str, object]:
    values: dict[str, object] = {
        f"{prefix}_{column}": getattr(state, attribute)
        for attribute, column in _STATE_SCALAR_FIELDS
    }
    values.update(
        {
            f"{prefix}_cell_temperature_{index}_k": float(temperature)
            for index, temperature in enumerate(state.cell_temperatures_k)
        }
    )
    values.update(
        {
            f"{prefix}_cell_resistance_{index}_ohm": float(resistance)
            for index, resistance in enumerate(state.cell_resistances_ohm)
        }
    )
    if len(values) != len(_STATE_SCALAR_FIELDS) + 2 * cell_count:
        raise ValueError("state cell count does not match dataset schema")
    return values


def _csv_content(records: tuple[TransitionRecord, ...], columns: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row: dict[str, object] = {
            "trajectory_id": record.trajectory_id,
            "step_index": record.step_index,
            "time_step_s": record.time_step_s,
            "action_coolant_mass_flow_kg_s": record.action.coolant_mass_flow_kg_s,
            "metadata_json": json.dumps(
                dict(record.metadata),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }
        row.update(
            _state_values(record.state, "state", len(record.state.cell_temperatures_k))
        )
        row.update(
            _state_values(
                record.next_state,
                "next_state",
                len(record.next_state.cell_temperatures_k),
            )
        )
        writer.writerow(row)
    return stream.getvalue()


def _json_content(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def save_dataset_splits(
    splits: DatasetSplits,
    output_directory: str | os.PathLike[str],
) -> Path:
    """Persist splits under a newly created directory without overwriting."""
    if type(splits) is not DatasetSplits:
        raise TypeError("splits must be exactly a DatasetSplits")

    cell_count = splits.cell_count
    columns = _column_names(cell_count)
    split_records = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    csv_contents = {
        name: _csv_content(records, columns) for name, records in split_records.items()
    }
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "seed": splits.seed,
        "validation_fraction": splits.validation_fraction,
        "test_fraction": splits.test_fraction,
        "cell_count": cell_count,
        "columns": columns,
        "splits": {
            name: {
                "row_count": len(records),
                "trajectory_count": len({record.trajectory_id for record in records}),
                "trajectory_ids": sorted({record.trajectory_id for record in records}),
            }
            for name, records in split_records.items()
        },
    }
    manifest_content = _json_content(manifest)

    output_path = Path(output_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(exist_ok=False)
    for name, content in csv_contents.items():
        (output_path / f"{name}.csv").write_text(content, encoding="utf-8", newline="")
    (output_path / "manifest.json").write_text(
        manifest_content, encoding="utf-8", newline=""
    )
    return output_path
