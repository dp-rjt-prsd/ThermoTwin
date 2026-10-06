"""Validated raw transition and generation provenance artifact loading."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data.dataset import _STATE_SCALAR_FIELDS
from thermotwin.data.model_dataset import (
    _SPLITS,
    ModelDatasets,
    _dataset_directory_path,
    _expected_columns,
    _read_manifest,
    _read_split,
    _validate_manifest,
    load_model_datasets,
)
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.data.types import (
    DatasetMetadataValue,
    DatasetSamplingConfig,
    TransitionRecord,
)

_SCHEMA_VERSION = "1.0"
_GENERATION_KEYS = frozenset(
    {
        "schema_version",
        "seed",
        "trajectory_count",
        "base_environment_config",
        "sampling_config",
        "source_config_files",
    }
)


def _duplicate_key_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"generation provenance contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"generation provenance contains nonfinite constant {value}")


def _read_generation_config(path: Path) -> EnvironmentConfig:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(
                stream,
                object_pairs_hook=_duplicate_key_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read valid generation_config.json: {error}"
        ) from error
    if type(raw) is not dict:
        raise ValueError("generation_config.json must be an object")
    if (
        type(raw.get("schema_version")) is not str
        or raw.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("generation_config.json schema_version must be '1.0'")
    if set(raw) != _GENERATION_KEYS:
        raise ValueError("generation_config.json has invalid keys")
    seed = raw["seed"]
    count = raw["trajectory_count"]
    if type(seed) is not int or seed < 0:
        raise ValueError("generation_config.json seed must be a nonnegative integer")
    if type(count) is not int or count < 3:
        raise ValueError("generation_config.json trajectory_count must be at least 3")
    sources = raw["source_config_files"]
    if type(sources) is not list or any(type(item) is not str for item in sources):
        raise ValueError("generation_config.json source_config_files must be strings")

    try:
        sampling = _strict_dataclass_dict(raw["sampling_config"], DatasetSamplingConfig)
        for field_name in (
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
        ):
            value = sampling[field_name]
            if type(value) is not list:
                raise ValueError(f"{field_name} must be a JSON array")
            sampling[field_name] = tuple(value)
        DatasetSamplingConfig(**sampling)

        environment = raw["base_environment_config"]
        if type(environment) is not dict or set(environment) != {
            "pack",
            "cooling",
            "simulation",
        }:
            raise ValueError("must contain exactly pack, cooling, and simulation")
        pack = _strict_dataclass_dict(environment["pack"], PackConfig)
        cooling = _strict_dataclass_dict(environment["cooling"], CoolingConfig)
        simulation = _strict_dataclass_dict(environment["simulation"], SimulationConfig)
        for field_name in (
            "initial_cell_temperatures_k",
            "initial_cell_resistances_ohm",
        ):
            value = simulation[field_name]
            if value is not None:
                if type(value) is not list:
                    raise ValueError(f"{field_name} must be a JSON array or null")
                simulation[field_name] = tuple(value)
        config = EnvironmentConfig(
            PackConfig(**pack), CoolingConfig(**cooling), SimulationConfig(**simulation)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "invalid generation_config.json base_environment_config or "
            f"sampling_config: {error}"
        ) from error
    return config


def _strict_dataclass_dict(value: object, contract: type[Any]) -> dict[str, Any]:
    expected = {field.name for field in fields(contract)}
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{contract.__name__} fields do not match the required schema")
    return cast(dict[str, Any], value)


def _metadata(value: object, split: str, row: int) -> dict[str, DatasetMetadataValue]:
    if not isinstance(value, str):
        raise ValueError(f"{split}.csv row {row} metadata_json must be a JSON object")
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_duplicate_key_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"{split}.csv row {row} metadata_json is invalid: {error}"
        ) from error
    if type(parsed) is not dict:
        raise ValueError(f"{split}.csv row {row} metadata_json must be an object")
    result: dict[str, DatasetMetadataValue] = {}
    for key, item in parsed.items():
        if type(key) is not str or not key.strip():
            raise ValueError(
                f"{split}.csv row {row} metadata keys must be nonempty strings"
            )
        if item is not None and type(item) not in (str, int, float, bool):
            raise ValueError(f"{split}.csv row {row} metadata values must be scalars")
        if type(item) is float and not math.isfinite(item):
            raise ValueError(f"{split}.csv row {row} metadata values must be finite")
        result[key] = cast(DatasetMetadataValue, item)
    return result


def _numeric_cell(frame: pd.DataFrame, row: int, column: str) -> float:
    value: object = frame.at[row, column]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"CSV column {column} must contain finite numeric values")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"CSV column {column} must contain finite numeric values")
    return normalized


def _text_cell(frame: pd.DataFrame, row: int, column: str) -> str:
    value: object = frame.at[row, column]
    if type(value) is not str:
        raise ValueError(f"CSV column {column} must contain text values")
    return value


def _state(frame: pd.DataFrame, row: int, prefix: str, cell_count: int) -> BatteryState:
    scalar_values = {
        attribute: _numeric_cell(frame, row, f"{prefix}_{column}")
        for attribute, column in _STATE_SCALAR_FIELDS
    }
    return BatteryState(
        **scalar_values,
        cell_temperatures_k=np.asarray(
            [
                _numeric_cell(frame, row, f"{prefix}_cell_temperature_{index}_k")
                for index in range(cell_count)
            ],
            dtype=np.float64,
        ),
        cell_resistances_ohm=np.asarray(
            [
                _numeric_cell(frame, row, f"{prefix}_cell_resistance_{index}_ohm")
                for index in range(cell_count)
            ],
            dtype=np.float64,
        ),
    )


def _records(
    frame: pd.DataFrame, split: str, cell_count: int
) -> tuple[TransitionRecord, ...]:
    records: list[TransitionRecord] = []
    next_step: dict[str, int] = {}
    for row in range(len(frame)):
        trajectory_id = _text_cell(frame, row, "trajectory_id")
        step_value = _numeric_cell(frame, row, "step_index")
        if step_value < 0.0 or step_value != math.floor(step_value):
            raise ValueError(f"{split}.csv row {row} step_index must be an integer")
        step_index = int(step_value)
        expected_step = next_step.get(trajectory_id, 0)
        if step_index != expected_step:
            raise ValueError(
                f"{split}.csv row {row} trajectory {trajectory_id!r} is out of order: "
                f"expected step {expected_step}, got {step_index}"
            )
        next_step[trajectory_id] = expected_step + 1
        try:
            records.append(
                TransitionRecord(
                    trajectory_id=trajectory_id,
                    step_index=step_index,
                    time_step_s=_numeric_cell(frame, row, "time_step_s"),
                    state=_state(frame, row, "state", cell_count),
                    action=CoolingAction(
                        _numeric_cell(frame, row, "action_coolant_mass_flow_kg_s")
                    ),
                    next_state=_state(frame, row, "next_state", cell_count),
                    metadata=_metadata(
                        _text_cell(frame, row, "metadata_json"), split, row
                    ),
                )
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{split}.csv row {row} is not a valid transition: {error}"
            ) from error
    return tuple(records)


def _record_matrices(
    records: tuple[TransitionRecord, ...],
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [
            [
                record.state.current_a,
                record.state.soc_fraction,
                *record.state.cell_temperatures_k,
                record.state.ambient_temperature_k,
                record.state.coolant_inlet_temperature_k,
                record.action.coolant_mass_flow_kg_s,
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            record.next_state.cell_temperatures_k - record.state.cell_temperatures_k
            for record in records
        ],
        dtype=np.float64,
    )
    return features, targets


@dataclass(frozen=True, slots=True)
class TransitionArtifacts:
    """Normalized model splits paired with the exact raw rows and provenance."""

    model_datasets: ModelDatasets
    train: tuple[TransitionRecord, ...]
    validation: tuple[TransitionRecord, ...]
    test: tuple[TransitionRecord, ...]
    base_environment_config: EnvironmentConfig


def load_transition_artifacts(
    directory: str | os.PathLike[str],
    normalization: NormalizationStats | None = None,
) -> TransitionArtifacts:
    """Load normalized model data, raw ordered transitions, and generation config."""
    root = _dataset_directory_path(directory)
    if not root.is_dir():
        raise ValueError("dataset directory must exist and be a directory")
    generation_config = _read_generation_config(root / "generation_config.json")
    model_datasets = load_model_datasets(root, normalization)
    manifest = _read_manifest(root / "manifest.json")
    cell_count, ids_by_split, rows_by_split = _validate_manifest(manifest)
    if cell_count != model_datasets.cell_count:
        raise ValueError("raw transition cell count does not match model datasets")
    if generation_config.pack.cell_count != cell_count:
        raise ValueError("base environment cell count does not match dataset schema")

    records: dict[str, tuple[TransitionRecord, ...]] = {}
    for split in _SPLITS:
        frame = _read_split(
            root / f"{split}.csv",
            split,
            _expected_columns(cell_count),
            ids_by_split[split],
            rows_by_split[split],
        )
        split_records = _records(frame, split, cell_count)
        dataset = getattr(model_datasets, split)
        if (
            len(split_records) != len(dataset)
            or tuple(dict.fromkeys(record.trajectory_id for record in split_records))
            != dataset.trajectory_ids
        ):
            raise ValueError(
                f"{split} raw transition order does not align with model rows"
            )
        raw_features, raw_targets = _record_matrices(split_records)
        expected_features = model_datasets.normalization.transform_features(
            raw_features
        ).astype(np.float32)
        expected_targets = model_datasets.normalization.transform_targets(
            raw_targets
        ).astype(np.float32)
        if not np.array_equal(
            dataset.features, expected_features
        ) or not np.array_equal(dataset.targets, expected_targets):
            raise ValueError(
                f"{split} raw transition values do not align with model rows"
            )
        records[split] = split_records

    return TransitionArtifacts(
        model_datasets=model_datasets,
        train=records["train"],
        validation=records["validation"],
        test=records["test"],
        base_environment_config=generation_config,
    )
