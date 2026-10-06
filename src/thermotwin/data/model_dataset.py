"""Validated, train-normalized transition datasets for thermal models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from thermotwin.data.preprocessing import NormalizationStats, fit_normalization

_SCHEMA_VERSION = "1.0"
_SPLITS = ("train", "validation", "test")
_STATE_SCALARS = (
    "time_s",
    "current_a",
    "soc_fraction",
    "ambient_temperature_k",
    "coolant_inlet_temperature_k",
    "coolant_mass_flow_kg_s",
)


def _cell_count(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("manifest cell_count must be a positive integer")
    return value


def feature_names(cell_count: int) -> tuple[str, ...]:
    """Build feature names in the model's fixed input order."""
    count = _cell_count(cell_count)
    return (
        "state_current_a",
        "state_soc_fraction",
        *(f"state_cell_temperature_{index}_k" for index in range(count)),
        "state_ambient_temperature_k",
        "state_coolant_inlet_temperature_k",
        "action_coolant_mass_flow_kg_s",
    )


def target_names(cell_count: int) -> tuple[str, ...]:
    """Build one-step cell temperature-delta names in cell order."""
    count = _cell_count(cell_count)
    return tuple(f"delta_cell_temperature_{index}_k" for index in range(count))


def _expected_columns(cell_count: int) -> tuple[str, ...]:
    columns = ["trajectory_id", "step_index", "time_step_s"]
    for prefix in ("state", "action", "next_state"):
        if prefix == "action":
            columns.append("action_coolant_mass_flow_kg_s")
        else:
            columns.extend(f"{prefix}_{field}" for field in _STATE_SCALARS)
            columns.extend(
                f"{prefix}_cell_temperature_{index}_k"
                for index in range(cell_count)
            )
            columns.extend(
                f"{prefix}_cell_resistance_{index}_ohm"
                for index in range(cell_count)
            )
    columns.append("metadata_json")
    return tuple(columns)


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"manifest contains invalid numeric constant {value}")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_object_pairs_no_duplicates,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read valid manifest.json: {error}") from error
    if type(value) is not dict:
        raise ValueError("manifest must be a JSON object")
    if (
        type(value.get("schema_version")) is not str
        or value["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version must be '1.0'")
    return cast(dict[str, Any], value)


def _manifest_ids(value: object, split: str) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"manifest {split} trajectory_ids must be a nonempty array")
    identifiers = tuple(value)
    if any(type(item) is not str or not item.strip() for item in identifiers):
        raise ValueError(
            f"manifest {split} trajectory_ids must contain nonempty strings"
        )
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"manifest {split} trajectory_ids must be unique")
    return identifiers


def _validate_manifest(
    manifest: dict[str, Any],
) -> tuple[int, dict[str, tuple[str, ...]], dict[str, int]]:
    cell_count = _cell_count(manifest.get("cell_count"))
    columns = manifest.get("columns")
    expected_columns = _expected_columns(cell_count)
    if type(columns) is not list or tuple(columns) != expected_columns:
        raise ValueError("manifest columns do not match the required dataset schema")

    split_manifest = manifest.get("splits")
    if type(split_manifest) is not dict or set(split_manifest) != set(_SPLITS):
        raise ValueError("manifest splits must define train, validation, and test")

    ids_by_split: dict[str, tuple[str, ...]] = {}
    rows_by_split: dict[str, int] = {}
    for split in _SPLITS:
        metadata = split_manifest[split]
        if type(metadata) is not dict:
            raise ValueError(f"manifest {split} metadata must be an object")
        row_count = metadata.get("row_count")
        trajectory_count = metadata.get("trajectory_count")
        if type(row_count) is not int or row_count <= 0:
            raise ValueError(f"manifest {split} row_count must be a positive integer")
        if type(trajectory_count) is not int or trajectory_count <= 0:
            raise ValueError(
                f"manifest {split} trajectory_count must be a positive integer"
            )
        ids = _manifest_ids(metadata.get("trajectory_ids"), split)
        if trajectory_count != len(ids):
            raise ValueError(f"manifest {split} trajectory_count does not match IDs")
        ids_by_split[split] = ids
        rows_by_split[split] = row_count

    for index, split in enumerate(_SPLITS):
        for other in _SPLITS[index + 1 :]:
            if set(ids_by_split[split]) & set(ids_by_split[other]):
                raise ValueError("trajectory IDs must be disjoint between splits")
    return cell_count, ids_by_split, rows_by_split


def _read_split(
    path: Path,
    split: str,
    expected_columns: tuple[str, ...],
    expected_ids: tuple[str, ...],
    expected_rows: int,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
        raise ValueError(f"could not read {split}.csv: {error}") from error
    if tuple(frame.columns) != expected_columns:
        raise ValueError(f"{split}.csv columns do not match manifest schema")
    if len(frame) != expected_rows:
        raise ValueError(
            f"{split}.csv row count {len(frame)} does not match "
            f"manifest row_count {expected_rows}"
        )
    identifiers = frame["trajectory_id"]
    if identifiers.isna().any() or (identifiers.str.strip() == "").any():
        raise ValueError(f"{split}.csv trajectory IDs must be nonempty strings")
    observed_ids = set(identifiers.tolist())
    if observed_ids != set(expected_ids):
        raise ValueError(f"{split}.csv trajectory membership does not match manifest")

    for name in expected_columns:
        if name in {"trajectory_id", "metadata_json"}:
            continue
        values = frame[name]
        if values.dtype.kind == "b":
            raise ValueError(
                f"{split}.csv column {name} must contain finite numeric values"
            )
        try:
            numeric = pd.to_numeric(values, errors="raise")
            numeric_values = numeric.to_numpy(dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{split}.csv column {name} must contain finite numeric values"
            ) from error
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"{split}.csv column {name} must contain finite numeric values"
            )
        frame[name] = numeric_values
    step_values = frame["step_index"].to_numpy(dtype=np.float64)
    if np.any(step_values < 0.0) or np.any(step_values != np.floor(step_values)):
        raise ValueError(f"{split}.csv step_index values must be nonnegative integers")
    return frame


def _raw_arrays(
    frame: pd.DataFrame, count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features = frame.loc[:, list(feature_names(count))].to_numpy(
        dtype=np.float64, copy=True
    )
    raw_next = frame.loc[
        :,
        [f"next_state_cell_temperature_{index}_k" for index in range(count)],
    ].to_numpy(dtype=np.float64, copy=True)
    raw_state = frame.loc[
        :,
        [f"state_cell_temperature_{index}_k" for index in range(count)],
    ].to_numpy(dtype=np.float64, copy=True)
    targets = raw_next - raw_state
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("model features and temperature deltas must be finite")
    return features, targets


@dataclass(frozen=True, slots=True)
class ThermalTransitionDataset(Dataset[tuple[Tensor, Tensor]]):
    """Map-style, immutable copy of normalized transition rows."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        targets = np.asarray(self.targets)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must be a nonempty rank-2 array")
        if (
            targets.ndim != 2
            or targets.shape[0] != features.shape[0]
            or targets.shape[1] == 0
        ):
            raise ValueError("targets must be rank 2 with matching nonempty rows")
        if features.dtype.kind not in "iuf" or targets.dtype.kind not in "iuf":
            raise ValueError("features and targets must contain real numeric values")
        owned_features = np.array(features, dtype=np.float32, copy=True)
        owned_targets = np.array(targets, dtype=np.float32, copy=True)
        if (
            not np.isfinite(owned_features).all()
            or not np.isfinite(owned_targets).all()
        ):
            raise ValueError("features and targets must contain only finite values")
        if type(self.trajectory_ids) is not tuple or not self.trajectory_ids:
            raise ValueError("trajectory_ids must be a nonempty immutable tuple")
        if any(
            type(identifier) is not str or not identifier.strip()
            for identifier in self.trajectory_ids
        ):
            raise ValueError("trajectory_ids must contain nonempty strings")
        if len(self.trajectory_ids) != len(set(self.trajectory_ids)):
            raise ValueError("trajectory_ids must be unique")
        owned_features.setflags(write=False)
        owned_targets.setflags(write=False)
        object.__setattr__(self, "features", owned_features)
        object.__setattr__(self, "targets", owned_targets)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def target_count(self) -> int:
        return int(self.targets.shape[1])


@dataclass(frozen=True, slots=True)
class ModelDatasets:
    """Three trajectory-disjoint datasets and their train-fitted statistics."""

    train: ThermalTransitionDataset
    validation: ThermalTransitionDataset
    test: ThermalTransitionDataset
    normalization: NormalizationStats
    cell_count: int


def _dataset_directory_path(directory: object) -> Path:
    if isinstance(directory, bool) or not isinstance(directory, (str, os.PathLike)):
        raise TypeError("directory must be a string or path-like directory")
    try:
        return Path(directory)
    except (TypeError, ValueError, OSError) as error:
        raise ValueError(f"invalid dataset directory path: {error}") from error


def load_model_datasets(
    directory: str | os.PathLike[str],
    normalization: NormalizationStats | None = None,
) -> ModelDatasets:
    """Load split artifacts, fitting on train rows or reusing supplied statistics."""
    root = _dataset_directory_path(directory)
    if not root.is_dir():
        raise ValueError("dataset directory must exist and be a directory")

    manifest = _read_manifest(root / "manifest.json")
    count, ids_by_split, rows_by_split = _validate_manifest(manifest)
    columns = _expected_columns(count)
    frames: dict[str, pd.DataFrame] = {}
    raw_features: dict[str, NDArray[np.float64]] = {}
    raw_targets: dict[str, NDArray[np.float64]] = {}
    for split in _SPLITS:
        frame = _read_split(
            root / f"{split}.csv",
            split,
            columns,
            ids_by_split[split],
            rows_by_split[split],
        )
        features, targets = _raw_arrays(frame, count)
        frames[split] = frame
        raw_features[split] = features
        raw_targets[split] = targets

    feature_order = feature_names(count)
    target_order = target_names(count)
    if normalization is None:
        stats = fit_normalization(
            raw_features["train"], raw_targets["train"], feature_order, target_order
        )
    else:
        if type(normalization) is not NormalizationStats:
            raise TypeError("normalization must be a NormalizationStats or None")
        if normalization.feature_names != feature_order:
            raise ValueError("normalization feature_names do not match dataset schema")
        if normalization.target_names != target_order:
            raise ValueError("normalization target_names do not match dataset schema")
        if (
            normalization.feature_mean.size != len(feature_order)
            or normalization.feature_scale.size != len(feature_order)
            or normalization.target_mean.size != len(target_order)
            or normalization.target_scale.size != len(target_order)
        ):
            raise ValueError("normalization dimensions do not match dataset schema")
        stats = normalization
    datasets: dict[str, ThermalTransitionDataset] = {}
    for split in _SPLITS:
        frame = frames[split]
        # Preserve the order of first appearance in each CSV for audit metadata.
        ids = tuple(dict.fromkeys(frame["trajectory_id"].tolist()))
        datasets[split] = ThermalTransitionDataset(
            stats.transform_features(raw_features[split]).astype(np.float32),
            stats.transform_targets(raw_targets[split]).astype(np.float32),
            ids,
        )
    return ModelDatasets(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        stats,
        count,
    )
