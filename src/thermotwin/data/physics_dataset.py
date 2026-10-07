"""Row-aligned simulator heat-balance contexts for thermal-model training."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from thermotwin.core import EnvironmentConfig
from thermotwin.data.model_dataset import (
    ModelDatasets,
    ThermalTransitionDataset,
    feature_names,
    target_names,
)
from thermotwin.data.transition_artifacts import TransitionArtifacts
from thermotwin.data.types import TransitionRecord
from thermotwin.physics.cell import joule_heating_w
from thermotwin.physics.cooling import (
    ambient_heat_exchange_w,
    cooling_heat_removal_w,
)
from thermotwin.physics.pack import neighbor_conduction_w
from thermotwin.physics.topology import pack_edges

_SPLITS = ("train", "validation", "test")


def _cell_count(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("cell_count must be a positive integer")
    return value


def physics_context_names(cell_count: int) -> tuple[str, ...]:
    """Name context values in their fixed training order."""
    count = _cell_count(cell_count)
    return (
        "time_step_s",
        "cell_thermal_capacity_j_k",
        *(f"net_heat_cell_{index}_w" for index in range(count)),
    )


def _metadata_scale(record: TransitionRecord, name: str) -> float:
    value = record.metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata {name} must be a positive finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"metadata {name} must be a positive finite scalar")
    return scalar


def physics_context(
    record: TransitionRecord,
    base_environment_config: EnvironmentConfig,
    *,
    current_a: float | None = None,
    coolant_mass_flow_kg_s: float | None = None,
) -> NDArray[np.float64]:
    """Build ``dt, thermal capacity, net heat per cell`` from authoritative physics.

    ``current_a`` and ``coolant_mass_flow_kg_s`` optionally replace the recorded
    values, giving the energy balance at an unlabeled collocation point.
    """
    if type(record) is not TransitionRecord:
        raise ValueError("record must be a TransitionRecord")
    if type(base_environment_config) is not EnvironmentConfig:
        raise ValueError("base_environment_config must be an EnvironmentConfig")
    pack = base_environment_config.pack
    state = record.state
    cell_count = state.cell_temperatures_k.size
    if cell_count != pack.cell_count:
        raise ValueError(
            "record cell count must match base environment pack cell count"
        )

    neighbor_scale = _metadata_scale(record, "neighbor_thermal_resistance_scale")
    cooling_scale = _metadata_scale(record, "cooling_effectiveness_scale")
    sampled_pack = replace(
        pack,
        neighbor_thermal_resistance_k_w=(
            pack.neighbor_thermal_resistance_k_w * neighbor_scale
        ),
    )
    base_cooling = base_environment_config.cooling
    sampled_cooling = replace(
        base_cooling,
        base_heat_transfer_coefficient_w_m2_k=(
            base_cooling.base_heat_transfer_coefficient_w_m2_k * cooling_scale
        ),
        flow_heat_transfer_gain=base_cooling.flow_heat_transfer_gain * cooling_scale,
    )

    current = state.current_a if current_a is None else float(current_a)
    flow = (
        record.action.coolant_mass_flow_kg_s
        if coolant_mass_flow_kg_s is None
        else float(coolant_mass_flow_kg_s)
    )
    generated_heat = joule_heating_w(current, state.cell_resistances_ohm)
    cooling_heat = cooling_heat_removal_w(
        state.cell_temperatures_k,
        state.coolant_inlet_temperature_k,
        flow,
        pack.cell_cooling_surface_area_m2,
        sampled_cooling,
    )
    neighbor_heat = neighbor_conduction_w(
        state.cell_temperatures_k,
        pack_edges(sampled_pack),
        sampled_pack.neighbor_thermal_resistance_k_w,
    )
    ambient_heat = ambient_heat_exchange_w(
        state.cell_temperatures_k,
        state.ambient_temperature_k,
        pack.ambient_thermal_resistance_k_w,
    )
    net_heat = generated_heat - cooling_heat + neighbor_heat + ambient_heat
    capacity = pack.cell_mass_kg * pack.cell_specific_heat_capacity_j_kg_k
    context = np.asarray((record.time_step_s, capacity, *net_heat), dtype=np.float64)
    if not np.isfinite(context).all():
        raise ValueError("physics context must contain only finite values")
    return context


def _immutable_float32_matrix(value: object, name: str) -> NDArray[np.float32]:
    try:
        array = np.asarray(value)
        if np.iscomplexobj(array) or array.dtype.kind not in "iuf":
            raise ValueError
        owned = np.array(array, dtype=np.float32, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real numeric matrix") from error
    if owned.ndim != 2 or owned.shape[0] == 0 or owned.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty rank-2 matrix")
    if not np.isfinite(owned).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(owned.tobytes(), dtype=np.float32).reshape(owned.shape)


@dataclass(frozen=True, slots=True)
class PhysicsTransitionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Immutable normalized transitions paired with per-row physics contexts."""

    features: NDArray[np.float32]
    targets: NDArray[np.float32]
    physics_contexts: NDArray[np.float32]
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        features = _immutable_float32_matrix(self.features, "features")
        targets = _immutable_float32_matrix(self.targets, "targets")
        contexts = _immutable_float32_matrix(self.physics_contexts, "physics_contexts")
        if (
            targets.shape[0] != features.shape[0]
            or contexts.shape[0] != features.shape[0]
        ):
            raise ValueError(
                "features, targets, and physics_contexts must have matching rows"
            )
        if contexts.shape[1] != targets.shape[1] + 2:
            raise ValueError(
                "physics_contexts width must equal two scalars plus target cells"
            )
        if type(self.trajectory_ids) is not tuple or not self.trajectory_ids:
            raise ValueError("trajectory_ids must be a nonempty immutable tuple")
        if any(
            type(identifier) is not str or not identifier.strip()
            for identifier in self.trajectory_ids
        ):
            raise ValueError("trajectory_ids must contain nonempty strings")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("trajectory_ids must be unique")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "physics_contexts", contexts)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def row_count(self) -> int:
        return len(self)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(self.targets[index], dtype=torch.float32),
            torch.tensor(self.physics_contexts[index], dtype=torch.float32),
        )

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def target_count(self) -> int:
        return int(self.targets.shape[1])

    @property
    def context_count(self) -> int:
        return int(self.physics_contexts.shape[1])

    @property
    def cell_count(self) -> int:
        return self.context_count - 2


@dataclass(frozen=True, slots=True)
class PhysicsDatasets:
    """Train, validation, and test splits paired with immutable physics data."""

    train: PhysicsTransitionDataset
    validation: PhysicsTransitionDataset
    test: PhysicsTransitionDataset


def _split_dataset(
    split_name: str,
    source: ThermalTransitionDataset,
    records: tuple[TransitionRecord, ...],
    artifacts: TransitionArtifacts,
) -> PhysicsTransitionDataset:
    if type(records) is not tuple or not records:
        raise ValueError(f"{split_name} records must be a nonempty tuple")
    if len(records) != len(source):
        raise ValueError(f"{split_name} physics rows do not align with model rows")

    raw_features = np.asarray(
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
    raw_targets = np.asarray(
        [
            record.next_state.cell_temperatures_k - record.state.cell_temperatures_k
            for record in records
        ],
        dtype=np.float64,
    )
    normalization = artifacts.model_datasets.normalization
    expected_features = normalization.transform_features(raw_features).astype(
        np.float32
    )
    expected_targets = normalization.transform_targets(raw_targets).astype(np.float32)
    if not np.array_equal(source.features, expected_features) or not np.array_equal(
        source.targets, expected_targets
    ):
        raise ValueError(f"{split_name} physics records do not align with model rows")
    identifiers = tuple(dict.fromkeys(record.trajectory_id for record in records))
    if identifiers != source.trajectory_ids:
        raise ValueError(
            f"{split_name} trajectory order does not align with model rows"
        )
    contexts = np.stack(
        [
            physics_context(record, artifacts.base_environment_config)
            for record in records
        ]
    ).astype(np.float32)
    return PhysicsTransitionDataset(
        source.features,
        source.targets,
        contexts,
        source.trajectory_ids,
    )


def build_collocation_dataset(
    artifacts: TransitionArtifacts,
    samples_per_row: int,
    *,
    seed: int,
    max_current_a: float,
) -> PhysicsTransitionDataset:
    """Unlabeled physics-only rows built from the training split.

    Each training state is copied ``samples_per_row`` times with a random pack
    current in ``[0, max_current_a]`` and a random legal coolant flow. Only the
    energy-balance residual is defined for these rows; their targets are zero
    placeholders that the trainer never uses.
    """
    if type(artifacts) is not TransitionArtifacts:
        raise ValueError("artifacts must be a TransitionArtifacts")
    if isinstance(samples_per_row, bool) or not isinstance(samples_per_row, int):
        raise ValueError("samples_per_row must be a positive integer")
    if samples_per_row <= 0:
        raise ValueError("samples_per_row must be a positive integer")
    if not math.isfinite(max_current_a) or max_current_a <= 0.0:
        raise ValueError("max_current_a must be positive and finite")
    config = artifacts.base_environment_config
    cooling = config.cooling
    rng = np.random.default_rng(seed)
    raw_features: list[list[float]] = []
    contexts: list[NDArray[np.float64]] = []
    for record in artifacts.train:
        state = record.state
        for _ in range(samples_per_row):
            current = float(rng.uniform(0.0, max_current_a))
            flow = float(
                rng.uniform(
                    cooling.minimum_coolant_mass_flow_kg_s,
                    cooling.maximum_coolant_mass_flow_kg_s,
                )
            )
            raw_features.append(
                [
                    current,
                    state.soc_fraction,
                    *state.cell_temperatures_k,
                    state.ambient_temperature_k,
                    state.coolant_inlet_temperature_k,
                    flow,
                ]
            )
            contexts.append(
                physics_context(
                    record,
                    config,
                    current_a=current,
                    coolant_mass_flow_kg_s=flow,
                )
            )
    normalization = artifacts.model_datasets.normalization
    features = normalization.transform_features(
        np.asarray(raw_features, dtype=np.float64)
    ).astype(np.float32)
    cell_count = artifacts.model_datasets.cell_count
    return PhysicsTransitionDataset(
        features,
        np.zeros((features.shape[0], cell_count), dtype=np.float32),
        np.stack(contexts).astype(np.float32),
        artifacts.model_datasets.train.trajectory_ids,
    )


def build_physics_datasets(artifacts: TransitionArtifacts) -> PhysicsDatasets:
    """Pair normalized model rows with contexts computed from raw transitions."""
    if type(artifacts) is not TransitionArtifacts:
        raise ValueError("artifacts must be a TransitionArtifacts")
    models: ModelDatasets = artifacts.model_datasets
    cell_count = models.cell_count
    config = artifacts.base_environment_config
    if type(config) is not EnvironmentConfig or config.pack.cell_count != cell_count:
        raise ValueError("base environment cell count must match model datasets")
    if models.train.feature_count != len(feature_names(cell_count)):
        raise ValueError("model feature count does not match configured cell count")
    if models.train.target_count != len(target_names(cell_count)):
        raise ValueError("model target count does not match configured cell count")

    splits = {
        name: _split_dataset(
            name,
            getattr(models, name),
            getattr(artifacts, name),
            artifacts,
        )
        for name in _SPLITS
    }
    return PhysicsDatasets(splits["train"], splits["validation"], splits["test"])
