"""Public contracts and operations for synthetic thermal datasets."""

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

__all__ = [
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
