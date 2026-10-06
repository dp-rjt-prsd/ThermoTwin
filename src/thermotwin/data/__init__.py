"""Public contracts and operations for synthetic thermal datasets."""

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

__all__ = [
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
