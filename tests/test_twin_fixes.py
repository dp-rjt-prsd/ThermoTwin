"""Tests for segmented coolant flow, collocation rows, and the comfort reference."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from test_physics_dataset import _artifacts
from test_predictive_objective import _config, _rollout
from test_sampling import base_environment_config, sampling_config  # noqa: F401
from thermotwin.control.objective import evaluate_predictive_candidate
from thermotwin.control.predictive_types import PredictiveObjectiveWeights
from thermotwin.core.types import EnvironmentConfig
from thermotwin.data.physics_dataset import (
    build_collocation_dataset,
    physics_context,
)
from thermotwin.data.sampling import sample_dataset_scenario
from thermotwin.data.types import DatasetSamplingConfig
from thermotwin.models.config import PhysicsLossConfig


def test_constant_flow_is_unchanged_when_segments_are_not_configured(
    base_environment_config: EnvironmentConfig,  # noqa: F811
    sampling_config: DatasetSamplingConfig,  # noqa: F811
) -> None:
    scenario = sample_dataset_scenario(base_environment_config, sampling_config, 3, 1)

    assert sampling_config.coolant_flow_segment_s is None
    assert scenario.coolant_flow_profile(0.0) == scenario.coolant_flow_profile(19.0)
    assert "coolant_flow_segment_s" not in scenario.metadata


def test_segmented_flow_changes_between_segments_and_stays_in_range(
    base_environment_config: EnvironmentConfig,  # noqa: F811
    sampling_config: DatasetSamplingConfig,  # noqa: F811
) -> None:
    segmented = replace(sampling_config, coolant_flow_segment_s=5.0)
    plain = sample_dataset_scenario(base_environment_config, sampling_config, 3, 1)
    scenario = sample_dataset_scenario(base_environment_config, segmented, 3, 1)

    flows = [scenario.coolant_flow_profile(t) for t in (0.0, 4.0, 5.0, 10.0, 15.0)]
    lower, upper = segmented.coolant_mass_flow_kg_s_range
    assert flows[0] == flows[1] == plain.coolant_flow_profile(0.0)
    assert len({flows[0], flows[2], flows[3], flows[4]}) == 4
    assert all(lower <= flow <= upper for flow in flows)
    assert scenario.metadata["coolant_flow_segment_count"] == 4
    # The other sampled conditions are identical with and without segments.
    assert scenario.environment_config == plain.environment_config
    again = sample_dataset_scenario(base_environment_config, segmented, 3, 1)
    assert again.coolant_flow_profile(12.0) == scenario.coolant_flow_profile(12.0)


@pytest.mark.parametrize("segment", [0.0, -1.0, 2.5])
def test_segment_must_be_positive_multiple_of_time_step(
    sampling_config: DatasetSamplingConfig,  # noqa: F811
    segment: float,
) -> None:
    with pytest.raises(ValueError):
        replace(sampling_config, coolant_flow_segment_s=segment)


def test_physics_context_overrides_follow_joule_and_cooling_physics() -> None:
    artifacts = _artifacts()
    record = artifacts.train[0]
    config = artifacts.base_environment_config
    base = physics_context(record, config)
    no_current = physics_context(record, config, current_a=0.0)
    more_current = physics_context(record, config, current_a=50.0)
    more_flow = physics_context(
        record,
        config,
        coolant_mass_flow_kg_s=config.cooling.maximum_coolant_mass_flow_kg_s,
    )

    np.testing.assert_allclose(
        physics_context(record, config, current_a=record.state.current_a), base
    )
    assert np.all(more_current[2:] > no_current[2:])
    hot_cells = (
        record.state.cell_temperatures_k > record.state.coolant_inlet_temperature_k
    )
    assert np.all(more_flow[2:][hot_cells] <= base[2:][hot_cells])


def test_collocation_rows_sample_current_and_flow_within_bounds() -> None:
    artifacts = _artifacts()
    dataset = build_collocation_dataset(artifacts, 5, seed=7, max_current_a=300.0)
    stats = artifacts.model_datasets.normalization
    raw = dataset.features * stats.feature_scale + stats.feature_mean
    cooling = artifacts.base_environment_config.cooling

    assert len(dataset) == 5 * len(artifacts.train)
    assert np.all((raw[:, 0] >= 0.0) & (raw[:, 0] <= 300.0))
    assert np.all(
        (raw[:, -1] >= cooling.minimum_coolant_mass_flow_kg_s - 1e-6)
        & (raw[:, -1] <= cooling.maximum_coolant_mass_flow_kg_s + 1e-6)
    )
    assert not np.any(dataset.targets)
    again = build_collocation_dataset(artifacts, 5, seed=7, max_current_a=300.0)
    np.testing.assert_array_equal(dataset.features, again.features)
    features, targets, contexts = dataset[0]
    assert features.dtype == targets.dtype == contexts.dtype == torch.float32


@pytest.mark.parametrize(
    ("samples", "current"), [(0, 300.0), (-1, 300.0), (2, 0.0), (2, float("nan"))]
)
def test_collocation_rejects_invalid_settings(samples: int, current: float) -> None:
    with pytest.raises(ValueError):
        build_collocation_dataset(_artifacts(), samples, seed=1, max_current_a=current)


def test_physics_loss_config_validates_collocation_settings() -> None:
    config = PhysicsLossConfig(0.1, 10.0, collocation_samples_per_row=2)
    assert config.collocation_samples_per_row == 2
    assert PhysicsLossConfig(0.1, 10.0).collocation_samples_per_row == 0
    with pytest.raises(ValueError):
        PhysicsLossConfig(0.1, 10.0, collocation_samples_per_row=-1)
    with pytest.raises(ValueError):
        PhysicsLossConfig(0.1, 10.0, collocation_max_current_a=0.0)


def test_comfort_reference_removes_peak_cost_below_reference() -> None:
    rollout = _rollout()  # worst predicted cell is 304 K
    absolute = evaluate_predictive_candidate(rollout, _config(peak=3.0))
    config = _config(peak=3.0)
    above = replace(
        config,
        objective_weights=replace(
            config.objective_weights, peak_temperature_reference_k=302.0
        ),
    )
    below = replace(
        config,
        objective_weights=replace(
            config.objective_weights, peak_temperature_reference_k=310.0
        ),
    )

    assert absolute.peak_temperature_cost == pytest.approx(3.0 * 304.0)
    assert evaluate_predictive_candidate(rollout, above).peak_temperature_cost == (
        pytest.approx(3.0 * 2.0)
    )
    assert evaluate_predictive_candidate(rollout, below).peak_temperature_cost == 0.0


def test_comfort_reference_must_be_nonnegative() -> None:
    with pytest.raises(ValueError):
        PredictiveObjectiveWeights(
            1.0, 1.0, 1.0, 1.0, peak_temperature_reference_k=-1.0
        )
