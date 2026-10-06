from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data.model_dataset import (
    ModelDatasets,
    ThermalTransitionDataset,
    feature_names,
    target_names,
)
from thermotwin.data.physics_dataset import (
    PhysicsTransitionDataset,
    build_physics_datasets,
    physics_context,
    physics_context_names,
)
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.data.transition_artifacts import TransitionArtifacts
from thermotwin.data.types import TransitionRecord
from thermotwin.physics.environment import BatteryEnvironment


def _config() -> EnvironmentConfig:
    return EnvironmentConfig(
        PackConfig(1, 2, 1.0, 1000.0, 0.01, 2.0, 5.0, 10.0, 0.02),
        CoolingConfig(0.0, 0.2, 100.0, 20.0, 1.0),
        SimulationConfig(1.0, 0.8, 300.0, 300.0, 290.0, 0.0),
    )


def _record(
    config: EnvironmentConfig,
    *,
    temperatures: tuple[float, float] = (300.0, 300.0),
    current: float = 0.0,
    resistances: tuple[float, float] = (0.01, 0.01),
    flow: float = 0.1,
    ambient: float = 300.0,
    coolant: float = 290.0,
    dt: float = 1.0,
    neighbor_scale: float = 1.0,
    cooling_scale: float = 1.0,
    trajectory: str = "run-a",
    step: int = 0,
) -> TransitionRecord:
    sampled_pack = replace(
        config.pack,
        neighbor_thermal_resistance_k_w=(
            config.pack.neighbor_thermal_resistance_k_w * neighbor_scale
        ),
    )
    sampled_cooling = replace(
        config.cooling,
        base_heat_transfer_coefficient_w_m2_k=(
            config.cooling.base_heat_transfer_coefficient_w_m2_k * cooling_scale
        ),
        flow_heat_transfer_gain=config.cooling.flow_heat_transfer_gain * cooling_scale,
    )
    sampled_config = replace(config, pack=sampled_pack, cooling=sampled_cooling)
    state = BatteryState(
        time_s=float(step),
        current_a=current,
        soc_fraction=0.8,
        ambient_temperature_k=ambient,
        coolant_inlet_temperature_k=coolant,
        coolant_mass_flow_kg_s=flow,
        cell_temperatures_k=np.asarray(temperatures),
        cell_resistances_ohm=np.asarray(resistances),
    )
    action = CoolingAction(flow)
    environment = BatteryEnvironment(lambda _time: current)
    environment.reset(sampled_config)
    environment._state = state  # Use the exact persisted row as the simulator state.
    next_state = environment.step(action, dt)
    return TransitionRecord(
        trajectory,
        step,
        dt,
        state,
        action,
        next_state,
        {
            "neighbor_thermal_resistance_scale": neighbor_scale,
            "cooling_effectiveness_scale": cooling_scale,
        },
    )


def _artifacts() -> TransitionArtifacts:
    config = _config()
    split_records = {
        "train": (_record(config, trajectory="train-a"),),
        "validation": (_record(config, trajectory="validation-a", current=2.0),),
        "test": (_record(config, trajectory="test-a", current=3.0),),
    }
    stats = NormalizationStats(
        feature_names(2),
        target_names(2),
        np.zeros(7),
        np.ones(7),
        np.zeros(2),
        np.ones(2),
    )
    datasets: dict[str, ThermalTransitionDataset] = {}
    for split, records in split_records.items():
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
            ]
        )
        raw_targets = np.asarray(
            [
                record.next_state.cell_temperatures_k - record.state.cell_temperatures_k
                for record in records
            ]
        )
        datasets[split] = ThermalTransitionDataset(
            stats.transform_features(raw_features).astype(np.float32),
            stats.transform_targets(raw_targets).astype(np.float32),
            tuple(dict.fromkeys(record.trajectory_id for record in records)),
        )
    return TransitionArtifacts(
        ModelDatasets(
            datasets["train"],
            datasets["validation"],
            datasets["test"],
            stats,
            2,
        ),
        split_records["train"],
        split_records["validation"],
        split_records["test"],
        config,
    )


def test_physics_context_names_have_fixed_order() -> None:
    assert physics_context_names(2) == (
        "time_step_s",
        "cell_thermal_capacity_j_k",
        "net_heat_cell_0_w",
        "net_heat_cell_1_w",
    )


def test_physics_context_is_heat_balance_and_reproduces_euler_deltas() -> None:
    config = _config()
    record = _record(
        config,
        temperatures=(305.0, 300.0),
        current=20.0,
        resistances=(0.02, 0.01),
        ambient=295.0,
        coolant=290.0,
        dt=0.5,
        neighbor_scale=1.5,
        cooling_scale=0.75,
    )

    context = physics_context(record, config)

    assert context.dtype == np.float64
    assert context.shape == (4,)
    assert context[0] == record.time_step_s
    expected_capacity = (
        config.pack.cell_mass_kg * config.pack.cell_specific_heat_capacity_j_kg_k
    )
    assert context[1] == expected_capacity
    np.testing.assert_allclose(
        record.next_state.cell_temperatures_k - record.state.cell_temperatures_k,
        context[2:] * context[0] / context[1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_physics_context_equilibrium_and_joule_heating_scaling() -> None:
    config = _config()
    balanced = _record(
        config,
        temperatures=(300.0, 300.0),
        ambient=300.0,
        coolant=300.0,
        flow=0.0,
    )
    assert np.max(np.abs(physics_context(balanced, config)[2:])) < 1e-12

    low_heat = _record(config, current=10.0, flow=0.0)
    high_resistance = _record(config, current=10.0, resistances=(0.02, 0.02), flow=0.0)
    high_current = _record(config, current=20.0, flow=0.0)
    assert (
        physics_context(high_resistance, config)[2]
        > physics_context(low_heat, config)[2]
    )
    assert (
        physics_context(high_current, config)[2] > physics_context(low_heat, config)[2]
    )


def test_physics_context_cooling_conduction_and_ambient_signs() -> None:
    config = _config()
    no_cooling = _record(
        config, temperatures=(310.0, 300.0), coolant=290.0, flow=0.0, current=0.0
    )
    more_cooling = _record(
        config, temperatures=(310.0, 300.0), coolant=290.0, flow=0.2, current=0.0
    )
    assert np.sum(physics_context(more_cooling, config)[2:]) < np.sum(
        physics_context(no_cooling, config)[2:]
    )

    no_ambient = _record(config, temperatures=(310.0, 300.0), ambient=300.0, flow=0.0)
    hot_ambient = _record(config, temperatures=(310.0, 300.0), ambient=320.0, flow=0.0)
    assert np.sum(physics_context(hot_ambient, config)[2:]) > np.sum(
        physics_context(no_ambient, config)[2:]
    )

    conduction_only = _record(
        config,
        temperatures=(310.0, 300.0),
        ambient=305.0,
        coolant=310.0,
        flow=0.0,
        current=0.0,
    )
    weak_conduction_config = replace(
        config,
        pack=replace(config.pack, neighbor_thermal_resistance_k_w=1e20),
    )
    conduction = (
        physics_context(conduction_only, config)[2:]
        - physics_context(conduction_only, weak_conduction_config)[2:]
    )
    assert conduction[0] < 0.0 < conduction[1]
    assert np.sum(conduction) == pytest.approx(0.0, abs=1e-12)


def test_context_rejects_unvalidated_metadata_and_incompatible_record() -> None:
    config = _config()
    for invalid in (None, True, 0.0, "1.2"):
        record = _record(config)
        metadata = dict(record.metadata)
        metadata["cooling_effectiveness_scale"] = invalid
        bad = replace(record, metadata=metadata)
        with pytest.raises(ValueError, match="cooling_effectiveness_scale"):
            physics_context(bad, config)

    with pytest.raises(ValueError, match="cell count"):
        wrong_pack = PackConfig(1, 3, 1, 1000, 0.01, 2, 5, 10, 0.02)
        physics_context(_record(config), replace(config, pack=wrong_pack))


def test_physics_dataset_owns_immutable_rows_and_preserves_alignment() -> None:
    original_features = np.array([[1.0, 2.0]])
    original_targets = np.array([[3.0, 4.0]])
    original_context = np.array([[0.5, 1000.0, 6.0, 7.0]])
    dataset = PhysicsTransitionDataset(
        original_features, original_targets, original_context, ("run-a",)
    )
    original_features[0, 0] = 99.0
    original_targets[0, 0] = 99.0
    original_context[0, 0] = 99.0
    assert dataset.features[0, 0] == 1.0
    assert dataset.targets[0, 0] == 3.0
    assert dataset.physics_contexts[0, 0] == 0.5
    with pytest.raises(ValueError):
        dataset.physics_contexts.setflags(write=True)
    x, y, context = dataset[0]
    assert x.dtype == y.dtype == context.dtype == torch.float32
    assert x.shape == (2,) and y.shape == (2,) and context.shape == (4,)


@pytest.mark.parametrize(
    ("features", "targets", "contexts", "ids", "message"),
    [
        ([[1.0]], [[1.0, 2.0]], [[1.0, 2.0, 3.0]], ("run-a",), "context"),
        ([[np.nan]], [[1.0]], [[1.0, 2.0, 3.0]], ("run-a",), "finite"),
        ([[1.0]], [[1.0]], [[1.0, 2.0, 3.0]], (), "trajectory_ids"),
        ([[1.0], [2.0]], [[1.0]], [[1.0, 2.0, 3.0]], ("run-a",), "matching"),
    ],
)
def test_physics_dataset_validates_dimensions_and_rows(
    features: list[list[float]],
    targets: list[list[float]],
    contexts: list[list[float]],
    ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PhysicsTransitionDataset(features, targets, contexts, ids)  # type: ignore[arg-type]


def test_build_physics_datasets_keeps_split_rows_and_counts_aligned() -> None:
    artifacts = _artifacts()

    result = build_physics_datasets(artifacts)

    for split in ("train", "validation", "test"):
        built = getattr(result, split)
        source = getattr(artifacts.model_datasets, split)
        records = getattr(artifacts, split)
        assert len(built) == len(source) == len(records)
        assert built.feature_count == source.feature_count
        assert built.target_count == source.target_count == 2
        assert built.context_count == 4
        np.testing.assert_array_equal(built.features, source.features)
        np.testing.assert_array_equal(built.targets, source.targets)
        np.testing.assert_array_equal(
            built.physics_contexts,
            np.stack(
                [
                    physics_context(record, artifacts.base_environment_config)
                    for record in records
                ]
            ).astype(np.float32),
        )
        assert built.trajectory_ids == source.trajectory_ids


def test_build_rejects_misaligned_model_rows() -> None:
    artifacts = _artifacts()
    bad_train = ThermalTransitionDataset(
        artifacts.model_datasets.train.features + 1.0,
        artifacts.model_datasets.train.targets,
        artifacts.model_datasets.train.trajectory_ids,
    )
    bad_models = replace(artifacts.model_datasets, train=bad_train)
    with pytest.raises(ValueError, match="align"):
        build_physics_datasets(replace(artifacts, model_datasets=bad_models))
