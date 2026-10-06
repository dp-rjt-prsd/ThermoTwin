from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data.types import (
    DatasetMetadataValue,
    DatasetScenario,
    TransitionRecord,
)
from thermotwin.physics.environment import BatteryEnvironment


def make_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=2,
            cell_mass_kg=0.8,
            cell_specific_heat_capacity_j_kg_k=900.0,
            nominal_cell_resistance_ohm=0.015,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.5,
            ambient_thermal_resistance_k_w=8.0,
            cell_cooling_surface_area_m2=0.02,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.01,
            maximum_coolant_mass_flow_kg_s=0.1,
            base_heat_transfer_coefficient_w_m2_k=5.0,
            flow_heat_transfer_gain=500.0,
            flow_exponent=0.8,
        ),
        simulation=SimulationConfig(
            default_time_step_s=0.5,
            initial_soc_fraction=0.8,
            initial_cell_temperature_k=300.0,
            ambient_temperature_k=295.0,
            coolant_inlet_temperature_k=290.0,
            initial_current_a=99.0,
            initial_cell_resistances_ohm=(0.015, 0.03),
        ),
    )


def make_scenario(
    trajectory_id: str = "run_01",
    *,
    duration_s: float = 1.5,
    time_step_s: float = 0.5,
    current_profile: Any = None,
    coolant_flow_profile: Any = None,
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> DatasetScenario:
    return DatasetScenario(
        trajectory_id=trajectory_id,
        environment_config=make_environment_config(),
        current_profile=current_profile or (lambda time_s: 4.0 + time_s),
        coolant_flow_profile=coolant_flow_profile
        or (lambda time_s: 0.02 + time_s / 100),
        duration_s=duration_s,
        time_step_s=time_step_s,
        metadata=metadata or {"profile": "unit-test"},
    )


def test_generate_trajectory_records_exact_authoritative_transitions() -> None:
    from thermotwin.data.generate import generate_trajectory

    queried_flows: list[float] = []

    def coolant_flow(time_s: float) -> float:
        queried_flows.append(time_s)
        return 0.02 + time_s / 100.0

    scenario = make_scenario(coolant_flow_profile=coolant_flow)

    records = generate_trajectory(scenario)

    assert len(records) == 3
    assert queried_flows == [0.0, 0.5, 1.0]
    assert all(type(record) is TransitionRecord for record in records)
    for index, record in enumerate(records):
        assert record.trajectory_id == "run_01"
        assert record.step_index == index
        assert record.time_step_s == 0.5
        assert record.state.time_s == index * 0.5
        assert record.next_state.time_s == (index + 1) * 0.5
        assert record.state.current_a == 4.0 + index * 0.5
        assert record.next_state.current_a == 4.0 + (index + 1) * 0.5
        assert record.action.coolant_mass_flow_kg_s == pytest.approx(
            0.02 + index * 0.5 / 100.0
        )
        assert (
            record.next_state.coolant_mass_flow_kg_s
            == record.action.coolant_mass_flow_kg_s
        )
        assert record.state.soc_fraction == pytest.approx(
            0.8 - sum((4.0 + step * 0.5) * 0.5 for step in range(index)) / 180_000
        )
        assert dict(record.metadata) == {"profile": "unit-test"}
    assert records[0].state.time_s == 0.0
    assert records[0].state.current_a == 4.0
    assert records[0].state.soc_fraction == 0.8


def test_generate_trajectory_preserves_state_continuity_and_metadata_copy() -> None:
    from thermotwin.data.generate import generate_trajectory

    source_metadata: dict[str, DatasetMetadataValue] = {
        "scenario": "warm",
        "seed": 5,
    }
    scenario = make_scenario(metadata=source_metadata)

    records = generate_trajectory(scenario)
    source_metadata["scenario"] = "changed"

    assert all(
        previous.next_state is following.state
        for previous, following in zip(records, records[1:], strict=False)
    )
    assert all(
        dict(record.metadata) == {"scenario": "warm", "seed": 5}
        for record in records
    )
    assert all(record.metadata is not scenario.metadata for record in records)
    with pytest.raises(TypeError):
        records[0].metadata["scenario"] = "changed"  # type: ignore[index]


def test_generate_trajectory_repeats_deterministically() -> None:
    from thermotwin.data.generate import generate_trajectory

    scenario = make_scenario()

    first = generate_trajectory(scenario)
    second = generate_trajectory(scenario)

    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.trajectory_id == right.trajectory_id
        assert left.step_index == right.step_index
        assert left.state.time_s == right.state.time_s
        assert left.state.current_a == right.state.current_a
        assert left.state.soc_fraction == right.state.soc_fraction
        assert left.action == right.action
        np.testing.assert_array_equal(
            left.state.cell_temperatures_k, right.state.cell_temperatures_k
        )
        np.testing.assert_array_equal(
            left.next_state.cell_temperatures_k, right.next_state.cell_temperatures_k
        )


def test_generate_dataset_concatenates_scenarios_in_input_order() -> None:
    from thermotwin.data.generate import generate_dataset

    scenarios = (
        make_scenario("second", duration_s=1.0),
        make_scenario("first", duration_s=0.5),
        make_scenario("third", duration_s=1.5),
    )

    records = generate_dataset(scenarios)

    assert [record.trajectory_id for record in records] == [
        "second",
        "second",
        "first",
        "third",
        "third",
        "third",
    ]
    assert [record.step_index for record in records] == [0, 1, 0, 0, 1, 2]
    assert records[0].next_state is records[1].state
    assert records[1].next_state is not records[2].state
    assert records[3].state.time_s == 0.0


def test_generate_trajectory_requires_exact_dataset_scenario_type() -> None:
    from thermotwin.data.generate import generate_trajectory

    class DerivedScenario(DatasetScenario):
        pass

    scenario = make_scenario()
    derived = DerivedScenario(
        trajectory_id="derived",
        environment_config=scenario.environment_config,
        current_profile=scenario.current_profile,
        coolant_flow_profile=scenario.coolant_flow_profile,
        duration_s=scenario.duration_s,
        time_step_s=scenario.time_step_s,
    )
    with pytest.raises(TypeError, match="DatasetScenario"):
        generate_trajectory(derived)


@pytest.mark.parametrize(
    "scenarios",
    [(), [], "not-scenarios", [make_scenario(), object()]],
)
def test_generate_dataset_rejects_invalid_or_empty_scenarios_before_environment(
    scenarios: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thermotwin.data.generate as generate

    resets: list[EnvironmentConfig] = []
    class TrackedEnvironment(BatteryEnvironment):
        def reset(self, config: EnvironmentConfig) -> Any:
            resets.append(config)
            return super().reset(config)

    monkeypatch.setattr(generate, "BatteryEnvironment", TrackedEnvironment)

    with pytest.raises((TypeError, ValueError)):
        generate.generate_dataset(scenarios)

    assert resets == []


def test_generate_dataset_rejects_subclass_scenario_before_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thermotwin.data.generate as generate

    class DerivedScenario(DatasetScenario):
        pass

    scenario = make_scenario()
    derived = DerivedScenario(
        trajectory_id="derived",
        environment_config=scenario.environment_config,
        current_profile=scenario.current_profile,
        coolant_flow_profile=scenario.coolant_flow_profile,
        duration_s=scenario.duration_s,
        time_step_s=scenario.time_step_s,
    )
    reset_called = False

    class TrackedEnvironment(BatteryEnvironment):
        def reset(self, config: EnvironmentConfig) -> Any:
            nonlocal reset_called
            reset_called = True
            return super().reset(config)

    monkeypatch.setattr(generate, "BatteryEnvironment", TrackedEnvironment)

    with pytest.raises(TypeError, match="DatasetScenario"):
        generate.generate_dataset((scenario, derived))
    assert not reset_called


def test_generate_dataset_rejects_duplicate_ids_before_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import thermotwin.data.generate as generate

    reset_called = False

    class TrackedEnvironment(BatteryEnvironment):
        def reset(self, config: EnvironmentConfig) -> Any:
            nonlocal reset_called
            reset_called = True
            return super().reset(config)

    monkeypatch.setattr(generate, "BatteryEnvironment", TrackedEnvironment)

    with pytest.raises(ValueError, match="unique"):
        generate.generate_dataset(
            (make_scenario("duplicate"), make_scenario("duplicate"))
        )
    assert not reset_called


@pytest.mark.parametrize("flow", [-0.01, 0.0, 0.2, float("nan"), float("inf")])
def test_generate_trajectory_rejects_illegal_coolant_flow(flow: float) -> None:
    from thermotwin.data.generate import generate_trajectory

    scenario = make_scenario(coolant_flow_profile=lambda _time_s: flow)

    with pytest.raises(ValueError):
        generate_trajectory(scenario)
