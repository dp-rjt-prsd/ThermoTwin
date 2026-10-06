from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from thermotwin.core import (
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.physics.environment import BatteryEnvironment


def make_config(
    *, fault: ResistanceFaultConfig | None = None, **simulation_overrides: Any
) -> EnvironmentConfig:
    simulation_values: dict[str, Any] = {
        "default_time_step_s": 1.0,
        "initial_soc_fraction": 0.5,
        "initial_cell_temperature_k": 300.0,
        "ambient_temperature_k": 300.0,
        "coolant_inlet_temperature_k": 300.0,
        "initial_current_a": 0.0,
    }
    simulation_values.update(simulation_overrides)
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=3,
            cell_mass_kg=1.0,
            cell_specific_heat_capacity_j_kg_k=10.0,
            nominal_cell_resistance_ohm=0.5,
            cell_capacity_ah=10.0,
            neighbor_thermal_resistance_k_w=1e12,
            ambient_thermal_resistance_k_w=1e12,
            cell_cooling_surface_area_m2=0.1,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.0,
            maximum_coolant_mass_flow_kg_s=1.0,
            base_heat_transfer_coefficient_w_m2_k=1.0,
            flow_heat_transfer_gain=10.0,
            flow_exponent=1.0,
        ),
        simulation=SimulationConfig(**simulation_values),
        resistance_fault=fault,
    )


def test_fault_starting_at_zero_is_visible_on_reset() -> None:
    environment = BatteryEnvironment(lambda _: 2.0)

    state = environment.reset(make_config(fault=ResistanceFaultConfig(1, 0.0, 3.0)))

    np.testing.assert_array_equal(state.cell_resistances_ohm, [0.5, 1.5, 0.5])


def test_fault_activates_after_baseline_interval_arrives_at_boundary() -> None:
    environment = BatteryEnvironment(lambda _: 2.0)
    reference = BatteryEnvironment(lambda _: 2.0)
    environment.reset(make_config(fault=ResistanceFaultConfig(1, 2.0, 3.0)))
    reference.reset(make_config())

    before_boundary = environment.step(CoolingAction(0.0), 1.0)
    reference.step(CoolingAction(0.0), 1.0)
    at_boundary = environment.step(CoolingAction(0.0), 1.0)
    reference_at_boundary = reference.step(CoolingAction(0.0), 1.0)

    np.testing.assert_array_equal(before_boundary.cell_resistances_ohm, [0.5] * 3)
    # The [1, 2] interval used baseline resistance, and the state at t=2
    # advertises the resistance that will govern the next interval.
    np.testing.assert_array_equal(at_boundary.cell_resistances_ohm, [0.5, 1.5, 0.5])
    np.testing.assert_array_equal(
        at_boundary.cell_temperatures_k, reference_at_boundary.cell_temperatures_k
    )


def test_faulted_interval_multiplies_only_selected_cell_once_and_increases_heat() -> (
    None
):
    fault = ResistanceFaultConfig(1, 1.0, 4.0)
    environment = BatteryEnvironment(lambda _: 2.0)
    reference = BatteryEnvironment(lambda _: 2.0)
    config = make_config(
        fault=fault,
        initial_cell_resistances_ohm=(0.25, 0.75, 1.25),
    )
    baseline_config = make_config(initial_cell_resistances_ohm=(0.25, 0.75, 1.25))
    environment.reset(config)
    reference.reset(baseline_config)

    environment.step(CoolingAction(0.0), 1.0)
    reference.step(CoolingAction(0.0), 1.0)
    once = environment.step(CoolingAction(0.0), 1.0)
    baseline = reference.step(CoolingAction(0.0), 1.0)

    np.testing.assert_array_equal(once.cell_resistances_ohm, [0.25, 3.0, 1.25])
    assert baseline.cell_resistances_ohm[1] == 0.75
    temperature_increase = once.cell_temperatures_k - baseline.cell_temperatures_k
    np.testing.assert_allclose(
        temperature_increase,
        [0.0, 0.9, 0.0],
        atol=1e-12,
        rtol=0.0,
    )

    twice = environment.step(CoolingAction(0.0), 1.0)
    np.testing.assert_array_equal(twice.cell_resistances_ohm, [0.25, 3.0, 1.25])


def test_fault_rejects_cell_index_outside_pack_on_reset() -> None:
    environment = BatteryEnvironment(lambda _: 0.0)

    with pytest.raises(ValueError, match="cell_index"):
        environment.reset(make_config(fault=ResistanceFaultConfig(3, 0.0, 2.0)))


def test_reset_restores_fault_schedule_deterministically() -> None:
    environment = BatteryEnvironment(lambda _: 2.0)
    config = make_config(fault=ResistanceFaultConfig(2, 1.0, 2.0))

    first_initial = environment.reset(config)
    first_before = environment.step(CoolingAction(0.0), 1.0)
    first_active = environment.step(CoolingAction(0.0), 1.0)
    second_initial = environment.reset(config)
    second_before = environment.step(CoolingAction(0.0), 1.0)
    second_active = environment.step(CoolingAction(0.0), 1.0)

    for first, second in (
        (first_initial, second_initial),
        (first_before, second_before),
        (first_active, second_active),
    ):
        np.testing.assert_array_equal(
            first.cell_resistances_ohm, second.cell_resistances_ohm
        )
        np.testing.assert_array_equal(
            first.cell_temperatures_k, second.cell_temperatures_k
        )


def test_fault_free_trajectory_unchanged() -> None:
    environment = BatteryEnvironment(lambda time_s: 2.0 + time_s)
    reference = BatteryEnvironment(lambda time_s: 2.0 + time_s)
    resistances = (0.25, 0.75, 1.25)
    environment.reset(
        make_config(
            fault=ResistanceFaultConfig(1, 100.0, 3.0),
            initial_cell_resistances_ohm=resistances,
        )
    )
    reference.reset(make_config(initial_cell_resistances_ohm=resistances))

    for dt in (0.25, 1.0, 3.0):
        state = environment.step(CoolingAction(0.2), dt)
        reference_state = reference.step(CoolingAction(0.2), dt)
        np.testing.assert_array_equal(
            state.cell_temperatures_k, reference_state.cell_temperatures_k
        )
        np.testing.assert_array_equal(
            state.cell_resistances_ohm, reference_state.cell_resistances_ohm
        )


def test_fault_free_step_preserves_resistances_on_current_persisted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = BatteryEnvironment(lambda _: 2.0)
    reference = BatteryEnvironment(lambda _: 2.0)
    environment.reset(make_config())
    persisted_resistances = np.asarray([0.02, 0.01, 0.03])
    persisted_state = replace(
        environment.state,
        cell_resistances_ohm=persisted_resistances,
    )
    monkeypatch.setattr(environment, "_state", persisted_state)
    reference.reset(
        make_config(initial_cell_resistances_ohm=tuple(persisted_resistances))
    )

    resumed = environment.step(CoolingAction(0.0), 1.0)
    expected = reference.step(CoolingAction(0.0), 1.0)

    np.testing.assert_array_equal(resumed.cell_resistances_ohm, persisted_resistances)
    np.testing.assert_array_equal(
        resumed.cell_temperatures_k, expected.cell_temperatures_k
    )
