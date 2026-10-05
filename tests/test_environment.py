from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.physics.environment import BatteryEnvironment


def make_config(**simulation_overrides: Any) -> EnvironmentConfig:
    temperature = 300.0
    simulation_values: dict[str, Any] = {
        "default_time_step_s": 1.0,
        "initial_soc_fraction": 0.5,
        "initial_cell_temperature_k": temperature,
        "ambient_temperature_k": temperature,
        "coolant_inlet_temperature_k": temperature,
        "initial_current_a": 0.0,
    }
    simulation_values.update(simulation_overrides)
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=2,
            cell_mass_kg=1.0,
            cell_specific_heat_capacity_j_kg_k=10.0,
            nominal_cell_resistance_ohm=0.5,
            cell_capacity_ah=10.0,
            neighbor_thermal_resistance_k_w=2.0,
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
    )


def test_reset_creates_configured_initial_state_and_minimum_flow() -> None:
    config = make_config(initial_soc_fraction=0.7)
    environment = BatteryEnvironment(lambda time_s: 2.0 + time_s)

    state = environment.reset(config)

    assert environment.state is state
    assert isinstance(state, BatteryState)
    assert state.time_s == 0.0
    assert state.current_a == 2.0
    assert state.soc_fraction == 0.7
    assert state.coolant_mass_flow_kg_s == 0.0
    np.testing.assert_array_equal(state.cell_temperatures_k, [300.0, 300.0])
    np.testing.assert_array_equal(state.cell_resistances_ohm, [0.5, 0.5])


def test_reset_rejects_per_cell_temperature_count_that_does_not_match_pack() -> None:
    environment = BatteryEnvironment(lambda _: 0.0)

    with pytest.raises(ValueError, match="initial_cell_temperatures_k"):
        environment.reset(make_config(initial_cell_temperatures_k=(300.0,)))


def test_zero_current_and_uniform_temperatures_remain_at_equilibrium() -> None:
    environment = BatteryEnvironment(lambda _: 0.0)
    environment.reset(make_config())

    state = environment.step(CoolingAction(0.4), 10.0)

    np.testing.assert_array_equal(state.cell_temperatures_k, [300.0, 300.0])


def test_step_matches_analytical_joule_heating_and_advances_profile() -> None:
    environment = BatteryEnvironment(lambda time_s: 10.0 + time_s)
    environment.reset(make_config())

    state = environment.step(CoolingAction(0.0), 2.0)

    # Interval heat is 10**2 * 0.5 = 50 W; C_th = 10 J/K.
    np.testing.assert_allclose(state.cell_temperatures_k, [310.0, 310.0])
    assert state.time_s == 2.0
    assert state.current_a == 12.0
    assert state.soc_fraction == pytest.approx(0.5 - 10.0 * 2.0 / (3600 * 10.0))


def test_higher_coolant_flow_lowers_next_temperature() -> None:
    low_environment = BatteryEnvironment(lambda _: 4.0)
    high_environment = BatteryEnvironment(lambda _: 4.0)
    config = make_config(initial_cell_temperature_k=320.0)
    low_environment.reset(config)
    high_environment.reset(config)

    low = low_environment.step(CoolingAction(0.1), 1.0)
    high = high_environment.step(CoolingAction(0.9), 1.0)

    assert np.all(high.cell_temperatures_k < low.cell_temperatures_k)


def test_environment_preserves_uniform_pack_without_net_conduction() -> None:
    config = make_config()
    environment = BatteryEnvironment(lambda _: 0.0)
    environment.reset(config)

    state = environment.step(CoolingAction(0.0), 1.0)

    np.testing.assert_array_equal(state.cell_temperatures_k, [300.0, 300.0])


def test_environment_conduction_moves_heat_hot_to_cold_conservatively() -> None:
    config = make_config(initial_cell_temperatures_k=(310.0, 290.0))
    environment = BatteryEnvironment(lambda _: 0.0)
    initial = environment.reset(config)

    state = environment.step(CoolingAction(0.0), 1.0)

    assert state.cell_temperatures_k[0] < initial.cell_temperatures_k[0]
    assert state.cell_temperatures_k[1] > initial.cell_temperatures_k[1]
    assert np.sum(state.cell_temperatures_k) == pytest.approx(
        np.sum(initial.cell_temperatures_k), abs=1e-9
    )


def test_signed_current_updates_soc_and_applied_flow_is_recorded() -> None:
    environment = BatteryEnvironment(lambda _: -10.0)
    environment.reset(make_config())

    state = environment.step(CoolingAction(0.6), 360.0)

    assert state.soc_fraction == pytest.approx(0.6)
    assert state.coolant_mass_flow_kg_s == 0.6


def test_state_is_immutable_and_environment_is_deterministic() -> None:
    config = make_config()
    first = BatteryEnvironment(lambda time_s: time_s + 1.0)
    second = BatteryEnvironment(lambda time_s: time_s + 1.0)
    first.reset(config)
    second.reset(config)

    state_a = first.step(CoolingAction(0.3), 0.25)
    state_b = second.step(CoolingAction(0.3), 0.25)

    assert state_a.time_s == state_b.time_s
    assert state_a.current_a == state_b.current_a
    np.testing.assert_array_equal(
        state_a.cell_temperatures_k,
        state_b.cell_temperatures_k,
    )
    with pytest.raises(ValueError):
        state_a.cell_temperatures_k[0] = 1.0
    with pytest.raises(AttributeError):
        state_a.time_s = 9.0  # type: ignore[misc]


def test_step_requires_reset_and_constructor_requires_callable() -> None:
    with pytest.raises(ValueError):
        BatteryEnvironment(None)  # type: ignore[arg-type]
    environment = BatteryEnvironment(lambda _: 0.0)
    with pytest.raises(RuntimeError):
        _ = environment.state
    with pytest.raises(RuntimeError):
        environment.step(CoolingAction(0.0), 1.0)


def test_current_profile_values_are_validated_by_battery_state() -> None:
    bad_at_reset = BatteryEnvironment(lambda _: float("nan"))
    with pytest.raises(ValueError, match="current_a"):
        bad_at_reset.reset(make_config())

    def invalid_after_reset(time_s: float) -> float:
        return 0.0 if time_s == 0.0 else float("nan")

    bad_after_step = BatteryEnvironment(invalid_after_reset)
    initial = bad_after_step.reset(make_config())
    with pytest.raises(ValueError, match="current_a"):
        bad_after_step.step(CoolingAction(0.0), 1.0)
    assert bad_after_step.state is initial


@pytest.mark.parametrize("dt", [0.0, -1.0, True, float("nan"), float("inf")])
def test_step_rejects_invalid_time_step(dt: Any) -> None:
    environment = BatteryEnvironment(lambda _: 0.0)
    environment.reset(make_config())

    with pytest.raises(ValueError):
        environment.step(CoolingAction(0.0), dt)


def test_step_rejects_wrong_action_type_and_out_of_range_flow() -> None:
    environment = BatteryEnvironment(lambda _: 0.0)
    environment.reset(make_config())

    with pytest.raises(TypeError):
        environment.step(None, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        environment.step(CoolingAction(1.1), 1.0)
