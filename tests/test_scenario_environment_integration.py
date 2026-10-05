from __future__ import annotations

import pytest

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.physics import BatteryEnvironment
from thermotwin.scenarios import constant_current_profile, fast_charging_profile


def make_config() -> EnvironmentConfig:
    temperature_k = 300.0
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=1,
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
            maximum_coolant_mass_flow_kg_s=0.0,
            base_heat_transfer_coefficient_w_m2_k=1.0,
            flow_heat_transfer_gain=1.0,
            flow_exponent=1.0,
        ),
        simulation=SimulationConfig(
            default_time_step_s=1.0,
            initial_soc_fraction=0.5,
            initial_cell_temperature_k=temperature_k,
            ambient_temperature_k=temperature_k,
            coolant_inlet_temperature_k=temperature_k,
            initial_current_a=0.0,
        ),
    )


def test_constant_discharge_profile_drives_environment_until_end() -> None:
    environment = BatteryEnvironment(constant_current_profile(10.0, 2.0))
    initial = environment.reset(make_config())

    during_profile = environment.step(CoolingAction(0.0), 1.0)
    at_profile_end = environment.step(CoolingAction(0.0), 1.0)

    assert isinstance(initial, BatteryState)
    assert during_profile.time_s == 1.0
    assert during_profile.current_a == 10.0
    assert during_profile.cell_temperatures_k[0] > initial.cell_temperatures_k[0]
    assert during_profile.soc_fraction < initial.soc_fraction
    assert at_profile_end.time_s == 2.0
    assert at_profile_end.current_a == 0.0
    assert at_profile_end.soc_fraction < during_profile.soc_fraction


def test_fast_charging_profile_raises_soc_and_state_remains_immutable() -> None:
    environment = BatteryEnvironment(fast_charging_profile(12.0, 10.0))
    initial = environment.reset(make_config())

    charging = environment.step(CoolingAction(0.0), 2.0)

    assert charging.time_s == 2.0
    assert charging.current_a == pytest.approx(-12.0)
    assert charging.soc_fraction > initial.soc_fraction
    assert isinstance(charging, BatteryState)
    with pytest.raises(ValueError):
        charging.cell_temperatures_k[0] = 301.0
