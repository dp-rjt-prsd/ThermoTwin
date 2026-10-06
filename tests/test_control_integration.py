"""Closed-loop integration checks for the baseline controllers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import thermotwin.control as control_package
from thermotwin.control import (
    Controller,
    PIDController,
    ThresholdController,
    pump_energy_j,
    pump_power_w,
)
from thermotwin.core import (
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.physics import BatteryEnvironment
from thermotwin.scenarios import constant_current_profile


def make_environment_config() -> EnvironmentConfig:
    cooling = CoolingConfig(
        minimum_coolant_mass_flow_kg_s=0.01,
        maximum_coolant_mass_flow_kg_s=0.1,
        base_heat_transfer_coefficient_w_m2_k=25.0,
        flow_heat_transfer_gain=3.0,
        flow_exponent=1.0,
    )
    return EnvironmentConfig(
        pack=PackConfig(
            rows=3,
            columns=4,
            cell_mass_kg=0.9,
            cell_specific_heat_capacity_j_kg_k=950.0,
            nominal_cell_resistance_ohm=0.02,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.2,
            ambient_thermal_resistance_k_w=20.0,
            cell_cooling_surface_area_m2=0.035,
        ),
        cooling=cooling,
        simulation=SimulationConfig(
            default_time_step_s=1.0,
            initial_soc_fraction=0.7,
            initial_cell_temperature_k=312.0,
            ambient_temperature_k=300.0,
            coolant_inlet_temperature_k=295.0,
            initial_current_a=0.0,
            initial_cell_temperatures_k=(
                316.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
                314.0,
            ),
        ),
    )


def make_threshold_controller(cooling: CoolingConfig) -> Controller:
    return ThresholdController(
        cooling,
        temperature_boundaries_k=(303.15, 308.15, 313.15),
        coolant_mass_flows_kg_s=(0.04, 0.07),
    )


def make_pid_controller(cooling: CoolingConfig) -> Controller:
    return PIDController(
        cooling,
        target_temperature_k=308.15,
        kp_kg_s_k=0.01,
        ki_kg_s_k_s=0.0001,
        kd_kg_s_k_per_s=0.0,
    )


def run_closed_loop(
    controller: Controller,
    config: EnvironmentConfig,
    current_profile: Callable[[float], float],
) -> tuple[list[float], list[np.ndarray], list[float]]:
    environment = BatteryEnvironment(current_profile)
    current_state = environment.reset(config)
    controller.reset()
    flows: list[float] = []
    temperature_history: list[np.ndarray] = [
        current_state.cell_temperatures_k.copy()
    ]
    times = [current_state.time_s]

    for step_index in range(1, 13):
        action = controller.act(current_state)
        assert isinstance(action, CoolingAction)
        assert (
            config.cooling.minimum_coolant_mass_flow_kg_s
            <= action.coolant_mass_flow_kg_s
            <= config.cooling.maximum_coolant_mass_flow_kg_s
        )
        current_state = environment.step(
            action, config.simulation.default_time_step_s
        )
        assert current_state.time_s == (
            step_index * config.simulation.default_time_step_s
        )
        flows.append(action.coolant_mass_flow_kg_s)
        temperature_history.append(current_state.cell_temperatures_k.copy())
        times.append(current_state.time_s)

    return flows, temperature_history, times


def test_control_package_exports_public_api_and_controllers_satisfy_protocol() -> None:
    assert control_package.__all__ == [
        "Controller",
        "ThresholdController",
        "PIDController",
        "pump_power_w",
        "pump_energy_j",
    ]
    assert control_package.Controller is Controller
    assert control_package.ThresholdController is ThresholdController
    assert control_package.PIDController is PIDController
    assert control_package.pump_power_w is pump_power_w
    assert control_package.pump_energy_j is pump_energy_j

    cooling = make_environment_config().cooling
    assert isinstance(make_threshold_controller(cooling), Controller)
    assert isinstance(make_pid_controller(cooling), Controller)


def test_threshold_controller_runs_deterministic_closed_loop() -> None:
    config = make_environment_config()
    profile = constant_current_profile(35.0, total_duration_s=12.0)
    controller = make_threshold_controller(config.cooling)

    first_run = run_closed_loop(controller, config, profile)
    controller.reset()
    repeated_run = run_closed_loop(controller, config, profile)

    assert max(first_run[0]) > config.cooling.minimum_coolant_mass_flow_kg_s
    assert repeated_run[0] == first_run[0]
    assert repeated_run[2] == first_run[2]
    np.testing.assert_array_equal(repeated_run[1], first_run[1])


def test_pid_controller_runs_deterministic_closed_loop() -> None:
    config = make_environment_config()
    profile = constant_current_profile(35.0, total_duration_s=12.0)
    controller = make_pid_controller(config.cooling)

    first_run = run_closed_loop(controller, config, profile)
    controller.reset()
    repeated_run = run_closed_loop(controller, config, profile)

    assert max(first_run[0]) > config.cooling.minimum_coolant_mass_flow_kg_s
    assert repeated_run[0] == first_run[0]
    assert repeated_run[2] == first_run[2]
    np.testing.assert_array_equal(repeated_run[1], first_run[1])
