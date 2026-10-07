from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.control import ThresholdController
from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.experiments.simulation_service import SimulationSession
from thermotwin.scenarios import constant_current_profile
from thermotwin.visualization.live_contracts import SimulationRequest


def make_environment_config(**overrides: Any) -> EnvironmentConfig:
    simulation_values: dict[str, Any] = {
        "default_time_step_s": 1.0,
        "initial_soc_fraction": 0.7,
        "initial_cell_temperature_k": 300.0,
        "ambient_temperature_k": 300.0,
        "coolant_inlet_temperature_k": 295.0,
        "initial_current_a": 0.0,
    }
    simulation_values.update(overrides)
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


def make_session(duration_s: float = 3.0) -> SimulationSession:
    config = make_environment_config()
    request = SimulationRequest(
        scenario_name="service-test",
        environment_config=config,
        current_profile=constant_current_profile(2.0, duration_s),
        controller_name="threshold",
        model_name=None,
        duration_s=duration_s,
        time_step_s=1.0,
        seed=1729,
    )
    controller = ThresholdController(
        config.cooling,
        temperature_boundaries_k=(301.0, 302.0, 303.0),
        coolant_mass_flows_kg_s=(0.2, 0.4),
    )
    return SimulationSession(request, controller)


def test_session_requires_start_before_step_and_supports_reset() -> None:
    session = make_session()

    with pytest.raises(RuntimeError, match="started before stepping"):
        session.step()

    initial_state = session.start()
    assert initial_state.time_s == 0.0
    assert session.state is initial_state
    assert session.snapshots == ()

    session.step()
    assert len(session.snapshots) == 1
    session.reset()  # type: ignore[unreachable]
    assert session.state is None
    assert session.snapshots == ()


def test_session_runs_to_requested_duration_and_rejects_extra_steps() -> None:
    session = make_session()

    snapshots = session.run_to_completion()

    assert session.complete
    assert len(snapshots) == 3
    assert [snapshot.state.time_s for snapshot in snapshots] == [1.0, 2.0, 3.0]
    assert all(
        snapshot.state.cell_temperatures_k.shape == (2,) for snapshot in snapshots
    )
    with pytest.raises(RuntimeError, match="already complete"):
        session.step()


def test_session_replays_identically_for_same_request_and_controller() -> None:
    first = make_session().run_to_completion()
    second = make_session().run_to_completion()

    np.testing.assert_array_equal(
        [snapshot.control_action.coolant_mass_flow_kg_s for snapshot in first],
        [snapshot.control_action.coolant_mass_flow_kg_s for snapshot in second],
    )
    for first_snapshot, second_snapshot in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            first_snapshot.state.cell_temperatures_k,
            second_snapshot.state.cell_temperatures_k,
        )
