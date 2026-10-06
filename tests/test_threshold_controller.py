"""Behavioral tests for the deterministic threshold controller."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from thermotwin.control.base import Controller
from thermotwin.control.threshold import ThresholdController
from thermotwin.core import BatteryState, CoolingAction, CoolingConfig


def _cooling_config() -> CoolingConfig:
    return CoolingConfig(
        minimum_coolant_mass_flow_kg_s=0.0,
        maximum_coolant_mass_flow_kg_s=0.1,
        base_heat_transfer_coefficient_w_m2_k=100.0,
        flow_heat_transfer_gain=50.0,
        flow_exponent=0.8,
    )


def _state(*temperatures_k: float) -> BatteryState:
    temperatures = np.asarray(temperatures_k, dtype=np.float64)
    return BatteryState(
        time_s=10.0,
        current_a=20.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=293.15,
        coolant_mass_flow_kg_s=0.0,
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=np.full(temperatures.shape, 0.01),
    )


def _controller(
    boundaries: Sequence[float] = (303.15, 308.15, 313.15),
    intermediate_flows: Sequence[float] = (0.025, 0.06),
) -> ThresholdController:
    return ThresholdController(
        cooling_config=_cooling_config(),
        temperature_boundaries_k=boundaries,
        coolant_mass_flows_kg_s=intermediate_flows,
    )


@pytest.mark.parametrize(
    ("temperature_k", "expected_flow"),
    [
        (303.15, 0.025),
        (308.15, 0.06),
        (313.15, 0.1),
    ],
)
def test_exact_temperature_boundaries_select_the_next_stage(
    temperature_k: float, expected_flow: float
) -> None:
    action = _controller().act(_state(temperature_k))

    assert action == CoolingAction(expected_flow)


@pytest.mark.parametrize(
    ("temperature_k", "expected_flow"),
    [
        (math.nextafter(303.15, -math.inf), 0.0),
        (math.nextafter(303.15, math.inf), 0.025),
        (math.nextafter(308.15, -math.inf), 0.025),
        (math.nextafter(308.15, math.inf), 0.06),
        (math.nextafter(313.15, -math.inf), 0.06),
        (math.nextafter(313.15, math.inf), 0.1),
    ],
)
def test_values_immediately_around_boundaries_select_expected_stage(
    temperature_k: float, expected_flow: float
) -> None:
    assert _controller().act(_state(temperature_k)).coolant_mass_flow_kg_s == (
        expected_flow
    )


def test_lowest_stage_applies_below_first_boundary_and_uses_maximum_cell() -> None:
    action = _controller().act(_state(299.0, 304.0, 302.0))

    assert action.coolant_mass_flow_kg_s == 0.025


def test_arbitrary_boundaries_define_the_same_four_stage_behavior() -> None:
    controller = _controller(boundaries=(280.0, 290.0, 300.0))

    assert controller.act(_state(279.9)).coolant_mass_flow_kg_s == 0.0
    assert controller.act(_state(280.0)).coolant_mass_flow_kg_s == 0.025
    assert controller.act(_state(290.0)).coolant_mass_flow_kg_s == 0.06
    assert controller.act(_state(300.0)).coolant_mass_flow_kg_s == 0.1


@pytest.mark.parametrize(
    "boundaries",
    [
        (),
        (300.0, 310.0),
        (300.0, 310.0, 320.0, 330.0),
        (300.0, 300.0, 320.0),
        (310.0, 300.0, 320.0),
        (300.0, 310.0, 0.0),
        (300.0, 310.0, math.inf),
        (300.0, 310.0, math.nan),
        (True, 310.0, 320.0),
    ],
)
def test_invalid_temperature_boundaries_are_rejected(
    boundaries: Sequence[float],
) -> None:
    with pytest.raises(ValueError):
        _controller(boundaries=boundaries)


@pytest.mark.parametrize(
    "intermediate_flows",
    [
        (),
        (0.02,),
        (0.01, 0.02, 0.03),
        (-0.01, 0.02),
        (0.02, 0.01),
        (0.02, math.inf),
        (0.02, math.nan),
        (0.02, 0.11),
        (0.02, True),
    ],
)
def test_invalid_intermediate_flows_are_rejected(
    intermediate_flows: Sequence[float],
) -> None:
    with pytest.raises(ValueError):
        _controller(intermediate_flows=intermediate_flows)


def test_intermediate_flows_may_equal_each_other_and_configured_bounds() -> None:
    controller = ThresholdController(
        cooling_config=_cooling_config(),
        temperature_boundaries_k=(300.0, 310.0, 320.0),
        coolant_mass_flows_kg_s=(0.0, 0.0),
    )

    assert controller.act(_state(300.0)).coolant_mass_flow_kg_s == 0.0
    assert controller.act(_state(310.0)).coolant_mass_flow_kg_s == 0.0


def test_intermediate_flows_must_respect_nonzero_configured_minimum() -> None:
    config = CoolingConfig(
        minimum_coolant_mass_flow_kg_s=0.02,
        maximum_coolant_mass_flow_kg_s=0.1,
        base_heat_transfer_coefficient_w_m2_k=100.0,
        flow_heat_transfer_gain=50.0,
        flow_exponent=0.8,
    )

    with pytest.raises(ValueError):
        ThresholdController(
            cooling_config=config,
            temperature_boundaries_k=(300.0, 310.0, 320.0),
            coolant_mass_flows_kg_s=(0.01, 0.05),
        )


def test_rejects_non_cooling_config() -> None:
    with pytest.raises(ValueError):
        ThresholdController(
            cooling_config=object(),  # type: ignore[arg-type]
            temperature_boundaries_k=(303.15, 308.15, 313.15),
            coolant_mass_flows_kg_s=(0.025, 0.06),
        )


def test_reset_is_noop_and_controller_satisfies_protocol() -> None:
    controller = _controller()

    assert isinstance(controller, Controller)
    controller.reset()
    assert controller.act(_state(303.15)) == CoolingAction(0.025)


def test_actions_are_legal_and_act_does_not_mutate_state() -> None:
    controller = _controller()
    state = _state(302.0, 307.0, 312.0)
    original_temperatures = state.cell_temperatures_k.copy()
    original_resistances = state.cell_resistances_ohm.copy()

    for temperature_k in (302.0, 304.0, 309.0, 314.0):
        action = controller.act(_state(temperature_k))
        assert _cooling_config().minimum_coolant_mass_flow_kg_s <= (
            action.coolant_mass_flow_kg_s
        ) <= _cooling_config().maximum_coolant_mass_flow_kg_s

    controller.act(state)
    np.testing.assert_array_equal(state.cell_temperatures_k, original_temperatures)
    np.testing.assert_array_equal(state.cell_resistances_ohm, original_resistances)
