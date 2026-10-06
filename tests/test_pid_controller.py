"""Behavioral tests for the resettable PID coolant controller."""

from __future__ import annotations

import numpy as np
import pytest

from thermotwin.control.base import Controller
from thermotwin.control.pid import PIDController
from thermotwin.core import BatteryState, CoolingAction, CoolingConfig


def _cooling_config() -> CoolingConfig:
    return CoolingConfig(
        minimum_coolant_mass_flow_kg_s=0.1,
        maximum_coolant_mass_flow_kg_s=0.9,
        base_heat_transfer_coefficient_w_m2_k=100.0,
        flow_heat_transfer_gain=10.0,
        flow_exponent=1.0,
    )


def _state(time_s: float, maximum_temperature_k: float) -> BatteryState:
    temperatures = np.array([maximum_temperature_k - 1.0, maximum_temperature_k])
    return BatteryState(
        time_s=time_s,
        current_a=10.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=293.15,
        coolant_mass_flow_kg_s=0.1,
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=np.array([0.01, 0.01]),
    )


def _controller(*, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0) -> PIDController:
    return PIDController(_cooling_config(), 300.0, kp, ki, kd)


def test_first_call_uses_proportional_control_without_integral_or_derivative() -> None:
    controller = _controller(kp=0.1, ki=1.0, kd=1.0)

    action = controller.act(_state(5.0, 302.0))

    assert action.coolant_mass_flow_kg_s == pytest.approx(0.3)


def test_integral_gain_increases_flow_using_observed_elapsed_seconds() -> None:
    controller = _controller(ki=0.1)
    controller.act(_state(0.0, 302.0))

    action = controller.act(_state(2.0, 302.0))

    assert action.coolant_mass_flow_kg_s == pytest.approx(0.5)


def test_derivative_gain_uses_change_in_maximum_temperature_per_second() -> None:
    controller = _controller(kd=0.1)
    controller.act(_state(1.0, 301.0))

    action = controller.act(_state(3.0, 303.0))

    assert action.coolant_mass_flow_kg_s == pytest.approx(0.2)


def test_output_saturates_at_configured_minimum_and_maximum() -> None:
    controller = _controller(kp=1.0)

    too_cold = controller.act(_state(0.0, 299.0))
    too_hot = controller.act(_state(1.0, 400.0))

    assert too_cold.coolant_mass_flow_kg_s == 0.1
    assert too_hot.coolant_mass_flow_kg_s == 0.9


def test_nonincreasing_time_is_rejected_without_corrupting_controller_state() -> None:
    controller = _controller(ki=0.1, kd=0.1)
    controller.act(_state(1.0, 301.0))
    controller.act(_state(3.0, 302.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        controller.act(_state(2.0, 305.0))

    after_rejection = controller.act(_state(5.0, 303.0))
    reference = _controller(ki=0.1, kd=0.1)
    reference.act(_state(1.0, 301.0))
    reference.act(_state(3.0, 302.0))
    expected = reference.act(_state(5.0, 303.0))

    assert after_rejection == expected


def test_reset_restores_deterministic_initial_behavior() -> None:
    controller = _controller(kp=0.02, ki=0.01, kd=0.03)
    observations = [_state(1.0, 302.0), _state(4.0, 303.0), _state(7.0, 301.0)]

    first_actions = [controller.act(observation) for observation in observations]
    controller.reset()
    second_actions = [controller.act(observation) for observation in observations]

    assert second_actions == first_actions


def test_actions_remain_in_configured_flow_range_and_satisfy_protocol() -> None:
    controller = _controller(kp=100.0, ki=100.0, kd=100.0)

    assert isinstance(controller, Controller)
    for state in (_state(0.0, 295.0), _state(1.0, 310.0), _state(2.0, 300.0)):
        action = controller.act(state)
        assert isinstance(action, CoolingAction)
        assert 0.1 <= action.coolant_mass_flow_kg_s <= 0.9


@pytest.mark.parametrize(
    ("target", "kp", "ki", "kd"),
    [
        (0.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0, 0.0),
        (300.0, -1.0, 0.0, 0.0),
        (300.0, 0.0, -1.0, 0.0),
        (300.0, 0.0, 0.0, -1.0),
        (300.0, float("nan"), 0.0, 0.0),
        (300.0, 0.0, float("inf"), 0.0),
        (300.0, 0.0, 0.0, float("inf")),
    ],
)
def test_constructor_rejects_invalid_target_or_gains(
    target: float, kp: float, ki: float, kd: float
) -> None:
    with pytest.raises(ValueError):
        PIDController(_cooling_config(), target, kp, ki, kd)
