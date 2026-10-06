from __future__ import annotations

import math

import pytest

from thermotwin.control.objective import pump_energy_j, pump_power_w


def test_pump_power_matches_configured_cubic_approximation() -> None:
    flow = 0.05
    coefficient = 120_000.0

    assert pump_power_w(flow, coefficient) == pytest.approx(coefficient * flow**3)


def test_pump_power_scales_cubically_with_flow() -> None:
    coefficient = 2_000.0
    low_flow_power = pump_power_w(0.02, coefficient)
    doubled_flow_power = pump_power_w(0.04, coefficient)

    assert doubled_flow_power == pytest.approx(8.0 * low_flow_power)


def test_pump_power_is_monotonic_in_flow_and_coefficient() -> None:
    baseline = pump_power_w(0.03, 1_000.0)

    assert pump_power_w(0.04, 1_000.0) > baseline
    assert pump_power_w(0.03, 2_000.0) > baseline


@pytest.mark.parametrize(
    ("flow", "coefficient"),
    [
        (0.0, 1_000.0),
        (0.1, 0.0),
        (0.0, 0.0),
    ],
)
def test_pump_power_returns_zero_when_flow_or_coefficient_is_zero(
    flow: float, coefficient: float
) -> None:
    assert pump_power_w(flow, coefficient) == 0.0


def test_pump_energy_matches_power_times_duration() -> None:
    flow = 0.04
    duration = 15.0
    coefficient = 50_000.0

    assert pump_energy_j(flow, duration, coefficient) == pytest.approx(
        pump_power_w(flow, coefficient) * duration
    )


@pytest.mark.parametrize(
    ("flow", "duration", "coefficient"),
    [
        (0.0, 10.0, 1_000.0),
        (0.1, 0.0, 1_000.0),
        (0.1, 10.0, 0.0),
    ],
)
def test_pump_energy_returns_zero_for_zero_flow_duration_or_coefficient(
    flow: float, duration: float, coefficient: float
) -> None:
    assert pump_energy_j(flow, duration, coefficient) == 0.0


@pytest.mark.parametrize("value", [True, False, "0.1", None, complex(1.0, 0.0)])
def test_pump_power_rejects_non_real_scalar_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        pump_power_w(value, 1_000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pump_power_w(0.1, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_pump_power_rejects_negative_or_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        pump_power_w(value, 1_000.0)
    with pytest.raises(ValueError):
        pump_power_w(0.1, value)


@pytest.mark.parametrize("value", [True, "1.0", None, complex(1.0, 0.0)])
def test_pump_energy_rejects_non_real_scalar_inputs(value: object) -> None:
    with pytest.raises(ValueError):
        pump_energy_j(value, 2.0, 1_000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pump_energy_j(0.1, value, 1_000.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pump_energy_j(0.1, 2.0, value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [-1.0, float("nan"), float("inf"), -float("inf")]
)
def test_pump_energy_rejects_negative_or_nonfinite_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        pump_energy_j(value, 2.0, 1_000.0)
    with pytest.raises(ValueError):
        pump_energy_j(0.1, value, 1_000.0)
    with pytest.raises(ValueError):
        pump_energy_j(0.1, 2.0, value)


def test_pump_calculations_reject_nonfinite_results() -> None:
    with pytest.raises(ValueError, match="finite"):
        pump_power_w(1e200, 1.0)
    with pytest.raises(ValueError, match="finite"):
        pump_energy_j(0.1, 1e308, 1e308)


def test_analytical_pump_example_has_expected_units_and_values() -> None:
    # With k=10_000 W*s^3/kg^3 and flow=0.1 kg/s, power is 10 W.
    power = pump_power_w(0.1, 10_000.0)
    energy = pump_energy_j(0.1, 30.0, 10_000.0)

    assert math.isclose(power, 10.0)
    assert math.isclose(energy, 300.0)
