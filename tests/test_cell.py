from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.physics.cell import (
    integrate_temperature_euler,
    joule_heating_w,
    temperature_rate_k_s,
)


def test_joule_heating_is_quadratic_in_current_and_linear_in_resistance() -> None:
    resistances = np.array([0.01, 0.02], dtype=np.float32)

    heating = joule_heating_w(10.0, resistances)

    np.testing.assert_allclose(heating, [1.0, 2.0])
    np.testing.assert_allclose(joule_heating_w(20.0, resistances), 4.0 * heating)
    np.testing.assert_allclose(joule_heating_w(-10.0, resistances), heating)
    assert heating.dtype == np.float64
    assert heating.ndim == 1
    assert not np.shares_memory(heating, resistances)


def test_joule_heating_returns_zero_at_zero_current() -> None:
    np.testing.assert_array_equal(joule_heating_w(0.0, [0.01, 0.02]), [0.0, 0.0])


def test_temperature_rate_uses_net_heat_over_thermal_capacity() -> None:
    heat = np.array([180.0, -90.0], dtype=np.float32)

    rates = temperature_rate_k_s(heat, 0.8, 900.0)

    np.testing.assert_allclose(rates, [0.25, -0.125])
    assert rates.dtype == np.float64
    assert not np.shares_memory(rates, heat)


def test_euler_integration_returns_an_independent_temperature_vector() -> None:
    temperatures = np.array([300.0, 301.0], dtype=np.float32)
    rates = np.array([2.0, -1.0], dtype=np.float32)

    updated = integrate_temperature_euler(temperatures, rates, 0.5)

    np.testing.assert_allclose(updated, [301.0, 300.5])
    assert updated.dtype == np.float64
    assert not np.shares_memory(updated, temperatures)
    assert not np.shares_memory(updated, rates)


@pytest.mark.parametrize(
    "resistances",
    [[], np.array(0.01), [[0.01]], [0.01, np.nan], [0.01, np.inf]],
)
def test_joule_heating_rejects_invalid_resistance_vectors(resistances: Any) -> None:
    with pytest.raises(ValueError):
        joule_heating_w(10.0, resistances)


@pytest.mark.parametrize("resistance", [[0.0], [-0.01]])
def test_joule_heating_requires_positive_resistances(resistance: Any) -> None:
    with pytest.raises(ValueError):
        joule_heating_w(10.0, resistance)


@pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
def test_joule_heating_rejects_non_finite_current(current: float) -> None:
    with pytest.raises(ValueError, match="current_a"):
        joule_heating_w(current, [0.01])


@pytest.mark.parametrize("heat", [[], np.array(1.0), [[1.0]], [np.nan], [np.inf]])
def test_temperature_rate_rejects_invalid_net_heat_vectors(heat: Any) -> None:
    with pytest.raises(ValueError):
        temperature_rate_k_s(heat, 1.0, 1.0)


@pytest.mark.parametrize(
    ("mass", "capacity"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (np.nan, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, np.inf),
    ],
)
def test_temperature_rate_requires_positive_finite_thermal_parameters(
    mass: float, capacity: float
) -> None:
    with pytest.raises(ValueError):
        temperature_rate_k_s([1.0], mass, capacity)


@pytest.mark.parametrize(
    ("temperatures", "rates", "dt"),
    [
        ([], [], 1.0),
        (np.array(300.0), np.array(1.0), 1.0),
        ([[300.0]], [[1.0]], 1.0),
        ([300.0], [1.0, 2.0], 1.0),
        ([300.0], [np.nan], 1.0),
        ([300.0], [np.inf], 1.0),
        ([0.0], [1.0], 1.0),
        ([np.nan], [1.0], 1.0),
        ([300.0], [1.0], 0.0),
        ([300.0], [1.0], np.inf),
    ],
)
def test_euler_integration_rejects_invalid_inputs(
    temperatures: Any, rates: Any, dt: float
) -> None:
    with pytest.raises(ValueError):
        integrate_temperature_euler(temperatures, rates, dt)


def test_euler_integration_rejects_nonpositive_or_nonfinite_result() -> None:
    with pytest.raises(ValueError, match="temperature"):
        integrate_temperature_euler([1.0], [-2.0], 1.0)
    with pytest.raises(ValueError, match="temperature"):
        integrate_temperature_euler([1e308], [1e308], 2.0)


@pytest.mark.parametrize("scalar", [0.0, -1.0, np.nan, np.inf])
def test_temperature_rate_rejects_invalid_scalar_thermal_parameters(
    scalar: float,
) -> None:
    with pytest.raises(ValueError):
        temperature_rate_k_s([1.0], scalar, 1.0)
    with pytest.raises(ValueError):
        temperature_rate_k_s([1.0], 1.0, scalar)


@pytest.mark.parametrize("dt", [-1.0, np.nan, -np.inf])
def test_euler_integration_requires_positive_finite_time_step(dt: float) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        integrate_temperature_euler([300.0], [1.0], dt)
