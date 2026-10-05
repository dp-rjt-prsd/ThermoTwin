from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.core import CoolingConfig
from thermotwin.physics.cooling import (
    ambient_heat_exchange_w,
    cooling_heat_removal_w,
    heat_transfer_coefficient_w_m2_k,
)


def make_cooling(**overrides: Any) -> CoolingConfig:
    values: dict[str, Any] = {
        "minimum_coolant_mass_flow_kg_s": 0.0,
        "maximum_coolant_mass_flow_kg_s": 0.1,
        "base_heat_transfer_coefficient_w_m2_k": 5.0,
        "flow_heat_transfer_gain": 500.0,
        "flow_exponent": 0.8,
    }
    return CoolingConfig(**(values | overrides))


def test_heat_transfer_coefficient_matches_configured_flow_law() -> None:
    config = make_cooling()
    flow = 0.04

    result = heat_transfer_coefficient_w_m2_k(flow, config)

    assert result == pytest.approx(5.0 + 500.0 * flow**0.8)


@pytest.mark.parametrize("flow", [0.0, 0.1])
def test_heat_transfer_coefficient_accepts_inclusive_flow_limits(flow: float) -> None:
    result = heat_transfer_coefficient_w_m2_k(flow, make_cooling())

    assert np.isfinite(result)


@pytest.mark.parametrize(
    "flow", [-0.01, 0.1001, float("nan"), float("inf"), -float("inf")]
)
def test_heat_transfer_coefficient_rejects_illegal_flow(flow: float) -> None:
    with pytest.raises(ValueError, match="coolant_mass_flow_kg_s"):
        heat_transfer_coefficient_w_m2_k(flow, make_cooling())


def test_cooling_heat_removal_has_signed_direction_and_increases_with_flow() -> None:
    temperatures = np.array([310.0, 300.0, 290.0])
    coolant_temperature = 300.0
    config = make_cooling()

    low = cooling_heat_removal_w(temperatures, coolant_temperature, 0.02, 0.5, config)
    high = cooling_heat_removal_w(temperatures, coolant_temperature, 0.08, 0.5, config)

    expected = heat_transfer_coefficient_w_m2_k(0.02, config) * 5.0
    assert low == pytest.approx([expected, 0.0, -expected])
    assert high[0] > low[0]
    assert high[2] < low[2]
    assert high.dtype == np.float64
    assert high.ndim == 1


def test_cooling_heat_removal_returns_independent_float64_vector() -> None:
    source = np.array([300.0, 301.0], dtype=np.float32)

    result = cooling_heat_removal_w(source, 290.0, 0.03, 0.02, make_cooling())

    assert result.dtype == np.float64
    assert result.ndim == 1
    assert not np.shares_memory(result, source)
    result[0] = -1.0
    assert source[0] == 300.0


@pytest.mark.parametrize(
    "temperatures",
    [[], np.array(300.0), [[300.0]], [0.0], [-1.0], [float("nan")], [float("inf")]],
)
def test_cooling_heat_removal_rejects_invalid_cell_temperatures(
    temperatures: Any,
) -> None:
    with pytest.raises(ValueError, match="cell_temperatures_k"):
        cooling_heat_removal_w(temperatures, 290.0, 0.03, 0.02, make_cooling())


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_cooling_heat_removal_rejects_invalid_scalar_inputs(value: float) -> None:
    with pytest.raises(ValueError):
        cooling_heat_removal_w([300.0], value, 0.03, 0.02, make_cooling())
    with pytest.raises(ValueError):
        cooling_heat_removal_w([300.0], 290.0, 0.03, value, make_cooling())


def test_ambient_heat_exchange_has_signed_direction() -> None:
    result = ambient_heat_exchange_w([290.0, 300.0, 310.0], 300.0, 2.0)

    assert result == pytest.approx([5.0, 0.0, -5.0])
    assert result.dtype == np.float64
    assert result.ndim == 1


@pytest.mark.parametrize(
    "temperatures",
    [[], np.array(300.0), [[300.0]], [0.0], [-1.0], [float("nan")], [float("inf")]],
)
def test_ambient_heat_exchange_rejects_invalid_cell_temperatures(
    temperatures: Any,
) -> None:
    with pytest.raises(ValueError, match="cell_temperatures_k"):
        ambient_heat_exchange_w(temperatures, 300.0, 2.0)


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_ambient_heat_exchange_rejects_invalid_ambient_or_resistance(
    value: float,
) -> None:
    with pytest.raises(ValueError):
        ambient_heat_exchange_w([300.0], value, 2.0)
    with pytest.raises(ValueError):
        ambient_heat_exchange_w([300.0], 300.0, value)
