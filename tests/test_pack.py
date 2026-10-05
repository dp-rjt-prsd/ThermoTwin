from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.physics.pack import neighbor_conduction_w
from thermotwin.physics.topology import rectangular_grid_edges


def test_equal_temperature_cells_exchange_no_heat() -> None:
    temperatures = np.full(12, 298.15)
    result = neighbor_conduction_w(
        temperatures, rectangular_grid_edges(3, 4), 1.5
    )

    np.testing.assert_array_equal(result, np.zeros(12))


def test_conduction_moves_heat_from_hot_cell_to_cold_neighbor() -> None:
    temperatures = np.array([310.0, 290.0])

    result = neighbor_conduction_w(temperatures, ((0, 1),), 2.0)

    np.testing.assert_allclose(result, [-10.0, 10.0])


def test_canonical_pack_conduction_is_conservative() -> None:
    temperatures = np.linspace(290.0, 310.0, 12)

    result = neighbor_conduction_w(
        temperatures, rectangular_grid_edges(3, 4), 1.5
    )

    assert result.shape == temperatures.shape
    assert np.sum(result) == pytest.approx(0.0, abs=1e-12)


def test_empty_edges_return_an_independent_float64_zero_vector() -> None:
    temperatures = np.array([290.0, 300.0])

    result = neighbor_conduction_w(temperatures, (), 1.0)

    assert result.shape == temperatures.shape
    assert result.dtype == np.float64
    assert not np.shares_memory(result, temperatures)
    np.testing.assert_array_equal(result, np.zeros_like(temperatures))


@pytest.mark.parametrize(
    "temperatures",
    [
        [],
        np.array([]),
        np.array([[290.0, 291.0]]),
        np.array([0.0]),
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        ["not a temperature"],
    ],
)
def test_rejects_invalid_temperature_arrays(temperatures: Any) -> None:
    with pytest.raises(ValueError):
        neighbor_conduction_w(temperatures, (), 1.0)


@pytest.mark.parametrize("resistance", [0.0, -1.0, np.nan, np.inf, True])
def test_rejects_invalid_thermal_resistance(resistance: Any) -> None:
    with pytest.raises(ValueError):
        neighbor_conduction_w([290.0, 300.0], ((0, 1),), resistance)


@pytest.mark.parametrize(
    "edges",
    [
        ((-1, 0),),
        ((0, 2),),
        ((1, 1),),
        ((1, 0),),
        ((0, 1), (0, 1)),
        ((0, True),),
        ((0, 1.0),),
        ((0,),),
        ((0, 1, 2),),
    ],
)
def test_rejects_invalid_edges(edges: Any) -> None:
    with pytest.raises(ValueError):
        neighbor_conduction_w([290.0, 300.0], edges, 1.0)
