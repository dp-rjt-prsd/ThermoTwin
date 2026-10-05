from __future__ import annotations

from typing import Any

import pytest

from thermotwin.core import PackConfig
from thermotwin.physics import Edge, pack_edges, rectangular_grid_edges


def make_pack(**overrides: Any) -> PackConfig:
    values: dict[str, Any] = {
        "rows": 3,
        "columns": 4,
        "cell_mass_kg": 0.8,
        "cell_specific_heat_capacity_j_kg_k": 900.0,
        "nominal_cell_resistance_ohm": 0.015,
        "cell_capacity_ah": 50.0,
        "neighbor_thermal_resistance_k_w": 1.5,
        "ambient_thermal_resistance_k_w": 8.0,
        "cell_cooling_surface_area_m2": 0.02,
    }
    return PackConfig(**(values | overrides))


def test_rectangular_grid_edges_returns_frozen_3_by_4_order() -> None:
    expected: tuple[Edge, ...] = (
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (0, 4), (4, 8),
        (1, 5), (5, 9),
        (2, 6), (6, 10),
        (3, 7), (7, 11),
    )

    assert rectangular_grid_edges(3, 4) == expected
    assert pack_edges(make_pack()) == expected


def test_edges_are_unique_undirected_pairs_with_lower_endpoint_first() -> None:
    edges = rectangular_grid_edges(3, 4)

    assert len(edges) == len(set(edges)) == 17
    assert all(first < second for first, second in edges)


def test_single_cell_grid_has_no_edges() -> None:
    assert rectangular_grid_edges(1, 1) == ()


@pytest.mark.parametrize("rows", [0, -1, 1.5, True])
def test_grid_rejects_invalid_row_dimensions(rows: Any) -> None:
    with pytest.raises(ValueError):
        rectangular_grid_edges(rows, 2)


@pytest.mark.parametrize("columns", [0, -1, 2.5, False])
def test_grid_rejects_invalid_column_dimensions(columns: Any) -> None:
    with pytest.raises(ValueError):
        rectangular_grid_edges(2, columns)
