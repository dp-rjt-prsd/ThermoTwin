"""Canonical cell adjacency for rectangular battery packs."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

from thermotwin.core import PackConfig

Edge: TypeAlias = tuple[int, int]


def _validated_dimension(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def rectangular_grid_edges(rows: int, columns: int) -> tuple[Edge, ...]:
    """Return unique neighboring cell pairs in row-major index order.

    Horizontal edges are emitted row by row, followed by vertical edges column
    by column. Each pair is stored with its lower index first.
    """
    rows = _validated_dimension("rows", rows)
    columns = _validated_dimension("columns", columns)

    horizontal = tuple(
        (row * columns + column, row * columns + column + 1)
        for row in range(rows)
        for column in range(columns - 1)
    )
    vertical = tuple(
        (row * columns + column, (row + 1) * columns + column)
        for column in range(columns)
        for row in range(rows - 1)
    )
    return horizontal + vertical


def pack_edges(pack: PackConfig) -> tuple[Edge, ...]:
    """Return the canonical adjacency for a configured pack."""
    return rectangular_grid_edges(pack.rows, pack.columns)
