"""Pure pack-level thermal calculations."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermotwin.physics.topology import Edge


def _validated_resistance(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("thermal_resistance_k_w must be a positive finite number")
    try:
        resistance = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "thermal_resistance_k_w must be a positive finite number"
        ) from error
    if not math.isfinite(resistance) or resistance <= 0.0:
        raise ValueError("thermal_resistance_k_w must be a positive finite number")
    return resistance


def _validated_edge(edge: object, cell_count: int) -> Edge:
    if not isinstance(edge, tuple) or len(edge) != 2:
        raise ValueError("each edge must be a pair of integer endpoints")

    first, second = edge
    if (
        isinstance(first, (bool, np.bool_))
        or isinstance(second, (bool, np.bool_))
        or not isinstance(first, Integral)
        or not isinstance(second, Integral)
    ):
        raise ValueError("edge endpoints must be integers, not booleans")

    first_index = int(first)
    second_index = int(second)
    if not (0 <= first_index < cell_count) or not (0 <= second_index < cell_count):
        raise ValueError("edge endpoints must be within the temperature array")
    if first_index >= second_index:
        raise ValueError("each edge must be canonically ordered with i < j")

    return first_index, second_index


def _validated_edges(edges: object, cell_count: int) -> tuple[Edge, ...]:
    if not isinstance(edges, tuple):
        raise ValueError("edges must be a tuple of canonical (i, j) pairs")

    validated_edges: list[Edge] = []
    seen_edges: set[Edge] = set()
    for edge in edges:
        canonical_edge = _validated_edge(edge, cell_count)
        if canonical_edge in seen_edges:
            raise ValueError(f"duplicate edge {canonical_edge}")
        seen_edges.add(canonical_edge)
        validated_edges.append(canonical_edge)

    return tuple(validated_edges)


def neighbor_conduction_w(
    cell_temperatures_k: ArrayLike,
    edges: tuple[Edge, ...],
    thermal_resistance_k_w: float,
) -> NDArray[np.float64]:
    """Return each cell's net heat flow from its neighboring cells in watts.

    For every undirected edge ``(i, j)``, heat enters ``i`` at the rate
    ``(T[j] - T[i]) / R_th`` and enters ``j`` at the equal, opposite rate.
    """
    try:
        temperatures = np.array(cell_temperatures_k, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "cell_temperatures_k must be a nonempty one-dimensional array "
            "of positive finite Kelvin values"
        ) from error

    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError(
            "cell_temperatures_k must be a nonempty one-dimensional array"
        )
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError(
            "cell_temperatures_k must contain positive finite Kelvin values"
        )

    resistance = _validated_resistance(thermal_resistance_k_w)

    validated_edges = _validated_edges(edges, temperatures.size)

    heat_flow = np.zeros(temperatures.shape, dtype=np.float64)
    for first_index, second_index in validated_edges:
        pair_flow = (
            temperatures[second_index] - temperatures[first_index]
        ) / resistance
        heat_flow[first_index] += pair_flow
        heat_flow[second_index] -= pair_flow

    return heat_flow
