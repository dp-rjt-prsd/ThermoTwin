"""Pure per-cell thermal calculations using SI units."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _positive_finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float64_vector(
    name: str, values: ArrayLike, *, strictly_positive: bool = False
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must contain real values")
        array = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite numeric vector") from error

    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if strictly_positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def joule_heating_w(
    current_a: float, cell_resistances_ohm: ArrayLike
) -> NDArray[np.float64]:
    """Return per-cell irreversible heat generation, ``I**2 * R``, in watts."""
    current = _finite_scalar("current_a", current_a)
    resistances = _float64_vector(
        "cell_resistances_ohm", cell_resistances_ohm, strictly_positive=True
    )
    with np.errstate(over="ignore", invalid="ignore"):
        heating = resistances * (current * current)
    if not np.all(np.isfinite(heating)):
        raise ValueError("joule heating must be finite")
    return heating


def temperature_rate_k_s(
    net_heat_w: ArrayLike,
    cell_mass_kg: float,
    cell_specific_heat_capacity_j_kg_k: float,
) -> NDArray[np.float64]:
    """Convert per-cell net heat in watts to temperature rates in K/s."""
    net_heat = _float64_vector("net_heat_w", net_heat_w)
    mass = _positive_finite_scalar("cell_mass_kg", cell_mass_kg)
    specific_heat = _positive_finite_scalar(
        "cell_specific_heat_capacity_j_kg_k",
        cell_specific_heat_capacity_j_kg_k,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        thermal_capacity = mass * specific_heat
        if not math.isfinite(thermal_capacity):
            raise ValueError("cell thermal capacity must be finite")
        rates = net_heat / thermal_capacity
    if not np.all(np.isfinite(rates)):
        raise ValueError("temperature rates must be finite")
    return rates


def integrate_temperature_euler(
    cell_temperatures_k: ArrayLike,
    temperature_rate_k_s: ArrayLike,
    dt_s: float,
) -> NDArray[np.float64]:
    """Advance cell temperatures by one explicit Euler time step."""
    temperatures = _float64_vector(
        "cell_temperatures_k", cell_temperatures_k, strictly_positive=True
    )
    rates = _float64_vector("temperature_rate_k_s", temperature_rate_k_s)
    dt = _positive_finite_scalar("dt_s", dt_s)
    if temperatures.shape != rates.shape:
        raise ValueError(
            "cell_temperatures_k and temperature_rate_k_s must have equal shapes"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        updated = temperatures + rates * dt
    if not np.all(np.isfinite(updated)) or np.any(updated <= 0.0):
        raise ValueError("resulting cell temperatures must be finite and positive")
    return updated
