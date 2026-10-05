"""Pure liquid-cooling and ambient heat-exchange calculations."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from thermotwin.core.types import CoolingConfig


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(name: str, value: object) -> float:
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _temperature_vector(cell_temperatures_k: ArrayLike) -> NDArray[np.float64]:
    try:
        raw = np.asarray(cell_temperatures_k)
        if np.iscomplexobj(raw):
            raise ValueError
        temperatures = np.array(cell_temperatures_k, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "cell_temperatures_k must contain positive finite values"
        ) from error
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("cell_temperatures_k must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("cell_temperatures_k must contain positive finite values")
    return temperatures


def heat_transfer_coefficient_w_m2_k(
    coolant_mass_flow_kg_s: float, config: CoolingConfig
) -> float:
    """Calculate the effective heat-transfer coefficient for coolant flow."""
    flow = _finite_scalar("coolant_mass_flow_kg_s", coolant_mass_flow_kg_s)
    if not (
        config.minimum_coolant_mass_flow_kg_s
        <= flow
        <= config.maximum_coolant_mass_flow_kg_s
    ):
        raise ValueError("coolant_mass_flow_kg_s is outside configured limits")

    coefficient = float(
        config.base_heat_transfer_coefficient_w_m2_k
        + config.flow_heat_transfer_gain * flow**config.flow_exponent
    )
    if not math.isfinite(coefficient):
        raise ValueError("heat-transfer coefficient must be finite")
    return coefficient


def cooling_heat_removal_w(
    cell_temperatures_k: ArrayLike,
    coolant_inlet_temperature_k: float,
    coolant_mass_flow_kg_s: float,
    cell_surface_area_m2: float,
    config: CoolingConfig,
) -> NDArray[np.float64]:
    """Return signed per-cell heat transfer from cells to the coolant, in watts."""
    temperatures = _temperature_vector(cell_temperatures_k)
    coolant_temperature = _positive_scalar(
        "coolant_inlet_temperature_k", coolant_inlet_temperature_k
    )
    area = _positive_scalar("cell_surface_area_m2", cell_surface_area_m2)
    coefficient = heat_transfer_coefficient_w_m2_k(coolant_mass_flow_kg_s, config)
    heat_removal = coefficient * area * (temperatures - coolant_temperature)
    if not np.all(np.isfinite(heat_removal)):
        raise ValueError("cooling heat removal must be finite")
    return np.asarray(heat_removal, dtype=np.float64)


def ambient_heat_exchange_w(
    cell_temperatures_k: ArrayLike,
    ambient_temperature_k: float,
    ambient_thermal_resistance_k_w: float,
) -> NDArray[np.float64]:
    """Return signed per-cell heat transfer from ambient air, in watts."""
    temperatures = _temperature_vector(cell_temperatures_k)
    ambient_temperature = _positive_scalar(
        "ambient_temperature_k", ambient_temperature_k
    )
    resistance = _positive_scalar(
        "ambient_thermal_resistance_k_w", ambient_thermal_resistance_k_w
    )
    heat_exchange = (ambient_temperature - temperatures) / resistance
    if not np.all(np.isfinite(heat_exchange)):
        raise ValueError("ambient heat exchange must be finite")
    return np.asarray(heat_exchange, dtype=np.float64)
