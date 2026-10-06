"""Frozen value contracts shared by ThermoTwin subsystems."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
CurrentProfile = Callable[[float], float]


def _require_finite(name: str, value: object) -> float:
    """Return a scalar as float, rejecting nonnumeric and nonfinite inputs."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_positive(name: str, value: object) -> None:
    if _require_finite(name, value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: object) -> None:
    if _require_finite(name, value) < 0.0:
        raise ValueError(f"{name} must be nonnegative")


def _require_soc(name: str, value: object) -> None:
    scalar = _require_finite(name, value)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_dimension(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _readonly_positive_vector(name: str, value: Any) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain positive finite values") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")

    # Back the stored view with immutable bytes so callers cannot re-enable writes.
    return np.frombuffer(array.tobytes(), dtype=np.float64)


def _normalize_positive_real_tuple(
    value: object, field_name: str
) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{field_name} must be a nonempty tuple or None")

    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{field_name} must contain positive finite real values")
        normalized_item = float(item)
        if not math.isfinite(normalized_item) or normalized_item <= 0.0:
            raise ValueError(f"{field_name} must contain positive finite real values")
        normalized.append(normalized_item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class CoolingAction:
    """Coolant command in kilograms per second."""

    coolant_mass_flow_kg_s: float

    def __post_init__(self) -> None:
        _require_nonnegative("coolant_mass_flow_kg_s", self.coolant_mass_flow_kg_s)


@dataclass(frozen=True, slots=True)
class BatteryState:
    """Instantaneous battery state using SI units."""

    time_s: float
    current_a: float
    soc_fraction: float
    ambient_temperature_k: float
    coolant_inlet_temperature_k: float
    coolant_mass_flow_kg_s: float
    cell_temperatures_k: FloatArray
    cell_resistances_ohm: FloatArray

    def __post_init__(self) -> None:
        _require_nonnegative("time_s", self.time_s)
        _require_finite("current_a", self.current_a)
        _require_soc("soc_fraction", self.soc_fraction)
        _require_positive("ambient_temperature_k", self.ambient_temperature_k)
        _require_positive(
            "coolant_inlet_temperature_k", self.coolant_inlet_temperature_k
        )
        _require_nonnegative("coolant_mass_flow_kg_s", self.coolant_mass_flow_kg_s)

        temperatures = _readonly_positive_vector(
            "cell_temperatures_k", self.cell_temperatures_k
        )
        resistances = _readonly_positive_vector(
            "cell_resistances_ohm", self.cell_resistances_ohm
        )
        if temperatures.shape != resistances.shape:
            raise ValueError(
                "cell_temperatures_k and cell_resistances_ohm must have equal shapes"
            )
        object.__setattr__(self, "cell_temperatures_k", temperatures)
        object.__setattr__(self, "cell_resistances_ohm", resistances)


@dataclass(frozen=True, slots=True)
class PackConfig:
    """Lumped pack and per-cell physical parameters."""

    rows: int
    columns: int
    cell_mass_kg: float
    cell_specific_heat_capacity_j_kg_k: float
    nominal_cell_resistance_ohm: float
    cell_capacity_ah: float
    neighbor_thermal_resistance_k_w: float
    ambient_thermal_resistance_k_w: float
    cell_cooling_surface_area_m2: float

    def __post_init__(self) -> None:
        _require_dimension("rows", self.rows)
        _require_dimension("columns", self.columns)
        for name in (
            "cell_mass_kg",
            "cell_specific_heat_capacity_j_kg_k",
            "nominal_cell_resistance_ohm",
            "cell_capacity_ah",
            "neighbor_thermal_resistance_k_w",
            "ambient_thermal_resistance_k_w",
            "cell_cooling_surface_area_m2",
        ):
            _require_positive(name, getattr(self, name))

    @property
    def cell_count(self) -> int:
        return self.rows * self.columns


@dataclass(frozen=True, slots=True)
class CoolingConfig:
    """Liquid cooling parameters in SI units."""

    minimum_coolant_mass_flow_kg_s: float
    maximum_coolant_mass_flow_kg_s: float
    base_heat_transfer_coefficient_w_m2_k: float
    flow_heat_transfer_gain: float
    flow_exponent: float

    def __post_init__(self) -> None:
        _require_nonnegative(
            "minimum_coolant_mass_flow_kg_s", self.minimum_coolant_mass_flow_kg_s
        )
        _require_nonnegative(
            "maximum_coolant_mass_flow_kg_s", self.maximum_coolant_mass_flow_kg_s
        )
        if self.minimum_coolant_mass_flow_kg_s > self.maximum_coolant_mass_flow_kg_s:
            raise ValueError(
                "minimum_coolant_mass_flow_kg_s must not exceed "
                "maximum_coolant_mass_flow_kg_s"
            )
        for name in (
            "base_heat_transfer_coefficient_w_m2_k",
            "flow_heat_transfer_gain",
            "flow_exponent",
        ):
            _require_positive(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Initial conditions and default integration step."""

    default_time_step_s: float
    initial_soc_fraction: float
    initial_cell_temperature_k: float
    ambient_temperature_k: float
    coolant_inlet_temperature_k: float
    initial_current_a: float
    initial_cell_temperatures_k: tuple[float, ...] | None = None
    initial_cell_resistances_ohm: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _require_positive("default_time_step_s", self.default_time_step_s)
        _require_soc("initial_soc_fraction", self.initial_soc_fraction)
        for name in (
            "initial_cell_temperature_k",
            "ambient_temperature_k",
            "coolant_inlet_temperature_k",
        ):
            _require_positive(name, getattr(self, name))
        _require_finite("initial_current_a", self.initial_current_a)

        temperatures = self.initial_cell_temperatures_k
        if temperatures is not None:
            object.__setattr__(
                self,
                "initial_cell_temperatures_k",
                _normalize_positive_real_tuple(
                    temperatures, "initial_cell_temperatures_k"
                ),
            )

        resistances = self.initial_cell_resistances_ohm
        if resistances is not None:
            object.__setattr__(
                self,
                "initial_cell_resistances_ohm",
                _normalize_positive_real_tuple(
                    resistances, "initial_cell_resistances_ohm"
                ),
            )


@dataclass(frozen=True, slots=True)
class ResistanceFaultConfig:
    """Scheduled single-cell resistance fault, retained as simulator provenance."""

    cell_index: int
    start_time_s: float
    resistance_multiplier: float

    def __post_init__(self) -> None:
        _require_nonnegative_integer("cell_index", self.cell_index)

        for name in ("start_time_s", "resistance_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            try:
                scalar = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError(f"{name} must be a finite real number") from error
            if not math.isfinite(scalar):
                raise ValueError(f"{name} must be finite")
            if name == "start_time_s" and scalar < 0.0:
                raise ValueError("start_time_s must be nonnegative")
            if name == "resistance_multiplier" and scalar <= 1.0:
                raise ValueError("resistance_multiplier must be greater than 1")


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Configuration bundle for a battery environment."""

    pack: PackConfig
    cooling: CoolingConfig
    simulation: SimulationConfig
    resistance_fault: ResistanceFaultConfig | None = None
