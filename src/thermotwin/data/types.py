"""Immutable contracts for reproducible synthetic dataset scenarios and samples."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CurrentProfile,
    EnvironmentConfig,
)

DatasetMetadataValue: TypeAlias = str | int | float | bool | None
CoolantFlowProfile: TypeAlias = Callable[[float], float]

_TIME_REL_TOLERANCE = 1e-12
_TIME_ABS_TOLERANCE_S = 1e-12
_TRAJECTORY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z", re.ASCII)
_ALLOWED_PROFILE_NAMES = frozenset(
    {"constant", "pulse", "stop_and_go", "aggressive", "fast_charging"}
)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _positive_finite(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _normalize_step_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("step_index must be a nonnegative integer")
    return int(value)


def _normalize_trajectory_id(value: object) -> str:
    if not isinstance(value, str) or _TRAJECTORY_ID.fullmatch(value) is None:
        raise ValueError(
            "trajectory_id must be 1-64 ASCII alphanumeric, hyphen, or underscore "
            "characters and start with an alphanumeric character"
        )
    return value


def _normalize_metadata(value: object) -> Mapping[str, DatasetMetadataValue]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")

    copied: dict[str, DatasetMetadataValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be nonempty strings")
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise ValueError(
                "metadata values must be scalar strings, numbers, booleans, or None"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("metadata float values must be finite")
        copied[key] = item
    return MappingProxyType(copied)


def _validate_step_multiple(duration_s: float, time_step_s: float) -> int:
    try:
        step_count = round(duration_s / time_step_s)
    except OverflowError as error:
        raise ValueError(
            "duration_s must be an integer number of time steps"
        ) from error
    if step_count <= 0 or not math.isclose(
        duration_s,
        step_count * time_step_s,
        rel_tol=_TIME_REL_TOLERANCE,
        abs_tol=_TIME_ABS_TOLERANCE_S,
    ):
        raise ValueError("duration_s must be an integer number of time steps")
    return step_count


@dataclass(frozen=True, slots=True)
class DatasetScenario:
    """Fixed simulation inputs for one independently split trajectory."""

    trajectory_id: str
    environment_config: EnvironmentConfig
    current_profile: CurrentProfile
    coolant_flow_profile: CoolantFlowProfile
    duration_s: float
    time_step_s: float
    metadata: Mapping[str, DatasetMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_id", _normalize_trajectory_id(self.trajectory_id)
        )
        if type(self.environment_config) is not EnvironmentConfig:
            raise ValueError("environment_config must be an EnvironmentConfig")
        if not callable(self.current_profile):
            raise ValueError("current_profile must be callable")
        if not callable(self.coolant_flow_profile):
            raise ValueError("coolant_flow_profile must be callable")

        duration = _positive_finite("duration_s", self.duration_s)
        time_step = _positive_finite("time_step_s", self.time_step_s)
        _validate_step_multiple(duration, time_step)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "time_step_s", time_step)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def step_count(self) -> int:
        """Number of fixed simulation steps represented by this scenario."""
        return round(float(self.duration_s) / float(self.time_step_s))


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One immutable simulator transition with applied action provenance."""

    trajectory_id: str
    step_index: int
    time_step_s: float
    state: BatteryState
    action: CoolingAction
    next_state: BatteryState
    metadata: Mapping[str, DatasetMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "trajectory_id", _normalize_trajectory_id(self.trajectory_id)
        )
        object.__setattr__(self, "step_index", _normalize_step_index(self.step_index))
        time_step = _positive_finite("time_step_s", self.time_step_s)
        object.__setattr__(self, "time_step_s", time_step)
        if type(self.state) is not BatteryState:
            raise ValueError("state must be a BatteryState")
        if type(self.action) is not CoolingAction:
            raise ValueError("action must be a CoolingAction")
        if type(self.next_state) is not BatteryState:
            raise ValueError("next_state must be a BatteryState")
        cell_count = self.state.cell_temperatures_k.size
        if cell_count == 0 or self.next_state.cell_temperatures_k.size != cell_count:
            raise ValueError(
                "state and next_state must have matching nonempty cell counts"
            )
        elapsed = self.next_state.time_s - self.state.time_s
        if not math.isclose(
            elapsed,
            time_step,
            rel_tol=_TIME_REL_TOLERANCE,
            abs_tol=_TIME_ABS_TOLERANCE_S,
        ):
            raise ValueError("next_state time must follow state by time_step_s")
        if not math.isclose(
            self.next_state.coolant_mass_flow_kg_s,
            self.action.coolant_mass_flow_kg_s,
            rel_tol=_TIME_REL_TOLERANCE,
            abs_tol=_TIME_ABS_TOLERANCE_S,
        ):
            raise ValueError("next_state coolant flow must match the action")
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class DatasetSamplingConfig:
    """Validated finite ranges used to sample dataset trajectory conditions."""

    duration_s: float
    time_step_s: float
    initial_soc_fraction_range: tuple[float, float]
    initial_cell_temperature_k_range: tuple[float, float]
    ambient_temperature_k_range: tuple[float, float]
    coolant_inlet_temperature_k_range: tuple[float, float]
    peak_current_a_range: tuple[float, float]
    coolant_mass_flow_kg_s_range: tuple[float, float]
    cell_resistance_scale_range: tuple[float, float]
    neighbor_thermal_resistance_scale_range: tuple[float, float]
    cooling_effectiveness_scale_range: tuple[float, float]
    allowed_profile_names: tuple[str, ...]

    def __post_init__(self) -> None:
        duration = _positive_finite("duration_s", self.duration_s)
        time_step = _positive_finite("time_step_s", self.time_step_s)
        _validate_step_multiple(duration, time_step)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "time_step_s", time_step)

        range_names = (
            "initial_soc_fraction_range",
            "initial_cell_temperature_k_range",
            "ambient_temperature_k_range",
            "coolant_inlet_temperature_k_range",
            "peak_current_a_range",
            "coolant_mass_flow_kg_s_range",
            "cell_resistance_scale_range",
            "neighbor_thermal_resistance_scale_range",
            "cooling_effectiveness_scale_range",
        )
        for name in range_names:
            normalized = self._normalize_range(name, getattr(self, name))
            object.__setattr__(self, name, normalized)

        self._validate_domain("initial_soc_fraction_range", 0.0, 1.0, inclusive=True)
        for name in (
            "initial_cell_temperature_k_range",
            "ambient_temperature_k_range",
            "coolant_inlet_temperature_k_range",
            "peak_current_a_range",
        ):
            self._validate_domain(name, 0.0, None)
        self._validate_domain("coolant_mass_flow_kg_s_range", 0.0, None, inclusive=True)
        for name in (
            "cell_resistance_scale_range",
            "neighbor_thermal_resistance_scale_range",
            "cooling_effectiveness_scale_range",
        ):
            self._validate_domain(name, 0.0, None)

        profiles = self.allowed_profile_names
        if not isinstance(profiles, tuple) or not profiles:
            raise ValueError("allowed_profile_names must be a nonempty tuple")
        if any(
            not isinstance(name, str) or name not in _ALLOWED_PROFILE_NAMES
            for name in profiles
        ):
            raise ValueError("allowed_profile_names contains an unsupported profile")
        if len(set(profiles)) != len(profiles):
            raise ValueError("allowed_profile_names must contain unique names")

    @staticmethod
    def _normalize_range(name: str, value: object) -> tuple[float, float]:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(f"{name} must be a two-value tuple")
        lower = _finite_real(f"{name}[0]", value[0])
        upper = _finite_real(f"{name}[1]", value[1])
        if lower >= upper:
            raise ValueError(f"{name} must be strictly ascending")
        return (lower, upper)

    def _validate_domain(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float | None,
        *,
        inclusive: bool = False,
    ) -> None:
        lower, upper = getattr(self, name)
        lower_invalid = lower < lower_bound if inclusive else lower <= lower_bound
        if lower_invalid or (upper_bound is not None and upper > upper_bound):
            if name == "initial_soc_fraction_range":
                raise ValueError(f"{name} endpoints must be in [0, 1]")
            relation = "nonnegative" if inclusive else "positive"
            raise ValueError(f"{name} endpoints must be {relation}")
