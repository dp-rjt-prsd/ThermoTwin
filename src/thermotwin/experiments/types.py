"""Immutable contracts for reproducible thermal experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

from thermotwin.core.types import BatteryState, CurrentProfile, EnvironmentConfig
from thermotwin.diagnostics.metrics import ExperimentMetrics

MetadataValue = str | int | float | bool | None

_TIME_REL_TOLERANCE = 1e-12
_TIME_ABS_TOLERANCE_S = 1e-12


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


def _nonempty_name(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _positive_finite(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _normalize_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _normalize_metadata(
    value: object,
) -> Mapping[str, MetadataValue]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping or None")

    copied: dict[str, MetadataValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be nonempty strings")
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise ValueError(
                "metadata values must be scalar strings, numbers, booleans, or None"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("metadata numeric values must be finite")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ExperimentScenario:
    """Fixed reproducible inputs for one experiment trajectory."""

    name: str
    environment_config: EnvironmentConfig
    current_profile: CurrentProfile
    duration_s: float
    time_step_s: float
    thermal_limit_k: float
    pump_power_coefficient_w_s3_kg3: float
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_name("name", self.name)
        if not isinstance(self.environment_config, EnvironmentConfig):
            raise ValueError("environment_config must be an EnvironmentConfig")
        if not callable(self.current_profile):
            raise ValueError("current_profile must be callable")

        duration = _positive_finite("duration_s", self.duration_s)
        time_step = _positive_finite("time_step_s", self.time_step_s)
        thermal_limit = _positive_finite("thermal_limit_k", self.thermal_limit_k)
        coefficient = _finite_real(
            "pump_power_coefficient_w_s3_kg3",
            self.pump_power_coefficient_w_s3_kg3,
        )
        if coefficient < 0.0:
            raise ValueError("pump_power_coefficient_w_s3_kg3 must be nonnegative")

        try:
            step_count = round(duration / time_step)
        except OverflowError as error:
            raise ValueError(
                "duration_s must be an integer number of time steps"
            ) from error
        if step_count <= 0 or not math.isclose(
            duration,
            step_count * time_step,
            rel_tol=_TIME_REL_TOLERANCE,
            abs_tol=_TIME_ABS_TOLERANCE_S,
        ):
            raise ValueError("duration_s must be an integer number of time steps")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "time_step_s", time_step)
        object.__setattr__(self, "thermal_limit_k", thermal_limit)
        object.__setattr__(self, "pump_power_coefficient_w_s3_kg3", coefficient)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))

    @property
    def step_count(self) -> int:
        """Number of fixed integration steps in this scenario."""
        return round(float(self.duration_s) / float(self.time_step_s))


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One post-step battery observation and its applied-step pump power."""

    state: BatteryState
    pump_power_w: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, BatteryState):
            raise ValueError("state must be a BatteryState")
        power = _finite_real("pump_power_w", self.pump_power_w)
        if power < 0.0:
            raise ValueError("pump_power_w must be nonnegative")
        object.__setattr__(self, "pump_power_w", power)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Immutable experiment trace and precomputed aggregate metrics."""

    scenario_name: str
    controller_name: str
    seed: int
    time_step_s: float
    records: tuple[ExperimentRecord, ...]
    metrics: ExperimentMetrics
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_name("scenario_name", self.scenario_name)
        _nonempty_name("controller_name", self.controller_name)
        seed = _normalize_seed(self.seed)
        time_step = _positive_finite("time_step_s", self.time_step_s)
        if not isinstance(self.records, (tuple, list)) or not self.records:
            raise ValueError(
                "records must be a nonempty sequence of ExperimentRecord values"
            )

        records = tuple(self.records)
        if any(not isinstance(record, ExperimentRecord) for record in records):
            raise ValueError("records must contain only ExperimentRecord values")

        expected_cell_count = records[0].state.cell_temperatures_k.size
        previous_time: float | None = None
        for record in records:
            state = record.state
            current_time = float(state.time_s)
            cell_count = state.cell_temperatures_k.size
            if cell_count != expected_cell_count:
                raise ValueError("records must have the same cell count")
            if previous_time is not None:
                elapsed = current_time - previous_time
                if elapsed <= 0.0:
                    raise ValueError("record state times must be strictly increasing")
                if not math.isclose(
                    elapsed,
                    time_step,
                    rel_tol=_TIME_REL_TOLERANCE,
                    abs_tol=_TIME_ABS_TOLERANCE_S,
                ):
                    raise ValueError(
                        "record state times must have fixed time_step_s spacing"
                    )
            previous_time = current_time

        if not isinstance(self.metrics, ExperimentMetrics):
            raise ValueError("metrics must be an ExperimentMetrics")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "time_step_s", time_step)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
