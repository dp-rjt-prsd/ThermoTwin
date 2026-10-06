"""Immutable DTOs returned by dashboard artifact parsers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from thermotwin.visualization.artifact_io import finite, frozen_array, metric_mapping

_HORIZONS = (30.0, 60.0, 120.0)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ControllerTrace:
    """One controller's measured time series and immutable saved metrics."""

    label: str
    time_s: FloatArray
    current_a: FloatArray
    soc_fraction: FloatArray
    ambient_temperature_k: FloatArray
    coolant_inlet_temperature_k: FloatArray
    coolant_mass_flow_kg_s: FloatArray
    pump_power_w: FloatArray
    cell_temperatures_k: FloatArray
    cell_resistances_ohm: FloatArray
    metrics: Mapping[str, float | int]

    def __post_init__(self) -> None:
        vector_names = (
            "time_s",
            "current_a",
            "soc_fraction",
            "ambient_temperature_k",
            "coolant_inlet_temperature_k",
            "coolant_mass_flow_kg_s",
            "pump_power_w",
        )
        vectors = {
            name: frozen_array(getattr(self, name), name) for name in vector_names
        }
        length = vectors["time_s"].size
        if length == 0 or any(value.shape != (length,) for value in vectors.values()):
            raise ValueError(
                "controller trace vectors must have one shared nonzero length"
            )
        temperatures = frozen_array(self.cell_temperatures_k, "cell_temperatures_k")
        resistances = frozen_array(self.cell_resistances_ohm, "cell_resistances_ohm")
        if temperatures.shape != (length, 12) or resistances.shape != (length, 12):
            raise ValueError("controller trace cell matrices must have shape (N, 12)")
        if np.any(temperatures <= 0.0) or np.any(resistances <= 0.0):
            raise ValueError("controller temperatures/resistances must be positive")
        if np.any(np.diff(vectors["time_s"]) <= 0.0):
            raise ValueError("controller trace time_s must increase strictly")
        if np.any(vectors["coolant_mass_flow_kg_s"] < 0.0) or np.any(
            vectors["pump_power_w"] < 0.0
        ):
            raise ValueError(
                "controller coolant flow and pump power must be nonnegative"
            )
        if np.any(vectors["soc_fraction"] < 0.0) or np.any(
            vectors["soc_fraction"] > 1.0
        ):
            raise ValueError("controller soc_fraction must be in [0, 1]")
        for name, value in vectors.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "cell_temperatures_k", temperatures)
        object.__setattr__(self, "cell_resistances_ohm", resistances)
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(metric_mapping(dict(self.metrics), "metrics"))),
        )


@dataclass(frozen=True, slots=True)
class ForecastSnapshot:
    """One cell-temperature prediction at an exact forecast horizon."""

    variant: str
    horizon_s: float
    source_time_s: float
    source_cell_temperatures_k: FloatArray
    predicted_cell_temperatures_k: FloatArray
    action_coolant_mass_flow_kg_s: float

    def __post_init__(self) -> None:
        for name in ("source_cell_temperatures_k", "predicted_cell_temperatures_k"):
            values = frozen_array(getattr(self, name), name)
            if values.shape != (12,) or np.any(values <= 0.0):
                raise ValueError(f"{name} must contain 12 positive finite values")
            object.__setattr__(self, name, values)
        for name in ("horizon_s", "source_time_s", "action_coolant_mass_flow_kg_s"):
            object.__setattr__(
                self, name, finite(getattr(self, name), name, minimum=0.0)
            )
        if self.horizon_s not in _HORIZONS:
            raise ValueError("forecast horizon_s must be 30, 60, or 120")


@dataclass(frozen=True, slots=True)
class FaultEvent:
    """Persisted anomaly event."""

    cell_index: int
    anomaly_score: float
    detection_time_s: float
    predicted_hotspot_temperature_k: float
    severity_k: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.cell_index, bool)
            or not isinstance(self.cell_index, int)
            or not 0 <= self.cell_index < 12
        ):
            raise ValueError("fault event cell_index must be in [0, 11]")
        for name in ("anomaly_score", "detection_time_s", "severity_k"):
            object.__setattr__(
                self, name, finite(getattr(self, name), name, minimum=0.0)
            )
        temperature = finite(
            self.predicted_hotspot_temperature_k,
            "predicted_hotspot_temperature_k",
            minimum=0.0,
        )
        if temperature <= 0.0:
            raise ValueError("predicted_hotspot_temperature_k must be positive")
        object.__setattr__(self, "predicted_hotspot_temperature_k", temperature)


@dataclass(frozen=True, slots=True)
class FaultTrace:
    """Measured/predicted fault trace; NaN score rows mean detector warmup."""

    time_s: FloatArray
    measured_temperature_k: FloatArray
    predicted_temperature_k: FloatArray
    residual_k: FloatArray
    scores: FloatArray
    coolant_mass_flow_kg_s: FloatArray
    pump_power_w: FloatArray

    def __post_init__(self) -> None:
        time = frozen_array(self.time_s, "fault.time_s")
        if time.ndim != 1 or time.size == 0 or np.any(np.diff(time) <= 0.0):
            raise ValueError("fault.time_s must be nonempty and increase strictly")
        for name in (
            "measured_temperature_k",
            "predicted_temperature_k",
            "residual_k",
            "scores",
        ):
            matrix = frozen_array(
                getattr(self, name), name, allow_nan=(name == "scores")
            )
            if matrix.shape != (time.size, 12):
                raise ValueError(f"fault.{name} must have shape (N, 12)")
            if name in ("measured_temperature_k", "predicted_temperature_k") and np.any(
                matrix <= 0.0
            ):
                raise ValueError(f"fault.{name} must be positive")
            if name == "scores" and (
                np.any(matrix < 0.0)
                or np.any(
                    np.any(np.isnan(matrix), axis=1) & ~np.all(np.isnan(matrix), axis=1)
                )
            ):
                raise ValueError(
                    "fault.scores must be nonnegative or a null warmup row"
                )
            object.__setattr__(self, name, matrix)
        for name in ("coolant_mass_flow_kg_s", "pump_power_w"):
            vector = frozen_array(getattr(self, name), name)
            if vector.shape != time.shape or np.any(vector < 0.0):
                raise ValueError(
                    f"fault.{name} must be nonnegative and match trace length"
                )
            object.__setattr__(self, name, vector)
        object.__setattr__(self, "time_s", time)


@dataclass(frozen=True, slots=True)
class FaultSummary:
    """Measured anomaly outcome, with injected fault truth kept separately."""

    model_variant: str
    fault_cell_index: int
    fault_start_time_s: float
    thermal_limit_k: float
    threshold_crossing_time_s: float | None
    detection_time_s: float | None
    lead_time_s: float | None
    detected_fault_cell: bool
    peak_temperature_k: float
    temperature_spread_k: float
    pump_energy_j: float
    thermal_violation_count: int
    anomaly_event_count: int
    events: tuple[FaultEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        if len(self.events) != self.anomaly_event_count:
            raise ValueError("fault summary event count does not match events")


@dataclass(frozen=True, slots=True)
class DashboardData:
    """All validated dashboard-facing artifacts, detached from source files."""

    controller_traces: Mapping[str, ControllerTrace]
    model_forecasts: Mapping[str, tuple[ForecastSnapshot, ...]]
    fault_trace: FaultTrace
    fault_summary: FaultSummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "controller_traces", MappingProxyType(dict(self.controller_traces))
        )
        object.__setattr__(
            self, "model_forecasts", MappingProxyType(dict(self.model_forecasts))
        )


__all__ = [
    "ControllerTrace",
    "DashboardData",
    "FaultEvent",
    "FaultSummary",
    "FaultTrace",
    "ForecastSnapshot",
]
