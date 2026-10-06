"""Measured thermal-threshold and anomaly lead-time metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from thermotwin.core.types import BatteryState
from thermotwin.diagnostics.anomaly import AnomalyEvent


def _require_states(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("states must be a sequence of BatteryState values")
    if len(value) == 0:
        raise ValueError("states must be nonempty")
    return value


def _finite_positive_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _optional_cell_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("cell_index must be a nonnegative integer or None")
    return int(value)


def _finite_nonnegative_time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative real number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_signed_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class FaultLeadTimeMetric:
    """Immutable anomaly timing relative to measured threshold crossing."""

    detection_time_s: float | None
    threshold_crossing_time_s: float | None
    lead_time_s: float | None

    def __post_init__(self) -> None:
        detection = self.detection_time_s
        crossing = self.threshold_crossing_time_s
        lead = self.lead_time_s
        if detection is not None:
            detection = _finite_nonnegative_time("detection_time_s", detection)
            object.__setattr__(self, "detection_time_s", detection)
        if crossing is not None:
            crossing = _finite_nonnegative_time("threshold_crossing_time_s", crossing)
            object.__setattr__(self, "threshold_crossing_time_s", crossing)

        if detection is None or crossing is None:
            if lead is not None:
                raise ValueError(
                    "lead_time_s must be None when either event time is unavailable"
                )
            return

        if lead is None:
            raise ValueError(
                "lead_time_s is required when both event times are available"
            )
        normalized_lead = _finite_signed_real("lead_time_s", lead)
        expected_lead = crossing - detection
        if not math.isfinite(expected_lead) or normalized_lead != expected_lead:
            raise ValueError(
                "lead_time_s must equal threshold_crossing_time_s - detection_time_s"
            )
        object.__setattr__(self, "lead_time_s", normalized_lead)


def first_threshold_crossing_time_s(
    states: Sequence[BatteryState],
    thermal_limit_k: float,
    cell_index: int | None = None,
) -> float | None:
    """Return first observed strict threshold crossing, or ``None`` if absent.

    Crossing is calculated only from measured ``BatteryState`` cell
    temperatures. The full sequence is validated before any crossing is
    returned, so malformed samples after a crossing cannot be ignored.
    """
    state_values = _require_states(states)
    thermal_limit = _finite_positive_real("thermal_limit_k", thermal_limit_k)
    cell_index = _optional_cell_index(cell_index)

    expected_cell_count: int | None = None
    previous_time_s: float | None = None
    first_crossing: float | None = None
    for state in state_values:
        if not isinstance(state, BatteryState):
            raise TypeError("states must contain BatteryState values")
        temperatures = state.cell_temperatures_k
        if temperatures.ndim != 1 or temperatures.size == 0:
            raise ValueError(
                "each state must have a nonempty one-dimensional cell array"
            )
        if expected_cell_count is None:
            expected_cell_count = int(temperatures.size)
        elif temperatures.size != expected_cell_count:
            raise ValueError("states must have a constant cell count")
        if not np.all(np.isfinite(temperatures)):
            raise ValueError("cell temperatures must be finite")

        time_s = float(state.time_s)
        if previous_time_s is not None and time_s <= previous_time_s:
            raise ValueError("state times must increase strictly")
        previous_time_s = time_s

        if cell_index is not None and cell_index >= temperatures.size:
            raise ValueError("cell_index must be within the state cell count")
        if first_crossing is None:
            observed_temperature = (
                float(temperatures[cell_index])
                if cell_index is not None
                else float(np.max(temperatures))
            )
            if observed_temperature > thermal_limit:
                first_crossing = time_s
    return first_crossing


def calculate_fault_lead_time_metric(
    anomaly_event: AnomalyEvent | None,
    threshold_crossing_time_s: float | None,
) -> FaultLeadTimeMetric:
    """Combine detector event timing with a crossing measured from states."""
    if anomaly_event is not None and not isinstance(anomaly_event, AnomalyEvent):
        raise TypeError("anomaly_event must be an AnomalyEvent or None")
    detection_time = None if anomaly_event is None else anomaly_event.detection_time_s
    if detection_time is None or threshold_crossing_time_s is None:
        return FaultLeadTimeMetric(detection_time, threshold_crossing_time_s, None)
    detection = _finite_nonnegative_time("detection_time_s", detection_time)
    crossing = _finite_nonnegative_time(
        "threshold_crossing_time_s", threshold_crossing_time_s
    )
    return FaultLeadTimeMetric(detection, crossing, crossing - detection)
