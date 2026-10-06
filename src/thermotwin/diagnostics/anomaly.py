"""Stateful detection of persistent per-cell thermal forecast residuals."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from thermotwin.core.types import BatteryState

FloatArray = NDArray[np.float64]


def _integer_at_least(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_copy(values: NDArray[np.float64]) -> FloatArray:
    """Copy an array into immutable bytes-backed storage."""
    return np.frombuffer(
        np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64
    )


def _validate_state_pair(
    measured: object, predicted: object
) -> tuple[BatteryState, BatteryState]:
    if not isinstance(measured, BatteryState):
        raise TypeError("measured must be a BatteryState")
    if not isinstance(predicted, BatteryState):
        raise TypeError("predicted must be a BatteryState")
    if measured.time_s != predicted.time_s:
        raise ValueError("measured and predicted states must have equal time_s")
    measured_temperatures = measured.cell_temperatures_k
    predicted_temperatures = predicted.cell_temperatures_k
    if measured_temperatures.shape != predicted_temperatures.shape:
        raise ValueError("measured and predicted states must have equal cell shapes")
    if measured_temperatures.ndim != 1 or measured_temperatures.size == 0:
        raise ValueError("states must contain a nonempty one-dimensional cell array")
    if not np.all(np.isfinite(measured_temperatures)) or not np.all(
        np.isfinite(predicted_temperatures)
    ):
        raise ValueError("cell temperatures must be finite")
    return measured, predicted


def calculate_temperature_residual(
    measured: BatteryState, predicted: BatteryState
) -> FloatArray:
    """Return measured minus forecast cell temperatures in kelvin."""
    measured_state, predicted_state = _validate_state_pair(measured, predicted)
    residual = measured_state.cell_temperatures_k - predicted_state.cell_temperatures_k
    if not np.all(np.isfinite(residual)):
        raise ValueError("temperature residuals must be finite")
    return _readonly_copy(residual)


@dataclass(frozen=True, slots=True)
class ThermalAnomalyConfig:
    """Validated rolling-window and persistence settings for anomaly detection."""

    rolling_window_steps: int = 10
    score_threshold: float = 3.0
    persistence_steps: int = 3
    epsilon_k: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rolling_window_steps",
            _integer_at_least("rolling_window_steps", self.rolling_window_steps, 2),
        )
        object.__setattr__(
            self,
            "persistence_steps",
            _integer_at_least("persistence_steps", self.persistence_steps, 2),
        )
        for name in ("score_threshold", "epsilon_k"):
            if _finite_real(name, getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class AnomalyEvent:
    """First persistent anomalous forecast residual for one cell."""

    cell_index: int
    anomaly_score: float
    detection_time_s: float
    predicted_hotspot_temperature_k: float
    severity_k: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cell_index",
            _integer_at_least("cell_index", self.cell_index, 0),
        )
        if _finite_real("anomaly_score", self.anomaly_score) < 0.0:
            raise ValueError("anomaly_score must be nonnegative")
        if _finite_real("detection_time_s", self.detection_time_s) < 0.0:
            raise ValueError("detection_time_s must be nonnegative")
        if (
            _finite_real(
                "predicted_hotspot_temperature_k", self.predicted_hotspot_temperature_k
            )
            <= 0.0
        ):
            raise ValueError("predicted_hotspot_temperature_k must be positive")
        if _finite_real("severity_k", self.severity_k) < 0.0:
            raise ValueError("severity_k must be nonnegative")


class ThermalAnomalyDetector:
    """Calibrate from prior residuals and emit once after persistent exceedance."""

    def __init__(self, config: ThermalAnomalyConfig | None = None) -> None:
        normalized_config = config if config is not None else ThermalAnomalyConfig()
        if not isinstance(normalized_config, ThermalAnomalyConfig):
            raise TypeError("config must be a ThermalAnomalyConfig")
        self._config = normalized_config
        self.reset()

    @property
    def config(self) -> ThermalAnomalyConfig:
        """Return the immutable detector configuration."""
        return self._config

    def reset(self) -> None:
        """Clear calibration, persistence, and diagnostic history."""
        self._residual_windows: list[deque[float]] = []
        self._cell_count: int | None = None
        self._last_time_s: float | None = None
        self._latest_residual: FloatArray | None = None
        self._latest_scores: FloatArray | None = None
        self._consecutive_exceedances: list[int] = []
        self._events: list[AnomalyEvent] = []
        self._detected_cells: set[int] = set()

    @property
    def latest_residual_k(self) -> FloatArray | None:
        """Return a defensive copy of the most recent residual vector."""
        if self._latest_residual is None:
            return None
        return np.array(self._latest_residual, copy=True)

    @property
    def latest_scores(self) -> FloatArray | None:
        """Return a defensive copy of the most recently calculated scores."""
        if self._latest_scores is None:
            return None
        return np.array(self._latest_scores, copy=True)

    @property
    def events(self) -> tuple[AnomalyEvent, ...]:
        """Return immutable first-detection events in detection order."""
        return tuple(self._events)

    def update(
        self, measured: BatteryState, predicted: BatteryState
    ) -> tuple[AnomalyEvent, ...]:
        """Score a matched state pair, then add its residual to calibration."""
        measured_state, predicted_state = _validate_state_pair(measured, predicted)
        if self._last_time_s is not None and measured_state.time_s <= self._last_time_s:
            raise ValueError("state times must increase strictly between updates")
        residual = calculate_temperature_residual(measured_state, predicted_state)
        cell_count = int(residual.size)
        if self._cell_count is not None and cell_count != self._cell_count:
            raise ValueError("cell count must remain constant between updates")

        self._cell_count = cell_count
        self._last_time_s = measured_state.time_s
        self._latest_residual = residual
        emitted: list[AnomalyEvent] = []
        window_size = self.config.rolling_window_steps
        if not self._residual_windows:
            self._residual_windows = [
                deque(maxlen=window_size) for _ in range(cell_count)
            ]
            self._consecutive_exceedances = [0] * cell_count
        calibrated = all(
            len(window) == window_size for window in self._residual_windows
        )
        if calibrated:
            prior = np.asarray([tuple(window) for window in self._residual_windows])
            mean = np.mean(prior, axis=1)
            standard_deviation = np.std(prior, axis=1)
            scores = np.abs(residual - mean) / (
                standard_deviation + self.config.epsilon_k
            )
            if not np.all(np.isfinite(scores)):
                raise ValueError("anomaly scores must be finite")
            self._latest_scores = _readonly_copy(scores)
            for cell_index, score_value in enumerate(scores):
                score = float(score_value)
                # Exclude threshold exceedances from baseline calibration so
                # persistent faults cannot absorb themselves into the window.
                if score > self.config.score_threshold:
                    self._consecutive_exceedances[cell_index] += 1
                    if (
                        self._consecutive_exceedances[cell_index]
                        >= self.config.persistence_steps
                        and cell_index not in self._detected_cells
                    ):
                        event = AnomalyEvent(
                            cell_index=cell_index,
                            anomaly_score=score,
                            detection_time_s=measured_state.time_s,
                            predicted_hotspot_temperature_k=float(
                                predicted_state.cell_temperatures_k[cell_index]
                            ),
                            severity_k=abs(float(residual[cell_index])),
                        )
                        self._events.append(event)
                        emitted.append(event)
                        self._detected_cells.add(cell_index)
                else:
                    self._consecutive_exceedances[cell_index] = 0
                    self._residual_windows[cell_index].append(
                        float(residual[cell_index])
                    )
        else:
            self._latest_scores = None
            for cell_index, value in enumerate(residual):
                self._residual_windows[cell_index].append(float(value))
        return tuple(emitted)
