"""Held-out trajectory evaluation at exact thermal forecast horizons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from thermotwin.core.types import BatteryState
from thermotwin.data.types import TransitionRecord
from thermotwin.models.evaluation import RegressionMetrics, compute_regression_metrics
from thermotwin.models.forecaster import (
    ForecastConditions,
    ForecastRollout,
    ThermalForecaster,
)

_TIME_REL_TOLERANCE = 1e-12
_TIME_ABS_TOLERANCE_S = 1e-12


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite positive real number") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return normalized


@dataclass(frozen=True, slots=True)
class ForecastHorizonMetrics:
    """Physical-unit metrics for one exact horizon and its held-out forecasts."""

    horizon_s: float
    step_count: int
    forecast_count: int
    metrics: RegressionMetrics

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "horizon_s", _positive_finite("horizon_s", self.horizon_s)
        )
        if type(self.step_count) is not int or self.step_count <= 0:
            raise ValueError("step_count must be a positive integer")
        if type(self.forecast_count) is not int or self.forecast_count <= 0:
            raise ValueError("forecast_count must be a positive integer")
        if type(self.metrics) is not RegressionMetrics:
            raise ValueError("metrics must be RegressionMetrics")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation."""
        return {
            "horizon_s": self.horizon_s,
            "step_count": self.step_count,
            "forecast_count": self.forecast_count,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ForecastEvaluation:
    """Immutable ordered collection of forecast metrics across horizons."""

    horizons: tuple[ForecastHorizonMetrics, ...]

    def __post_init__(self) -> None:
        if type(self.horizons) is not tuple or not self.horizons:
            raise ValueError("horizons must be a nonempty tuple")
        if any(type(item) is not ForecastHorizonMetrics for item in self.horizons):
            raise ValueError("horizons must contain ForecastHorizonMetrics")
        horizon_seconds = tuple(item.horizon_s for item in self.horizons)
        if any(
            right <= left
            for left, right in zip(horizon_seconds, horizon_seconds[1:], strict=False)
        ):
            raise ValueError("horizon metrics must be strictly increasing")
        forecast_counts = {item.forecast_count for item in self.horizons}
        cell_counts = {len(item.metrics.per_cell_mae_kelvin) for item in self.horizons}
        if len(forecast_counts) != 1:
            raise ValueError("all horizons must have a consistent forecast count")
        if len(cell_counts) != 1:
            raise ValueError("all horizons must have a consistent cell count")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready results indexed by horizon seconds."""
        return {
            "horizons": {str(item.horizon_s): item.to_dict() for item in self.horizons}
        }


def _same_state(left: BatteryState, right: BatteryState) -> bool:
    return (
        left.time_s == right.time_s
        and left.current_a == right.current_a
        and left.soc_fraction == right.soc_fraction
        and left.ambient_temperature_k == right.ambient_temperature_k
        and left.coolant_inlet_temperature_k == right.coolant_inlet_temperature_k
        and left.coolant_mass_flow_kg_s == right.coolant_mass_flow_kg_s
        and np.array_equal(left.cell_temperatures_k, right.cell_temperatures_k)
        and np.array_equal(left.cell_resistances_ohm, right.cell_resistances_ohm)
    )


def _group_and_validate_records(
    records: tuple[TransitionRecord, ...], forecaster: ThermalForecaster
) -> tuple[tuple[TransitionRecord, ...], ...]:
    if type(records) is not tuple or not records:
        raise ValueError("records must be a nonempty immutable tuple")
    if any(not isinstance(record, TransitionRecord) for record in records):
        raise ValueError("records must contain TransitionRecord values")

    time_step = _positive_finite("forecaster.time_step_s", forecaster.time_step_s)
    cell_count = forecaster.cell_count
    if type(cell_count) is not int or cell_count <= 0:
        raise ValueError("forecaster.cell_count must be a positive integer")

    grouped: dict[str, list[TransitionRecord]] = {}
    for record in records:
        grouped.setdefault(record.trajectory_id, []).append(record)

    result: list[tuple[TransitionRecord, ...]] = []
    for trajectory_id, trajectory_records in grouped.items():
        prior: TransitionRecord | None = None
        for expected_index, record in enumerate(trajectory_records):
            if record.step_index != expected_index:
                raise ValueError(
                    f"trajectory {trajectory_id!r} step indices must be zero-based "
                    "and contiguous in record order"
                )
            if not math.isclose(
                record.time_step_s,
                time_step,
                rel_tol=_TIME_REL_TOLERANCE,
                abs_tol=_TIME_ABS_TOLERANCE_S,
            ):
                raise ValueError(
                    f"trajectory {trajectory_id!r} timestep must equal the "
                    "forecaster timestep"
                )
            for state in (record.state, record.next_state):
                if state.cell_temperatures_k.size != cell_count:
                    raise ValueError(
                        f"trajectory {trajectory_id!r} cell count does not match "
                        "the forecaster"
                    )
            if prior is not None and not _same_state(prior.next_state, record.state):
                raise ValueError(
                    f"trajectory {trajectory_id!r} state chain is not exactly "
                    "continuous"
                )
            prior = record
        result.append(tuple(trajectory_records))
    return tuple(result)


def evaluate_forecast_horizons(
    forecaster: ThermalForecaster,
    records: tuple[TransitionRecord, ...],
    horizons_s: tuple[float, ...] = (30.0, 60.0, 120.0),
) -> ForecastEvaluation:
    """Evaluate final-state predictions from one origin per held-out trajectory.

    Each rollout receives only recorded actions and post-action exogenous
    conditions. Observed future temperatures are used solely as final targets.
    """
    if not isinstance(forecaster, ThermalForecaster):
        raise TypeError("forecaster must conform to ThermalForecaster")
    if type(horizons_s) is not tuple or not horizons_s:
        raise ValueError("horizons_s must be a nonempty immutable tuple")
    normalized_horizons = tuple(
        _positive_finite(f"horizons_s[{index}]", value)
        for index, value in enumerate(horizons_s)
    )
    if any(
        right <= left
        for left, right in zip(
            normalized_horizons, normalized_horizons[1:], strict=False
        )
    ):
        raise ValueError("horizons_s must be unique and strictly increasing")

    trajectories = _group_and_validate_records(records, forecaster)
    time_step = float(forecaster.time_step_s)
    step_counts: list[int] = []
    for horizon in normalized_horizons:
        ratio = horizon / time_step
        steps = round(ratio)
        if steps <= 0 or not math.isclose(
            horizon,
            steps * time_step,
            rel_tol=_TIME_REL_TOLERANCE,
            abs_tol=_TIME_ABS_TOLERANCE_S,
        ):
            raise ValueError(
                f"horizon {horizon:g} s must be an exact integer multiple of "
                "the forecaster timestep"
            )
        step_counts.append(steps)

    largest_step_count = step_counts[-1]
    for trajectory in trajectories:
        if len(trajectory) < largest_step_count:
            raise ValueError(
                f"every trajectory must contain at least {largest_step_count} "
                "records for the largest horizon"
            )

    horizon_results: list[ForecastHorizonMetrics] = []
    for horizon, step_count in zip(normalized_horizons, step_counts, strict=True):
        predictions: list[NDArray[np.float64]] = []
        observations: list[NDArray[np.float64]] = []
        for trajectory in trajectories:
            selected_records = trajectory[:step_count]
            actions = tuple(record.action for record in selected_records)
            conditions = tuple(
                ForecastConditions.from_state(record.next_state)
                for record in selected_records
            )
            rollout = forecaster.rollout(
                trajectory[0].state,
                actions,
                future_conditions=conditions,
            )
            if not isinstance(rollout, ForecastRollout):
                raise ValueError("forecaster.rollout must return ForecastRollout")
            if len(rollout.states) != step_count:
                raise ValueError("forecaster rollout length does not match horizon")
            predicted = rollout.states[-1].cell_temperatures_k
            observed = selected_records[-1].next_state.cell_temperatures_k
            if predicted.size != forecaster.cell_count:
                raise ValueError("forecast cell count does not match forecaster")
            predictions.append(np.asarray(predicted, dtype=np.float64))
            observations.append(np.asarray(observed, dtype=np.float64))
        horizon_results.append(
            ForecastHorizonMetrics(
                horizon_s=horizon,
                step_count=step_count,
                forecast_count=len(trajectories),
                metrics=compute_regression_metrics(
                    np.stack(predictions), np.stack(observations)
                ),
            )
        )
    return ForecastEvaluation(tuple(horizon_results))
