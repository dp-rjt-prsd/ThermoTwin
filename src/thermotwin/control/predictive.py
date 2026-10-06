"""Receding-horizon coolant selection over a thermal forecaster."""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from thermotwin.control.objective import evaluate_predictive_candidate
from thermotwin.control.predictive_types import (
    PredictiveCandidateEvaluation,
    PredictiveControllerConfig,
)
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.models.forecaster import ForecastRollout, ThermalForecaster


class PredictiveController:
    """Evaluate constant-flow candidates and apply the best first action."""

    _forecaster: ThermalForecaster
    _config: PredictiveControllerConfig
    _candidate_flows: tuple[float, ...]
    _cell_count: int
    _last_candidate_evaluations: tuple[PredictiveCandidateEvaluation, ...] | None
    _selected_evaluation: PredictiveCandidateEvaluation | None

    def __init__(
        self,
        forecaster: ThermalForecaster,
        config: PredictiveControllerConfig,
    ) -> None:
        if not isinstance(forecaster, ThermalForecaster):
            raise TypeError("forecaster must satisfy the ThermalForecaster protocol")
        if type(config) is not PredictiveControllerConfig:
            raise TypeError("config must be a PredictiveControllerConfig")

        cell_count = _positive_cell_count(forecaster.cell_count)
        _positive_time_step(forecaster.time_step_s)

        self._forecaster = forecaster
        self._config = config
        self._cell_count = int(cell_count)
        self._candidate_flows = tuple(
            float(flow)
            for flow in np.linspace(
                config.minimum_coolant_mass_flow_kg_s,
                config.maximum_coolant_mass_flow_kg_s,
                config.candidate_count,
            )
        )
        self._last_candidate_evaluations = None
        self._selected_evaluation = None

    @property
    def last_candidate_evaluations(
        self,
    ) -> tuple[PredictiveCandidateEvaluation, ...] | None:
        """Immutable candidate diagnostics from the most recent successful call."""
        return self._last_candidate_evaluations

    @property
    def selected_evaluation(self) -> PredictiveCandidateEvaluation | None:
        """Chosen candidate diagnostics, or ``None`` before a successful call."""
        return self._selected_evaluation

    def reset(self) -> None:
        """Clear per-decision diagnostics before a new trajectory begins."""
        self._last_candidate_evaluations = None
        self._selected_evaluation = None

    def act(self, state: BatteryState) -> CoolingAction:
        """Forecast each bounded flow and return the lowest-cost first action."""
        self._last_candidate_evaluations = None
        self._selected_evaluation = None
        if not isinstance(state, BatteryState):
            raise TypeError("state must be a BatteryState")
        if state.cell_temperatures_k.size != self._cell_count:
            raise ValueError(f"state cell count must be {self._cell_count}")

        evaluations: list[PredictiveCandidateEvaluation] = []
        for flow in self._candidate_flows:
            action = CoolingAction(flow)
            actions = (action,) * self._config.horizon_steps
            rollout = self._forecaster.rollout(state, actions)
            if type(rollout) is not ForecastRollout:
                raise TypeError("forecaster rollout must be a ForecastRollout")
            if not _same_state(rollout.initial_state, state):
                raise ValueError(
                    "rollout initial state must match the requested state"
                )
            if rollout.actions != actions:
                raise ValueError(
                    "rollout candidate actions must match requested actions"
                )
            if rollout.step_count != self._config.horizon_steps:
                raise ValueError("rollout step count must match configured horizon")
            if rollout.cell_count != self._cell_count:
                raise ValueError("rollout cell count must match forecaster cell_count")
            evaluations.append(evaluate_predictive_candidate(rollout, self._config))

        candidate_evaluations = tuple(evaluations)
        selected = min(
            candidate_evaluations,
            key=lambda evaluation: (
                evaluation.total_cost,
                evaluation.action.coolant_mass_flow_kg_s,
            ),
        )
        self._last_candidate_evaluations = candidate_evaluations
        self._selected_evaluation = selected
        return selected.action


__all__ = ["PredictiveController"]


def _positive_cell_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("forecaster cell_count must be a positive integer")
    return int(value)


def _positive_time_step(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("forecaster time_step_s must be finite and positive")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "forecaster time_step_s must be finite and positive"
        ) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("forecaster time_step_s must be finite and positive")
    return result


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
