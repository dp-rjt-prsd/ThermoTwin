"""Immutable contracts for the interactive ThermoTwin dashboard."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CurrentProfile,
    EnvironmentConfig,
)
from thermotwin.experiments.types import MetadataValue


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _positive(name: str, value: object) -> float:
    normalized = _finite(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _mapping(
    value: Mapping[str, MetadataValue] | None, name: str
) -> Mapping[str, MetadataValue]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, MetadataValue] = {}
    for key, item in value.items():
        normalized_key = _nonempty(f"{name} key", key)
        if not isinstance(item, (str, int, float, bool)) and item is not None:
            raise ValueError(f"{name} values must be scalar or None")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} values must be finite")
        normalized[normalized_key] = item
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """Validated user inputs for one deterministic interactive simulation."""

    scenario_name: str
    environment_config: EnvironmentConfig
    current_profile: CurrentProfile
    controller_name: str
    model_name: str | None
    duration_s: float
    time_step_s: float
    seed: int
    metadata: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_name", _nonempty("scenario_name", self.scenario_name)
        )
        if not isinstance(self.environment_config, EnvironmentConfig):
            raise ValueError("environment_config must be an EnvironmentConfig")
        if not callable(self.current_profile):
            raise ValueError("current_profile must be callable")
        object.__setattr__(
            self, "controller_name", _nonempty("controller_name", self.controller_name)
        )
        if self.model_name is not None:
            object.__setattr__(
                self, "model_name", _nonempty("model_name", self.model_name)
            )
        object.__setattr__(self, "duration_s", _positive("duration_s", self.duration_s))
        object.__setattr__(
            self, "time_step_s", _positive("time_step_s", self.time_step_s)
        )
        step_count = round(self.duration_s / self.time_step_s)
        if step_count <= 0 or not math.isclose(
            self.duration_s,
            step_count * self.time_step_s,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("duration_s must be an integer number of time steps")
        object.__setattr__(self, "seed", _seed(self.seed))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def step_count(self) -> int:
        """Number of fixed integration steps requested by the user."""
        return round(self.duration_s / self.time_step_s)


@dataclass(frozen=True, slots=True)
class SimulationSnapshot:
    """One live simulator observation and optional model forecast."""

    state: BatteryState
    control_action: CoolingAction
    forecast_states: tuple[BatteryState, ...] = ()
    thermal_terms_w: Mapping[str, float] = field(default_factory=dict)
    diagnostics: Mapping[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, BatteryState):
            raise ValueError("state must be a BatteryState")
        if not isinstance(self.control_action, CoolingAction):
            raise ValueError("control_action must be a CoolingAction")
        if type(self.forecast_states) is not tuple:
            raise ValueError("forecast_states must be a tuple")
        if any(not isinstance(state, BatteryState) for state in self.forecast_states):
            raise ValueError("forecast_states must contain BatteryState values")
        terms: dict[str, float] = {}
        for key, value in self.thermal_terms_w.items():
            terms[_nonempty("thermal term key", key)] = _finite(
                f"thermal_terms_w[{key!r}]", value
            )
        object.__setattr__(self, "thermal_terms_w", MappingProxyType(terms))
        object.__setattr__(
            self, "diagnostics", _mapping(self.diagnostics, "diagnostics")
        )


@dataclass(frozen=True, slots=True)
class ControllerComparison:
    """Synchronized live-run metrics with an explicit reference run."""

    run_ids: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float | int]]
    reference_run_id: str
    evidence_note: str

    def __post_init__(self) -> None:
        if not self.run_ids or any(
            not isinstance(run_id, str) or not run_id.strip() for run_id in self.run_ids
        ):
            raise ValueError("run_ids must contain nonempty strings")
        run_ids = tuple(run_id.strip() for run_id in self.run_ids)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids must be unique")
        if self.reference_run_id not in run_ids:
            raise ValueError("reference_run_id must identify one run")
        if not isinstance(self.evidence_note, str) or not self.evidence_note.strip():
            raise ValueError("evidence_note must be nonempty")
        normalized: dict[str, Mapping[str, float | int]] = {}
        for run_id in run_ids:
            if run_id not in self.metrics:
                raise ValueError(f"metrics missing run {run_id!r}")
            values: dict[str, float | int] = {}
            for key, value in self.metrics[run_id].items():
                values[_nonempty("metric key", key)] = _finite(
                    f"metrics[{run_id!r}][{key!r}]", value
                )
            normalized[run_id] = MappingProxyType(values)
        object.__setattr__(self, "run_ids", run_ids)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))
        object.__setattr__(self, "reference_run_id", self.reference_run_id.strip())
        object.__setattr__(self, "evidence_note", self.evidence_note.strip())


@dataclass(frozen=True, slots=True)
class FormulaSpec:
    """Display metadata for one authoritative physical or evaluation formula."""

    formula_id: str
    title: str
    latex: str
    explanation: str
    symbols: Mapping[str, str]
    units: Mapping[str, str]
    approximation_note: str | None = None
    implementation_provenance: str = ""

    def __post_init__(self) -> None:
        for name in ("formula_id", "title", "latex", "explanation"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(self, "symbols", _mapping(self.symbols, "symbols"))
        object.__setattr__(self, "units", _mapping(self.units, "units"))
        if self.approximation_note is not None:
            object.__setattr__(
                self,
                "approximation_note",
                _nonempty("approximation_note", self.approximation_note),
            )
        object.__setattr__(
            self,
            "implementation_provenance",
            self.implementation_provenance.strip(),
        )


@dataclass(frozen=True, slots=True)
class FormulaEvaluation:
    """Live substituted terms and result for one formula at one snapshot."""

    formula_id: str
    snapshot_time_s: float
    substituted_terms: Mapping[str, float]
    result: float | None
    validity_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "formula_id", _nonempty("formula_id", self.formula_id))
        object.__setattr__(
            self,
            "snapshot_time_s",
            _finite("snapshot_time_s", self.snapshot_time_s, minimum=0.0),
        )
        terms: dict[str, float] = {}
        for key, value in self.substituted_terms.items():
            terms[_nonempty("substituted term key", key)] = _finite(
                f"substituted_terms[{key!r}]", value
            )
        object.__setattr__(self, "substituted_terms", MappingProxyType(terms))
        if self.result is not None:
            object.__setattr__(self, "result", _finite("result", self.result))
        if self.validity_reason is not None:
            object.__setattr__(
                self,
                "validity_reason",
                _nonempty("validity_reason", self.validity_reason),
            )


@dataclass(frozen=True, slots=True)
class PresentationStep:
    """One guided-demo step and the visual/formula content it activates."""

    step_id: str
    title: str
    explanation: str
    visual_id: str
    formula_ids: tuple[str, ...] = ()
    preset_transition: str | None = None

    def __post_init__(self) -> None:
        for name in ("step_id", "title", "explanation", "visual_id"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if type(self.formula_ids) is not tuple or any(
            not isinstance(value, str) or not value.strip()
            for value in self.formula_ids
        ):
            raise ValueError("formula_ids must be a tuple of nonempty strings")
        if self.preset_transition is not None:
            object.__setattr__(
                self,
                "preset_transition",
                _nonempty("preset_transition", self.preset_transition),
            )


__all__ = [
    "ControllerComparison",
    "FormulaEvaluation",
    "FormulaSpec",
    "PresentationStep",
    "SimulationRequest",
    "SimulationSnapshot",
]
