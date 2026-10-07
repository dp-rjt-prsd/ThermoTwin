"""Immutable, validated contracts for the Phase 12 benchmark matrix."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, TypeAlias

ScenarioId: TypeAlias = Literal[
    "normal",
    "high_current",
    "hot_ambient",
    "weak_cooling",
    "fast_charging",
    "resistance_fault",
]
VariantId: TypeAlias = Literal[
    "threshold",
    "pid",
    "predictive_data_only",
    "predictive_physics_informed",
]
ModelVariant: TypeAlias = Literal["data_only", "physics_informed"]
TrainingMode: TypeAlias = Literal["smoke", "full"]
BenchmarkMode: TypeAlias = Literal["quick", "full"]
EvaluationDistribution: TypeAlias = Literal["in_distribution", "out_of_distribution"]
PersistentInputAssumption: TypeAlias = Literal["current_value_persistence"]
ActionHandling: TypeAlias = Literal["constant_candidate_sequence"]

SCENARIO_IDS: tuple[ScenarioId, ...] = (
    "normal",
    "high_current",
    "hot_ambient",
    "weak_cooling",
    "fast_charging",
    "resistance_fault",
)
VARIANT_IDS: tuple[VariantId, ...] = (
    "threshold",
    "pid",
    "predictive_data_only",
    "predictive_physics_informed",
)
FORECAST_HORIZONS_S: tuple[float, ...] = (30.0, 60.0, 120.0)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_TIME_REL_TOLERANCE = 1e-12
_TIME_ABS_TOLERANCE_S = 1e-12


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


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


def _nonnegative_real(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return normalized


def _positive_real(name: str, value: object) -> float:
    normalized = _finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _one_of(name: str, value: object, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(allowed)
        raise ValueError(f"{name} must be one of: {choices}")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Path and content digest identifying one persisted artifact."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _nonempty_string("path", self.path)
        _digest("sha256", self.sha256)


@dataclass(frozen=True, slots=True)
class ModelProvenance:
    """Training and artifact provenance for one predictive model variant."""

    model_variant: ModelVariant
    training_mode: TrainingMode
    training_seed: int
    checkpoint: ArtifactReference
    model_config: ArtifactReference
    normalization: ArtifactReference
    comparison_provenance: ArtifactReference
    dataset_manifests: tuple[ArtifactReference, ...]

    def __post_init__(self) -> None:
        _one_of("model_variant", self.model_variant, ("data_only", "physics_informed"))
        _one_of("training_mode", self.training_mode, ("smoke", "full"))
        object.__setattr__(
            self,
            "training_seed",
            _nonnegative_integer("training_seed", self.training_seed),
        )
        for name in (
            "checkpoint",
            "model_config",
            "normalization",
            "comparison_provenance",
        ):
            if type(getattr(self, name)) is not ArtifactReference:
                raise ValueError(f"{name} must be an ArtifactReference")
        if not isinstance(self.dataset_manifests, (tuple, list)):
            raise ValueError("dataset_manifests must be a nonempty immutable sequence")
        manifests = tuple(self.dataset_manifests)
        if not manifests:
            raise ValueError("dataset_manifests must be nonempty")
        if any(type(reference) is not ArtifactReference for reference in manifests):
            raise ValueError("dataset_manifests must contain ArtifactReference values")
        if len(set(manifests)) != len(manifests):
            raise ValueError("dataset_manifests must contain unique references")
        object.__setattr__(self, "dataset_manifests", manifests)


@dataclass(frozen=True, slots=True)
class ForecastAssumptions:
    """Explicit exogenous-input and control assumptions used by forecasts."""

    current_assumption: PersistentInputAssumption
    soc_assumption: PersistentInputAssumption
    ambient_temperature_assumption: PersistentInputAssumption
    coolant_inlet_temperature_assumption: PersistentInputAssumption
    action_handling: ActionHandling
    uses_future_simulator_truth: Literal[False]

    def __post_init__(self) -> None:
        for name in (
            "current_assumption",
            "soc_assumption",
            "ambient_temperature_assumption",
            "coolant_inlet_temperature_assumption",
        ):
            _one_of(name, getattr(self, name), ("current_value_persistence",))
        _one_of(
            "action_handling", self.action_handling, ("constant_candidate_sequence",)
        )
        if self.uses_future_simulator_truth is not False:
            raise ValueError("uses_future_simulator_truth must be exactly False")


@dataclass(frozen=True, slots=True)
class ControllerMetrics:
    """Aggregate thermal-safety and coolant-energy metrics for one controller."""

    peak_temperature_k: float
    mean_temperature_k: float
    maximum_temperature_spread_k: float
    pump_energy_j: float
    thermal_violation_count: int
    thermal_violation_duration_s: float

    def __post_init__(self) -> None:
        for name in ("peak_temperature_k", "mean_temperature_k"):
            object.__setattr__(self, name, _positive_real(name, getattr(self, name)))
        if self.peak_temperature_k < self.mean_temperature_k:
            raise ValueError("peak_temperature_k must be >= mean_temperature_k")
        for name in (
            "maximum_temperature_spread_k",
            "pump_energy_j",
            "thermal_violation_duration_s",
        ):
            object.__setattr__(self, name, _nonnegative_real(name, getattr(self, name)))
        object.__setattr__(
            self,
            "thermal_violation_count",
            _nonnegative_integer(
                "thermal_violation_count", self.thermal_violation_count
            ),
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluationMetrics:
    """One-step model accuracy and physics-residual summaries."""

    model_variant: ModelVariant
    distribution: EvaluationDistribution
    mean_absolute_error_k: float
    root_mean_square_error_k: float
    maximum_cell_error_k: float
    mean_absolute_physics_residual_w: float
    root_mean_square_physics_residual_w: float
    maximum_absolute_physics_residual_w: float
    sample_count: int

    def __post_init__(self) -> None:
        _one_of("model_variant", self.model_variant, ("data_only", "physics_informed"))
        _one_of(
            "distribution",
            self.distribution,
            ("in_distribution", "out_of_distribution"),
        )
        for name in (
            "mean_absolute_error_k",
            "root_mean_square_error_k",
            "maximum_cell_error_k",
            "mean_absolute_physics_residual_w",
            "root_mean_square_physics_residual_w",
            "maximum_absolute_physics_residual_w",
        ):
            object.__setattr__(self, name, _nonnegative_real(name, getattr(self, name)))
        if self.root_mean_square_error_k < self.mean_absolute_error_k:
            raise ValueError(
                "root_mean_square_error_k must be >= mean_absolute_error_k"
            )
        if self.maximum_cell_error_k < max(
            self.mean_absolute_error_k, self.root_mean_square_error_k
        ):
            raise ValueError(
                "maximum_cell_error_k must be >= mean_absolute_error_k and "
                "root_mean_square_error_k"
            )
        if (
            self.root_mean_square_physics_residual_w
            < self.mean_absolute_physics_residual_w
        ):
            raise ValueError(
                "root_mean_square_physics_residual_w must be >= "
                "mean_absolute_physics_residual_w"
            )
        if self.maximum_absolute_physics_residual_w < max(
            self.mean_absolute_physics_residual_w,
            self.root_mean_square_physics_residual_w,
        ):
            raise ValueError(
                "maximum_absolute_physics_residual_w must be >= mean and RMS "
                "physics residuals"
            )
        object.__setattr__(
            self, "sample_count", _positive_integer("sample_count", self.sample_count)
        )


@dataclass(frozen=True, slots=True)
class ForecastHorizonMetrics:
    """Forecast error summary for one model, distribution, and horizon."""

    model_variant: ModelVariant
    distribution: EvaluationDistribution
    horizon_s: float
    mean_absolute_error_k: float
    root_mean_square_error_k: float
    maximum_cell_error_k: float
    forecast_count: int

    def __post_init__(self) -> None:
        _one_of("model_variant", self.model_variant, ("data_only", "physics_informed"))
        _one_of(
            "distribution",
            self.distribution,
            ("in_distribution", "out_of_distribution"),
        )
        horizon = _finite_real("horizon_s", self.horizon_s)
        if horizon not in FORECAST_HORIZONS_S:
            raise ValueError("horizon_s must be one of 30.0, 60.0, or 120.0")
        object.__setattr__(self, "horizon_s", horizon)
        for name in (
            "mean_absolute_error_k",
            "root_mean_square_error_k",
            "maximum_cell_error_k",
        ):
            object.__setattr__(self, name, _nonnegative_real(name, getattr(self, name)))
        if self.root_mean_square_error_k < self.mean_absolute_error_k:
            raise ValueError(
                "root_mean_square_error_k must be >= mean_absolute_error_k"
            )
        if self.maximum_cell_error_k < max(
            self.mean_absolute_error_k, self.root_mean_square_error_k
        ):
            raise ValueError(
                "maximum_cell_error_k must be >= mean_absolute_error_k and "
                "root_mean_square_error_k"
            )
        object.__setattr__(
            self,
            "forecast_count",
            _positive_integer("forecast_count", self.forecast_count),
        )


@dataclass(frozen=True, slots=True)
class FaultDetectionOutcome:
    """Observed detector outcome, kept separate from injected-fault truth."""

    suspected_cell_index: int | None = None
    detection_time_s: float | None = None
    threshold_crossing_time_s: float | None = None
    lead_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.suspected_cell_index is not None:
            cell_index: object = self.suspected_cell_index
            if isinstance(cell_index, bool) or not isinstance(cell_index, Integral):
                raise ValueError("suspected_cell_index must be an integer from 0 to 11")
            normalized_cell_index = int(cell_index)
            if not 0 <= normalized_cell_index <= 11:
                raise ValueError("suspected_cell_index must be an integer from 0 to 11")
            object.__setattr__(self, "suspected_cell_index", normalized_cell_index)

        for name in ("detection_time_s", "threshold_crossing_time_s", "lead_time_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_real(name, value))

        if (self.suspected_cell_index is None) != (self.detection_time_s is None):
            if self.detection_time_s is None:
                raise ValueError(
                    "detection_time_s is required with suspected_cell_index"
                )
            raise ValueError("suspected_cell_index is required with detection_time_s")

        if self.detection_time_s is None:
            if self.suspected_cell_index is not None:
                raise ValueError("suspected_cell_index requires detection_time_s")
            if self.lead_time_s is not None:
                raise ValueError("lead_time_s requires detection_time_s")
        if self.lead_time_s is not None:
            if self.threshold_crossing_time_s is None:
                raise ValueError("lead_time_s requires threshold_crossing_time_s")
            expected_lead = self.threshold_crossing_time_s - self.detection_time_s  # type: ignore[operator]
            if expected_lead < 0.0 or not math.isclose(
                self.lead_time_s,
                expected_lead,
                rel_tol=_TIME_REL_TOLERANCE,
                abs_tol=_TIME_ABS_TOLERANCE_S,
            ):
                raise ValueError(
                    "lead_time_s must equal threshold_crossing_time_s minus "
                    "detection_time_s and be nonnegative"
                )
        elif (
            self.detection_time_s is not None
            and self.threshold_crossing_time_s is not None
            and self.threshold_crossing_time_s < self.detection_time_s
        ):
            raise ValueError(
                "threshold_crossing_time_s must not precede detection_time_s"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkCellResult:
    """One scenario/controller cell in the complete comparison matrix."""

    mode: BenchmarkMode
    scenario_id: ScenarioId
    variant_id: VariantId
    seed: int
    scenario_configuration_sha256: str
    controller_metrics: ControllerMetrics
    model_provenance: ModelProvenance | None = None
    forecast_assumptions: ForecastAssumptions | None = None
    fault_detection: FaultDetectionOutcome | None = None

    def __post_init__(self) -> None:
        _one_of("mode", self.mode, ("quick", "full"))
        _one_of("scenario_id", self.scenario_id, SCENARIO_IDS)
        _one_of("variant_id", self.variant_id, VARIANT_IDS)
        object.__setattr__(self, "seed", _nonnegative_integer("seed", self.seed))
        _digest("scenario_configuration_sha256", self.scenario_configuration_sha256)
        if type(self.controller_metrics) is not ControllerMetrics:
            raise ValueError("controller_metrics must be a ControllerMetrics")

        predictive = self.variant_id in (
            "predictive_data_only",
            "predictive_physics_informed",
        )
        if predictive:
            if type(self.model_provenance) is not ModelProvenance:
                raise ValueError("model_provenance is required for predictive variants")
            expected_model = (
                "data_only"
                if self.variant_id == "predictive_data_only"
                else "physics_informed"
            )
            if self.model_provenance.model_variant != expected_model:
                raise ValueError("model_provenance.model_variant must match variant_id")
            if type(self.forecast_assumptions) is not ForecastAssumptions:
                raise ValueError(
                    "forecast_assumptions is required for predictive variants"
                )
            if self.mode == "full" and self.model_provenance.training_mode != "full":
                raise ValueError("training_mode must be full for a full benchmark")
        elif self.model_provenance is not None:
            raise ValueError("model_provenance is prohibited for reactive variants")
        elif self.forecast_assumptions is not None:
            raise ValueError("forecast_assumptions is prohibited for reactive variants")

        if (
            self.fault_detection is not None
            and type(self.fault_detection) is not FaultDetectionOutcome
        ):
            raise ValueError("fault_detection must be a FaultDetectionOutcome or None")


@dataclass(frozen=True, slots=True)
class BenchmarkMatrix:
    """Complete six-scenario by four-variant benchmark result matrix."""

    mode: BenchmarkMode
    cells: tuple[BenchmarkCellResult, ...]

    def __post_init__(self) -> None:
        _one_of("mode", self.mode, ("quick", "full"))
        if not isinstance(self.cells, (tuple, list)):
            raise ValueError(
                "cells must be an immutable sequence of BenchmarkCellResult"
            )
        cells = tuple(self.cells)
        if any(type(cell) is not BenchmarkCellResult for cell in cells):
            raise ValueError("cells must contain BenchmarkCellResult values")
        if any(cell.mode != self.mode for cell in cells):
            raise ValueError("every cell mode must equal matrix mode")

        seen: set[tuple[str, str]] = set()
        for cell in cells:
            key = (cell.scenario_id, cell.variant_id)
            if key in seen:
                raise ValueError(
                    f"duplicate benchmark cell for scenario_id={key[0]} "
                    f"and variant_id={key[1]}"
                )
            seen.add(key)

        required = {
            (scenario, variant) for scenario in SCENARIO_IDS for variant in VARIANT_IDS
        }
        missing = required - seen
        if missing:
            ordered_missing = tuple(
                (scenario, variant)
                for scenario in SCENARIO_IDS
                for variant in VARIANT_IDS
                if (scenario, variant) in missing
            )
            raise ValueError(f"missing benchmark cells: {ordered_missing}")
        if seen != required:
            raise ValueError("cells contain unknown scenario/controller pairs")

        by_scenario: dict[str, list[BenchmarkCellResult]] = {}
        for cell in cells:
            by_scenario.setdefault(cell.scenario_id, []).append(cell)
        for scenario_id, scenario_cells in by_scenario.items():
            seeds = {cell.seed for cell in scenario_cells}
            if len(seeds) != 1:
                raise ValueError(f"all cells for {scenario_id} must share seed")
            digests = {cell.scenario_configuration_sha256 for cell in scenario_cells}
            if len(digests) != 1:
                raise ValueError(
                    f"all cells for {scenario_id} must share "
                    "scenario_configuration_sha256"
                )
        object.__setattr__(self, "cells", cells)


__all__ = [
    "ActionHandling",
    "ArtifactReference",
    "BenchmarkCellResult",
    "BenchmarkMatrix",
    "BenchmarkMode",
    "ControllerMetrics",
    "EvaluationDistribution",
    "FaultDetectionOutcome",
    "FORECAST_HORIZONS_S",
    "ForecastAssumptions",
    "ForecastHorizonMetrics",
    "ModelEvaluationMetrics",
    "ModelProvenance",
    "ModelVariant",
    "PersistentInputAssumption",
    "SCENARIO_IDS",
    "ScenarioId",
    "TrainingMode",
    "VARIANT_IDS",
    "VariantId",
]
