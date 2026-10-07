"""Fair execution of the complete Phase 12 controller benchmark matrix."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeVar, cast

from thermotwin.experiments.benchmark_controllers import (
    BENCHMARK_FORECAST_ASSUMPTIONS,
    BenchmarkControllerParameters,
    build_benchmark_controllers,
)
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    BenchmarkCellResult,
    BenchmarkMatrix,
    BenchmarkMode,
    ControllerMetrics,
    ModelProvenance,
    ModelVariant,
    ScenarioId,
    VariantId,
)
from thermotwin.experiments.runner import run_experiment
from thermotwin.experiments.types import ExperimentResult
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.scenarios.benchmark import BenchmarkScenarioCase

ExecutionKey = tuple[ScenarioId, VariantId]
_Key = TypeVar("_Key")
_Value = TypeVar("_Value")
_FORECASTER_KEYS: Final[tuple[ModelVariant, ...]] = (
    "data_only",
    "physics_informed",
)
_PREDICTIVE_VARIANTS: Final[dict[VariantId, ModelVariant]] = {
    "predictive_data_only": "data_only",
    "predictive_physics_informed": "physics_informed",
}


@dataclass(frozen=True, slots=True)
class BenchmarkExecution:
    """Validated aggregate matrix and immutable evidence traces."""

    matrix: BenchmarkMatrix
    results: Mapping[ExecutionKey, ExperimentResult]

    def __post_init__(self) -> None:
        if type(self.matrix) is not BenchmarkMatrix:
            raise ValueError("matrix must be exactly a BenchmarkMatrix")
        if not isinstance(self.results, Mapping):
            raise ValueError("results must be a mapping")
        copied = dict(self.results)
        required = {
            (scenario_id, variant_id)
            for scenario_id in SCENARIO_IDS
            for variant_id in VARIANT_IDS
        }
        if set(copied) != required:
            raise ValueError("results must contain exactly the 24 matrix keys")
        if any(type(result) is not ExperimentResult for result in copied.values()):
            raise ValueError("results must contain ExperimentResult values")
        cells = {
            (cell.scenario_id, cell.variant_id): cell for cell in self.matrix.cells
        }
        for key, result in copied.items():
            if key[0] not in SCENARIO_IDS or key[1] not in VARIANT_IDS:
                raise ValueError("results contain an invalid scenario/variant key")
            cell = cells[key]
            if result.scenario.name != key[0]:
                raise ValueError("result scenario identity does not match its key")
            if result.seed != cell.seed:
                raise ValueError("result seed does not match its matrix cell")
            if _controller_metrics(result) != cell.controller_metrics:
                raise ValueError("result metrics do not match its matrix cell")
        for scenario_id in SCENARIO_IDS:
            scenario_results = tuple(
                copied[(scenario_id, variant_id)] for variant_id in VARIANT_IDS
            )
            if any(
                result.scenario is not scenario_results[0].scenario
                for result in scenario_results[1:]
            ):
                raise ValueError(
                    "all results for a scenario must retain the identical "
                    "scenario object"
                )
        object.__setattr__(self, "results", MappingProxyType(copied))


def _controller_metrics(result: ExperimentResult) -> ControllerMetrics:
    metrics = result.metrics
    return ControllerMetrics(
        peak_temperature_k=metrics.peak_temperature_k,
        mean_temperature_k=metrics.mean_temperature_k,
        maximum_temperature_spread_k=metrics.maximum_temperature_spread_k,
        pump_energy_j=metrics.pump_energy_j,
        thermal_violation_count=metrics.thermal_violation_count,
        thermal_violation_duration_s=metrics.time_above_thermal_limit_s,
    )


def _exact_mapping_keys(
    name: str, value: object, expected: AbstractSet[_Key]
) -> Mapping[_Key, _Value]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be exact strings")
    received = set(value)
    if received != expected:
        missing = sorted(expected - received, key=repr)
        extra = sorted(received - expected, key=repr)
        raise ValueError(f"{name} keys mismatch; missing={missing}, extra={extra}")
    return cast(Mapping[_Key, _Value], value)


def _validated_controller_parameters(value: object) -> BenchmarkControllerParameters:
    if type(value) is not BenchmarkControllerParameters:
        raise ValueError(
            "controller_parameters must be exactly BenchmarkControllerParameters"
        )
    return value


def run_benchmark_matrix(
    cases: Mapping[ScenarioId, BenchmarkScenarioCase],
    controller_parameters: BenchmarkControllerParameters,
    forecasters: Mapping[ModelVariant, ThermalForecaster],
    model_provenance: Mapping[ModelVariant, ModelProvenance],
    *,
    mode: BenchmarkMode,
) -> BenchmarkExecution:
    """Run all scenarios against the same four-controller comparison matrix."""
    if mode not in ("quick", "full"):
        raise ValueError("mode must be exactly 'quick' or 'full'")
    validated_parameters = _validated_controller_parameters(
        cast(object, controller_parameters)
    )
    expected_scenarios = set(SCENARIO_IDS)
    case_mapping: Mapping[ScenarioId, BenchmarkScenarioCase] = _exact_mapping_keys(
        "cases", cases, expected_scenarios
    )
    forecaster_mapping: Mapping[ModelVariant, ThermalForecaster] = _exact_mapping_keys(
        "forecasters", forecasters, set(_FORECASTER_KEYS)
    )
    provenance_mapping: Mapping[ModelVariant, ModelProvenance] = _exact_mapping_keys(
        "model_provenance", model_provenance, set(_FORECASTER_KEYS)
    )

    ordered_cases: list[BenchmarkScenarioCase] = []
    for scenario_id in SCENARIO_IDS:
        case = case_mapping[scenario_id]
        if type(case) is not BenchmarkScenarioCase:
            raise ValueError(f"cases[{scenario_id!r}] must be BenchmarkScenarioCase")
        if case.scenario_id != scenario_id:
            raise ValueError(f"cases[{scenario_id!r}] has a mismatched scenario_id")
        if case.scenario.name != scenario_id:
            raise ValueError(f"cases[{scenario_id!r}] scenario name must match its key")
        if case.scenario.metadata.get("mode") != mode:
            raise ValueError(f"cases[{scenario_id!r}] metadata mode must be {mode!r}")
        ordered_cases.append(case)

    copied_forecasters: dict[ModelVariant, ThermalForecaster] = {}
    copied_provenance: dict[ModelVariant, ModelProvenance] = {}
    for key in _FORECASTER_KEYS:
        forecaster = forecaster_mapping[key]
        provenance = provenance_mapping[key]
        if not isinstance(forecaster, ThermalForecaster):
            raise ValueError(f"forecasters[{key!r}] must satisfy ThermalForecaster")
        if type(provenance) is not ModelProvenance:
            raise ValueError(f"model_provenance[{key!r}] must be ModelProvenance")
        if provenance.model_variant != key:
            raise ValueError(f"model_provenance[{key!r}] model variant mismatch")
        if mode == "full" and provenance.training_mode != "full":
            raise ValueError("training_mode must be full for a full benchmark")
        copied_forecasters[key] = forecaster
        copied_provenance[key] = provenance

    results: dict[ExecutionKey, ExperimentResult] = {}
    cells: list[BenchmarkCellResult] = []
    for case in ordered_cases:
        # Each scenario receives a new controller set, including fresh PID state.
        controllers = build_benchmark_controllers(
            case.scenario.environment_config.cooling,
            validated_parameters,
            copied_forecasters,
        )
        for variant_id in VARIANT_IDS:
            controller = controllers[variant_id]
            result = run_experiment(case.scenario, controller, case.seed)
            execution_key = (case.scenario_id, variant_id)
            results[execution_key] = result
            predictive_model = _PREDICTIVE_VARIANTS.get(variant_id)
            cell_model_provenance = (
                None
                if predictive_model is None
                else copied_provenance[predictive_model]
            )
            cells.append(
                BenchmarkCellResult(
                    mode=mode,
                    scenario_id=case.scenario_id,
                    variant_id=variant_id,
                    seed=case.seed,
                    scenario_configuration_sha256=case.configuration_sha256,
                    controller_metrics=_controller_metrics(result),
                    model_provenance=cell_model_provenance,
                    forecast_assumptions=(
                        BENCHMARK_FORECAST_ASSUMPTIONS
                        if predictive_model is not None
                        else None
                    ),
                    fault_detection=None,
                )
            )

    matrix = BenchmarkMatrix(mode=mode, cells=tuple(cells))
    execution = BenchmarkExecution(matrix=matrix, results=results)
    for case in ordered_cases:
        for variant_id in VARIANT_IDS:
            result = execution.results[(case.scenario_id, variant_id)]
            if result.scenario is not case.scenario:
                raise ValueError("runner did not retain the identical scenario object")
            if result.seed != case.seed:
                raise ValueError("runner did not retain the scenario seed")
    return execution


__all__ = ["BenchmarkExecution", "ExecutionKey", "run_benchmark_matrix"]
