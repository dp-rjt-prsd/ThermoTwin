"""Aggregate validated Phase 12 artifacts without rerunning experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from thermotwin.experiments.benchmark_serialization import load_benchmark_matrix
from thermotwin.experiments.benchmark_types import (
    FORECAST_HORIZONS_S,
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkMatrix,
    BenchmarkMode,
    ControllerMetrics,
    EvaluationDistribution,
    FaultDetectionOutcome,
    ModelEvaluationMetrics,
    ModelProvenance,
    ModelVariant,
    ScenarioId,
    TrainingMode,
    VariantId,
)
from thermotwin.models.evaluation import RegressionMetrics

EvidenceStatus = Literal["workflow_only", "descriptive_synthetic_benchmark"]
ComparisonDirection = Literal["data_only_lower", "physics_informed_lower", "tie"]

_MODELS: tuple[ModelVariant, ...] = ("data_only", "physics_informed")
_DISTRIBUTIONS: tuple[EvaluationDistribution, ...] = (
    "in_distribution",
    "out_of_distribution",
)
_REGRESSION_FIELDS = {
    "mae_kelvin",
    "rmse_kelvin",
    "per_cell_mae_kelvin",
    "per_cell_rmse_kelvin",
    "maximum_absolute_error_kelvin",
    "worst_cell_index",
}
_COMPARISON_METRIC_FIELDS = {
    "regression",
    "mean_absolute_physics_residual_w",
    "root_mean_square_physics_residual_w",
    "maximum_absolute_physics_residual_w",
}
_COMPARISON_PROVENANCE_FIELDS = {
    "artifact_schema_version",
    "datasets",
    "seed",
    "dataset_seeds",
    "package_versions",
    "config_source",
    "configuration_source_path",
    "normalization",
    "distribution_labels",
    "smoke_test",
}
_FORECAST_PROVENANCE_FIELDS = {
    "artifact_schema_version",
    "seeds",
    "package_versions",
    "source_paths",
    "copied_artifacts",
    "horizons_s",
    "time_step_s",
    "forecast_counts",
    "normalization_source",
    "distribution_label",
    "split_label",
    "smoke_test",
}
_FAULT_SUMMARY_FIELDS = {
    "thermal_limit_k",
    "threshold_crossing_time_s",
    "lead_time",
    "detected_fault_cell",
    "peak_temperature_k",
    "temperature_spread_k",
    "pump_energy_j",
    "thermal_violation_count",
    "anomaly_event_count",
}
_FAULT_PROVENANCE_REQUIRED = {
    "schema_version",
    "scenario",
    "environment_config",
    "controller_class_name",
    "seed",
    "result_metadata",
    "record_count",
    "workflow",
    "model_variant",
    "inference_inputs",
    "hidden_fault_truth",
    "claim_scope",
}
_ROOT_FIELDS = {
    "schema_version",
    "evidence_status",
    "benchmark_mode",
    "benchmark_seed",
    "comparison_seed",
    "id_dataset_seed",
    "ood_dataset_seed",
    "model_provenance",
    "model_metrics",
    "forecast_metrics",
    "controller_metrics",
    "fault_detection",
    "comparisons",
    "source_artifacts",
    "limitations",
}
_SERIALIZED_SCHEMA_VERSION = 1
_ARTIFACT_NAME = "final_evaluation.json"


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is prohibited: {value}")


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required saved artifact is missing or invalid: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_json_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read saved artifact {path.name}: {error}"
        ) from error


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _exact(value: object, fields: set[str], name: str) -> dict[str, object]:
    result = _mapping(value, name)
    if set(result) != fields:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(fields - set(result))}, "
            f"extra={sorted(set(result) - fields)}"
        )
    return result


def _finite(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return number


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: object, name: str) -> Path:
    if type(relative) is not str:
        raise ValueError(f"{name} must be a canonical relative path")
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in relative
        or pure.as_posix() != relative
    ):
        raise ValueError(f"{name} must be a canonical relative path")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise ValueError(f"{name} escapes its evidence directory") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} is missing or invalid")
    return path


def _parse_regression(value: object, name: str) -> RegressionMetrics:
    fields = _exact(value, _REGRESSION_FIELDS, name)
    mae_cells = fields["per_cell_mae_kelvin"]
    rmse_cells = fields["per_cell_rmse_kelvin"]
    if not isinstance(mae_cells, list) or not isinstance(rmse_cells, list):
        raise ValueError(f"{name} per-cell metrics must be arrays")
    if len(mae_cells) != 12 or len(rmse_cells) != 12:
        raise ValueError(f"{name} must contain metrics for exactly 12 cells")
    worst_cell_index = fields["worst_cell_index"]
    if isinstance(worst_cell_index, bool) or type(worst_cell_index) is not int:
        raise ValueError(f"{name}.worst_cell_index must be an integer")
    return RegressionMetrics(
        mae_kelvin=_finite(
            f"{name}.mae_kelvin", fields["mae_kelvin"], nonnegative=True
        ),
        rmse_kelvin=_finite(
            f"{name}.rmse_kelvin", fields["rmse_kelvin"], nonnegative=True
        ),
        per_cell_mae_kelvin=tuple(
            _finite(f"{name}.per_cell_mae_kelvin", item, nonnegative=True)
            for item in mae_cells
        ),
        per_cell_rmse_kelvin=tuple(
            _finite(f"{name}.per_cell_rmse_kelvin", item, nonnegative=True)
            for item in rmse_cells
        ),
        maximum_absolute_error_kelvin=_finite(
            f"{name}.maximum_absolute_error_kelvin",
            fields["maximum_absolute_error_kelvin"],
            nonnegative=True,
        ),
        worst_cell_index=worst_cell_index,
    )


@dataclass(frozen=True, slots=True)
class FinalForecastMetrics:
    """Saved error summary for one model, distribution, and horizon."""

    model_variant: ModelVariant
    distribution: EvaluationDistribution
    horizon_s: float
    mae_kelvin: float
    rmse_kelvin: float
    maximum_absolute_cell_error_kelvin: float
    forecast_count: int
    step_count: int

    def __post_init__(self) -> None:
        if self.model_variant not in _MODELS or self.distribution not in _DISTRIBUTIONS:
            raise ValueError("forecast metric model or distribution is invalid")
        horizon = _finite("horizon_s", self.horizon_s, nonnegative=True)
        if horizon not in FORECAST_HORIZONS_S:
            raise ValueError("forecast metric horizon is unsupported")
        object.__setattr__(self, "horizon_s", horizon)
        for name in (
            "mae_kelvin",
            "rmse_kelvin",
            "maximum_absolute_cell_error_kelvin",
        ):
            object.__setattr__(
                self,
                name,
                _finite(name, getattr(self, name), nonnegative=True),
            )
        if (
            self.rmse_kelvin < self.mae_kelvin
            or self.maximum_absolute_cell_error_kelvin
            < max(self.mae_kelvin, self.rmse_kelvin)
        ):
            raise ValueError("forecast error metrics are internally inconsistent")
        object.__setattr__(
            self,
            "forecast_count",
            _positive_integer("forecast_count", self.forecast_count),
        )
        object.__setattr__(
            self, "step_count", _positive_integer("step_count", self.step_count)
        )


@dataclass(frozen=True, slots=True)
class ScenarioControllerMetrics:
    """Saved controller results for one scenario and variant."""

    scenario_id: ScenarioId
    variant_id: VariantId
    metrics: ControllerMetrics

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS or self.variant_id not in VARIANT_IDS:
            raise ValueError("scenario/controller identity is invalid")
        if type(self.metrics) is not ControllerMetrics:
            raise ValueError("metrics must be ControllerMetrics")


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Deterministic lower-value comparison with a neutral direction label."""

    scope: str
    metric_name: str
    data_only_value: float
    physics_informed_value: float
    lower_value_variant: ComparisonDirection

    def __post_init__(self) -> None:
        if not self.scope.strip() or not self.metric_name.strip():
            raise ValueError("comparison scope and metric name must be nonempty")
        left = _finite("data_only_value", self.data_only_value, nonnegative=True)
        right = _finite(
            "physics_informed_value", self.physics_informed_value, nonnegative=True
        )
        object.__setattr__(self, "data_only_value", left)
        object.__setattr__(self, "physics_informed_value", right)
        expected: ComparisonDirection = (
            "data_only_lower"
            if left < right
            else "physics_informed_lower"
            if right < left
            else "tie"
        )
        if self.lower_value_variant != expected:
            raise ValueError("comparison direction does not match saved values")


@dataclass(frozen=True, slots=True)
class SourceArtifactDigest:
    """SHA-256 identity for one input artifact used by the aggregation."""

    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be nonempty")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class FinalEvaluation:
    """Immutable, machine-derived summary of validated saved experiments."""

    evidence_status: EvidenceStatus
    benchmark_mode: BenchmarkMode
    benchmark_seed: int
    comparison_seed: int
    id_dataset_seed: int
    ood_dataset_seed: int
    model_provenance: tuple[ModelProvenance, ...]
    model_metrics: tuple[ModelEvaluationMetrics, ...]
    forecast_metrics: tuple[FinalForecastMetrics, ...]
    controller_metrics: tuple[ScenarioControllerMetrics, ...]
    fault_detection: FaultDetectionOutcome
    comparisons: tuple[MetricComparison, ...]
    source_artifacts: tuple[SourceArtifactDigest, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        expected_status = (
            "workflow_only"
            if self.benchmark_mode == "quick"
            else "descriptive_synthetic_benchmark"
        )
        if self.evidence_status != expected_status:
            raise ValueError("evidence status does not match benchmark mode")
        for name in (
            "benchmark_seed",
            "comparison_seed",
            "id_dataset_seed",
            "ood_dataset_seed",
        ):
            object.__setattr__(self, name, _positive_or_zero(name, getattr(self, name)))
        if self.id_dataset_seed == self.ood_dataset_seed:
            raise ValueError("ID and OOD dataset seeds must differ")
        if type(self.model_provenance) is not tuple or len(self.model_provenance) != 2:
            raise ValueError("model_provenance must contain exactly two models")
        if tuple(item.model_variant for item in self.model_provenance) != _MODELS:
            raise ValueError("model provenance must use canonical model order")
        if any(
            item.training_seed != self.comparison_seed
            or (self.benchmark_mode == "full" and item.training_mode != "full")
            for item in self.model_provenance
        ):
            raise ValueError("model provenance is incompatible with evaluation mode")
        if len({item.training_mode for item in self.model_provenance}) != 1:
            raise ValueError("both models must use the same training mode")
        if type(self.model_metrics) is not tuple or len(self.model_metrics) != 4:
            raise ValueError(
                "model_metrics must contain two models by two distributions"
            )
        model_keys = {
            (item.model_variant, item.distribution) for item in self.model_metrics
        }
        if model_keys != {
            (model, dist) for model in _MODELS for dist in _DISTRIBUTIONS
        }:
            raise ValueError("model_metrics is missing or duplicates a required pair")
        if type(self.forecast_metrics) is not tuple or len(self.forecast_metrics) != 12:
            raise ValueError(
                "forecast_metrics must contain two models, distributions, and horizons"
            )
        forecast_keys = {
            (item.model_variant, item.distribution, item.horizon_s)
            for item in self.forecast_metrics
        }
        expected_forecasts = {
            (model, dist, horizon)
            for model in _MODELS
            for dist in _DISTRIBUTIONS
            for horizon in FORECAST_HORIZONS_S
        }
        if forecast_keys != expected_forecasts:
            raise ValueError(
                "forecast_metrics is missing or duplicates a required horizon"
            )
        if (
            type(self.controller_metrics) is not tuple
            or len(self.controller_metrics) != 24
        ):
            raise ValueError(
                "controller_metrics must contain the complete six by four matrix"
            )
        controller_keys = {
            (item.scenario_id, item.variant_id) for item in self.controller_metrics
        }
        expected_controller_keys = {
            (scenario, variant) for scenario in SCENARIO_IDS for variant in VARIANT_IDS
        }
        if controller_keys != expected_controller_keys:
            raise ValueError(
                "controller_metrics is missing or duplicates a matrix cell"
            )
        if type(self.fault_detection) is not FaultDetectionOutcome:
            raise ValueError("fault_detection must be a FaultDetectionOutcome")
        if type(self.comparisons) is not tuple or not self.comparisons:
            raise ValueError("comparisons must be a nonempty immutable sequence")
        if type(self.source_artifacts) is not tuple or not self.source_artifacts:
            raise ValueError("source_artifacts must be a nonempty immutable sequence")
        ids = tuple(item.artifact_id for item in self.source_artifacts)
        if len(ids) != len(set(ids)):
            raise ValueError("source_artifacts must have unique IDs")
        if (
            type(self.limitations) is not tuple
            or not self.limitations
            or any(
                type(item) is not str or not item.strip() for item in self.limitations
            )
        ):
            raise ValueError("limitations must be nonempty strings")


def _positive_or_zero(name: str, value: object) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _load_comparison_sources(
    comparison_root: Path, benchmark_root: Path, matrix: BenchmarkMatrix
) -> tuple[
    dict[tuple[ModelVariant, EvaluationDistribution], ModelEvaluationMetrics],
    dict[str, object],
    dict[EvaluationDistribution, dict[str, object]],
    dict[str, bytes],
]:
    if not comparison_root.is_dir() or comparison_root.is_symlink():
        raise ValueError("comparison directory is missing or invalid")
    source_names = (
        "data_only.pt",
        "physics_informed.pt",
        "model_config.json",
        "normalization.json",
        "provenance.json",
        "id_comparison_metrics.json",
        "ood_comparison_metrics.json",
    )
    source_bytes: dict[str, bytes] = {}
    for name in source_names:
        path = comparison_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"comparison artifact is missing or invalid: {name}")
        source_bytes[name] = path.read_bytes()

    comparison_provenance = _mapping(
        _read_json(comparison_root / "provenance.json"), "comparison provenance"
    )
    _exact(
        comparison_provenance, _COMPARISON_PROVENANCE_FIELDS, "comparison provenance"
    )
    if comparison_provenance["artifact_schema_version"] != "1.0":
        raise ValueError("comparison provenance schema is unsupported")
    benchmark_manifest_path = benchmark_root / "comparison" / "provenance.json"
    benchmark_comparison = _mapping(
        _read_json(benchmark_manifest_path), "benchmark comparison provenance"
    )
    normalized = json.loads(json.dumps(comparison_provenance))
    datasets = _mapping(normalized["datasets"], "comparison datasets")
    for distribution in _DISTRIBUTIONS:
        dataset = _mapping(datasets[distribution], f"{distribution} dataset")
        dataset["directory"] = f"dataset/{distribution}"
        dataset["manifest_path"] = f"dataset/{distribution}/manifest.json"
    if normalized != benchmark_comparison:
        raise ValueError("comparison provenance does not match benchmark evidence")

    raw_provenance = comparison_provenance
    mode = matrix.mode
    smoke = raw_provenance["smoke_test"]
    if type(smoke) is not bool:
        raise ValueError("comparison smoke flag must be a boolean")
    if mode == "full" and smoke:
        raise ValueError("a full benchmark requires full-trained comparison evidence")
    seed = _positive_or_zero("comparison seed", raw_provenance["seed"])
    dataset_seeds = _exact(
        raw_provenance["dataset_seeds"], set(_DISTRIBUTIONS), "comparison dataset seeds"
    )
    embedded_datasets = _mapping(raw_provenance["datasets"], "comparison datasets")
    manifests: dict[EvaluationDistribution, dict[str, object]] = {}
    for distribution in _DISTRIBUTIONS:
        source_dataset = _mapping(
            embedded_datasets[distribution], f"{distribution} dataset"
        )
        manifest = _mapping(source_dataset["manifest"], f"{distribution} manifest")
        if manifest.get("distribution") is not None:
            raise ValueError("dataset manifest has unexpected distribution field")
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("cell_count") != 12
            or manifest.get("seed") != dataset_seeds[distribution]
        ):
            raise ValueError(f"{distribution} dataset provenance is incompatible")
        manifests[distribution] = manifest
        bundled_manifest = _mapping(
            _read_json(benchmark_root / "dataset" / distribution / "manifest.json"),
            f"bundled {distribution} manifest",
        )
        if bundled_manifest != manifest:
            raise ValueError(f"{distribution} dataset manifest differs from benchmark")

    cells_by_model: dict[ModelVariant, ModelProvenance] = {}
    for cell in matrix.cells:
        provenance = cell.model_provenance
        if provenance is None:
            continue
        cells_by_model.setdefault(provenance.model_variant, provenance)
        if provenance.training_seed != seed:
            raise ValueError("benchmark model training seed differs from comparison")
        if provenance.training_mode != ("smoke" if smoke else "full"):
            raise ValueError("benchmark model training mode differs from comparison")
        for reference, name in (
            (provenance.checkpoint, f"{provenance.model_variant}.pt"),
            (provenance.model_config, "model_config.json"),
            (provenance.normalization, "normalization.json"),
        ):
            bundled = benchmark_root / reference.path
            if (
                _sha256_bytes(source_bytes[name]) != reference.sha256
                or _sha256_file(comparison_root / name) != reference.sha256
                or _sha256_file(bundled) != reference.sha256
            ):
                raise ValueError(
                    f"comparison {name} does not match benchmark reference"
                )
        for reference, _distribution in zip(
            provenance.dataset_manifests, _DISTRIBUTIONS, strict=True
        ):
            bundled = benchmark_root / reference.path
            if _sha256_file(bundled) != reference.sha256:
                raise ValueError("benchmark dataset reference digest mismatch")
    if set(cells_by_model) != set(_MODELS):
        raise ValueError("benchmark matrix is missing model provenance")
    for provenance in cells_by_model.values():
        if provenance.comparison_provenance.sha256 != _sha256_file(
            benchmark_root / provenance.comparison_provenance.path
        ):
            raise ValueError("benchmark comparison provenance reference mismatch")

    model_metrics: dict[
        tuple[ModelVariant, EvaluationDistribution], ModelEvaluationMetrics
    ] = {}
    for distribution, filename in (
        ("in_distribution", "id_comparison_metrics.json"),
        ("out_of_distribution", "ood_comparison_metrics.json"),
    ):
        document = _mapping(_read_json(comparison_root / filename), filename)
        if set(document) != set(_MODELS):
            raise ValueError(f"{filename} must contain both model variants")
        split_document = _mapping(manifests[distribution]["splits"], "dataset splits")
        test_split = _mapping(split_document["test"], "test split")
        sample_count = _positive_integer("test row_count", test_split["row_count"])
        for model in _MODELS:
            values = _exact(
                document[model], _COMPARISON_METRIC_FIELDS, f"{filename}.{model}"
            )
            regression = _parse_regression(
                values["regression"], f"{filename}.{model}.regression"
            )
            model_metrics[(model, distribution)] = ModelEvaluationMetrics(
                model_variant=model,
                distribution=distribution,
                mean_absolute_error_k=regression.mae_kelvin,
                root_mean_square_error_k=regression.rmse_kelvin,
                maximum_cell_error_k=regression.maximum_absolute_error_kelvin,
                mean_absolute_physics_residual_w=_finite(
                    "mean_absolute_physics_residual_w",
                    values["mean_absolute_physics_residual_w"],
                    nonnegative=True,
                ),
                root_mean_square_physics_residual_w=_finite(
                    "root_mean_square_physics_residual_w",
                    values["root_mean_square_physics_residual_w"],
                    nonnegative=True,
                ),
                maximum_absolute_physics_residual_w=_finite(
                    "maximum_absolute_physics_residual_w",
                    values["maximum_absolute_physics_residual_w"],
                    nonnegative=True,
                ),
                sample_count=sample_count,
            )
    return model_metrics, raw_provenance, manifests, source_bytes


def _expected_copies() -> dict[str, str]:
    return {
        "data_only_checkpoint": "checkpoints/data_only.pt",
        "physics_informed_checkpoint": "checkpoints/physics_informed.pt",
        "model_config": "model_config.json",
        "normalization": "normalization.json",
        "evaluation_test_csv": "evaluated_dataset/test.csv",
        "evaluation_manifest": "evaluated_dataset/manifest.json",
        "evaluation_generation_config": "evaluated_dataset/generation_config.json",
    }


def _validate_saved_test_csv(path: Path, manifest: dict[str, object]) -> None:
    """Check copied forecast inputs against the saved dataset manifest shape."""
    split_values = _mapping(manifest.get("splits"), "forecast dataset splits")
    test_split = _mapping(split_values.get("test"), "forecast test split")
    row_count = _positive_integer(
        "forecast test row_count", test_split.get("row_count")
    )
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(type(item) is not str for item in columns)
        or len(columns) != len(set(columns))
    ):
        raise ValueError("forecast dataset manifest columns are invalid")
    trajectory_index = (
        columns.index("trajectory_id") if "trajectory_id" in columns else -1
    )
    metadata_index = (
        columns.index("metadata_json") if "metadata_json" in columns else -1
    )
    if trajectory_index < 0 or metadata_index < 0:
        raise ValueError(
            "forecast dataset schema requires trajectory_id and metadata_json"
        )
    raw_trajectory_ids = test_split.get("trajectory_ids")
    if (
        not isinstance(raw_trajectory_ids, list)
        or not raw_trajectory_ids
        or any(
            type(value) is not str or not value.strip() for value in raw_trajectory_ids
        )
        or len(raw_trajectory_ids) != len(set(raw_trajectory_ids))
    ):
        raise ValueError("forecast test split trajectory_ids are invalid")
    expected_trajectory_ids = set(raw_trajectory_ids)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"forecast test CSV is invalid: {error}") from error
    if header != columns or len(rows) != row_count:
        raise ValueError("forecast test CSV does not match its dataset manifest")
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("forecast test CSV rows do not match the manifest columns")
    observed_trajectory_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if column_index == trajectory_index:
                if not value.strip() or value not in expected_trajectory_ids:
                    raise ValueError(
                        "forecast test CSV contains a trajectory_id absent from "
                        "its manifest"
                    )
                observed_trajectory_ids.add(value)
                continue
            if column_index == metadata_index:
                try:
                    metadata = json.loads(
                        value,
                        object_pairs_hook=_json_object,
                        parse_constant=_reject_constant,
                    )
                except (ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"forecast test CSV metadata_json is invalid at row {row_index}"
                    ) from error
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "forecast test CSV metadata_json must be an object at row "
                        f"{row_index}"
                    )
                _require_finite_json(metadata, f"metadata_json row {row_index}")
                continue
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(
                    "forecast test CSV contains a nonnumeric value"
                ) from error
            if not math.isfinite(number):
                raise ValueError(
                    "forecast test CSV contains non-finite value at "
                    f"{row_index}:{column_index}"
                )
    if observed_trajectory_ids != expected_trajectory_ids:
        raise ValueError(
            "forecast test CSV observed trajectory IDs do not match its manifest"
        )


def _require_finite_json(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{name}[{index}]")


def _forecast_metrics_for_distribution(
    forecast_root: Path,
    distribution: EvaluationDistribution,
    expected_smoke: bool,
    comparison_seed: int,
    source_bytes: dict[str, bytes],
    expected_manifest: dict[str, object],
) -> tuple[FinalForecastMetrics, ...]:
    if not forecast_root.is_dir() or forecast_root.is_symlink():
        raise ValueError(f"{distribution} forecast directory is missing or invalid")
    provenance = _mapping(
        _read_json(forecast_root / "provenance.json"), "forecast provenance"
    )
    _exact(provenance, _FORECAST_PROVENANCE_FIELDS, "forecast provenance")
    if provenance["artifact_schema_version"] != "1.0":
        raise ValueError("forecast provenance schema is unsupported")
    if provenance["distribution_label"] != distribution:
        raise ValueError("forecast distribution label does not match its input role")
    if (
        provenance["split_label"] != "test"
        or provenance["smoke_test"] is not expected_smoke
    ):
        raise ValueError("forecast split or evidence mode is incompatible")
    seeds = _exact(
        provenance["seeds"], {"training", "evaluation_dataset"}, "forecast seeds"
    )
    if (
        _positive_or_zero("forecast training seed", seeds["training"])
        != comparison_seed
    ):
        raise ValueError("forecast training seed differs from comparison")
    if _positive_or_zero(
        "forecast dataset seed", seeds["evaluation_dataset"]
    ) != expected_manifest.get("seed"):
        raise ValueError("forecast dataset seed differs from comparison provenance")
    if provenance["horizons_s"] != list(FORECAST_HORIZONS_S):
        raise ValueError("forecast horizons must be exactly 30, 60, and 120 seconds")
    timestep = _finite("forecast time_step_s", provenance["time_step_s"])
    if timestep <= 0.0:
        raise ValueError("forecast time_step_s must be positive")

    copy_entries = _exact(
        provenance["copied_artifacts"], set(_expected_copies()), "copied artifacts"
    )
    for key, relative in _expected_copies().items():
        entry = _exact(copy_entries[key], {"source", "copy"}, f"copied artifact {key}")
        if (
            type(entry["source"]) is not str
            or not entry["source"]
            or entry["copy"] != relative
        ):
            raise ValueError(
                f"copied artifact {key} has an invalid source or destination"
            )
        copied = _safe_child(forecast_root, relative, f"copied artifact {key}")
        if key == "data_only_checkpoint":
            expected_bytes = source_bytes["data_only.pt"]
        elif key == "physics_informed_checkpoint":
            expected_bytes = source_bytes["physics_informed.pt"]
        elif key == "model_config":
            expected_bytes = source_bytes["model_config.json"]
        elif key == "normalization":
            expected_bytes = source_bytes["normalization.json"]
        elif key == "evaluation_manifest":
            copied_manifest = _mapping(_read_json(copied), "forecast dataset manifest")
            if copied_manifest != expected_manifest:
                raise ValueError("forecast evaluation manifest differs from comparison")
            continue
        elif key == "evaluation_generation_config":
            generation = _mapping(_read_json(copied), "forecast generation config")
            if generation.get("seed") != expected_manifest.get("seed"):
                raise ValueError("forecast generation config seed is incompatible")
            sources = generation.get("source_config_files")
            if not isinstance(sources, list) or any(
                type(item) is not str for item in sources
            ):
                raise ValueError("forecast dataset source labels are invalid")
            expected_config = (
                "configs/dataset_ood.yaml"
                if distribution == "out_of_distribution"
                else "configs/dataset.yaml"
            )
            if sources != [expected_config]:
                raise ValueError(
                    "forecast dataset source label conflicts with distribution"
                )
            continue
        elif key == "evaluation_test_csv":
            _validate_saved_test_csv(copied, expected_manifest)
            continue
        else:
            expected_bytes = b""
        if copied.read_bytes() != expected_bytes:
            raise ValueError(f"forecast copied artifact {key} differs from comparison")

    counts = _exact(
        provenance["forecast_counts"],
        {str(horizon) for horizon in FORECAST_HORIZONS_S},
        "forecast counts",
    )
    forecast_count_by_horizon = {
        str(horizon): _positive_integer(
            f"forecast count {horizon}", counts[str(horizon)]
        )
        for horizon in FORECAST_HORIZONS_S
    }
    results: list[FinalForecastMetrics] = []
    for model in _MODELS:
        filename = f"{model}_forecast_metrics.json"
        document = _mapping(_read_json(forecast_root / filename), filename)
        horizon_values = _exact(
            document["horizons"],
            {str(horizon) for horizon in FORECAST_HORIZONS_S},
            f"{filename}.horizons",
        )
        model_horizons: list[FinalForecastMetrics] = []
        for horizon in FORECAST_HORIZONS_S:
            key = str(horizon)
            item = _exact(
                horizon_values[key],
                {"horizon_s", "step_count", "forecast_count", "metrics"},
                f"{filename}.{key}",
            )
            if _finite("horizon_s", item["horizon_s"]) != horizon:
                raise ValueError("forecast horizon key and value do not match")
            expected_steps = horizon / timestep
            step_count = _positive_integer("step_count", item["step_count"])
            if not math.isclose(
                expected_steps, step_count, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    "forecast step_count does not match horizon and timestep"
                )
            count = _positive_integer("forecast_count", item["forecast_count"])
            if count != forecast_count_by_horizon[key]:
                raise ValueError("forecast count differs from provenance")
            metrics = _parse_regression(item["metrics"], f"{filename}.{key}.metrics")
            if len(metrics.per_cell_mae_kelvin) != 12:
                raise ValueError("forecast metrics must cover exactly 12 cells")
            model_horizons.append(
                FinalForecastMetrics(
                    model_variant=model,
                    distribution=distribution,
                    horizon_s=horizon,
                    mae_kelvin=metrics.mae_kelvin,
                    rmse_kelvin=metrics.rmse_kelvin,
                    maximum_absolute_cell_error_kelvin=(
                        metrics.maximum_absolute_error_kelvin
                    ),
                    forecast_count=count,
                    step_count=step_count,
                )
            )
        if len({item.forecast_count for item in model_horizons}) != 1:
            raise ValueError("forecast counts must be consistent across horizons")
        results.extend(model_horizons)
    return tuple(results)


def _fault_outcome(fault_root: Path) -> FaultDetectionOutcome:
    if not fault_root.is_dir() or fault_root.is_symlink():
        raise ValueError("fault-detection directory is missing or invalid")
    provenance = _mapping(
        _read_json(fault_root / "provenance.json"), "fault provenance"
    )
    if not _FAULT_PROVENANCE_REQUIRED.issubset(provenance):
        raise ValueError("fault provenance is incomplete")
    if (
        provenance["schema_version"] != "1.0"
        or provenance["model_variant"] not in _MODELS
    ):
        raise ValueError("fault provenance schema or model is invalid")
    if not isinstance(provenance["hidden_fault_truth"], dict):
        raise ValueError("fault evidence lacks simulator-owned fault truth")
    truth = _exact(
        provenance["hidden_fault_truth"],
        {"cell_index", "start_time_s", "resistance_multiplier"},
        "hidden fault truth",
    )
    cell_index = truth["cell_index"]
    if (
        isinstance(cell_index, bool)
        or type(cell_index) is not int
        or not 0 <= cell_index < 12
    ):
        raise ValueError("fault truth cell index is invalid")
    _finite("fault start_time_s", truth["start_time_s"], nonnegative=True)
    if _finite("fault resistance_multiplier", truth["resistance_multiplier"]) <= 1.0:
        raise ValueError("fault resistance multiplier must exceed one")
    record_count = _positive_integer("fault record_count", provenance["record_count"])
    del record_count

    summary = _exact(
        _read_json(fault_root / "fault_summary.json"),
        _FAULT_SUMMARY_FIELDS,
        "fault summary",
    )
    for name in (
        "thermal_limit_k",
        "peak_temperature_k",
        "temperature_spread_k",
        "pump_energy_j",
    ):
        _finite(f"fault {name}", summary[name], nonnegative=True)
    violations = summary["thermal_violation_count"]
    event_count = summary["anomaly_event_count"]
    if isinstance(violations, bool) or type(violations) is not int or violations < 0:
        raise ValueError("fault thermal_violation_count is invalid")
    if isinstance(event_count, bool) or type(event_count) is not int or event_count < 0:
        raise ValueError("fault anomaly_event_count is invalid")
    crossing = summary["threshold_crossing_time_s"]
    crossing_value: float | None = None
    if crossing is not None:
        crossing_value = _finite(
            "threshold_crossing_time_s", crossing, nonnegative=True
        )
    lead = _exact(
        summary["lead_time"],
        {"detection_time_s", "threshold_crossing_time_s", "lead_time_s"},
        "fault lead time",
    )
    if lead["threshold_crossing_time_s"] != crossing_value:
        raise ValueError("fault crossing time differs from lead-time evidence")
    events = _read_json(fault_root / "anomaly_events.json")
    if not isinstance(events, list) or len(events) != event_count:
        raise ValueError("fault events do not match saved event count")
    target_events: list[dict[str, object]] = []
    for index, raw in enumerate(events):
        event = _exact(
            raw,
            {
                "cell_index",
                "anomaly_score",
                "detection_time_s",
                "predicted_hotspot_temperature_k",
                "severity_k",
            },
            f"fault event {index}",
        )
        event_cell = event["cell_index"]
        if (
            isinstance(event_cell, bool)
            or type(event_cell) is not int
            or not 0 <= event_cell < 12
        ):
            raise ValueError("fault event cell index is invalid")
        _finite("anomaly_score", event["anomaly_score"], nonnegative=True)
        _finite("detection_time_s", event["detection_time_s"], nonnegative=True)
        _finite(
            "predicted_hotspot_temperature_k",
            event["predicted_hotspot_temperature_k"],
            nonnegative=True,
        )
        _finite("severity_k", event["severity_k"], nonnegative=True)
        if event_cell == cell_index:
            target_events.append(event)
    detection = lead["detection_time_s"]
    detection_value: float | None = None
    if detection is not None:
        detection_value = _finite("fault detection_time_s", detection, nonnegative=True)
        matching = [
            event for event in target_events if event["detection_time_s"] == detection
        ]
        if not matching:
            raise ValueError("fault detection time has no matching target-cell event")
        suspected_cell: int | None = cell_index
    else:
        suspected_cell = None
        if lead["lead_time_s"] is not None:
            raise ValueError("fault lead time is unavailable without detection")
    if type(summary["detected_fault_cell"]) is not bool or summary[
        "detected_fault_cell"
    ] != (detection is not None):
        raise ValueError("fault detected flag differs from observed detector output")
    lead_value = lead["lead_time_s"]
    if lead_value is not None:
        _finite("fault lead_time_s", lead_value)
        if detection_value is None or crossing_value is None:
            raise ValueError("fault lead time requires detection and crossing")
        if lead_value != crossing_value - detection_value:
            raise ValueError("fault lead time is inconsistent with saved times")
    return FaultDetectionOutcome(
        suspected_cell_index=suspected_cell,
        detection_time_s=detection_value,
        threshold_crossing_time_s=crossing_value,
        lead_time_s=(
            _finite("fault lead_time_s", lead_value) if lead_value is not None else None
        ),
    )


def _direction(left: float, right: float) -> ComparisonDirection:
    if left < right:
        return "data_only_lower"
    if right < left:
        return "physics_informed_lower"
    return "tie"


def _comparisons(
    model_metrics: tuple[ModelEvaluationMetrics, ...],
    forecast_metrics: tuple[FinalForecastMetrics, ...],
) -> tuple[MetricComparison, ...]:
    comparisons: list[MetricComparison] = []
    for distribution in _DISTRIBUTIONS:
        paired = {
            item.model_variant: item
            for item in model_metrics
            if item.distribution == distribution
        }
        pairs = (
            (
                "mae_kelvin",
                paired["data_only"].mean_absolute_error_k,
                paired["physics_informed"].mean_absolute_error_k,
            ),
            (
                "rmse_kelvin",
                paired["data_only"].root_mean_square_error_k,
                paired["physics_informed"].root_mean_square_error_k,
            ),
            (
                "mean_absolute_physics_residual_w",
                paired["data_only"].mean_absolute_physics_residual_w,
                paired["physics_informed"].mean_absolute_physics_residual_w,
            ),
            (
                "root_mean_square_physics_residual_w",
                paired["data_only"].root_mean_square_physics_residual_w,
                paired["physics_informed"].root_mean_square_physics_residual_w,
            ),
        )
        for metric_name, left_value, right_value in pairs:
            comparisons.append(
                MetricComparison(
                    f"{distribution}/one_step",
                    metric_name,
                    left_value,
                    right_value,
                    _direction(left_value, right_value),
                )
            )
    for distribution in _DISTRIBUTIONS:
        for horizon in FORECAST_HORIZONS_S:
            paired_forecast = {
                item.model_variant: item
                for item in forecast_metrics
                if item.distribution == distribution and item.horizon_s == horizon
            }
            forecast_left = paired_forecast["data_only"]
            forecast_right = paired_forecast["physics_informed"]
            comparisons.append(
                MetricComparison(
                    f"{distribution}/{horizon:g}s",
                    "mae_kelvin",
                    forecast_left.mae_kelvin,
                    forecast_right.mae_kelvin,
                    _direction(forecast_left.mae_kelvin, forecast_right.mae_kelvin),
                )
            )
    return tuple(comparisons)


def _source_digest(artifact_id: str, path: Path) -> SourceArtifactDigest:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source artifact is missing or invalid: {artifact_id}")
    return SourceArtifactDigest(artifact_id, _sha256_file(path))


def aggregate_final_evaluation(
    benchmark_directory: str | os.PathLike[str],
    comparison_directory: str | os.PathLike[str],
    id_forecast_directory: str | os.PathLike[str],
    ood_forecast_directory: str | os.PathLike[str],
    fault_detection_directory: str | os.PathLike[str],
) -> FinalEvaluation:
    """Aggregate metrics exclusively from validated persisted experiment outputs."""
    benchmark_root = Path(benchmark_directory)
    matrix = load_benchmark_matrix(benchmark_root)
    comparison_root = Path(comparison_directory)
    model_metric_map, comparison_provenance, manifests, comparison_bytes = (
        _load_comparison_sources(comparison_root, benchmark_root, matrix)
    )
    smoke = comparison_provenance["smoke_test"]
    if type(smoke) is not bool:
        raise ValueError("comparison smoke flag is invalid")
    expected_smoke = smoke

    forecast_metrics = (
        *_forecast_metrics_for_distribution(
            Path(id_forecast_directory),
            "in_distribution",
            expected_smoke,
            cast(int, comparison_provenance["seed"]),
            comparison_bytes,
            manifests["in_distribution"],
        ),
        *_forecast_metrics_for_distribution(
            Path(ood_forecast_directory),
            "out_of_distribution",
            expected_smoke,
            cast(int, comparison_provenance["seed"]),
            comparison_bytes,
            manifests["out_of_distribution"],
        ),
    )
    model_metrics = tuple(
        model_metric_map[(model, distribution)]
        for model in _MODELS
        for distribution in _DISTRIBUTIONS
    )
    controller_metrics = tuple(
        ScenarioControllerMetrics(
            cell.scenario_id,
            cell.variant_id,
            cell.controller_metrics,
        )
        for cell in matrix.cells
    )
    provenance_by_model = {
        cell.model_provenance.model_variant: cell.model_provenance
        for cell in matrix.cells
        if cell.model_provenance is not None
    }
    model_provenance = tuple(provenance_by_model[model] for model in _MODELS)
    fault_detection = _fault_outcome(Path(fault_detection_directory))
    comparisons = _comparisons(model_metrics, forecast_metrics)

    source_artifacts: list[SourceArtifactDigest] = [
        _source_digest("benchmark/manifest.json", benchmark_root / "manifest.json")
    ]
    for name in (
        "provenance.json",
        "data_only.pt",
        "physics_informed.pt",
        "model_config.json",
        "normalization.json",
        "id_comparison_metrics.json",
        "ood_comparison_metrics.json",
    ):
        source_artifacts.append(
            _source_digest(f"comparison/{name}", comparison_root / name)
        )
    for distribution in _DISTRIBUTIONS:
        source_artifacts.append(
            _source_digest(
                f"benchmark/dataset/{distribution}/manifest.json",
                benchmark_root / "dataset" / distribution / "manifest.json",
            )
        )
    for label, directory in (
        ("id", Path(id_forecast_directory)),
        ("ood", Path(ood_forecast_directory)),
    ):
        for name in (
            "provenance.json",
            "data_only_forecast_metrics.json",
            "physics_informed_forecast_metrics.json",
        ):
            source_artifacts.append(
                _source_digest(f"forecast/{label}/{name}", directory / name)
            )
        for artifact_id, relative in _expected_copies().items():
            source_artifacts.append(
                _source_digest(
                    f"forecast/{label}/evidence/{artifact_id}", directory / relative
                )
            )
    for name in ("provenance.json", "fault_summary.json", "anomaly_events.json"):
        source_artifacts.append(
            _source_digest(f"fault/{name}", Path(fault_detection_directory) / name)
        )

    limitations: tuple[str, ...] = (
        "Results describe only the configured synthetic scenarios and saved runs.",
        "No significance, confidence interval, safety certification, or real-world "
        "generalization is established.",
        "Fault artifacts contain no checkpoint/configuration hashes or evidence-mode "
        "field, so their detector model cannot be cryptographically bound to the twin.",
    )
    if matrix.mode == "quick":
        limitations = (
            "Quick inputs are workflow-only evidence; "
            "performance claims are not supported.",
            *limitations,
        )
    dataset_seeds = _mapping(
        comparison_provenance["dataset_seeds"], "comparison dataset seeds"
    )
    return FinalEvaluation(
        evidence_status=(
            "workflow_only"
            if matrix.mode == "quick"
            else "descriptive_synthetic_benchmark"
        ),
        benchmark_mode=matrix.mode,
        benchmark_seed=matrix.cells[0].seed,
        comparison_seed=cast(int, comparison_provenance["seed"]),
        id_dataset_seed=cast(int, dataset_seeds["in_distribution"]),
        ood_dataset_seed=cast(int, dataset_seeds["out_of_distribution"]),
        model_provenance=model_provenance,
        model_metrics=model_metrics,
        forecast_metrics=forecast_metrics,
        controller_metrics=controller_metrics,
        fault_detection=fault_detection,
        comparisons=comparisons,
        source_artifacts=tuple(source_artifacts),
        limitations=limitations,
    )


def _reference_from_dict(value: object, name: str) -> ArtifactReference:
    raw = _exact(value, {"path", "sha256"}, name)
    if type(raw["path"]) is not str or type(raw["sha256"]) is not str:
        raise ValueError(f"{name} path and digest must be strings")
    return ArtifactReference(raw["path"], raw["sha256"])


def _provenance_from_dict(value: object) -> ModelProvenance:
    raw = _exact(
        value,
        {
            "model_variant",
            "training_mode",
            "training_seed",
            "checkpoint",
            "model_config",
            "normalization",
            "comparison_provenance",
            "dataset_manifests",
        },
        "model provenance",
    )
    manifests = raw["dataset_manifests"]
    if not isinstance(manifests, list):
        raise ValueError("dataset_manifests must be an array")
    return ModelProvenance(
        model_variant=cast(ModelVariant, raw["model_variant"]),
        training_mode=cast(TrainingMode, raw["training_mode"]),
        training_seed=raw["training_seed"],  # type: ignore[arg-type]
        checkpoint=_reference_from_dict(raw["checkpoint"], "checkpoint"),
        model_config=_reference_from_dict(raw["model_config"], "model_config"),
        normalization=_reference_from_dict(raw["normalization"], "normalization"),
        comparison_provenance=_reference_from_dict(
            raw["comparison_provenance"], "comparison_provenance"
        ),
        dataset_manifests=tuple(
            _reference_from_dict(item, "dataset manifest") for item in manifests
        ),
    )


def _evaluation_from_dict(value: object) -> FinalEvaluation:
    raw = _exact(value, _ROOT_FIELDS, "final evaluation")
    if raw["schema_version"] != _SERIALIZED_SCHEMA_VERSION:
        raise ValueError("final evaluation schema is unsupported")
    model_values, forecast_values = raw["model_metrics"], raw["forecast_metrics"]
    controller_values = raw["controller_metrics"]
    provenance_values = raw["model_provenance"]
    comparison_values = raw["comparisons"]
    digest_values, limitation_values = raw["source_artifacts"], raw["limitations"]
    if not all(
        isinstance(items, list)
        for items in (
            model_values,
            forecast_values,
            controller_values,
            provenance_values,
            comparison_values,
            digest_values,
            limitation_values,
        )
    ):
        raise ValueError("final evaluation collections must be arrays")

    model_metrics: list[ModelEvaluationMetrics] = []
    for item in cast(list[object], model_values):
        fields = _exact(
            item,
            {
                "model_variant",
                "distribution",
                "mean_absolute_error_k",
                "root_mean_square_error_k",
                "maximum_cell_error_k",
                "mean_absolute_physics_residual_w",
                "root_mean_square_physics_residual_w",
                "maximum_absolute_physics_residual_w",
                "sample_count",
            },
            "model metric",
        )
        model_metrics.append(ModelEvaluationMetrics(**fields))  # type: ignore[arg-type]

    forecast_metrics: list[FinalForecastMetrics] = []
    for item in cast(list[object], forecast_values):
        fields = _exact(
            item,
            {
                "model_variant",
                "distribution",
                "horizon_s",
                "mae_kelvin",
                "rmse_kelvin",
                "maximum_absolute_cell_error_kelvin",
                "forecast_count",
                "step_count",
            },
            "forecast metric",
        )
        forecast_metrics.append(FinalForecastMetrics(**fields))  # type: ignore[arg-type]

    controller_metrics: list[ScenarioControllerMetrics] = []
    for item in cast(list[object], controller_values):
        fields = _exact(
            item, {"scenario_id", "variant_id", "metrics"}, "controller metric"
        )
        metric = _exact(
            fields["metrics"],
            {
                "peak_temperature_k",
                "mean_temperature_k",
                "maximum_temperature_spread_k",
                "pump_energy_j",
                "thermal_violation_count",
                "thermal_violation_duration_s",
            },
            "controller metric values",
        )
        controller_metrics.append(
            ScenarioControllerMetrics(
                cast(ScenarioId, fields["scenario_id"]),
                cast(VariantId, fields["variant_id"]),
                ControllerMetrics(**metric),  # type: ignore[arg-type]
            )
        )

    comparisons: list[MetricComparison] = []
    for item in cast(list[object], comparison_values):
        fields = _exact(
            item,
            {
                "scope",
                "metric_name",
                "data_only_value",
                "physics_informed_value",
                "lower_value_variant",
            },
            "metric comparison",
        )
        comparisons.append(MetricComparison(**fields))  # type: ignore[arg-type]

    source_artifacts: list[SourceArtifactDigest] = []
    for item in cast(list[object], digest_values):
        fields = _exact(item, {"artifact_id", "sha256"}, "source artifact digest")
        source_artifacts.append(SourceArtifactDigest(**fields))  # type: ignore[arg-type]

    fault_fields = _exact(
        raw["fault_detection"],
        {
            "suspected_cell_index",
            "detection_time_s",
            "threshold_crossing_time_s",
            "lead_time_s",
        },
        "fault detection outcome",
    )
    return FinalEvaluation(
        evidence_status=cast(EvidenceStatus, raw["evidence_status"]),
        benchmark_mode=cast(BenchmarkMode, raw["benchmark_mode"]),
        benchmark_seed=raw["benchmark_seed"],  # type: ignore[arg-type]
        comparison_seed=raw["comparison_seed"],  # type: ignore[arg-type]
        id_dataset_seed=raw["id_dataset_seed"],  # type: ignore[arg-type]
        ood_dataset_seed=raw["ood_dataset_seed"],  # type: ignore[arg-type]
        model_provenance=tuple(
            _provenance_from_dict(item)
            for item in cast(list[object], provenance_values)
        ),
        model_metrics=tuple(model_metrics),
        forecast_metrics=tuple(forecast_metrics),
        controller_metrics=tuple(controller_metrics),
        fault_detection=FaultDetectionOutcome(**fault_fields),  # type: ignore[arg-type]
        comparisons=tuple(comparisons),
        source_artifacts=tuple(source_artifacts),
        limitations=tuple(cast(list[str], limitation_values)),
    )


def save_final_evaluation(
    evaluation: FinalEvaluation,
    output_directory: str | os.PathLike[str],
) -> Path:
    """Atomically save a validated final evaluation into a new directory."""
    if type(evaluation) is not FinalEvaluation:
        raise TypeError("evaluation must be exactly a FinalEvaluation")
    requested = Path(output_directory)
    if requested.name in ("", ".", "..") or ".partial-" in requested.name:
        raise ValueError("output_directory must be a final result directory")
    parent = requested.parent.resolve()
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"final evaluation already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".{requested.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(exist_ok=False)
    try:
        content = {"schema_version": _SERIALIZED_SCHEMA_VERSION, **asdict(evaluation)}
        serialized = (
            json.dumps(content, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
        artifact = stage / _ARTIFACT_NAME
        with artifact.open("x", encoding="utf-8", newline="") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(stage, destination)
        return destination
    except BaseException:
        if stage.exists() and stage.parent == parent and ".partial-" in stage.name:
            shutil.rmtree(stage)
        raise


def load_final_evaluation(
    directory: str | os.PathLike[str],
) -> FinalEvaluation:
    """Strictly load a final evaluation directory containing one JSON file."""
    root = Path(directory)
    if ".partial-" in root.name or root.is_symlink() or not root.is_dir():
        raise ValueError("final evaluation must be a complete regular directory")
    entries = {path.name for path in root.iterdir()}
    if entries != {_ARTIFACT_NAME}:
        raise ValueError(
            "final evaluation directory contains missing or unexpected files"
        )
    raw = _mapping(_read_json(root / _ARTIFACT_NAME), "final evaluation artifact")
    return _evaluation_from_dict(raw)


__all__ = [
    "FinalEvaluation",
    "FinalForecastMetrics",
    "MetricComparison",
    "ScenarioControllerMetrics",
    "SourceArtifactDigest",
    "aggregate_final_evaluation",
    "load_final_evaluation",
    "save_final_evaluation",
]
