"""Portable, tamper-detecting persistence for complete benchmark executions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkCellResult,
    BenchmarkMatrix,
    ControllerMetrics,
    FaultDetectionOutcome,
    ForecastAssumptions,
    ModelProvenance,
    ModelVariant,
    TrainingMode,
)
from thermotwin.experiments.serialization import save_experiment_result
from thermotwin.models.config import MLPConfig
from thermotwin.models.mlp import ThermalMLP

if TYPE_CHECKING:
    import os as os_types

    from thermotwin.experiments.benchmark import BenchmarkExecution

_SCHEMA_VERSION = 2
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_METRIC_FIELDS = {
    "peak_temperature_k",
    "mean_temperature_k",
    "maximum_temperature_spread_k",
    "pump_energy_j",
    "thermal_violation_count",
    "thermal_violation_duration_s",
}
_PROVENANCE_FIELDS = {
    "model_variant",
    "training_mode",
    "training_seed",
    "checkpoint",
    "model_config",
    "normalization",
    "comparison_provenance",
    "dataset_manifests",
}
_ASSUMPTION_FIELDS = {
    "current_assumption",
    "soc_assumption",
    "ambient_temperature_assumption",
    "coolant_inlet_temperature_assumption",
    "action_handling",
    "uses_future_simulator_truth",
}
_FAULT_FIELDS = {
    "suspected_cell_index",
    "detection_time_s",
    "threshold_crossing_time_s",
    "lead_time_s",
}
_CELL_FIELDS = {
    "scenario_id",
    "variant_id",
    "seed",
    "scenario_configuration_sha256",
    "controller_metrics",
    "model_provenance",
    "forecast_assumptions",
    "fault_detection",
    "artifacts",
    "artifact_sha256",
}
_ARTIFACTS = ("timeseries", "metrics", "provenance")
_EVIDENCE_PATHS = (
    "comparison/data_only.pt",
    "comparison/physics_informed.pt",
    "comparison/model_config.json",
    "comparison/normalization.json",
    "comparison/provenance.json",
    "dataset/in_distribution/manifest.json",
    "dataset/out_of_distribution/manifest.json",
)


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is prohibited: {value}")


def _read_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_json_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid benchmark JSON file: {path.name}") from error


def _exact_fields(value: object, fields: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )
    return cast(dict[str, object], value)


def _safe_json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_path(scenario_id: str, variant_id: str, artifact: str) -> str:
    extension = "csv" if artifact == "timeseries" else "json"
    return f"runs/{scenario_id}/{variant_id}/{artifact}.{extension}"


def _manifest_cell(
    cell: BenchmarkCellResult,
    paths: Mapping[str, str],
    hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "scenario_id": cell.scenario_id,
        "variant_id": cell.variant_id,
        "seed": cell.seed,
        "scenario_configuration_sha256": cell.scenario_configuration_sha256,
        "controller_metrics": asdict(cell.controller_metrics),
        "model_provenance": (
            None if cell.model_provenance is None else asdict(cell.model_provenance)
        ),
        "forecast_assumptions": (
            None
            if cell.forecast_assumptions is None
            else asdict(cell.forecast_assumptions)
        ),
        "fault_detection": (
            None if cell.fault_detection is None else asdict(cell.fault_detection)
        ),
        "artifacts": dict(paths),
        "artifact_sha256": dict(hashes),
    }


def _canonical_evidence_paths(provenance: ModelProvenance) -> tuple[str, ...]:
    expected = (
        f"comparison/{provenance.model_variant}.pt",
        "comparison/model_config.json",
        "comparison/normalization.json",
        "comparison/provenance.json",
        "dataset/in_distribution/manifest.json",
        "dataset/out_of_distribution/manifest.json",
    )
    actual = (
        provenance.checkpoint.path,
        provenance.model_config.path,
        provenance.normalization.path,
        provenance.comparison_provenance.path,
        *(reference.path for reference in provenance.dataset_manifests),
    )
    if actual != expected:
        raise ValueError("model provenance references are not canonical bundle paths")
    return expected


def _validate_evidence_payloads(
    matrix: BenchmarkMatrix, evidence_files: Mapping[str, bytes]
) -> dict[str, str]:
    if set(evidence_files) != set(_EVIDENCE_PATHS):
        raise ValueError("evidence artifacts must contain the exact required paths")
    if any(type(payload) is not bytes for payload in evidence_files.values()):
        raise ValueError("evidence artifact payloads must be bytes")
    hashes = {
        path: hashlib.sha256(evidence_files[path]).hexdigest()
        for path in _EVIDENCE_PATHS
    }
    seen_variants: set[str] = set()
    for cell in matrix.cells:
        provenance = cell.model_provenance
        if provenance is None:
            continue
        paths = _canonical_evidence_paths(provenance)
        references = (
            provenance.checkpoint,
            provenance.model_config,
            provenance.normalization,
            provenance.comparison_provenance,
            *provenance.dataset_manifests,
        )
        if any(
            reference.sha256 != hashes[path]
            for reference, path in zip(references, paths, strict=True)
        ):
            raise ValueError("model provenance digest does not match evidence bytes")
        seen_variants.add(provenance.model_variant)
    if seen_variants != {"data_only", "physics_informed"}:
        raise ValueError("benchmark evidence must cover both model variants")
    return hashes


def save_benchmark_execution(
    execution: BenchmarkExecution,
    output_directory: str | os_types.PathLike[str],
    *,
    evidence_files: Mapping[str, bytes],
) -> Path:
    """Atomically publish a new complete benchmark bundle without overwrite."""
    from thermotwin.experiments.benchmark import BenchmarkExecution

    if type(execution) is not BenchmarkExecution:
        raise TypeError("execution must be exactly a BenchmarkExecution")
    evidence_hashes = _validate_evidence_payloads(execution.matrix, evidence_files)
    try:
        requested = Path(output_directory)
    except (TypeError, ValueError) as error:
        raise ValueError("output_directory must be a filesystem path") from error
    if requested.name in ("", ".", "..") or ".partial-" in requested.name:
        raise ValueError("output_directory must be a final bundle directory")

    # Resolve the parent so the stage is always a sibling of the exact target.
    parent = requested.parent.resolve()
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"benchmark output already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"benchmark output already exists: {destination}")

    stage = parent / f".{requested.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(exist_ok=False)
    try:
        for relative_path in _EVIDENCE_PATHS:
            evidence_path = stage / PurePosixPath(relative_path)
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_path.write_bytes(evidence_files[relative_path])
        cells_by_key = {
            (cell.scenario_id, cell.variant_id): cell for cell in execution.matrix.cells
        }
        manifest_cells: list[dict[str, object]] = []
        for scenario_id in SCENARIO_IDS:
            for variant_id in VARIANT_IDS:
                key = (scenario_id, variant_id)
                result = execution.results[key]
                run_root = stage / "runs" / scenario_id
                save_experiment_result(result, run_root, variant_id)
                paths = {
                    artifact: _artifact_path(scenario_id, variant_id, artifact)
                    for artifact in _ARTIFACTS
                }
                hashes = {
                    artifact: _sha256(stage / PurePosixPath(paths[artifact]))
                    for artifact in _ARTIFACTS
                }
                manifest_cells.append(
                    _manifest_cell(cells_by_key[key], paths, hashes)
                )

        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "complete": True,
            "mode": execution.matrix.mode,
            "evidence": evidence_hashes,
            "cells": manifest_cells,
        }
        (stage / "manifest.json").write_text(
            _safe_json_text(manifest), encoding="utf-8", newline=""
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"benchmark output already exists: {destination}")
        os.rename(stage, destination)
        return destination
    except BaseException:
        if stage.exists() and stage.parent == parent and ".partial-" in stage.name:
            shutil.rmtree(stage)
        raise


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_numeric_fields(
    value: dict[str, object], names: tuple[str, ...], *, positive: bool = False
) -> None:
    for name in names:
        number = _finite_number(value[name], name)
        if positive and number <= 0.0:
            raise ValueError(f"{name} must be positive")


def _validate_saved_environment(value: object) -> None:
    environment = _exact_fields(
        value,
        {"pack", "cooling", "simulation", "resistance_fault"},
        "saved environment configuration",
    )
    pack = _exact_fields(
        environment["pack"],
        {
            "rows",
            "columns",
            "cell_mass_kg",
            "cell_specific_heat_capacity_j_kg_k",
            "nominal_cell_resistance_ohm",
            "cell_capacity_ah",
            "neighbor_thermal_resistance_k_w",
            "ambient_thermal_resistance_k_w",
            "cell_cooling_surface_area_m2",
        },
        "saved pack configuration",
    )
    if pack["rows"] != 3 or pack["columns"] != 4:
        raise ValueError("saved benchmark pack must contain exactly 12 cells")
    _validate_numeric_fields(
        pack,
        (
            "cell_mass_kg",
            "cell_specific_heat_capacity_j_kg_k",
            "nominal_cell_resistance_ohm",
            "cell_capacity_ah",
            "neighbor_thermal_resistance_k_w",
            "ambient_thermal_resistance_k_w",
            "cell_cooling_surface_area_m2",
        ),
        positive=True,
    )
    cooling = _exact_fields(
        environment["cooling"],
        {
            "minimum_coolant_mass_flow_kg_s",
            "maximum_coolant_mass_flow_kg_s",
            "base_heat_transfer_coefficient_w_m2_k",
            "flow_heat_transfer_gain",
            "flow_exponent",
        },
        "saved cooling configuration",
    )
    minimum_flow = _finite_number(
        cooling["minimum_coolant_mass_flow_kg_s"],
        "minimum_coolant_mass_flow_kg_s",
    )
    maximum_flow = _finite_number(
        cooling["maximum_coolant_mass_flow_kg_s"],
        "maximum_coolant_mass_flow_kg_s",
    )
    if minimum_flow > maximum_flow:
        raise ValueError("saved coolant limits are inconsistent")
    _validate_numeric_fields(
        cooling,
        (
            "base_heat_transfer_coefficient_w_m2_k",
            "flow_heat_transfer_gain",
            "flow_exponent",
        ),
        positive=True,
    )
    simulation = _exact_fields(
        environment["simulation"],
        {
            "default_time_step_s",
            "initial_soc_fraction",
            "initial_cell_temperature_k",
            "ambient_temperature_k",
            "coolant_inlet_temperature_k",
            "initial_current_a",
            "initial_cell_temperatures_k",
            "initial_cell_resistances_ohm",
        },
        "saved simulation configuration",
    )
    _validate_numeric_fields(
        simulation,
        (
            "default_time_step_s",
            "initial_cell_temperature_k",
            "ambient_temperature_k",
            "coolant_inlet_temperature_k",
        ),
        positive=True,
    )
    _finite_number(simulation["initial_current_a"], "initial_current_a")
    soc = _finite_number(simulation["initial_soc_fraction"], "initial SOC")
    if not 0.0 <= soc <= 1.0:
        raise ValueError("saved initial SOC must be in [0, 1]")
    for vector_name in (
        "initial_cell_temperatures_k",
        "initial_cell_resistances_ohm",
    ):
        vector = simulation[vector_name]
        if vector is not None:
            if not isinstance(vector, list) or len(vector) != 12:
                raise ValueError(f"{vector_name} must be null or contain 12 values")
            for item in vector:
                if _finite_number(item, vector_name) <= 0.0:
                    raise ValueError(f"{vector_name} values must be positive")
    fault = environment["resistance_fault"]
    if fault is not None:
        fault_values = _exact_fields(
            fault,
            {"cell_index", "start_time_s", "resistance_multiplier"},
            "saved resistance fault",
        )
        cell_index = fault_values["cell_index"]
        if (
            isinstance(cell_index, bool)
            or not isinstance(cell_index, int)
            or not 0 <= cell_index < 12
        ):
            raise ValueError("saved resistance fault cell index is invalid")
        _finite_number(fault_values["start_time_s"], "fault start time")
        multiplier = _finite_number(
            fault_values["resistance_multiplier"], "fault multiplier"
        )
        if multiplier <= 1.0:
            raise ValueError("saved resistance multiplier must exceed one")


def _parse_metrics(value: object) -> ControllerMetrics:
    values = _exact_fields(value, _METRIC_FIELDS, "controller_metrics")
    count = values["thermal_violation_count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("thermal_violation_count must be an integer")
    return ControllerMetrics(
        peak_temperature_k=_finite_number(values["peak_temperature_k"], "peak"),
        mean_temperature_k=_finite_number(values["mean_temperature_k"], "mean"),
        maximum_temperature_spread_k=_finite_number(
            values["maximum_temperature_spread_k"], "spread"
        ),
        pump_energy_j=_finite_number(values["pump_energy_j"], "pump energy"),
        thermal_violation_count=count,
        thermal_violation_duration_s=_finite_number(
            values["thermal_violation_duration_s"], "violation duration"
        ),
    )


def _parse_reference(value: object, name: str) -> ArtifactReference:
    values = _exact_fields(value, {"path", "sha256"}, name)
    if not isinstance(values["path"], str) or not isinstance(values["sha256"], str):
        raise ValueError(f"{name} values must be strings")
    return ArtifactReference(values["path"], values["sha256"])


def _parse_provenance(value: object) -> ModelProvenance | None:
    if value is None:
        return None
    values = _exact_fields(value, _PROVENANCE_FIELDS, "model_provenance")
    manifests_value = values["dataset_manifests"]
    if not isinstance(manifests_value, list):
        raise ValueError("dataset_manifests must be an array")
    return ModelProvenance(
        model_variant=cast(ModelVariant, values["model_variant"]),
        training_mode=cast(TrainingMode, values["training_mode"]),
        training_seed=values["training_seed"],  # type: ignore[arg-type]
        checkpoint=_parse_reference(values["checkpoint"], "checkpoint"),
        model_config=_parse_reference(values["model_config"], "model_config"),
        normalization=_parse_reference(values["normalization"], "normalization"),
        comparison_provenance=_parse_reference(
            values["comparison_provenance"], "comparison_provenance"
        ),
        dataset_manifests=tuple(
            _parse_reference(item, "dataset manifest") for item in manifests_value
        ),
    )


def _parse_assumptions(value: object) -> ForecastAssumptions | None:
    if value is None:
        return None
    values = _exact_fields(value, _ASSUMPTION_FIELDS, "forecast_assumptions")
    return ForecastAssumptions(**values)  # type: ignore[arg-type]


def _parse_fault(value: object) -> FaultDetectionOutcome | None:
    if value is None:
        return None
    values = _exact_fields(value, _FAULT_FIELDS, "fault_detection")
    return FaultDetectionOutcome(**values)  # type: ignore[arg-type]


def _expected_inventory() -> set[str]:
    inventory = {"manifest.json", "runs", "comparison", "dataset"}
    inventory.update(_EVIDENCE_PATHS)
    inventory.update(
        {
            "dataset/in_distribution",
            "dataset/out_of_distribution",
        }
    )
    for scenario_id in SCENARIO_IDS:
        scenario_path = f"runs/{scenario_id}"
        inventory.add(scenario_path)
        for variant_id in VARIANT_IDS:
            run_path = f"{scenario_path}/{variant_id}"
            inventory.add(run_path)
            inventory.update(
                f"{run_path}/{artifact}."
                f"{('csv' if artifact == 'timeseries' else 'json')}"
                for artifact in _ARTIFACTS
            )
    return inventory


def _inventory(root: Path) -> set[str]:
    found: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("benchmark bundle must not contain symlinks")
            found.add(path.relative_to(root).as_posix())
    return found


def _validate_run_files(
    root: Path,
    scenario_id: str,
    variant_id: str,
    cell: dict[str, object],
    metrics: ControllerMetrics,
    provenance: ModelProvenance | None,
) -> None:
    artifacts = _exact_fields(cell["artifacts"], set(_ARTIFACTS), "artifacts")
    hashes = _exact_fields(cell["artifact_sha256"], set(_ARTIFACTS), "artifact_sha256")
    for artifact in _ARTIFACTS:
        expected_path = _artifact_path(scenario_id, variant_id, artifact)
        path_value = artifacts[artifact]
        if path_value != expected_path or not isinstance(path_value, str):
            raise ValueError("artifact paths must be canonical relative paths")
        pure = PurePosixPath(path_value)
        if pure.is_absolute() or ".." in pure.parts or "\\" in path_value:
            raise ValueError("unsafe artifact path")
        expected_hash = hashes[artifact]
        if (
            not isinstance(expected_hash, str)
            or _DIGEST_RE.fullmatch(expected_hash) is None
        ):
            raise ValueError("artifact SHA-256 must be lowercase hexadecimal")
        artifact_file = root.joinpath(*pure.parts)
        if not artifact_file.is_file() or artifact_file.is_symlink():
            raise ValueError(f"missing artifact: {path_value}")
        if _sha256(artifact_file) != expected_hash:
            raise ValueError(f"artifact digest mismatch: {path_value}")

    metrics_file = root / PurePosixPath(cast(str, artifacts["metrics"]))
    stored_metrics = _exact_fields(
        _read_json(metrics_file),
        {
            "peak_temperature_k",
            "mean_temperature_k",
            "maximum_temperature_spread_k",
            "pump_energy_j",
            "thermal_violation_count",
            "time_above_thermal_limit_s",
        },
        "saved ExperimentMetrics",
    )
    for field in (
        "peak_temperature_k",
        "mean_temperature_k",
        "maximum_temperature_spread_k",
        "pump_energy_j",
        "time_above_thermal_limit_s",
    ):
        _finite_number(stored_metrics[field], f"saved metric {field}")
    stored_count = stored_metrics["thermal_violation_count"]
    if isinstance(stored_count, bool) or not isinstance(stored_count, int):
        raise ValueError("saved thermal_violation_count must be an integer")
    expected_metric_values = {
        "peak_temperature_k": metrics.peak_temperature_k,
        "mean_temperature_k": metrics.mean_temperature_k,
        "maximum_temperature_spread_k": metrics.maximum_temperature_spread_k,
        "pump_energy_j": metrics.pump_energy_j,
        "thermal_violation_count": metrics.thermal_violation_count,
        "time_above_thermal_limit_s": metrics.thermal_violation_duration_s,
    }
    if stored_metrics != expected_metric_values:
        raise ValueError("saved run metrics do not match manifest metrics")

    provenance_file = root / PurePosixPath(cast(str, artifacts["provenance"]))
    saved = _exact_fields(
        _read_json(provenance_file),
        {
            "schema_version",
            "scenario",
            "environment_config",
            "scenario_metadata",
            "controller_class_name",
            "seed",
            "result_metadata",
            "record_count",
        },
        "saved run provenance",
    )
    scenario = _exact_fields(
        saved["scenario"],
        {
            "name",
            "duration_s",
            "time_step_s",
            "thermal_limit_k",
            "pump_power_coefficient_w_s3_kg3",
        },
        "saved scenario provenance",
    )
    if saved["schema_version"] != "1.0" or scenario["name"] != scenario_id:
        raise ValueError("saved scenario provenance is inconsistent")
    _validate_numeric_fields(
        scenario,
        (
            "duration_s",
            "time_step_s",
            "thermal_limit_k",
        ),
        positive=True,
    )
    pump_coefficient = _finite_number(
        scenario["pump_power_coefficient_w_s3_kg3"], "pump coefficient"
    )
    if pump_coefficient < 0.0:
        raise ValueError("saved pump coefficient must be nonnegative")
    record_count = saved["record_count"]
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
    ):
        raise ValueError("saved run record count is invalid")
    if saved["seed"] != cell["seed"]:
        raise ValueError("saved seed or record count is invalid")
    metadata = saved["scenario_metadata"]
    result_metadata = saved["result_metadata"]
    if not isinstance(metadata, dict) or not isinstance(result_metadata, dict):
        raise ValueError("saved scenario metadata must be objects")
    if metadata.get("mode") not in ("quick", "full") or result_metadata != metadata:
        raise ValueError("saved scenario metadata is inconsistent")
    _validate_saved_environment(saved["environment_config"])
    if not isinstance(saved["controller_class_name"], str):
        raise ValueError("controller class provenance is invalid")
    expected_class = {
        "threshold": "ThresholdController",
        "pid": "PIDController",
        "predictive_data_only": "PredictiveController",
        "predictive_physics_informed": "PredictiveController",
    }[variant_id]
    if saved["controller_class_name"] != expected_class:
        raise ValueError("controller provenance does not match matrix variant")
    if variant_id.startswith("predictive_"):
        if provenance is None:
            raise ValueError("predictive matrix cell must have model provenance")
        if provenance.model_variant != variant_id.removeprefix("predictive_"):
            raise ValueError("predictive model provenance does not match variant")
    elif provenance is not None:
        raise ValueError("reactive matrix cell cannot have model provenance")


def _validate_dataset_manifest(value: object, label: str) -> dict[str, object]:
    manifest = _exact_fields(
        value,
        {
            "schema_version",
            "seed",
            "validation_fraction",
            "test_fraction",
            "cell_count",
            "columns",
            "splits",
        },
        f"{label} dataset manifest",
    )
    if manifest["schema_version"] != "1.0" or manifest["cell_count"] != 12:
        raise ValueError(f"{label} dataset manifest is incompatible")
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        raise ValueError(f"{label} dataset manifest seed is invalid")
    validation_fraction = _finite_number(
        manifest["validation_fraction"], f"{label} validation fraction"
    )
    test_fraction = _finite_number(manifest["test_fraction"], f"{label} test fraction")
    if (
        validation_fraction < 0.0
        or test_fraction < 0.0
        or validation_fraction + test_fraction >= 1.0
    ):
        raise ValueError(f"{label} dataset split fractions are invalid")
    columns = manifest["columns"]
    if (
        not isinstance(columns, list)
        or len(columns) < 13
        or any(type(column) is not str for column in columns)
    ):
        raise ValueError(f"{label} dataset manifest columns are invalid")
    splits = _exact_fields(
        manifest["splits"], {"train", "validation", "test"}, f"{label} splits"
    )
    for split_name, split_value in splits.items():
        split = _exact_fields(
            split_value,
            {"row_count", "trajectory_count", "trajectory_ids"},
            f"{label} {split_name} split",
        )
        row_count, trajectory_count = split["row_count"], split["trajectory_count"]
        ids = split["trajectory_ids"]
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
            or isinstance(trajectory_count, bool)
            or not isinstance(trajectory_count, int)
            or trajectory_count < 1
            or not isinstance(ids, list)
            or len(ids) != trajectory_count
            or any(type(item) is not str or not item for item in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"{label} {split_name} split is invalid")
    return manifest


def _validate_embedded_evidence(
    root: Path,
    evidence: dict[str, object],
    provenances: tuple[ModelProvenance, ...],
) -> None:
    if set(evidence) != set(_EVIDENCE_PATHS):
        raise ValueError("benchmark evidence inventory is incomplete or unexpected")
    evidence_bytes: dict[str, bytes] = {}
    for path in _EVIDENCE_PATHS:
        pure = PurePosixPath(path)
        digest = evidence[path]
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in path
            or pure.as_posix() != path
            or not isinstance(digest, str)
            or _DIGEST_RE.fullmatch(digest) is None
        ):
            raise ValueError("benchmark evidence paths or digests are invalid")
        artifact = root.joinpath(*pure.parts)
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"missing evidence artifact: {path}")
        if _sha256(artifact) != digest:
            raise ValueError(f"evidence artifact digest mismatch: {path}")
        evidence_bytes[path] = artifact.read_bytes()

    if len(provenances) == 0:
        raise ValueError("benchmark evidence has no predictive model provenance")
    provenance_by_variant: dict[str, ModelProvenance] = {}
    for provenance in provenances:
        previous = provenance_by_variant.setdefault(
            provenance.model_variant, provenance
        )
        if previous != provenance:
            raise ValueError("model provenance differs across benchmark scenarios")
        paths = _canonical_evidence_paths(provenance)
        references = (
            provenance.checkpoint,
            provenance.model_config,
            provenance.normalization,
            provenance.comparison_provenance,
            *provenance.dataset_manifests,
        )
        if any(
            hashlib.sha256(evidence_bytes[path]).hexdigest() != reference.sha256
            for reference, path in zip(references, paths, strict=True)
        ):
            raise ValueError("model provenance digest does not match bundle evidence")

    config = _exact_fields(
        _read_json(root / "comparison/model_config.json"),
        {"input_dim", "output_dim", "hidden_widths", "activation"},
        "embedded model configuration",
    )
    try:
        model_config = MLPConfig(
            input_dim=config["input_dim"],  # type: ignore[arg-type]
            output_dim=config["output_dim"],  # type: ignore[arg-type]
            hidden_widths=tuple(config["hidden_widths"]),  # type: ignore[arg-type]
            activation=config["activation"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as error:
        raise ValueError("embedded model configuration is invalid") from error
    if model_config.input_dim != 17 or model_config.output_dim != 12:
        raise ValueError("embedded model configuration is incompatible with 12 cells")

    normalization = _exact_fields(
        _read_json(root / "comparison/normalization.json"),
        {
            "schema_version",
            "feature_names",
            "target_names",
            "feature_mean",
            "feature_scale",
            "target_mean",
            "target_scale",
        },
        "embedded normalization",
    )
    if type(normalization["schema_version"]) is not int or normalization[
        "schema_version"
    ] != 1:
        raise ValueError("embedded normalization schema is unsupported")
    try:
        stats = NormalizationStats(
            feature_names=tuple(normalization["feature_names"]),  # type: ignore[arg-type]
            target_names=tuple(normalization["target_names"]),  # type: ignore[arg-type]
            feature_mean=np.asarray(normalization["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(normalization["feature_scale"], dtype=np.float64),
            target_mean=np.asarray(normalization["target_mean"], dtype=np.float64),
            target_scale=np.asarray(normalization["target_scale"], dtype=np.float64),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("embedded normalization is invalid") from error
    if (
        stats.feature_names != feature_names(12)
        or stats.target_names != target_names(12)
        or len(stats.feature_mean) != model_config.input_dim
        or len(stats.target_mean) != model_config.output_dim
    ):
        raise ValueError("embedded normalization is incompatible with model schema")

    for variant in ("data_only", "physics_informed"):
        checkpoint_path = root / f"comparison/{variant}.pt"
        try:
            state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if (
                not isinstance(state, dict)
                or not state
                or any(
                    not isinstance(key, str) or not isinstance(value, torch.Tensor)
                    for key, value in state.items()
                )
            ):
                raise ValueError("checkpoint is not a tensor-only state_dict")
            ThermalMLP(model_config).load_state_dict(state, strict=True)
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"embedded {variant} checkpoint is incompatible"
            ) from error

    manifests: dict[str, dict[str, object]] = {}
    for label in ("in_distribution", "out_of_distribution"):
        path = f"dataset/{label}/manifest.json"
        manifests[label] = _validate_dataset_manifest(
            _read_json(root / path), label
        )
    comparison = _exact_fields(
        _read_json(root / "comparison/provenance.json"),
        {
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
        },
        "embedded comparison provenance",
    )
    datasets = _exact_fields(
        comparison["datasets"],
        {"in_distribution", "out_of_distribution"},
        "embedded comparison datasets",
    )
    dataset_seeds = _exact_fields(
        comparison["dataset_seeds"],
        {"in_distribution", "out_of_distribution"},
        "embedded dataset seeds",
    )
    for label, manifest in manifests.items():
        expected_fields = {
            "directory",
            "manifest_path",
            "manifest",
            "distribution",
        } | ({"normalization_source"} if label == "out_of_distribution" else set())
        dataset = _exact_fields(
            datasets[label], expected_fields, f"embedded {label} dataset"
        )
        if dataset["manifest"] != manifest:
            raise ValueError("embedded dataset provenance does not match manifest")
        if dataset["directory"] != f"dataset/{label}" or dataset[
            "manifest_path"
        ] != f"dataset/{label}/manifest.json":
            raise ValueError("embedded dataset paths are not normalized")
        expected_dataset_seed = dataset_seeds[label]
        if (
            isinstance(expected_dataset_seed, bool)
            or not isinstance(expected_dataset_seed, int)
            or expected_dataset_seed < 0
        ):
            raise ValueError("embedded dataset seed is invalid")
        if manifest["seed"] != expected_dataset_seed:
            raise ValueError("embedded dataset manifest seed is inconsistent")
        if dataset["distribution"] != label:
            raise ValueError("embedded dataset distribution is inconsistent")
        if label == "out_of_distribution" and dataset[
            "normalization_source"
        ] != "in_distribution":
            raise ValueError("embedded OOD normalization source is inconsistent")
    package_versions = cast(dict[object, object], comparison["package_versions"])
    if (
        comparison["artifact_schema_version"] != "1.0"
        or isinstance(comparison["seed"], bool)
        or not isinstance(comparison["seed"], int)
        or comparison["seed"] < 0
        or comparison["distribution_labels"]
        != ["in_distribution", "out_of_distribution"]
        or type(comparison["smoke_test"]) is not bool
        or not isinstance(comparison["package_versions"], dict)
        or not comparison["package_versions"]
        or any(
            type(name) is not str
            or not name
            or type(version) is not str
            or not version
            for name, version in package_versions.items()
        )
        or type(comparison["config_source"]) is not str
        or not comparison["config_source"]
        or type(comparison["configuration_source_path"]) is not str
        or not comparison["configuration_source_path"]
        or comparison["normalization"]
        != {
            "artifact": "normalization.json",
            "fit_dataset": "in_distribution",
            "reused_for": ["in_distribution", "out_of_distribution"],
        }
    ):
        raise ValueError("embedded comparison provenance is incompatible")
    training_mode = "smoke" if comparison["smoke_test"] else "full"
    if any(item.training_mode != training_mode for item in provenances):
        raise ValueError("model training mode does not match comparison provenance")
    if any(item.training_seed != comparison["seed"] for item in provenances):
        raise ValueError("model training seed does not match comparison provenance")


def load_benchmark_matrix(directory: str | os_types.PathLike[str]) -> BenchmarkMatrix:
    """Load and validate the exact portable artifact set without executing it."""
    root = Path(directory)
    if ".partial-" in root.name:
        raise ValueError("partial benchmark directories cannot be loaded")
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ValueError("benchmark directory must be an existing regular directory")
    if _inventory(root) != _expected_inventory():
        raise ValueError("benchmark directory contains missing or unexpected files")

    manifest = _exact_fields(
        _read_json(root / "manifest.json"),
        {"schema_version", "complete", "mode", "evidence", "cells"},
        "benchmark manifest",
    )
    if (
        manifest["schema_version"] != _SCHEMA_VERSION
        or manifest["complete"] is not True
    ):
        raise ValueError("benchmark manifest is unsupported or incomplete")
    mode = manifest["mode"]
    if mode not in ("quick", "full"):
        raise ValueError("benchmark manifest mode is invalid")
    raw_cells = manifest["cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) != 24:
        raise ValueError("benchmark manifest must contain exactly 24 cells")
    evidence = _exact_fields(
        manifest["evidence"], set(_EVIDENCE_PATHS), "benchmark evidence inventory"
    )

    cells: list[BenchmarkCellResult] = []
    provenances: list[ModelProvenance] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_cell in enumerate(raw_cells):
        values = _exact_fields(raw_cell, _CELL_FIELDS, f"cells[{index}]")
        scenario_id, variant_id = values["scenario_id"], values["variant_id"]
        if scenario_id not in SCENARIO_IDS or variant_id not in VARIANT_IDS:
            raise ValueError("manifest contains an unknown scenario or variant")
        key = (scenario_id, variant_id)
        if key in seen:
            raise ValueError("manifest contains a duplicate matrix cell")
        seen.add(key)
        expected_key = [
            (scenario, variant)
            for scenario in SCENARIO_IDS
            for variant in VARIANT_IDS
        ][index]
        if key != expected_key:
            raise ValueError("manifest cells must use canonical scenario/variant order")
        metrics = _parse_metrics(values["controller_metrics"])
        provenance = _parse_provenance(values["model_provenance"])
        if provenance is not None:
            provenances.append(provenance)
        assumptions = _parse_assumptions(values["forecast_assumptions"])
        fault = _parse_fault(values["fault_detection"])
        cell = BenchmarkCellResult(
            mode=mode,
            scenario_id=scenario_id,
            variant_id=variant_id,
            seed=values["seed"],  # type: ignore[arg-type]
            scenario_configuration_sha256=values[
                "scenario_configuration_sha256"
            ],  # type: ignore[arg-type]
            controller_metrics=metrics,
            model_provenance=provenance,
            forecast_assumptions=assumptions,
            fault_detection=fault,
        )
        _validate_run_files(root, scenario_id, variant_id, values, metrics, provenance)
        saved_provenance = _read_json(
            root / _artifact_path(scenario_id, variant_id, "provenance")
        )
        saved_metadata = cast(dict[str, object], saved_provenance)["scenario_metadata"]
        if cast(dict[str, object], saved_metadata).get("mode") != mode:
            raise ValueError("saved run mode does not match benchmark manifest")
        cells.append(cell)
    required = {
        (scenario, variant) for scenario in SCENARIO_IDS for variant in VARIANT_IDS
    }
    if seen != required:
        raise ValueError("manifest matrix is incomplete")
    _validate_embedded_evidence(root, evidence, tuple(provenances))
    return BenchmarkMatrix(mode=mode, cells=tuple(cells))


__all__ = ["load_benchmark_matrix", "save_benchmark_execution"]
