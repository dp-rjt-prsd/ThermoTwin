"""Portable, tamper-detecting persistence for complete benchmark executions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    import os as os_types

    from thermotwin.experiments.benchmark import BenchmarkExecution

_SCHEMA_VERSION = 1
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


def save_benchmark_execution(
    execution: BenchmarkExecution,
    output_directory: str | os_types.PathLike[str],
) -> Path:
    """Atomically publish a new complete benchmark bundle without overwrite."""
    from thermotwin.experiments.benchmark import BenchmarkExecution

    if type(execution) is not BenchmarkExecution:
        raise TypeError("execution must be exactly a BenchmarkExecution")
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
    inventory = {"manifest.json", "runs"}
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
        {"schema_version", "complete", "mode", "cells"},
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

    cells: list[BenchmarkCellResult] = []
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
    return BenchmarkMatrix(mode=mode, cells=tuple(cells))


__all__ = ["load_benchmark_matrix", "save_benchmark_execution"]
