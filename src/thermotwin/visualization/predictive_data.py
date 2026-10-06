"""Parsers for Phase 9 comparison, trace, and forecast artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from thermotwin.visualization.artifact_io import (
    exact_fields,
    finite,
    float_matrix,
    frozen_array,
    json_mapping,
    metric_mapping,
    nonnegative_integer,
    number_column,
    read_csv,
    read_json,
    recompute_metrics,
    safe_artifact,
)
from thermotwin.visualization.contracts import ControllerTrace, ForecastSnapshot

_RUN_VARIANTS = {
    "threshold": "reactive_threshold",
    "pid": "reactive_pid",
    "predictive_data_only": "data_only",
    "predictive_physics_informed": "physics_informed",
}
_TRACE_COLUMNS = (
    "time_s",
    "current_a",
    "soc_fraction",
    "ambient_temperature_k",
    "coolant_inlet_temperature_k",
    "coolant_mass_flow_kg_s",
    "pump_power_w",
)
_HORIZONS = (30.0, 60.0, 120.0)


def _load_controller(
    root: Path,
    run_id: str,
    label: str,
    entry: Mapping[str, Any],
    expected_provenance: object | None,
) -> tuple[ControllerTrace, dict[str, Any]]:
    exact_fields(entry, ("variant", "metrics", "artifact_path"), f"runs.{run_id}")
    if entry.get("variant") != label:
        raise ValueError(
            f"comparison_summary.json runs.{run_id}.variant must be {label!r}"
        )
    run_path = safe_artifact(
        root, entry.get("artifact_path"), f"runs.{run_id}.artifact_path"
    )
    summary_metrics = metric_mapping(entry.get("metrics"), f"runs.{run_id}.metrics")
    saved_metrics = metric_mapping(
        read_json(run_path / "metrics.json"), f"{run_id}/metrics.json"
    )
    if dict(summary_metrics) != dict(saved_metrics):
        raise ValueError(f"runs.{run_id}.metrics does not exactly match metrics.json")

    provenance = json_mapping(
        read_json(run_path / "provenance.json"), f"{run_id}/provenance.json"
    )
    if provenance.get("schema_version") != "1.0":
        raise ValueError(f"{run_id}/provenance.json has unsupported schema_version")
    exact_fields(
        provenance,
        (
            "schema_version",
            "scenario",
            "environment_config",
            "scenario_metadata",
            "controller_class_name",
            "seed",
            "result_metadata",
            "record_count",
        ),
        f"{run_id}/provenance.json",
    )
    nonnegative_integer(provenance.get("seed"), f"{run_id}/provenance.seed")
    current_provenance = {
        key: provenance.get(key)
        for key in ("scenario", "environment_config", "scenario_metadata", "seed")
    }
    if expected_provenance is not None and current_provenance != expected_provenance:
        raise ValueError(
            f"{run_id}/provenance.json scenario provenance differs across runs"
        )
    scenario = json_mapping(
        current_provenance["scenario"], f"{run_id}/provenance.scenario"
    )
    environment = json_mapping(
        current_provenance["environment_config"],
        f"{run_id}/provenance.environment_config",
    )
    pack = json_mapping(
        environment.get("pack"), f"{run_id}/provenance.environment_config.pack"
    )
    if pack.get("rows") != 3 or pack.get("columns") != 4:
        raise ValueError(f"{run_id}/provenance pack must be 3x4 (12 cells)")

    columns = [*_TRACE_COLUMNS]
    columns.extend(f"cell_temperature_{index}_k" for index in range(12))
    columns.extend(f"cell_resistance_{index}_ohm" for index in range(12))
    rows = read_csv(run_path / "timeseries.csv", columns, f"{run_id}/timeseries.csv")
    record_count = provenance.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(rows)
    ):
        raise ValueError(
            f"{run_id}/provenance.record_count differs from timeseries.csv"
        )
    time_step = finite(
        scenario.get("time_step_s"), "scenario.time_step_s", minimum=0.0
    )
    thermal_limit = finite(
        scenario.get("thermal_limit_k"), "scenario.thermal_limit_k", minimum=0.0
    )
    trace = ControllerTrace(
        label=label,
        time_s=number_column(rows, "time_s", minimum=0.0),
        current_a=number_column(rows, "current_a"),
        soc_fraction=number_column(rows, "soc_fraction", minimum=0.0),
        ambient_temperature_k=number_column(rows, "ambient_temperature_k", minimum=0.0),
        coolant_inlet_temperature_k=number_column(
            rows, "coolant_inlet_temperature_k", minimum=0.0
        ),
        coolant_mass_flow_kg_s=number_column(
            rows, "coolant_mass_flow_kg_s", minimum=0.0
        ),
        pump_power_w=number_column(rows, "pump_power_w", minimum=0.0),
        cell_temperatures_k=float_matrix(
            rows,
            [f"cell_temperature_{index}_k" for index in range(12)],
            "cell_temperatures_k",
            positive=True,
        ),
        cell_resistances_ohm=float_matrix(
            rows,
            [f"cell_resistance_{index}_ohm" for index in range(12)],
            "cell_resistances_ohm",
            positive=True,
        ),
        thermal_limit_k=thermal_limit,
        metrics=summary_metrics,
    )
    recomputed_metrics = recompute_metrics(
        trace.cell_temperatures_k,
        trace.pump_power_w,
        time_step,
        trace.thermal_limit_k,
    )
    if dict(recomputed_metrics) != dict(summary_metrics):
        raise ValueError(f"{run_id} saved metrics do not match its timeseries.csv")
    if not math.isclose(trace.time_s[0], time_step, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{run_id}/timeseries.csv first time does not match scenario timestep"
        )
    return trace, current_provenance


def _load_forecasts(
    root: Path,
    relative: object,
    traces: Mapping[str, ControllerTrace],
    provenance: Mapping[str, Any],
) -> Mapping[str, tuple[ForecastSnapshot, ...]]:
    path = safe_artifact(
        root, relative, "comparison_summary.json.forecast_snapshots_path"
    )
    payload = json_mapping(read_json(path), "forecast_snapshots.json")
    exact_fields(
        payload,
        ("schema_version", "pack", "horizons_s", "variants"),
        "forecast_snapshots.json",
    )
    if payload.get("schema_version") != 1:
        raise ValueError("forecast_snapshots.json.schema_version must be 1")
    pack = json_mapping(payload.get("pack"), "forecast_snapshots.json.pack")
    exact_fields(pack, ("rows", "columns"), "forecast_snapshots.json.pack")
    if pack != {"rows": 3, "columns": 4}:
        raise ValueError("forecast_snapshots.json schema/pack must describe 3x4 cells")
    if payload.get("horizons_s") != [30, 60, 120]:
        raise ValueError("forecast_snapshots.json horizons_s must be [30, 60, 120]")
    variants = json_mapping(payload.get("variants"), "forecast_snapshots.json.variants")
    exact_fields(variants, ("data_only", "physics_informed"), "forecast variants")
    environment = json_mapping(provenance["environment_config"], "environment_config")
    cooling = json_mapping(environment.get("cooling"), "environment_config.cooling")
    minimum_flow = finite(
        cooling.get("minimum_coolant_mass_flow_kg_s"),
        "cooling.minimum_coolant_mass_flow_kg_s",
        minimum=0.0,
    )
    maximum_flow = finite(
        cooling.get("maximum_coolant_mass_flow_kg_s"),
        "cooling.maximum_coolant_mass_flow_kg_s",
        minimum=0.0,
    )
    result: dict[str, tuple[ForecastSnapshot, ...]] = {}
    for variant, run_id in (
        ("data_only", "predictive_data_only"),
        ("physics_informed", "predictive_physics_informed"),
    ):
        raw_variant = json_mapping(variants[variant], f"forecast variants.{variant}")
        exact_fields(
            raw_variant,
            ("source", "action_coolant_mass_flow_kg_s", "predictions"),
            f"forecast variants.{variant}",
        )
        source = json_mapping(raw_variant.get("source"), f"{variant}.source")
        exact_fields(
            source,
            ("measured_time_s", "cell_temperatures_k"),
            f"forecast variants.{variant}.source",
        )
        source_time = finite(
            source.get("measured_time_s"),
            f"{variant}.source.measured_time_s",
            minimum=0.0,
        )
        source_temperatures = frozen_array(
            source.get("cell_temperatures_k"), f"{variant}.source.cell_temperatures_k"
        )
        if source_temperatures.shape != (12,) or np.any(source_temperatures <= 0.0):
            raise ValueError(
                f"{variant}.source.cell_temperatures_k must contain 12 positive values"
            )
        trace = traces[run_id]
        if source_time != float(trace.time_s[-1]) or not np.array_equal(
            source_temperatures, trace.cell_temperatures_k[-1]
        ):
            raise ValueError(
                f"{variant}.source does not match {run_id} final measured row"
            )
        action_flow = finite(
            raw_variant.get("action_coolant_mass_flow_kg_s"),
            f"{variant}.action_coolant_mass_flow_kg_s",
            minimum=0.0,
        )
        if not minimum_flow <= action_flow <= maximum_flow:
            raise ValueError(
                f"{variant}.action_coolant_mass_flow_kg_s is outside legal limits"
            )
        if action_flow != float(trace.coolant_mass_flow_kg_s[-1]):
            raise ValueError(
                f"{variant} forecast action differs from final predictive trace action"
            )
        predictions = raw_variant.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(_HORIZONS):
            raise ValueError(
                f"{variant}.predictions must contain exactly three horizons"
            )
        snapshots: list[ForecastSnapshot] = []
        for raw_prediction, horizon in zip(predictions, _HORIZONS, strict=True):
            prediction = json_mapping(raw_prediction, f"{variant}.predictions")
            exact_fields(
                prediction,
                ("horizon_s", "cell_temperatures_k"),
                f"{variant}.predictions",
            )
            actual_horizon = finite(
                prediction.get("horizon_s"), f"{variant}.predictions.horizon_s"
            )
            if actual_horizon != horizon:
                raise ValueError(
                    f"{variant}.predictions horizons must be ordered 30, 60, 120"
                )
            temperatures = frozen_array(
                prediction.get("cell_temperatures_k"),
                f"{variant}.predictions.cell_temperatures_k",
            )
            if temperatures.shape != (12,) or np.any(temperatures <= 0.0):
                raise ValueError(
                    f"{variant}.predictions.cell_temperatures_k must contain 12 "
                    "positive values"
                )
            snapshots.append(
                ForecastSnapshot(
                    variant,
                    horizon,
                    source_time,
                    source_temperatures,
                    temperatures,
                    action_flow,
                )
            )
        result[variant] = tuple(snapshots)
    return MappingProxyType(result)


def load_predictive_data(
    predictive_directory: str | Path,
) -> tuple[Mapping[str, ControllerTrace], Mapping[str, tuple[ForecastSnapshot, ...]]]:
    root = Path(predictive_directory).resolve()
    summary = json_mapping(
        read_json(root / "comparison_summary.json"), "comparison_summary.json"
    )
    exact_fields(
        summary,
        ("seed", "scenario", "runs", "forecast_snapshots_path"),
        "comparison_summary.json",
    )
    nonnegative_integer(summary.get("seed"), "comparison_summary.json.seed")
    entries = json_mapping(summary.get("runs"), "comparison_summary.json.runs")
    exact_fields(entries, _RUN_VARIANTS, "comparison_summary.json.runs")
    traces: dict[str, ControllerTrace] = {}
    provenance: dict[str, Any] | None = None
    for run_id, variant_label in _RUN_VARIANTS.items():
        entry = json_mapping(entries[run_id], f"runs.{run_id}")
        trace, current_provenance = _load_controller(
            root, run_id, variant_label, entry, provenance
        )
        if provenance is None:
            provenance = current_provenance
        traces[run_id] = trace
    if provenance is None:
        raise ValueError("comparison_summary.json contains no controller runs")

    scenario = json_mapping(provenance["scenario"], "run provenance.scenario")
    summary_scenario = json_mapping(
        summary.get("scenario"), "comparison_summary.json.scenario"
    )
    exact_fields(
        summary_scenario,
        ("name", "duration_s", "time_step_s", "step_count", "metadata"),
        "comparison_summary.json.scenario",
    )
    for key in ("name", "duration_s", "time_step_s"):
        if summary_scenario.get(key) != scenario.get(key):
            raise ValueError(
                f"comparison_summary.json.scenario.{key} differs from provenance"
            )
    duration = finite(scenario.get("duration_s"), "scenario.duration_s", minimum=0.0)
    time_step = finite(scenario.get("time_step_s"), "scenario.time_step_s", minimum=0.0)
    if summary.get("seed") != provenance.get("seed"):
        raise ValueError("comparison_summary.json.seed differs from provenance")
    if summary_scenario.get("metadata") != provenance.get("scenario_metadata"):
        raise ValueError(
            "comparison_summary.json.scenario.metadata differs from provenance"
        )
    reference_times = traces["threshold"].time_s
    for run_id, trace in traces.items():
        if trace.time_s.shape != reference_times.shape or not np.array_equal(
            trace.time_s, reference_times
        ):
            raise ValueError(f"{run_id} trace time axis differs from threshold trace")
        if not np.allclose(np.diff(trace.time_s), time_step, rtol=1e-12, atol=1e-12):
            raise ValueError(f"{run_id} trace times do not match scenario timestep")
        if not math.isclose(
            float(trace.time_s[-1]), duration, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"{run_id} trace final time differs from scenario duration"
            )
    step_count = nonnegative_integer(
        summary_scenario.get("step_count"),
        "comparison_summary.json.scenario.step_count",
    )
    if step_count != len(reference_times):
        raise ValueError(
            "comparison_summary.json.scenario.step_count differs from trace length"
        )
    forecasts = _load_forecasts(
        root, summary.get("forecast_snapshots_path"), traces, provenance
    )
    return MappingProxyType(traces), forecasts


__all__ = ["load_predictive_data"]
