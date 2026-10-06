"""Parser for Phase 10 fault, anomaly, and lead-time artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from thermotwin.visualization.artifact_io import (
    csv_finite,
    exact_fields,
    finite,
    float_matrix,
    json_mapping,
    metric_mapping,
    nonnegative_integer,
    number_column,
    parse_inline_json,
    read_csv,
    read_json,
    recompute_metrics,
)
from thermotwin.visualization.contracts import FaultEvent, FaultSummary, FaultTrace

_COMMON_COLUMNS = (
    "time_s",
    "current_a",
    "soc_fraction",
    "ambient_temperature_k",
    "coolant_inlet_temperature_k",
    "coolant_mass_flow_kg_s",
    "pump_power_w",
)


def _parse_event(value: object, field: str) -> FaultEvent:
    raw = json_mapping(value, field)
    event_fields = (
        "cell_index",
        "anomaly_score",
        "detection_time_s",
        "predicted_hotspot_temperature_k",
        "severity_k",
    )
    exact_fields(raw, event_fields, field)
    cell = raw["cell_index"]
    if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < 12:
        raise ValueError(f"{field}.cell_index must be in [0, 11]")
    return FaultEvent(
        cell,
        finite(raw["anomaly_score"], f"{field}.anomaly_score", minimum=0.0),
        finite(raw["detection_time_s"], f"{field}.detection_time_s", minimum=0.0),
        finite(
            raw["predicted_hotspot_temperature_k"],
            f"{field}.predicted_hotspot_temperature_k",
            minimum=0.0,
        ),
        finite(raw["severity_k"], f"{field}.severity_k", minimum=0.0),
    )


def load_fault_data(
    fault_directory: str | Path,
) -> tuple[FaultTrace, FaultSummary]:
    directory = Path(fault_directory).resolve()
    run = (
        directory
        if (directory / "provenance.json").is_file()
        else directory / "fault-detection"
    )
    provenance = json_mapping(
        read_json(run / "provenance.json"), "fault-detection/provenance.json"
    )
    if provenance.get("schema_version") != "1.0":
        raise ValueError("fault provenance schema_version must be '1.0'")
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
            "workflow",
            "model_variant",
            "inference_inputs",
            "hidden_fault_truth",
            "claim_scope",
        ),
        "fault provenance.json",
    )
    nonnegative_integer(provenance.get("seed"), "fault provenance.seed")
    truth = json_mapping(
        provenance.get("hidden_fault_truth"), "provenance.hidden_fault_truth"
    )
    exact_fields(
        truth,
        ("cell_index", "start_time_s", "resistance_multiplier"),
        "provenance.hidden_fault_truth",
    )
    environment = json_mapping(
        provenance.get("environment_config"), "provenance.environment_config"
    )
    scenario = json_mapping(provenance.get("scenario"), "provenance.scenario")
    pack = json_mapping(environment.get("pack"), "provenance.environment_config.pack")
    if pack.get("rows") != 3 or pack.get("columns") != 4:
        raise ValueError("fault provenance pack must be 3x4 (12 cells)")
    inference = json_mapping(
        provenance.get("inference_inputs"), "provenance.inference_inputs"
    )
    if set(inference) != {"controller", "forecaster", "detector"}:
        raise ValueError(
            "provenance inference_inputs must keep hidden fault truth separate"
        )
    cell_index = truth.get("cell_index")
    if (
        isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or not 0 <= cell_index < 12
    ):
        raise ValueError("provenance.hidden_fault_truth.cell_index must be in [0, 11]")
    fault_start = finite(
        truth.get("start_time_s"),
        "provenance.hidden_fault_truth.start_time_s",
        minimum=0.0,
    )
    resistance_multiplier = finite(
        truth.get("resistance_multiplier"),
        "provenance.hidden_fault_truth.resistance_multiplier",
        minimum=1.0,
    )
    if resistance_multiplier <= 1.0:
        raise ValueError(
            "provenance.hidden_fault_truth.resistance_multiplier must exceed 1"
        )
    model_variant = provenance.get("model_variant")
    if model_variant not in ("data_only", "physics_informed"):
        raise ValueError(
            "provenance.model_variant must be data_only or physics_informed"
        )

    columns = list(_COMMON_COLUMNS)
    columns.extend(f"measured_temperature_{index}_k" for index in range(12))
    columns.extend(f"predicted_temperature_{index}_k" for index in range(12))
    columns.extend(f"residual_{index}_k" for index in range(12))
    columns.extend(f"score_{index}" for index in range(12))
    columns.append("new_events_json")
    rows = read_csv(
        run / "anomaly_trace.csv", columns, "fault-detection/anomaly_trace.csv"
    )
    record_count = provenance.get("record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(rows)
    ):
        raise ValueError("fault provenance.record_count differs from anomaly_trace.csv")
    time = number_column(rows, "time_s", minimum=0.0)
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("anomaly_trace.csv time_s must increase strictly")
    time_step = finite(
        scenario.get("time_step_s"), "provenance.scenario.time_step_s", minimum=0.0
    )
    duration = finite(
        scenario.get("duration_s"), "provenance.scenario.duration_s", minimum=0.0
    )
    if not math.isclose(float(time[0]), time_step, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("anomaly_trace.csv first time differs from scenario timestep")
    if not math.isclose(float(time[-1]), duration, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("anomaly_trace.csv final time differs from scenario duration")
    if not np.allclose(np.diff(time), time_step, rtol=1e-12, atol=1e-12):
        raise ValueError("anomaly_trace.csv times differ from scenario timestep")
    for name in ("current_a", "ambient_temperature_k", "coolant_inlet_temperature_k"):
        number_column(rows, name)
    soc = number_column(rows, "soc_fraction", minimum=0.0)
    if np.any(soc > 1.0):
        raise ValueError("anomaly_trace.csv soc_fraction must be in [0, 1]")
    measured = float_matrix(
        rows,
        [f"measured_temperature_{index}_k" for index in range(12)],
        "measured temperatures",
        positive=True,
    )
    predicted = float_matrix(
        rows,
        [f"predicted_temperature_{index}_k" for index in range(12)],
        "predicted temperatures",
        positive=True,
    )
    residuals = float_matrix(
        rows, [f"residual_{index}_k" for index in range(12)], "residuals"
    )
    if not np.allclose(measured - predicted, residuals, rtol=1e-12, atol=1e-12):
        raise ValueError(
            "anomaly_trace.csv residuals do not equal measured minus predicted "
            "temperatures"
        )

    standard_columns = list(_COMMON_COLUMNS)
    standard_columns.extend(f"cell_temperature_{index}_k" for index in range(12))
    standard_columns.extend(f"cell_resistance_{index}_ohm" for index in range(12))
    standard_rows = read_csv(
        run / "timeseries.csv", standard_columns, "fault-detection/timeseries.csv"
    )
    if len(standard_rows) != len(rows) or len(standard_rows) != record_count:
        raise ValueError("fault timeseries.csv and anomaly_trace.csv row counts differ")
    standard_columns_values = {
        "time_s": number_column(standard_rows, "time_s", minimum=0.0),
        "current_a": number_column(standard_rows, "current_a"),
        "soc_fraction": number_column(standard_rows, "soc_fraction", minimum=0.0),
        "ambient_temperature_k": number_column(
            standard_rows, "ambient_temperature_k", minimum=0.0
        ),
        "coolant_inlet_temperature_k": number_column(
            standard_rows, "coolant_inlet_temperature_k", minimum=0.0
        ),
        "coolant_mass_flow_kg_s": number_column(
            standard_rows, "coolant_mass_flow_kg_s", minimum=0.0
        ),
        "pump_power_w": number_column(standard_rows, "pump_power_w", minimum=0.0),
    }
    standard_temperatures = float_matrix(
        standard_rows,
        [f"cell_temperature_{index}_k" for index in range(12)],
        "timeseries cell temperatures",
        positive=True,
    )
    float_matrix(
        standard_rows,
        [f"cell_resistance_{index}_ohm" for index in range(12)],
        "timeseries cell resistances",
        positive=True,
    )
    anomaly_shared_columns = {
        "time_s": time,
        "current_a": number_column(rows, "current_a"),
        "soc_fraction": soc,
        "ambient_temperature_k": number_column(rows, "ambient_temperature_k"),
        "coolant_inlet_temperature_k": number_column(
            rows, "coolant_inlet_temperature_k"
        ),
        "coolant_mass_flow_kg_s": number_column(
            rows, "coolant_mass_flow_kg_s", minimum=0.0
        ),
        "pump_power_w": number_column(rows, "pump_power_w", minimum=0.0),
    }
    for name, expected_values in standard_columns_values.items():
        if not np.array_equal(expected_values, anomaly_shared_columns[name]):
            raise ValueError(
                f"fault timeseries.csv {name} differs from anomaly_trace.csv"
            )
    if not np.array_equal(standard_temperatures, measured):
        raise ValueError(
            "fault timeseries.csv cell temperatures differ from anomaly_trace.csv"
        )

    score_rows: list[list[float]] = []
    emitted_events: list[FaultEvent] = []
    for row_index, row in enumerate(rows):
        blank = [row[f"score_{index}"] == "" for index in range(12)]
        if any(blank) and not all(blank):
            raise ValueError(
                f"anomaly_trace.csv score row {row_index} must be fully null "
                "during warmup"
            )
        row_scores = [
            math.nan
            if blank[index]
            else csv_finite(
                row[f"score_{index}"],
                f"anomaly_trace.csv score_{index} row {row_index}",
                minimum=0.0,
            )
            for index in range(12)
        ]
        score_rows.append(row_scores)
        embedded = parse_inline_json(
            row["new_events_json"],
            f"anomaly_trace.csv new_events_json row {row_index}",
        )
        if not isinstance(embedded, list):
            raise ValueError("anomaly_trace.csv new_events_json must be a JSON list")
        row_events = tuple(
            _parse_event(item, "anomaly_trace.csv new event") for item in embedded
        )
        for event in row_events:
            if event.detection_time_s != float(time[row_index]):
                raise ValueError(
                    "anomaly event detection time differs from trace row time"
                )
            if (
                math.isnan(row_scores[event.cell_index])
                or event.anomaly_score != row_scores[event.cell_index]
            ):
                raise ValueError("anomaly event score differs from its trace score")
            if (
                event.predicted_hotspot_temperature_k
                != predicted[row_index, event.cell_index]
            ):
                raise ValueError(
                    "anomaly event temperature differs from predicted trace"
                )
            if event.severity_k != abs(residuals[row_index, event.cell_index]):
                raise ValueError("anomaly event severity differs from residual trace")
        emitted_events.extend(row_events)

    event_payload = read_json(run / "anomaly_events.json")
    if not isinstance(event_payload, list):
        raise ValueError("anomaly_events.json must be a JSON list")
    events = tuple(
        _parse_event(item, "anomaly_events.json event") for item in event_payload
    )
    if tuple(emitted_events) != events:
        raise ValueError(
            "anomaly_events.json differs from events embedded in anomaly trace"
        )

    scores = np.asarray(score_rows, dtype=np.float64)
    trace = FaultTrace(
        time_s=time,
        measured_temperature_k=measured,
        predicted_temperature_k=predicted,
        residual_k=residuals,
        scores=scores,
        coolant_mass_flow_kg_s=number_column(
            rows, "coolant_mass_flow_kg_s", minimum=0.0
        ),
        pump_power_w=number_column(rows, "pump_power_w", minimum=0.0),
    )

    summary = json_mapping(read_json(run / "fault_summary.json"), "fault_summary.json")
    exact_fields(
        summary,
        (
            "thermal_limit_k",
            "threshold_crossing_time_s",
            "lead_time",
            "detected_fault_cell",
            "peak_temperature_k",
            "temperature_spread_k",
            "pump_energy_j",
            "thermal_violation_count",
            "anomaly_event_count",
        ),
        "fault_summary.json",
    )
    thermal_limit = finite(
        summary.get("thermal_limit_k"), "fault_summary.thermal_limit_k", minimum=0.0
    )
    if thermal_limit != finite(
        scenario.get("thermal_limit_k"),
        "provenance.scenario.thermal_limit_k",
        minimum=0.0,
    ):
        raise ValueError("fault_summary thermal limit differs from scenario provenance")
    crossings = np.flatnonzero(measured[:, cell_index] > thermal_limit)
    crossing_time = None if len(crossings) == 0 else float(time[int(crossings[0])])
    stored_crossing = summary.get("threshold_crossing_time_s")
    if stored_crossing is not None:
        stored_crossing = finite(
            stored_crossing, "fault_summary.threshold_crossing_time_s", minimum=0.0
        )
    if stored_crossing != crossing_time:
        raise ValueError(
            "fault_summary.threshold_crossing_time_s differs from measured "
            "target-cell trace"
        )
    lead = json_mapping(summary.get("lead_time"), "fault_summary.lead_time")
    exact_fields(
        lead,
        ("detection_time_s", "threshold_crossing_time_s", "lead_time_s"),
        "fault_summary.lead_time",
    )
    stored_detection = lead.get("detection_time_s")
    if stored_detection is not None:
        stored_detection = finite(
            stored_detection, "fault_summary.lead_time.detection_time_s", minimum=0.0
        )
    stored_lead_crossing = lead.get("threshold_crossing_time_s")
    if stored_lead_crossing is not None:
        stored_lead_crossing = finite(
            stored_lead_crossing,
            "fault_summary.lead_time.threshold_crossing_time_s",
            minimum=0.0,
        )
    stored_lead = lead.get("lead_time_s")
    if stored_lead is not None:
        stored_lead = finite(stored_lead, "fault_summary.lead_time.lead_time_s")
    target_events = [event for event in events if event.cell_index == cell_index]
    detection_time = None if not target_events else target_events[0].detection_time_s
    if stored_detection != detection_time or stored_lead_crossing != crossing_time:
        raise ValueError("fault_summary lead event times disagree with trace/events")
    expected_lead = (
        None
        if detection_time is None or crossing_time is None
        else crossing_time - detection_time
    )
    if stored_lead != expected_lead:
        raise ValueError("fault_summary.lead_time_s is inconsistent with event times")
    if detection_time is not None and detection_time < fault_start:
        raise ValueError("fault detection event precedes hidden fault start time")
    detected = summary.get("detected_fault_cell")
    if type(detected) is not bool or detected != (detection_time is not None):
        raise ValueError(
            "fault_summary.detected_fault_cell disagrees with target-cell event"
        )
    event_count = summary.get("anomaly_event_count")
    if (
        isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count != len(events)
    ):
        raise ValueError(
            "fault_summary.anomaly_event_count differs from anomaly_events.json"
        )
    saved_metrics = metric_mapping(
        read_json(run / "metrics.json"), "fault-detection/metrics.json"
    )
    recomputed = recompute_metrics(
        standard_temperatures,
        standard_columns_values["pump_power_w"],
        time_step,
        thermal_limit,
    )
    if dict(recomputed) != dict(saved_metrics):
        raise ValueError("fault metrics.json does not match timeseries.csv")
    summary_metric_fields = {
        "peak_temperature_k": summary.get("peak_temperature_k"),
        "maximum_temperature_spread_k": summary.get("temperature_spread_k"),
        "pump_energy_j": summary.get("pump_energy_j"),
        "thermal_violation_count": summary.get("thermal_violation_count"),
    }
    for name, value in summary_metric_fields.items():
        if saved_metrics[name] != value or recomputed[name] != value:
            raise ValueError(
                f"fault_summary.{name} differs from metrics.json or trace metrics"
            )
    violation_count = finite(
        summary.get("thermal_violation_count"),
        "fault_summary.thermal_violation_count",
        minimum=0.0,
    )
    if violation_count != int(violation_count):
        raise ValueError("fault_summary.thermal_violation_count must be an integer")
    fault_summary = FaultSummary(
        model_variant=model_variant,
        fault_cell_index=cell_index,
        fault_start_time_s=fault_start,
        thermal_limit_k=thermal_limit,
        threshold_crossing_time_s=crossing_time,
        detection_time_s=detection_time,
        lead_time_s=expected_lead,
        detected_fault_cell=detected,
        peak_temperature_k=finite(
            summary.get("peak_temperature_k"),
            "fault_summary.peak_temperature_k",
            minimum=0.0,
        ),
        temperature_spread_k=finite(
            summary.get("temperature_spread_k"),
            "fault_summary.temperature_spread_k",
            minimum=0.0,
        ),
        pump_energy_j=finite(
            summary.get("pump_energy_j"), "fault_summary.pump_energy_j", minimum=0.0
        ),
        thermal_violation_count=int(violation_count),
        anomaly_event_count=len(events),
        events=events,
    )
    return trace, fault_summary


__all__ = ["load_fault_data"]
