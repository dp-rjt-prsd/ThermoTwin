from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from thermotwin.visualization.data import load_dashboard_data

METRICS = {
    "peak_temperature_k": 301.0,
    "mean_temperature_k": 300.5,
    "maximum_temperature_spread_k": 0.0,
    "pump_energy_j": 2.0,
    "thermal_violation_count": 1,
    "time_above_thermal_limit_s": 1.0,
}
RUNS = {
    "threshold": "reactive_threshold",
    "pid": "reactive_pid",
    "predictive_data_only": "data_only",
    "predictive_physics_informed": "physics_informed",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scenario() -> dict[str, object]:
    return {
        "name": "dashboard-test",
        "duration_s": 2.0,
        "time_step_s": 1.0,
        "thermal_limit_k": 300.5,
        "pump_power_coefficient_w_s3_kg3": 10.0,
    }


def _controller_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for time_s, offset in ((1.0, 0.0), (2.0, 1.0)):
        row: dict[str, object] = {
            "time_s": time_s,
            "current_a": 10.0,
            "soc_fraction": 0.5,
            "ambient_temperature_k": 300.0,
            "coolant_inlet_temperature_k": 290.0,
            "coolant_mass_flow_kg_s": 0.1,
            "pump_power_w": 1.0,
        }
        row.update({f"cell_temperature_{i}_k": 300.0 + offset for i in range(12)})
        row.update({f"cell_resistance_{i}_ohm": 0.01 for i in range(12)})
        rows.append(row)
    return rows


def _fault_artifacts(root: Path) -> None:
    fault = root / "fault-detection"
    run_rows: list[dict[str, object]] = []
    standard_rows: list[dict[str, object]] = []
    for time_s, hottest in ((1.0, 300.0), (2.0, 321.0), (3.0, 322.0)):
        row: dict[str, object] = {
            "time_s": time_s,
            "current_a": 10.0,
            "soc_fraction": 0.5,
            "ambient_temperature_k": 300.0,
            "coolant_inlet_temperature_k": 290.0,
            "coolant_mass_flow_kg_s": 0.1,
            "pump_power_w": 1.0,
            "new_events_json": "[]",
        }
        row.update(
            {
                f"measured_temperature_{i}_k": hottest if i == 4 else 300.0
                for i in range(12)
            }
        )
        row.update({f"predicted_temperature_{i}_k": 300.0 for i in range(12)})
        row.update(
            {f"residual_{i}_k": hottest - 300.0 if i == 4 else 0.0 for i in range(12)}
        )
        row.update({f"score_{i}": "" if time_s == 1.0 else 0.5 for i in range(12)})
        run_rows.append(row)
        standard_row: dict[str, object] = {
            "time_s": time_s,
            "current_a": 10.0,
            "soc_fraction": 0.5,
            "ambient_temperature_k": 300.0,
            "coolant_inlet_temperature_k": 290.0,
            "coolant_mass_flow_kg_s": 0.1,
            "pump_power_w": 1.0,
        }
        standard_row.update(
            {f"cell_temperature_{i}_k": hottest if i == 4 else 300.0 for i in range(12)}
        )
        standard_row.update({f"cell_resistance_{i}_ohm": 0.01 for i in range(12)})
        standard_rows.append(standard_row)
    _write_csv(fault / "anomaly_trace.csv", run_rows)
    _write_csv(fault / "timeseries.csv", standard_rows)
    _write_json(fault / "anomaly_events.json", [])
    provenance = {
        "schema_version": "1.0",
        "scenario": {
            **_scenario(),
            "duration_s": 3.0,
            "thermal_limit_k": 320.0,
        },
        "environment_config": {"pack": {"rows": 3, "columns": 4}},
        "scenario_metadata": {"profile": "test"},
        "controller_class_name": "test-controller",
        "seed": 7,
        "result_metadata": {},
        "record_count": 3,
        "workflow": "fault test",
        "model_variant": "physics_informed",
        "inference_inputs": {
            "controller": ["observed temperatures"],
            "forecaster": ["current state and action"],
            "detector": ["measured and predicted states"],
        },
        "hidden_fault_truth": {
            "cell_index": 4,
            "start_time_s": 1.5,
            "resistance_multiplier": 2.0,
        },
        "claim_scope": "test",
    }
    _write_json(fault / "provenance.json", provenance)
    temperature_values: list[float] = []
    for row_index in range(3):
        for cell in range(12):
            value = standard_rows[row_index][f"cell_temperature_{cell}_k"]
            if not isinstance(value, (int, float)):
                raise AssertionError("fixture cell temperature must be numeric")
            temperature_values.append(float(value))
    temperature_scale = max(temperature_values)
    mean_temperature = temperature_scale * (
        math.fsum(value / temperature_scale for value in temperature_values)
        / len(temperature_values)
    )
    _write_json(
        fault / "fault_summary.json",
        {
            "thermal_limit_k": 320.0,
            "threshold_crossing_time_s": 2.0,
            "lead_time": {
                "detection_time_s": None,
                "threshold_crossing_time_s": 2.0,
                "lead_time_s": None,
            },
            "detected_fault_cell": False,
            "peak_temperature_k": 322.0,
            "temperature_spread_k": 22.0,
            "pump_energy_j": 3.0,
            "thermal_violation_count": 2,
            "anomaly_event_count": 0,
        },
    )
    _write_json(
        fault / "metrics.json",
        {
            "peak_temperature_k": 322.0,
            "mean_temperature_k": mean_temperature,
            "maximum_temperature_spread_k": 22.0,
            "pump_energy_j": 3.0,
            "thermal_violation_count": 2,
            "time_above_thermal_limit_s": 2.0,
        },
    )


def make_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    predictive = tmp_path / "predictive"
    runs: dict[str, object] = {}
    provenance = {
        "schema_version": "1.0",
        "scenario": _scenario(),
        "environment_config": {
            "pack": {"rows": 3, "columns": 4},
            "cooling": {
                "minimum_coolant_mass_flow_kg_s": 0.0,
                "maximum_coolant_mass_flow_kg_s": 0.1,
            },
        },
        "scenario_metadata": {"profile": "test"},
        "controller_class_name": "test-controller",
        "seed": 7,
        "result_metadata": {},
        "record_count": 2,
    }
    for run_id, label in RUNS.items():
        run_dir = predictive / run_id
        _write_csv(run_dir / "timeseries.csv", _controller_rows())
        _write_json(run_dir / "metrics.json", METRICS)
        _write_json(run_dir / "provenance.json", provenance)
        runs[run_id] = {
            "variant": label,
            "metrics": METRICS,
            "artifact_path": run_id,
        }
    _write_json(
        predictive / "forecast_snapshots.json",
        {
            "schema_version": 2,
            "pack": {"rows": 3, "columns": 4},
            "horizons_s": [30, 60, 120],
            "source_run_id": "threshold",
            "source": {
                "measured_time_s": 2.0,
                "cell_temperatures_k": [301.0] * 12,
            },
            "action_coolant_mass_flow_kg_s": 0.1,
            "variants": {
                variant: {
                    "predictions": [
                        {
                            "horizon_s": horizon,
                            "cell_temperatures_k": [
                                301.0 + horizon / 100 + offset
                            ]
                            * 12,
                        }
                        for horizon in (30, 60, 120)
                    ],
                }
                for variant, offset in (
                    ("data_only", 0.0),
                    ("physics_informed", 0.5),
                )
            },
        },
    )
    _write_json(
        predictive / "comparison_summary.json",
        {
            "seed": 7,
            "scenario": {
                "name": "dashboard-test",
                "duration_s": 2.0,
                "time_step_s": 1.0,
                "step_count": 2,
                "metadata": {"profile": "test"},
            },
            "runs": runs,
            "forecast_snapshots_path": "forecast_snapshots.json",
        },
    )
    fault_root = tmp_path / "fault-parent"
    _fault_artifacts(fault_root)
    return predictive, fault_root


def test_loads_valid_artifacts_and_returns_immutable_data(tmp_path: Path) -> None:
    predictive, fault = make_artifacts(tmp_path)

    data = load_dashboard_data(predictive, fault)

    assert set(data.controller_traces) == set(RUNS)
    trace = data.controller_traces["threshold"]
    assert trace.cell_temperatures_k.shape == (2, 12)
    assert trace.cell_temperatures_k.dtype == np.float64
    assert data.controller_traces["predictive_data_only"].thermal_limit_k == 300.5
    with pytest.raises(ValueError):
        trace.cell_temperatures_k[0, 0] = 0.0
    with pytest.raises(TypeError):
        cast(dict[str, object], data.controller_traces)["other"] = trace
    assert tuple(item.horizon_s for item in data.model_forecasts["data_only"]) == (
        30,
        60,
        120,
    )
    data_only = data.model_forecasts["data_only"]
    physics_informed = data.model_forecasts["physics_informed"]
    assert data_only[0].source_time_s == physics_informed[0].source_time_s == 2.0
    assert data_only[0].action_coolant_mass_flow_kg_s == 0.1
    assert physics_informed[0].action_coolant_mass_flow_kg_s == 0.1
    np.testing.assert_array_equal(
        data_only[0].source_cell_temperatures_k,
        physics_informed[0].source_cell_temperatures_k,
    )
    for snapshots in (data_only, physics_informed):
        for snapshot in snapshots:
            np.testing.assert_array_equal(
                snapshot.source_cell_temperatures_k,
                trace.cell_temperatures_k[-1],
            )
            assert not snapshot.source_cell_temperatures_k.flags.writeable
            with pytest.raises(ValueError):
                snapshot.source_cell_temperatures_k[0] = 0.0
    assert not np.array_equal(
        data_only[0].predicted_cell_temperatures_k,
        physics_informed[0].predicted_cell_temperatures_k,
    )
    assert data.fault_trace.scores.shape == (3, 12)
    assert np.isnan(data.fault_trace.scores[0]).all()


def test_csv_column_order_is_not_semantic(tmp_path: Path) -> None:
    predictive, fault = make_artifacts(tmp_path)
    trace_path = predictive / "threshold" / "timeseries.csv"
    rows = _controller_rows()
    reordered = [dict(reversed(tuple(row.items()))) for row in rows]
    _write_csv(trace_path, reordered)

    data = load_dashboard_data(predictive, fault)

    assert data.controller_traces["threshold"].cell_temperatures_k.shape == (2, 12)


def test_accepts_fault_run_directory_directly(tmp_path: Path) -> None:
    predictive, fault_parent = make_artifacts(tmp_path)

    data = load_dashboard_data(predictive, fault_parent / "fault-detection")

    assert data.fault_summary.fault_cell_index == 4
    assert data.fault_summary.threshold_crossing_time_s == 2.0


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "malformed",
        "extra_column",
        "duplicate",
        "nonfinite",
        "traversal",
        "metric",
        "trace",
        "snapshot",
        "wrong_source_run",
        "source_mismatch",
        "action_mismatch",
        "variant_source_forbidden",
        "variant_action_forbidden",
        "fault_lead",
        "fault_provenance",
        "scenario_provenance",
        "fault_trace_conflict",
        "fault_record_count",
        "trace_metrics",
        "scenario_thermal_limit_string",
        "scenario_thermal_limit_zero",
        "extra_fixed_field",
        "extra_fault_provenance_field",
        "fault_schema_version",
        "json_numeric_string",
        "json_seed_string",
    ],
)
def test_rejects_invalid_artifacts(tmp_path: Path, mutation: str) -> None:
    predictive, fault = make_artifacts(tmp_path)
    summary_path = predictive / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        (predictive / "pid" / "metrics.json").unlink()
    elif mutation == "malformed":
        rows = _controller_rows()
        for row in rows:
            del row["cell_temperature_11_k"]
        _write_csv(predictive / "threshold" / "timeseries.csv", rows)
    elif mutation == "extra_column":
        rows = _controller_rows()
        for row in rows:
            row["unexpected_column"] = 1.0
        _write_csv(predictive / "threshold" / "timeseries.csv", rows)
    elif mutation == "duplicate":
        summary_path.write_text('{"runs":{},"runs":{}}', encoding="utf-8")
    elif mutation == "nonfinite":
        (predictive / "threshold" / "metrics.json").write_text(
            '{"peak_temperature_k":NaN}', encoding="utf-8"
        )
    elif mutation == "traversal":
        summary["runs"]["threshold"]["artifact_path"] = "../threshold"
        _write_json(summary_path, summary)
    elif mutation == "metric":
        summary["runs"]["threshold"]["metrics"]["peak_temperature_k"] = 999.0
        _write_json(summary_path, summary)
    elif mutation == "trace":
        rows = _controller_rows()
        rows[1]["time_s"] = 1.0
        _write_csv(predictive / "threshold" / "timeseries.csv", rows)
    elif mutation == "snapshot":
        snapshots_path = predictive / "forecast_snapshots.json"
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        snapshots["source"]["measured_time_s"] = 1.0
        _write_json(snapshots_path, snapshots)
    elif mutation == "wrong_source_run":
        snapshots_path = predictive / "forecast_snapshots.json"
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        snapshots["source_run_id"] = "predictive_data_only"
        _write_json(snapshots_path, snapshots)
    elif mutation == "source_mismatch":
        snapshots_path = predictive / "forecast_snapshots.json"
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        snapshots["source"]["cell_temperatures_k"][0] += 0.1
        _write_json(snapshots_path, snapshots)
    elif mutation == "action_mismatch":
        snapshots_path = predictive / "forecast_snapshots.json"
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        snapshots["action_coolant_mass_flow_kg_s"] = 0.05
        _write_json(snapshots_path, snapshots)
    elif mutation in ("variant_source_forbidden", "variant_action_forbidden"):
        snapshots_path = predictive / "forecast_snapshots.json"
        snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
        key = "source" if mutation == "variant_source_forbidden" else (
            "action_coolant_mass_flow_kg_s"
        )
        snapshots["variants"]["data_only"][key] = None
        _write_json(snapshots_path, snapshots)
    elif mutation == "fault_lead":
        summary_path = fault / "fault-detection" / "fault_summary.json"
        fault_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fault_summary["lead_time"]["lead_time_s"] = 100.0
        _write_json(summary_path, fault_summary)
    elif mutation == "fault_provenance":
        provenance_path = fault / "fault-detection" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["inference_inputs"]["hidden_fault_truth"] = True
        _write_json(provenance_path, provenance)
    elif mutation == "scenario_provenance":
        provenance_path = predictive / "pid" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["seed"] = 8
        _write_json(provenance_path, provenance)
    elif mutation in ("scenario_thermal_limit_string", "scenario_thermal_limit_zero"):
        thermal_limit: object = "300.5" if mutation.endswith("string") else 0.0
        for run_id in RUNS:
            provenance_path = predictive / run_id / "provenance.json"
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance["scenario"]["thermal_limit_k"] = thermal_limit
            _write_json(provenance_path, provenance)
    elif mutation == "fault_trace_conflict":
        with (fault / "fault-detection" / "timeseries.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = cast(
                list[dict[str, object]], list(csv.DictReader(stream))
            )
        rows[1]["current_a"] = "11.0"
        _write_csv(fault / "fault-detection" / "timeseries.csv", rows)
    elif mutation == "fault_record_count":
        provenance_path = fault / "fault-detection" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["record_count"] = 4
        _write_json(provenance_path, provenance)
    elif mutation == "trace_metrics":
        metrics_path = predictive / "threshold" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["peak_temperature_k"] = 999.0
        _write_json(metrics_path, metrics)
        summary["runs"]["threshold"]["metrics"]["peak_temperature_k"] = 999.0
        _write_json(summary_path, summary)
    elif mutation == "extra_fixed_field":
        summary["unexpected"] = True
        _write_json(summary_path, summary)
    elif mutation == "extra_fault_provenance_field":
        provenance_path = fault / "fault-detection" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["unexpected"] = True
        _write_json(provenance_path, provenance)
    elif mutation == "fault_schema_version":
        provenance_path = fault / "fault-detection" / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["schema_version"] = "9.9"
        _write_json(provenance_path, provenance)
    elif mutation == "json_numeric_string":
        metrics_path = predictive / "threshold" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["peak_temperature_k"] = "301.0"
        _write_json(metrics_path, metrics)
        summary["runs"]["threshold"]["metrics"]["peak_temperature_k"] = "301.0"
        _write_json(summary_path, summary)
    elif mutation == "json_seed_string":
        summary["seed"] = "7"
        _write_json(summary_path, summary)

    with pytest.raises(
        (ValueError, FileNotFoundError),
        match=(
            "(?i)(artifact|field|path|json|metrics|time|source|lead|truth|"
            "schema|trace|missing|duplicate|finite|positive)"
        ),
    ):
        load_dashboard_data(predictive, fault)
