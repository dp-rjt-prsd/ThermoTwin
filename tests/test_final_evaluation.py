from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

import thermotwin.experiments.final_evaluation as final_evaluation_module
from test_benchmark_serialization import _save_bundle
from thermotwin.experiments.final_evaluation import (
    aggregate_final_evaluation,
    load_final_evaluation,
    save_final_evaluation,
)

_MODELS = ("data_only", "physics_informed")
_DISTRIBUTIONS = ("in_distribution", "out_of_distribution")
_HORIZONS = (30.0, 60.0, 120.0)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _write_nonfinite_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=True, indent=2, sort_keys=True) + "\n")


def _regression(mae: float) -> dict[str, Any]:
    return {
        "mae_kelvin": mae,
        "rmse_kelvin": mae + 0.1,
        "per_cell_mae_kelvin": [mae] * 12,
        "per_cell_rmse_kelvin": [mae + 0.1] * 12,
        "maximum_absolute_error_kelvin": mae + 0.2,
        "worst_cell_index": 0,
    }


def _model_metrics(mae: float) -> dict[str, Any]:
    return {
        "regression": _regression(mae),
        "mean_absolute_physics_residual_w": 0.5,
        "root_mean_square_physics_residual_w": 0.7,
        "maximum_absolute_physics_residual_w": 1.0,
    }


def _make_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    benchmark = _save_bundle(root, "benchmark")
    comparison = root / "comparison-input"
    comparison.mkdir()
    evidence = benchmark / "comparison"
    for filename in (
        "data_only.pt",
        "physics_informed.pt",
        "model_config.json",
        "normalization.json",
        "provenance.json",
    ):
        shutil.copyfile(evidence / filename, comparison / filename)
    for _label, filename in (
        ("in_distribution", "id_comparison_metrics.json"),
        ("out_of_distribution", "ood_comparison_metrics.json"),
    ):
        (comparison / filename).write_text(
            json.dumps(
                {
                    model: _model_metrics(0.5 if model == "data_only" else 0.6)
                    for model in _MODELS
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    forecast_directories: list[Path] = []
    comparison_provenance = json.loads(
        (comparison / "provenance.json").read_text(encoding="utf-8")
    )
    for distribution in _DISTRIBUTIONS:
        forecast = root / f"forecast-{distribution}"
        forecast.mkdir()
        (forecast / "checkpoints").mkdir()
        copied_sources = {
            "data_only_checkpoint": ("data_only.pt", "checkpoints/data_only.pt"),
            "physics_informed_checkpoint": (
                "physics_informed.pt",
                "checkpoints/physics_informed.pt",
            ),
            "model_config": ("model_config.json", "model_config.json"),
            "normalization": ("normalization.json", "normalization.json"),
        }
        copied_artifacts: dict[str, dict[str, str]] = {}
        for key, (source_name, relative) in copied_sources.items():
            shutil.copyfile(comparison / source_name, forecast / relative)
            copied_artifacts[key] = {
                "source": f"source/{source_name}",
                "copy": relative,
            }

        source_manifest = benchmark / "dataset" / distribution / "manifest.json"
        dataset_manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        test_split = dataset_manifest["splits"]["test"]
        csv_header = dataset_manifest["columns"]
        test_csv = ",".join(csv_header) + "\n"
        test_csv += "".join(
            ",".join("0" for _ in csv_header) + "\n"
            for _ in range(test_split["row_count"])
        )
        dataset_files = {
            "evaluation_test_csv": (
                test_csv.encode(),
                "evaluated_dataset/test.csv",
            ),
            "evaluation_manifest": (
                source_manifest.read_bytes(),
                "evaluated_dataset/manifest.json",
            ),
            "evaluation_generation_config": (
                json.dumps(
                    {
                        "seed": dataset_manifest["seed"],
                        "source_config_files": [
                            "configs/dataset_ood.yaml"
                            if distribution == "out_of_distribution"
                            else "configs/dataset.yaml"
                        ],
                    },
                    sort_keys=True,
                ).encode(),
                "evaluated_dataset/generation_config.json",
            ),
        }
        for key, (payload, relative) in dataset_files.items():
            path = forecast / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            copied_artifacts[key] = {"source": f"source/{relative}", "copy": relative}

        counts = {str(horizon): 4 for horizon in _HORIZONS}
        forecast_provenance = {
            "artifact_schema_version": "1.0",
            "seeds": {
                "training": comparison_provenance["seed"],
                "evaluation_dataset": dataset_manifest["seed"],
            },
            "package_versions": {"torch": "test"},
            "source_paths": {"comparison_directory": "source/comparison"},
            "copied_artifacts": copied_artifacts,
            "horizons_s": list(_HORIZONS),
            "time_step_s": 1.0,
            "forecast_counts": counts,
            "normalization_source": "source/comparison/normalization.json",
            "distribution_label": distribution,
            "split_label": "test",
            "smoke_test": True,
        }
        _write_json(forecast / "provenance.json", forecast_provenance)
        for model in _MODELS:
            horizons = {
                str(horizon): {
                    "horizon_s": horizon,
                    "step_count": int(horizon),
                    "forecast_count": counts[str(horizon)],
                    "metrics": _regression(0.4 if model == "data_only" else 0.45),
                }
                for horizon in _HORIZONS
            }
            _write_json(
                forecast / f"{model}_forecast_metrics.json", {"horizons": horizons}
            )
        forecast_directories.append(forecast)

    fault = root / "fault-detection"
    fault.mkdir()
    lead_time = {
        "detection_time_s": 20.0,
        "threshold_crossing_time_s": 24.0,
        "lead_time_s": 4.0,
    }
    _write_json(
        fault / "fault_summary.json",
        {
            "thermal_limit_k": 313.15,
            "threshold_crossing_time_s": 24.0,
            "lead_time": lead_time,
            "detected_fault_cell": True,
            "peak_temperature_k": 314.0,
            "temperature_spread_k": 3.0,
            "pump_energy_j": 100.0,
            "thermal_violation_count": 1,
            "anomaly_event_count": 1,
        },
    )
    _write_json(
        fault / "anomaly_events.json",
        [
            {
                "cell_index": 5,
                "anomaly_score": 4.0,
                "detection_time_s": 20.0,
                "predicted_hotspot_temperature_k": 312.0,
                "severity_k": 2.0,
            }
        ],
    )
    _write_json(
        fault / "provenance.json",
        {
            "schema_version": "1.0",
            "scenario": {"name": "simulator-owned-resistance-fault"},
            "environment_config": {},
            "controller_class_name": "ThresholdController",
            "seed": 42,
            "result_metadata": {},
            "record_count": 30,
            "workflow": (
                "simulator-owned resistance fault and learned residual detection"
            ),
            "model_variant": "physics_informed",
            "inference_inputs": {"controller": ["BatteryState.cell_temperatures_k"]},
            "hidden_fault_truth": {
                "cell_index": 5,
                "start_time_s": 12.0,
                "resistance_multiplier": 3.0,
            },
            "claim_scope": (
                "fault-induced thermal anomaly detection; no thermal-runaway model"
            ),
        },
    )
    return (
        benchmark,
        comparison,
        forecast_directories[0],
        forecast_directories[1],
        fault,
    )


def test_aggregation_reports_saved_values_and_round_trips_atomically(
    tmp_path: Path,
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    evaluation = aggregate_final_evaluation(
        benchmark, comparison, id_forecast, ood_forecast, fault
    )
    assert evaluation.evidence_status == "workflow_only"
    assert len(evaluation.model_metrics) == 4
    assert len(evaluation.forecast_metrics) == 12
    assert len(evaluation.controller_metrics) == 24
    assert evaluation.fault_detection.suspected_cell_index == 5
    assert evaluation.fault_detection.lead_time_s == 4.0
    assert evaluation.comparisons
    assert any("workflow-only" in item for item in evaluation.limitations)
    assert any("cryptographically bound" in item for item in evaluation.limitations)
    destination = save_final_evaluation(evaluation, tmp_path / "final")
    assert load_final_evaluation(destination) == evaluation
    with pytest.raises(FileExistsError):
        save_final_evaluation(evaluation, destination)
    saved_path = destination / "final_evaluation.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    saved["extra"] = "tampered"
    _write_json(saved_path, saved)
    with pytest.raises(ValueError, match="fields mismatch"):
        load_final_evaluation(destination)


def test_missing_forecast_horizon_fails_closed(tmp_path: Path) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    metrics_path = id_forecast / "data_only_forecast_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    del metrics["horizons"]["60.0"]
    _write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="horizons"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


def test_forecast_distribution_mismatch_fails_closed(tmp_path: Path) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    provenance_path = id_forecast / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["distribution_label"] = "out_of_distribution"
    _write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="distribution"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


def test_comparison_mode_mismatch_fails_closed(tmp_path: Path) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    provenance_path = comparison / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["smoke_test"] = False
    _write_json(provenance_path, provenance)
    with pytest.raises(ValueError, match="provenance|mode"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


@pytest.mark.parametrize(
    ("artifact_family", "metric_name", "nonfinite"),
    (
        ("comparison", "id_comparison_metrics.json", float("nan")),
        ("comparison", "ood_comparison_metrics.json", float("inf")),
        ("forecast", "data_only_forecast_metrics.json", float("nan")),
        ("forecast", "physics_informed_forecast_metrics.json", float("inf")),
    ),
)
def test_nonfinite_saved_metrics_fail_closed(
    tmp_path: Path,
    artifact_family: str,
    metric_name: str,
    nonfinite: float,
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    directory = comparison if artifact_family == "comparison" else id_forecast
    document = json.loads((directory / metric_name).read_text(encoding="utf-8"))
    if artifact_family == "comparison":
        document["data_only"]["regression"]["mae_kelvin"] = nonfinite
    else:
        document["horizons"]["30.0"]["metrics"]["mae_kelvin"] = nonfinite
    _write_nonfinite_json(directory / metric_name, document)
    with pytest.raises(ValueError, match="non-finite|finite"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


@pytest.mark.parametrize(
    "mismatch", ("comparison_seed", "forecast_training", "forecast_dataset")
)
def test_seed_mismatches_fail_closed(tmp_path: Path, mismatch: str) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    if mismatch == "comparison_seed":
        path = comparison / "provenance.json"
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance["seed"] += 1
        _write_json(path, provenance)
    else:
        path = id_forecast / "provenance.json"
        provenance = json.loads(path.read_text(encoding="utf-8"))
        key = "training" if mismatch == "forecast_training" else "evaluation_dataset"
        provenance["seeds"][key] += 1
        _write_json(path, provenance)
    with pytest.raises(ValueError, match="seed|provenance"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


@pytest.mark.parametrize(
    ("location", "artifact_name"),
    (
        ("comparison", "model_config.json"),
        ("comparison", "normalization.json"),
        ("forecast", "model_config.json"),
        ("forecast", "normalization.json"),
    ),
)
def test_changed_config_or_normalization_bytes_fail_closed(
    tmp_path: Path, location: str, artifact_name: str
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    target_directory = comparison if location == "comparison" else id_forecast
    artifact_path = target_directory / artifact_name
    if location == "forecast":
        artifact_path = id_forecast / (
            "model_config.json"
            if artifact_name == "model_config.json"
            else "normalization.json"
        )
    document = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact_name == "model_config.json":
        document["hidden_widths"] = [9]
    else:
        document["feature_mean"][0] = 0.25
    _write_json(artifact_path, document)
    with pytest.raises(ValueError, match="match|differs|digest|artifact"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_final_evaluation_directory_inventory_is_exact(
    tmp_path: Path, mutation: str
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    evaluation = aggregate_final_evaluation(
        benchmark, comparison, id_forecast, ood_forecast, fault
    )
    destination = save_final_evaluation(evaluation, tmp_path / "final")
    artifact = destination / "final_evaluation.json"
    if mutation == "missing":
        artifact.unlink()
    else:
        (destination / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_final_evaluation(destination)


def test_save_failure_cleans_only_its_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    evaluation = aggregate_final_evaluation(
        benchmark, comparison, id_forecast, ood_forecast, fault
    )
    unrelated = tmp_path / ".final.partial-unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("preserve", encoding="utf-8")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(final_evaluation_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected persistence failure"):
        save_final_evaluation(evaluation, tmp_path / "final")

    assert not (tmp_path / "final").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "preserve"
    assert list(tmp_path.glob(".final.partial-*")) == [unrelated]


@pytest.mark.parametrize("fault_mismatch", ("lead_time", "detected_flag", "event_cell"))
def test_fault_detection_inconsistency_fails_closed(
    tmp_path: Path, fault_mismatch: str
) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    summary_path = fault / "fault_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if fault_mismatch == "lead_time":
        summary["lead_time"]["lead_time_s"] = 5.0
    elif fault_mismatch == "detected_flag":
        summary["detected_fault_cell"] = False
    else:
        events = json.loads((fault / "anomaly_events.json").read_text(encoding="utf-8"))
        events[0]["cell_index"] = 4
        _write_json(fault / "anomaly_events.json", events)
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="fault|target-cell|lead time"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


def test_benchmark_checkpoint_must_match_comparison_artifact(tmp_path: Path) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    (comparison / "data_only.pt").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="checkpoint|artifact|does not match"):
        aggregate_final_evaluation(
            benchmark, comparison, id_forecast, ood_forecast, fault
        )


def test_unavailable_fault_events_remain_unavailable(tmp_path: Path) -> None:
    benchmark, comparison, id_forecast, ood_forecast, fault = _make_inputs(tmp_path)
    summary_path = fault / "fault_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "threshold_crossing_time_s": None,
            "lead_time": {
                "detection_time_s": None,
                "threshold_crossing_time_s": None,
                "lead_time_s": None,
            },
            "detected_fault_cell": False,
            "anomaly_event_count": 0,
        }
    )
    _write_json(summary_path, summary)
    _write_json(fault / "anomaly_events.json", [])
    evaluation = aggregate_final_evaluation(
        benchmark, comparison, id_forecast, ood_forecast, fault
    )
    assert evaluation.fault_detection.suspected_cell_index is None
    assert evaluation.fault_detection.threshold_crossing_time_s is None
    assert evaluation.fault_detection.lead_time_s is None
