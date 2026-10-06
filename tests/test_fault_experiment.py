from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from thermotwin.control.threshold import ThresholdController
from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.diagnostics.anomaly import ThermalAnomalyConfig
from thermotwin.diagnostics.fault_metrics import FaultLeadTimeMetric
from thermotwin.experiments.fault_detection import (
    FaultDetectionStep,
    run_fault_detection_experiment,
    save_fault_detection_result,
)
from thermotwin.experiments.types import ExperimentScenario
from thermotwin.models.forecaster import ForecastConditions, ForecastRollout


def _state(time_s: float, temperatures: list[float]) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=10.0,
        soc_fraction=0.8,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=293.15,
        coolant_mass_flow_kg_s=0.0,
        cell_temperatures_k=np.asarray(temperatures),
        cell_resistances_ohm=np.full(len(temperatures), 0.015),
    )


def _scenario() -> ExperimentScenario:
    pack = PackConfig(1, 2, 0.8, 900.0, 0.015, 50.0, 1.5, 8.0, 0.02)
    cooling = CoolingConfig(0.0, 0.1, 5.0, 500.0, 0.8)
    simulation = SimulationConfig(1.0, 0.8, 298.15, 298.15, 293.15, 10.0)
    environment = EnvironmentConfig(
        pack, cooling, simulation, ResistanceFaultConfig(1, 1.0, 3.0)
    )
    return ExperimentScenario(
        "fault-test", environment, lambda _time: 10.0, 3.0, 1.0, 299.0, 1000.0
    )


class _RecordingForecaster:
    time_step_s = 1.0
    cell_count = 2

    def __init__(self) -> None:
        self.inputs: list[tuple[BatteryState, CoolingAction]] = []

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        self.inputs.append((state, action))
        return _state(state.time_s + 1.0, [298.15, 298.15])

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: tuple[ForecastConditions, ...] | None = None,
    ) -> ForecastRollout:
        raise AssertionError("fault workflow must use one-step predictions only")


def test_fault_workflow_uses_state_action_and_records_exact_residual_sequence() -> None:
    scenario = _scenario()
    controller = ThresholdController(
        scenario.environment_config.cooling, [303, 308, 313], [0.025, 0.06]
    )
    forecaster = _RecordingForecaster()
    result = run_fault_detection_experiment(
        scenario,
        controller,
        forecaster,
        ThermalAnomalyConfig(rolling_window_steps=2, persistence_steps=2),
        seed=7,
    )

    assert len(forecaster.inputs) == scenario.step_count
    assert all(len(inputs) == 2 for inputs in forecaster.inputs)
    assert [tuple(record.residuals_k) for record in result.trace] == [
        tuple(
            record.measured_state.cell_temperatures_k - record.predicted_temperatures_k
        )
        for record in result.trace
    ]
    assert result.trace[0].scores is None
    assert result.trace[1].scores is None
    assert result.trace[2].scores is not None
    assert not result.trace[0].residuals_k.flags.writeable
    assert result.result.records[-1].state.time_s == 3.0


@pytest.mark.parametrize(
    ("predicted", "residuals", "scores"),
    [
        ([np.nan, 298.0], [0.0, 0.0], None),
        ([0.0, 298.0], [0.0, 0.0], None),
        ([298.0, 298.0], [np.inf, 0.0], None),
        ([298.0, 298.0], [0.0, 0.0], [-1.0, 0.0]),
        ([298.0, 298.0], [0.0, 0.0], [np.nan, 0.0]),
    ],
)
def test_fault_trace_step_rejects_invalid_persisted_arrays(
    predicted: list[float], residuals: list[float], scores: list[float] | None
) -> None:
    with pytest.raises(ValueError):
        FaultDetectionStep(
            _state(1.0, [298.0, 298.0]),
            np.asarray(predicted),
            np.asarray(residuals),
            None if scores is None else np.asarray(scores),
            0.0,
            (),
        )


def test_fault_result_rejects_inconsistent_lead_time_summary() -> None:
    scenario = _scenario()
    controller = ThresholdController(
        scenario.environment_config.cooling, [303, 308, 313], [0.025, 0.06]
    )
    result = run_fault_detection_experiment(
        scenario, controller, _RecordingForecaster(), ThermalAnomalyConfig(2, 3.0, 2), 7
    )
    with pytest.raises(ValueError, match="threshold crossing"):
        replace(
            result,
            threshold_crossing_time_s=1.0,
            lead_time=FaultLeadTimeMetric(None, None, None),
        )


def test_saved_fault_artifacts_separate_truth_from_inference_inputs(
    tmp_path: Path,
) -> None:
    scenario = _scenario()
    controller = ThresholdController(
        scenario.environment_config.cooling, [303, 308, 313], [0.025, 0.06]
    )
    result = run_fault_detection_experiment(
        scenario, controller, _RecordingForecaster(), ThermalAnomalyConfig(2, 3.0, 2), 7
    )
    output = save_fault_detection_result(result, tmp_path)

    assert {
        "timeseries.csv",
        "metrics.json",
        "anomaly_trace.csv",
        "anomaly_events.json",
        "fault_summary.json",
        "provenance.json",
    } <= {path.name for path in output.iterdir()}
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert set(provenance["inference_inputs"]) == {
        "controller",
        "forecaster",
        "detector",
    }
    assert all(
        "fault" not in input_name.lower()
        for inputs in provenance["inference_inputs"].values()
        for input_name in inputs
    )
    assert provenance["hidden_fault_truth"]["cell_index"] == 1
    assert "resistance_fault" not in provenance["environment_config"]
    summary = json.loads((output / "fault_summary.json").read_text(encoding="utf-8"))
    lead = summary["lead_time"]
    if lead["detection_time_s"] is None or lead["threshold_crossing_time_s"] is None:
        assert lead["lead_time_s"] is None
    else:
        assert lead["lead_time_s"] == (
            lead["threshold_crossing_time_s"] - lead["detection_time_s"]
        )
    with (output / "anomaly_trace.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(result.trace)
    with pytest.raises(FileExistsError):
        save_fault_detection_result(result, tmp_path)


def test_fault_smoke_cli_is_deterministic_and_non_overwriting(tmp_path: Path) -> None:
    outputs = [tmp_path / "first", tmp_path / "second"]
    hashes: list[dict[str, str]] = []
    for output in outputs:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_fault_detection.py",
                "--smoke-test",
                "--seed",
                "17",
                "--output-directory",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        files = output / "fault-detection"
        assert (files / "timeseries.csv").is_file()
        provenance = json.loads((files / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["seed"] == 17
        hashes.append(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files.iterdir()
            }
        )
    assert hashes[0] == hashes[1]
    summary = json.loads(
        (outputs[0] / "fault-detection" / "fault_summary.json").read_text(
            encoding="utf-8"
        )
    )
    lead = summary["lead_time"]
    if lead["detection_time_s"] is None or lead["threshold_crossing_time_s"] is None:
        assert lead["lead_time_s"] is None
    else:
        assert lead["lead_time_s"] == (
            lead["threshold_crossing_time_s"] - lead["detection_time_s"]
        )
    rerun = subprocess.run(
        [
            sys.executable,
            "scripts/run_fault_detection.py",
            "--smoke-test",
            "--output-directory",
            str(outputs[0]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode != 0
    assert "already exists" in rerun.stderr
