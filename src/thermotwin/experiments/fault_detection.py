"""Reproducible simulator-owned resistance fault and residual detection run."""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from thermotwin.control.base import Controller
from thermotwin.control.objective import pump_power_w
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.diagnostics.anomaly import (
    AnomalyEvent,
    ThermalAnomalyConfig,
    ThermalAnomalyDetector,
)
from thermotwin.diagnostics.fault_metrics import (
    FaultLeadTimeMetric,
    calculate_fault_lead_time_metric,
    first_threshold_crossing_time_s,
)
from thermotwin.diagnostics.metrics import calculate_experiment_metrics
from thermotwin.experiments.serialization import save_experiment_result
from thermotwin.experiments.types import (
    ExperimentRecord,
    ExperimentResult,
    ExperimentScenario,
)
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.physics.environment import BatteryEnvironment

if TYPE_CHECKING:
    import os

FloatArray = NDArray[np.float64]


def _immutable_array(values: NDArray[np.float64]) -> FloatArray:
    return np.frombuffer(
        np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64
    )


def _normalize_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class FaultDetectionStep:
    """Immutable measured, predicted, and detector output for one step."""

    measured_state: BatteryState
    predicted_temperatures_k: FloatArray
    residuals_k: FloatArray
    scores: FloatArray | None
    pump_power_w: float
    events: tuple[AnomalyEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.measured_state, BatteryState):
            raise TypeError("measured_state must be a BatteryState")
        predicted = _immutable_array(self.predicted_temperatures_k)
        residuals = _immutable_array(self.residuals_k)
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
            raise ValueError("predicted temperatures must be positive and finite")
        if not np.all(np.isfinite(residuals)):
            raise ValueError("residuals must be finite")
        if predicted.shape != self.measured_state.cell_temperatures_k.shape:
            raise ValueError("predicted temperatures must match measured cell count")
        if residuals.shape != predicted.shape:
            raise ValueError("residuals must match predicted cell count")
        scores = None if self.scores is None else _immutable_array(self.scores)
        if scores is not None and (
            not np.all(np.isfinite(scores)) or np.any(scores < 0.0)
        ):
            raise ValueError("scores must be finite and nonnegative")
        if scores is not None and scores.shape != predicted.shape:
            raise ValueError("scores must match predicted cell count")
        if not math.isfinite(self.pump_power_w) or self.pump_power_w < 0.0:
            raise ValueError("pump_power_w must be finite and nonnegative")
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, AnomalyEvent) for event in self.events
        ):
            raise TypeError("events must be a tuple of AnomalyEvent values")
        object.__setattr__(self, "predicted_temperatures_k", predicted)
        object.__setattr__(self, "residuals_k", residuals)
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True, slots=True)
class FaultDetectionResult:
    """Closed-loop trace and measured metrics, with truth kept in scenario."""

    result: ExperimentResult
    trace: tuple[FaultDetectionStep, ...]
    events: tuple[AnomalyEvent, ...]
    threshold_crossing_time_s: float | None
    lead_time: FaultLeadTimeMetric
    model_variant: str

    def __post_init__(self) -> None:
        if not isinstance(self.result, ExperimentResult):
            raise TypeError("result must be an ExperimentResult")
        if not isinstance(self.trace, tuple) or not self.trace:
            raise ValueError("trace must be a nonempty tuple")
        if len(self.trace) != len(self.result.records):
            raise ValueError("trace and experiment records must have equal lengths")
        if any(not isinstance(step, FaultDetectionStep) for step in self.trace):
            raise TypeError("trace must contain FaultDetectionStep values")
        trace_events = tuple(event for step in self.trace for event in step.events)
        if not isinstance(self.events, tuple) or self.events != trace_events:
            raise ValueError("events must match the events emitted in the trace")
        for step, record in zip(self.trace, self.result.records, strict=True):
            if step.measured_state is not record.state:
                raise ValueError("trace measurements must match experiment records")
            if step.pump_power_w != record.pump_power_w:
                raise ValueError("trace pump power must match experiment records")
        if not isinstance(self.lead_time, FaultLeadTimeMetric):
            raise TypeError("lead_time must be a FaultLeadTimeMetric")
        if self.threshold_crossing_time_s != self.lead_time.threshold_crossing_time_s:
            raise ValueError("threshold crossing must match the lead-time metric")
        if type(self.model_variant) is not str or self.model_variant not in {
            "data_only",
            "physics_informed",
        }:
            raise ValueError("model_variant must be data_only or physics_informed")


def run_fault_detection_experiment(
    scenario: ExperimentScenario,
    controller: Controller,
    forecaster: ThermalForecaster,
    detector_config: ThermalAnomalyConfig,
    seed: int,
    *,
    model_variant: str = "physics_informed",
) -> FaultDetectionResult:
    """Run the simulator and update detection from measured/forecast states.

    Fault truth is consulted only after all detector updates, to calculate the
    target-cell threshold and matched-event lead-time summary.
    """
    if type(scenario) is not ExperimentScenario:
        raise TypeError("scenario must be exactly an ExperimentScenario")
    if not isinstance(controller, Controller):
        raise TypeError("controller must implement the Controller protocol")
    if not isinstance(forecaster, ThermalForecaster):
        raise TypeError("forecaster must implement the ThermalForecaster protocol")
    if not isinstance(detector_config, ThermalAnomalyConfig):
        raise TypeError("detector_config must be a ThermalAnomalyConfig")
    normalized_seed = _normalize_seed(seed)
    if type(model_variant) is not str or model_variant not in {
        "data_only",
        "physics_informed",
    }:
        raise ValueError("model_variant must be data_only or physics_informed")

    environment = BatteryEnvironment(scenario.current_profile)
    controller.reset()
    detector = ThermalAnomalyDetector(detector_config)
    current_state = environment.reset(scenario.environment_config)
    if forecaster.cell_count != current_state.cell_temperatures_k.size:
        raise ValueError("forecaster cell count must match the simulator pack")
    if not math.isclose(forecaster.time_step_s, scenario.time_step_s):
        raise ValueError("forecaster time_step_s must match scenario time_step_s")

    trace: list[FaultDetectionStep] = []
    experiment_records: list[ExperimentRecord] = []
    states: list[BatteryState] = []
    pump_powers: list[float] = []
    all_events: list[AnomalyEvent] = []
    for _ in range(scenario.step_count):
        action = controller.act(current_state)
        if not isinstance(action, CoolingAction):
            raise TypeError("controller.act must return a CoolingAction")
        # Exactly one learned prediction per chosen state/action pair, before
        # the simulator step; detector receives only measured/predicted state.
        expected_state = forecaster.predict_next(current_state, action)
        if not isinstance(expected_state, BatteryState):
            raise TypeError("forecaster.predict_next must return a BatteryState")
        measured_state = environment.step(action, scenario.time_step_s)
        emitted = detector.update(measured_state, expected_state)
        residuals = detector.latest_residual_k
        scores = detector.latest_scores
        if residuals is None:
            raise RuntimeError("anomaly detector did not retain its residual")
        power = pump_power_w(
            action.coolant_mass_flow_kg_s,
            scenario.pump_power_coefficient_w_s3_kg3,
        )
        record = FaultDetectionStep(
            measured_state,
            expected_state.cell_temperatures_k,
            residuals,
            scores,
            power,
            emitted,
        )
        trace.append(record)
        experiment_records.append(ExperimentRecord(measured_state, power))
        states.append(measured_state)
        pump_powers.append(power)
        all_events.extend(emitted)
        current_state = measured_state

    metrics = calculate_experiment_metrics(
        states, pump_powers, scenario.time_step_s, scenario.thermal_limit_k
    )
    experiment_result = ExperimentResult(
        scenario,
        type(controller).__name__,
        normalized_seed,
        tuple(experiment_records),
        metrics,
        scenario.metadata,
    )
    # Fault labels enter evaluation only after all measured inference steps.
    fault = scenario.environment_config.resistance_fault
    if fault is None:
        raise ValueError("fault detection scenario requires a resistance fault")
    crossing = first_threshold_crossing_time_s(
        states, scenario.thermal_limit_k, fault.cell_index
    )
    target_event = next(
        (event for event in all_events if event.cell_index == fault.cell_index), None
    )
    lead_time = calculate_fault_lead_time_metric(target_event, crossing)
    return FaultDetectionResult(
        experiment_result,
        tuple(trace),
        tuple(all_events),
        crossing,
        lead_time,
        model_variant,
    )


def _json_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def _trace_csv(result: FaultDetectionResult) -> str:
    cell_count = result.result.records[0].state.cell_temperatures_k.size
    fields = [
        "time_s",
        "current_a",
        "soc_fraction",
        "ambient_temperature_k",
        "coolant_inlet_temperature_k",
        "coolant_mass_flow_kg_s",
        "pump_power_w",
        *(f"measured_temperature_{i}_k" for i in range(cell_count)),
        *(f"predicted_temperature_{i}_k" for i in range(cell_count)),
        *(f"residual_{i}_k" for i in range(cell_count)),
        *(f"score_{i}" for i in range(cell_count)),
        "new_events_json",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for step in result.trace:
        state = step.measured_state
        row: dict[str, object] = {
            "time_s": state.time_s,
            "current_a": state.current_a,
            "soc_fraction": state.soc_fraction,
            "ambient_temperature_k": state.ambient_temperature_k,
            "coolant_inlet_temperature_k": state.coolant_inlet_temperature_k,
            "coolant_mass_flow_kg_s": state.coolant_mass_flow_kg_s,
            "pump_power_w": step.pump_power_w,
            "new_events_json": json.dumps(
                [asdict(event) for event in step.events],
                allow_nan=False,
                sort_keys=True,
            ),
        }
        for index in range(cell_count):
            row[f"measured_temperature_{index}_k"] = float(
                state.cell_temperatures_k[index]
            )
            row[f"predicted_temperature_{index}_k"] = float(
                step.predicted_temperatures_k[index]
            )
            row[f"residual_{index}_k"] = float(step.residuals_k[index])
            row[f"score_{index}"] = (
                None if step.scores is None else float(step.scores[index])
            )
        writer.writerow(row)
    return stream.getvalue()


def save_fault_detection_result(
    result: FaultDetectionResult,
    output_directory: str | os.PathLike[str],
) -> Path:
    """Save standard experiment files and fault-specific evidence in a new run."""
    if type(result) is not FaultDetectionResult:
        raise TypeError("result must be exactly a FaultDetectionResult")
    fault = result.result.scenario.environment_config.resistance_fault
    if fault is None:
        raise ValueError("fault detection result must retain its fault scenario")
    destination = save_experiment_result(
        result.result, output_directory, "fault-detection"
    )

    provenance_path = destination / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    environment_config = provenance["environment_config"]
    environment_config.pop("resistance_fault", None)
    provenance.update(
        {
            "workflow": (
                "simulator-owned resistance fault and learned residual detection"
            ),
            "model_variant": result.model_variant,
            "inference_inputs": {
                "controller": ["BatteryState.cell_temperatures_k"],
                "forecaster": [
                    "BatteryState.current_a",
                    "BatteryState.soc_fraction",
                    "BatteryState.cell_temperatures_k",
                    "BatteryState.ambient_temperature_k",
                    "BatteryState.coolant_inlet_temperature_k",
                    "CoolingAction.coolant_mass_flow_kg_s",
                ],
                "detector": [
                    "measured BatteryState.time_s",
                    "measured BatteryState.cell_temperatures_k",
                    "predicted BatteryState.time_s",
                    "predicted BatteryState.cell_temperatures_k",
                ],
            },
            "hidden_fault_truth": {
                "cell_index": fault.cell_index,
                "start_time_s": fault.start_time_s,
                "resistance_multiplier": fault.resistance_multiplier,
            },
            "claim_scope": (
                "fault-induced thermal anomaly detection; no thermal-runaway model"
            ),
        }
    )
    _json_write(provenance_path, provenance)
    (destination / "anomaly_trace.csv").write_text(
        _trace_csv(result), encoding="utf-8", newline=""
    )
    _json_write(
        destination / "anomaly_events.json",
        [asdict(event) for event in result.events],
    )
    _json_write(
        destination / "fault_summary.json",
        {
            "thermal_limit_k": result.result.scenario.thermal_limit_k,
            "threshold_crossing_time_s": result.threshold_crossing_time_s,
            "lead_time": asdict(result.lead_time),
            "detected_fault_cell": result.lead_time.detection_time_s is not None,
            "peak_temperature_k": result.result.metrics.peak_temperature_k,
            "temperature_spread_k": result.result.metrics.maximum_temperature_spread_k,
            "pump_energy_j": result.result.metrics.pump_energy_j,
            "thermal_violation_count": result.result.metrics.thermal_violation_count,
            "anomaly_event_count": len(result.events),
        },
    )
    return destination


__all__ = [
    "FaultDetectionResult",
    "FaultDetectionStep",
    "run_fault_detection_experiment",
    "save_fault_detection_result",
]
