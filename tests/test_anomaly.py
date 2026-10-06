from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from thermotwin.core.types import BatteryState
from thermotwin.diagnostics.anomaly import (
    AnomalyEvent,
    ThermalAnomalyConfig,
    ThermalAnomalyDetector,
    calculate_temperature_residual,
)


def _state(time_s: float, temperatures: tuple[float, ...]) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=0.1,
        cell_temperatures_k=np.asarray(temperatures),
        cell_resistances_ohm=np.full(len(temperatures), 0.01),
    )


def _pair(
    time_s: float,
    residuals: tuple[float, ...],
    predicted: tuple[float, ...] | None = None,
) -> tuple[BatteryState, BatteryState]:
    forecast = predicted or tuple(300.0 for _ in residuals)
    return _state(
        time_s, tuple(p + e for p, e in zip(forecast, residuals, strict=True))
    ), _state(time_s, forecast)


def test_residual_is_measured_minus_predicted_and_immutable() -> None:
    measured, predicted = _pair(1.0, (2.0, -3.0))

    residual = calculate_temperature_residual(measured, predicted)

    np.testing.assert_array_equal(residual, [2.0, -3.0])
    with pytest.raises(ValueError):
        residual[0] = 99.0


def test_detector_warms_up_on_prior_window_before_scoring() -> None:
    detector = ThermalAnomalyDetector(ThermalAnomalyConfig(rolling_window_steps=3))

    for time_s in range(1, 4):
        measured, predicted = _pair(float(time_s), (0.0, 0.0))
        assert detector.update(measured, predicted) == ()
    assert detector.latest_scores is None
    measured, predicted = _pair(4.0, (0.0, 0.0))
    assert detector.update(measured, predicted) == ()
    np.testing.assert_array_equal(detector.latest_scores, [0.0, 0.0])


def test_noisy_warmup_does_not_emit_before_calibration_is_full() -> None:
    detector = ThermalAnomalyDetector(
        ThermalAnomalyConfig(rolling_window_steps=3, score_threshold=3.0)
    )
    for time_s, residual in enumerate((0.0, 0.2, -0.1, 0.1), start=1):
        measured, predicted = _pair(float(time_s), (residual,))
        assert detector.update(measured, predicted) == ()
    assert detector.events == ()
    assert detector.latest_scores is not None


def test_isolated_spike_does_not_trigger_persistent_detection() -> None:
    detector = ThermalAnomalyDetector(
        ThermalAnomalyConfig(
            rolling_window_steps=2, score_threshold=5.0, persistence_steps=2
        )
    )
    for warmup_step in range(1, 3):
        measured, predicted = _pair(float(warmup_step), (0.0,))
        detector.update(measured, predicted)

    for time_s, residual in (
        (3.0, 10.0),
        (4.0, 0.0),
        (5.0, 10.0),
        (6.0, 0.0),
    ):
        measured, predicted = _pair(time_s, (residual,))
        assert detector.update(measured, predicted) == ()
    assert detector.events == ()


def test_persistent_anomaly_emits_first_event_with_exact_values_and_time() -> None:
    detector = ThermalAnomalyDetector(
        ThermalAnomalyConfig(
            rolling_window_steps=2, score_threshold=5.0, persistence_steps=2
        )
    )
    for time_s in (1.0, 2.0):
        measured, predicted = _pair(time_s, (0.0, 0.0), (301.0, 302.0))
        detector.update(measured, predicted)

    for time_s in (3.0, 4.0):
        measured, predicted = _pair(time_s, (0.0, 10.0), (301.0, 302.0))
        events = detector.update(measured, predicted)
    assert len(events) == 1
    event = events[0]
    assert event.cell_index == 1
    assert event.detection_time_s == 4.0
    assert event.predicted_hotspot_temperature_k == 302.0
    assert event.severity_k == 10.0
    assert event.anomaly_score == pytest.approx(10.0 / detector.config.epsilon_k)
    assert detector.events == (event,)
    with pytest.raises(FrozenInstanceError):
        event.cell_index = 0  # type: ignore[misc]


def test_cells_have_independent_persistence_and_only_first_detection_is_emitted() -> (
    None
):
    detector = ThermalAnomalyDetector(
        ThermalAnomalyConfig(
            rolling_window_steps=2, score_threshold=5.0, persistence_steps=2
        )
    )
    for time_s in (1.0, 2.0):
        measured, predicted = _pair(time_s, (0.0, 0.0))
        detector.update(measured, predicted)
    measured, predicted = _pair(3.0, (10.0, 0.0))
    assert detector.update(measured, predicted) == ()
    measured, predicted = _pair(4.0, (10.0, 10.0))
    assert [event.cell_index for event in detector.update(measured, predicted)] == [0]
    measured, predicted = _pair(5.0, (10.0, 10.0))
    assert [event.cell_index for event in detector.update(measured, predicted)] == [1]
    assert [event.cell_index for event in detector.events] == [0, 1]


def test_latest_diagnostics_are_defensive() -> None:
    detector = ThermalAnomalyDetector(ThermalAnomalyConfig(rolling_window_steps=2))
    for time_s in (1.0, 2.0, 3.0):
        measured, predicted = _pair(time_s, (1.0,))
        detector.update(measured, predicted)
    residual = detector.latest_residual_k
    scores = detector.latest_scores
    assert residual is not None and scores is not None
    residual[0] = 50.0
    scores[0] = 50.0
    assert detector.latest_residual_k is not None
    assert detector.latest_residual_k[0] == 1.0
    assert detector.latest_scores is not None
    assert detector.latest_scores[0] == 0.0


def test_reset_clears_detector_and_replays_deterministically() -> None:
    detector = ThermalAnomalyDetector(
        ThermalAnomalyConfig(
            rolling_window_steps=2, score_threshold=5.0, persistence_steps=2
        )
    )
    trace = ((1.0, 0.0), (2.0, 0.0), (3.0, 10.0), (4.0, 10.0))

    def run_trace() -> tuple[AnomalyEvent, ...]:
        for time_s, residual_value in trace:
            measured, predicted = _pair(time_s, (residual_value,))
            detector.update(measured, predicted)
        return detector.events

    first_events = run_trace()
    assert len(first_events) == 1
    detector.reset()
    assert detector.events == ()
    assert detector.latest_residual_k is None
    assert detector.latest_scores is None
    assert run_trace() == first_events


@pytest.mark.parametrize(
    ("rolling_window_steps", "score_threshold", "persistence_steps", "epsilon_k"),
    [(1, 3.0, 2, 1e-6), (2, 0.0, 2, 1e-6), (2, 3.0, 1, 1e-6), (2, 3.0, 2, 0.0)],
)
def test_configuration_rejects_invalid_values(
    rolling_window_steps: int,
    score_threshold: float,
    persistence_steps: int,
    epsilon_k: float,
) -> None:
    with pytest.raises(ValueError):
        ThermalAnomalyConfig(
            rolling_window_steps, score_threshold, persistence_steps, epsilon_k
        )


def test_configuration_rejects_noninteger_step_counts() -> None:
    with pytest.raises(ValueError):
        ThermalAnomalyConfig(rolling_window_steps=True)
    with pytest.raises(ValueError):
        ThermalAnomalyConfig(rolling_window_steps=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ThermalAnomalyConfig(persistence_steps="2")  # type: ignore[arg-type]


def test_event_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        AnomalyEvent(-1, 1.0, 2.0, 300.0, 1.0)
    with pytest.raises(ValueError):
        AnomalyEvent(0, float("inf"), 2.0, 300.0, 1.0)


def test_event_rejects_noninteger_cell_index() -> None:
    with pytest.raises(ValueError):
        AnomalyEvent(True, 1.0, 2.0, 300.0, 1.0)
    with pytest.raises(ValueError):
        AnomalyEvent(0.5, 1.0, 2.0, 300.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AnomalyEvent("0", 1.0, 2.0, 300.0, 1.0)  # type: ignore[arg-type]


def test_residual_and_update_validate_type_time_and_cell_shape() -> None:
    measured, predicted = _pair(1.0, (0.0, 0.0))
    with pytest.raises(TypeError):
        calculate_temperature_residual(object(), predicted)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        calculate_temperature_residual(measured, _state(2.0, (300.0, 300.0)))
    with pytest.raises(ValueError):
        calculate_temperature_residual(measured, _state(1.0, (300.0,)))

    detector = ThermalAnomalyDetector(ThermalAnomalyConfig(rolling_window_steps=2))
    with pytest.raises(ValueError):
        detector.update(measured, _state(2.0, (300.0, 300.0)))
    detector.update(measured, predicted)
    with pytest.raises(ValueError):
        detector.update(_state(1.0, (300.0, 300.0)), predicted)
    with pytest.raises(ValueError):
        detector.update(_state(2.0, (300.0,)), _state(2.0, (300.0,)))


def test_nonfinite_state_rejected_and_no_fault_label_api() -> None:
    with pytest.raises(ValueError):
        _state(1.0, (float("nan"),))
    assert not hasattr(ThermalAnomalyDetector, "fault_label")
    assert "fault_label" not in ThermalAnomalyDetector.update.__annotations__
