from typing import cast

import numpy as np
import pytest

from thermotwin.core.types import BatteryState
from thermotwin.diagnostics.anomaly import AnomalyEvent
from thermotwin.diagnostics.fault_metrics import (
    FaultLeadTimeMetric,
    calculate_fault_lead_time_metric,
    first_threshold_crossing_time_s,
)


def _state(time_s: float, temperatures_k: list[float]) -> BatteryState:
    count = len(temperatures_k)
    return BatteryState(
        time_s=time_s,
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=298.15,
        coolant_mass_flow_kg_s=0.1,
        cell_temperatures_k=np.asarray(temperatures_k, dtype=np.float64),
        cell_resistances_ohm=np.full(count, 0.01, dtype=np.float64),
    )


def _event(detection_time_s: float) -> AnomalyEvent:
    return AnomalyEvent(
        cell_index=0,
        anomaly_score=4.0,
        detection_time_s=detection_time_s,
        predicted_hotspot_temperature_k=310.0,
        severity_k=2.0,
    )


@pytest.mark.parametrize(
    ("detection_time", "crossing_time", "expected"),
    [(4.0, 10.0, 6.0), (10.0, 10.0, 0.0), (12.0, 10.0, -2.0)],
)
def test_metric_reports_signed_lead_time(
    detection_time: float, crossing_time: float, expected: float
) -> None:
    metric = calculate_fault_lead_time_metric(_event(detection_time), crossing_time)

    assert metric == FaultLeadTimeMetric(detection_time, crossing_time, expected)


@pytest.mark.parametrize(
    ("event", "crossing", "detection", "lead"),
    [(_event(3.0), None, 3.0, None), (None, 9.0, None, None), (None, None, None, None)],
)
def test_metric_preserves_unavailable_events(
    event: AnomalyEvent | None,
    crossing: float | None,
    detection: float | None,
    lead: float | None,
) -> None:
    metric = calculate_fault_lead_time_metric(event, crossing)

    assert metric.detection_time_s == detection
    assert metric.threshold_crossing_time_s == crossing
    assert metric.lead_time_s == lead


def test_metric_is_immutable_and_rejects_inconsistent_values() -> None:
    metric = FaultLeadTimeMetric(2.0, 5.0, 3.0)

    with pytest.raises(AttributeError):
        metric.lead_time_s = 99.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="lead_time_s"):
        FaultLeadTimeMetric(2.0, 5.0, 4.0)
    with pytest.raises(ValueError, match="lead_time_s"):
        FaultLeadTimeMetric(2.0, 5.0, None)
    with pytest.raises(ValueError, match="lead_time_s"):
        FaultLeadTimeMetric(None, 5.0, 0.0)


def test_crossing_uses_first_observed_state_strictly_over_limit() -> None:
    states = [
        _state(0.0, [300.0, 301.0]),
        _state(2.0, [302.0, 303.0]),
        _state(4.0, [304.0, 306.0]),
    ]

    assert first_threshold_crossing_time_s(states, 303.0) == 4.0


def test_exact_limit_does_not_count_as_crossing() -> None:
    states = [_state(0.0, [300.0]), _state(3.0, [303.0])]

    assert first_threshold_crossing_time_s(states, 303.0) is None


def test_crossing_can_be_selected_cell_or_pack_maximum() -> None:
    states = [_state(0.0, [300.0, 300.0]), _state(1.0, [304.0, 302.0])]

    assert first_threshold_crossing_time_s(states, 303.0, cell_index=1) is None
    assert first_threshold_crossing_time_s(states, 303.0) == 1.0
    assert first_threshold_crossing_time_s(states, 303.0, cell_index=0) == 1.0


@pytest.mark.parametrize(
    "states",
    [
        [],
        [_state(0.0, [300.0]), _state(0.0, [301.0])],
        [_state(1.0, [300.0]), _state(0.0, [301.0])],
    ],
)
def test_crossing_rejects_empty_or_nonincreasing_trajectory(
    states: list[BatteryState],
) -> None:
    with pytest.raises(ValueError):
        first_threshold_crossing_time_s(states, 303.0)


def test_crossing_rejects_cell_count_changes_and_non_states() -> None:
    with pytest.raises(ValueError, match="cell count"):
        first_threshold_crossing_time_s(
            [_state(0.0, [300.0]), _state(1.0, [301.0, 302.0])], 303.0
        )
    with pytest.raises((TypeError, ValueError), match="BatteryState"):
        first_threshold_crossing_time_s([_state(0.0, [300.0]), object()], 303.0)  # type: ignore[list-item]


@pytest.mark.parametrize("cell_index", [-1, 1.5, True, "0"])
def test_crossing_rejects_invalid_cell_index(cell_index: object) -> None:
    with pytest.raises(ValueError, match="cell_index"):
        first_threshold_crossing_time_s(
            [_state(0.0, [300.0])], 303.0, cell_index=cast(int | None, cell_index)
        )


def test_crossing_rejects_cell_index_outside_cell_count() -> None:
    with pytest.raises(ValueError, match="within"):
        first_threshold_crossing_time_s([_state(0.0, [300.0])], 303.0, cell_index=1)


@pytest.mark.parametrize(
    "thermal_limit", [0.0, -1.0, float("inf"), float("nan"), True, "303"]
)
def test_crossing_rejects_invalid_thermal_limit(thermal_limit: object) -> None:
    with pytest.raises(ValueError, match="thermal_limit_k"):
        first_threshold_crossing_time_s([_state(0.0, [300.0])], thermal_limit)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "metric",
    [
        (True, 4.0, 3.0),
        (-1.0, 4.0, 5.0),
        (float("inf"), 4.0, None),
        (2.0, float("nan"), None),
        (2.0, 4.0, float("inf")),
        (None, None, 0.0),
    ],
)
def test_metric_rejects_invalid_times_or_unavailable_lead(
    metric: tuple[float | None, float | None, float | None],
) -> None:
    with pytest.raises(ValueError):
        FaultLeadTimeMetric(*metric)


def test_metric_builder_rejects_wrong_event_type() -> None:
    with pytest.raises(TypeError, match="AnomalyEvent"):
        calculate_fault_lead_time_metric(object(), 5.0)  # type: ignore[arg-type]
