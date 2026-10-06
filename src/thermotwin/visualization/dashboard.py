"""Artifact-only Streamlit dashboard for saved ThermoTwin experiments."""

from __future__ import annotations

import math
import os

import streamlit as st
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

from thermotwin.visualization.contracts import (
    ControllerTrace,
    DashboardData,
    FaultSummary,
    FaultTrace,
)
from thermotwin.visualization.data import load_dashboard_data
from thermotwin.visualization.plots import (
    controller_comparison_figure,
    coolant_flow_figure,
    fault_diagnostics_figure,
    forecast_comparison_figure,
    pack_heatmap_figure,
    temperature_timeseries_figure,
)

_CONTROLLERS = {
    "Reactive threshold": "threshold",
    "Reactive PID": "pid",
    "Predictive (data-only)": "predictive_data_only",
    "Predictive (physics-informed)": "predictive_physics_informed",
}
_MODELS = {
    "Data-only": "data_only",
    "Physics-informed": "physics_informed",
}
_METRIC_LABELS = (
    ("peak_temperature_k", "Peak temperature", "K"),
    ("mean_temperature_k", "Mean temperature", "K"),
    ("maximum_temperature_spread_k", "Temperature spread", "K"),
    ("pump_energy_j", "Cooling energy", "J"),
    ("thermal_violation_count", "Thermal violations", ""),
    ("time_above_thermal_limit_s", "Time above limit", "s"),
)


def _format_metric(value: float | int, unit: str) -> str:
    if unit:
        return f"{value:.2f} {unit}"
    return f"{value:d}" if isinstance(value, int) else f"{value:.2f}"


def _fault_value(value: float | int | None, unit: str) -> str:
    if value is None:
        return "Unavailable"
    return _format_metric(value, unit)


def _temperature_history_figure(
    trace: ControllerTrace, *, title: str
) -> Figure:
    return temperature_timeseries_figure(
        trace, thermal_limit_k=trace.thermal_limit_k, title=title
    )


def _fault_indicators(
    trace: FaultTrace, summary: FaultSummary
) -> tuple[float | None, float]:
    """Return score/hotspot from the target cell's event or saved trace."""
    cell_index = summary.fault_cell_index
    cell_events = [event for event in summary.events if event.cell_index == cell_index]
    if cell_events:
        event = max(cell_events, key=lambda item: item.detection_time_s)
        return event.anomaly_score, event.predicted_hotspot_temperature_k

    score = next(
        (
            float(value)
            for value in reversed(trace.scores[:, cell_index])
            if math.isfinite(float(value))
        ),
        None,
    )
    hotspot = float(trace.predicted_temperature_k[:, cell_index].max())
    return score, hotspot


def render_dashboard(data: DashboardData) -> None:
    """Render saved predictive and fault artifacts without running a simulator."""
    st.title("ThermoTwin Battery Thermal Digital Twin")
    st.caption(
        "A physics-informed twin forecasts cell temperatures to support proactive "
        "cooling and reports fault-induced thermal anomaly signals. These saved "
        "synthetic experiment artifacts are not evidence of production safety "
        "performance."
    )

    selected_controller_label = st.selectbox(
        "Controller", tuple(_CONTROLLERS), key="controller_selector"
    )
    selected_controller = _CONTROLLERS[selected_controller_label]
    selected_model_label = st.selectbox(
        "Thermal twin model", tuple(_MODELS), key="model_selector"
    )
    selected_model = _MODELS[selected_model_label]
    horizon = st.selectbox(
        "Forecast horizon (seconds)", (30, 60, 120), key="horizon_selector"
    )

    trace = data.controller_traces[selected_controller]
    summary = data.fault_summary
    metrics = st.columns(6)
    for column, (key, label, unit) in zip(metrics, _METRIC_LABELS, strict=True):
        value = trace.metrics[key]
        column.metric(label, _format_metric(value, unit))

    snapshot = next(
        item
        for item in data.model_forecasts[selected_model]
        if item.horizon_s == float(horizon)
    )
    st.subheader("Measured and predicted pack temperatures")
    measured_column, predicted_column = st.columns(2)
    measured_column.plotly_chart(
        pack_heatmap_figure(
            trace.cell_temperatures_k[-1], title="Final measured pack temperatures"
        ),
        width="stretch",
        key="measured_heatmap",
    )
    predicted_column.plotly_chart(
        pack_heatmap_figure(
            snapshot.predicted_cell_temperatures_k,
            title=f"Predicted pack temperatures at {horizon} s",
        ),
        width="stretch",
        key="predicted_heatmap",
    )

    st.subheader("Selected controller history")
    st.plotly_chart(
        _temperature_history_figure(
            trace,
            title=f"Measured temperatures - {selected_controller_label}",
        ),
        width="stretch",
        key="temperature_history",
    )
    st.plotly_chart(
        coolant_flow_figure(
            trace, title=f"Saved coolant flow - {selected_controller_label}"
        ),
        width="stretch",
        key="coolant_flow_history",
    )

    st.subheader("Saved experiment comparisons")
    st.plotly_chart(
        controller_comparison_figure(data.controller_traces),
        width="stretch",
        key="controller_comparison",
    )
    st.plotly_chart(
        forecast_comparison_figure(data.model_forecasts),
        width="stretch",
        key="model_forecast_comparison",
    )

    fault_trace = data.fault_trace
    cell_index = summary.fault_cell_index
    anomaly_score, predicted_hotspot = _fault_indicators(fault_trace, summary)
    st.subheader("Fault-induced thermal anomaly")
    fault_columns = st.columns(3)
    fault_columns[0].markdown(
        "**Suspected / configured cell**\n\n"
        f"C{cell_index + 1} ("
        f"{'detected' if summary.detected_fault_cell else 'configured'})"
    )
    fault_columns[1].markdown(
        "**Anomaly score**\n\n" + _fault_value(anomaly_score, "")
    )
    fault_columns[2].markdown(
        "**Predicted hotspot**\n\n" + _format_metric(predicted_hotspot, "K")
    )
    fault_columns[0].markdown(
        "**Detection time**\n\n" + _fault_value(summary.detection_time_s, "s")
    )
    fault_columns[1].markdown(
        "**Measured threshold crossing**\n\n"
        + _fault_value(summary.threshold_crossing_time_s, "s")
    )
    fault_columns[2].markdown(
        "**Measured lead time**\n\n" + _fault_value(summary.lead_time_s, "s")
    )
    st.plotly_chart(
        fault_diagnostics_figure(fault_trace, summary),
        width="stretch",
        key="fault_diagnostics",
    )


def main() -> None:
    """Load user-selected artifact roots and render the dashboard when valid."""
    st.set_page_config(page_title="ThermoTwin", layout="wide")
    with st.sidebar:
        st.header("Experiment artifacts")
        predictive_directory = st.text_input(
            "Predictive results directory",
            value=os.environ.get("THERMOTWIN_PREDICTIVE_RESULTS", ""),
            key="predictive_directory",
        )
        fault_directory = st.text_input(
            "Fault results directory",
            value=os.environ.get("THERMOTWIN_FAULT_RESULTS", ""),
            key="fault_directory",
        )
    if not predictive_directory.strip() or not fault_directory.strip():
        st.title("ThermoTwin Battery Thermal Digital Twin")
        st.caption(
            "Review saved temperature forecasts, cooling comparisons, and "
            "fault-induced thermal anomaly diagnostics."
        )
        st.info(
            "Enter both artifact directories in the sidebar. Point them to saved "
            "predictive comparison and fault detection results."
        )
        return
    try:
        data = load_dashboard_data(predictive_directory, fault_directory)
    except (OSError, UnicodeError, ValueError) as exc:
        st.title("ThermoTwin Battery Thermal Digital Twin")
        st.error(f"Could not load dashboard artifacts: {exc}")
        return
    render_dashboard(data)


if __name__ == "__main__":
    main()
