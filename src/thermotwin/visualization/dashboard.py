"""Streamlit entry point for the ThermoTwin judge-facing thermal dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

from thermotwin.experiments.final_evaluation import FinalEvaluation
from thermotwin.visualization.contracts import (
    ControllerTrace,
    DashboardData,
    FaultEvent,
    FaultSummary,
)
from thermotwin.visualization.dashboard_views import (
    render_application,
    render_source_inputs,
)
from thermotwin.visualization.data import load_dashboard_data, load_evaluation
from thermotwin.visualization.plots import (
    controller_comparison_figure,
    coolant_flow_figure,
    fault_diagnostics_figure,
    forecast_comparison_figure,
    pack_heatmap_figure,
    temperature_timeseries_figure,
)
from thermotwin.visualization.presentation import GLOBAL_CSS

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


_DEFAULT_EVALUATION_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "results" / "phase12-final-evaluation"
)


@st.cache_resource(show_spinner="Validating saved ThermoTwin evidence…")
def _load_dashboard_data_cached(
    predictive_directory: str, fault_directory: str
) -> DashboardData:
    """Cache strict immutable artifact parsing for responsive navigation."""
    return load_dashboard_data(predictive_directory, fault_directory)


@st.cache_resource(show_spinner="Loading benchmark evaluation…")
def _load_evaluation_cached(directory: str) -> FinalEvaluation:
    return load_evaluation(directory)


def _default_evaluation_directory() -> str:
    configured = os.environ.get("THERMOTWIN_EVALUATION_RESULTS")
    if configured is not None:
        return configured
    if _DEFAULT_EVALUATION_DIRECTORY.is_dir():
        return str(_DEFAULT_EVALUATION_DIRECTORY)
    return ""


def _format_metric(value: float | int, unit: str) -> str:
    if unit:
        return f"{value:.2f} {unit}"
    return f"{value:d}" if isinstance(value, int) else f"{value:.2f}"


def _fault_value(value: float | int | None, unit: str) -> str:
    if value is None:
        return "Unavailable"
    return _format_metric(value, unit)


def _temperature_history_figure(trace: ControllerTrace, *, title: str) -> Figure:
    return temperature_timeseries_figure(
        trace, thermal_limit_k=trace.thermal_limit_k, title=title
    )


def _fault_indicators(
    summary: FaultSummary,
) -> FaultEvent | None:
    """Select the strongest detector event, then earliest time and lowest cell ID.

    Ranking uses only persisted detector evidence. Injected-fault truth is reserved
    for the separately labeled evaluation panel.
    """
    if not summary.events:
        return None
    return max(
        summary.events,
        key=lambda event: (
            event.anomaly_score,
            -event.detection_time_s,
            -event.cell_index,
        ),
    )


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
    st.caption(
        "Shared measured source: threshold-run data | "
        f"source time: {snapshot.source_time_s:.2f} s | "
        f"selected model: {selected_model_label} | "
        f"forecast horizon: {horizon} s | "
        "constant saved coolant action: "
        f"{snapshot.action_coolant_mass_flow_kg_s:.4f} kg/s"
    )
    measured_column, predicted_column = st.columns(2)
    measured_column.plotly_chart(
        pack_heatmap_figure(
            snapshot.source_cell_temperatures_k,
            title=(
                "Shared threshold-run measured pack temperatures at "
                f"{snapshot.source_time_s:.2f} s"
            ),
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
    event = _fault_indicators(summary)
    suspected_cell = f"C{event.cell_index + 1}" if event is not None else "Unavailable"
    anomaly_score = event.anomaly_score if event is not None else None
    predicted_hotspot = (
        event.predicted_hotspot_temperature_k if event is not None else None
    )
    detection_time = event.detection_time_s if event is not None else None
    severity = event.severity_k if event is not None else None
    st.subheader("Fault detection (operator view)")
    fault_columns = st.columns(3)
    fault_columns[0].markdown("**Suspected cell**\n\n" + suspected_cell)
    fault_columns[1].markdown("**Anomaly score**\n\n" + _fault_value(anomaly_score, ""))
    fault_columns[1].markdown("**Severity**\n\n" + _fault_value(severity, "K"))
    fault_columns[2].markdown(
        "**Predicted hotspot**\n\n" + _fault_value(predicted_hotspot, "K")
    )
    fault_columns[0].markdown(
        "**Detection time**\n\n" + _fault_value(detection_time, "s")
    )
    st.subheader("Evaluation-only injected-fault truth/outcome")
    st.markdown(
        f"Configured injected-fault cell: C{summary.fault_cell_index + 1} | "
        f"fault start: {_fault_value(summary.fault_start_time_s, 's')} | "
        "measured threshold crossing: "
        f"{_fault_value(summary.threshold_crossing_time_s, 's')} | "
        f"measured lead time: {_fault_value(summary.lead_time_s, 's')}"
    )
    st.plotly_chart(
        fault_diagnostics_figure(fault_trace, summary),
        width="stretch",
        key="fault_diagnostics",
    )


def _initialise_sources() -> tuple[str, str, str]:
    """Seed artifact paths from the environment once, then read session state."""
    defaults = {
        "predictive_directory": os.environ.get("THERMOTWIN_PREDICTIVE_RESULTS", ""),
        "fault_directory": os.environ.get("THERMOTWIN_FAULT_RESULTS", ""),
        "evaluation_directory": _default_evaluation_directory(),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return (
        str(st.session_state["predictive_directory"]).strip(),
        str(st.session_state["fault_directory"]).strip(),
        str(st.session_state["evaluation_directory"]).strip(),
    )


def _landing(message: str, *, error: bool = False) -> None:
    st.title("ThermoTwin · Battery Thermal Digital Twin")
    st.caption(
        "A judge-focused thermal control room for saved temperature forecasts, "
        "cooling comparisons, and fault-induced anomaly evidence."
    )
    if error:
        st.error(message)
    else:
        st.info(message)
    render_source_inputs()


def main() -> None:
    """Load validated artifact roots and render the V2 judge experience."""
    st.set_page_config(
        page_title="ThermoTwin · Thermal Digital Twin",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    predictive_directory, fault_directory, evaluation_directory = _initialise_sources()
    if not predictive_directory or not fault_directory:
        _landing(
            "Enter both artifact directories below. Point them to saved predictive "
            "comparison and fault detection results."
        )
        return
    try:
        data = _load_dashboard_data_cached(predictive_directory, fault_directory)
    except (OSError, UnicodeError, ValueError) as exc:
        _landing(f"Could not load dashboard artifacts: {exc}", error=True)
        return
    evaluation: FinalEvaluation | None = None
    if evaluation_directory:
        try:
            evaluation = _load_evaluation_cached(evaluation_directory)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            st.toast(f"Benchmark evaluation not loaded: {exc}")
    render_application(
        data,
        evaluation=evaluation,
        predictive_directory=predictive_directory,
        fault_directory=fault_directory,
        evaluation_directory=evaluation_directory,
    )


if __name__ == "__main__":
    main()
