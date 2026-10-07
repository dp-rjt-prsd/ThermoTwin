"""Judge-facing Streamlit views for the ThermoTwin V2 presentation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from thermotwin.control.objective import pump_power_w
from thermotwin.experiments.simulation_service import SimulationSession
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.visualization.contracts import (
    ControllerTrace,
    DashboardData,
    FaultEvent,
    ForecastSnapshot,
)
from thermotwin.visualization.live_contracts import (
    SimulationRequest,
    SimulationSnapshot,
)
from thermotwin.visualization.live_simulation import (
    LiveConfiguration,
    load_forecaster,
    load_live_configuration,
    make_controller,
    make_current_profile,
    make_environment,
)
from thermotwin.visualization.plots import (
    controller_comparison_figure,
    fault_diagnostics_figure,
    forecast_comparison_figure,
    pack_heatmap_figure,
    temperature_timeseries_figure,
)
from thermotwin.visualization.presentation import (
    FORMULAS,
    STORY_STEPS,
    FormulaSpec,
    StoryStep,
    apply_plot_theme,
)

_CONTROLLERS: Mapping[str, str] = {
    "Reactive threshold": "threshold",
    "Reactive PID": "pid",
    "Predictive · data-only twin": "predictive_data_only",
    "Predictive · physics-informed twin": "predictive_physics_informed",
}
_MODELS: Mapping[str, str] = {
    "Data-only twin": "data_only",
    "Physics-informed twin": "physics_informed",
}
_VIEWS = (
    "Guided Demo",
    "Live Thermal Lab",
    "Cooling Comparison",
    "Fault Lab",
    "Saved Evidence",
)
_ACCENTS = {
    "warning": "tt-warning",
    "danger": "tt-danger",
    "coolant": "tt-coolant",
    "forecast": "tt-forecast",
}
_DEFAULT_COMPARISON_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "results" / "phase12-final-comparison"
)


def _celsius(temperature_k: float) -> float:
    return float(temperature_k) - 273.15


def _temperature(temperature_k: float) -> str:
    return f"{_celsius(temperature_k):.2f} °C"


def _optional(value: float | None, unit: str) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.2f} {unit}".rstrip()


def _primary_fault_event(data: DashboardData) -> FaultEvent | None:
    if not data.fault_summary.events:
        return None
    return max(
        data.fault_summary.events,
        key=lambda event: (
            event.anomaly_score,
            -event.detection_time_s,
            -event.cell_index,
        ),
    )


def _snapshot(data: DashboardData, model_key: str, horizon_s: int) -> ForecastSnapshot:
    return next(
        snapshot
        for snapshot in data.model_forecasts[model_key]
        if snapshot.horizon_s == float(horizon_s)
    )


def _hero(eyebrow: str, title: str, copy: str, badges: Sequence[str]) -> None:
    badge_markup = "".join(f'<span class="tt-badge">{badge}</span>' for badge in badges)
    st.markdown(
        f"""
        <section class="tt-hero">
          <div class="tt-eyebrow">{eyebrow}</div>
          <div class="tt-title">{title}</div>
          <div class="tt-copy">{copy}</div>
          <div>{badge_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _panel(title: str, copy: str, accent: str = "coolant") -> None:
    accent_class = _ACCENTS.get(accent, "tt-coolant")
    st.markdown(
        f"""
        <div class="tt-panel {accent_class}">
          <div class="tt-panel-title">{title}</div>
          <div class="tt-panel-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _formula(spec: FormulaSpec, *, expanded: bool = False) -> None:
    with st.expander(f"∑  {spec.title}", expanded=expanded):
        st.latex(spec.latex)
        st.markdown(spec.explanation)
        st.markdown("**Symbols and units**")
        for symbol in spec.symbols:
            st.markdown(f"- {symbol}")
        if spec.approximation is not None:
            st.warning(spec.approximation)
        st.caption(f"Implementation provenance: `{spec.provenance}`")


def _formula_group(
    formula_ids: Sequence[str], *, title: str = "Physics behind this view"
) -> None:
    st.markdown(f"### {title}")
    for index, formula_id in enumerate(formula_ids):
        _formula(FORMULAS[formula_id], expanded=index == 0)


def _plot(figure: go.Figure, *, key: str) -> None:
    st.plotly_chart(apply_plot_theme(figure), width="stretch", key=key)


def _live_temperature_figure(
    snapshots: Sequence[SimulationSnapshot], thermal_limit_k: float
) -> go.Figure:
    states = [snapshot.state for snapshot in snapshots]
    temperatures = np.stack([state.cell_temperatures_k for state in states])
    times = np.array([state.time_s for state in states])
    figure = go.Figure()
    for cell_index in range(12):
        figure.add_trace(
            go.Scatter(
                x=times,
                y=temperatures[:, cell_index] - 273.15,
                mode="lines",
                name=f"C{cell_index + 1}",
                line=dict(width=1.4),
                hovertemplate=(
                    f"C{cell_index + 1}<br>Time: %{{x:.1f}} s<br>"
                    "Temperature: %{y:.2f} °C<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=np.max(temperatures, axis=1) - 273.15,
            mode="lines",
            name="Tmax",
            line=dict(color="#F06449", width=3),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=np.full(times.shape, thermal_limit_k - 273.15),
            mode="lines",
            name="Thermal limit",
            line=dict(color="#F4B942", width=2, dash="dot"),
        )
    )
    figure.update_layout(title="Live cell-temperature history")
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Temperature (°C)")
    return figure


def _live_coolant_figure(snapshots: Sequence[SimulationSnapshot]) -> go.Figure:
    states = [snapshot.state for snapshot in snapshots]
    times = [state.time_s for state in states]
    flow = [snapshot.control_action.coolant_mass_flow_kg_s for snapshot in snapshots]
    figure = go.Figure(
        go.Scatter(
            x=times,
            y=flow,
            mode="lines+markers",
            name="Coolant command",
            line=dict(color="#27C2D1", width=2.5),
            hovertemplate="Time: %{x:.1f} s<br>Flow: %{y:.4f} kg/s<extra></extra>",
        )
    )
    figure.update_layout(title="Live coolant command")
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Mass flow (kg/s)", rangemode="tozero")
    return figure


@st.cache_resource(show_spinner=False)
def _live_configuration_cached() -> LiveConfiguration:
    return load_live_configuration()


@st.cache_resource(show_spinner="Loading selected thermal twin…")
def _live_forecaster_cached(
    model_name: str, comparison_directory: str
) -> ThermalForecaster:
    configuration = _live_configuration_cached()
    return load_forecaster(
        configuration,
        model_name=model_name,
        comparison_directory=comparison_directory,
    )


def _metric_strip(trace: ControllerTrace) -> None:
    metrics = trace.metrics
    columns = st.columns(6)
    columns[0].metric(
        "Peak temperature", _temperature(float(metrics["peak_temperature_k"]))
    )
    columns[1].metric(
        "Mean temperature", _temperature(float(metrics["mean_temperature_k"]))
    )
    columns[2].metric(
        "Maximum spread",
        f"{float(metrics['maximum_temperature_spread_k']):.2f} °C",
    )
    columns[3].metric("Pump energy", f"{float(metrics['pump_energy_j']):.2f} J")
    columns[4].metric("Thermal violations", str(metrics["thermal_violation_count"]))
    columns[5].metric(
        "Time above limit", f"{float(metrics['time_above_thermal_limit_s']):.2f} s"
    )


def _sidebar_controls() -> tuple[str, str, int]:
    st.sidebar.markdown("## ThermoTwin")
    st.sidebar.caption("Physics-informed battery thermal digital twin")
    st.sidebar.markdown("### Evidence controls")
    controller_label = st.sidebar.selectbox(
        "Controller",
        tuple(_CONTROLLERS),
        index=3,
        key="v2_controller_selector",
    )
    model_label = st.sidebar.selectbox(
        "Thermal twin model",
        tuple(_MODELS),
        index=1,
        key="v2_model_selector",
    )
    horizon_s = st.sidebar.selectbox(
        "Forecast horizon (seconds)",
        (30, 60, 120),
        index=1,
        key="v2_horizon_selector",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Physics & formulas")
    formula_title = st.sidebar.selectbox(
        "Equation explorer",
        tuple(spec.title for spec in FORMULAS.values()),
        key="v2_formula_selector",
    )
    selected_formula = next(
        spec for spec in FORMULAS.values() if spec.title == formula_title
    )
    with st.sidebar.expander("Open selected equation", expanded=False):
        st.latex(selected_formula.latex)
        st.markdown(selected_formula.explanation)
        st.caption(f"Source: `{selected_formula.provenance}`")
    st.sidebar.info(
        "All values come from validated synthetic experiment artifacts. "
        "They are not evidence of production safety performance."
    )
    return (
        _CONTROLLERS[controller_label],
        _MODELS[model_label],
        int(horizon_s),
    )


def _story_rail() -> StoryStep:
    labels = {f"{step.number:02d} · {step.title}": step for step in STORY_STEPS}
    selected = st.radio(
        "Story sequence",
        tuple(labels),
        key="guided_story_step",
    )
    return labels[selected]


def _guided_status(data: DashboardData, trace: ControllerTrace) -> None:
    event = _primary_fault_event(data)
    current_temperatures = trace.cell_temperatures_k[-1]
    st.markdown("### Live telemetry")
    st.caption(f"Saved state at {trace.time_s[-1]:.1f} s")
    _panel(
        "Current telemetry",
        f"Tmax {_temperature(float(np.max(current_temperatures)))} · mean "
        f"{_temperature(float(np.mean(current_temperatures)))} · spread "
        f"{float(np.ptp(current_temperatures)):.2f} °C · coolant "
        f"{trace.coolant_mass_flow_kg_s[-1]:.4f} kg/s",
        "coolant",
    )
    _panel(
        "Safety limit",
        f"Configured threshold: {_temperature(trace.thermal_limit_k)}. "
        f"Saved violations: {trace.metrics['thermal_violation_count']}.",
        "warning",
    )
    if event is None:
        _panel(
            "Anomaly monitor",
            "No persisted detector event is available for the selected artifact.",
            "forecast",
        )
    else:
        _panel(
            "Anomaly detected",
            f"Suspected C{event.cell_index + 1} · score {event.anomaly_score:.2f} · "
            f"predicted hotspot {_temperature(event.predicted_hotspot_temperature_k)}",
            "danger",
        )


def render_guided_demo(
    data: DashboardData, controller_key: str, model_key: str, horizon_s: int
) -> None:
    trace = data.controller_traces[controller_key]
    snapshot = _snapshot(data, model_key, horizon_s)
    _hero(
        "Judge-guided thermal story",
        "See heat build, move, and get controlled before it becomes a hotspot.",
        "Follow the physical chain from electrical load to proactive cooling and "
        "fault-induced anomaly evidence. ML stays behind the scenes; heat flow "
        "and measured outcomes remain in front.",
        ("12-cell pack", f"{horizon_s} s forecast", "Artifact-backed", "SI physics"),
    )
    story_column, evidence_column, status_column = st.columns([1.05, 2.9, 1.15])
    with story_column:
        step = _story_rail()
        _panel(
            f"Step {step.number}: {step.title}",
            step.summary,
            step.accent,
        )
        st.caption(
            "Select a step to connect the thermal visual to its governing equation."
        )
    with evidence_column:
        st.markdown("### Measured vs forecast pack temperatures")
        st.caption(
            "Shared measured source: threshold-run data · "
            f"source {snapshot.source_time_s:.2f} s · horizon {horizon_s} s · "
            f"saved coolant action {snapshot.action_coolant_mass_flow_kg_s:.4f} kg/s"
        )
        measured, predicted = st.columns(2)
        with measured:
            _plot(
                pack_heatmap_figure(
                    snapshot.source_cell_temperatures_k,
                    title="Measured thermal state",
                ),
                key="v2_guided_measured_heatmap",
            )
        with predicted:
            _plot(
                pack_heatmap_figure(
                    snapshot.predicted_cell_temperatures_k,
                    title=f"Twin forecast · +{horizon_s} s",
                ),
                key="v2_guided_predicted_heatmap",
            )
        _panel(
            "Physical energy flow",
            "Electrical I²R heat enters each cell. Neighbor conduction redistributes "
            "it. Ambient exchange and the liquid cold plate remove it. The predictive "
            "controller evaluates the future balance before choosing flow.",
            "coolant",
        )
    with status_column:
        _guided_status(data, trace)
    _metric_strip(trace)
    comparison, formula_column = st.columns([2.15, 1])
    with comparison:
        _plot(
            controller_comparison_figure(data.controller_traces),
            key="v2_guided_controller_comparison",
        )
    with formula_column:
        _formula_group(step.formula_ids, title="Equation for this story step")
    summary = data.fault_summary
    st.markdown(
        f"""
        <div class="tt-truth">
          <div class="tt-panel-title">Evaluation-only injected-fault truth</div>
          <div class="tt-panel-copy">
            Configured cell C{summary.fault_cell_index + 1} · fault start
            {summary.fault_start_time_s:.2f} s · threshold crossing
            {_optional(summary.threshold_crossing_time_s, "s")} · measured lead time
            {_optional(summary.lead_time_s, "s")}. This truth is not supplied to the
            digital twin or operator detector during inference.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_lab(data: DashboardData, controller_key: str) -> None:
    configuration = _live_configuration_cached()
    _hero(
        "Interactive thermal lab",
        "Change the conditions and run the authoritative pack simulator.",
        "The simulator remains ground truth. Controllers act on the current state, "
        "and an optional saved twin forecasts future cell temperatures without "
        "receiving future simulator values.",
        ("Live simulator", "12-cell pack", "Deterministic run"),
    )
    controller_options = tuple(_CONTROLLERS)
    default_controller = controller_options[0]
    controls = st.columns(4)
    profile_name = controls[0].selectbox(
        "Current profile", configuration.profile_names, key="v2_live_profile"
    )
    controller_name = controls[1].selectbox(
        "Controller",
        controller_options,
        index=controller_options.index(default_controller),
        key="v2_live_controller",
    )
    model_name = controls[2].selectbox(
        "Thermal twin",
        ("No forecast", "Data-only twin", "Physics-informed twin"),
        index=0,
        key="v2_live_model",
    )
    duration_s = controls[3].slider(
        "Duration (s)", 10, 120, 30, 10, key="v2_live_duration"
    )
    physical = st.columns(4)
    ambient_c = physical[0].slider("Ambient (°C)", 10.0, 50.0, 25.0, 1.0)
    coolant_c = physical[1].slider("Coolant inlet (°C)", 5.0, 40.0, 20.0, 1.0)
    initial_c = physical[2].slider("Initial cell temp (°C)", 15.0, 45.0, 25.0, 1.0)
    peak_current = physical[3].slider("Peak current (A)", 20.0, 180.0, 120.0, 10.0)
    fault_enabled = st.checkbox(
        "Inject a resistance fault", key="v2_live_fault_enabled"
    )
    fault_cell = 0
    fault_multiplier = 1.0
    fault_start = 0.0
    if fault_enabled:
        fault_controls = st.columns(3)
        fault_cell = (
            fault_controls[0].selectbox(
                "Faulted cell", tuple(range(1, 13)), key="v2_live_fault_cell"
            )
            - 1
        )
        fault_multiplier = fault_controls[1].slider(
            "Resistance multiplier", 1.1, 5.0, 2.0, 0.1, key="v2_live_fault_multiplier"
        )
        fault_start = fault_controls[2].slider(
            "Fault start (s)",
            0.0,
            float(duration_s - 1),
            10.0,
            1.0,
            key="v2_live_fault_start",
        )
    action_columns = st.columns([1, 1, 4])
    run_clicked = action_columns[0].button(
        "Run simulation", type="primary", use_container_width=True, key="v2_live_run"
    )
    reset_clicked = action_columns[1].button(
        "Reset", use_container_width=True, key="v2_live_reset"
    )
    action_columns[2].caption(
        "Every run uses the physical simulator as ground truth. Predictive control "
        "loads the selected saved twin only when its artifact is available."
    )
    signature = (
        profile_name,
        controller_name,
        model_name,
        duration_s,
        ambient_c,
        coolant_c,
        initial_c,
        peak_current,
        fault_enabled,
        fault_cell,
        fault_multiplier,
        fault_start,
    )
    if reset_clicked:
        st.session_state.pop("v2_live_result", None)
        st.session_state.pop("v2_live_signature", None)
    if run_clicked or "v2_live_result" not in st.session_state:
        try:
            environment = make_environment(
                configuration,
                ambient_temperature_k=ambient_c + 273.15,
                coolant_inlet_temperature_k=coolant_c + 273.15,
                initial_soc_fraction=0.8,
                initial_cell_temperature_k=initial_c + 273.15,
                fault_cell_index=fault_cell if fault_enabled else None,
                fault_start_time_s=fault_start,
                resistance_multiplier=fault_multiplier,
            )
            profile = make_current_profile(
                profile_name, duration_s=float(duration_s), peak_current_a=peak_current
            )
            selected_forecaster = None
            if model_name != "No forecast":
                selected_forecaster = _live_forecaster_cached(
                    model_name, str(_DEFAULT_COMPARISON_DIRECTORY)
                )
            selected_controller = make_controller(
                configuration,
                controller_name,
                forecaster=selected_forecaster,
            )
            request = SimulationRequest(
                scenario_name=profile_name,
                environment_config=environment,
                current_profile=profile,
                controller_name=controller_name,
                model_name=None if model_name == "No forecast" else model_name,
                duration_s=float(duration_s),
                time_step_s=environment.simulation.default_time_step_s,
                seed=1729,
                metadata={"fault_enabled": fault_enabled},
            )
            session = SimulationSession(
                request,
                selected_controller,
                selected_forecaster,
                forecast_horizon_steps=12 if selected_forecaster else 0,
            )
            st.session_state.v2_live_result = session.run_to_completion()
            st.session_state.v2_live_signature = signature
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            st.error(f"Live simulation could not run: {exc}")
            return
    snapshots = st.session_state.get("v2_live_result")
    if not snapshots:
        st.info("Run the simulation to populate the live thermal lab.")
        return
    st.caption(f"Active run: {st.session_state.get('v2_live_signature', signature)}")
    index = st.slider(
        "Timeline",
        min_value=0,
        max_value=len(snapshots) - 1,
        value=len(snapshots) - 1,
        key="v2_live_timeline",
    )
    snapshot = snapshots[index]
    state = snapshot.state
    temperatures = state.cell_temperatures_k
    telemetry = st.columns(6)
    telemetry[0].metric("Time", f"{state.time_s:.1f} s")
    telemetry[1].metric("Current", f"{state.current_a:.1f} A")
    telemetry[2].metric("SOC", f"{state.soc_fraction * 100:.1f} %")
    telemetry[3].metric("Ambient", _temperature(state.ambient_temperature_k))
    telemetry[4].metric(
        "Coolant inlet", _temperature(state.coolant_inlet_temperature_k)
    )
    telemetry[5].metric(
        "Coolant flow", f"{snapshot.control_action.coolant_mass_flow_kg_s:.4f} kg/s"
    )
    heatmap, context = st.columns([1.8, 1])
    with heatmap:
        _plot(
            pack_heatmap_figure(
                temperatures,
                title=f"Live 12-cell thermal state · {state.time_s:.1f} s",
            ),
            key="v2_live_heatmap",
        )
        if snapshot.forecast_states:
            _plot(
                pack_heatmap_figure(
                    snapshot.forecast_states[-1].cell_temperatures_k,
                    title="Twin forecast at selected horizon",
                ),
                key="v2_live_forecast_heatmap",
            )
        else:
            st.info("This controller run has no thermal-twin forecast selected.")
    with context:
        hottest = int(np.argmax(temperatures))
        _panel(
            "Hottest cell",
            f"C{hottest + 1} is {_temperature(float(temperatures[hottest]))}; "
            f"pack spread is {float(np.ptp(temperatures)):.2f} °C.",
            "warning",
        )
        terms = snapshot.thermal_terms_w
        _panel(
            "Energy balance terms",
            f"Generation {terms['Q_gen_total_w']:.2f} W · cooling "
            f"{terms['Q_cool_total_w']:.2f} W · ambient "
            f"{terms['Q_ambient_total_w']:.2f} W · neighbor net "
            f"{terms['Q_neighbor_net_w']:.2f} W.",
            "coolant",
        )
        pump_power = pump_power_w(
            snapshot.control_action.coolant_mass_flow_kg_s,
            configuration.pump_power_coefficient_w_s3_kg3,
        )
        _panel(
            "Live equation substitution",
            f"C{hottest + 1}: ({state.current_a:.2f} A)² × "
            f"{state.cell_resistances_ohm[hottest]:.5f} Ω = "
            f"{state.current_a**2 * state.cell_resistances_ohm[hottest]:.3f} W. "
            f"Pump power is {pump_power:.3f} W.",
            "forecast",
        )
        _formula_group(("energy_balance", "joule", "cooling", "pump"))
    history, cooling = st.columns(2)
    with history:
        _plot(
            _live_temperature_figure(snapshots, configuration.thermal_limit_k),
            key="v2_live_temperature_history",
        )
    with cooling:
        _plot(_live_coolant_figure(snapshots), key="v2_live_coolant_history")


def _comparison_table(traces: Mapping[str, ControllerTrace]) -> pd.DataFrame:
    rows: list[dict[str, str | float | int]] = []
    names = {value: key for key, value in _CONTROLLERS.items()}
    for key, trace in traces.items():
        rows.append(
            {
                "Controller": names[key],
                "Peak (°C)": round(
                    _celsius(float(trace.metrics["peak_temperature_k"])), 2
                ),
                "Mean (°C)": round(
                    _celsius(float(trace.metrics["mean_temperature_k"])), 2
                ),
                "Max spread (°C)": round(
                    float(trace.metrics["maximum_temperature_spread_k"]), 2
                ),
                "Pump energy (J)": round(float(trace.metrics["pump_energy_j"]), 2),
                "Violations": int(trace.metrics["thermal_violation_count"]),
                "Time above limit (s)": round(
                    float(trace.metrics["time_above_thermal_limit_s"]), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def render_cooling_comparison(data: DashboardData, controller_key: str) -> None:
    _hero(
        "Cooling strategy evidence",
        "Compare thermal safety, uniformity, and pump effort on equal footing.",
        "The panels report saved outcomes for reactive and predictive controllers. "
        "They show tradeoffs; they do not manufacture a winner or imply statistical "
        "significance from one trajectory.",
        ("4 controllers", "6 physical metrics", "Common evidence source"),
    )
    selected_trace = data.controller_traces[controller_key]
    _metric_strip(selected_trace)
    _plot(
        controller_comparison_figure(data.controller_traces),
        key="v2_cooling_controller_matrix",
    )
    st.dataframe(
        _comparison_table(data.controller_traces),
        use_container_width=True,
        hide_index=True,
    )
    model_chart, formulas = st.columns([2, 1])
    with model_chart:
        _plot(
            forecast_comparison_figure(data.model_forecasts),
            key="v2_cooling_forecast_comparison",
        )
    with formulas:
        _formula_group(("threshold", "pid", "predictive_cost", "pump", "spread"))


def _anomaly_score_figure(data: DashboardData) -> go.Figure:
    trace = data.fault_trace
    figure = go.Figure()
    for cell_index in range(12):
        figure.add_trace(
            go.Scatter(
                x=trace.time_s,
                y=trace.scores[:, cell_index],
                mode="lines",
                name=f"C{cell_index + 1}",
                line=dict(width=1.5),
                hovertemplate=(
                    f"C{cell_index + 1}<br>Time: %{{x:.2f}} s<br>"
                    "Score: %{y:.2f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(title="Rolling per-cell anomaly scores")
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Normalized anomaly score")
    return figure


def render_fault_lab(data: DashboardData) -> None:
    summary = data.fault_summary
    event = _primary_fault_event(data)
    threshold_crossing = _optional(summary.threshold_crossing_time_s, "s")
    lead_time = _optional(summary.lead_time_s, "s")
    _hero(
        "Fault-induced thermal anomaly",
        "Watch excess resistance turn into a measurable hotspot precursor.",
        "The operator sees only observed-versus-expected thermal behavior. The "
        "configured fault cell and start time stay isolated in the evaluation panel.",
        ("Resistance fault", "Persistent detector", "No fault-label leakage"),
    )
    operator, truth = st.columns([1.6, 1])
    with operator:
        st.markdown("### Operator view")
        if event is None:
            st.info("No persisted anomaly event is available.")
        else:
            cards = st.columns(4)
            cards[0].metric("Suspected cell", f"C{event.cell_index + 1}")
            cards[1].metric("Anomaly score", f"{event.anomaly_score:.2f}")
            cards[2].metric("Detection time", f"{event.detection_time_s:.2f} s")
            cards[3].metric(
                "Predicted hotspot",
                _temperature(event.predicted_hotspot_temperature_k),
            )
        _panel(
            "Detection principle",
            "A single noisy sample does not create an event. The score must exceed "
            "its threshold persistently against the rolling residual baseline.",
            "forecast",
        )
    with truth:
        st.markdown("### Evaluation-only truth")
        st.markdown(
            f"""
            <div class="tt-truth">
              <strong>Configured cell:</strong> C{summary.fault_cell_index + 1}<br>
              <strong>Fault start:</strong> {summary.fault_start_time_s:.2f} s<br>
              <strong>Threshold crossing:</strong> {threshold_crossing}<br>
              <strong>Lead time:</strong> {lead_time}<br>
              <strong>Detected configured cell:</strong> {summary.detected_fault_cell}
            </div>
            """,
            unsafe_allow_html=True,
        )
    diagnostics, scores = st.columns(2)
    with diagnostics:
        _plot(
            fault_diagnostics_figure(data.fault_trace, summary),
            key="v2_fault_diagnostics",
        )
    with scores:
        _plot(_anomaly_score_figure(data), key="v2_fault_scores")
    _formula_group(
        ("joule", "anomaly_residual", "anomaly_score", "lead_time"),
        title="From resistance fault to early warning",
    )


def render_saved_evidence(
    data: DashboardData,
    controller_key: str,
    model_key: str,
    horizon_s: int,
    predictive_directory: str,
    fault_directory: str,
) -> None:
    trace = data.controller_traces[controller_key]
    snapshot = _snapshot(data, model_key, horizon_s)
    _hero(
        "Immutable experiment evidence",
        "Trace every visual back to validated saved artifacts.",
        "This view preserves the original dashboard's artifact-first workflow "
        "while presenting provenance and scientific limitations more clearly.",
        ("Validated loaders", "No hidden recomputation", "Synthetic evidence"),
    )
    roots = st.columns(2)
    roots[0].text_input(
        "Predictive evidence directory",
        value=str(Path(predictive_directory)),
        disabled=True,
        key="v2_predictive_root_display",
    )
    roots[1].text_input(
        "Fault evidence directory",
        value=str(Path(fault_directory)),
        disabled=True,
        key="v2_fault_root_display",
    )
    _metric_strip(trace)
    measured, predicted = st.columns(2)
    with measured:
        _plot(
            pack_heatmap_figure(
                snapshot.source_cell_temperatures_k,
                title=f"Shared measured source · {snapshot.source_time_s:.2f} s",
            ),
            key="v2_evidence_measured_heatmap",
        )
    with predicted:
        _plot(
            pack_heatmap_figure(
                snapshot.predicted_cell_temperatures_k,
                title=f"Saved {horizon_s} s forecast",
            ),
            key="v2_evidence_predicted_heatmap",
        )
    _plot(
        temperature_timeseries_figure(
            trace,
            thermal_limit_k=trace.thermal_limit_k,
            title="Selected controller temperature history",
        ),
        key="v2_evidence_temperature_history",
    )
    comparison, forecast = st.columns(2)
    with comparison:
        _plot(
            controller_comparison_figure(data.controller_traces),
            key="v2_evidence_controller_comparison",
        )
    with forecast:
        _plot(
            forecast_comparison_figure(data.model_forecasts),
            key="v2_evidence_forecast_comparison",
        )
    st.info(
        "Evidence status: these synthetic saved artifacts demonstrate the workflow. "
        "The interface does not claim production safety or guaranteed superiority."
    )
    export_frame = pd.DataFrame(
        {
            "time_s": trace.time_s,
            "current_a": trace.current_a,
            "soc_fraction": trace.soc_fraction,
            "ambient_temperature_k": trace.ambient_temperature_k,
            "coolant_inlet_temperature_k": trace.coolant_inlet_temperature_k,
            "coolant_mass_flow_kg_s": trace.coolant_mass_flow_kg_s,
            "pump_power_w": trace.pump_power_w,
            **{
                f"cell_{index + 1}_temperature_k": trace.cell_temperatures_k[:, index]
                for index in range(12)
            },
        }
    )
    export_summary = {
        "schema_version": 1,
        "controller": controller_key,
        "model": model_key,
        "forecast_horizon_s": horizon_s,
        "predictive_artifact_root": str(Path(predictive_directory)),
        "fault_artifact_root": str(Path(fault_directory)),
        "metrics": dict(trace.metrics),
        "forecast_source_time_s": snapshot.source_time_s,
        "forecast_action_coolant_mass_flow_kg_s": (
            snapshot.action_coolant_mass_flow_kg_s
        ),
        "scientific_scope": (
            "synthetic experiment evidence; not production safety validation"
        ),
    }
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download selected trace · CSV",
        export_frame.to_csv(index=False),
        file_name=f"thermotwin-{controller_key}-trace.csv",
        mime="text/csv",
        use_container_width=True,
    )
    downloads[1].download_button(
        "Download evidence summary · JSON",
        json.dumps(export_summary, indent=2, sort_keys=True),
        file_name="thermotwin-evidence-summary.json",
        mime="application/json",
        width="stretch",
    )
    _formula_group(
        ("mae", "rmse", "physics_residual"), title="How evidence is quantified"
    )


def render_application(
    data: DashboardData,
    *,
    predictive_directory: str,
    fault_directory: str,
) -> None:
    """Render the complete V2 judge-facing dashboard."""
    controller_key, model_key, horizon_s = _sidebar_controls()
    st.title("ThermoTwin · Battery Thermal Digital Twin")
    st.markdown(
        '<div class="tt-eyebrow">ThermoTwin · Battery thermal digital twin</div>',
        unsafe_allow_html=True,
    )
    active_view = st.radio(
        "Workspace",
        _VIEWS,
        horizontal=True,
        label_visibility="collapsed",
        key="v2_workspace",
    )
    if active_view == "Guided Demo":
        render_guided_demo(data, controller_key, model_key, horizon_s)
    elif active_view == "Live Thermal Lab":
        render_live_lab(data, controller_key)
    elif active_view == "Cooling Comparison":
        render_cooling_comparison(data, controller_key)
    elif active_view == "Fault Lab":
        render_fault_lab(data)
    else:
        render_saved_evidence(
            data,
            controller_key,
            model_key,
            horizon_s,
            predictive_directory,
            fault_directory,
        )


__all__ = [
    "render_application",
    "render_cooling_comparison",
    "render_fault_lab",
    "render_guided_demo",
    "render_live_lab",
    "render_saved_evidence",
]
