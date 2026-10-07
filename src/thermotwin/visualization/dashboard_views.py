"""Judge-facing Streamlit views for the ThermoTwin dashboard.

Four views, each with one job:

* Guided Demo — a seven-step physical story where every step drives its own visual.
* Live Thermal Lab — run the authoritative simulator and play the result back.
* Controllers & Twins — saved benchmark and model evidence, plus provenance.
* Fault Lab — operator-visible anomaly evidence, kept apart from injected truth.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from thermotwin.control.objective import pump_power_w
from thermotwin.experiments.final_evaluation import FinalEvaluation
from thermotwin.experiments.simulation_service import SimulationSession
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.physics.topology import pack_edges
from thermotwin.visualization.contracts import (
    ControllerTrace,
    DashboardData,
    FaultEvent,
    ForecastSnapshot,
)
from thermotwin.visualization.judge_figures import (
    CONTROLLER_METRICS,
    add_heat_flow_arrows,
    anomaly_score_figure,
    controller_scorecard,
    coolant_history,
    fault_story_figure,
    model_error_figure,
    pack_grid,
    pack_map,
    residual_figure,
    scenario_controller_figure,
    shared_range_c,
    temperature_history,
)
from thermotwin.visualization.live_contracts import (
    SimulationRequest,
    SimulationSnapshot,
)
from thermotwin.visualization.live_simulation import (
    LiveConfiguration,
    load_detector_settings,
    load_forecaster,
    load_live_configuration,
    make_controller,
    make_current_profile,
    make_environment,
)
from thermotwin.visualization.presentation import (
    FORMULAS,
    STORY_STEPS,
    FormulaSpec,
    StoryStep,
)

_CONTROLLERS: Mapping[str, str] = {
    "Reactive threshold": "threshold",
    "Reactive PID": "pid",
    "Predictive · data-only twin": "predictive_data_only",
    "Predictive · physics-informed twin": "predictive_physics_informed",
}
_PREDICTIVE_TWINS: Mapping[str, str] = {
    "Predictive · data-only twin": "Data-only twin",
    "Predictive · physics-informed twin": "Physics-informed twin",
}
_MODELS: Mapping[str, str] = {
    "Data-only twin": "data_only",
    "Physics-informed twin": "physics_informed",
}
_VIEWS = ("Guided Demo", "Live Thermal Lab", "Controllers & Twins", "Fault Lab")
_ACCENTS = {
    "warning": "tt-warning",
    "danger": "tt-danger",
    "coolant": "tt-coolant",
    "forecast": "tt-forecast",
}
_DEFAULT_COMPARISON_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "results" / "phase12-final-comparison"
)
_PLAYBACK_INTERVAL_S = 0.5
_SPEEDS = {"1×": 1, "2×": 2, "4×": 4}

SideCard = tuple[str, str, str]


# ---------------------------------------------------------------- formatting


def _celsius(temperature_k: float) -> float:
    return float(temperature_k) - 273.15


def _temperature(temperature_k: float) -> str:
    return f"{_celsius(temperature_k):.1f} °C"


def _optional(value: float | None, unit: str) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.1f} {unit}".rstrip()


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


# ---------------------------------------------------------------- components


def _hero(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <section class="tt-hero">
          <div class="tt-eyebrow">{eyebrow}</div>
          <div class="tt-title">{title}</div>
          <div class="tt-copy">{copy}</div>
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


def _chips(items: Sequence[tuple[str, str]]) -> None:
    markup = "".join(
        f'<span class="tt-chip">{label} <b>{value}</b></span>' for label, value in items
    )
    st.markdown(f"<div>{markup}</div>", unsafe_allow_html=True)


def _formula(spec: FormulaSpec, *, expanded: bool = False) -> None:
    with st.expander(f"∑  {spec.title}", expanded=expanded):
        st.latex(spec.latex)
        st.markdown(spec.explanation)
        st.markdown("**Symbols and units**")
        for symbol in spec.symbols:
            st.markdown(f"- {symbol}")
        if spec.approximation is not None:
            st.caption(f"Approximation: {spec.approximation}")
        st.caption(f"Implementation: `{spec.provenance}`")


def _formula_group(
    formula_ids: Sequence[str], *, title: str = "Physics behind this view"
) -> None:
    st.markdown(f"### {title}")
    columns = st.columns(2)
    for index, formula_id in enumerate(formula_ids):
        with columns[index % 2]:
            _formula(FORMULAS[formula_id], expanded=index < 2)


def _plot(figure: go.Figure, *, key: str) -> None:
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _time_picker(label: str, times: Sequence[float], *, key: str, default: int) -> int:
    """Pick a sample index by its time; a single-sample trace needs no control."""
    count = len(times)
    if count <= 1:
        return 0
    if key not in st.session_state:
        st.session_state[key] = min(max(default, 0), count - 1)
    return int(
        st.select_slider(
            label,
            options=list(range(count)),
            format_func=lambda index: f"{float(times[index]):.0f} s",
            key=key,
        )
    )


def _side_cards(cards: Sequence[SideCard]) -> None:
    for title, copy, accent in cards:
        _panel(title, copy, accent)


# ---------------------------------------------------------------- cached loaders


@st.cache_resource(show_spinner=False)
def _live_configuration_cached() -> LiveConfiguration:
    return load_live_configuration()


@st.cache_resource(show_spinner=False)
def _detector_settings_cached() -> dict[str, float]:
    return load_detector_settings()


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


# ---------------------------------------------------------------- guided demo


def _step_energy(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[SideCard]:
    trace = data.controller_traces["threshold"]
    times = [float(value) for value in trace.time_s]
    peak_index = int(np.argmax(np.abs(trace.current_a)))
    index = _time_picker("Time in saved run", times, key="g1_time", default=peak_index)
    current = float(trace.current_a[index])
    resistances = np.asarray(trace.cell_resistances_ohm[index], dtype=float)
    heat_w = current**2 * resistances
    all_heat = (np.asarray(trace.current_a)[:, None] ** 2) * np.asarray(
        trace.cell_resistances_ohm
    )
    top = max(float(all_heat.max()) * 1.05, 1.0)
    _plot(
        pack_grid(
            heat_w,
            title=f"Joule heat generated in each cell · t = {times[index]:.0f} s",
            value_range=(0.0, top),
            unit="W",
            colorbar_title="W",
            decimals=0,
        ),
        key="v2_guided_primary",
    )
    st.caption(
        "Source: saved reactive-threshold run. Q = I²R is evaluated per cell from the "
        "recorded pack current and cell resistance."
    )
    multiplier = _detector_settings_cached()["resistance_multiplier"]
    nominal = float(np.median(resistances))
    return [
        (
            "What to notice",
            f"At <b>{current:.0f} A</b> each healthy cell with R = "
            f"<b>{nominal * 1000:.1f} mΩ</b> generates <b>{current**2 * nominal:.0f} W</b>. "
            "Doubling current quadruples heat.",
            "warning",
        ),
        (
            "Fault what-if",
            f"The fault experiments raise one cell's resistance ×{multiplier:g}. At the "
            f"same current that cell would generate "
            f"<b>{current**2 * nominal * multiplier:.0f} W</b>, a local hotspot.",
            "danger",
        ),
    ]


def _step_heat_moves(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[SideCard]:
    trace = data.fault_trace
    configuration = _live_configuration_cached()
    times = [float(value) for value in trace.time_s]
    index = _time_picker(
        "Time in saved fault run", times, key="g2_time", default=len(times) - 1
    )
    temperatures = np.asarray(trace.measured_temperature_k[index], dtype=float)
    hottest = int(np.argmax(temperatures))
    figure = pack_map(
        temperatures,
        title=f"Measured pack temperature · t = {times[index]:.0f} s",
        range_c=shared_range_c(trace.measured_temperature_k),
        highlight=hottest,
    )
    thermal_resistance = configuration.environment.pack.neighbor_thermal_resistance_k_w
    flows = add_heat_flow_arrows(
        figure,
        temperatures,
        pack_edges(configuration.environment.pack),
        thermal_resistance,
    )
    _plot(figure, key="v2_guided_primary")
    st.caption(
        "Source: saved fault run. Arrows show the strongest conduction paths, "
        f"(T_hot − T_cold) / R_th with R_th = {thermal_resistance:g} K/W."
    )
    strongest = (
        f"Strongest path C{flows[0][0] + 1} → C{flows[0][1] + 1}: <b>{flows[0][2]:.1f} W</b>."
        if flows
        else "The pack is uniform here, so no conduction flow is visible yet."
    )
    return [
        (
            "What to notice",
            f"C{hottest + 1} is hottest at <b>{_temperature(float(temperatures[hottest]))}</b>; "
            f"spread is <b>{float(np.ptp(temperatures)):.2f} K</b>. {strongest}",
            "danger",
        ),
        (
            "Why it matters",
            "Conduction spreads a hotspot into its neighbours, so a single bad cell "
            "raises the whole pack's peak and non-uniformity.",
            "coolant",
        ),
    ]


def _flow_changes(times: Sequence[float], flows: Sequence[float]) -> list[str]:
    changes: list[str] = []
    previous = None
    for time_s, flow in zip(times, flows, strict=True):
        if previous is not None and abs(flow - previous) > 1e-9:
            changes.append(f"{time_s:.0f} s → {flow:.3f} kg/s")
        previous = flow
    return changes


def _step_cooling(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[SideCard]:
    trace = data.fault_trace
    limit = data.fault_summary.thermal_limit_k
    _plot(
        temperature_history(
            trace.time_s,
            trace.measured_temperature_k,
            thermal_limit_k=limit,
            title="Pack temperature (saved fault run)",
            height=290,
        ),
        key="v2_guided_primary",
    )
    _plot(
        coolant_history(
            trace.time_s,
            trace.coolant_mass_flow_kg_s,
            trace.pump_power_w,
            title="Reactive threshold controller: coolant command",
            height=260,
        ),
        key="v2_guided_secondary",
    )
    changes = _flow_changes(
        [float(value) for value in trace.time_s],
        [float(value) for value in trace.coolant_mass_flow_kg_s],
    )
    return [
        (
            "What to notice",
            "The reactive controller only steps up flow after Tmax crosses each band: "
            + ("; ".join(changes) if changes else "no flow change in this run")
            + ".",
            "coolant",
        ),
        (
            "Cost of cooling",
            "Pump power grows with ṁ³, so going from 0.06 to 0.1 kg/s multiplies pump "
            "power by about 4.6×.",
            "warning",
        ),
    ]


def _step_forecast(
    data: DashboardData, evaluation: FinalEvaluation | None
) -> list[SideCard]:
    controls = st.columns(2)
    model_label = controls[0].radio(
        "Thermal twin", tuple(_MODELS), index=1, horizontal=True, key="g4_model"
    )
    horizon_s = int(
        controls[1].radio(
            "Forecast horizon",
            (30, 60, 120),
            horizontal=True,
            key="g4_horizon",
            format_func=lambda value: f"+{value} s",
        )
    )
    model_key = _MODELS[model_label]
    snapshot = _snapshot(data, model_key, horizon_s)
    value_range = shared_range_c(
        snapshot.source_cell_temperatures_k, snapshot.predicted_cell_temperatures_k
    )
    measured, predicted = st.columns(2)
    with measured:
        _plot(
            pack_map(
                snapshot.source_cell_temperatures_k,
                title=f"Measured now · t = {snapshot.source_time_s:.0f} s",
                range_c=value_range,
                height=320,
            ),
            key="v2_guided_measured_heatmap",
        )
    with predicted:
        _plot(
            pack_map(
                snapshot.predicted_cell_temperatures_k,
                title=f"{model_label} forecast · +{horizon_s} s",
                range_c=value_range,
                height=320,
            ),
            key="v2_guided_predicted_heatmap",
        )
    st.caption(
        "Both maps share one colour scale. Source: threshold-run state; the forecast "
        f"holds coolant at the saved {snapshot.action_coolant_mass_flow_kg_s:.3f} kg/s."
    )
    one_step_error = float(np.nanmean(np.abs(data.fault_trace.residual_k[:10])))
    cards: list[SideCard] = [
        (
            "Short-horizon tracking",
            f"Before the fault starts, the twin's one-step error in the saved fault run "
            f"averages <b>{one_step_error:.2f} K</b>.",
            "forecast",
        )
    ]
    if evaluation is not None:
        mae = next(
            (
                item.mae_kelvin
                for item in evaluation.forecast_metrics
                if item.model_variant == model_key
                and item.distribution == "in_distribution"
                and item.horizon_s == float(horizon_s)
            ),
            None,
        )
        if mae is not None:
            cards.append(
                (
                    f"Measured error at +{horizon_s} s",
                    f"Saved evaluation MAE for this twin: <b>{mae:.2f} K</b> "
                    "(in-distribution). Treat long-horizon maps with that error in mind.",
                    "warning" if mae > 5 else "forecast",
                )
            )
    return cards


def _step_controller(
    data: DashboardData, evaluation: FinalEvaluation | None
) -> list[SideCard]:
    if evaluation is None:
        _plot(
            temperature_history(
                data.controller_traces["threshold"].time_s,
                data.controller_traces["threshold"].cell_temperatures_k,
                thermal_limit_k=data.controller_traces["threshold"].thermal_limit_k,
                title="Saved run temperature",
            ),
            key="v2_guided_primary",
        )
        return [
            (
                "Benchmark not loaded",
                "Add the final-evaluation directory in the sidebar to compare "
                "controllers across six scenarios.",
                "warning",
            )
        ]
    _plot(
        scenario_controller_figure(evaluation, "pump_energy_j", height=400),
        key="v2_guided_primary",
    )
    table = controller_scorecard(evaluation)
    st.dataframe(
        table, hide_index=True, width="stretch", column_config=_SCORECARD_FORMAT
    )
    by_name = {row["Controller"]: row for row in table.to_dict("records")}
    reactive = by_name["Reactive threshold"]
    predictive = by_name["Predictive · physics-informed twin"]
    saving = 1.0 - float(predictive["Mean pump energy (J)"]) / max(
        float(reactive["Mean pump energy (J)"]), 1e-9
    )
    peak_gap = float(predictive["Mean peak (°C)"]) - float(reactive["Mean peak (°C)"])
    return [
        (
            "What to notice",
            f"Across six scenarios the physics-informed predictive controller used "
            f"<b>{saving:.0%} less pump energy</b> than the threshold controller, with mean "
            f"peak temperature <b>{peak_gap:+.2f} K</b> different.",
            "forecast",
        ),
        (
            "Evidence scope",
            f"Benchmark mode <b>{evaluation.benchmark_mode}</b>, status "
            f"<b>{evaluation.evidence_status}</b>, seed {evaluation.benchmark_seed}. "
            "Descriptive, not statistically significant.",
            "warning",
        ),
    ]


def _step_fault(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[SideCard]:
    _plot(
        fault_story_figure(data.fault_trace, data.fault_summary),
        key="v2_guided_primary",
    )
    event = _primary_fault_event(data)
    if event is None:
        return [
            ("Detector", "No persisted detector event in this artifact.", "warning")
        ]
    return [
        (
            "Operator sees",
            f"Suspected <b>C{event.cell_index + 1}</b> at <b>{event.detection_time_s:.0f} s</b> "
            f"(score {event.anomaly_score:.0f}). The detector never receives the fault label.",
            "danger",
        ),
        (
            "Lead time",
            f"The cell crossed the limit at "
            f"<b>{_optional(data.fault_summary.threshold_crossing_time_s, 's')}</b>, giving "
            f"<b>{_optional(data.fault_summary.lead_time_s, 's')}</b> of warning.",
            "forecast",
        ),
    ]


def _step_evidence(
    data: DashboardData, evaluation: FinalEvaluation | None
) -> list[SideCard]:
    if evaluation is None:
        st.info("Load the final-evaluation directory to show model evidence.")
        return []
    _plot(model_error_figure(evaluation, height=360), key="v2_guided_primary")
    _plot(residual_figure(evaluation, height=290), key="v2_guided_secondary")
    return [
        (
            "Honest reading",
            "One-step error is small; error grows sharply with horizon. Out of "
            "distribution, the physics-informed twin has lower error and residual.",
            "forecast",
        ),
        (
            "Limitations",
            "<br>".join(f"• {item}" for item in evaluation.limitations[:3]),
            "warning",
        ),
    ]


_STEP_RENDERERS: Mapping[
    int, Callable[[DashboardData, FinalEvaluation | None], list[SideCard]]
] = {
    1: _step_energy,
    2: _step_heat_moves,
    3: _step_cooling,
    4: _step_forecast,
    5: _step_controller,
    6: _step_fault,
    7: _step_evidence,
}
_STEP_LABELS = tuple(f"{step.number:02d} · {step.title}" for step in STORY_STEPS)


def _move_step(offset: int) -> None:
    current = st.session_state.get("guided_story_step", _STEP_LABELS[0])
    index = _STEP_LABELS.index(current) + offset
    st.session_state["guided_story_step"] = _STEP_LABELS[
        min(max(index, 0), len(_STEP_LABELS) - 1)
    ]


def _story_rail() -> StoryStep:
    with st.container(key="tt_story"):
        selected = st.radio("Story", _STEP_LABELS, key="guided_story_step")
    index = _STEP_LABELS.index(selected)
    st.markdown(
        f'<div class="tt-progress"><span style="width:{(index + 1) / len(_STEP_LABELS):.0%}">'
        "</span></div>",
        unsafe_allow_html=True,
    )
    previous, following = st.columns(2)
    previous.button(
        "◀ Prev",
        key="guided_prev",
        on_click=_move_step,
        args=(-1,),
        disabled=index == 0,
        width="stretch",
    )
    following.button(
        "Next ▶",
        key="guided_next",
        on_click=_move_step,
        args=(1,),
        disabled=index == len(_STEP_LABELS) - 1,
        type="primary",
        width="stretch",
    )
    return STORY_STEPS[index]


def render_guided_demo(data: DashboardData, evaluation: FinalEvaluation | None) -> None:
    _hero(
        "Guided demo",
        "Follow heat from the cell to the cold plate, then see the twin act on it.",
        "Seven steps. Each one shows the saved evidence for one physical idea and the "
        "equation behind it. Use Next ▶ to walk through.",
    )
    rail, main, side = st.columns([1.0, 2.7, 1.15], gap="medium")
    with rail:
        step = _story_rail()
    with main:
        st.markdown(f"### {step.number}. {step.title}")
        st.caption(step.summary)
        cards = _STEP_RENDERERS[step.number](data, evaluation)
    with side:
        _side_cards(cards)
    _formula_group(step.formula_ids, title="Equations for this step")


# ---------------------------------------------------------------- live lab


def _run_live_simulation(configuration: LiveConfiguration) -> None:
    state = st.session_state
    duration_s = int(state.lab_duration)
    controller_name = str(state.lab_controller)
    predictive = controller_name in _PREDICTIVE_TWINS
    twin_name = (
        _PREDICTIVE_TWINS[controller_name]
        if predictive
        else (None if state.lab_overlay == "None" else str(state.lab_overlay))
    )
    fault_enabled = bool(state.lab_fault)
    fault_start = float(state.lab_fault_start)
    if fault_enabled and fault_start >= duration_s:
        fault_start = float(duration_s // 2)
        st.warning(f"Fault start moved to {fault_start:.0f} s to fit inside the run.")
    environment = make_environment(
        configuration,
        ambient_temperature_k=float(state.lab_ambient) + 273.15,
        coolant_inlet_temperature_k=float(state.lab_coolant) + 273.15,
        initial_soc_fraction=0.8,
        initial_cell_temperature_k=float(state.lab_initial) + 273.15,
        fault_cell_index=int(state.lab_fault_cell) - 1 if fault_enabled else None,
        fault_start_time_s=fault_start,
        resistance_multiplier=float(state.lab_fault_multiplier),
    )
    profile = make_current_profile(
        str(state.lab_profile),
        duration_s=float(duration_s),
        peak_current_a=float(state.lab_current),
    )
    forecaster = (
        _live_forecaster_cached(twin_name, str(_DEFAULT_COMPARISON_DIRECTORY))
        if twin_name is not None
        else None
    )
    controller = make_controller(configuration, controller_name, forecaster=forecaster)
    request = SimulationRequest(
        scenario_name=str(state.lab_profile),
        environment_config=environment,
        current_profile=profile,
        controller_name=controller_name,
        model_name=twin_name,
        duration_s=float(duration_s),
        time_step_s=environment.simulation.default_time_step_s,
        seed=1729,
        metadata={"fault_enabled": fault_enabled},
    )
    session = SimulationSession(
        request,
        controller,
        forecaster,
        forecast_horizon_steps=12 if forecaster is not None else 0,
    )
    snapshots = session.run_to_completion()
    state.lab_result = snapshots
    state.lab_run_info = [
        ("Profile", str(state.lab_profile)),
        ("Peak current", f"{float(state.lab_current):.0f} A"),
        ("Duration", f"{duration_s} s"),
        ("Ambient", f"{float(state.lab_ambient):.0f} °C"),
        ("Coolant", f"{float(state.lab_coolant):.0f} °C"),
        ("Controller", controller_name),
        ("Twin", twin_name or "none"),
        (
            "Fault",
            f"C{int(state.lab_fault_cell)} ×{float(state.lab_fault_multiplier):.1f} "
            f"from {fault_start:.0f} s"
            if fault_enabled
            else "none",
        ),
    ]
    state.lab_frame = 0
    state.lab_playing = True


def _conditions_form(configuration: LiveConfiguration) -> bool:
    with st.form("live_conditions"):
        load, environment, cooling, fault = st.columns(4, gap="medium")
        with load:
            st.markdown("**Electrical load**")
            st.selectbox(
                "Current profile", configuration.profile_names, key="lab_profile"
            )
            st.slider("Peak current (A)", 20, 240, 160, 10, key="lab_current")
            st.slider("Duration (s)", 10, 120, 60, 10, key="lab_duration")
        with environment:
            st.markdown("**Environment**")
            st.slider("Ambient (°C)", 10, 50, 30, 1, key="lab_ambient")
            st.slider("Initial cell temperature (°C)", 15, 45, 28, 1, key="lab_initial")
        with cooling:
            st.markdown("**Cooling & twin**")
            st.selectbox(
                "Controller", tuple(_CONTROLLERS), index=3, key="lab_controller"
            )
            st.slider("Coolant inlet (°C)", 5, 40, 20, 1, key="lab_coolant")
            st.selectbox(
                "Forecast overlay (reactive controllers)",
                ("None", "Data-only twin", "Physics-informed twin"),
                key="lab_overlay",
                help="Predictive controllers always use their own twin for decisions.",
            )
        with fault:
            st.markdown("**Fault injection**")
            st.checkbox("Inject resistance fault", key="lab_fault")
            st.selectbox(
                "Faulted cell", tuple(range(1, 13)), index=5, key="lab_fault_cell"
            )
            st.slider("Resistance ×", 1.5, 5.0, 3.0, 0.5, key="lab_fault_multiplier")
            st.number_input("Fault start (s)", 0, 119, 10, 1, key="lab_fault_start")
        return st.form_submit_button("▶  Run simulation", type="primary")


def _toggle_playback(frame_count: int) -> None:
    state = st.session_state
    if (
        not state.get("lab_playing", False)
        and int(state.get("lab_frame", 0)) >= frame_count - 1
    ):
        state.lab_frame = 0
    state.lab_playing = not state.get("lab_playing", False)


def _live_playback(
    snapshots: Sequence[SimulationSnapshot], configuration: LiveConfiguration
) -> None:
    states = [snapshot.state for snapshot in snapshots]
    times = np.array([state.time_s for state in states])
    temperatures = np.stack([state.cell_temperatures_k for state in states])
    flows = np.array(
        [snapshot.control_action.coolant_mass_flow_kg_s for snapshot in snapshots]
    )
    pump = np.array(
        [
            pump_power_w(flow, configuration.pump_power_coefficient_w_s3_kg3)
            for flow in flows
        ]
    )
    forecast_arrays = [
        snapshot.forecast_states[-1].cell_temperatures_k
        for snapshot in snapshots
        if snapshot.forecast_states
    ]
    value_range = shared_range_c(temperatures, *forecast_arrays)
    frame_count = len(snapshots)
    playing = bool(st.session_state.get("lab_playing", False))

    @st.fragment(run_every=_PLAYBACK_INTERVAL_S if playing else None)
    def _player() -> None:
        state = st.session_state
        if state.get("lab_playing", False):
            step = _SPEEDS.get(str(state.get("lab_speed", "1×")), 1)
            state.lab_frame = min(
                int(state.get("lab_frame", 0)) + step, frame_count - 1
            )
        controls = st.columns([1, 2.4, 5], vertical_alignment="center")
        if controls[0].button(
            "⏸ Pause" if state.get("lab_playing", False) else "▶ Play",
            key="lab_play",
            on_click=_toggle_playback,
            args=(frame_count,),
            width="stretch",
        ):
            # The timer interval is fixed when the fragment is defined, so a full
            # rerun is needed to start or stop the playback clock.
            st.rerun()
        controls[1].radio(
            "Speed",
            tuple(_SPEEDS),
            horizontal=True,
            key="lab_speed",
            label_visibility="collapsed",
        )
        with controls[2]:
            if frame_count > 1:
                st.slider(
                    "Timeline (s)",
                    0,
                    frame_count - 1,
                    key="lab_frame",
                    format="%d",
                    label_visibility="collapsed",
                )
        frame = int(state.get("lab_frame", 0))
        snapshot = snapshots[frame]
        live_state = snapshot.state
        cells = live_state.cell_temperatures_k
        telemetry = st.columns(6)
        telemetry[0].metric("Time", f"{live_state.time_s:.0f} s")
        telemetry[1].metric("Current", f"{live_state.current_a:.0f} A")
        telemetry[2].metric("SOC", f"{live_state.soc_fraction * 100:.1f} %")
        telemetry[3].metric("Tmax", _temperature(float(np.max(cells))))
        telemetry[4].metric("Spread", f"{float(np.ptp(cells)):.2f} K")
        telemetry[5].metric("Coolant flow", f"{flows[frame]:.3f} kg/s")
        hottest = int(np.argmax(cells))
        maps = st.columns(2 if snapshot.forecast_states else 1)
        with maps[0]:
            _plot(
                pack_map(
                    cells,
                    title=f"Simulator (ground truth) · t = {live_state.time_s:.0f} s",
                    range_c=value_range,
                    highlight=hottest if float(np.ptp(cells)) > 0.05 else None,
                ),
                key="v2_live_heatmap",
            )
        if snapshot.forecast_states:
            with maps[1]:
                horizon = len(snapshot.forecast_states)
                _plot(
                    pack_map(
                        snapshot.forecast_states[-1].cell_temperatures_k,
                        title=f"Twin forecast · +{horizon} s",
                        range_c=value_range,
                    ),
                    key="v2_live_forecast_heatmap",
                )
        terms = snapshot.thermal_terms_w
        balance = st.columns(3)
        with balance[0]:
            _panel(
                "Energy balance (pack)",
                f"Generated <b>{terms['Q_gen_total_w']:.0f} W</b> · cold plate "
                f"<b>{terms['Q_cool_total_w']:.0f} W</b> · ambient "
                f"<b>{terms['Q_ambient_total_w']:.0f} W</b>",
                "coolant",
            )
        with balance[1]:
            resistance = float(live_state.cell_resistances_ohm[hottest])
            _panel(
                f"Hottest cell C{hottest + 1}",
                f"({live_state.current_a:.0f} A)² × {resistance * 1000:.1f} mΩ = "
                f"<b>{live_state.current_a**2 * resistance:.0f} W</b> at "
                f"{_temperature(float(cells[hottest]))}",
                "warning",
            )
        with balance[2]:
            _panel(
                "Pump",
                f"{flows[frame]:.3f} kg/s → <b>{pump[frame]:.1f} W</b> "
                f"(energy so far {float(np.sum(pump[: frame + 1])):.0f} J)",
                "forecast",
            )
        history, cooling = st.columns(2)
        with history:
            _plot(
                temperature_history(
                    times,
                    temperatures,
                    thermal_limit_k=configuration.thermal_limit_k,
                    title="Cell temperatures",
                    cursor_s=float(live_state.time_s),
                ),
                key="v2_live_temperature_history",
            )
        with cooling:
            _plot(
                coolant_history(
                    times,
                    flows,
                    pump,
                    title="Coolant command",
                    cursor_s=float(live_state.time_s),
                ),
                key="v2_live_coolant_history",
            )
        if state.get("lab_playing", False) and frame >= frame_count - 1:
            state.lab_playing = False
            st.rerun()

    _player()


def render_live_lab(data: DashboardData) -> None:
    del data
    configuration = _live_configuration_cached()
    _hero(
        "Live thermal lab",
        "Set the conditions, run the pack simulator, and play it back.",
        "The simulator is ground truth. Controllers see only the current state; "
        "predictive controllers plan with their saved twin, never future simulator values.",
    )
    if _conditions_form(configuration):
        try:
            with st.spinner("Simulating the 12-cell pack…"):
                _run_live_simulation(configuration)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            st.error(f"Live simulation could not run: {exc}")
            return
    snapshots = st.session_state.get("lab_result")
    if not snapshots:
        st.info("Choose the run conditions above and press **Run simulation**.")
        return
    _chips(st.session_state.get("lab_run_info", []))
    _live_playback(snapshots, configuration)
    _formula_group(("energy_balance", "joule", "cooling", "pump"))


# ---------------------------------------------------------------- controllers & twins

_SCORECARD_FORMAT = {
    "Mean peak (°C)": st.column_config.NumberColumn(format="%.2f"),
    "Worst peak (°C)": st.column_config.NumberColumn(format="%.2f"),
    "Mean spread (K)": st.column_config.NumberColumn(format="%.2f"),
    "Mean pump energy (J)": st.column_config.NumberColumn(format="%.1f"),
    "Time > limit (s)": st.column_config.NumberColumn(format="%.0f"),
}


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
                "Max spread (K)": round(
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


def _provenance_and_downloads(
    data: DashboardData,
    evaluation: FinalEvaluation | None,
    controller_key: str,
    directories: Mapping[str, str],
) -> None:
    trace = data.controller_traces[controller_key]
    with st.expander("Provenance, limitations & downloads"):
        for label, directory in directories.items():
            st.markdown(f"**{label}**")
            st.code(directory or "not loaded", language=None)
        if evaluation is not None:
            st.markdown("**Stated limitations**")
            for item in evaluation.limitations:
                st.markdown(f"- {item}")
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
                    f"cell_{index + 1}_temperature_k": trace.cell_temperatures_k[
                        :, index
                    ]
                    for index in range(12)
                },
            }
        )
        export_summary = {
            "schema_version": 1,
            "controller": controller_key,
            "artifact_roots": dict(directories),
            "metrics": dict(trace.metrics),
            "evidence_status": evaluation.evidence_status if evaluation else None,
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
            width="stretch",
        )
        downloads[1].download_button(
            "Download evidence summary · JSON",
            json.dumps(export_summary, indent=2, sort_keys=True),
            file_name="thermotwin-evidence-summary.json",
            mime="application/json",
            width="stretch",
        )


def render_controllers_and_twins(
    data: DashboardData,
    evaluation: FinalEvaluation | None,
    directories: Mapping[str, str],
) -> None:
    _hero(
        "Controllers & twins",
        "Does predicting the pack change the cooling decision?",
        "Saved benchmark results for four controllers across six scenarios, and the "
        "accuracy of the two thermal twins they rely on.",
    )
    if evaluation is None:
        st.info(
            "Benchmark evaluation not loaded. Set **Benchmark evaluation directory** in "
            "the sidebar (for example `results/phase12-final-evaluation`)."
        )
    else:
        _chips(
            [
                ("Benchmark", evaluation.benchmark_mode),
                ("Evidence", evaluation.evidence_status),
                ("Seed", str(evaluation.benchmark_seed)),
                ("Scenarios", "6"),
                ("Controllers", "4"),
            ]
        )
        st.markdown("### Controller benchmark")
        metric_labels = {
            label: key for key, (label, _, _) in CONTROLLER_METRICS.items()
        }
        chosen = st.radio(
            "Metric", tuple(metric_labels), index=3, horizontal=True, key="ct_metric"
        )
        _plot(
            scenario_controller_figure(evaluation, metric_labels[chosen]),
            key="v2_cooling_controller_matrix",
        )
        table, note = st.columns([2.2, 1], gap="medium")
        with table:
            st.markdown("**Six-scenario summary**")
            st.dataframe(
                controller_scorecard(evaluation),
                hide_index=True,
                width="stretch",
                column_config=_SCORECARD_FORMAT,
            )
        with note:
            _panel(
                "Reading the benchmark",
                "Predictive controllers trade a fraction of a kelvin of peak "
                "temperature for much lower pump energy in most scenarios. These are "
                "short quick-mode runs, so treat them as descriptive.",
                "warning",
            )
        st.markdown("### Thermal twin accuracy")
        errors, residuals = st.columns([1.5, 1], gap="medium")
        with errors:
            _plot(model_error_figure(evaluation), key="v2_cooling_forecast_comparison")
        with residuals:
            _plot(residual_figure(evaluation), key="v2_cooling_residuals")
    st.markdown("### Saved controller run (time series)")
    controller_label = st.selectbox(
        "Controller", tuple(_CONTROLLERS), index=0, key="ct_trace_controller"
    )
    controller_key = _CONTROLLERS[controller_label]
    trace = data.controller_traces[controller_key]
    history, cooling = st.columns(2)
    with history:
        _plot(
            temperature_history(
                trace.time_s,
                trace.cell_temperatures_k,
                thermal_limit_k=trace.thermal_limit_k,
                title=f"{controller_label}: temperatures",
            ),
            key="v2_evidence_temperature_history",
        )
    with cooling:
        _plot(
            coolant_history(
                trace.time_s,
                trace.coolant_mass_flow_kg_s,
                trace.pump_power_w,
                title=f"{controller_label}: coolant",
            ),
            key="v2_evidence_coolant_history",
        )
    st.dataframe(
        _comparison_table(data.controller_traces), hide_index=True, width="stretch"
    )
    outcomes = {
        tuple(sorted(item.metrics.items())) for item in data.controller_traces.values()
    }
    if len(outcomes) == 1:
        st.caption(
            "In this saved run all four controllers produced identical results: the "
            "load never heated the pack enough for any controller to start cooling. "
            "Use the benchmark above or the Live Thermal Lab to see them diverge."
        )
    _provenance_and_downloads(data, evaluation, controller_key, directories)
    _formula_group(
        ("threshold", "pid", "predictive_cost", "pump", "mae", "physics_residual")
    )


# ---------------------------------------------------------------- fault lab


def render_fault_lab(data: DashboardData) -> None:
    summary = data.fault_summary
    trace = data.fault_trace
    event = _primary_fault_event(data)
    _hero(
        "Fault lab",
        "Excess resistance turns into a measurable hotspot precursor.",
        "The operator panel uses only observed-versus-expected temperatures. The "
        "injected fault cell and start time appear only in the evaluation panel.",
    )
    st.markdown("### Operator view")
    cards = st.columns(5)
    if event is None:
        st.info("No persisted anomaly event is available.")
    else:
        cards[0].metric("Suspected cell", f"C{event.cell_index + 1}")
        cards[1].metric("Anomaly score", f"{event.anomaly_score:.0f}")
        cards[2].metric("Detected at", f"{event.detection_time_s:.0f} s")
        cards[3].metric(
            "Predicted hotspot", _temperature(event.predicted_hotspot_temperature_k)
        )
        cards[4].metric("Detector events", str(summary.anomaly_event_count))
    story, scores = st.columns(2, gap="medium")
    with story:
        _plot(fault_story_figure(trace, summary), key="v2_fault_diagnostics")
    with scores:
        _plot(
            anomaly_score_figure(
                trace,
                suspected_cell=None if event is None else event.cell_index,
                score_threshold=_detector_settings_cached()["score_threshold"],
            ),
            key="v2_fault_scores",
        )
    pack, truth = st.columns([1.3, 1], gap="medium")
    with pack:
        times = [float(value) for value in trace.time_s]
        default = (
            int(np.argmin(np.abs(np.asarray(times) - event.detection_time_s)))
            if event is not None
            else len(times) - 1
        )
        index = _time_picker("Pack state at", times, key="fault_time", default=default)
        _plot(
            pack_map(
                trace.measured_temperature_k[index],
                title=f"Measured pack · t = {times[index]:.0f} s",
                range_c=shared_range_c(trace.measured_temperature_k),
                highlight=None if event is None else event.cell_index,
            ),
            key="v2_fault_pack",
        )
    with truth:
        st.markdown("### Evaluation-only truth")
        st.markdown(
            f"""
            <div class="tt-truth">
              Evaluation-only injected-fault truth. Configured cell
              C{summary.fault_cell_index + 1}.<br>
              <strong>Fault start:</strong> {summary.fault_start_time_s:.1f} s<br>
              <strong>Threshold crossing:</strong>
              {_optional(summary.threshold_crossing_time_s, "s")}<br>
              <strong>Lead time:</strong> {_optional(summary.lead_time_s, "s")}<br>
              <strong>Detector found the configured cell:</strong>
              {"yes" if summary.detected_fault_cell else "no"}<br>
              <strong>Peak:</strong> {_temperature(summary.peak_temperature_k)} ·
              <strong>spread:</strong> {summary.temperature_spread_k:.2f} K
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "This is fault-induced thermal anomaly detection, not thermal-runaway "
            "detection: the simulator does not model runaway chemistry."
        )
    _formula_group(
        ("joule", "anomaly_residual", "anomaly_score", "lead_time"),
        title="From resistance fault to early warning",
    )


# ---------------------------------------------------------------- application


def _sidebar_equations() -> None:
    st.sidebar.markdown("### Equation explorer")
    formula_title = st.sidebar.selectbox(
        "Equation explorer",
        tuple(spec.title for spec in FORMULAS.values()),
        key="v2_formula_selector",
        label_visibility="collapsed",
    )
    selected = next(spec for spec in FORMULAS.values() if spec.title == formula_title)
    st.sidebar.latex(selected.latex)
    st.sidebar.caption(selected.explanation)
    st.sidebar.info(
        "All values come from validated synthetic experiment artifacts. They are not "
        "evidence of production safety performance."
    )


def render_application(
    data: DashboardData,
    *,
    evaluation: FinalEvaluation | None = None,
    predictive_directory: str,
    fault_directory: str,
    evaluation_directory: str = "",
) -> None:
    """Render the complete judge-facing dashboard."""
    _sidebar_equations()
    st.markdown(
        '<div class="tt-brand"><span class="tt-brand-name">ThermoTwin</span>'
        '<span class="tt-brand-tag">Physics-informed digital twin for predictive EV '
        "battery cooling · 12-cell pack · synthetic evidence</span></div>",
        unsafe_allow_html=True,
    )
    with st.container(key="tt_nav"):
        active_view = st.radio(
            "Workspace",
            _VIEWS,
            horizontal=True,
            label_visibility="collapsed",
            key="v2_workspace",
        )
    if active_view == "Guided Demo":
        render_guided_demo(data, evaluation)
    elif active_view == "Live Thermal Lab":
        render_live_lab(data)
    elif active_view == "Controllers & Twins":
        render_controllers_and_twins(
            data,
            evaluation,
            {
                "Predictive run": predictive_directory,
                "Fault run": fault_directory,
                "Benchmark evaluation": evaluation_directory,
            },
        )
    else:
        render_fault_lab(data)


__all__ = [
    "render_application",
    "render_controllers_and_twins",
    "render_fault_lab",
    "render_guided_demo",
    "render_live_lab",
]
