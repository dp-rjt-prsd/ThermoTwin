"""Judge-facing Streamlit views for the ThermoTwin dashboard.

Layout follows the control-room design: a top navigation bar, a left rail
(story steps or run conditions), a central evidence canvas built from header-bar
cards, and a right "live telemetry / safety monitoring" rail.

* Guided Demo — seven physical story steps, each with its own evidence visual.
* Live Thermal Lab — run the authoritative simulator and play the result back.
* Controllers & Twins — saved benchmark and model evidence, plus provenance.
* Fault Lab — operator-visible anomaly evidence, kept apart from injected truth.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from html import escape
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
    anomaly_score_figure,
    conduction_flows,
    controller_scorecard,
    coolant_history,
    fault_story_figure,
    model_error_figure,
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
from thermotwin.visualization.ui_components import (
    anomaly_card,
    brand,
    card_header,
    chips,
    flow_card,
    kpi,
    kpi_grid,
    notice,
    pack_tiles,
    page_heading,
    rail_title,
    scale_legend,
    setting_block,
    truth_strip,
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
_MODEL_LABELS = {value: key for key, value in _MODELS.items()}
_VIEWS = ("Guided Demo", "Live Thermal Lab", "Controllers & Twins", "Fault Lab")
_DEFAULT_COMPARISON_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "results" / "phase12-final-comparison"
)
_PLAYBACK_INTERVAL_S = 0.5
_SPEEDS = {"1×": 1, "2×": 2, "4×": 4}
_DISCLAIMER = (
    "These saved synthetic experiment artifacts are not evidence of production "
    "safety performance."
)
SOURCE_INPUTS: tuple[tuple[str, str], ...] = (
    ("predictive_directory", "Predictive results directory"),
    ("fault_directory", "Fault results directory"),
    ("evaluation_directory", "Benchmark evaluation directory (optional)"),
)

Notice = tuple[str, str, str]


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


def _kelvin_labels(values_k: Sequence[float] | np.ndarray) -> list[str]:
    return [f"{float(value):.2f} K" for value in np.asarray(values_k).reshape(-1)]


# ---------------------------------------------------------------- components


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@contextmanager
def _card(
    key: str,
    label: str,
    meta: str = "",
    *,
    icon_name: str | None = None,
    tone: str = "muted",
) -> Iterator[None]:
    with st.container(key=f"tt_card_{key}"):
        _html(card_header(label, meta, icon_name=icon_name, tone=tone))
        yield


def _bare(figure: go.Figure, *, top: int = 34) -> go.Figure:
    """Drop the in-chart title: the card header carries it."""
    figure.update_layout(title_text="", margin=dict(t=top))
    return figure


def _plot(figure: go.Figure, *, key: str, top: int = 34) -> None:
    st.plotly_chart(
        _bare(figure, top=top),
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        },
    )


def _tiles(key: str, markup: str, legend: str | None = None) -> None:
    with st.container(key=key):
        _html(markup + (legend or ""))


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


def _notices(items: Sequence[Notice]) -> None:
    if not items:
        return
    for column, (title, copy, tone) in zip(st.columns(len(items)), items, strict=True):
        with column:
            _html(notice(title, copy, tone))


def _ensure_index(key: str, count: int, default: int) -> int:
    if count <= 1:
        return 0
    if key not in st.session_state:
        st.session_state[key] = min(max(default, 0), count - 1)
    return int(st.session_state[key])


def _time_picker(label: str, times: Sequence[float], *, key: str) -> int:
    """Pick a sample index by its time; a single-sample trace needs no control."""
    if len(times) <= 1:
        return 0
    return int(
        st.select_slider(
            label,
            options=list(range(len(times))),
            format_func=lambda index: f"{float(times[index]):.0f} s",
            key=key,
            label_visibility="collapsed",
        )
    )


def _truth_chips(data: DashboardData) -> str:
    summary = data.fault_summary
    lead = summary.lead_time_s
    return truth_strip(
        [
            ("Injected fault", f"C{summary.fault_cell_index + 1}", "primary"),
            ("Fault start", f"{summary.fault_start_time_s:.2f} s", "primary"),
            (
                "Limit crossing",
                _optional(summary.threshold_crossing_time_s, "s"),
                "danger",
            ),
            (
                "Lead time",
                "Unavailable" if lead is None else f"+{lead:.2f} s",
                "forecast",
            ),
            (
                "Detector found configured cell",
                "yes" if summary.detected_fault_cell else "no",
                "primary",
            ),
        ]
    )


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


# ---------------------------------------------------------------- right rails


def _safety_block(data: DashboardData) -> None:
    summary = data.fault_summary
    event = _primary_fault_event(data)
    trace = data.fault_trace
    _html('<div class="tt-rail-sub">Safety monitoring</div>')
    if event is None:
        _html(
            anomaly_card(
                "No anomaly event",
                None,
                ["The saved detector raised no event."],
                tone="ok",
            )
        )
    else:
        _html(
            anomaly_card(
                "Anomaly detected",
                f"{event.anomaly_score:.2f}",
                [
                    f"Suspected cell: <b>C{event.cell_index + 1}</b>",
                    f"Predicted hotspot: <b>{_temperature(event.predicted_hotspot_temperature_k)}</b>",
                    f"Detected at <b>{event.detection_time_s:.0f} s</b>",
                ],
            )
        )
    times = np.asarray(trace.time_s, dtype=float)
    step = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    above = np.max(trace.measured_temperature_k, axis=1) > summary.thermal_limit_k
    _html(
        kpi_grid(
            [
                kpi("Violations", str(summary.thermal_violation_count)),
                kpi("Time > limit", f"{float(above.sum()) * step:.2f}", "s"),
            ]
        )
    )


def _telemetry_rail(data: DashboardData) -> None:
    """Right rail for saved evidence: the final state of the saved fault run."""
    trace = data.fault_trace
    summary = data.fault_summary
    measured = np.asarray(trace.measured_temperature_k, dtype=float)
    with st.container(key="tt_rail_right"):
        _html(rail_title("Live telemetry", live=True))
        st.caption(f"Saved fault-detection run · t = {float(trace.time_s[-1]):.0f} s")
        _html(
            setting_block(
                [
                    ("Evidence run", "Resistance-fault scenario"),
                    (
                        "Thermal twin model",
                        _MODEL_LABELS.get(summary.model_variant, summary.model_variant),
                    ),
                ]
            )
        )
        _html(
            kpi_grid(
                [
                    kpi(
                        "Peak temp", f"{_celsius(summary.peak_temperature_k):.2f}", "°C"
                    ),
                    kpi("Mean temp", f"{_celsius(float(measured.mean())):.2f}", "°C"),
                    kpi("Temp spread", f"{summary.temperature_spread_k:.2f}", "K"),
                    kpi("Cooling energy", f"{summary.pump_energy_j:.1f}", "J"),
                ]
            )
        )
        _html(
            flow_card(
                float(trace.coolant_mass_flow_kg_s[-1]), float(trace.pump_power_w[-1])
            )
        )
        _safety_block(data)


# ---------------------------------------------------------------- guided demo

_STEP_HEADINGS = {
    1: "Electrical load creates heat in every cell",
    2: "Heat spreads from hot cells to their neighbours",
    3: "The cold plate removes heat, at a pump cost",
    4: "Measured vs. Forecast Pack Temperatures",
    5: "Choosing coolant flow: safety vs. pump energy",
    6: "A resistance fault becomes a measurable hotspot",
    7: "What the saved evidence shows",
}


def _step_energy(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[Notice]:
    trace = data.controller_traces["threshold"]
    times = [float(value) for value in trace.time_s]
    peak_index = int(np.argmax(np.abs(trace.current_a)))
    index = _ensure_index("g1_time", len(times), peak_index)
    current = float(trace.current_a[index])
    resistances = np.asarray(trace.cell_resistances_ohm[index], dtype=float)
    heat_w = current**2 * resistances
    all_heat = (np.asarray(trace.current_a)[:, None] ** 2) * np.asarray(
        trace.cell_resistances_ohm
    )
    top = max(float(all_heat.max()) * 1.05, 1.0)
    with _card(
        "energy",
        "Joule heat per cell",
        f"t = {times[index]:.0f} s · I = {current:.0f} A",
        icon_name="zap",
        tone="warning",
    ):
        _time_picker("Time in saved run", times, key="g1_time")
        _tiles(
            "v2_guided_primary",
            pack_tiles(
                heat_w,
                value_range=(0.0, top),
                unit="W",
                decimals=0,
                secondary=[f"R = {value * 1000:.1f} mΩ" for value in resistances],
            ),
            scale_legend((0.0, top), "W"),
        )
    st.caption(
        "Source: saved reactive-threshold run. Q = I²R per cell from the recorded "
        "pack current and cell resistance."
    )
    multiplier = _detector_settings_cached()["resistance_multiplier"]
    nominal = float(np.median(resistances))
    return [
        (
            "What to notice",
            f"At <b>{current:.0f} A</b> each healthy cell (R = <b>{nominal * 1000:.1f} mΩ</b>) "
            f"generates <b>{current**2 * nominal:.0f} W</b>. Doubling current quadruples heat.",
            "warning",
        ),
        (
            "Fault what-if",
            f"The fault experiments raise one cell's resistance ×{multiplier:g}. At the same "
            f"current that cell would generate <b>{current**2 * nominal * multiplier:.0f} W</b>.",
            "danger",
        ),
    ]


def _step_heat_moves(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[Notice]:
    trace = data.fault_trace
    configuration = _live_configuration_cached()
    times = [float(value) for value in trace.time_s]
    index = _ensure_index("g2_time", len(times), len(times) - 1)
    temperatures = np.asarray(trace.measured_temperature_k[index], dtype=float)
    hottest = int(np.argmax(temperatures))
    resistance = configuration.environment.pack.neighbor_thermal_resistance_k_w
    flows = conduction_flows(
        temperatures, pack_edges(configuration.environment.pack), resistance
    )
    value_range = shared_range_c(trace.measured_temperature_k)
    with _card(
        "conduction",
        "Measured pack · conduction paths",
        f"t = {times[index]:.0f} s",
        icon_name="flame",
        tone="danger",
    ):
        _time_picker("Time in saved fault run", times, key="g2_time")
        _tiles(
            "v2_guided_primary",
            pack_tiles(
                temperatures - 273.15,
                value_range=value_range,
                unit="°C",
                secondary=_kelvin_labels(temperatures),
                highlight=hottest if float(np.ptp(temperatures)) > 0.05 else None,
                flows=flows[:8],
            ),
            scale_legend(value_range, "°C"),
        )
    st.caption(
        f"Source: saved fault run. Connectors show hot → cold conduction, (T_hot − T_cold) / R_th "
        f"with R_th = {resistance:g} K/W."
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
            "Conduction spreads a hotspot into its neighbours, so one bad cell raises the "
            "pack's peak temperature and non-uniformity.",
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
) -> list[Notice]:
    trace = data.fault_trace
    limit = data.fault_summary.thermal_limit_k
    with _card(
        "cooling_temp", "Pack temperature", "saved fault run", icon_name="activity"
    ):
        _plot(
            temperature_history(
                trace.time_s,
                trace.measured_temperature_k,
                thermal_limit_k=limit,
                title="Pack temperature",
                height=270,
            ),
            key="v2_guided_primary",
        )
    with _card(
        "cooling_flow",
        "Coolant command · reactive threshold",
        "",
        icon_name="waves",
        tone="coolant",
    ):
        _plot(
            coolant_history(
                trace.time_s,
                trace.coolant_mass_flow_kg_s,
                trace.pump_power_w,
                title="Coolant",
                height=240,
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
            "Pump power grows with ṁ³: going from 0.06 to 0.1 kg/s multiplies pump power "
            "by about 4.6×.",
            "warning",
        ),
    ]


def _step_forecast(
    data: DashboardData, evaluation: FinalEvaluation | None
) -> list[Notice]:
    controls = st.columns([1.2, 1])
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
    measured_k = np.asarray(snapshot.source_cell_temperatures_k, dtype=float)
    predicted_k = np.asarray(snapshot.predicted_cell_temperatures_k, dtype=float)
    value_range = shared_range_c(measured_k, predicted_k)
    measured, predicted = st.columns(2, gap="medium")
    with (
        measured,
        _card("measured", "Measured state", f"T = {snapshot.source_time_s:.2f} s"),
    ):
        _tiles(
            "v2_guided_measured_heatmap",
            pack_tiles(
                measured_k - 273.15,
                value_range=value_range,
                unit="°C",
                secondary=_kelvin_labels(measured_k),
                highlight=int(np.argmax(measured_k))
                if float(np.ptp(measured_k)) > 0.05
                else None,
            ),
        )
    with (
        predicted,
        _card(
            "forecast",
            "Forecast state",
            f"+{horizon_s:.2f} s horizon",
            icon_name="forward",
            tone="forecast",
        ),
    ):
        _tiles(
            "v2_guided_predicted_heatmap",
            pack_tiles(
                predicted_k - 273.15,
                value_range=value_range,
                unit="°C",
                secondary=_kelvin_labels(predicted_k),
                forecast=True,
            ),
        )
    _html(scale_legend(value_range, "°C"))
    st.caption(
        f"Both maps share one colour scale. Source: threshold-run state; the {model_label.lower()} "
        f"holds coolant at the saved {snapshot.action_coolant_mass_flow_kg_s:.3f} kg/s."
    )
    one_step_error = float(np.nanmean(np.abs(data.fault_trace.residual_k[:10])))
    notices: list[Notice] = [
        (
            "Short-horizon tracking",
            "Before the fault starts, the twin's one-step error in the saved fault run "
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
            notices.append(
                (
                    f"Measured error at +{horizon_s} s",
                    f"Saved evaluation MAE for this twin: <b>{mae:.2f} K</b> "
                    "(in-distribution). Read long-horizon maps with that error in mind.",
                    "warning" if mae > 5 else "forecast",
                )
            )
    return notices


def _step_controller(
    data: DashboardData, evaluation: FinalEvaluation | None
) -> list[Notice]:
    if evaluation is None:
        trace = data.controller_traces["threshold"]
        with _card("saved_run", "Saved run temperature"):
            _plot(
                temperature_history(
                    trace.time_s,
                    trace.cell_temperatures_k,
                    thermal_limit_k=trace.thermal_limit_k,
                    title="Saved run",
                ),
                key="v2_guided_primary",
            )
        return [
            (
                "Benchmark not loaded",
                "Add the benchmark evaluation directory under Evidence to compare "
                "controllers across six scenarios.",
                "warning",
            )
        ]
    with _card(
        "benchmark",
        "Pump energy by scenario",
        f"{evaluation.benchmark_mode} benchmark",
        icon_name="gauge",
    ):
        _plot(
            scenario_controller_figure(evaluation, "pump_energy_j", height=400),
            key="v2_guided_primary",
            top=16,
        )
    table = controller_scorecard(evaluation)
    with _card("scorecard", "Six-scenario summary"):
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
            f"<b>{saving:.0%} less pump energy</b> than threshold control, with mean peak "
            f"temperature <b>{peak_gap:+.2f} K</b> different.",
            "forecast",
        ),
        (
            "Evidence scope",
            f"Benchmark <b>{evaluation.benchmark_mode}</b>, status "
            f"<b>{evaluation.evidence_status}</b>, seed {evaluation.benchmark_seed}. "
            "Descriptive, not statistically significant.",
            "warning",
        ),
    ]


def _step_fault(
    data: DashboardData, _evaluation: FinalEvaluation | None
) -> list[Notice]:
    summary = data.fault_summary
    with _card(
        "fault_story",
        f"C{summary.fault_cell_index + 1} vs the rest of the pack",
        "saved fault run",
        icon_name="alert",
        tone="danger",
    ):
        _plot(
            fault_story_figure(data.fault_trace, summary),
            key="v2_guided_primary",
            top=16,
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
            f"The cell crossed the limit at <b>{_optional(summary.threshold_crossing_time_s, 's')}</b>, "
            f"giving <b>{_optional(summary.lead_time_s, 's')}</b> of warning.",
            "forecast",
        ),
    ]


def _step_evidence(
    _data: DashboardData, evaluation: FinalEvaluation | None
) -> list[Notice]:
    if evaluation is None:
        st.info("Load the benchmark evaluation directory to show model evidence.")
        return []
    errors, residuals = st.columns([1.5, 1], gap="medium")
    with (
        errors,
        _card(
            "errors",
            "Twin forecast error by horizon",
            "MAE · log scale",
            icon_name="activity",
        ),
    ):
        _plot(
            model_error_figure(evaluation, height=340), key="v2_guided_primary", top=40
        )
    with (
        residuals,
        _card(
            "residuals",
            "Physics residual",
            "lower = more consistent",
            icon_name="sigma",
        ),
    ):
        _plot(
            residual_figure(evaluation, height=340), key="v2_guided_secondary", top=40
        )
    return [
        (
            "Honest reading",
            "One-step error is small; error grows sharply with horizon. Out of distribution, "
            "the physics-informed twin has lower error and residual.",
            "forecast",
        ),
        (
            "Limitations",
            "<br>".join(f"• {escape(item)}" for item in evaluation.limitations[:2]),
            "warning",
        ),
    ]


_STEP_RENDERERS: Mapping[
    int, Callable[[DashboardData, FinalEvaluation | None], list[Notice]]
] = {
    1: _step_energy,
    2: _step_heat_moves,
    3: _step_cooling,
    4: _step_forecast,
    5: _step_controller,
    6: _step_fault,
    7: _step_evidence,
}


def _set_step(number: int) -> None:
    st.session_state["guided_step"] = min(max(number, 1), len(STORY_STEPS))


def _story_rail() -> StoryStep:
    current = int(st.session_state.setdefault("guided_step", 1))
    with st.container(key="tt_rail_left"):
        _html(
            '<div class="tt-rail-eyebrow">Story sequence</div>'
            '<div class="tt-rail-heading">From cell heat to proactive cooling</div>'
        )
        for step in STORY_STEPS:
            active = "on_" if step.number == current else ""
            with st.container(key=f"tt_step_{active}{step.number}"):
                st.button(
                    step.title,
                    key=f"guided_step_{step.number}",
                    on_click=_set_step,
                    args=(step.number,),
                    width="stretch",
                )
                _html(f'<div class="tt-step-desc">{escape(step.summary)}</div>')
        _html(
            f'<div class="tt-progress"><span style="width:{current / len(STORY_STEPS):.0%}"></span></div>'
        )
        previous, following = st.columns(2)
        previous.button(
            "◀ Prev",
            key="guided_prev",
            on_click=_set_step,
            args=(current - 1,),
            disabled=current == 1,
            width="stretch",
        )
        following.button(
            "Next ▶",
            key="guided_next",
            on_click=_set_step,
            args=(current + 1,),
            disabled=current == len(STORY_STEPS),
            type="primary",
            width="stretch",
        )
    return STORY_STEPS[current - 1]


def render_guided_demo(data: DashboardData, evaluation: FinalEvaluation | None) -> None:
    rail, center, right = st.columns([1.0, 3.2, 1.25], gap="medium")
    with rail:
        step = _story_rail()
    with center:
        _html(page_heading(_STEP_HEADINGS[step.number], step.summary, _DISCLAIMER))
        notices = _STEP_RENDERERS[step.number](data, evaluation)
        _notices(notices)
        _html(_truth_chips(data))
        _formula_group(step.formula_ids, title="Equations for this step")
    with right:
        _telemetry_rail(data)


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
    state.lab_result = session.run_to_completion()
    state.lab_run_info = {
        "controller": controller_name,
        "twin": twin_name or "none",
        "chips": [
            ("Profile", str(state.lab_profile)),
            ("Peak current", f"{float(state.lab_current):.0f} A"),
            ("Duration", f"{duration_s} s"),
            ("Ambient", f"{float(state.lab_ambient):.0f} °C"),
            ("Coolant", f"{float(state.lab_coolant):.0f} °C"),
            (
                "Fault",
                f"C{int(state.lab_fault_cell)} ×{float(state.lab_fault_multiplier):.1f} from {fault_start:.0f} s"
                if fault_enabled
                else "none",
            ),
        ],
    }
    state.lab_frame = 0
    state.lab_playing = True


def _conditions_form(configuration: LiveConfiguration) -> bool:
    with st.form("live_conditions", border=False):
        _html('<div class="tt-rail-sub">Electrical load</div>')
        st.selectbox("Current profile", configuration.profile_names, key="lab_profile")
        st.slider("Peak current (A)", 20, 240, 160, 10, key="lab_current")
        st.slider("Duration (s)", 10, 120, 60, 10, key="lab_duration")
        _html('<div class="tt-rail-sub">Environment</div>')
        st.slider("Ambient (°C)", 10, 50, 30, 1, key="lab_ambient")
        st.slider("Initial cell temperature (°C)", 15, 45, 28, 1, key="lab_initial")
        _html('<div class="tt-rail-sub">Cooling & twin</div>')
        st.selectbox("Controller", tuple(_CONTROLLERS), index=3, key="lab_controller")
        st.slider("Coolant inlet (°C)", 5, 40, 20, 1, key="lab_coolant")
        st.selectbox(
            "Forecast overlay (reactive controllers)",
            ("None", "Data-only twin", "Physics-informed twin"),
            key="lab_overlay",
            help="Predictive controllers always use their own twin for decisions.",
        )
        _html('<div class="tt-rail-sub">Fault injection</div>')
        st.checkbox("Inject resistance fault", key="lab_fault")
        fault_cell, multiplier = st.columns(2)
        fault_cell.selectbox("Cell", tuple(range(1, 13)), index=5, key="lab_fault_cell")
        multiplier.number_input(
            "Resistance ×", 1.5, 5.0, 3.0, 0.5, key="lab_fault_multiplier"
        )
        st.number_input("Fault start (s)", 0, 119, 10, 1, key="lab_fault_start")
        return st.form_submit_button(
            "▶ Run simulation", type="primary", width="stretch"
        )


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
    limit_k = configuration.thermal_limit_k
    run_info = st.session_state.get("lab_run_info", {})
    playing = bool(st.session_state.get("lab_playing", False))

    @st.fragment(run_every=_PLAYBACK_INTERVAL_S if playing else None)
    def _player() -> None:
        state = st.session_state
        if state.get("lab_playing", False):
            step = _SPEEDS.get(str(state.get("lab_speed", "1×")), 1)
            state.lab_frame = min(
                int(state.get("lab_frame", 0)) + step, frame_count - 1
            )
        center, right = st.columns([3.2, 1.25], gap="medium")
        with center:
            controls = st.columns([1, 1.9, 5], vertical_alignment="center")
            if controls[0].button(
                "⏸ Pause" if state.get("lab_playing", False) else "▶ Play",
                key="lab_play",
                on_click=_toggle_playback,
                args=(frame_count,),
                width="stretch",
            ):
                # The timer interval is fixed when the fragment is defined, so a
                # full rerun is needed to start or stop the playback clock.
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
            cells = np.asarray(live_state.cell_temperatures_k, dtype=float)
            hottest = int(np.argmax(cells))
            maps = st.columns(2 if snapshot.forecast_states else 1, gap="medium")
            with (
                maps[0],
                _card(
                    "live_sim",
                    "Simulator · ground truth",
                    f"T = {live_state.time_s:.2f} s",
                ),
            ):
                _tiles(
                    "v2_live_heatmap",
                    pack_tiles(
                        cells - 273.15,
                        value_range=value_range,
                        unit="°C",
                        secondary=_kelvin_labels(cells),
                        highlight=hottest if float(np.ptp(cells)) > 0.05 else None,
                    ),
                )
            if snapshot.forecast_states:
                forecast_cells = np.asarray(
                    snapshot.forecast_states[-1].cell_temperatures_k, dtype=float
                )
                with (
                    maps[1],
                    _card(
                        "live_forecast",
                        "Twin forecast",
                        f"+{len(snapshot.forecast_states):.2f} s horizon",
                        icon_name="forward",
                        tone="forecast",
                    ),
                ):
                    _tiles(
                        "v2_live_forecast_heatmap",
                        pack_tiles(
                            forecast_cells - 273.15,
                            value_range=value_range,
                            unit="°C",
                            secondary=_kelvin_labels(forecast_cells),
                            forecast=True,
                        ),
                    )
            _html(scale_legend(value_range, "°C"))
            history, cooling = st.columns(2, gap="medium")
            with (
                history,
                _card("live_history", "Cell temperatures", "", icon_name="activity"),
            ):
                _plot(
                    temperature_history(
                        times,
                        temperatures,
                        thermal_limit_k=limit_k,
                        title="Cell temperatures",
                        cursor_s=float(live_state.time_s),
                    ),
                    key="v2_live_temperature_history",
                )
            with (
                cooling,
                _card(
                    "live_coolant",
                    "Coolant command",
                    "",
                    icon_name="waves",
                    tone="coolant",
                ),
            ):
                _plot(
                    coolant_history(
                        times,
                        flows,
                        pump,
                        title="Coolant",
                        cursor_s=float(live_state.time_s),
                    ),
                    key="v2_live_coolant_history",
                )
            terms = snapshot.thermal_terms_w
            resistance = float(live_state.cell_resistances_ohm[hottest])
            _notices(
                [
                    (
                        "Energy balance (pack)",
                        f"Generated <b>{terms['Q_gen_total_w']:.0f} W</b> · cold plate "
                        f"<b>{terms['Q_cool_total_w']:.0f} W</b> · ambient <b>{terms['Q_ambient_total_w']:.0f} W</b>",
                        "coolant",
                    ),
                    (
                        f"Hottest cell C{hottest + 1}",
                        f"({live_state.current_a:.0f} A)² × {resistance * 1000:.1f} mΩ = "
                        f"<b>{live_state.current_a**2 * resistance:.0f} W</b> at {_temperature(float(cells[hottest]))}",
                        "warning",
                    ),
                ]
            )
        with right, st.container(key="tt_rail_right"):
            _html(
                rail_title("Live telemetry", live=bool(state.get("lab_playing", False)))
            )
            _html(
                setting_block(
                    [
                        ("Active controller", str(run_info.get("controller", "—"))),
                        ("Thermal twin model", str(run_info.get("twin", "none"))),
                    ]
                )
            )
            _html(
                kpi_grid(
                    [
                        kpi("Time", f"{live_state.time_s:.0f}", "s"),
                        kpi("Current", f"{live_state.current_a:.0f}", "A"),
                        kpi("Peak temp", f"{_celsius(float(cells.max())):.2f}", "°C"),
                        kpi("Mean temp", f"{_celsius(float(cells.mean())):.2f}", "°C"),
                        kpi("Temp spread", f"{float(np.ptp(cells)):.2f}", "K"),
                        kpi(
                            "Pump energy",
                            f"{float(np.sum(pump[: frame + 1])):.1f}",
                            "J",
                        ),
                        kpi("SOC", f"{live_state.soc_fraction * 100:.1f}", "%"),
                        kpi("Frame", f"{frame + 1}/{frame_count}"),
                    ]
                )
            )
            _html(flow_card(float(flows[frame]), float(pump[frame])))
            _html('<div class="tt-rail-sub">Safety monitoring</div>')
            above = temperatures[: frame + 1].max(axis=1) > limit_k
            if cells.max() > limit_k:
                _html(
                    anomaly_card(
                        "Above thermal limit",
                        None,
                        [
                            f"Hottest cell: <b>C{hottest + 1}</b>",
                            f"Tmax <b>{_temperature(float(cells.max()))}</b> > limit {_temperature(limit_k)}",
                        ],
                    )
                )
            else:
                _html(
                    anomaly_card(
                        "Within thermal limit",
                        None,
                        [
                            f"Tmax <b>{_temperature(float(cells.max()))}</b>",
                            f"limit {_temperature(limit_k)}",
                        ],
                        tone="ok",
                    )
                )
            _html(
                kpi_grid(
                    [
                        kpi("Violations", str(int(above.sum()))),
                        kpi("Time > limit", f"{float(above.sum()):.2f}", "s"),
                    ]
                )
            )
        if state.get("lab_playing", False) and frame >= frame_count - 1:
            state.lab_playing = False
            st.rerun()

    _player()


def render_live_lab(data: DashboardData) -> None:
    del data
    configuration = _live_configuration_cached()
    rail, main = st.columns([1.0, 4.45], gap="medium")
    with rail, st.container(key="tt_rail_left"):
        _html(
            '<div class="tt-rail-eyebrow">Run conditions</div>'
            '<div class="tt-rail-heading">Configure the 12-cell pack</div>'
        )
        submitted = _conditions_form(configuration)
    with main:
        _html(
            page_heading(
                "Live Thermal Lab",
                "The simulator is ground truth. Controllers see only the current state; "
                "predictive controllers plan with their saved twin, never future simulator values.",
            )
        )
        if submitted:
            try:
                with st.spinner("Simulating the 12-cell pack…"):
                    _run_live_simulation(configuration)
            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                st.error(f"Live simulation could not run: {exc}")
                return
        snapshots = st.session_state.get("lab_result")
        if not snapshots:
            st.info(
                "Choose the run conditions on the left and press **Run simulation**."
            )
            _formula_group(("energy_balance", "joule", "cooling", "pump"))
            return
        _html(chips(st.session_state.get("lab_run_info", {}).get("chips", [])))
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
            "scientific_scope": "synthetic experiment evidence; not production safety validation",
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


def _benchmark_rail(evaluation: FinalEvaluation | None) -> None:
    with st.container(key="tt_rail_right"):
        _html(rail_title("Benchmark summary"))
        if evaluation is None:
            st.caption("Benchmark evaluation not loaded.")
            return
        _html(
            setting_block(
                [
                    ("Benchmark mode", evaluation.benchmark_mode),
                    ("Evidence status", evaluation.evidence_status),
                ]
            )
        )
        table = controller_scorecard(evaluation)
        by_name = {row["Controller"]: row for row in table.to_dict("records")}
        reactive = by_name["Reactive threshold"]
        predictive = by_name["Predictive · physics-informed twin"]
        saving = 1.0 - float(predictive["Mean pump energy (J)"]) / max(
            float(reactive["Mean pump energy (J)"]), 1e-9
        )
        _html(
            kpi_grid(
                [
                    kpi(
                        "Threshold pump",
                        f"{float(reactive['Mean pump energy (J)']):.1f}",
                        "J",
                    ),
                    kpi(
                        "PI-twin pump",
                        f"{float(predictive['Mean pump energy (J)']):.1f}",
                        "J",
                        tone="forecast",
                    ),
                    kpi("Energy saving", f"{saving:.0%}", tone="forecast"),
                    kpi(
                        "Peak Δ",
                        f"{float(predictive['Mean peak (°C)']) - float(reactive['Mean peak (°C)']):+.2f}",
                        "K",
                    ),
                ]
            )
        )
        _html('<div class="tt-rail-sub">Limitations</div>')
        for item in evaluation.limitations[:3]:
            st.caption(f"• {item}")


def render_controllers_and_twins(
    data: DashboardData,
    evaluation: FinalEvaluation | None,
    directories: Mapping[str, str],
) -> None:
    center, right = st.columns([3.6, 1.25], gap="medium")
    with center:
        _html(
            page_heading(
                "Does predicting the pack change the cooling decision?",
                "Saved benchmark results for four controllers across six scenarios, and the "
                "accuracy of the two thermal twins they rely on.",
                _DISCLAIMER,
            )
        )
        if evaluation is None:
            st.info(
                "Benchmark evaluation not loaded. Set **Benchmark evaluation directory** under "
                "Evidence in the top bar (for example `results/phase12-final-evaluation`)."
            )
        else:
            _html(
                chips(
                    [
                        ("Benchmark", evaluation.benchmark_mode),
                        ("Evidence", evaluation.evidence_status),
                        ("Seed", str(evaluation.benchmark_seed)),
                        ("Scenarios", "6"),
                        ("Controllers", "4"),
                    ]
                )
            )
            metric_labels = {
                label: key for key, (label, _, _) in CONTROLLER_METRICS.items()
            }
            with _card(
                "benchmark",
                "Controller benchmark",
                "6 scenarios × 4 controllers",
                icon_name="gauge",
            ):
                chosen = st.radio(
                    "Metric",
                    tuple(metric_labels),
                    index=3,
                    horizontal=True,
                    key="ct_metric",
                    label_visibility="collapsed",
                )
                _plot(
                    scenario_controller_figure(evaluation, metric_labels[chosen]),
                    key="v2_cooling_controller_matrix",
                    top=16,
                )
            with _card("scorecard", "Six-scenario summary"):
                st.dataframe(
                    controller_scorecard(evaluation),
                    hide_index=True,
                    width="stretch",
                    column_config=_SCORECARD_FORMAT,
                )
            errors, residuals = st.columns([1.5, 1], gap="medium")
            with (
                errors,
                _card(
                    "errors",
                    "Twin forecast error by horizon",
                    "MAE · log scale",
                    icon_name="activity",
                ),
            ):
                _plot(
                    model_error_figure(evaluation),
                    key="v2_cooling_forecast_comparison",
                    top=40,
                )
            with (
                residuals,
                _card(
                    "residuals",
                    "Physics residual",
                    "lower = more consistent",
                    icon_name="sigma",
                ),
            ):
                _plot(residual_figure(evaluation), key="v2_cooling_residuals", top=40)
        controller_label = st.selectbox(
            "Saved controller run",
            tuple(_CONTROLLERS),
            index=0,
            key="ct_trace_controller",
        )
        controller_key = _CONTROLLERS[controller_label]
        trace = data.controller_traces[controller_key]
        history, cooling = st.columns(2, gap="medium")
        with (
            history,
            _card(
                "trace_temp",
                f"{controller_label}: temperatures",
                "",
                icon_name="activity",
            ),
        ):
            _plot(
                temperature_history(
                    trace.time_s,
                    trace.cell_temperatures_k,
                    thermal_limit_k=trace.thermal_limit_k,
                    title="Temperatures",
                ),
                key="v2_evidence_temperature_history",
            )
        with (
            cooling,
            _card(
                "trace_flow",
                f"{controller_label}: coolant",
                "",
                icon_name="waves",
                tone="coolant",
            ),
        ):
            _plot(
                coolant_history(
                    trace.time_s,
                    trace.coolant_mass_flow_kg_s,
                    trace.pump_power_w,
                    title="Coolant",
                ),
                key="v2_evidence_coolant_history",
            )
        st.dataframe(
            _comparison_table(data.controller_traces), hide_index=True, width="stretch"
        )
        outcomes = {
            tuple(sorted(item.metrics.items()))
            for item in data.controller_traces.values()
        }
        if len(outcomes) == 1:
            st.caption(
                "In this saved run all four controllers produced identical results: the load "
                "never heated the pack enough for any controller to start cooling. Use the "
                "benchmark above or the Live Thermal Lab to see them diverge."
            )
        _provenance_and_downloads(data, evaluation, controller_key, directories)
        _formula_group(
            ("threshold", "pid", "predictive_cost", "pump", "mae", "physics_residual")
        )
    with right:
        _benchmark_rail(evaluation)


# ---------------------------------------------------------------- fault lab


def render_fault_lab(data: DashboardData) -> None:
    summary = data.fault_summary
    trace = data.fault_trace
    event = _primary_fault_event(data)
    center, right = st.columns([3.6, 1.25], gap="medium")
    with center:
        _html(
            page_heading(
                "Excess resistance turns into a measurable hotspot precursor",
                "The operator view uses only observed-versus-expected temperatures. The "
                "injected fault cell and start time appear only in the evaluation-only strip.",
                "Fault-induced thermal anomaly detection, not thermal-runaway detection.",
            )
        )
        story, scores = st.columns(2, gap="medium")
        with (
            story,
            _card(
                "fault_story",
                "Measured vs twin expectation",
                f"C{summary.fault_cell_index + 1}",
                icon_name="alert",
                tone="danger",
            ),
        ):
            _plot(
                fault_story_figure(trace, summary), key="v2_fault_diagnostics", top=16
            )
        with (
            scores,
            _card(
                "fault_scores",
                "Detector score per cell",
                "rolling residual z-score",
                icon_name="activity",
            ),
        ):
            _plot(
                anomaly_score_figure(
                    trace,
                    suspected_cell=None if event is None else event.cell_index,
                    score_threshold=_detector_settings_cached()["score_threshold"],
                ),
                key="v2_fault_scores",
                top=40,
            )
        times = [float(value) for value in trace.time_s]
        default = (
            int(np.argmin(np.abs(np.asarray(times) - event.detection_time_s)))
            if event is not None
            else len(times) - 1
        )
        index = _ensure_index("fault_time", len(times), default)
        measured_k = np.asarray(trace.measured_temperature_k[index], dtype=float)
        value_range = shared_range_c(trace.measured_temperature_k)
        with _card("fault_pack", "Measured pack state", f"T = {times[index]:.2f} s"):
            _time_picker("Pack state at", times, key="fault_time")
            _tiles(
                "v2_fault_pack",
                pack_tiles(
                    measured_k - 273.15,
                    value_range=value_range,
                    unit="°C",
                    secondary=_kelvin_labels(measured_k),
                    highlight=None if event is None else event.cell_index,
                ),
                scale_legend(value_range, "°C"),
            )
        _html(_truth_chips(data))
        _formula_group(
            ("joule", "anomaly_residual", "anomaly_score", "lead_time"),
            title="From resistance fault to early warning",
        )
    with right, st.container(key="tt_rail_right"):
        _html(rail_title("Operator view", live=True))
        _html(
            setting_block(
                [
                    (
                        "Detector twin",
                        _MODEL_LABELS.get(summary.model_variant, summary.model_variant),
                    ),
                    ("Detector events", str(summary.anomaly_event_count)),
                ]
            )
        )
        _html(
            kpi_grid(
                [
                    kpi(
                        "Peak temp", f"{_celsius(summary.peak_temperature_k):.2f}", "°C"
                    ),
                    kpi("Temp spread", f"{summary.temperature_spread_k:.2f}", "K"),
                    kpi(
                        "Detected at",
                        _optional(event.detection_time_s if event else None, "s"),
                    ),
                    kpi("Cooling energy", f"{summary.pump_energy_j:.1f}", "J"),
                ]
            )
        )
        _safety_block(data)


# ---------------------------------------------------------------- application


def render_source_inputs() -> None:
    """Artifact-directory inputs; values live in session state under fixed keys."""
    for key, label in SOURCE_INPUTS:
        st.text_input(label, key=key)


def _equation_explorer() -> None:
    formula_title = st.selectbox(
        "Equation explorer",
        tuple(spec.title for spec in FORMULAS.values()),
        key="v2_formula_selector",
    )
    selected = next(spec for spec in FORMULAS.values() if spec.title == formula_title)
    st.latex(selected.latex)
    st.caption(selected.explanation)
    st.caption(f"Implementation: `{selected.provenance}`")


def _header() -> str:
    with st.container(key="tt_header"):
        logo, navigation, tools = st.columns(
            [1.0, 5.2, 2.6], vertical_alignment="center"
        )
        logo.markdown(brand(), unsafe_allow_html=True)
        with navigation, st.container(key="tt_nav"):
            view = st.radio(
                "Workspace",
                _VIEWS,
                horizontal=True,
                label_visibility="collapsed",
                key="v2_workspace",
            )
        with tools:
            formulas, sources = st.columns(2)
            with formulas.popover(
                "Physics & Formulas", icon=":material/functions:", width="stretch"
            ):
                _equation_explorer()
            with sources.popover(
                "Evidence", icon=":material/folder_open:", width="stretch"
            ):
                render_source_inputs()
                st.caption(
                    "All values come from validated synthetic experiment artifacts. They are "
                    "not evidence of production safety performance."
                )
    return str(view)


def render_application(
    data: DashboardData,
    *,
    evaluation: FinalEvaluation | None = None,
    predictive_directory: str,
    fault_directory: str,
    evaluation_directory: str = "",
) -> None:
    """Render the complete judge-facing dashboard."""
    active_view = _header()
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
    "SOURCE_INPUTS",
    "render_application",
    "render_controllers_and_twins",
    "render_fault_lab",
    "render_guided_demo",
    "render_live_lab",
    "render_source_inputs",
]
