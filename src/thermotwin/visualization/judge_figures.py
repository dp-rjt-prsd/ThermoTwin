"""Celsius-first, dark-theme Plotly figures for the judge-facing dashboard.

``plots.py`` keeps the artifact-level Kelvin figures used by the legacy view and
its tests. The figures here present the same saved evidence for a judging panel:
shared colour scales, Celsius display units, and one message per chart.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from numpy.typing import NDArray
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ControllerMetrics,
    ScenarioId,
    VariantId,
)
from thermotwin.experiments.final_evaluation import FinalEvaluation
from thermotwin.visualization.contracts import FaultSummary, FaultTrace
from thermotwin.visualization.presentation import apply_plot_theme

FloatArray = NDArray[np.float64]

KELVIN_OFFSET = 273.15
TEXT = "#F3F7F9"
MUTED = "#A6B8C3"
GRID = "#1D3441"
COOLANT = "#27C2D1"
FORECAST = "#62D6A7"
WARNING = "#F4B942"
DANGER = "#F06449"
NEUTRAL = "#8FA3B0"

CONTROLLER_NAMES: Mapping[VariantId, str] = {
    "threshold": "Reactive threshold",
    "pid": "Reactive PID",
    "predictive_data_only": "Predictive · data-only twin",
    "predictive_physics_informed": "Predictive · physics-informed twin",
}
CONTROLLER_COLORS: Mapping[VariantId, str] = {
    "threshold": NEUTRAL,
    "pid": WARNING,
    "predictive_data_only": COOLANT,
    "predictive_physics_informed": FORECAST,
}
SCENARIO_NAMES: Mapping[ScenarioId, str] = {
    "normal": "Normal driving",
    "high_current": "High current",
    "hot_ambient": "Hot ambient",
    "weak_cooling": "Weak cooling",
    "fast_charging": "Fast charging",
    "resistance_fault": "Resistance fault",
}
MODEL_NAMES = {
    "data_only": "Data-only twin",
    "physics_informed": "Physics-informed twin",
}
MODEL_COLORS = {"data_only": COOLANT, "physics_informed": FORECAST}
# metric key -> (label, unit, converts kelvin to Celsius)
CONTROLLER_METRICS: Mapping[str, tuple[str, str, bool]] = {
    "peak_temperature_k": ("Peak cell temperature", "°C", True),
    "mean_temperature_k": ("Mean pack temperature", "°C", True),
    "maximum_temperature_spread_k": ("Maximum spread ΔT", "K", False),
    "pump_energy_j": ("Pump energy", "J", False),
    "thermal_violation_duration_s": ("Time above 40 °C limit", "s", False),
}
# Cold coolant blue through amber to hot red; readable on the dark surface.
THERMAL_SCALE = (
    (0.0, "#1C4E80"),
    (0.25, "#27C2D1"),
    (0.5, "#9AD57A"),
    (0.7, "#F4B942"),
    (0.85, "#F06449"),
    (1.0, "#B3202A"),
)


def to_celsius(values_k: float | FloatArray) -> float | FloatArray:
    """Convert kelvin values to Celsius for display."""
    return values_k - KELVIN_OFFSET


def shared_range_c(*arrays_k: Sequence[float] | FloatArray) -> tuple[float, float]:
    """Common Celsius colour limits, rounded outward, spanning at least 2 °C."""
    values = np.concatenate(
        [np.asarray(item, dtype=float).reshape(-1) for item in arrays_k]
    )
    values = values[np.isfinite(values)] - KELVIN_OFFSET
    if values.size == 0:
        raise ValueError("at least one finite temperature is required")
    low, high = math.floor(float(values.min())), math.ceil(float(values.max()))
    if high - low < 2:
        middle = (high + low) / 2.0
        low, high = math.floor(middle - 1.0), math.ceil(middle + 1.0)
    return float(low), float(high)


def _finish(figure: go.Figure, title: str, *, height: int) -> go.Figure:
    apply_plot_theme(figure)
    figure.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16)),
        height=height,
        margin=dict(l=10, r=10, t=52, b=10),
    )
    return figure


def pack_grid(
    values: Sequence[float] | FloatArray,
    *,
    title: str,
    value_range: tuple[float, float],
    unit: str,
    colorbar_title: str,
    highlight: int | None = None,
    hover_suffix: Sequence[str] | None = None,
    decimals: int = 1,
    height: int = 340,
) -> go.Figure:
    """Draw the C1..C12 3×4 pack, row-major, on a caller-supplied colour scale."""
    data = np.asarray(values, dtype=float)
    if data.shape != (12,):
        raise ValueError("pack_grid requires exactly 12 cell values")
    if not np.isfinite(data).all():
        raise ValueError("pack_grid values must be finite")
    low, high = value_range
    if not high > low:
        raise ValueError("value_range must be increasing")
    grid = data.reshape(3, 4)
    extra = list(hover_suffix) if hover_suffix is not None else [""] * 12
    hover = np.array(
        [
            f"C{index + 1}: {data[index]:.{decimals + 1}f} {unit}{extra[index]}"
            for index in range(12)
        ],
        dtype=object,
    ).reshape(3, 4)
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=[1, 2, 3, 4],
            y=[1, 2, 3],
            zmin=low,
            zmax=high,
            colorscale=[list(stop) for stop in THERMAL_SCALE],
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                thickness=12,
                len=0.82,
                y=0.55,
                outlinewidth=0,
            ),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )
    for index, value in enumerate(data):
        row, column = divmod(index, 4)
        position = (value - low) / (high - low)
        label_color = "#0B141C" if 0.3 <= position <= 0.78 else TEXT
        figure.add_annotation(
            x=column + 1,
            y=row + 1,
            text=(
                f"<span style='font-size:11px'>C{index + 1}</span><br>"
                f"<b>{value:.{decimals}f}</b>"
            ),
            showarrow=False,
            font=dict(color=label_color, size=15),
        )
    if highlight is not None:
        row, column = divmod(int(highlight), 4)
        figure.add_shape(
            type="rect",
            x0=column + 0.5,
            x1=column + 1.5,
            y0=row + 0.5,
            y1=row + 1.5,
            line=dict(color=TEXT, width=3),
        )
    figure.add_shape(
        type="rect",
        x0=0.5,
        x1=4.5,
        y0=3.62,
        y1=3.82,
        line=dict(width=0),
        fillcolor=COOLANT,
        opacity=0.55,
    )
    figure.add_annotation(
        x=2.5,
        y=3.72,
        text="liquid cold plate",
        showarrow=False,
        font=dict(size=11, color="#0B141C"),
    )
    figure.update_xaxes(visible=False, range=[0.45, 4.55])
    figure.update_yaxes(visible=False, range=[3.9, 0.45])
    return _finish(figure, title, height=height)


def pack_map(
    temperatures_k: Sequence[float] | FloatArray,
    *,
    title: str,
    range_c: tuple[float, float],
    highlight: int | None = None,
    height: int = 340,
) -> go.Figure:
    """Pack temperatures in Celsius, with kelvin retained in the hover text."""
    kelvin = np.asarray(temperatures_k, dtype=float)
    return pack_grid(
        kelvin - KELVIN_OFFSET,
        title=title,
        value_range=range_c,
        unit="°C",
        colorbar_title="°C",
        highlight=highlight,
        hover_suffix=[f" ({value:.2f} K)" for value in kelvin.reshape(-1)],
        height=height,
    )


def add_heat_flow_arrows(
    figure: go.Figure,
    temperatures_k: Sequence[float] | FloatArray,
    edges: Sequence[tuple[int, int]],
    thermal_resistance_k_w: float,
    *,
    count: int = 6,
    minimum_w: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Overlay the strongest hot-to-cold conduction flows, (T_hot−T_cold)/R_th."""
    temperatures = np.asarray(temperatures_k, dtype=float)
    flows: list[tuple[int, int, float]] = []
    for left, right in edges:
        difference = float(temperatures[left] - temperatures[right])
        hot, cold = (left, right) if difference >= 0 else (right, left)
        watts = abs(difference) / thermal_resistance_k_w
        if watts >= minimum_w:
            flows.append((hot, cold, watts))
    flows.sort(key=lambda item: item[2], reverse=True)
    for hot, cold, watts in flows[:count]:
        hot_row, hot_column = divmod(hot, 4)
        cold_row, cold_column = divmod(cold, 4)
        start_x, start_y = hot_column + 1, hot_row + 1
        end_x, end_y = cold_column + 1, cold_row + 1
        figure.add_annotation(
            x=start_x + 0.62 * (end_x - start_x),
            y=start_y + 0.62 * (end_y - start_y),
            ax=start_x + 0.38 * (end_x - start_x),
            ay=start_y + 0.38 * (end_y - start_y),
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=2.5,
            arrowcolor=TEXT,
            text="",
        )
        figure.add_annotation(
            x=(start_x + end_x) / 2,
            y=(start_y + end_y) / 2,
            text=f"{watts:.1f} W",
            showarrow=False,
            yshift=-12 if hot_row == cold_row else 0,
            xshift=0 if hot_row == cold_row else 24,
            font=dict(size=11, color=TEXT),
            bgcolor="rgba(7,16,24,.75)",
        )
    return flows


def temperature_history(
    times_s: Sequence[float] | FloatArray,
    temperatures_k: FloatArray,
    *,
    thermal_limit_k: float,
    title: str,
    cursor_s: float | None = None,
    highlight_cell: int | None = None,
    predicted_k: FloatArray | None = None,
    height: int = 330,
) -> go.Figure:
    """Cell temperatures (thin), Tmax (bold), mean (dashed), and the safety limit."""
    times = np.asarray(times_s, dtype=float)
    celsius = np.asarray(temperatures_k, dtype=float) - KELVIN_OFFSET
    figure = go.Figure()
    for cell in range(celsius.shape[1]):
        emphasised = cell == highlight_cell
        figure.add_trace(
            go.Scatter(
                x=times,
                y=celsius[:, cell],
                mode="lines",
                name=f"C{cell + 1}",
                showlegend=emphasised,
                line=dict(
                    color=DANGER if emphasised else "rgba(166,184,195,.28)",
                    width=3 if emphasised else 1,
                ),
                hovertemplate=f"C{cell + 1}: %{{y:.2f}} °C<extra></extra>",
            )
        )
    if predicted_k is not None and highlight_cell is not None:
        figure.add_trace(
            go.Scatter(
                x=times,
                y=np.asarray(predicted_k, dtype=float)[:, highlight_cell]
                - KELVIN_OFFSET,
                mode="lines",
                name=f"C{highlight_cell + 1} twin expectation",
                line=dict(color=FORECAST, width=2, dash="dash"),
                hovertemplate="Twin: %{y:.2f} °C<extra></extra>",
            )
        )
    if highlight_cell is None:
        figure.add_trace(
            go.Scatter(
                x=times,
                y=celsius.max(axis=1),
                mode="lines",
                name="Tmax",
                line=dict(color=DANGER, width=3),
                hovertemplate="Tmax: %{y:.2f} °C<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=celsius.mean(axis=1),
            mode="lines",
            name="Pack mean",
            line=dict(color=COOLANT, width=2, dash="dash"),
            hovertemplate="Mean: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.add_hline(
        y=thermal_limit_k - KELVIN_OFFSET,
        line=dict(color=WARNING, width=2, dash="dot"),
        annotation_text=f"limit {thermal_limit_k - KELVIN_OFFSET:.0f} °C",
        annotation_position="top left",
        annotation_font_color=WARNING,
    )
    if cursor_s is not None:
        figure.add_vline(x=cursor_s, line=dict(color=TEXT, width=1.5))
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Temperature (°C)")
    figure.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    figure = _finish(figure, title, height=height)
    figure.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return figure


def coolant_history(
    times_s: Sequence[float] | FloatArray,
    flow_kg_s: Sequence[float] | FloatArray,
    pump_power_w: Sequence[float] | FloatArray,
    *,
    title: str,
    cursor_s: float | None = None,
    height: int = 300,
) -> go.Figure:
    """Commanded coolant flow (step) with the approximate pump power it costs."""
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=list(times_s),
            y=list(flow_kg_s),
            mode="lines",
            line=dict(color=COOLANT, width=3, shape="hv"),
            name="Coolant flow",
            hovertemplate="Flow: %{y:.3f} kg/s<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=list(times_s),
            y=list(pump_power_w),
            mode="lines",
            line=dict(color=WARNING, width=2, dash="dot", shape="hv"),
            name="Pump power",
            hovertemplate="Pump: %{y:.1f} W<extra></extra>",
        ),
        secondary_y=True,
    )
    if cursor_s is not None:
        figure.add_vline(x=cursor_s, line=dict(color=TEXT, width=1.5))
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Flow (kg/s)", rangemode="tozero", secondary_y=False)
    figure.update_yaxes(
        title_text="Pump power (W)",
        rangemode="tozero",
        showgrid=False,
        secondary_y=True,
    )
    figure.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    figure = _finish(figure, title, height=height)
    figure.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return figure


def _controller_lookup(
    evaluation: FinalEvaluation,
) -> dict[tuple[ScenarioId, VariantId], ControllerMetrics]:
    return {
        (record.scenario_id, record.variant_id): record.metrics
        for record in evaluation.controller_metrics
    }


def _metric_value(metrics: ControllerMetrics, key: str) -> float:
    value = float(getattr(metrics, key))
    return value - KELVIN_OFFSET if CONTROLLER_METRICS[key][2] else value


def scenario_controller_figure(
    evaluation: FinalEvaluation, metric_key: str, *, height: int = 430
) -> go.Figure:
    """One metric across the six benchmark scenarios, grouped by controller."""
    if metric_key not in CONTROLLER_METRICS:
        raise ValueError(f"unsupported controller metric: {metric_key}")
    label, unit, is_temperature = CONTROLLER_METRICS[metric_key]
    lookup = _controller_lookup(evaluation)
    scenarios = [SCENARIO_NAMES[scenario] for scenario in SCENARIO_IDS]
    figure = go.Figure()
    all_values: list[float] = []
    for variant in VARIANT_IDS:
        values = [
            _metric_value(lookup[(scenario, variant)], metric_key)
            for scenario in SCENARIO_IDS
        ]
        all_values.extend(values)
        figure.add_trace(
            go.Bar(
                y=scenarios,
                x=values,
                orientation="h",
                name=CONTROLLER_NAMES[variant],
                marker_color=CONTROLLER_COLORS[variant],
                text=[
                    f"{value:.1f}" if value or is_temperature else ""
                    for value in values
                ],
                textposition="outside",
                textfont=dict(size=11, color=MUTED),
                cliponaxis=False,
                hovertemplate=(
                    f"%{{y}}<br>{CONTROLLER_NAMES[variant]}: %{{x:.2f}} {unit}"
                    "<extra></extra>"
                ),
            )
        )
    if is_temperature:
        low, high = min(all_values), max(all_values)
        pad = max(0.5, 0.08 * (high - low))
        figure.update_xaxes(range=[low - pad, high + 3 * pad])
    else:
        figure.update_xaxes(range=[0, max(all_values + [1.0]) * 1.18])
    figure.update_xaxes(title_text=f"{label} ({unit})")
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        legend=dict(orientation="h", y=-0.2, x=0, yanchor="top"),
    )
    figure = _finish(figure, f"{label} by scenario", height=height)
    figure.update_layout(margin=dict(l=10, r=30, t=52, b=70))
    return figure


def controller_scorecard(evaluation: FinalEvaluation) -> pd.DataFrame:
    """Six-scenario means (and violation totals) for each controller."""
    lookup = _controller_lookup(evaluation)
    rows: list[dict[str, object]] = []
    for variant in VARIANT_IDS:
        records = [lookup[(scenario, variant)] for scenario in SCENARIO_IDS]
        rows.append(
            {
                "Controller": CONTROLLER_NAMES[variant],
                "Mean peak (°C)": float(
                    np.mean([item.peak_temperature_k for item in records])
                )
                - KELVIN_OFFSET,
                "Worst peak (°C)": float(
                    np.max([item.peak_temperature_k for item in records])
                )
                - KELVIN_OFFSET,
                "Mean spread (K)": float(
                    np.mean([item.maximum_temperature_spread_k for item in records])
                ),
                "Mean pump energy (J)": float(
                    np.mean([item.pump_energy_j for item in records])
                ),
                "Time > limit (s)": float(
                    np.sum([item.thermal_violation_duration_s for item in records])
                ),
            }
        )
    return pd.DataFrame(rows)


def model_error_figure(evaluation: FinalEvaluation, *, height: int = 380) -> go.Figure:
    """MAE from one step to 120 s, in and out of distribution, both twins."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("In-distribution", "Out-of-distribution"),
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    labels = ["1 s", "30 s", "60 s", "120 s"]
    for column, distribution in enumerate(
        ("in_distribution", "out_of_distribution"), start=1
    ):
        for model in ("data_only", "physics_informed"):
            one_step = next(
                item.mean_absolute_error_k
                for item in evaluation.model_metrics
                if item.model_variant == model and item.distribution == distribution
            )
            horizons = {
                item.horizon_s: item.mae_kelvin
                for item in evaluation.forecast_metrics
                if item.model_variant == model and item.distribution == distribution
            }
            values = [one_step] + [
                horizons.get(h, math.nan) for h in (30.0, 60.0, 120.0)
            ]
            figure.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=MODEL_NAMES[model],
                    marker_color=MODEL_COLORS[model],
                    showlegend=column == 1,
                    text=[
                        f"{value:.2f}" if value < 10 else f"{value:.0f}"
                        for value in values
                    ],
                    textposition="outside",
                    textfont=dict(size=11, color=MUTED),
                    cliponaxis=False,
                    hovertemplate=(
                        f"{MODEL_NAMES[model]}<br>Horizon %{{x}}<br>"
                        "MAE %{y:.3f} K<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_yaxes(type="log", dtick=1, range=[-0.7, 2.75])
    figure.update_yaxes(title_text="Mean absolute error (K, log scale)", col=1)
    figure.update_xaxes(title_text="Forecast horizon")
    figure.update_layout(
        barmode="group",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right", yanchor="bottom"),
    )
    figure = _finish(figure, "Twin forecast error grows with horizon", height=height)
    figure.update_layout(margin=dict(l=10, r=10, t=90, b=10))
    return figure


def residual_figure(evaluation: FinalEvaluation, *, height: int = 320) -> go.Figure:
    """Mean absolute physics residual (energy-balance violation) per twin."""
    distributions = ("in_distribution", "out_of_distribution")
    figure = go.Figure()
    for model in ("data_only", "physics_informed"):
        values = [
            next(
                item.mean_absolute_physics_residual_w
                for item in evaluation.model_metrics
                if item.model_variant == model and item.distribution == distribution
            )
            for distribution in distributions
        ]
        figure.add_trace(
            go.Bar(
                x=["In-distribution", "Out-of-distribution"],
                y=values,
                name=MODEL_NAMES[model],
                marker_color=MODEL_COLORS[model],
                text=[f"{value:,.0f} W" for value in values],
                textposition="outside",
                textfont=dict(size=11, color=MUTED),
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:,.1f} W<extra></extra>",
            )
        )
    figure.update_yaxes(title_text="Mean |residual| (W)", rangemode="tozero")
    figure.update_layout(
        barmode="group",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
    )
    figure = _finish(
        figure, "Physics residual: lower is more consistent", height=height
    )
    figure.update_layout(margin=dict(l=10, r=10, t=80, b=10))
    return figure


def anomaly_score_figure(
    trace: FaultTrace,
    *,
    suspected_cell: int | None,
    score_threshold: float,
    height: int = 340,
) -> go.Figure:
    """Rolling per-cell anomaly scores with the suspected cell emphasised."""
    scores = np.asarray(trace.scores, dtype=float)
    figure = go.Figure()
    for cell in range(scores.shape[1]):
        emphasised = cell == suspected_cell
        values = np.where(scores[:, cell] > 0, scores[:, cell], np.nan)
        figure.add_trace(
            go.Scatter(
                x=trace.time_s,
                y=values,
                mode="lines",
                name=f"C{cell + 1}" + (" (suspected)" if emphasised else ""),
                showlegend=emphasised,
                line=dict(
                    color=DANGER if emphasised else "rgba(166,184,195,.30)",
                    width=3 if emphasised else 1,
                ),
                hovertemplate=f"C{cell + 1}: score %{{y:.1f}}<extra></extra>",
            )
        )
    figure.add_hline(
        y=score_threshold,
        line=dict(color=WARNING, width=2, dash="dot"),
        annotation_text=f"threshold {score_threshold:g}",
        annotation_position="bottom right",
        annotation_font_color=WARNING,
    )
    figure.update_yaxes(type="log", dtick=1, title_text="Anomaly score (log scale)")
    figure.update_xaxes(title_text="Time (s)")
    figure.update_layout(
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom")
    )
    figure = _finish(figure, "Detector score per cell", height=height)
    figure.update_layout(margin=dict(l=10, r=10, t=70, b=10))
    return figure


def fault_story_figure(
    trace: FaultTrace, summary: FaultSummary, *, height: int = 360
) -> go.Figure:
    """Faulted cell vs the rest of the pack, with fault/detection/limit markers."""
    cell = summary.fault_cell_index
    measured = np.asarray(trace.measured_temperature_k, dtype=float) - KELVIN_OFFSET
    others = np.delete(measured, cell, axis=1).mean(axis=1)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=measured[:, cell],
            mode="lines",
            name=f"C{cell + 1} measured",
            line=dict(color=DANGER, width=3),
            hovertemplate=f"C{cell + 1}: %{{y:.2f}} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=np.asarray(trace.predicted_temperature_k, dtype=float)[:, cell]
            - KELVIN_OFFSET,
            mode="lines",
            name=f"C{cell + 1} twin one-step expectation",
            line=dict(color=FORECAST, width=2, dash="dash"),
            hovertemplate="Twin: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=others,
            mode="lines",
            name="Mean of other cells",
            line=dict(color=COOLANT, width=2),
            hovertemplate="Others: %{y:.2f} °C<extra></extra>",
        )
    )
    figure.add_hline(
        y=summary.thermal_limit_k - KELVIN_OFFSET,
        line=dict(color=WARNING, width=2, dash="dot"),
        annotation_text=f"limit {summary.thermal_limit_k - KELVIN_OFFSET:.0f} °C",
        annotation_position="bottom right",
        annotation_font_color=WARNING,
    )
    markers = (
        ("fault injected", summary.fault_start_time_s, NEUTRAL, 1.0),
        ("detected", summary.detection_time_s, FORECAST, 0.88),
        ("limit crossed", summary.threshold_crossing_time_s, WARNING, 0.76),
    )
    for label, moment, color, y in markers:
        if moment is None:
            continue
        figure.add_vline(x=moment, line=dict(color=color, width=1.5, dash="dash"))
        figure.add_annotation(
            x=moment,
            y=y,
            yref="paper",
            text=f"{label} · {moment:.0f} s",
            showarrow=False,
            xanchor="left",
            xshift=4,
            font=dict(size=11, color=color),
            bgcolor="rgba(7,16,24,.8)",
        )
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Temperature (°C)")
    figure.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22, x=0, yanchor="top"),
    )
    figure = _finish(
        figure, f"C{cell + 1}: abnormal heating vs the pack", height=height
    )
    figure.update_layout(margin=dict(l=10, r=10, t=52, b=10))
    return figure


__all__ = [
    "CONTROLLER_METRICS",
    "CONTROLLER_NAMES",
    "SCENARIO_NAMES",
    "add_heat_flow_arrows",
    "anomaly_score_figure",
    "controller_scorecard",
    "coolant_history",
    "fault_story_figure",
    "model_error_figure",
    "pack_grid",
    "pack_map",
    "residual_figure",
    "scenario_controller_figure",
    "shared_range_c",
    "temperature_history",
    "to_celsius",
]
