"""Reusable Plotly figures built only from validated dashboard artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from numpy.typing import NDArray
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from thermotwin.visualization.contracts import (
    ControllerTrace,
    FaultSummary,
    FaultTrace,
    ForecastSnapshot,
)

_CONTROLLER_ORDER = (
    "threshold",
    "pid",
    "predictive_data_only",
    "predictive_physics_informed",
)
_CONTROLLER_NAMES = {
    "threshold": "Threshold",
    "pid": "PID",
    "predictive_data_only": "Predictive (data-only)",
    "predictive_physics_informed": "Predictive (physics-informed)",
}
_MODEL_ORDER = ("data_only", "physics_informed")
_MODEL_NAMES = {"data_only": "Data-only", "physics_informed": "Physics-informed"}
_HORIZONS = (30.0, 60.0, 120.0)
_METRICS = (
    ("peak_temperature_k", "Peak temperature (K)"),
    ("mean_temperature_k", "Mean temperature (K)"),
    ("maximum_temperature_spread_k", "Maximum temperature spread (K)"),
    ("pump_energy_j", "Pump energy (J)"),
    ("thermal_violation_count", "Thermal violations (count)"),
    ("time_above_thermal_limit_s", "Time above thermal limit (s)"),
)
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _require_type(value: object, expected: type[object], name: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}")


def _require_title(title: str) -> None:
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a nonempty string")


def _layout(figure: go.Figure, title: str) -> go.Figure:
    _require_title(title)
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13, color="#17212b"),
        margin=dict(l=65, r=35, t=75, b=65),
        hoverlabel=dict(font_size=13),
    )
    return figure


def pack_heatmap_figure(
    temperatures_k: Sequence[float] | NDArray[np.float64], *, title: str
) -> go.Figure:
    """Render the canonical C1..C12 row-major 3 by 4 pack map."""
    _require_title(title)
    if isinstance(temperatures_k, (str, bytes)):
        raise TypeError("temperatures_k must be a one-dimensional numeric sequence")
    values = np.asarray(temperatures_k)
    if values.shape != (12,):
        raise ValueError(
            "temperatures_k must contain exactly 12 values in one dimension"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("temperatures_k must contain numeric values")
    if not np.isfinite(values).all():
        raise ValueError("temperatures_k values must be finite")
    if np.any(values <= 0.0):
        raise ValueError("temperatures_k values must be positive kelvin values")
    grid = values.astype(float, copy=True).reshape(3, 4)
    labels = np.array([f"C{i}" for i in range(1, 13)], dtype=object).reshape(3, 4)
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=["Column 1", "Column 2", "Column 3", "Column 4"],
            y=["Row 1", "Row 2", "Row 3"],
            text=labels,
            texttemplate="%{text}<br>%{z:.1f} K",
            textfont=dict(size=13, color="#111111"),
            colorscale="Inferno",
            colorbar=dict(title=dict(text="Temperature (K)")),
            hovertemplate="%{text}<br>Temperature: %{z:.2f} Kelvin<extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    figure.update_xaxes(side="top", title_text="Pack column", ticks="")
    figure.update_yaxes(autorange="reversed", title_text="Pack row", ticks="")
    figure.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    return _layout(figure, title)


def temperature_timeseries_figure(
    trace: ControllerTrace, *, thermal_limit_k: float, title: str
) -> go.Figure:
    """Plot all saved cell temperatures and aggregate/limit references."""
    _require_type(trace, ControllerTrace, "trace")
    _require_title(title)
    if isinstance(thermal_limit_k, bool) or not isinstance(
        thermal_limit_k, (int, float)
    ):
        raise TypeError("thermal_limit_k must be a finite positive number")
    limit = float(thermal_limit_k)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("thermal_limit_k must be finite and positive")
    temperatures = trace.cell_temperatures_k
    figure = go.Figure()
    for cell_index in range(12):
        figure.add_trace(
            go.Scatter(
                x=trace.time_s,
                y=temperatures[:, cell_index],
                mode="lines",
                name=f"C{cell_index + 1}",
                line=dict(color=_COLORS[cell_index % len(_COLORS)], width=1.2),
                hovertemplate=(
                    f"Time: %{{x:.2f}} s<br>C{cell_index + 1}: "
                    "%{y:.2f} K<extra></extra>"
                ),
            )
        )
    maximum = np.max(temperatures, axis=1)
    mean = np.mean(temperatures, axis=1)
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=maximum,
            mode="lines",
            name="Measured Tmax",
            line=dict(color="#111111", width=3, dash="solid"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Measured Tmax: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=mean,
            mode="lines",
            name="Measured mean",
            line=dict(color="#5B6573", width=2.5, dash="dash"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Measured mean: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=np.full(trace.time_s.shape, limit),
            mode="lines",
            name="Thermal limit",
            line=dict(color="#B22222", width=2.5, dash="dot"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Thermal limit: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Temperature (K)")
    figure.update_layout(legend_title_text="Temperature traces")
    return _layout(figure, title)


def coolant_flow_figure(trace: ControllerTrace, *, title: str) -> go.Figure:
    """Plot saved coolant mass flow against its measured time axis."""
    _require_type(trace, ControllerTrace, "trace")
    _require_title(title)
    figure = go.Figure(
        go.Scatter(
            x=trace.time_s,
            y=trace.coolant_mass_flow_kg_s,
            mode="lines",
            name="Coolant mass flow",
            line=dict(color="#0072B2", width=2.5, dash="solid"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Coolant mass flow: %{y:.4f} kg/s<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Coolant mass flow (kg/s)", rangemode="tozero")
    return _layout(figure, title)


def controller_comparison_figure(
    traces: Mapping[str, ControllerTrace],
) -> go.Figure:
    """Compare the six persisted controller metrics in separately scaled panels."""
    if not isinstance(traces, Mapping):
        raise TypeError(
            "traces must be a mapping of controller keys to ControllerTrace"
        )
    if set(traces) != set(_CONTROLLER_ORDER):
        raise ValueError("traces must contain all four required controller keys")
    for key in _CONTROLLER_ORDER:
        _require_type(traces[key], ControllerTrace, f"traces[{key!r}]")
    rows, columns = 2, 3
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[label for _, label in _METRICS],
        horizontal_spacing=0.12,
        vertical_spacing=0.25,
    )
    for metric_index, (metric_key, metric_label) in enumerate(_METRICS):
        row, column = divmod(metric_index, columns)
        for controller_index, controller_key in enumerate(_CONTROLLER_ORDER):
            trace = traces[controller_key]
            if metric_key not in trace.metrics:
                raise ValueError(f"{controller_key} saved metrics lack {metric_key!r}")
            raw_value = trace.metrics[metric_key]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"{controller_key}.{metric_key} must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{controller_key}.{metric_key} must be finite")
            if value < 0.0:
                raise ValueError(f"{controller_key}.{metric_key} must be nonnegative")
            figure.add_trace(
                go.Bar(
                    x=[_CONTROLLER_NAMES[controller_key]],
                    y=[value],
                    name=_CONTROLLER_NAMES[controller_key],
                    legendgroup=controller_key,
                    showlegend=(metric_index == 0),
                    marker_color=_COLORS[controller_index],
                    hovertemplate=f"%{{x}}<br>{metric_label}: %{{y}}<extra></extra>",
                ),
                row=row + 1,
                col=column + 1,
            )
        figure.update_yaxes(title_text=metric_label, row=row + 1, col=column + 1)
    figure.update_layout(barmode="group", legend_title_text="Controller")
    return _layout(figure, "Saved controller metric comparison")


def forecast_comparison_figure(
    forecasts: Mapping[str, Sequence[ForecastSnapshot]],
) -> go.Figure:
    """Compare saved predicted peak temperature and spread by forecast horizon."""
    if not isinstance(forecasts, Mapping):
        raise TypeError("forecasts must map model variant names to snapshots")
    if set(forecasts) != set(_MODEL_ORDER):
        raise ValueError("forecasts must contain both required model variants")
    normalized: dict[str, tuple[ForecastSnapshot, ...]] = {}
    for variant in _MODEL_ORDER:
        raw_snapshots: object = forecasts[variant]
        if isinstance(raw_snapshots, (str, bytes)) or not isinstance(
            raw_snapshots, Sequence
        ):
            raise TypeError(f"forecasts[{variant!r}] must be a sequence of snapshots")
        snapshot_objects = cast(Sequence[object], raw_snapshots)
        items: list[ForecastSnapshot] = []
        for snapshot in snapshot_objects:
            _require_type(snapshot, ForecastSnapshot, "forecast snapshot")
            items.append(cast(ForecastSnapshot, snapshot))
        if len(items) != len(_HORIZONS):
            raise ValueError(f"forecasts[{variant!r}] must contain all three horizons")
        for snapshot, expected_horizon in zip(items, _HORIZONS, strict=True):
            if snapshot.variant != variant or snapshot.horizon_s != expected_horizon:
                raise ValueError(
                    f"{variant} snapshots must match variants and ordered "
                    "30/60/120-second horizons"
                )
        normalized[variant] = tuple(items)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Predicted peak temperature (K)",
            "Predicted temperature spread (K)",
        ),
        horizontal_spacing=0.14,
    )
    for variant_index, variant in enumerate(_MODEL_ORDER):
        snapshots = normalized[variant]
        horizons = [snapshot.horizon_s for snapshot in snapshots]
        peaks = [
            float(np.max(snapshot.predicted_cell_temperatures_k))
            for snapshot in snapshots
        ]
        spreads = [
            float(
                np.max(snapshot.predicted_cell_temperatures_k)
                - np.min(snapshot.predicted_cell_temperatures_k)
            )
            for snapshot in snapshots
        ]
        color = _COLORS[variant_index]
        for column, values, metric_name, dash in (
            (1, peaks, "peak", "solid"),
            (2, spreads, "spread", "dash"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=horizons,
                    y=values,
                    mode="lines+markers",
                    name=f"{_MODEL_NAMES[variant]} {metric_name}",
                    line=dict(color=color, width=2.5, dash=dash),
                    marker=dict(
                        symbol="circle" if variant_index == 0 else "square", size=8
                    ),
                    hovertemplate=(
                        "Horizon: %{x:.0f} s<br>"
                        f"{_MODEL_NAMES[variant]} predicted {metric_name}: "
                        "%{y:.2f} K<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.update_xaxes(title_text="Forecast horizon (s)", row=1, col=1)
    figure.update_xaxes(title_text="Forecast horizon (s)", row=1, col=2)
    figure.update_yaxes(title_text="Predicted peak temperature (K)", row=1, col=1)
    figure.update_yaxes(title_text="Predicted temperature spread (K)", row=1, col=2)
    figure.update_layout(legend_title_text="Model and statistic")
    return _layout(figure, "Saved forecast comparison")


def fault_diagnostics_figure(trace: FaultTrace, summary: FaultSummary) -> go.Figure:
    """Show a configured cell's measured fault trace and saved event times."""
    _require_type(trace, FaultTrace, "trace")
    _require_type(summary, FaultSummary, "summary")
    cell_index = summary.fault_cell_index
    if (
        isinstance(cell_index, bool)
        or not isinstance(cell_index, int)
        or not 0 <= cell_index < 12
    ):
        raise ValueError("summary.fault_cell_index must be an integer in [0, 11]")
    cell_name = f"C{cell_index + 1}"
    if (
        isinstance(summary.thermal_limit_k, bool)
        or not isinstance(summary.thermal_limit_k, (int, float))
        or not math.isfinite(float(summary.thermal_limit_k))
        or summary.thermal_limit_k <= 0.0
    ):
        raise ValueError("summary.thermal_limit_k must be finite and positive")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=trace.measured_temperature_k[:, cell_index],
            mode="lines+markers",
            name=f"{cell_name} measured",
            line=dict(color="#B22222", width=2.5, dash="solid"),
            marker=dict(symbol="circle", size=7),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Measured temperature: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=trace.predicted_temperature_k[:, cell_index],
            mode="lines+markers",
            name=f"{cell_name} predicted",
            line=dict(color="#0072B2", width=2.5, dash="dash"),
            marker=dict(symbol="square", size=6),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Predicted temperature: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=trace.time_s,
            y=np.full(trace.time_s.shape, summary.thermal_limit_k),
            mode="lines",
            name="Thermal limit",
            line=dict(color="#111111", width=2.5, dash="dot"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>Thermal limit: %{y:.2f} K<extra></extra>"
            ),
        )
    )
    markers = (
        ("Fault start", summary.fault_start_time_s, "#7A5195", "dash"),
        ("Detection", summary.detection_time_s, "#009E73", "dot"),
        (
            "Thermal-limit crossing",
            summary.threshold_crossing_time_s,
            "#D55E00",
            "dashdot",
        ),
    )
    for label, marker_time, color, dash in markers:
        if marker_time is not None:
            if (
                isinstance(marker_time, bool)
                or not isinstance(marker_time, (int, float))
                or not math.isfinite(float(marker_time))
                or float(marker_time) < 0.0
            ):
                raise ValueError(f"{label} time must be finite and nonnegative")
            figure.add_vline(
                x=float(marker_time),
                line_color=color,
                line_dash=dash,
                line_width=2,
                annotation_text=label,
                annotation_position="top",
                name=label,
            )
            figure.layout.shapes[-1].name = label
    figure.update_xaxes(title_text="Time (s)")
    figure.update_yaxes(title_text="Temperature (K)")
    figure.update_layout(legend_title_text=f"{cell_name} thermal diagnostics")
    return _layout(figure, f"Fault-induced thermal anomaly diagnostics - {cell_name}")


__all__ = [
    "controller_comparison_figure",
    "coolant_flow_figure",
    "fault_diagnostics_figure",
    "forecast_comparison_figure",
    "pack_heatmap_figure",
    "temperature_timeseries_figure",
]
