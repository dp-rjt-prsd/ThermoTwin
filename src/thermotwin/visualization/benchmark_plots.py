"""Accessible, deterministic figures from validated Phase 12 summaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]

from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ControllerMetrics,
    EvaluationDistribution,
    ModelEvaluationMetrics,
    ModelVariant,
    ScenarioId,
    VariantId,
)
from thermotwin.experiments.final_evaluation import (
    FinalEvaluation,
    FinalForecastMetrics,
)

_VARIANT_NAMES: dict[VariantId, str] = {
    "threshold": "Threshold",
    "pid": "PID",
    "predictive_data_only": "Predictive (data-only)",
    "predictive_physics_informed": "Predictive (physics-informed)",
}
_MODEL_ORDER: tuple[ModelVariant, ...] = ("data_only", "physics_informed")
_MODEL_NAMES: dict[ModelVariant, str] = {
    "data_only": "Data-only",
    "physics_informed": "Physics-informed",
}
_DISTRIBUTIONS: tuple[EvaluationDistribution, ...] = (
    "in_distribution",
    "out_of_distribution",
)
_DISTRIBUTION_NAMES: dict[EvaluationDistribution, str] = {
    "in_distribution": "In-distribution",
    "out_of_distribution": "Out-of-distribution",
}
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
_PATTERNS = ("", "/", "x", ".")
_SYMBOLS = ("circle", "square", "diamond", "triangle-up")
_SCENARIO_NAMES: dict[ScenarioId, str] = {
    "normal": "Normal",
    "high_current": "High current",
    "hot_ambient": "Hot ambient",
    "weak_cooling": "Weak cooling",
    "fast_charging": "Fast charging",
    "resistance_fault": "Resistance fault",
}
_CONTROLLER_METRICS: tuple[tuple[str, str, str], ...] = (
    ("peak_temperature_k", "Peak temperature", "K"),
    ("mean_temperature_k", "Mean temperature", "K"),
    ("maximum_temperature_spread_k", "Temperature spread", "K"),
    ("pump_energy_j", "Pump energy", "J"),
    ("thermal_violation_count", "Thermal violations", "count"),
    ("thermal_violation_duration_s", "Violation duration", "s"),
)
_MODEL_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mean_absolute_error_k", "Mean absolute error", "K"),
    ("root_mean_square_error_k", "Root mean square error", "K"),
    ("maximum_cell_error_k", "Maximum cell error", "K"),
)
_FORECAST_METRICS: tuple[tuple[str, str], ...] = (
    ("mae_kelvin", "Mean absolute error (K)"),
    ("rmse_kelvin", "Root mean square error (K)"),
    ("maximum_absolute_cell_error_kelvin", "Maximum cell error (K)"),
)
_RESIDUAL_METRICS: tuple[tuple[str, str, str], ...] = (
    ("mean_absolute_physics_residual_w", "Mean absolute residual", "W"),
    ("root_mean_square_physics_residual_w", "Root mean square residual", "W"),
    ("maximum_absolute_physics_residual_w", "Maximum absolute residual", "W"),
)


def _require_evaluation(evaluation: object) -> FinalEvaluation:
    if type(evaluation) is not FinalEvaluation:
        raise TypeError("evaluation must be exactly a FinalEvaluation")
    return evaluation


def _finish(figure: go.Figure, evaluation: FinalEvaluation, title: str) -> go.Figure:
    status = evaluation.evidence_status
    scenarios = ", ".join(_SCENARIO_NAMES[item] for item in SCENARIO_IDS)
    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=13, color="#17212b"),
        margin=dict(l=75, r=35, t=85, b=125),
        hoverlabel=dict(font_size=13),
        legend=dict(title_text="Identity (labels and patterns)"),
        annotations=[
            *list(figure.layout.annotations or ()),
            go.layout.Annotation(
                text=(
                    f"Evidence status: {status} · Synthetic, descriptive "
                    "configured-run evidence.<br>"
                    f"Scenario set: {scenarios}."
                ),
                x=0.0,
                y=-0.19,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                align="left",
                font=dict(size=11, color="#263746"),
            ),
        ],
    )
    return figure


def _controller_lookup(
    evaluation: FinalEvaluation,
) -> dict[tuple[ScenarioId, VariantId], ControllerMetrics]:
    return {
        (record.scenario_id, record.variant_id): record.metrics
        for record in evaluation.controller_metrics
    }


def controller_matrix_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Show the saved six-scenario by four-controller comparison matrix."""
    evaluation = _require_evaluation(evaluation)
    metrics = _controller_lookup(evaluation)
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=tuple(
            f"{label} ({unit})" for _, label, unit in _CONTROLLER_METRICS
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )
    for metric_index, (field, label, unit) in enumerate(_CONTROLLER_METRICS):
        row, column = divmod(metric_index, 2)
        for variant_index, variant in enumerate(VARIANT_IDS):
            values = [
                getattr(metrics[(scenario, variant)], field)
                for scenario in SCENARIO_IDS
            ]
            figure.add_trace(
                go.Bar(
                    x=[_SCENARIO_NAMES[scenario] for scenario in SCENARIO_IDS],
                    y=values,
                    name=_VARIANT_NAMES[variant],
                    legendgroup=variant,
                    showlegend=metric_index == 0,
                    marker=dict(
                        color=_COLORS[variant_index],
                        pattern=dict(shape=_PATTERNS[variant_index]),
                        line=dict(color="#263746", width=0.6),
                    ),
                    hovertemplate=(
                        f"Scenario: %{{x}}<br>Controller: {_VARIANT_NAMES[variant]}<br>"
                        f"{label}: %{{y}} {unit}<extra></extra>"
                    ),
                ),
                row=row + 1,
                col=column + 1,
            )
        figure.update_xaxes(title_text="Scenario", row=row + 1, col=column + 1)
        figure.update_yaxes(title_text=unit, row=row + 1, col=column + 1)
    figure.update_layout(barmode="group", height=1000)
    return _finish(figure, evaluation, "Controller comparison across saved scenarios")


def _model_metric_figure(
    evaluation: FinalEvaluation,
    metrics: tuple[tuple[str, str, str], ...],
    *,
    title: str,
    value_getter: Callable[[ModelEvaluationMetrics, str], float],
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=tuple(f"{label} ({unit})" for _, label, unit in metrics),
        horizontal_spacing=0.12,
    )
    model_lookup = {
        (item.model_variant, item.distribution): item
        for item in evaluation.model_metrics
    }
    for metric_index, (field, label, unit) in enumerate(metrics):
        for model_index, model in enumerate(_MODEL_ORDER):
            values = [
                value_getter(model_lookup[(model, distribution)], field)
                for distribution in _DISTRIBUTIONS
            ]
            figure.add_trace(
                go.Bar(
                    x=[_DISTRIBUTION_NAMES[item] for item in _DISTRIBUTIONS],
                    y=values,
                    name=_MODEL_NAMES[model],
                    legendgroup=model,
                    showlegend=metric_index == 0,
                    marker=dict(
                        color=_COLORS[model_index],
                        pattern=dict(shape=_PATTERNS[model_index]),
                        line=dict(color="#263746", width=0.6),
                    ),
                    hovertemplate=(
                        f"Distribution: %{{x}}<br>Model: {_MODEL_NAMES[model]}<br>"
                        f"{label}: %{{y}} {unit}<extra></extra>"
                    ),
                ),
                row=1,
                col=metric_index + 1,
            )
        figure.update_xaxes(
            title_text="Evaluation distribution", row=1, col=metric_index + 1
        )
        figure.update_yaxes(title_text=unit, row=1, col=metric_index + 1)
    figure.update_layout(barmode="group", height=500)
    return _finish(figure, evaluation, title)


def model_accuracy_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Compare saved one-step ID/OOD model error summaries."""
    evaluation = _require_evaluation(evaluation)

    def value(record: ModelEvaluationMetrics, field: str) -> float:
        return cast(float, getattr(record, field))

    return _model_metric_figure(
        evaluation,
        _MODEL_METRICS,
        title="One-step model error by evaluation distribution",
        value_getter=value,
    )


def forecast_error_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Compare saved ID/OOD rollout errors at 30, 60, and 120 seconds."""
    evaluation = _require_evaluation(evaluation)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=tuple(label for _, label in _FORECAST_METRICS),
        horizontal_spacing=0.12,
    )
    identities = tuple(
        (model, distribution)
        for model in _MODEL_ORDER
        for distribution in _DISTRIBUTIONS
    )
    for metric_index, (field, label) in enumerate(_FORECAST_METRICS):
        for identity_index, (model, distribution) in enumerate(identities):
            records: tuple[FinalForecastMetrics, ...] = tuple(
                sorted(
                    (
                        item
                        for item in evaluation.forecast_metrics
                        if item.model_variant == model
                        and item.distribution == distribution
                    ),
                    key=lambda item: item.horizon_s,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[item.horizon_s for item in records],
                    y=[getattr(item, field) for item in records],
                    mode="lines+markers",
                    name=(
                        f"{_MODEL_NAMES[model]} · "
                        f"{_DISTRIBUTION_NAMES[distribution]}"
                    ),
                    legendgroup=f"{model}-{distribution}",
                    showlegend=metric_index == 0,
                    line=dict(
                        color=_COLORS[identity_index],
                        dash="solid" if model == "data_only" else "dash",
                        width=2.2,
                    ),
                    marker=dict(
                        symbol="circle" if model == "data_only" else "square",
                        size=8,
                        line=dict(color="#263746", width=0.8),
                    ),
                    hovertemplate=(
                        "Model and distribution: %{fullData.name}<br>"
                        "Forecast horizon: %{x} s<br>"
                        f"{label}: %{{y}}<extra></extra>"
                    ),
                ),
                row=1,
                col=metric_index + 1,
            )
        figure.update_xaxes(
            title_text="Forecast horizon (s)", row=1, col=metric_index + 1
        )
        figure.update_yaxes(
            title_text="Temperature error (K)", row=1, col=metric_index + 1
        )
    figure.update_layout(height=500)
    return _finish(figure, evaluation, "Saved multi-step forecast errors")


def physics_residual_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Compare saved mean, RMS, and maximum physics residuals by model."""
    evaluation = _require_evaluation(evaluation)

    def value(record: ModelEvaluationMetrics, field: str) -> float:
        return cast(float, getattr(record, field))

    return _model_metric_figure(
        evaluation,
        _RESIDUAL_METRICS,
        title="Saved physics residual by model and distribution",
        value_getter=value,
    )


def thermal_energy_tradeoff_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Plot saved peak-temperature and pump-energy pairs per scenario/controller."""
    evaluation = _require_evaluation(evaluation)
    metrics = _controller_lookup(evaluation)
    figure = go.Figure()
    for variant_index, variant in enumerate(VARIANT_IDS):
        records = [metrics[(scenario, variant)] for scenario in SCENARIO_IDS]
        figure.add_trace(
            go.Scatter(
                x=[item.pump_energy_j for item in records],
                y=[item.peak_temperature_k for item in records],
                customdata=[
                    [
                        _SCENARIO_NAMES[scenario],
                        item.mean_temperature_k,
                        item.maximum_temperature_spread_k,
                        item.thermal_violation_count,
                        item.thermal_violation_duration_s,
                    ]
                    for scenario, item in zip(SCENARIO_IDS, records, strict=True)
                ],
                mode="markers+text",
                text=[_SCENARIO_NAMES[scenario] for scenario in SCENARIO_IDS],
                textposition="top center",
                name=_VARIANT_NAMES[variant],
                marker=dict(
                    color=_COLORS[variant_index],
                    symbol=_SYMBOLS[variant_index],
                    size=12,
                    line=dict(color="#263746", width=1),
                ),
                hovertemplate=(
                    "Scenario: %{customdata[0]}<br>"
                    f"Controller: {_VARIANT_NAMES[variant]}<br>"
                    "Pump energy: %{x} J<br>Peak temperature: %{y} K<br>"
                    "Mean temperature: %{customdata[1]} K<br>"
                    "Temperature spread: %{customdata[2]} K<br>"
                    "Thermal violations: %{customdata[3]}<br>"
                    "Violation duration: %{customdata[4]} s<extra></extra>"
                ),
            )
        )
    figure.update_xaxes(title_text="Pump energy (J)")
    figure.update_yaxes(title_text="Peak temperature (K)")
    return _finish(figure, evaluation, "Scenario thermal and cooling-energy tradeoff")


def _optional_time(value: float | None) -> str:
    return "Unavailable" if value is None else f"{value} s"


def fault_timing_figure(evaluation: FinalEvaluation) -> go.Figure:
    """Show observed fault timing and preserve missing values as unavailable."""
    evaluation = _require_evaluation(evaluation)
    fault = evaluation.fault_detection
    figure = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Observed fault field", "Saved result"],
                    fill_color="#e8eef2",
                    line_color="#263746",
                    align="left",
                ),
                cells=dict(
                    values=[
                        [
                            "Suspected cell",
                            "Detection time",
                            "Thermal-limit crossing time",
                            "Detection lead time",
                        ],
                        [
                            (
                                "Unavailable"
                                if fault.suspected_cell_index is None
                                else f"C{fault.suspected_cell_index + 1}"
                            ),
                            _optional_time(fault.detection_time_s),
                            _optional_time(fault.threshold_crossing_time_s),
                            _optional_time(fault.lead_time_s),
                        ],
                    ],
                    fill_color=[["#ffffff", "#f4f7f9"]],
                    line_color="#aab7c2",
                    align="left",
                    height=34,
                ),
                columnwidth=[2, 1],
                name="Observed detector result",
            )
        ]
    )
    figure.update_layout(height=360)
    return _finish(
        figure,
        evaluation,
        "Observed fault detection timing (aggregate model identity unavailable)",
    )


__all__ = [
    "controller_matrix_figure",
    "fault_timing_figure",
    "forecast_error_figure",
    "model_accuracy_figure",
    "physics_residual_figure",
    "thermal_energy_tradeoff_figure",
]
