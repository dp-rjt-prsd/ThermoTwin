from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test_dashboard_data import make_artifacts
from thermotwin.physics.topology import rectangular_grid_edges
from thermotwin.visualization.data import load_dashboard_data, load_evaluation
from thermotwin.visualization.judge_figures import (
    CONTROLLER_METRICS,
    add_heat_flow_arrows,
    anomaly_score_figure,
    controller_scorecard,
    fault_story_figure,
    model_error_figure,
    pack_grid,
    pack_map,
    scenario_controller_figure,
    shared_range_c,
    temperature_history,
)

EVALUATION = (
    Path(__file__).resolve().parents[1] / "results" / "phase12-final-evaluation"
)


def test_shared_range_covers_all_inputs_in_celsius_with_minimum_span() -> None:
    low, high = shared_range_c([300.0] * 12, [330.4] * 12)
    assert low <= 300.0 - 273.15 and high >= 330.4 - 273.15
    assert low == float(int(low)) and high == float(int(high))

    flat_low, flat_high = shared_range_c([299.15] * 12)
    assert flat_high - flat_low >= 2.0


def test_pack_map_uses_caller_range_and_celsius_labels() -> None:
    temperatures = np.linspace(298.15, 309.15, 12)
    figure = pack_map(temperatures, title="Pack", range_c=(20.0, 40.0), highlight=5)

    heatmap = figure.data[0]
    assert (heatmap.zmin, heatmap.zmax) == (20.0, 40.0)
    np.testing.assert_allclose(np.asarray(heatmap.z).reshape(-1), temperatures - 273.15)
    labels = [item.text for item in figure.layout.annotations if "C" in item.text]
    assert len(labels) == 12
    assert "<b>25.0</b>" in labels[0]
    assert "K)" in heatmap.customdata[0][0]
    assert any(shape.line.width == 3 for shape in figure.layout.shapes)


def test_pack_grid_rejects_bad_shapes_and_ranges() -> None:
    with pytest.raises(ValueError):
        pack_grid(
            [1.0] * 11, title="x", value_range=(0, 1), unit="W", colorbar_title="W"
        )
    with pytest.raises(ValueError):
        pack_grid(
            [1.0] * 12, title="x", value_range=(1, 1), unit="W", colorbar_title="W"
        )


def test_heat_flow_arrows_point_from_hot_to_cold_with_fourier_magnitude() -> None:
    temperatures = np.full(12, 300.0)
    temperatures[5] = 303.0
    figure = pack_map(temperatures, title="Pack", range_c=(25.0, 31.0))

    flows = add_heat_flow_arrows(
        figure, temperatures, rectangular_grid_edges(3, 4), 1.5
    )

    assert {hot for hot, _, _ in flows} == {5}
    assert {cold for _, cold, _ in flows} == {1, 4, 6, 9}
    assert all(watts == pytest.approx(2.0) for _, _, watts in flows)


def test_history_and_fault_figures_render_from_saved_artifacts(tmp_path: Path) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    trace = data.controller_traces["threshold"]

    history = temperature_history(
        trace.time_s,
        trace.cell_temperatures_k,
        thermal_limit_k=trace.thermal_limit_k,
        title="History",
        cursor_s=float(trace.time_s[0]),
    )
    assert history.layout.yaxis.title.text == "Temperature (°C)"
    tmax = next(item for item in history.data if item.name == "Tmax")
    np.testing.assert_allclose(tmax.y, trace.cell_temperatures_k.max(axis=1) - 273.15)

    story = fault_story_figure(data.fault_trace, data.fault_summary)
    assert "runaway" not in story.layout.title.text.lower()
    scores = anomaly_score_figure(
        data.fault_trace,
        suspected_cell=data.fault_summary.fault_cell_index,
        score_threshold=3.0,
    )
    assert scores.layout.yaxis.type == "log"
    assert [item.name for item in scores.data if item.showlegend] == [
        f"C{data.fault_summary.fault_cell_index + 1} (suspected)"
    ]


@pytest.mark.skipif(
    not (EVALUATION / "evaluation" / "final_evaluation.json").is_file(),
    reason="saved final evaluation artifact is not available locally",
)
def test_benchmark_figures_cover_every_scenario_and_controller() -> None:
    evaluation = load_evaluation(EVALUATION)

    for metric in CONTROLLER_METRICS:
        figure = scenario_controller_figure(evaluation, metric)
        assert len(figure.data) == 4
        assert all(len(trace.x) == 6 for trace in figure.data)
    with pytest.raises(ValueError):
        scenario_controller_figure(evaluation, "not_a_metric")

    table = controller_scorecard(evaluation)
    assert len(table) == 4
    peaks = [
        record.metrics.peak_temperature_k
        for record in evaluation.controller_metrics
        if record.variant_id == "threshold"
    ]
    assert table.loc[0, "Mean peak (°C)"] == pytest.approx(np.mean(peaks) - 273.15)

    errors = model_error_figure(evaluation)
    assert len(errors.data) == 4
    assert errors.layout.yaxis.type == "log"
