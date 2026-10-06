from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

from thermotwin.visualization.contracts import (
    ControllerTrace,
    FaultSummary,
    FaultTrace,
    ForecastSnapshot,
)
from thermotwin.visualization.plots import (
    controller_comparison_figure,
    coolant_flow_figure,
    fault_diagnostics_figure,
    forecast_comparison_figure,
    pack_heatmap_figure,
    temperature_timeseries_figure,
)

METRICS = {
    "peak_temperature_k": 310.0,
    "mean_temperature_k": 305.0,
    "maximum_temperature_spread_k": 5.0,
    "pump_energy_j": 12.0,
    "thermal_violation_count": 2,
    "time_above_thermal_limit_s": 4.0,
}
CONTROLLER_ORDER = (
    "threshold",
    "pid",
    "predictive_data_only",
    "predictive_physics_informed",
)


def _controller(label: str = "threshold") -> ControllerTrace:
    time = np.array([0.0, 1.0, 2.0])
    temperatures = np.arange(36, dtype=float).reshape(3, 12) + 300.0
    return ControllerTrace(
        label=label,
        time_s=time,
        current_a=np.ones(3),
        soc_fraction=np.full(3, 0.5),
        ambient_temperature_k=np.full(3, 295.0),
        coolant_inlet_temperature_k=np.full(3, 290.0),
        coolant_mass_flow_kg_s=np.array([0.01, 0.02, 0.03]),
        pump_power_w=np.array([1.0, 2.0, 3.0]),
        cell_temperatures_k=temperatures,
        cell_resistances_ohm=np.full((3, 12), 0.01),
        metrics=METRICS,
    )


def _forecasts() -> Mapping[str, tuple[ForecastSnapshot, ...]]:
    result: dict[str, tuple[ForecastSnapshot, ...]] = {}
    for variant, offset in (("data_only", 0.0), ("physics_informed", 1.0)):
        result[variant] = tuple(
            ForecastSnapshot(
                variant=variant,
                horizon_s=float(horizon),
                source_time_s=5.0,
                source_cell_temperatures_k=np.full(12, 300.0),
                predicted_cell_temperatures_k=np.full(
                    12, 300.0 + horizon / 100 + offset
                ),
                action_coolant_mass_flow_kg_s=0.02,
            )
            for horizon in (30, 60, 120)
        )
    return result


def _fault_inputs() -> tuple[FaultTrace, FaultSummary]:
    temperatures = np.full((3, 12), 300.0)
    temperatures[:, 4] = [301.0, 315.0, 321.0]
    prediction = np.full((3, 12), 300.0)
    trace = FaultTrace(
        time_s=np.array([0.0, 5.0, 10.0]),
        measured_temperature_k=temperatures,
        predicted_temperature_k=prediction,
        residual_k=temperatures - prediction,
        scores=np.zeros((3, 12)),
        coolant_mass_flow_kg_s=np.full(3, 0.02),
        pump_power_w=np.ones(3),
    )
    summary = FaultSummary(
        model_variant="physics_informed",
        fault_cell_index=4,
        fault_start_time_s=2.0,
        thermal_limit_k=320.0,
        threshold_crossing_time_s=10.0,
        detection_time_s=7.0,
        lead_time_s=3.0,
        detected_fault_cell=True,
        peak_temperature_k=321.0,
        temperature_spread_k=21.0,
        pump_energy_j=3.0,
        thermal_violation_count=1,
        anomaly_event_count=0,
        events=(),
    )
    return trace, summary


def test_pack_heatmap_maps_canonical_grid_and_preserves_input() -> None:
    temperatures = np.arange(12, dtype=float) + 290.0
    original = temperatures.copy()

    figure = pack_heatmap_figure(temperatures, title="Pack temperatures")

    assert isinstance(figure, Figure)
    heatmap = figure.data[0]
    assert heatmap.type == "heatmap"
    assert np.asarray(heatmap.z).shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(heatmap.z), temperatures.reshape(3, 4))
    assert [label.split("<br>")[0] for row in heatmap.text for label in row] == [
        f"C{index}" for index in range(1, 13)
    ]
    assert heatmap.colorbar.title.text == "Temperature (K)"
    assert "Kelvin" in heatmap.hovertemplate
    np.testing.assert_array_equal(temperatures, original)


def test_pack_heatmap_rejects_invalid_temperature_shapes_and_values() -> None:
    with pytest.raises(ValueError, match="12"):
        pack_heatmap_figure(np.ones((3, 4)), title="bad shape")
    with pytest.raises(ValueError, match="finite"):
        pack_heatmap_figure([300.0] * 11 + [float("nan")], title="bad value")
    with pytest.raises(ValueError, match="positive"):
        pack_heatmap_figure([300.0] * 11 + [0.0], title="bad value")


def test_temperature_and_coolant_figures_show_units_and_saved_data() -> None:
    trace = _controller()
    temperature_figure = temperature_timeseries_figure(
        trace, thermal_limit_k=320.0, title="Temperature history"
    )
    names = [item.name for item in temperature_figure.data]
    assert len([name for name in names if name.startswith("C")]) == 12
    assert {"Measured Tmax", "Measured mean", "Thermal limit"}.issubset(names)
    assert temperature_figure.layout.xaxis.title.text == "Time (s)"
    assert temperature_figure.layout.yaxis.title.text == "Temperature (K)"
    assert all("K" in (item.hovertemplate or "") for item in temperature_figure.data)

    coolant_figure = coolant_flow_figure(trace, title="Saved coolant flow")
    coolant = coolant_figure.data[0]
    assert list(coolant.x) == [0.0, 1.0, 2.0]
    assert list(coolant.y) == [0.01, 0.02, 0.03]
    assert coolant_figure.layout.xaxis.title.text == "Time (s)"
    assert coolant_figure.layout.yaxis.title.text == "Coolant mass flow (kg/s)"
    assert "kg/s" in coolant.hovertemplate


def test_controller_comparison_covers_saved_metrics_in_stable_order() -> None:
    traces = {key: _controller(key) for key in reversed(CONTROLLER_ORDER)}
    figure = controller_comparison_figure(traces)

    assert len(figure.data) == 24
    assert [figure.data[index].name for index in range(4)] == [
        "Threshold",
        "PID",
        "Predictive (data-only)",
        "Predictive (physics-informed)",
    ]
    titles = [annotation.text for annotation in figure.layout.annotations]
    assert titles == [
        "Peak temperature (K)",
        "Mean temperature (K)",
        "Maximum temperature spread (K)",
        "Pump energy (J)",
        "Thermal violations (count)",
        "Time above thermal limit (s)",
    ]


def test_forecast_comparison_has_two_variants_and_three_horizons() -> None:
    figure = forecast_comparison_figure(_forecasts())
    assert len(figure.data) == 4
    assert {item.name for item in figure.data} == {
        "Data-only peak",
        "Data-only spread",
        "Physics-informed peak",
        "Physics-informed spread",
    }
    for item in figure.data:
        assert list(item.x) == [30.0, 60.0, 120.0]
        assert "K" in item.hovertemplate
    peak_traces = {
        item.name: list(item.y) for item in figure.data if "peak" in item.name
    }
    assert peak_traces["Data-only peak"] == pytest.approx([300.3, 300.6, 301.2])
    assert peak_traces["Physics-informed peak"] == pytest.approx([301.3, 301.6, 302.2])
    assert figure.layout.xaxis.title.text == "Forecast horizon (s)"


def test_fault_diagnostics_has_target_cell_lines_limit_and_event_markers() -> None:
    trace, summary = _fault_inputs()
    figure = fault_diagnostics_figure(trace, summary)
    names = {item.name for item in figure.data}
    assert {
        "C5 measured",
        "C5 predicted",
        "Thermal limit",
    }.issubset(names)
    positions = {
        shape.name: shape.x0 for shape in figure.layout.shapes if shape.type == "line"
    }
    assert positions == {
        "Fault start": 2.0,
        "Detection": 7.0,
        "Thermal-limit crossing": 10.0,
    }
    assert figure.layout.title.text == "Fault-induced thermal anomaly diagnostics - C5"
    assert "runaway" not in figure.layout.title.text.lower()


def test_plot_functions_reject_wrong_dto_types_and_incomplete_comparisons() -> None:
    with pytest.raises(TypeError, match="ControllerTrace"):
        coolant_flow_figure(object(), title="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="four"):
        controller_comparison_figure({"threshold": _controller()})
    with pytest.raises(ValueError, match="variants"):
        forecast_comparison_figure({"data_only": _forecasts()["data_only"]})
    with pytest.raises(TypeError, match="FaultTrace"):
        fault_diagnostics_figure(object(), _fault_inputs()[1])  # type: ignore[arg-type]
