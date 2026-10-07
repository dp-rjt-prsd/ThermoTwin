from __future__ import annotations

import importlib.util
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from test_dashboard_data import make_artifacts
from thermotwin.visualization import dashboard as dashboard_module
from thermotwin.visualization.contracts import FaultEvent
from thermotwin.visualization.dashboard import (
    _fault_indicators,
    _temperature_history_figure,
)
from thermotwin.visualization.data import load_dashboard_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = PROJECT_ROOT / "src" / "thermotwin" / "visualization" / "dashboard.py"
LAUNCHER_PATH = PROJECT_ROOT / "scripts" / "run_dashboard.py"


def _run_dashboard(
    predictive: Path | str,
    fault: Path | str,
    monkeypatch: pytest.MonkeyPatch,
    evaluation: str = "",
) -> AppTest:
    monkeypatch.setenv("THERMOTWIN_PREDICTIVE_RESULTS", str(predictive))
    monkeypatch.setenv("THERMOTWIN_FAULT_RESULTS", str(fault))
    monkeypatch.setenv("THERMOTWIN_EVALUATION_RESULTS", evaluation)
    return AppTest.from_file(str(DASHBOARD_PATH), default_timeout=120).run()


def _markdown(app: AppTest) -> list[str]:
    return [item.value for item in app.markdown]


def test_dashboard_renders_guided_demo_with_story_navigation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)

    app = _run_dashboard(predictive, fault, monkeypatch)

    assert not app.exception
    assert not app.error
    markdown = _markdown(app)
    assert any("not evidence of production safety" in value for value in markdown)
    assert all("thermal-runaway" not in item.value for item in app.caption)
    assert any("tt-pack" in value for value in markdown)
    assert app.get_by_key("v2_guided_primary") is not None
    assert {item.label for item in app.selectbox} >= {"Equation explorer"}
    assert any("Live telemetry" in value for value in markdown)
    for expected in range(2, 8):
        app.button(key="guided_next").click().run()
        assert not app.exception
        assert not app.error
        assert app.session_state["guided_step"] == expected
    app.button(key="guided_prev").click().run()
    assert app.session_state["guided_step"] == 6
    app.button(key="guided_step_2").click().run()
    assert app.session_state["guided_step"] == 2
    assert any("tt-conn" in value for value in _markdown(app))


def test_guided_forecast_step_shows_measured_and_forecast_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    app = _run_dashboard(predictive, fault, monkeypatch)
    app.button(key="guided_step_4").click().run()

    assert not app.exception
    for horizon in (30, 60, 120):
        app.radio(key="g4_horizon").set_value(horizon).run()
        assert not app.exception
        assert app.get_by_key("v2_guided_measured_heatmap") is not None
        assert app.get_by_key("v2_guided_predicted_heatmap") is not None
        assert any(f"+{horizon:.2f} s horizon" in value for value in _markdown(app))
        assert any("tt-tile-forecast" in value for value in _markdown(app))


def test_dashboard_navigation_renders_every_judge_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    app = _run_dashboard(predictive, fault, monkeypatch)

    assert not app.exception
    assert tuple(app.radio(key="v2_workspace").options) == (
        "Guided Demo",
        "Live Thermal Lab",
        "Controllers & Twins",
        "Fault Lab",
    )

    app.radio(key="v2_workspace").set_value("Live Thermal Lab").run()
    assert not app.exception
    assert any("Run simulation" in item.value for item in app.info)

    app.radio(key="v2_workspace").set_value("Controllers & Twins").run()
    assert not app.exception
    assert any("Benchmark evaluation not loaded" in item.value for item in app.info)
    assert app.get_by_key("v2_evidence_temperature_history") is not None

    app.radio(key="v2_workspace").set_value("Fault Lab").run()
    assert not app.exception
    assert app.get_by_key("v2_fault_diagnostics") is not None
    assert app.get_by_key("v2_fault_scores") is not None
    markdown = _markdown(app)
    assert any("Evaluation-only ground truth" in value for value in markdown)
    assert any("Injected fault" in value and "C5" in value for value in markdown)
    assert any("Safety monitoring" in value for value in markdown)


def test_live_lab_runs_only_on_request_and_plays_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    app = _run_dashboard(predictive, fault, monkeypatch)
    app.radio(key="v2_workspace").set_value("Live Thermal Lab").run()
    with pytest.raises(KeyError):
        app.get_by_key("v2_live_heatmap")

    app.selectbox(key="lab_controller").set_value("Reactive threshold")
    app.slider(key="lab_duration").set_value(10)
    app.checkbox(key="lab_fault").check()
    app.number_input(key="lab_fault_start").set_value(40)
    next(item for item in app.button if "Run simulation" in item.label).click().run()

    assert not app.exception
    assert not app.error
    assert any("Fault start moved" in item.value for item in app.warning)
    assert app.get_by_key("v2_live_heatmap") is not None
    assert app.get_by_key("v2_live_temperature_history") is not None
    assert any("Active controller" in value for value in _markdown(app))

    app.button(key="lab_play").click().run()
    assert not app.exception
    app.slider(key="lab_frame").set_value(3).run()
    assert not app.exception
    assert any(
        "Time</div>" in value and ">4 <small>s</small>" in value
        for value in _markdown(app)
    )


def test_dashboard_benchmark_views_with_saved_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = PROJECT_ROOT / "results" / "phase12-final-evaluation"
    if not (evaluation / "evaluation" / "final_evaluation.json").is_file():
        pytest.skip("saved final evaluation artifact is not available locally")
    predictive, fault = make_artifacts(tmp_path)
    app = _run_dashboard(predictive, fault, monkeypatch, str(evaluation))
    app.radio(key="v2_workspace").set_value("Controllers & Twins").run()

    assert not app.exception
    assert app.get_by_key("v2_cooling_controller_matrix") is not None
    assert app.get_by_key("v2_cooling_forecast_comparison") is not None
    assert any("Benchmark summary" in value for value in _markdown(app))
    for option in app.radio(key="ct_metric").options:
        app.radio(key="ct_metric").set_value(option).run()
        assert not app.exception


def test_dashboard_shows_instructions_for_empty_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _run_dashboard("", "", monkeypatch)

    assert not app.exception
    assert any("Enter both artifact directories" in item.value for item in app.info)
    assert not app.metric


def test_dashboard_reports_invalid_artifact_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    (predictive / "comparison_summary.json").write_text("{bad json", encoding="utf-8")

    app = _run_dashboard(predictive, fault, monkeypatch)

    assert not app.exception
    assert any("Could not load dashboard artifacts" in item.value for item in app.error)
    assert not app.metric
    assert not any("Traceback" in item.value for item in app.error)


def test_dashboard_paths_can_be_set_in_source_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    monkeypatch.setenv("THERMOTWIN_PREDICTIVE_RESULTS", "")
    monkeypatch.setenv("THERMOTWIN_FAULT_RESULTS", "")
    monkeypatch.setenv("THERMOTWIN_EVALUATION_RESULTS", "")

    app = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=120).run()
    app.text_input[0].set_value(str(predictive))
    app.text_input[1].set_value(str(fault))
    app.run()

    assert not app.exception
    assert app.get_by_key("v2_guided_primary") is not None


def test_launcher_builds_child_command_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    captured: dict[str, object] = {}

    specification = importlib.util.spec_from_file_location(
        "test_thermotwin_dashboard_launcher", LAUNCHER_PATH
    )
    assert specification is not None and specification.loader is not None
    launcher_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher_module)
    launcher_namespace = cast(Any, launcher_module)
    launcher_main = cast(Callable[[Sequence[str] | None], int], launcher_namespace.main)
    launcher_subprocess = cast(Any, launcher_namespace.subprocess)
    launcher_sys = cast(ModuleType, launcher_namespace.sys)

    def fake_run(
        command: list[str],
        *,
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(launcher_subprocess, "run", fake_run)

    result = launcher_main(
        ["--predictive-directory", str(predictive), "--fault-directory", str(fault)]
    )

    command = cast(list[str], captured["command"])
    environment = cast(dict[str, str], captured["env"])
    assert result == 17
    assert command[:3] == [launcher_sys.executable, "-m", "streamlit"]
    assert command[3] == "run"
    assert command[4].endswith("dashboard.py")
    assert "--server.headless=true" in command
    assert environment["THERMOTWIN_PREDICTIVE_RESULTS"] == str(predictive)
    assert environment["THERMOTWIN_FAULT_RESULTS"] == str(fault)
    assert environment.get("THERMOTWIN_EVALUATION_RESULTS") == os.environ.get(
        "THERMOTWIN_EVALUATION_RESULTS"
    )
    assert captured["check"] is False

    export = tmp_path / "export"
    (export / "evaluation").mkdir(parents=True)
    (export / "evaluation" / "final_evaluation.json").write_text("{}", encoding="utf-8")
    launcher_main(
        [
            "--predictive-directory",
            str(predictive),
            "--fault-directory",
            str(fault),
            "--evaluation-directory",
            str(export),
        ]
    )
    environment = cast(dict[str, str], captured["env"])
    assert environment["THERMOTWIN_EVALUATION_RESULTS"] == str(export.resolve())
    with pytest.raises(SystemExit):
        launcher_main(
            [
                "--predictive-directory",
                str(predictive),
                "--fault-directory",
                str(fault),
                "--evaluation-directory",
                str(tmp_path / "missing"),
            ]
        )


def test_temperature_figure_uses_trace_limit_not_fault_limit(tmp_path: Path) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    trace = replace(data.controller_traces["threshold"], thermal_limit_k=337.0)

    figure = _temperature_history_figure(trace, title="Controller history")

    limit_trace = next(item for item in figure.data if item.name == "Thermal limit")
    assert set(limit_trace.y) == {337.0}


def test_fault_indicators_choose_strongest_event_not_latest_or_injected_cell(
    tmp_path: Path,
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    summary = data.fault_summary
    stronger_early_event = FaultEvent(
        cell_index=2,
        anomaly_score=213.88,
        detection_time_s=15.0,
        predicted_hotspot_temperature_k=312.75,
        severity_k=12.75,
    )
    weaker_late_event = FaultEvent(
        cell_index=1,
        anomaly_score=4.75,
        detection_time_s=23.0,
        predicted_hotspot_temperature_k=308.25,
        severity_k=3.25,
    )
    fallback_summary = replace(
        summary, anomaly_event_count=2, events=(weaker_late_event, stronger_early_event)
    )

    primary_event = _fault_indicators(fallback_summary)
    assert primary_event == stronger_early_event

    target_event = FaultEvent(
        cell_index=summary.fault_cell_index,
        anomaly_score=8.5,
        detection_time_s=6.0,
        predicted_hotspot_temperature_k=315.0,
        severity_k=15.0,
    )
    tied_event = FaultEvent(
        cell_index=2,
        anomaly_score=8.5,
        detection_time_s=6.0,
        predicted_hotspot_temperature_k=314.0,
        severity_k=14.0,
    )
    event_summary = replace(
        summary,
        anomaly_event_count=3,
        events=(weaker_late_event, target_event, tied_event),
    )
    event = _fault_indicators(event_summary)

    assert event == tied_event


def test_fault_operator_panel_uses_one_primary_detector_event_for_all_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    summary = data.fault_summary
    primary_event = FaultEvent(
        cell_index=5,
        anomaly_score=213.88,
        detection_time_s=15.0,
        predicted_hotspot_temperature_k=312.75,
        severity_k=12.75,
    )
    weak_later_event = FaultEvent(
        cell_index=1,
        anomaly_score=4.75,
        detection_time_s=23.0,
        predicted_hotspot_temperature_k=308.25,
        severity_k=3.25,
    )
    data = replace(
        data,
        fault_summary=replace(
            summary, anomaly_event_count=2, events=(weak_later_event, primary_event)
        ),
    )
    markdown: list[str] = []

    class _Column:
        def metric(self, *_args: object) -> None:
            return None

        def markdown(self, value: str) -> None:
            markdown.append(value)

        def plotly_chart(self, *_args: object, **_kwargs: object) -> None:
            return None

    fake_streamlit = SimpleNamespace(
        title=lambda *_args: None,
        caption=lambda *_args: None,
        selectbox=lambda label, options, **_kwargs: {
            "Controller": "Reactive threshold",
            "Thermal twin model": "Data-only",
            "Forecast horizon (seconds)": 120,
        }[label],
        columns=lambda count: [_Column() for _ in range(count)],
        subheader=lambda *_args: None,
        markdown=lambda value: markdown.append(value),
        plotly_chart=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(dashboard_module, "st", fake_streamlit)

    dashboard_module.render_dashboard(data)

    assert "**Suspected cell**\n\nC6" in markdown
    assert "**Anomaly score**\n\n213.88" in markdown
    assert "**Detection time**\n\n15.00 s" in markdown
    assert "**Predicted hotspot**\n\n312.75 K" in markdown
    assert "**Severity**\n\n12.75 K" in markdown
    assert any("Configured injected-fault cell: C5" in value for value in markdown)


def test_fault_indicators_show_no_suspected_cell_without_detector_events(
    tmp_path: Path,
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)

    assert _fault_indicators(data.fault_summary) is None


def test_dashboard_uses_shared_measured_snapshot_not_selected_controller_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    selected = data.controller_traces["predictive_data_only"]
    temperatures = selected.cell_temperatures_k.copy()
    temperatures[-1, :] = 350.0
    selected_controller = replace(selected, cell_temperatures_k=temperatures)
    traces = dict(data.controller_traces)
    traces["predictive_data_only"] = selected_controller
    dashboard_data = replace(data, controller_traces=traces)
    charts: dict[str, object] = {}
    captions: list[str] = []

    class _Column:
        def metric(self, *_args: object) -> None:
            return None

        def markdown(self, *_args: object) -> None:
            return None

        def plotly_chart(self, figure: object, *, key: str, **_kwargs: object) -> None:
            charts[key] = figure

    def _columns(count: int) -> list[_Column]:
        return [_Column() for _ in range(count)]

    fake_streamlit = SimpleNamespace(
        title=lambda *_args: None,
        caption=lambda value: captions.append(value),
        selectbox=lambda label, options, **_kwargs: {
            "Controller": "Predictive (data-only)",
            "Thermal twin model": "Data-only",
            "Forecast horizon (seconds)": 120,
        }[label],
        columns=_columns,
        subheader=lambda *_args: None,
        markdown=lambda *_args: None,
        plotly_chart=lambda figure, *, key, **_kwargs: charts.__setitem__(key, figure),
    )
    monkeypatch.setattr(dashboard_module, "st", fake_streamlit)

    dashboard_module.render_dashboard(dashboard_data)

    expected_source = data.model_forecasts["data_only"][2].source_cell_temperatures_k
    measured_figure = cast(Any, charts["measured_heatmap"])
    measured = np.asarray(measured_figure.data[0].z).reshape(-1)
    np.testing.assert_array_equal(measured, expected_source)
    assert not np.array_equal(measured, selected_controller.cell_temperatures_k[-1])
    assert any(
        "shared measured source: threshold-run data" in value.lower()
        for value in captions
    )
    assert any("source time: 2.00 s" in value.lower() for value in captions)
    assert any("selected model: data-only" in value.lower() for value in captions)
    assert any("forecast horizon: 120 s" in value.lower() for value in captions)
    assert any("coolant action: 0.1000 kg/s" in value.lower() for value in captions)
