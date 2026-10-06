from __future__ import annotations

import importlib.util
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
from streamlit.testing.v1 import AppTest

from test_dashboard_data import make_artifacts
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
    predictive: Path | str, fault: Path | str, monkeypatch: pytest.MonkeyPatch
) -> AppTest:
    monkeypatch.setenv("THERMOTWIN_PREDICTIVE_RESULTS", str(predictive))
    monkeypatch.setenv("THERMOTWIN_FAULT_RESULTS", str(fault))
    return AppTest.from_file(str(DASHBOARD_PATH)).run()


def test_dashboard_renders_artifacts_with_selectors_metrics_and_fault_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)

    app = _run_dashboard(predictive, fault, monkeypatch)

    assert not app.exception
    assert any("ThermoTwin" in item.value for item in app.title)
    assert any("fault-induced thermal anomaly" in item.value for item in app.caption)
    assert any("synthetic" in item.value for item in app.caption)
    assert all("thermal-runaway" not in item.value for item in app.caption)
    assert len(app.metric) == 6
    assert len(app.selectbox) == 3
    assert {item.label for item in app.selectbox} >= {
        "Controller",
        "Thermal twin model",
        "Forecast horizon (seconds)",
    }
    assert any(item.value == "301.00 K" for item in app.metric)
    assert any("Suspected / configured cell" in item.value for item in app.markdown)
    assert any("C5" in item.value for item in app.markdown)
    assert any("Unavailable" in item.value for item in app.markdown)
    chart_keys = (
        "measured_heatmap",
        "predicted_heatmap",
        "temperature_history",
        "coolant_flow_history",
        "controller_comparison",
        "model_forecast_comparison",
        "fault_diagnostics",
    )
    assert all(app.get_by_key(key) is not None for key in chart_keys)


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


def test_dashboard_paths_can_be_set_in_sidebar_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    monkeypatch.setenv("THERMOTWIN_PREDICTIVE_RESULTS", "")
    monkeypatch.setenv("THERMOTWIN_FAULT_RESULTS", "")

    app = AppTest.from_file(str(DASHBOARD_PATH)).run()
    app.text_input[0].set_value(str(predictive))
    app.text_input[1].set_value(str(fault))
    app.run()

    assert not app.exception
    assert len(app.metric) == 6


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
    launcher_main = cast(
        Callable[[Sequence[str] | None], int], launcher_namespace.main
    )
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
    assert captured["check"] is False


def test_temperature_figure_uses_trace_limit_not_fault_limit(tmp_path: Path) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    trace = replace(data.controller_traces["threshold"], thermal_limit_k=337.0)

    figure = _temperature_history_figure(trace, title="Controller history")

    limit_trace = next(item for item in figure.data if item.name == "Thermal limit")
    assert set(limit_trace.y) == {337.0}


def test_fault_indicators_use_target_cell_event_or_target_cell_trace(
    tmp_path: Path,
) -> None:
    predictive, fault = make_artifacts(tmp_path)
    data = load_dashboard_data(predictive, fault)
    summary = data.fault_summary
    unrelated_event = FaultEvent(
        cell_index=3,
        anomaly_score=99.0,
        detection_time_s=5.0,
        predicted_hotspot_temperature_k=399.0,
        severity_k=20.0,
    )
    predicted_temperatures = data.fault_trace.predicted_temperature_k.copy()
    predicted_temperatures[:, 3] = 399.0
    fault_trace = replace(
        data.fault_trace, predicted_temperature_k=predicted_temperatures
    )
    fallback_summary = replace(
        summary, anomaly_event_count=1, events=(unrelated_event,)
    )

    fallback_score, fallback_hotspot = _fault_indicators(
        fault_trace, fallback_summary
    )

    assert fallback_score == pytest.approx(0.5)
    assert fallback_hotspot == pytest.approx(300.0)

    target_event = FaultEvent(
        cell_index=summary.fault_cell_index,
        anomaly_score=8.5,
        detection_time_s=6.0,
        predicted_hotspot_temperature_k=315.0,
        severity_k=15.0,
    )
    event_summary = replace(
        summary,
        anomaly_event_count=2,
        events=(unrelated_event, target_event),
    )
    event_score, event_hotspot = _fault_indicators(data.fault_trace, event_summary)

    assert event_score == pytest.approx(8.5)
    assert event_hotspot == pytest.approx(315.0)
