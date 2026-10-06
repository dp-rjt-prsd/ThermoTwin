"""Acceptance tests for the reproducible closed-loop controller comparison."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "run_predictive.py"


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *arguments],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_smoke_comparison_saves_four_fair_runs_and_measured_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "predictive-smoke"
    completed = _run_cli(
        "--smoke-test", "--seed", "13", "--output-directory", str(output)
    )

    assert completed.returncode == 0, completed.stderr
    printed_summary = json.loads(completed.stdout)
    summary_path = output / "comparison_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert printed_summary == summary
    assert summary["seed"] == 13
    assert summary["scenario"]["name"] == "predictive-stop-and-go"
    assert set(summary["runs"]) == {
        "threshold",
        "pid",
        "predictive_data_only",
        "predictive_physics_informed",
    }

    provenance_values = []
    for run_id, run in summary["runs"].items():
        artifact_path = output / run["artifact_path"]
        assert artifact_path.is_dir()
        saved_metrics = json.loads(
            (artifact_path / "metrics.json").read_text(encoding="utf-8")
        )
        assert run["metrics"] == saved_metrics
        provenance = json.loads(
            (artifact_path / "provenance.json").read_text(encoding="utf-8")
        )
        provenance_values.append(provenance)
        assert provenance["seed"] == 13
        assert provenance["scenario"]["name"] == "predictive-stop-and-go"
        assert (artifact_path / "timeseries.csv").is_file()
        if run_id == "predictive_data_only":
            assert run["variant"] == "data_only"
        elif run_id == "predictive_physics_informed":
            assert run["variant"] == "physics_informed"

    assert len(provenance_values) == 4
    reference = provenance_values[0]
    for provenance in provenance_values[1:]:
        assert provenance["seed"] == reference["seed"]
        assert provenance["scenario"] == reference["scenario"]
        assert provenance["environment_config"] == reference["environment_config"]
        assert provenance["scenario_metadata"] == reference["scenario_metadata"]


def test_existing_output_is_rejected_before_any_run_directory_is_written(
    tmp_path: Path,
) -> None:
    comparison = tmp_path / "comparison"
    comparison.mkdir()
    output = tmp_path / "existing-output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    completed = _run_cli(
        "--smoke-test",
        "--comparison-directory",
        str(comparison),
        "--output-directory",
        str(output),
    )

    assert completed.returncode != 0
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert list(output.iterdir()) == [marker]


def test_missing_comparison_directory_is_reported_without_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "should-not-exist"
    completed = _run_cli(
        "--comparison-directory",
        str(tmp_path / "missing-comparison"),
        "--output-directory",
        str(output),
    )

    assert completed.returncode != 0
    assert "comparison" in completed.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize("seed", ["-1", "not-an-integer"])
def test_invalid_seed_is_rejected(seed: str, tmp_path: Path) -> None:
    completed = _run_cli("--seed", seed, "--output-directory", str(tmp_path / "unused"))

    assert completed.returncode == 2
    assert "seed" in completed.stderr.lower()
