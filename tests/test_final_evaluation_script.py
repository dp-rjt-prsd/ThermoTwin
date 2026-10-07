from __future__ import annotations

import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest
import scripts.final_evaluation as final_evaluation_script

from test_final_evaluation import _make_inputs

_FIGURES = (
    "controller_matrix.html",
    "model_accuracy.html",
    "forecast_errors.html",
    "physics_residuals.html",
    "thermal_energy_tradeoff.html",
    "fault_timing.html",
)


def _arguments(
    sources: tuple[Path, Path, Path, Path, Path], output: Path
) -> list[str]:
    benchmark, comparison, id_forecast, ood_forecast, fault = sources
    return [
        "--benchmark-directory",
        str(benchmark),
        "--comparison-directory",
        str(comparison),
        "--id-forecast-directory",
        str(id_forecast),
        "--ood-forecast-directory",
        str(ood_forecast),
        "--fault-directory",
        str(fault),
        "--output-directory",
        str(output),
    ]


def _create_sources(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    return _make_inputs(root / "source")


def _file_manifest(root: Path) -> dict[str, object]:
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_cli_atomically_exports_reloadable_evaluation_and_six_standalone_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sources = _create_sources(tmp_path)
    output = tmp_path / "published"

    assert final_evaluation_script.main(_arguments(sources, output)) == 0

    response = json.loads(capsys.readouterr().out)
    assert response["output_directory"] == str(output.resolve())
    assert response["evidence_status"] == "workflow_only"
    assert response["benchmark_mode"] == "quick"
    assert set(response["figures"]) == set(_FIGURES)
    assert (output / "evaluation/final_evaluation.json").is_file()
    assert {path.name for path in (output / "figures").iterdir()} == set(_FIGURES)
    manifest = _file_manifest(output)
    assert manifest["schema_version"] == 1
    assert manifest["evidence_status"] == "workflow_only"
    assert manifest["benchmark_mode"] == "quick"
    files = manifest["files"]
    assert isinstance(files, list)
    expected_paths = [
        "evaluation/final_evaluation.json",
        *(f"figures/{name}" for name in _FIGURES),
    ]
    assert [record["path"] for record in files] == expected_paths
    for record in files:
        path = output / record["path"]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    evaluation_text = (output / "evaluation/final_evaluation.json").read_text(
        encoding="utf-8"
    )
    assert '"evidence_status": "workflow_only"' in evaluation_text
    for name in _FIGURES:
        html = (output / "figures" / name).read_text(encoding="utf-8")
        assert "Evidence status: workflow_only" in html
        assert "Synthetic, descriptive" in html
        assert '<script src="https://cdn.plot.ly' not in html
        assert "Plotly.newPlot" in html
    assert not list(tmp_path.glob(".published.partial-*"))


def test_cli_repeated_export_is_byte_deterministic(tmp_path: Path) -> None:
    sources = _create_sources(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert final_evaluation_script.main(_arguments(sources, first)) == 0
    assert final_evaluation_script.main(_arguments(sources, second)) == 0

    first_manifest = (first / "manifest.json").read_bytes()
    second_manifest = (second / "manifest.json").read_bytes()
    assert first_manifest == second_manifest
    for relative in (
        "evaluation/final_evaluation.json",
        *(f"figures/{name}" for name in _FIGURES),
    ):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_existing_output_is_rejected_before_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("aggregation must not run for an existing output")

    monkeypatch.setattr(
        final_evaluation_script, "aggregate_final_evaluation", unexpected
    )
    assert final_evaluation_script.main(_arguments((output,) * 5, output)) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "Traceback" not in captured.err
    assert (output / "keep.txt").read_text(encoding="utf-8") == "preserve"


def test_unexpected_aggregation_type_error_propagates_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unexpected-type-error"

    def fail_unexpectedly(*_args: object, **_kwargs: object) -> object:
        raise TypeError("internal programming defect")

    monkeypatch.setattr(
        final_evaluation_script, "aggregate_final_evaluation", fail_unexpectedly
    )
    with pytest.raises(TypeError, match="internal programming defect"):
        final_evaluation_script.main(
            _arguments((tmp_path,) * 5, output)
        )
    assert not output.exists()


def test_missing_or_malformed_sources_fail_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_sources = tuple(tmp_path / name for name in ("b", "c", "id", "ood", "f"))
    assert final_evaluation_script.main(
        _arguments(missing_sources, tmp_path / "missing-output")
    ) == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not (tmp_path / "missing-output").exists()

    sources = _create_sources(tmp_path / "malformed")
    comparison = sources[1]
    (comparison / "provenance.json").write_text("{broken", encoding="utf-8")
    assert final_evaluation_script.main(
        _arguments(sources, tmp_path / "malformed-output")
    ) == 2
    captured = capsys.readouterr()
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err
    assert not (tmp_path / "malformed-output").exists()


def test_injected_mid_export_failure_cleans_only_owned_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sources = _create_sources(tmp_path)
    output = tmp_path / "failed"
    unrelated_stage = tmp_path / ".failed.partial-preserve"
    unrelated_stage.mkdir()
    (unrelated_stage / "keep.txt").write_text("preserve", encoding="utf-8")
    original_write_html = go.Figure.write_html
    write_count = 0

    def fail_on_third(
        figure: go.Figure, *args: object, **kwargs: object
    ) -> object:
        nonlocal write_count
        write_count += 1
        if write_count == 3:
            raise OSError("injected figure export failure")
        return original_write_html(figure, *args, **kwargs)

    monkeypatch.setattr(go.Figure, "write_html", fail_on_third)
    assert final_evaluation_script.main(_arguments(sources, output)) == 2
    captured = capsys.readouterr()
    assert "injected figure export failure" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
    assert (unrelated_stage / "keep.txt").read_text(encoding="utf-8") == "preserve"
    assert list(tmp_path.glob(".failed.partial-*")) == [unrelated_stage]


def test_help_lists_all_required_source_and_output_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        final_evaluation_script.main(["--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    for option in (
        "--benchmark-directory",
        "--comparison-directory",
        "--id-forecast-directory",
        "--ood-forecast-directory",
        "--fault-directory",
        "--output-directory",
    ):
        assert option in output
