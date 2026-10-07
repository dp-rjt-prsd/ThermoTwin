"""User-facing subprocess checks for the Phase 12 benchmark command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.experiments.benchmark_serialization import load_benchmark_matrix
from thermotwin.models.config import MLPConfig
from thermotwin.models.mlp import ThermalMLP

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "benchmark.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *arguments],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _make_comparison(directory: Path, *, smoke_test: bool = True) -> Path:
    directory.mkdir()
    model_config = MLPConfig(input_dim=17, output_dim=12, hidden_widths=(8,))
    (directory / "model_config.json").write_text(
        json.dumps(
            {
                "input_dim": model_config.input_dim,
                "output_dim": model_config.output_dim,
                "hidden_widths": list(model_config.hidden_widths),
                "activation": model_config.activation,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    names = feature_names(12)
    targets = target_names(12)
    normalization = {
        "schema_version": 1,
        "feature_names": names,
        "target_names": targets,
        "feature_mean": np.zeros(17).tolist(),
        "feature_scale": np.ones(17).tolist(),
        "target_mean": np.zeros(12).tolist(),
        "target_scale": np.ones(12).tolist(),
    }
    (directory / "normalization.json").write_text(
        json.dumps(normalization, sort_keys=True), encoding="utf-8"
    )
    model = ThermalMLP(model_config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    for variant in ("data_only", "physics_informed"):
        torch.save(model.state_dict(), directory / f"{variant}.pt")

    def manifest(seed: int) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "seed": seed,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "cell_count": 12,
            "columns": [f"column_{index}" for index in range(25)],
            "splits": {
                split: {
                    "row_count": 4,
                    "trajectory_count": 1,
                    "trajectory_ids": [f"{split}-trajectory"],
                }
                for split in ("train", "validation", "test")
            },
        }

    provenance = {
        "artifact_schema_version": "1.0",
        "datasets": {
            "in_distribution": {
                "directory": "C:/temporary/id",
                "manifest_path": "C:/temporary/id/manifest.json",
                "manifest": manifest(1729),
                "distribution": "in_distribution",
            },
            "out_of_distribution": {
                "directory": "C:/temporary/ood",
                "manifest_path": "C:/temporary/ood/manifest.json",
                "manifest": manifest(1730),
                "distribution": "out_of_distribution",
                "normalization_source": "in_distribution",
            },
        },
        "seed": 1729,
        "dataset_seeds": {
            "in_distribution": 1729,
            "out_of_distribution": 1730,
        },
        "package_versions": {"torch": "2.6.0"},
        "config_source": "configs/training.yaml",
        "configuration_source_path": "C:/repository/configs/training.yaml",
        "normalization": {
            "artifact": "normalization.json",
            "fit_dataset": "in_distribution",
            "reused_for": ["in_distribution", "out_of_distribution"],
        },
        "distribution_labels": ["in_distribution", "out_of_distribution"],
        "smoke_test": smoke_test,
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    return directory


def test_help_and_invalid_seed_are_standard_argparse_responses() -> None:
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "--comparison-directory" in help_result.stdout
    assert "--output-directory" in help_result.stdout

    invalid = _run("--quick", "--seed", "-1")
    assert invalid.returncode == 2
    assert "seed must be a nonnegative integer" in invalid.stderr
    assert "Traceback" not in invalid.stderr


def test_existing_output_is_rejected_before_comparison_loading(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    result = _run(
        "--quick",
        "--comparison-directory",
        str(tmp_path / "missing-comparison"),
        "--output-directory",
        str(output),
    )
    assert result.returncode == 2
    assert "output directory already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert "missing-comparison" not in result.stderr


def test_full_mode_requires_a_comparison_directory(tmp_path: Path) -> None:
    result = _run("--output-directory", str(tmp_path / "full"))
    assert result.returncode == 2
    assert "full mode requires --comparison-directory" in result.stderr
    assert not (tmp_path / "full").exists()
    assert "Traceback" not in result.stderr


def test_missing_and_malformed_comparison_artifacts_fail_closed(
    tmp_path: Path,
) -> None:
    missing = _run(
        "--quick",
        "--comparison-directory",
        str(tmp_path / "missing"),
        "--output-directory",
        str(tmp_path / "missing-output"),
    )
    assert missing.returncode == 2
    assert "comparison directory is missing or invalid" in missing.stderr
    assert not (tmp_path / "missing-output").exists()

    malformed_directory = tmp_path / "malformed"
    malformed_directory.mkdir()
    (malformed_directory / "provenance.json").write_text("{", encoding="utf-8")
    for filename in (
        "data_only.pt",
        "physics_informed.pt",
        "model_config.json",
        "normalization.json",
    ):
        (malformed_directory / filename).write_bytes(b"present")
    malformed = _run(
        "--quick",
        "--comparison-directory",
        str(malformed_directory),
        "--output-directory",
        str(tmp_path / "malformed-output"),
    )
    assert malformed.returncode == 2
    assert "could not read provenance.json" in malformed.stderr
    assert "Traceback" not in malformed.stderr
    assert not (tmp_path / "malformed-output").exists()


def test_quick_matrix_is_reloadable_and_reproducible_with_supplied_models(
    tmp_path: Path,
) -> None:
    comparison = _make_comparison(tmp_path / "comparison")
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = _run(
        "--quick",
        "--comparison-directory",
        str(comparison),
        "--output-directory",
        str(first_output),
    )
    assert first.returncode == 0, first.stderr
    first_summary = json.loads(first.stdout)
    assert first_summary == {
        "output_directory": str(first_output.resolve()),
        "mode": "quick",
        "scenario_count": 6,
        "variant_count": 4,
        "cell_count": 24,
        "seed": 1729,
        "complete": True,
        "evidence_status": "workflow_only",
    }
    assert len(load_benchmark_matrix(first_output).cells) == 24

    second = _run(
        "--quick",
        "--comparison-directory",
        str(comparison),
        "--output-directory",
        str(second_output),
    )
    assert second.returncode == 0, second.stderr
    assert len(load_benchmark_matrix(second_output).cells) == 24
    first_manifest = json.loads((first_output / "manifest.json").read_text())
    second_manifest = json.loads((second_output / "manifest.json").read_text())
    assert first_manifest == second_manifest
    for cell in first_manifest["cells"]:
        for artifact in ("metrics", "timeseries", "provenance"):
            relative = cell["artifacts"][artifact]
            assert (first_output / relative).read_bytes() == (
                second_output / relative
            ).read_bytes()


def test_quick_without_comparison_trains_in_temporary_directory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "automatic-quick"
    result = _run("--quick", "--seed", "37", "--output-directory", str(output))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["seed"] == 37
    assert summary["cell_count"] == 24
    matrix = load_benchmark_matrix(output)
    assert len(matrix.cells) == 24
    for cell in matrix.cells:
        if cell.model_provenance is not None:
            assert cell.model_provenance.training_mode == "smoke"
            assert cell.model_provenance.comparison_provenance.path == (
                "comparison/provenance.json"
            )


def test_full_mode_rejects_smoke_provenance_before_execution(tmp_path: Path) -> None:
    comparison = _make_comparison(tmp_path / "smoke-comparison", smoke_test=True)
    output = tmp_path / "full-output"
    result = _run(
        "--comparison-directory",
        str(comparison),
        "--output-directory",
        str(output),
    )
    assert result.returncode == 2
    assert "full mode rejects smoke-trained comparison artifacts" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
