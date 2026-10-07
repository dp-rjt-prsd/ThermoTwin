"""Aggregate validated Phase 12 evidence and export standalone figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from thermotwin.experiments.final_evaluation import (
    FinalEvaluation,
    aggregate_final_evaluation,
    load_final_evaluation,
    save_final_evaluation,
)
from thermotwin.visualization.benchmark_plots import (
    controller_matrix_figure,
    fault_timing_figure,
    forecast_error_figure,
    model_accuracy_figure,
    physics_residual_figure,
    thermal_energy_tradeoff_figure,
)

_MANIFEST_SCHEMA_VERSION = 1
_FIGURE_BUILDERS = (
    ("controller_matrix.html", controller_matrix_figure),
    ("model_accuracy.html", model_accuracy_figure),
    ("forecast_errors.html", forecast_error_figure),
    ("physics_residuals.html", physics_residual_figure),
    ("thermal_energy_tradeoff.html", thermal_energy_tradeoff_figure),
    ("fault_timing.html", fault_timing_figure),
)
_EVALUATION_PATH = "evaluation/final_evaluation.json"
_MANIFEST_FIELDS = {
    "schema_version",
    "evidence_status",
    "benchmark_mode",
    "files",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-directory", type=Path, required=True)
    parser.add_argument("--comparison-directory", type=Path, required=True)
    parser.add_argument("--id-forecast-directory", type=Path, required=True)
    parser.add_argument("--ood-forecast-directory", type=Path, required=True)
    parser.add_argument("--fault-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: object) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _expected_files() -> tuple[str, ...]:
    return (_EVALUATION_PATH, *(f"figures/{name}" for name, _ in _FIGURE_BUILDERS))


def _write_manifest(stage: Path, evaluation: FinalEvaluation) -> None:
    files = [
        {"path": relative, "sha256": _sha256(stage / PurePosixPath(relative))}
        for relative in _expected_files()
    ]
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "evidence_status": evaluation.evidence_status,
        "benchmark_mode": evaluation.benchmark_mode,
        "files": files,
    }
    with (stage / "manifest.json").open(
        "x", encoding="utf-8", newline=""
    ) as stream:
        stream.write(_json_text(manifest))
        stream.flush()
        os.fsync(stream.fileno())


def _export_figures(stage: Path, evaluation: FinalEvaluation) -> None:
    figure_directory = stage / "figures"
    figure_directory.mkdir()
    for filename, builder in _FIGURE_BUILDERS:
        figure = builder(evaluation)
        figure.write_html(
            figure_directory / filename,
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
            div_id=f"thermotwin-{filename.removesuffix('.html')}",
            config={"responsive": True, "displaylogo": False},
        )


def _strict_manifest(root: Path, evaluation: FinalEvaluation) -> list[dict[str, str]]:
    try:
        with (root / "manifest.json").open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"published manifest cannot be read: {error}") from error
    if type(raw) is not dict or set(raw) != _MANIFEST_FIELDS:
        raise ValueError("published manifest has unexpected fields")
    if raw["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise ValueError("published manifest schema is unsupported")
    if (
        raw["evidence_status"] != evaluation.evidence_status
        or raw["benchmark_mode"] != evaluation.benchmark_mode
    ):
        raise ValueError("published manifest evidence identity is inconsistent")
    files = raw["files"]
    if type(files) is not list or len(files) != len(_expected_files()):
        raise ValueError("published manifest file inventory is incomplete")
    validated: list[dict[str, str]] = []
    for index, record in enumerate(files):
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ValueError("published manifest file record is malformed")
        relative = record["path"]
        checksum = record["sha256"]
        if type(relative) is not str or relative != _expected_files()[index]:
            raise ValueError("published manifest paths are not canonical")
        if (
            type(checksum) is not str
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("published manifest digest is invalid")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise ValueError("published manifest contains an unsafe path")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"published file is missing or invalid: {relative}")
        if _sha256(path) != checksum:
            raise ValueError(f"published file digest mismatch: {relative}")
        validated.append({"path": relative, "sha256": checksum})
    return validated


def _verify_published(root: Path, evaluation: FinalEvaluation) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("published output is not a regular directory")
    expected_inventory = {
        "manifest.json",
        "evaluation",
        "figures",
        *_expected_files(),
    }
    found: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            if path.is_symlink():
                raise ValueError("published output must not contain symlinks")
            found.add(path.relative_to(root).as_posix())
    if found != expected_inventory:
        raise ValueError("published output contains missing or unexpected files")
    _strict_manifest(root, evaluation)
    reloaded = load_final_evaluation(root / "evaluation")
    if reloaded != evaluation:
        raise ValueError("published evaluation does not match aggregated evidence")


def _publish(evaluation: FinalEvaluation, output_directory: Path) -> Path:
    requested = Path(output_directory)
    if requested.name in ("", ".", "..") or ".partial-" in requested.name:
        raise ValueError("output directory must be a final bundle directory")
    parent = requested.parent.resolve()
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")
    parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"output directory already exists: {destination}")
    stage = parent / f".{requested.name}.partial-{uuid.uuid4().hex}"
    stage.mkdir(exist_ok=False)
    published = False
    try:
        save_final_evaluation(evaluation, stage / "evaluation")
        _export_figures(stage, evaluation)
        _write_manifest(stage, evaluation)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"output directory already exists: {destination}")
        os.rename(stage, destination)
        published = True
        _verify_published(destination, evaluation)
        return destination
    except BaseException:
        if stage.exists() and stage.parent == parent and ".partial-" in stage.name:
            shutil.rmtree(stage)
        if published and destination.exists() and destination.parent == parent:
            shutil.rmtree(destination)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory: Path = args.output_directory
    try:
        if output_directory.exists() or output_directory.is_symlink():
            raise FileExistsError(
                f"output directory already exists: {output_directory}"
            )
        evaluation = aggregate_final_evaluation(
            args.benchmark_directory,
            args.comparison_directory,
            args.id_forecast_directory,
            args.ood_forecast_directory,
            args.fault_directory,
        )
        output = _publish(evaluation, output_directory)
        response = {
            "output_directory": str(output.resolve()),
            "evaluation_file": str((output / _EVALUATION_PATH).resolve()),
            "figures": {
                filename: str((output / "figures" / filename).resolve())
                for filename, _ in _FIGURE_BUILDERS
            },
            "evidence_status": evaluation.evidence_status,
            "benchmark_mode": evaluation.benchmark_mode,
        }
        print(_json_text(response), end="")
        return 0
    except (
        OSError,
        ValueError,
    ) as error:
        print(
            f"final_evaluation: error: {error}\n"
            "Verify that every input directory contains compatible, validated "
            "saved evidence and choose a new output directory.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
