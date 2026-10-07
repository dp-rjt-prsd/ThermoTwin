from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from thermotwin.experiments.benchmark import BenchmarkExecution
from thermotwin.experiments.benchmark_serialization import (
    load_benchmark_matrix,
    save_benchmark_execution,
)
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    BenchmarkMode,
)


def _execution(mode: BenchmarkMode = "quick") -> BenchmarkExecution:
    from test_benchmark import _execute

    return _execute(mode)


def _save_bundle(
    tmp_path: Path, name: str, *, mode: BenchmarkMode = "quick"
) -> Path:
    return save_benchmark_execution(_execution(mode), tmp_path / name)


def _read_manifest(bundle: Path) -> dict[str, object]:
    value = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(bundle: Path, manifest: dict[str, object]) -> None:
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, allow_nan=False), encoding="utf-8"
    )


def _refresh_artifact_hash(
    bundle: Path, cell_index: int, artifact: str
) -> None:
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    cell = cells[cell_index]
    assert isinstance(cell, dict)
    artifacts = cell["artifacts"]
    hashes = cell["artifact_sha256"]
    assert isinstance(artifacts, dict) and isinstance(hashes, dict)
    artifact_path = bundle / str(artifacts[artifact])
    hashes[artifact] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _write_manifest(bundle, manifest)


def _expected_files() -> set[str]:
    files = {"manifest.json", "runs"}
    for scenario in SCENARIO_IDS:
        files.add(f"runs/{scenario}")
        for variant in VARIANT_IDS:
            path = f"runs/{scenario}/{variant}"
            files.add(path)
            files.update(
                {
                    f"{path}/timeseries.csv",
                    f"{path}/metrics.json",
                    f"{path}/provenance.json",
                }
            )
    return files


def _inventory(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
    }


def test_saves_and_reloads_complete_matrix_with_exact_inventory(tmp_path: Path) -> None:
    execution = _execution()
    destination = save_benchmark_execution(execution, tmp_path / "bundle")
    assert destination == tmp_path / "bundle"
    assert _inventory(destination) == _expected_files()
    reloaded = load_benchmark_matrix(destination)
    assert reloaded == execution.matrix
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert len(manifest["cells"]) == 24


def test_save_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_benchmark_execution(_execution(), destination)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_loader_rejects_tampered_missing_and_unexpected_files(tmp_path: Path) -> None:
    original = _save_bundle(tmp_path, "original")
    tampered = tmp_path / "tampered"
    shutil.copytree(original, tampered)
    target = tampered / "runs/normal/threshold/metrics.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_benchmark_matrix(tampered)

    missing = tmp_path / "missing"
    shutil.copytree(original, missing)
    (missing / "runs/normal/threshold/timeseries.csv").unlink()
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_benchmark_matrix(missing)

    extra = tmp_path / "extra"
    shutil.copytree(original, extra)
    (extra / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_benchmark_matrix(extra)


@pytest.mark.parametrize(
    "manifest_text",
    [
        '{"schema_version":1,"schema_version":1,"complete":true,"mode":"quick","cells":[]}',
        '{"schema_version":1,"complete":NaN,"mode":"quick","cells":[]}',
        '{"schema_version":1,"complete":false,"mode":"quick","cells":[]}',
    ],
)
def test_loader_rejects_duplicate_nonfinite_or_partial_manifest(
    tmp_path: Path, manifest_text: str
) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    (bundle / "manifest.json").write_text(manifest_text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_benchmark_matrix(bundle)


def test_loader_rejects_partial_named_directory(tmp_path: Path) -> None:
    bundle = _save_bundle(tmp_path, "valid")
    partial = tmp_path / "bundle.partial-incomplete"
    shutil.copytree(bundle, partial)
    with pytest.raises(ValueError, match="partial"):
        load_benchmark_matrix(partial)


def test_handled_save_failure_cleans_only_its_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import thermotwin.experiments.benchmark_serialization as serialization

    def fail(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected trace writer failure")

    monkeypatch.setattr(serialization, "save_experiment_result", fail)
    unrelated_stage = tmp_path / ".failed.partial-keep"
    unrelated_stage.mkdir()
    with pytest.raises(OSError, match="injected"):
        save_benchmark_execution(_execution(), tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert list(tmp_path.glob(".failed.partial-*")) == [unrelated_stage]


@pytest.mark.parametrize("artifact_path", ["../outside.csv", "absolute"])
def test_loader_rejects_noncanonical_traversal_and_absolute_paths(
    tmp_path: Path, artifact_path: str
) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    artifacts = cells[0]["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["timeseries"] = (
        str((bundle / "runs/normal/threshold/timeseries.csv").resolve())
        if artifact_path == "absolute"
        else artifact_path
    )
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="canonical relative paths"):
        load_benchmark_matrix(bundle)


def test_loader_rejects_manifest_seed_that_differs_from_saved_provenance(
    tmp_path: Path,
) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cells[0]["seed"] = int(cells[0]["seed"]) + 1
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="saved seed"):
        load_benchmark_matrix(bundle)


def test_loader_rejects_one_cell_configuration_digest_mismatch(
    tmp_path: Path,
) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cells[0]["scenario_configuration_sha256"] = "f" * 64
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="share scenario_configuration_sha256"):
        load_benchmark_matrix(bundle)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "reordered"])
def test_loader_rejects_duplicate_unknown_or_reordered_cells(
    tmp_path: Path, mutation: str
) -> None:
    bundle = _save_bundle(tmp_path, f"bundle-{mutation}")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    if mutation == "duplicate":
        assert isinstance(cells[0], dict) and isinstance(cells[1], dict)
        cells[1]["scenario_id"] = cells[0]["scenario_id"]
        cells[1]["variant_id"] = cells[0]["variant_id"]
    elif mutation == "unknown":
        assert isinstance(cells[0], dict)
        cells[0]["variant_id"] = "unknown"
    else:
        cells[0], cells[1] = cells[1], cells[0]
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="duplicate|unknown|canonical"):
        load_benchmark_matrix(bundle)


def test_loader_rejects_model_provenance_variant_mismatch(tmp_path: Path) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[2], dict)
    provenance = cells[2]["model_provenance"]
    assert isinstance(provenance, dict)
    provenance["model_variant"] = "physics_informed"
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="model_provenance.model_variant"):
        load_benchmark_matrix(bundle)


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [("metrics", "malformed"), ("provenance", "invalid")],
)
def test_loader_rejects_semantically_invalid_saved_artifact_after_hash_update(
    tmp_path: Path, artifact: str, mutation: str
) -> None:
    bundle = _save_bundle(tmp_path, f"bundle-{artifact}")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    paths = cells[0]["artifacts"]
    assert isinstance(paths, dict)
    path = bundle / str(paths[artifact])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(saved, dict)
    if artifact == "metrics":
        saved["peak_temperature_k"] = "invalid"
    else:
        saved["record_count"] = 0
    path.write_text(json.dumps(saved, allow_nan=False), encoding="utf-8")
    _refresh_artifact_hash(bundle, 0, artifact)
    with pytest.raises(ValueError):
        load_benchmark_matrix(bundle)


def test_full_matrix_rejects_smoke_model_provenance(tmp_path: Path) -> None:
    bundle = _save_bundle(tmp_path, "full-bundle", mode="full")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[2], dict)
    provenance = cells[2]["model_provenance"]
    assert isinstance(provenance, dict)
    provenance["training_mode"] = "smoke"
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="training_mode must be full"):
        load_benchmark_matrix(bundle)


def test_loader_rejects_symlink_in_bundle(tmp_path: Path) -> None:
    bundle = _save_bundle(tmp_path, "bundle")
    link = bundle / "unexpected-link"
    target = bundle / "manifest.json"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        load_benchmark_matrix(bundle)
