from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.experiments.benchmark import BenchmarkExecution
from thermotwin.experiments.benchmark_serialization import (
    load_benchmark_matrix,
    save_benchmark_execution,
)
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkMatrix,
    BenchmarkMode,
)
from thermotwin.models.config import MLPConfig
from thermotwin.models.mlp import ThermalMLP


def _execution(mode: BenchmarkMode = "quick") -> BenchmarkExecution:
    from test_benchmark import _execute

    return _execute(mode)


def _evidence(mode: BenchmarkMode) -> dict[str, bytes]:
    model_config = MLPConfig(17, 12, (8,))
    model_config_bytes = json.dumps(
        {
            "input_dim": 17,
            "output_dim": 12,
            "hidden_widths": [8],
            "activation": "gelu",
        },
        sort_keys=True,
        indent=2,
    ).encode() + b"\n"
    model = ThermalMLP(model_config)
    checkpoint_stream = io.BytesIO()
    torch.save(model.state_dict(), checkpoint_stream)
    checkpoint = checkpoint_stream.getvalue()
    normalization = {
        "schema_version": 1,
        "feature_names": feature_names(12),
        "target_names": target_names(12),
        "feature_mean": np.zeros(17).tolist(),
        "feature_scale": np.ones(17).tolist(),
        "target_mean": np.zeros(12).tolist(),
        "target_scale": np.ones(12).tolist(),
    }
    normalization_bytes = json.dumps(
        normalization, sort_keys=True, indent=2
    ).encode() + b"\n"

    def dataset_manifest(seed: int) -> dict[str, object]:
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

    manifests = {
        "in_distribution": dataset_manifest(17),
        "out_of_distribution": dataset_manifest(18),
    }
    comparison = {
        "artifact_schema_version": "1.0",
        "datasets": {
            label: {
                "directory": f"dataset/{label}",
                "manifest_path": f"dataset/{label}/manifest.json",
                "manifest": manifest,
                "distribution": label,
                **(
                    {"normalization_source": "in_distribution"}
                    if label == "out_of_distribution"
                    else {}
                ),
            }
            for label, manifest in manifests.items()
        },
        "seed": 17,
        "dataset_seeds": {"in_distribution": 17, "out_of_distribution": 18},
        "package_versions": {"torch": torch.__version__},
        "config_source": "configs/training.yaml",
        "configuration_source_path": "configs/training.yaml",
        "normalization": {
            "artifact": "normalization.json",
            "fit_dataset": "in_distribution",
            "reused_for": ["in_distribution", "out_of_distribution"],
        },
        "distribution_labels": ["in_distribution", "out_of_distribution"],
        "smoke_test": mode == "quick",
    }
    encode = lambda value: (  # noqa: E731
        json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    )
    return {
        "comparison/data_only.pt": checkpoint,
        "comparison/physics_informed.pt": checkpoint,
        "comparison/model_config.json": model_config_bytes,
        "comparison/normalization.json": normalization_bytes,
        "comparison/provenance.json": encode(comparison),
        "dataset/in_distribution/manifest.json": encode(manifests["in_distribution"]),
        "dataset/out_of_distribution/manifest.json": encode(
            manifests["out_of_distribution"]
        ),
    }


def _execution_with_evidence(
    mode: BenchmarkMode = "quick",
) -> tuple[BenchmarkExecution, dict[str, bytes]]:
    execution = _execution(mode)
    evidence = _evidence(mode)
    cells = []
    for cell in execution.matrix.cells:
        provenance = cell.model_provenance
        if provenance is not None:
            paths = (
                f"comparison/{provenance.model_variant}.pt",
                "comparison/model_config.json",
                "comparison/normalization.json",
                "comparison/provenance.json",
                "dataset/in_distribution/manifest.json",
                "dataset/out_of_distribution/manifest.json",
            )
            references = tuple(
                ArtifactReference(path, hashlib.sha256(evidence[path]).hexdigest())
                for path in paths
            )
            provenance = replace(
                provenance,
                checkpoint=references[0],
                model_config=references[1],
                normalization=references[2],
                comparison_provenance=references[3],
                dataset_manifests=references[4:],
            )
            cell = replace(cell, model_provenance=provenance)
        cells.append(cell)
    return replace(execution, matrix=BenchmarkMatrix(mode, tuple(cells))), evidence


def _save_bundle(
    tmp_path: Path, name: str, *, mode: BenchmarkMode = "quick"
) -> Path:
    execution, evidence = _execution_with_evidence(mode)
    return save_benchmark_execution(
        execution, tmp_path / name, evidence_files=evidence
    )


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
    files = {
        "manifest.json",
        "runs",
        "comparison",
        "dataset",
        "comparison/data_only.pt",
        "comparison/physics_informed.pt",
        "comparison/model_config.json",
        "comparison/normalization.json",
        "comparison/provenance.json",
        "dataset/in_distribution",
        "dataset/in_distribution/manifest.json",
        "dataset/out_of_distribution",
        "dataset/out_of_distribution/manifest.json",
    }
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
    execution, evidence = _execution_with_evidence()
    destination = save_benchmark_execution(
        execution, tmp_path / "bundle", evidence_files=evidence
    )
    assert destination == tmp_path / "bundle"
    assert _inventory(destination) == _expected_files()
    reloaded = load_benchmark_matrix(destination)
    assert reloaded == execution.matrix
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert len(manifest["cells"]) == 24


def test_loader_rejects_missing_extra_or_tampered_evidence(tmp_path: Path) -> None:
    original = _save_bundle(tmp_path, "evidence-original")
    missing = tmp_path / "evidence-missing"
    shutil.copytree(original, missing)
    (missing / "comparison/data_only.pt").unlink()
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_benchmark_matrix(missing)

    extra = tmp_path / "evidence-extra"
    shutil.copytree(original, extra)
    (extra / "comparison/unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or unexpected"):
        load_benchmark_matrix(extra)

    tampered = tmp_path / "evidence-tampered"
    shutil.copytree(original, tampered)
    with (tampered / "comparison/normalization.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="evidence artifact digest mismatch"):
        load_benchmark_matrix(tampered)


def test_loader_rejects_semantically_incompatible_evidence_after_hash_update(
    tmp_path: Path,
) -> None:
    bundle = _save_bundle(tmp_path, "evidence-incompatible")
    manifest = _read_manifest(bundle)
    evidence = manifest["evidence"]
    assert isinstance(evidence, dict)
    config_path = bundle / "comparison/model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["input_dim"] = 16
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    evidence["comparison/model_config.json"] = digest
    cells = manifest["cells"]
    assert isinstance(cells, list)
    for cell in cells:
        assert isinstance(cell, dict)
        provenance = cell["model_provenance"]
        if isinstance(provenance, dict):
            reference = provenance["model_config"]
            assert isinstance(reference, dict)
            reference["sha256"] = digest
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="incompatible with 12 cells"):
        load_benchmark_matrix(bundle)


def test_save_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        execution, evidence = _execution_with_evidence()
        save_benchmark_execution(execution, destination, evidence_files=evidence)
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
    execution, evidence = _execution_with_evidence()
    with pytest.raises(OSError, match="injected"):
        save_benchmark_execution(
            execution, tmp_path / "failed", evidence_files=evidence
        )
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


@pytest.mark.parametrize("path", ["../outside.pt", "C:/outside.pt", "comparison\\x.pt"])
def test_loader_rejects_noncanonical_evidence_references(
    tmp_path: Path, path: str
) -> None:
    bundle = _save_bundle(tmp_path, "bundle-evidence-path")
    manifest = _read_manifest(bundle)
    cells = manifest["cells"]
    assert isinstance(cells, list) and isinstance(cells[2], dict)
    provenance = cells[2]["model_provenance"]
    assert isinstance(provenance, dict)
    checkpoint = provenance["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["path"] = path
    _write_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="canonical bundle paths"):
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
