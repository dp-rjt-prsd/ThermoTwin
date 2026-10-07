"""Run ThermoTwin's deterministic six-scenario, four-controller benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from thermotwin.control.predictive_types import PredictiveObjectiveWeights
from thermotwin.experiments.benchmark import run_benchmark_matrix
from thermotwin.experiments.benchmark_controllers import BenchmarkControllerParameters
from thermotwin.experiments.benchmark_serialization import save_benchmark_execution
from thermotwin.experiments.benchmark_types import (
    ArtifactReference,
    ModelProvenance,
    ModelVariant,
    TrainingMode,
)
from thermotwin.models.artifacts import load_comparison_forecaster
from thermotwin.scenarios.benchmark import load_benchmark_scenarios

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_CONFIG = _PROJECT_ROOT / "configs" / "benchmark.yaml"
_PACK_CONFIG = _PROJECT_ROOT / "configs" / "pack.yaml"
_SIMULATION_CONFIG = _PROJECT_ROOT / "configs" / "simulation.yaml"
_CONTROLLER_CONFIG = _PROJECT_ROOT / "configs" / "controller.yaml"
_EXPECTED_PROVENANCE_FIELDS = {
    "artifact_schema_version",
    "datasets",
    "seed",
    "dataset_seeds",
    "package_versions",
    "config_source",
    "configuration_source_path",
    "normalization",
    "distribution_labels",
    "smoke_test",
}
_DATASET_FIELDS = {"directory", "manifest_path", "manifest", "distribution"}
_MANIFEST_FIELDS = {
    "schema_version",
    "seed",
    "validation_fraction",
    "test_fraction",
    "cell_count",
    "columns",
    "splits",
}


def _nonnegative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be a nonnegative integer"
        ) from error
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a nonnegative integer")
    return seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "run the 30-second workflow check (without --quick, run the "
            "600-second suite)"
        ),
    )
    parser.add_argument("--comparison-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--seed", type=_nonnegative_seed, default=1729)
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not read {label}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value} is prohibited")


def _exact_mapping(value: object, name: str, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")
    return cast(dict[str, Any], value)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys in every mapping."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError(
                "configuration mapping keys must be scalar values"
            ) from error
        if duplicate:
            raise ValueError(f"duplicate YAML field: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _read_yaml(path: Path, label: str, fields: set[str]) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"could not load {label}: {error}") from error
    return _exact_mapping(value, label, fields)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(
            f"could not read comparison artifact {path.name}: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_manifest(value: object, name: str, expected_seed: int) -> dict[str, Any]:
    manifest = _exact_mapping(value, name, _MANIFEST_FIELDS)
    if manifest["schema_version"] != "1.0":
        raise ValueError(f"{name}.schema_version must be '1.0'")
    if type(manifest["seed"]) is not int or manifest["seed"] != expected_seed:
        raise ValueError(f"{name}.seed does not match comparison dataset seed")
    _canonical_number(manifest["validation_fraction"], f"{name}.validation_fraction")
    _canonical_number(manifest["test_fraction"], f"{name}.test_fraction")
    if type(manifest["cell_count"]) is not int or manifest["cell_count"] != 12:
        raise ValueError(f"{name}.cell_count must be 12")
    if (
        type(manifest["columns"]) is not list
        or len(manifest["columns"]) < 13
        or any(type(column) is not str for column in manifest["columns"])
    ):
        raise ValueError(f"{name}.columns is invalid")
    splits = _exact_mapping(
        manifest["splits"], f"{name}.splits", {"train", "validation", "test"}
    )
    for split_name, raw_split in splits.items():
        split = _exact_mapping(
            raw_split,
            f"{name}.splits.{split_name}",
            {"row_count", "trajectory_count", "trajectory_ids"},
        )
        for count_name in ("row_count", "trajectory_count"):
            if type(split[count_name]) is not int or split[count_name] <= 0:
                raise ValueError(f"{name}.splits.{split_name}.{count_name} is invalid")
        identifiers = split["trajectory_ids"]
        if (
            type(identifiers) is not list
            or len(identifiers) != split["trajectory_count"]
            or any(type(identifier) is not str for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError(f"{name}.splits.{split_name}.trajectory_ids is invalid")
    return manifest


def _validate_comparison(
    directory: Path,
    *,
    require_full_training: bool,
    minimum_flow_kg_s: float,
    maximum_flow_kg_s: float,
) -> tuple[
    dict[ModelVariant, Any],
    dict[ModelVariant, ModelProvenance],
    dict[str, bytes],
]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"comparison directory is missing or invalid: {directory}")
    root = directory.resolve()
    artifact_names = (
        "data_only.pt",
        "physics_informed.pt",
        "model_config.json",
        "normalization.json",
        "provenance.json",
    )
    paths = {name: root / name for name in artifact_names}
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"comparison artifact is missing or invalid: {name}")

    provenance_document = _read_json(paths["provenance.json"], "provenance.json")
    provenance_document = _exact_mapping(
        provenance_document, "provenance.json", _EXPECTED_PROVENANCE_FIELDS
    )
    if provenance_document["artifact_schema_version"] != "1.0":
        raise ValueError("provenance artifact_schema_version must be '1.0'")
    if (
        type(provenance_document["config_source"]) is not str
        or not provenance_document["config_source"]
        or type(provenance_document["configuration_source_path"]) is not str
        or not provenance_document["configuration_source_path"]
    ):
        raise ValueError("provenance configuration source fields are invalid")
    package_versions = provenance_document["package_versions"]
    if (
        type(package_versions) is not dict
        or not package_versions
        or any(
            type(name) is not str or type(value) is not str
            for name, value in package_versions.items()
        )
    ):
        raise ValueError("provenance package_versions must map names to strings")
    seed = provenance_document["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("provenance seed must be a nonnegative integer")
    smoke_test = provenance_document["smoke_test"]
    if type(smoke_test) is not bool:
        raise ValueError("provenance smoke_test must be a boolean")
    if require_full_training and smoke_test:
        raise ValueError("full mode rejects smoke-trained comparison artifacts")
    if provenance_document["distribution_labels"] != [
        "in_distribution",
        "out_of_distribution",
    ]:
        raise ValueError("provenance distribution_labels are invalid")
    normalization = _exact_mapping(
        provenance_document["normalization"],
        "provenance normalization",
        {"artifact", "fit_dataset", "reused_for"},
    )
    if normalization != {
        "artifact": "normalization.json",
        "fit_dataset": "in_distribution",
        "reused_for": ["in_distribution", "out_of_distribution"],
    }:
        raise ValueError("provenance normalization source is invalid")
    dataset_seeds = _exact_mapping(
        provenance_document["dataset_seeds"],
        "provenance dataset_seeds",
        {"in_distribution", "out_of_distribution"},
    )
    manifests: dict[str, dict[str, Any]] = {}
    datasets = _exact_mapping(
        provenance_document["datasets"],
        "provenance datasets",
        {"in_distribution", "out_of_distribution"},
    )
    for label in ("in_distribution", "out_of_distribution"):
        expected_fields = _DATASET_FIELDS | (
            {"normalization_source"} if label == "out_of_distribution" else set()
        )
        dataset = _exact_mapping(datasets[label], f"dataset {label}", expected_fields)
        if (
            type(dataset["directory"]) is not str
            or not dataset["directory"]
            or type(dataset["manifest_path"]) is not str
            or not dataset["manifest_path"]
            or dataset["distribution"] != label
        ):
            raise ValueError(f"dataset provenance for {label} is invalid")
        if (
            label == "out_of_distribution"
            and dataset["normalization_source"] != "in_distribution"
        ):
            raise ValueError("OOD normalization source must be in_distribution")
        expected_dataset_seed = dataset_seeds[label]
        if type(expected_dataset_seed) is not int or expected_dataset_seed < 0:
            raise ValueError(f"dataset seed for {label} is invalid")
        manifests[label] = _validate_manifest(
            dataset["manifest"], f"{label} embedded manifest", expected_dataset_seed
        )

    # Normalize the temporary source paths so quick-run provenance remains
    # portable and deterministic when the generated source directory is removed.
    stable_provenance = json.loads(json.dumps(provenance_document))
    for label in ("in_distribution", "out_of_distribution"):
        stable_provenance["datasets"][label]["directory"] = f"dataset/{label}"
        stable_provenance["datasets"][label]["manifest_path"] = (
            f"dataset/{label}/manifest.json"
        )
    evidence_files = {
        f"comparison/{name}": path.read_bytes() for name, path in paths.items()
        if name != "provenance.json"
    }
    evidence_files["comparison/provenance.json"] = _artifact_json_bytes(
        stable_provenance
    )
    for label in ("in_distribution", "out_of_distribution"):
        evidence_files[f"dataset/{label}/manifest.json"] = _artifact_json_bytes(
            manifests[label]
        )
    model_config_reference = ArtifactReference(
        "comparison/model_config.json", _file_digest(paths["model_config.json"])
    )
    normalization_reference = ArtifactReference(
        "comparison/normalization.json",
        _bytes_digest(evidence_files["comparison/normalization.json"]),
    )
    comparison_reference = ArtifactReference(
        "comparison/provenance.json",
        _bytes_digest(evidence_files["comparison/provenance.json"]),
    )
    manifest_references = tuple(
        ArtifactReference(
            f"dataset/{label}/manifest.json",
            _bytes_digest(evidence_files[f"dataset/{label}/manifest.json"]),
        )
        for label in ("in_distribution", "out_of_distribution")
    )

    forecasters: dict[ModelVariant, Any] = {}
    model_provenance: dict[ModelVariant, ModelProvenance] = {}
    training_mode: TrainingMode = "smoke" if smoke_test else "full"
    for variant in ("data_only", "physics_informed"):
        forecasters[variant] = load_comparison_forecaster(
            root,
            variant,
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=minimum_flow_kg_s,
            maximum_coolant_flow_kg_s=maximum_flow_kg_s,
        )
        model_provenance[variant] = ModelProvenance(
            model_variant=variant,
            training_mode=training_mode,
            training_seed=seed,
            checkpoint=ArtifactReference(
                f"comparison/{variant}.pt",
                _bytes_digest(evidence_files[f"comparison/{variant}.pt"]),
            ),
            model_config=model_config_reference,
            normalization=normalization_reference,
            comparison_provenance=comparison_reference,
            dataset_manifests=manifest_references,
        )
    return forecasters, model_provenance, evidence_files


def _load_controller_parameters() -> tuple[BenchmarkControllerParameters, float, float]:
    pack = _read_yaml(_PACK_CONFIG, "pack.yaml", {"pack", "cooling"})
    cooling = _exact_mapping(
        pack["cooling"],
        "pack cooling",
        {
            "minimum_coolant_mass_flow_kg_s",
            "maximum_coolant_mass_flow_kg_s",
            "base_heat_transfer_coefficient_w_m2_k",
            "flow_heat_transfer_gain",
            "flow_exponent",
        },
    )
    minimum = _canonical_number(
        cooling["minimum_coolant_mass_flow_kg_s"], "minimum coolant flow"
    )
    maximum = _canonical_number(
        cooling["maximum_coolant_mass_flow_kg_s"], "maximum coolant flow"
    )
    if maximum <= minimum:
        raise ValueError("pack cooling maximum must exceed its minimum")
    controller = _read_yaml(
        _CONTROLLER_CONFIG,
        "controller.yaml",
        {
            "threshold_controller",
            "pid_controller",
            "predictive_controller",
            "thermal_safety_limit_k",
            "pump_power_coefficient_w_s3_kg3",
        },
    )
    benchmark = _read_yaml(
        _BENCHMARK_CONFIG,
        "benchmark.yaml",
        {
            "schema_version",
            "quick_duration_s",
            "full_duration_s",
            "thermal_limit_k",
            "pump_power_coefficient_w_s3_kg3",
            "scenarios",
        },
    )
    thermal_limit = _canonical_number(
        benchmark["thermal_limit_k"], "benchmark thermal limit"
    )
    pump_coefficient = _canonical_number(
        benchmark["pump_power_coefficient_w_s3_kg3"], "benchmark pump coefficient"
    )
    if (
        _canonical_number(controller["thermal_safety_limit_k"], "controller limit")
        != thermal_limit
    ):
        raise ValueError("controller thermal limit must match benchmark.yaml")
    if (
        _canonical_number(
            controller["pump_power_coefficient_w_s3_kg3"], "controller pump coefficient"
        )
        != pump_coefficient
    ):
        raise ValueError("controller pump coefficient must match benchmark.yaml")

    threshold = _exact_mapping(
        controller["threshold_controller"],
        "threshold_controller",
        {"temperature_boundaries_k", "stages"},
    )
    boundaries = threshold["temperature_boundaries_k"]
    stages = threshold["stages"]
    if type(boundaries) is not list or len(boundaries) != 3:
        raise ValueError("threshold temperature boundaries must contain three values")
    if type(stages) is not list or len(stages) != 4:
        raise ValueError("threshold stages must contain four entries")
    flows: list[float] = []
    stage_names = ("minimum", "low", "medium", "maximum")
    for index, (stage, stage_name) in enumerate(zip(stages, stage_names, strict=True)):
        values = _exact_mapping(
            stage, f"threshold stages[{index}]", {"name", "coolant_mass_flow_kg_s"}
        )
        if values["name"] != stage_name:
            raise ValueError(
                "threshold stage names must be minimum, low, medium, maximum"
            )
        flows.append(
            _canonical_number(values["coolant_mass_flow_kg_s"], "threshold stage flow")
        )
    if (
        flows[0] != minimum
        or flows[-1] != maximum
        or any(left > right for left, right in zip(flows, flows[1:], strict=False))
    ):
        raise ValueError("threshold stage flows must match configured cooling bounds")

    pid = _exact_mapping(
        controller["pid_controller"],
        "pid_controller",
        {
            "target_temperature_k",
            "kp_kg_s_k",
            "ki_kg_s_k_s",
            "kd_kg_s_k_per_s",
            "minimum_coolant_mass_flow_kg_s",
            "maximum_coolant_mass_flow_kg_s",
        },
    )
    predictive = _exact_mapping(
        controller["predictive_controller"],
        "predictive_controller",
        {
            "horizon_steps",
            "candidate_count",
            "minimum_coolant_mass_flow_kg_s",
            "maximum_coolant_mass_flow_kg_s",
            "thermal_safety_limit_k",
            "pump_power_coefficient_w_s3_kg3",
            "objective_weights",
        },
    )
    for name, values in (("PID", pid), ("predictive", predictive)):
        if (
            _canonical_number(
                values["minimum_coolant_mass_flow_kg_s"], f"{name} minimum"
            )
            != minimum
            or _canonical_number(
                values["maximum_coolant_mass_flow_kg_s"], f"{name} maximum"
            )
            != maximum
        ):
            raise ValueError(f"{name} cooling limits must match pack.yaml")
    if (
        _canonical_number(predictive["thermal_safety_limit_k"], "predictive limit")
        != thermal_limit
        or _canonical_number(
            predictive["pump_power_coefficient_w_s3_kg3"], "predictive pump coefficient"
        )
        != pump_coefficient
    ):
        raise ValueError("predictive limits must match benchmark.yaml")
    raw_weights = predictive["objective_weights"]
    if type(raw_weights) is not dict:
        raise ValueError("predictive objective weights must be a mapping")
    raw_weights = dict(raw_weights)
    peak_reference = raw_weights.pop("peak_temperature_reference_k", 0.0)
    weights = PredictiveObjectiveWeights(
        **_exact_mapping(
            raw_weights,
            "predictive objective weights",
            {
                "peak_temperature_weight",
                "temperature_spread_weight",
                "pump_energy_weight",
                "safety_penalty_weight",
            },
        ),
        peak_temperature_reference_k=_canonical_number(
            peak_reference, "predictive peak temperature reference"
        ),
    )
    span = maximum - minimum
    fractions = cast(
        tuple[float, float],
        tuple((flow - minimum) / span for flow in flows[1:-1]),
    )
    parameters = BenchmarkControllerParameters(
        threshold_temperature_boundaries_k=tuple(
            _canonical_number(boundary, "threshold boundary") for boundary in boundaries
        ),  # type: ignore[arg-type]
        threshold_intermediate_flow_fractions=fractions,
        pid_target_temperature_k=_canonical_number(
            pid["target_temperature_k"], "PID target"
        ),
        pid_kp_kg_s_k=_canonical_number(pid["kp_kg_s_k"], "PID kp"),
        pid_ki_kg_s_k_s=_canonical_number(pid["ki_kg_s_k_s"], "PID ki"),
        pid_kd_kg_s_k_per_s=_canonical_number(pid["kd_kg_s_k_per_s"], "PID kd"),
        predictive_horizon_steps=predictive["horizon_steps"],
        predictive_candidate_count=predictive["candidate_count"],
        thermal_safety_limit_k=thermal_limit,
        pump_power_coefficient_w_s3_kg3=pump_coefficient,
        objective_weights=weights,
    )
    return parameters, minimum, maximum


def _train_temporary_comparison(directory: Path) -> None:
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "compare_models.py"),
        "--smoke-test",
        "--output-directory",
        str(directory),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise RuntimeError(f"could not start comparison training: {error}") from error
    if result.returncode != 0:
        detail = (
            result.stderr or result.stdout or "comparison training failed"
        ).strip()
        detail = detail.splitlines()[-1] if detail else "comparison training failed"
        raise RuntimeError(f"comparison training failed: {detail}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "quick" if args.quick else "full"
    output = args.output_directory
    if output is None:
        output = _PROJECT_ROOT / "results" / f"benchmark-{mode}-{args.seed}"
    try:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output directory already exists: {output}")

        temporary: tempfile.TemporaryDirectory[str] | None = None
        comparison = args.comparison_directory
        if comparison is None:
            if mode == "full":
                raise ValueError("full mode requires --comparison-directory")
            temporary = tempfile.TemporaryDirectory(prefix="thermotwin-benchmark-")
            comparison = Path(temporary.name) / "comparison"
        try:
            if args.comparison_directory is None:
                _train_temporary_comparison(comparison)
            parameters, minimum_flow, maximum_flow = _load_controller_parameters()
            forecasters, provenance, evidence_files = _validate_comparison(
                comparison,
                require_full_training=mode == "full",
                minimum_flow_kg_s=minimum_flow,
                maximum_flow_kg_s=maximum_flow,
            )
            cases = load_benchmark_scenarios(
                _BENCHMARK_CONFIG,
                _PACK_CONFIG,
                _SIMULATION_CONFIG,
                mode=cast(Any, mode),
                seed=args.seed,
            )
            execution = run_benchmark_matrix(
                cases,
                parameters,
                forecasters,
                provenance,
                mode=cast(Any, mode),
            )
            saved = save_benchmark_execution(
                execution, output, evidence_files=evidence_files
            )
        finally:
            if temporary is not None:
                temporary.cleanup()

        summary = {
            "output_directory": str(saved.resolve()),
            "mode": mode,
            "scenario_count": 6,
            "variant_count": 4,
            "cell_count": 24,
            "seed": args.seed,
            "complete": True,
            "evidence_status": (
                "workflow_only"
                if mode == "quick"
                else "descriptive_synthetic_benchmark"
            ),
        }
        print(json.dumps(summary, allow_nan=False, sort_keys=True))
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
    ) as error:
        print(f"benchmark: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
