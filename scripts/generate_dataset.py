"""Generate reproducible synthetic simulator transitions for model training."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data import (
    DatasetSamplingConfig,
    generate_dataset,
    sample_dataset_scenario,
    save_dataset_splits,
    split_by_trajectory,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_VERSION = "1.0"
_SMOKE_MAX_STEPS = 10
_NORMAL_TRAJECTORY_COUNT = 100
_BASE_SOURCE_CONFIG_FILES = (
    "configs/pack.yaml",
    "configs/simulation.yaml",
)
_NORMAL_DATASET_CONFIG = "configs/dataset.yaml"
_OOD_DATASET_CONFIG = "configs/dataset_ood.yaml"
_ARTIFACT_FILENAMES = (
    "train.csv",
    "validation.csv",
    "test.csv",
    "manifest.json",
    "generation_config.json",
)


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


def _trajectory_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "trajectory count must be an integer of at least 3"
        ) from error
    if count < 3:
        raise argparse.ArgumentTypeError("trajectory count must be at least 3")
    return count


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        contents = yaml.safe_load(stream)
    if not isinstance(contents, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return contents


def _load_configs(
    *, ood: bool = False
) -> tuple[EnvironmentConfig, DatasetSamplingConfig, tuple[str, ...]]:
    dataset_config_file = _OOD_DATASET_CONFIG if ood else _NORMAL_DATASET_CONFIG
    source_config_files = (*_BASE_SOURCE_CONFIG_FILES, dataset_config_file)
    pack_yaml = _load_yaml(_PROJECT_ROOT / source_config_files[0])
    simulation_yaml = _load_yaml(_PROJECT_ROOT / source_config_files[1])
    dataset_yaml = _load_yaml(_PROJECT_ROOT / source_config_files[2])

    pack_values = dict(pack_yaml["pack"])
    cooling_values = dict(pack_yaml["cooling"])
    simulation_values = dict(simulation_yaml["simulation"])
    for field in ("initial_cell_temperatures_k", "initial_cell_resistances_ohm"):
        if simulation_values[field] is not None:
            simulation_values[field] = tuple(simulation_values[field])

    environment_config = EnvironmentConfig(
        pack=PackConfig(**pack_values),
        cooling=CoolingConfig(**cooling_values),
        simulation=SimulationConfig(**simulation_values),
    )

    sampling_values = dict(dataset_yaml["sampling"])
    for name, value in tuple(sampling_values.items()):
        if name.endswith("_range"):
            sampling_values[name] = tuple(value)
    sampling_values["allowed_profile_names"] = tuple(
        sampling_values["allowed_profile_names"]
    )
    sampling_config = DatasetSamplingConfig(**sampling_values)
    return environment_config, sampling_config, source_config_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"run trajectories of at most {_SMOKE_MAX_STEPS} simulation steps",
    )
    parser.add_argument(
        "--ood",
        action="store_true",
        help="sample from the shifted out-of-distribution conditions",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="directory in which to create dataset artifacts",
    )
    parser.add_argument("--seed", type=_nonnegative_seed, default=42)
    parser.add_argument(
        "--trajectory-count",
        type=_trajectory_count,
        help=(
            "number of independent trajectories (default: 6 in smoke mode, "
            f"{_NORMAL_TRAJECTORY_COUNT} otherwise)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate and persist a trajectory-split dataset, then print its summary."""
    args = _parser().parse_args(argv)
    environment_config, base_sampling_config, source_config_files = _load_configs(
        ood=args.ood
    )
    trajectory_count = args.trajectory_count
    if trajectory_count is None:
        trajectory_count = 6 if args.smoke_test else _NORMAL_TRAJECTORY_COUNT

    sampling_config = base_sampling_config
    if args.smoke_test:
        smoke_duration = min(
            base_sampling_config.duration_s,
            _SMOKE_MAX_STEPS * base_sampling_config.time_step_s,
        )
        sampling_config = replace(base_sampling_config, duration_s=smoke_duration)

    scenarios = tuple(
        sample_dataset_scenario(
            environment_config,
            sampling_config,
            args.seed,
            trajectory_index,
        )
        for trajectory_index in range(trajectory_count)
    )
    records = generate_dataset(scenarios)
    splits = split_by_trajectory(records, 0.2, 0.2, args.seed)

    output_directory = args.output_directory
    if output_directory is None:
        if args.smoke_test:
            temporary_parent = Path(tempfile.mkdtemp(prefix="thermotwin-dataset-"))
            output_directory = temporary_parent / "dataset"
        else:
            output_directory = (
                _PROJECT_ROOT / "data" / "processed" / f"dataset-{args.seed}"
            )

    artifact_directory = save_dataset_splits(splits, output_directory)
    generation_config = {
        "schema_version": _SCHEMA_VERSION,
        "seed": args.seed,
        "trajectory_count": trajectory_count,
        "base_environment_config": asdict(environment_config),
        "sampling_config": asdict(sampling_config),
        "source_config_files": list(source_config_files),
    }
    (artifact_directory / "generation_config.json").write_text(
        json.dumps(generation_config, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )

    split_records = {
        name: getattr(splits, name) for name in ("train", "validation", "test")
    }
    split_summary = {
        name: {
            "row_count": len(part),
            "trajectory_count": len({record.trajectory_id for record in part}),
            "trajectory_ids": sorted({record.trajectory_id for record in part}),
        }
        for name, part in split_records.items()
    }
    summary = {
        "output_directory": str(artifact_directory.resolve()),
        "seed": args.seed,
        "distribution": "out_of_distribution" if args.ood else "in_distribution",
        "trajectory_count": trajectory_count,
        "splits": split_summary,
        "total_row_count": len(records),
        "artifacts": sorted(_ARTIFACT_FILENAMES),
    }
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
