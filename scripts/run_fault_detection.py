"""Run a reproducible resistance-fault and learned residual-detection case."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from thermotwin.control import ThresholdController
from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.diagnostics.anomaly import ThermalAnomalyConfig
from thermotwin.experiments.fault_detection import (
    run_fault_detection_experiment,
    save_fault_detection_result,
)
from thermotwin.experiments.types import ExperimentScenario
from thermotwin.models.artifacts import load_comparison_forecaster
from thermotwin.scenarios.drive_cycles import constant_current_profile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "diagnostics.yaml"


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed must be a nonnegative integer"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be a nonnegative integer")
    return parsed


def _yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not load {path}: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{path.name} must be a YAML mapping")
    return cast(dict[str, Any], value)


def _mapping(value: object, name: str, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(keys))}")
    return cast(dict[str, Any], value)


def _scenario(
    smoke: bool,
) -> tuple[ExperimentScenario, ThermalAnomalyConfig, ThresholdController]:
    root = _mapping(
        _yaml(_CONFIG_PATH).get("fault_detection"),
        "fault_detection",
        {
            "fault_cell_index",
            "fault_start_time_s",
            "resistance_multiplier",
            "duration_s",
            "current_a",
            "smoke_test",
            "detector",
        },
    )
    scenario_keys = {
        "fault_cell_index",
        "fault_start_time_s",
        "resistance_multiplier",
        "duration_s",
        "current_a",
    }
    values = (
        _mapping(root["smoke_test"], "smoke_test", scenario_keys)
        if smoke
        else {key: root[key] for key in scenario_keys}
    )
    detector = ThermalAnomalyConfig(
        **_mapping(
            root["detector"],
            "detector",
            {
                "rolling_window_steps",
                "score_threshold",
                "persistence_steps",
                "epsilon_k",
            },
        )
    )
    pack_yaml = _yaml(_PROJECT_ROOT / "configs" / "pack.yaml")
    simulation_yaml = _yaml(_PROJECT_ROOT / "configs" / "simulation.yaml")
    controller_yaml = _yaml(_PROJECT_ROOT / "configs" / "controller.yaml")
    pack = PackConfig(**pack_yaml["pack"])
    cooling = CoolingConfig(**pack_yaml["cooling"])
    simulation_values = dict(simulation_yaml["simulation"])
    if simulation_values["initial_cell_temperatures_k"] is not None:
        simulation_values["initial_cell_temperatures_k"] = tuple(
            simulation_values["initial_cell_temperatures_k"]
        )
    simulation = SimulationConfig(**simulation_values)
    fault = ResistanceFaultConfig(
        values["fault_cell_index"],
        values["fault_start_time_s"],
        values["resistance_multiplier"],
    )
    environment = EnvironmentConfig(pack, cooling, simulation, fault)
    threshold_values = controller_yaml["threshold_controller"]
    controller = ThresholdController(
        cooling,
        threshold_values["temperature_boundaries_k"],
        [stage["coolant_mass_flow_kg_s"] for stage in threshold_values["stages"][1:-1]],
    )
    duration = float(values["duration_s"])
    scenario = ExperimentScenario(
        name="simulator-owned-resistance-fault",
        environment_config=environment,
        current_profile=constant_current_profile(float(values["current_a"]), duration),
        duration_s=duration,
        time_step_s=simulation.default_time_step_s,
        thermal_limit_k=float(controller_yaml["thermal_safety_limit_k"]),
        pump_power_coefficient_w_s3_kg3=float(
            controller_yaml["pump_power_coefficient_w_s3_kg3"]
        ),
        metadata={
            "current_profile": "constant_current",
            "current_a": float(values["current_a"]),
        },
    )
    return scenario, detector, controller


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--comparison-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--model-variant",
        choices=("data_only", "physics_informed"),
        default="physics_informed",
    )
    parser.add_argument("--seed", type=_seed, default=42)
    return parser


def _comparison_directory(
    args: argparse.Namespace,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if args.comparison_directory is not None:
        return Path(cast(str | Path, args.comparison_directory)), None
    if not args.smoke_test:
        return _PROJECT_ROOT / "results" / "model-comparison-1729", None
    temporary = tempfile.TemporaryDirectory(prefix="thermotwin-fault-comparison-")
    output = Path(temporary.name) / "comparison"
    try:
        subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "compare_models.py"),
                "--smoke-test",
                "--output-directory",
                str(output),
            ],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.cleanup()
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise RuntimeError(f"smoke comparison generation failed: {detail}") from error
    return output, temporary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comparison_temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        scenario, detector_config, controller = _scenario(args.smoke_test)
        output_root = (
            args.output_directory or _PROJECT_ROOT / "results" / "fault-detection"
        )
        if (output_root / "fault-detection").exists():
            raise FileExistsError(
                f"output directory already exists: {output_root / 'fault-detection'}"
            )
        comparison, comparison_temp = _comparison_directory(args)
        forecaster = load_comparison_forecaster(
            comparison,
            args.model_variant,
            time_step_s=scenario.time_step_s,
            minimum_coolant_flow_kg_s=scenario.environment_config.cooling.minimum_coolant_mass_flow_kg_s,
            maximum_coolant_flow_kg_s=scenario.environment_config.cooling.maximum_coolant_mass_flow_kg_s,
        )
        result = run_fault_detection_experiment(
            scenario,
            controller,
            forecaster,
            detector_config,
            args.seed,
            model_variant=args.model_variant,
        )
        output = save_fault_detection_result(result, output_root)
        print(
            json.dumps(
                {
                    "artifact_directory": str(output.resolve()),
                    "model_variant": result.model_variant,
                    "lead_time": asdict(result.lead_time),
                    "peak_temperature_k": result.result.metrics.peak_temperature_k,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        print(f"run_fault_detection: error: {error}", file=sys.stderr)
        return 2
    finally:
        if comparison_temp is not None:
            comparison_temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
