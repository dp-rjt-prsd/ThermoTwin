"""Run the configured threshold and PID controllers on one fixed scenario."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from thermotwin.control import PIDController, ThresholdController
from thermotwin.core import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.experiments import (
    ExperimentScenario,
    run_experiment,
    save_experiment_result,
)
from thermotwin.scenarios import stop_and_go_profile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_STEP_COUNT = 20
_NORMAL_STEP_COUNT = 120
_PEAK_DISCHARGE_CURRENT_A = 120.0


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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        contents = yaml.safe_load(stream)
    if not isinstance(contents, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return contents


def _environment_config() -> tuple[EnvironmentConfig, dict[str, Any]]:
    pack_yaml = _load_yaml(_PROJECT_ROOT / "configs" / "pack.yaml")
    simulation_yaml = _load_yaml(_PROJECT_ROOT / "configs" / "simulation.yaml")
    controller_yaml = _load_yaml(_PROJECT_ROOT / "configs" / "controller.yaml")

    pack = PackConfig(**pack_yaml["pack"])
    cooling = CoolingConfig(**pack_yaml["cooling"])
    simulation_values = dict(simulation_yaml["simulation"])
    initial_temperatures = simulation_values["initial_cell_temperatures_k"]
    if initial_temperatures is not None:
        simulation_values["initial_cell_temperatures_k"] = tuple(initial_temperatures)
    simulation = SimulationConfig(**simulation_values)
    environment_config = EnvironmentConfig(pack, cooling, simulation)

    threshold_values = controller_yaml["threshold_controller"]
    pid_values = controller_yaml["pid_controller"]
    if (
        pid_values["minimum_coolant_mass_flow_kg_s"]
        != cooling.minimum_coolant_mass_flow_kg_s
        or pid_values["maximum_coolant_mass_flow_kg_s"]
        != cooling.maximum_coolant_mass_flow_kg_s
    ):
        raise ValueError("PID flow bounds must match configured cooling limits")
    settings = {
        "threshold_boundaries_k": threshold_values["temperature_boundaries_k"],
        "threshold_flows_kg_s": [
            stage["coolant_mass_flow_kg_s"]
            for stage in threshold_values["stages"][1:-1]
        ],
        "pid": pid_values,
        "thermal_limit_k": controller_yaml["thermal_safety_limit_k"],
        "pump_power_coefficient_w_s3_kg3": controller_yaml[
            "pump_power_coefficient_w_s3_kg3"
        ],
    }
    return environment_config, settings


def _build_scenario(
    environment_config: EnvironmentConfig,
    settings: dict[str, Any],
    *,
    smoke_test: bool,
) -> ExperimentScenario:
    time_step_s = environment_config.simulation.default_time_step_s
    step_count = _SMOKE_STEP_COUNT if smoke_test else _NORMAL_STEP_COUNT
    duration_s = step_count * time_step_s
    cycle_duration_s = duration_s if smoke_test else duration_s / 2.0
    profile = stop_and_go_profile(
        peak_discharge_current_a=_PEAK_DISCHARGE_CURRENT_A,
        cycle_duration_s=cycle_duration_s,
        total_duration_s=duration_s,
    )
    return ExperimentScenario(
        name="baseline-stop-and-go",
        environment_config=environment_config,
        current_profile=profile,
        duration_s=duration_s,
        time_step_s=time_step_s,
        thermal_limit_k=settings["thermal_limit_k"],
        pump_power_coefficient_w_s3_kg3=settings[
            "pump_power_coefficient_w_s3_kg3"
        ],
        metadata={
            "current_profile": "stop_and_go",
            "peak_discharge_current_a": _PEAK_DISCHARGE_CURRENT_A,
            "cycle_duration_s": cycle_duration_s,
            "total_duration_s": duration_s,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a quick scenario of at most 30 simulation steps",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="directory in which to create controller run artifacts",
    )
    parser.add_argument("--seed", type=_nonnegative_seed, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run both reactive controllers and print their measured artifacts as JSON."""
    args = _parser().parse_args(argv)
    environment_config, settings = _environment_config()
    scenario = _build_scenario(
        environment_config,
        settings,
        smoke_test=args.smoke_test,
    )
    output_directory = args.output_directory
    if output_directory is None:
        if args.smoke_test:
            output_directory = Path(tempfile.mkdtemp(prefix="thermotwin-baseline-"))
        else:
            output_directory = _PROJECT_ROOT / "results" / "baseline"

    pid_settings = settings["pid"]
    controllers = (
        (
            "threshold-controller",
            ThresholdController(
                environment_config.cooling,
                settings["threshold_boundaries_k"],
                settings["threshold_flows_kg_s"],
            ),
        ),
        (
            "pid-controller",
            PIDController(
                environment_config.cooling,
                pid_settings["target_temperature_k"],
                pid_settings["kp_kg_s_k"],
                pid_settings["ki_kg_s_k_s"],
                pid_settings["kd_kg_s_k_per_s"],
            ),
        ),
    )

    runs: dict[str, dict[str, object]] = {}
    for run_id, controller in controllers:
        result = run_experiment(scenario, controller, args.seed)
        artifact_directory = save_experiment_result(
            result,
            output_directory,
            run_id,
        )
        runs[result.controller_name] = {
            "metrics": asdict(result.metrics),
            "artifact_directory": str(artifact_directory.resolve()),
        }

    print(json.dumps({"runs": runs}, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
