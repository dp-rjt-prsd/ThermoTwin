"""Compare reactive and predictive cooling on one reproducible scenario."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import yaml

from thermotwin.control import PIDController, ThresholdController
from thermotwin.control.predictive import PredictiveController
from thermotwin.control.predictive_types import (
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.core import (
    CoolingAction,
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
from thermotwin.models.artifacts import load_comparison_forecaster
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.scenarios import stop_and_go_profile

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_STEP_COUNT = 20
_NORMAL_STEP_COUNT = 120
_PEAK_DISCHARGE_CURRENT_A = 120.0
_FORECAST_HORIZONS_S = (30, 60, 120)
_DEFAULT_COMPARISON_DIRECTORY = _PROJECT_ROOT / "results" / "model-comparison-1729"


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


def _mapping(value: object, name: str, expected: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")
    return cast(dict[str, Any], value)


def _read_yaml(path: Path, expected: set[str]) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            contents = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not load {path}: {error}") from error
    return _mapping(contents, path.name, expected)


def _load_configuration() -> tuple[EnvironmentConfig, dict[str, Any]]:
    pack_yaml = _read_yaml(_PROJECT_ROOT / "configs" / "pack.yaml", {"pack", "cooling"})
    simulation_yaml = _read_yaml(
        _PROJECT_ROOT / "configs" / "simulation.yaml", {"simulation"}
    )
    controller_yaml = _read_yaml(
        _PROJECT_ROOT / "configs" / "controller.yaml",
        {
            "threshold_controller",
            "pid_controller",
            "predictive_controller",
            "thermal_safety_limit_k",
            "pump_power_coefficient_w_s3_kg3",
        },
    )
    pack = PackConfig(
        **_mapping(
            pack_yaml["pack"],
            "pack",
            {
                "rows",
                "columns",
                "cell_mass_kg",
                "cell_specific_heat_capacity_j_kg_k",
                "nominal_cell_resistance_ohm",
                "cell_capacity_ah",
                "neighbor_thermal_resistance_k_w",
                "ambient_thermal_resistance_k_w",
                "cell_cooling_surface_area_m2",
            },
        )
    )
    cooling = CoolingConfig(
        **_mapping(
            pack_yaml["cooling"],
            "cooling",
            {
                "minimum_coolant_mass_flow_kg_s",
                "maximum_coolant_mass_flow_kg_s",
                "base_heat_transfer_coefficient_w_m2_k",
                "flow_heat_transfer_gain",
                "flow_exponent",
            },
        )
    )
    simulation_values = _mapping(
        simulation_yaml["simulation"],
        "simulation",
        {
            "default_time_step_s",
            "initial_soc_fraction",
            "initial_cell_temperature_k",
            "ambient_temperature_k",
            "coolant_inlet_temperature_k",
            "initial_current_a",
            "initial_cell_temperatures_k",
            "initial_cell_resistances_ohm",
        },
    )
    for key in ("initial_cell_temperatures_k", "initial_cell_resistances_ohm"):
        if simulation_values[key] is not None:
            simulation_values[key] = tuple(simulation_values[key])
    simulation = SimulationConfig(**simulation_values)
    environment = EnvironmentConfig(pack, cooling, simulation)

    threshold = _mapping(
        controller_yaml["threshold_controller"],
        "threshold_controller",
        {"temperature_boundaries_k", "stages"},
    )
    stages = threshold["stages"]
    if type(stages) is not list or len(stages) != 4:
        raise ValueError("threshold_controller.stages must contain four entries")
    flows: list[float] = []
    for index, stage in enumerate(stages):
        stage_values = _mapping(
            stage,
            f"threshold_controller.stages[{index}]",
            {"name", "coolant_mass_flow_kg_s"},
        )
        flows.append(stage_values["coolant_mass_flow_kg_s"])
    if (
        flows[0] != cooling.minimum_coolant_mass_flow_kg_s
        or flows[-1] != cooling.maximum_coolant_mass_flow_kg_s
    ):
        raise ValueError("threshold flow bounds must match configured cooling limits")

    pid_values = _mapping(
        controller_yaml["pid_controller"],
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
    if (
        pid_values["minimum_coolant_mass_flow_kg_s"]
        != cooling.minimum_coolant_mass_flow_kg_s
        or pid_values["maximum_coolant_mass_flow_kg_s"]
        != cooling.maximum_coolant_mass_flow_kg_s
    ):
        raise ValueError("PID flow bounds must match configured cooling limits")

    predictive_values = _mapping(
        controller_yaml["predictive_controller"],
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
    predictive_values = dict(predictive_values)
    predictive_values["objective_weights"] = PredictiveObjectiveWeights(
        **_mapping(
            predictive_values["objective_weights"],
            "predictive_controller.objective_weights",
            {
                "peak_temperature_weight",
                "temperature_spread_weight",
                "pump_energy_weight",
                "safety_penalty_weight",
            },
        )
    )
    predictive = PredictiveControllerConfig(**predictive_values)
    if (
        predictive.minimum_coolant_mass_flow_kg_s
        != cooling.minimum_coolant_mass_flow_kg_s
        or predictive.maximum_coolant_mass_flow_kg_s
        != cooling.maximum_coolant_mass_flow_kg_s
    ):
        raise ValueError("predictive flow bounds must match configured cooling limits")

    return environment, {
        "threshold_boundaries_k": threshold["temperature_boundaries_k"],
        "threshold_flows_kg_s": flows[1:-1],
        "pid": pid_values,
        "predictive": predictive,
        "thermal_limit_k": controller_yaml["thermal_safety_limit_k"],
        "pump_power_coefficient_w_s3_kg3": controller_yaml[
            "pump_power_coefficient_w_s3_kg3"
        ],
    }


def _build_scenario(
    environment: EnvironmentConfig,
    settings: dict[str, Any],
    *,
    smoke_test: bool,
) -> ExperimentScenario:
    time_step_s = environment.simulation.default_time_step_s
    step_count = _SMOKE_STEP_COUNT if smoke_test else _NORMAL_STEP_COUNT
    duration_s = step_count * time_step_s
    cycle_duration_s = duration_s if smoke_test else duration_s / 2.0
    profile = stop_and_go_profile(
        peak_discharge_current_a=_PEAK_DISCHARGE_CURRENT_A,
        cycle_duration_s=cycle_duration_s,
        total_duration_s=duration_s,
    )
    return ExperimentScenario(
        name="predictive-stop-and-go",
        environment_config=environment,
        current_profile=profile,
        duration_s=duration_s,
        time_step_s=time_step_s,
        thermal_limit_k=settings["thermal_limit_k"],
        pump_power_coefficient_w_s3_kg3=settings["pump_power_coefficient_w_s3_kg3"],
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
        help="run 20 steps; train temporary comparison models if needed",
    )
    parser.add_argument("--comparison-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--seed", type=_nonnegative_seed, default=42)
    return parser


def _prepare_comparison_directory(
    supplied: Path | None, *, smoke_test: bool
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if supplied is not None:
        if not supplied.is_dir():
            raise FileNotFoundError(f"comparison directory does not exist: {supplied}")
        return supplied, None
    if not smoke_test:
        if not _DEFAULT_COMPARISON_DIRECTORY.is_dir():
            raise FileNotFoundError(
                "Phase 7 comparison directory is missing; provide "
                "--comparison-directory or run scripts/compare_models.py first"
            )
        return _DEFAULT_COMPARISON_DIRECTORY, None

    temporary = tempfile.TemporaryDirectory(prefix="thermotwin-predictive-models-")
    destination = Path(temporary.name) / "comparison"
    command = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "compare_models.py"),
        "--smoke-test",
        "--output-directory",
        str(destination),
    ]
    try:
        subprocess.run(
            command,
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.cleanup()
        detail = (
            (error.stderr or "").strip() or "comparison training exited unsuccessfully"
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise RuntimeError(
            f"could not train smoke comparison models: {detail}"
        ) from error
    return destination, temporary


def _new_output_directory(path: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"output directory already exists: {path}") from error
    return path.resolve()


def _json_text(value: object) -> str:
    try:
        return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"comparison summary is not valid JSON: {error}") from error


def _forecast_snapshots(
    results: dict[str, Any],
    forecasters: dict[str, ThermalForecaster],
    *,
    rows: int,
    columns: int,
) -> dict[str, object]:
    """Build post-run forecasts using only each run's final measured state."""
    expected_cell_count = rows * columns
    variants: dict[str, object] = {}
    for variant, run_id in (
        ("data_only", "predictive_data_only"),
        ("physics_informed", "predictive_physics_informed"),
    ):
        result = results[run_id]
        forecaster = forecasters[variant]
        source_state = result.records[-1].state
        if source_state.cell_temperatures_k.shape != (expected_cell_count,):
            raise ValueError(
                "measured source state does not match configured pack size"
            )
        if forecaster.cell_count != expected_cell_count:
            raise ValueError(
                "forecaster cell count does not match configured pack size"
            )

        action_flow = float(source_state.coolant_mass_flow_kg_s)
        action = CoolingAction(action_flow)
        predictions: list[dict[str, object]] = []
        for horizon_s in _FORECAST_HORIZONS_S:
            step_count = round(horizon_s / forecaster.time_step_s)
            if step_count <= 0 or not math.isclose(
                horizon_s,
                step_count * forecaster.time_step_s,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"forecast horizon {horizon_s}s is not an exact timestep multiple"
                )
            rollout = forecaster.rollout(
                source_state,
                tuple(action for _ in range(step_count)),
            )
            temperatures = rollout.cell_temperatures_k
            if temperatures.shape != (step_count, expected_cell_count):
                raise ValueError("forecast rollout has an unexpected temperature shape")
            if not all(
                math.isfinite(float(value)) and float(value) > 0.0
                for value in temperatures.flat
            ):
                raise ValueError(
                    "forecast rollout temperatures must be finite and positive"
                )
            predictions.append(
                {
                    "horizon_s": horizon_s,
                    "cell_temperatures_k": [float(value) for value in temperatures[-1]],
                }
            )

        source_temperatures = [
            float(value) for value in source_state.cell_temperatures_k
        ]
        if not all(
            math.isfinite(value) and value > 0.0 for value in source_temperatures
        ):
            raise ValueError("measured source temperatures must be finite and positive")
        variants[variant] = {
            "source": {
                "measured_time_s": float(source_state.time_s),
                "cell_temperatures_k": source_temperatures,
            },
            "action_coolant_mass_flow_kg_s": action_flow,
            "predictions": predictions,
        }
    return {
        "schema_version": 1,
        "pack": {"rows": rows, "columns": columns},
        "horizons_s": list(_FORECAST_HORIZONS_S),
        "variants": variants,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run and persist all four controller variants on identical inputs."""
    args = _parser().parse_args(argv)
    output_directory = args.output_directory
    if output_directory is None:
        output_directory = _PROJECT_ROOT / "results" / f"predictive-{args.seed}"
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")

    environment, settings = _load_configuration()
    comparison_directory, temporary_comparison = _prepare_comparison_directory(
        args.comparison_directory, smoke_test=args.smoke_test
    )
    try:
        scenario = _build_scenario(environment, settings, smoke_test=args.smoke_test)
        predictive_config = settings["predictive"]
        common_forecaster_args = {
            "time_step_s": scenario.time_step_s,
            "minimum_coolant_flow_kg_s": (
                environment.cooling.minimum_coolant_mass_flow_kg_s
            ),
            "maximum_coolant_flow_kg_s": (
                environment.cooling.maximum_coolant_mass_flow_kg_s
            ),
        }
        data_forecaster = load_comparison_forecaster(
            comparison_directory, "data_only", **common_forecaster_args
        )
        physics_forecaster = load_comparison_forecaster(
            comparison_directory, "physics_informed", **common_forecaster_args
        )
        forecasters: dict[str, ThermalForecaster] = {
            "data_only": data_forecaster,
            "physics_informed": physics_forecaster,
        }
        pid = settings["pid"]
        controllers = (
            (
                "threshold",
                "reactive_threshold",
                None,
                ThresholdController(
                    environment.cooling,
                    settings["threshold_boundaries_k"],
                    settings["threshold_flows_kg_s"],
                ),
            ),
            (
                "pid",
                "reactive_pid",
                None,
                PIDController(
                    environment.cooling,
                    pid["target_temperature_k"],
                    pid["kp_kg_s_k"],
                    pid["ki_kg_s_k_s"],
                    pid["kd_kg_s_k_per_s"],
                ),
            ),
            (
                "predictive_data_only",
                "data_only",
                "data_only",
                PredictiveController(data_forecaster, predictive_config),
            ),
            (
                "predictive_physics_informed",
                "physics_informed",
                "physics_informed",
                PredictiveController(physics_forecaster, predictive_config),
            ),
        )

        results = [
            (run_id, variant, run_experiment(scenario, controller, args.seed))
            for run_id, variant, _model_variant, controller in controllers
        ]
        output = _new_output_directory(output_directory)
        runs: dict[str, dict[str, object]] = {}
        for run_id, variant, result in results:
            artifact_directory = save_experiment_result(result, output, run_id)
            runs[run_id] = {
                "variant": variant,
                "metrics": asdict(result.metrics),
                "artifact_path": artifact_directory.relative_to(output).as_posix(),
            }

        result_by_id = {run_id: result for run_id, _variant, result in results}
        snapshot_path = output / "forecast_snapshots.json"
        snapshot_text = _json_text(
            _forecast_snapshots(
                result_by_id,
                forecasters,
                rows=environment.pack.rows,
                columns=environment.pack.columns,
            )
        )
        snapshot_path.write_text(snapshot_text, encoding="utf-8", newline="")

        summary: dict[str, object] = {
            "seed": args.seed,
            "scenario": {
                "name": scenario.name,
                "duration_s": scenario.duration_s,
                "time_step_s": scenario.time_step_s,
                "step_count": scenario.step_count,
                "metadata": dict(scenario.metadata),
            },
            "runs": runs,
            "forecast_snapshots_path": snapshot_path.relative_to(output).as_posix(),
        }
        summary_text = _json_text(summary)
        (output / "comparison_summary.json").write_text(
            summary_text, encoding="utf-8", newline=""
        )
        print(summary_text, end="")
        return 0
    finally:
        if temporary_comparison is not None:
            temporary_comparison.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
