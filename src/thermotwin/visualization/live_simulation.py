"""Configuration and factory helpers for the interactive thermal lab."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import yaml

from thermotwin.control import PIDController, ThresholdController
from thermotwin.control.base import Controller
from thermotwin.control.predictive import PredictiveController
from thermotwin.control.predictive_types import (
    PredictiveControllerConfig,
    PredictiveObjectiveWeights,
)
from thermotwin.core import (
    CoolingConfig,
    CurrentProfile,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.models.artifacts import ComparisonVariant, load_comparison_forecaster
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.scenarios import (
    aggressive_drive_profile,
    constant_current_profile,
    pulse_current_profile,
    stop_and_go_profile,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_COMPARISON_DIRECTORY = _PROJECT_ROOT / "results" / "phase12-final-comparison"

_PROFILE_NAMES = (
    "Normal stop-and-go",
    "Aggressive drive",
    "Sustained load",
    "Pulse load",
)


class LiveConfiguration:
    """Validated project defaults used by the interactive dashboard."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        threshold_boundaries_k: tuple[float, ...],
        threshold_flows_kg_s: tuple[float, ...],
        pid_values: dict[str, float],
        predictive_values: dict[str, Any],
        thermal_limit_k: float,
        pump_power_coefficient_w_s3_kg3: float,
    ) -> None:
        self.environment = environment
        self.threshold_boundaries_k = threshold_boundaries_k
        self.threshold_flows_kg_s = threshold_flows_kg_s
        self.pid_values = pid_values
        self.predictive_values: dict[str, Any] = predictive_values
        self.thermal_limit_k = thermal_limit_k
        self.pump_power_coefficient_w_s3_kg3 = pump_power_coefficient_w_s3_kg3

    @property
    def profile_names(self) -> tuple[str, ...]:
        return _PROFILE_NAMES


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if type(value) is not dict:
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return cast(dict[str, Any], value)


def load_live_configuration() -> LiveConfiguration:
    """Load physical and controller defaults from the repository YAML files."""
    pack_values = _read_yaml(_PROJECT_ROOT / "configs" / "pack.yaml")
    simulation_values = _read_yaml(_PROJECT_ROOT / "configs" / "simulation.yaml")
    controller_values = _read_yaml(_PROJECT_ROOT / "configs" / "controller.yaml")
    simulation = dict(simulation_values["simulation"])
    for key in ("initial_cell_temperatures_k", "initial_cell_resistances_ohm"):
        if simulation[key] is not None:
            simulation[key] = tuple(simulation[key])
    environment = EnvironmentConfig(
        pack=PackConfig(**pack_values["pack"]),
        cooling=CoolingConfig(**pack_values["cooling"]),
        simulation=SimulationConfig(**simulation),
    )
    threshold = controller_values["threshold_controller"]
    threshold_stages = threshold["stages"]
    threshold_flows = tuple(
        float(stage["coolant_mass_flow_kg_s"]) for stage in threshold_stages[1:-1]
    )
    return LiveConfiguration(
        environment,
        tuple(float(value) for value in threshold["temperature_boundaries_k"]),
        threshold_flows,
        {
            key: float(value)
            for key, value in controller_values["pid_controller"].items()
            if key.endswith(("_k", "_kg_s_k", "_kg_s_k_s", "_kg_s_k_per_s"))
            or key
            in {
                "target_temperature_k",
                "minimum_coolant_mass_flow_kg_s",
                "maximum_coolant_mass_flow_kg_s",
            }
        },
        dict(controller_values["predictive_controller"]),
        float(controller_values["thermal_safety_limit_k"]),
        float(controller_values["pump_power_coefficient_w_s3_kg3"]),
    )


def load_detector_settings() -> dict[str, float]:
    """Read the anomaly-detector threshold and fault multiplier used by the runs."""
    values = _read_yaml(_PROJECT_ROOT / "configs" / "diagnostics.yaml")
    fault = values["fault_detection"]
    return {
        "score_threshold": float(fault["detector"]["score_threshold"]),
        "resistance_multiplier": float(fault["resistance_multiplier"]),
    }


def make_environment(
    configuration: LiveConfiguration,
    *,
    ambient_temperature_k: float,
    coolant_inlet_temperature_k: float,
    initial_soc_fraction: float,
    initial_cell_temperature_k: float,
    fault_cell_index: int | None = None,
    fault_start_time_s: float = 0.0,
    resistance_multiplier: float = 1.0,
) -> EnvironmentConfig:
    """Apply user controls to immutable project defaults."""
    simulation = replace(
        configuration.environment.simulation,
        ambient_temperature_k=ambient_temperature_k,
        coolant_inlet_temperature_k=coolant_inlet_temperature_k,
        initial_soc_fraction=initial_soc_fraction,
        initial_cell_temperature_k=initial_cell_temperature_k,
        initial_cell_temperatures_k=None,
    )
    fault = None
    if fault_cell_index is not None:
        if resistance_multiplier <= 1.0:
            raise ValueError("resistance_multiplier must be greater than 1")
        fault = ResistanceFaultConfig(
            cell_index=fault_cell_index,
            start_time_s=fault_start_time_s,
            resistance_multiplier=resistance_multiplier,
        )
    return replace(
        configuration.environment,
        simulation=simulation,
        resistance_fault=fault,
    )


def make_current_profile(
    name: str, *, duration_s: float, peak_current_a: float
) -> CurrentProfile:
    """Create one deterministic current profile for a selected lab preset."""
    if name == "Normal stop-and-go":
        return stop_and_go_profile(peak_current_a, duration_s, duration_s)
    if name == "Aggressive drive":
        return aggressive_drive_profile(peak_current_a, duration_s, duration_s)
    if name == "Sustained load":
        return constant_current_profile(peak_current_a, duration_s)
    if name == "Pulse load":
        pulse_duration = min(10.0, duration_s / 2.0)
        return pulse_current_profile(
            0.15 * peak_current_a,
            peak_current_a,
            duration_s / 4.0,
            pulse_duration,
            duration_s,
        )
    raise ValueError(f"unknown current profile: {name}")


def load_forecaster(
    configuration: LiveConfiguration,
    *,
    model_name: str,
    comparison_directory: str | Path = _DEFAULT_COMPARISON_DIRECTORY,
) -> ThermalForecaster:
    """Load a saved twin variant for interactive forecast/control decisions."""
    variant_value = {
        "Data-only twin": "data_only",
        "Physics-informed twin": "physics_informed",
    }.get(model_name)
    if variant_value is None:
        raise ValueError(f"unknown model name: {model_name}")
    variant = cast(ComparisonVariant, variant_value)
    simulation = configuration.environment.simulation
    return load_comparison_forecaster(
        comparison_directory,
        variant,
        time_step_s=simulation.default_time_step_s,
        minimum_coolant_flow_kg_s=(
            configuration.environment.cooling.minimum_coolant_mass_flow_kg_s
        ),
        maximum_coolant_flow_kg_s=(
            configuration.environment.cooling.maximum_coolant_mass_flow_kg_s
        ),
    )


def make_controller(
    configuration: LiveConfiguration,
    controller_name: str,
    *,
    forecaster: ThermalForecaster | None = None,
) -> Controller:
    """Build a controller using the same settings as the experiment scripts."""
    cooling = configuration.environment.cooling
    if controller_name == "Reactive threshold":
        return ThresholdController(
            cooling,
            configuration.threshold_boundaries_k,
            configuration.threshold_flows_kg_s,
        )
    if controller_name == "Reactive PID":
        values = configuration.pid_values
        return PIDController(
            cooling,
            values["target_temperature_k"],
            values["kp_kg_s_k"],
            values["ki_kg_s_k_s"],
            values["kd_kg_s_k_per_s"],
        )
    if forecaster is None:
        raise ValueError("a forecaster is required for predictive control")
    values = configuration.predictive_values
    weights = cast(dict[str, float], values["objective_weights"])
    predictive_config = PredictiveControllerConfig(
        horizon_steps=int(values["horizon_steps"]),
        candidate_count=int(values["candidate_count"]),
        minimum_coolant_mass_flow_kg_s=float(values["minimum_coolant_mass_flow_kg_s"]),
        maximum_coolant_mass_flow_kg_s=float(values["maximum_coolant_mass_flow_kg_s"]),
        thermal_safety_limit_k=float(values["thermal_safety_limit_k"]),
        pump_power_coefficient_w_s3_kg3=float(
            values["pump_power_coefficient_w_s3_kg3"]
        ),
        objective_weights=PredictiveObjectiveWeights(
            peak_temperature_weight=float(weights["peak_temperature_weight"]),
            temperature_spread_weight=float(weights["temperature_spread_weight"]),
            pump_energy_weight=float(weights["pump_energy_weight"]),
            safety_penalty_weight=float(weights["safety_penalty_weight"]),
        ),
    )
    return PredictiveController(forecaster, predictive_config)


__all__ = [
    "LiveConfiguration",
    "load_detector_settings",
    "load_forecaster",
    "load_live_configuration",
    "make_controller",
    "make_current_profile",
    "make_environment",
]
