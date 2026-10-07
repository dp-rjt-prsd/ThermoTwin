"""Strict, deterministic scenario factory for the Phase 12 benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from thermotwin.core.types import (
    CoolingConfig,
    CurrentProfile,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    BenchmarkMode,
    ScenarioId,
)
from thermotwin.experiments.types import ExperimentScenario, MetadataValue
from thermotwin.scenarios.charging import fast_charging_profile
from thermotwin.scenarios.drive_cycles import (
    aggressive_drive_profile,
    stop_and_go_profile,
)

__all__ = ["BenchmarkScenarioCase", "load_benchmark_scenarios"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_BENCHMARK_FIELDS = {
    "schema_version",
    "quick_duration_s",
    "full_duration_s",
    "thermal_limit_k",
    "pump_power_coefficient_w_s3_kg3",
    "scenarios",
}
_SCENARIO_FIELDS = {
    "profile",
    "peak_current_a",
    "cycle_duration_s",
    "ambient_temperature_k",
    "maximum_coolant_mass_flow_kg_s",
    "fast_charge_taper_start_fraction",
    "fast_charge_final_current_fraction",
    "resistance_fault",
}
_PACK_FIELDS = {
    "rows",
    "columns",
    "cell_mass_kg",
    "cell_specific_heat_capacity_j_kg_k",
    "nominal_cell_resistance_ohm",
    "cell_capacity_ah",
    "neighbor_thermal_resistance_k_w",
    "ambient_thermal_resistance_k_w",
    "cell_cooling_surface_area_m2",
}
_COOLING_FIELDS = {
    "minimum_coolant_mass_flow_kg_s",
    "maximum_coolant_mass_flow_kg_s",
    "base_heat_transfer_coefficient_w_m2_k",
    "flow_heat_transfer_gain",
    "flow_exponent",
}
_SIMULATION_FIELDS = {
    "default_time_step_s",
    "initial_soc_fraction",
    "initial_cell_temperature_k",
    "ambient_temperature_k",
    "coolant_inlet_temperature_k",
    "initial_current_a",
    "initial_cell_temperatures_k",
    "initial_cell_resistances_ohm",
}
_FAULT_FIELDS = {
    "cell_index",
    "resistance_multiplier",
    "quick_start_time_s",
    "full_start_time_s",
}


@dataclass(frozen=True, slots=True)
class BenchmarkScenarioCase:
    """One immutable scenario together with its reproducibility identity."""

    scenario_id: ScenarioId
    seed: int
    scenario: ExperimentScenario
    configuration_sha256: str

    def __post_init__(self) -> None:
        if self.scenario_id not in SCENARIO_IDS:
            raise ValueError(f"unknown scenario_id: {self.scenario_id!r}")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        object.__setattr__(self, "seed", int(self.seed))
        if type(self.scenario) is not ExperimentScenario:
            raise ValueError("scenario must be an ExperimentScenario")
        if not isinstance(self.configuration_sha256, str) or not _SHA256_RE.fullmatch(
            self.configuration_sha256
        ):
            raise ValueError("configuration_sha256 must be lowercase SHA-256 hex")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys instead of silently losing one."""


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


def _exact_fields(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(unknown))}")
        raise ValueError(f"{name} has invalid fields ({'; '.join(details)})")
    return cast(dict[str, object], value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_real(name: str, value: object) -> float:
    result = _finite_real(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _read_yaml(path_value: str | Path, name: str) -> dict[str, object]:
    try:
        path = Path(path_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} path must identify a readable file") from error
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} configuration is not a readable file: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.load(file, Loader=_UniqueKeySafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("duplicate YAML"):
            raise ValueError(
                f"{name} configuration has invalid YAML: {error}"
            ) from error
        raise ValueError(
            f"{name} configuration is not readable valid YAML: {path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{name} configuration root must be a YAML mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ValueError(f"{name} configuration keys must be strings")
    return cast(dict[str, object], loaded)


def _read_pack(path: str | Path) -> tuple[PackConfig, CoolingConfig]:
    document = _exact_fields(
        _read_yaml(path, "pack"), {"pack", "cooling"}, "pack configuration"
    )
    pack_values = _exact_fields(document["pack"], _PACK_FIELDS, "pack")
    cooling_values = _exact_fields(document["cooling"], _COOLING_FIELDS, "cooling")
    pack = PackConfig(
        rows=_integer("pack.rows", pack_values["rows"], minimum=1),
        columns=_integer("pack.columns", pack_values["columns"], minimum=1),
        cell_mass_kg=_positive_real("pack.cell_mass_kg", pack_values["cell_mass_kg"]),
        cell_specific_heat_capacity_j_kg_k=_positive_real(
            "pack.cell_specific_heat_capacity_j_kg_k",
            pack_values["cell_specific_heat_capacity_j_kg_k"],
        ),
        nominal_cell_resistance_ohm=_positive_real(
            "pack.nominal_cell_resistance_ohm",
            pack_values["nominal_cell_resistance_ohm"],
        ),
        cell_capacity_ah=_positive_real(
            "pack.cell_capacity_ah", pack_values["cell_capacity_ah"]
        ),
        neighbor_thermal_resistance_k_w=_positive_real(
            "pack.neighbor_thermal_resistance_k_w",
            pack_values["neighbor_thermal_resistance_k_w"],
        ),
        ambient_thermal_resistance_k_w=_positive_real(
            "pack.ambient_thermal_resistance_k_w",
            pack_values["ambient_thermal_resistance_k_w"],
        ),
        cell_cooling_surface_area_m2=_positive_real(
            "pack.cell_cooling_surface_area_m2",
            pack_values["cell_cooling_surface_area_m2"],
        ),
    )
    if (pack.rows, pack.columns) != (3, 4):
        raise ValueError("pack must have exactly 3x4 cells for the benchmark")
    cooling = CoolingConfig(
        minimum_coolant_mass_flow_kg_s=_nonnegative_real(
            "cooling.minimum_coolant_mass_flow_kg_s",
            cooling_values["minimum_coolant_mass_flow_kg_s"],
        ),
        maximum_coolant_mass_flow_kg_s=_nonnegative_real(
            "cooling.maximum_coolant_mass_flow_kg_s",
            cooling_values["maximum_coolant_mass_flow_kg_s"],
        ),
        base_heat_transfer_coefficient_w_m2_k=_positive_real(
            "cooling.base_heat_transfer_coefficient_w_m2_k",
            cooling_values["base_heat_transfer_coefficient_w_m2_k"],
        ),
        flow_heat_transfer_gain=_positive_real(
            "cooling.flow_heat_transfer_gain", cooling_values["flow_heat_transfer_gain"]
        ),
        flow_exponent=_positive_real(
            "cooling.flow_exponent", cooling_values["flow_exponent"]
        ),
    )
    return pack, cooling


def _optional_temperature_vector(
    name: str, value: object, *, positive: bool
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 12:
        raise ValueError(f"{name} must be null or a list of exactly 12 values")
    validator = _positive_real if positive else _finite_real
    return tuple(
        validator(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _read_simulation(path: str | Path) -> SimulationConfig:
    document = _exact_fields(
        _read_yaml(path, "simulation"), {"simulation"}, "simulation configuration"
    )
    values = _exact_fields(document["simulation"], _SIMULATION_FIELDS, "simulation")
    return SimulationConfig(
        default_time_step_s=_positive_real(
            "simulation.default_time_step_s", values["default_time_step_s"]
        ),
        initial_soc_fraction=_finite_real(
            "simulation.initial_soc_fraction", values["initial_soc_fraction"]
        ),
        initial_cell_temperature_k=_positive_real(
            "simulation.initial_cell_temperature_k",
            values["initial_cell_temperature_k"],
        ),
        ambient_temperature_k=_positive_real(
            "simulation.ambient_temperature_k", values["ambient_temperature_k"]
        ),
        coolant_inlet_temperature_k=_positive_real(
            "simulation.coolant_inlet_temperature_k",
            values["coolant_inlet_temperature_k"],
        ),
        initial_current_a=_finite_real(
            "simulation.initial_current_a", values["initial_current_a"]
        ),
        initial_cell_temperatures_k=_optional_temperature_vector(
            "simulation.initial_cell_temperatures_k",
            values["initial_cell_temperatures_k"],
            positive=True,
        ),
        initial_cell_resistances_ohm=_optional_temperature_vector(
            "simulation.initial_cell_resistances_ohm",
            values["initial_cell_resistances_ohm"],
            positive=True,
        ),
    )


def _canonical_environment(environment: EnvironmentConfig) -> dict[str, object]:
    result = asdict(environment)
    return result


def _configuration_digest(
    *,
    scenario_id: ScenarioId,
    mode: BenchmarkMode,
    environment: EnvironmentConfig,
    profile_id: str,
    profile_parameters: Mapping[str, float],
    duration_s: float,
    time_step_s: float,
    thermal_limit_k: float,
    pump_coefficient: float,
) -> str:
    payload = {
        "scenario_id": scenario_id,
        "mode": mode,
        "environment": _canonical_environment(environment),
        "profile_id": profile_id,
        "profile_parameters": dict(profile_parameters),
        "duration_s": duration_s,
        "time_step_s": time_step_s,
        "thermal_limit_k": thermal_limit_k,
        "pump_power_coefficient_w_s3_kg3": pump_coefficient,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _profile_and_metadata(
    scenario_id: ScenarioId,
    config: dict[str, object],
    duration_s: float,
) -> tuple[CurrentProfile, str, dict[str, float], dict[str, MetadataValue]]:
    profile_id = config["profile"]
    if profile_id not in ("stop_and_go", "aggressive_drive", "fast_charging"):
        raise ValueError(f"scenario {scenario_id} has unknown profile: {profile_id!r}")

    peak_current = _positive_real(
        f"scenarios.{scenario_id}.peak_current_a", config["peak_current_a"]
    )
    parameters: dict[str, float] = {"peak_current_a": peak_current}
    metadata: dict[str, MetadataValue] = {"profile_id": profile_id}
    if profile_id == "fast_charging":
        if config["cycle_duration_s"] is not None:
            raise ValueError(
                f"scenarios.{scenario_id}.cycle_duration_s must be null for charging"
            )
        taper_start = _finite_real(
            f"scenarios.{scenario_id}.fast_charge_taper_start_fraction",
            config["fast_charge_taper_start_fraction"],
        )
        final_fraction = _finite_real(
            f"scenarios.{scenario_id}.fast_charge_final_current_fraction",
            config["fast_charge_final_current_fraction"],
        )
        parameters.update(
            {
                "taper_start_fraction": taper_start,
                "final_current_fraction": final_fraction,
                "duration_s": duration_s,
            }
        )
        metadata.update(
            {
                "taper_start_fraction": taper_start,
                "final_current_fraction": final_fraction,
            }
        )
        profile = fast_charging_profile(
            peak_current, duration_s, taper_start, final_fraction
        )
    else:
        if (
            config["fast_charge_taper_start_fraction"] is not None
            or config["fast_charge_final_current_fraction"] is not None
        ):
            raise ValueError(
                f"scenario {scenario_id} has charging parameters for a drive profile"
            )
        cycle_duration = _positive_real(
            f"scenarios.{scenario_id}.cycle_duration_s", config["cycle_duration_s"]
        )
        if cycle_duration > duration_s:
            raise ValueError(
                f"scenarios.{scenario_id}.cycle_duration_s must not exceed "
                f"{duration_s} s duration"
            )
        parameters["cycle_duration_s"] = cycle_duration
        metadata["cycle_duration_s"] = cycle_duration
        if profile_id == "stop_and_go":
            profile = stop_and_go_profile(peak_current, cycle_duration, duration_s)
        else:
            profile = aggressive_drive_profile(peak_current, cycle_duration, duration_s)
    metadata["peak_current_a"] = peak_current
    return profile, profile_id, parameters, metadata


def load_benchmark_scenarios(
    benchmark_config_path: str | Path,
    pack_config_path: str | Path,
    simulation_config_path: str | Path,
    *,
    mode: BenchmarkMode,
    seed: int,
) -> Mapping[ScenarioId, BenchmarkScenarioCase]:
    """Load six strict benchmark cases from YAML and freeze their provenance."""
    if mode not in ("quick", "full") or not isinstance(mode, str):
        raise ValueError("mode must be exactly 'quick' or 'full'")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    normalized_seed = int(seed)

    benchmark_document = _exact_fields(
        _read_yaml(benchmark_config_path, "benchmark"),
        _BENCHMARK_FIELDS,
        "benchmark configuration",
    )
    schema_version = _integer(
        "benchmark.schema_version", benchmark_document["schema_version"], minimum=1
    )
    if schema_version != 1:
        raise ValueError("benchmark.schema_version must equal 1")
    durations = {
        "quick": _positive_real(
            "benchmark.quick_duration_s", benchmark_document["quick_duration_s"]
        ),
        "full": _positive_real(
            "benchmark.full_duration_s", benchmark_document["full_duration_s"]
        ),
    }
    if durations != {"quick": 30.0, "full": 600.0}:
        raise ValueError("benchmark durations must be quick=30 s and full=600 s")
    thermal_limit = _positive_real(
        "benchmark.thermal_limit_k", benchmark_document["thermal_limit_k"]
    )
    pump_coefficient = _nonnegative_real(
        "benchmark.pump_power_coefficient_w_s3_kg3",
        benchmark_document["pump_power_coefficient_w_s3_kg3"],
    )
    pack, base_cooling = _read_pack(pack_config_path)
    base_simulation = _read_simulation(simulation_config_path)
    if (
        base_simulation.initial_cell_temperatures_k is not None
        and len(base_simulation.initial_cell_temperatures_k) != pack.cell_count
    ):
        raise ValueError(
            "simulation initial cell temperatures must match the 12-cell pack"
        )
    if (
        base_simulation.initial_cell_resistances_ohm is not None
        and len(base_simulation.initial_cell_resistances_ohm) != pack.cell_count
    ):
        raise ValueError(
            "simulation initial cell resistances must match the 12-cell pack"
        )
    if not math.isclose(base_simulation.default_time_step_s, 1.0):
        raise ValueError(
            "benchmark requires simulation.default_time_step_s to be exactly 1 s"
        )

    raw_scenarios = benchmark_document["scenarios"]
    if not isinstance(raw_scenarios, dict):
        raise ValueError("benchmark.scenarios must be a YAML mapping")
    expected_ids = set(SCENARIO_IDS)
    actual_ids = set(raw_scenarios)
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        unknown = actual_ids - expected_ids
        details = []
        if missing:
            details.append(f"missing scenario IDs: {', '.join(sorted(missing))}")
        if unknown:
            details.append(
                f"unknown scenario IDs: {', '.join(sorted(map(str, unknown)))}"
            )
        raise ValueError(
            "benchmark.scenarios has invalid scenario IDs (" + "; ".join(details) + ")"
        )

    duration = durations[mode]
    cases: dict[ScenarioId, BenchmarkScenarioCase] = {}
    for raw_id in SCENARIO_IDS:
        scenario_id = raw_id
        values = _exact_fields(
            raw_scenarios[scenario_id], _SCENARIO_FIELDS, f"scenarios.{scenario_id}"
        )
        profile, profile_id, profile_parameters, metadata = _profile_and_metadata(
            scenario_id, values, duration
        )
        simulation = base_simulation
        cooling = base_cooling
        fault: ResistanceFaultConfig | None = None
        ambient_override = values["ambient_temperature_k"]
        if ambient_override is not None:
            ambient = _positive_real(
                f"scenarios.{scenario_id}.ambient_temperature_k", ambient_override
            )
            simulation = replace(simulation, ambient_temperature_k=ambient)
            profile_parameters["ambient_temperature_k"] = ambient
        maximum_flow_override = values["maximum_coolant_mass_flow_kg_s"]
        if maximum_flow_override is not None:
            maximum_flow = _nonnegative_real(
                f"scenarios.{scenario_id}.maximum_coolant_mass_flow_kg_s",
                maximum_flow_override,
            )
            if maximum_flow < cooling.minimum_coolant_mass_flow_kg_s:
                raise ValueError(
                    f"scenarios.{scenario_id} maximum coolant flow must be >= "
                    "the configured minimum"
                )
            cooling = replace(cooling, maximum_coolant_mass_flow_kg_s=maximum_flow)
            profile_parameters["maximum_coolant_mass_flow_kg_s"] = maximum_flow

        raw_fault = values["resistance_fault"]
        if raw_fault is not None:
            fault_values = _exact_fields(
                raw_fault, _FAULT_FIELDS, f"scenarios.{scenario_id}.resistance_fault"
            )
            cell_index = _integer(
                f"scenarios.{scenario_id}.resistance_fault.cell_index",
                fault_values["cell_index"],
                minimum=0,
            )
            if cell_index >= pack.cell_count:
                raise ValueError("resistance_fault.cell_index must be in [0, 11]")
            multiplier = _finite_real(
                f"scenarios.{scenario_id}.resistance_fault.resistance_multiplier",
                fault_values["resistance_multiplier"],
            )
            if multiplier <= 1.0:
                raise ValueError("resistance_fault.resistance_multiplier must exceed 1")
            quick_start = _nonnegative_real(
                f"scenarios.{scenario_id}.resistance_fault.quick_start_time_s",
                fault_values["quick_start_time_s"],
            )
            full_start = _nonnegative_real(
                f"scenarios.{scenario_id}.resistance_fault.full_start_time_s",
                fault_values["full_start_time_s"],
            )
            fault_start = quick_start if mode == "quick" else full_start
            if fault_start >= duration:
                raise ValueError(
                    "resistance fault start time must be within the selected "
                    "benchmark duration"
                )
            fault = ResistanceFaultConfig(cell_index, fault_start, multiplier)
            profile_parameters.update(
                {
                    "fault_cell_index": float(cell_index),
                    "fault_resistance_multiplier": multiplier,
                    "fault_start_time_s": fault_start,
                }
            )
            metadata.update(
                {
                    "fault_cell_index": cell_index,
                    "fault_resistance_multiplier": multiplier,
                    "fault_start_time_s": fault_start,
                }
            )
        environment = EnvironmentConfig(
            pack=pack,
            cooling=cooling,
            simulation=simulation,
            resistance_fault=fault,
        )
        metadata["mode"] = mode
        digest = _configuration_digest(
            scenario_id=scenario_id,
            mode=mode,
            environment=environment,
            profile_id=profile_id,
            profile_parameters=profile_parameters,
            duration_s=duration,
            time_step_s=base_simulation.default_time_step_s,
            thermal_limit_k=thermal_limit,
            pump_coefficient=pump_coefficient,
        )
        scenario = ExperimentScenario(
            name=scenario_id,
            environment_config=environment,
            current_profile=profile,
            duration_s=duration,
            time_step_s=base_simulation.default_time_step_s,
            thermal_limit_k=thermal_limit,
            pump_power_coefficient_w_s3_kg3=pump_coefficient,
            metadata=metadata,
        )
        cases[scenario_id] = BenchmarkScenarioCase(
            scenario_id=scenario_id,
            seed=normalized_seed,
            scenario=scenario,
            configuration_sha256=digest,
        )
    return MappingProxyType(cases)
