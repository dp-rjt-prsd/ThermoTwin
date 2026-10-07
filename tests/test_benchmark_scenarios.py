from __future__ import annotations

import copy
import json
import re
import shutil
from collections.abc import Mapping, MutableMapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
import yaml

from thermotwin.core.types import EnvironmentConfig
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    BenchmarkMode,
    ScenarioId,
)
from thermotwin.experiments.types import ExperimentScenario
from thermotwin.scenarios.benchmark import (
    BenchmarkScenarioCase,
    load_benchmark_scenarios,
)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "configs" / "benchmark.yaml"
PACK = ROOT / "configs" / "pack.yaml"
SIMULATION = ROOT / "configs" / "simulation.yaml"


def load(
    mode: object = "quick",
    seed: object = 17,
    *,
    paths: tuple[Path, Path, Path] | None = None,
) -> Mapping[ScenarioId, BenchmarkScenarioCase]:
    benchmark_path, pack_path, simulation_path = paths or (BENCHMARK, PACK, SIMULATION)
    return load_benchmark_scenarios(
        benchmark_path,
        pack_path,
        simulation_path,
        mode=cast(BenchmarkMode, mode),
        seed=cast(int, seed),
    )


def _copy_configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = (
        tmp_path / "benchmark.yaml",
        tmp_path / "pack.yaml",
        tmp_path / "simulation.yaml",
    )
    for source, target in zip((BENCHMARK, PACK, SIMULATION), paths, strict=True):
        shutil.copyfile(source, target)
    return paths


def _read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_loader_returns_six_immutable_cases_in_canonical_order() -> None:
    cases = load()

    assert isinstance(cases, MappingProxyType)
    assert tuple(cases) == SCENARIO_IDS
    assert all(type(case) is BenchmarkScenarioCase for case in cases.values())
    assert all(type(case.scenario) is ExperimentScenario for case in cases.values())
    assert all(
        type(case.scenario.environment_config) is EnvironmentConfig
        for case in cases.values()
    )
    assert all(
        case.scenario.environment_config.pack.cell_count == 12
        for case in cases.values()
    )
    with pytest.raises(TypeError):
        cast(MutableMapping[ScenarioId, BenchmarkScenarioCase], cases)["normal"] = (
            cases["normal"]
        )
    with pytest.raises((AttributeError, TypeError)):
        cast(MutableMapping[str, object], cases["normal"].scenario.metadata)["mode"] = (
            "full"
        )


def test_modes_seed_determinism_profiles_and_fault_timing() -> None:
    quick_a = load("quick", 17)
    quick_b = load("quick", 999)
    full = load("full", 17)

    for scenario_id in SCENARIO_IDS:
        quick_case = quick_a[scenario_id]
        assert quick_case.scenario.duration_s == 30.0
        assert quick_case.scenario.time_step_s == 1.0
        assert quick_case.scenario.step_count == 30
        assert quick_case.scenario.metadata["mode"] == "quick"
        assert (
            quick_case.configuration_sha256 == quick_b[scenario_id].configuration_sha256
        )
        assert quick_case.scenario.current_profile(0.0) == quick_b[
            scenario_id
        ].scenario.current_profile(0.0)
        assert full[scenario_id].scenario.duration_s == 600.0
        assert full[scenario_id].scenario.step_count == 600
        assert full[scenario_id].scenario.metadata["mode"] == "full"
        assert quick_case.configuration_sha256 != full[scenario_id].configuration_sha256

    assert quick_a["normal"].scenario.current_profile(5.0) == 120.0
    assert quick_a["high_current"].scenario.current_profile(5.0) == 240.0
    assert quick_a["fast_charging"].scenario.current_profile(0.0) == -150.0
    charging = quick_a["fast_charging"].scenario.current_profile
    assert all(charging(t) <= 0.0 for t in (0, 20, 21, 25, 29, 30))
    assert charging(30.0) == 0.0
    assert (
        quick_a["resistance_fault"].scenario.environment_config.resistance_fault
        is not None
    )
    assert (
        quick_a[
            "resistance_fault"
        ].scenario.environment_config.resistance_fault.start_time_s
        == 10.0
    )
    assert (
        full["resistance_fault"].scenario.environment_config.resistance_fault
        is not None
    )
    assert (
        full[
            "resistance_fault"
        ].scenario.environment_config.resistance_fault.start_time_s
        == 120.0
    )


def test_scenarios_differ_only_by_declared_profile_or_physical_override() -> None:
    cases = load()
    normal = cases["normal"].scenario
    normal_environment = normal.environment_config

    for scenario_id in ("high_current", "fast_charging"):
        assert cases[scenario_id].scenario.environment_config == normal_environment

    hot = cases["hot_ambient"].scenario.environment_config
    assert hot.pack == normal_environment.pack
    assert hot.cooling == normal_environment.cooling
    assert hot.resistance_fault is None
    assert hot.simulation == replace(
        normal_environment.simulation, ambient_temperature_k=313.15
    )

    weak = cases["weak_cooling"].scenario.environment_config
    assert weak.pack == normal_environment.pack
    assert weak.simulation == normal_environment.simulation
    assert weak.resistance_fault is None
    assert (
        weak.cooling.minimum_coolant_mass_flow_kg_s
        == normal_environment.cooling.minimum_coolant_mass_flow_kg_s
    )
    assert weak.cooling.maximum_coolant_mass_flow_kg_s == 0.025
    assert (
        weak.cooling.base_heat_transfer_coefficient_w_m2_k
        == normal_environment.cooling.base_heat_transfer_coefficient_w_m2_k
    )
    assert (
        weak.cooling.flow_heat_transfer_gain
        == normal_environment.cooling.flow_heat_transfer_gain
    )
    assert weak.cooling.flow_exponent == normal_environment.cooling.flow_exponent

    fault = cases["resistance_fault"].scenario.environment_config
    assert fault.pack == normal_environment.pack
    assert fault.cooling == normal_environment.cooling
    assert fault.simulation == normal_environment.simulation
    assert fault.resistance_fault is not None
    assert fault.resistance_fault.cell_index == 4
    assert fault.resistance_fault.resistance_multiplier == 2.5

    assert len({case.scenario.thermal_limit_k for case in cases.values()}) == 1
    assert (
        len({case.scenario.pump_power_coefficient_w_s3_kg3 for case in cases.values()})
        == 1
    )
    assert {
        case.scenario.pump_power_coefficient_w_s3_kg3 for case in cases.values()
    } == {100000.0}


def test_configuration_digest_is_canonical_sensitive_to_configuration_not_seed(
    tmp_path: Path,
) -> None:
    paths = _copy_configs(tmp_path)
    first = load("quick", 1, paths=paths)
    second = load("quick", 2, paths=paths)
    assert first["normal"].configuration_sha256 == second["normal"].configuration_sha256

    changed = _read_yaml(paths[0])
    scenarios = changed["scenarios"]
    assert isinstance(scenarios, dict)
    normal_scenario = scenarios["normal"]
    assert isinstance(normal_scenario, dict)
    normal_scenario["peak_current_a"] = 121.0
    _write_yaml(paths[0], changed)
    third = load("quick", 1, paths=paths)
    assert third["normal"].configuration_sha256 != first["normal"].configuration_sha256
    assert (
        third["high_current"].configuration_sha256
        == first["high_current"].configuration_sha256
    )

    # Reordering YAML mappings must not change the effective digest.
    reordered = copy.deepcopy(changed)
    scenarios = reordered["scenarios"]
    assert isinstance(scenarios, dict)
    reordered["scenarios"] = dict(reversed(list(scenarios.items())))
    _write_yaml(paths[0], reordered)
    fourth = load("quick", 1, paths=paths)
    assert fourth["normal"].configuration_sha256 == third["normal"].configuration_sha256
    assert re.fullmatch(r"[0-9a-f]{64}", fourth["normal"].configuration_sha256)

    original_normal = first["normal"].scenario
    assert original_normal.current_profile(5.0) == 120.0
    assert original_normal.metadata["peak_current_a"] == 120.0


@pytest.mark.parametrize("mode", ["smoke", "Quick", "", None])
def test_rejects_invalid_mode(mode: object) -> None:
    with pytest.raises(ValueError, match="mode"):
        load(mode=mode)


@pytest.mark.parametrize("seed", [True, -1, 1.5, "1"])
def test_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        load(seed=seed)


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), 0, -1, "bad"])
def test_rejects_invalid_benchmark_numeric_values(tmp_path: Path, bad: object) -> None:
    paths = _copy_configs(tmp_path)
    data = _read_yaml(paths[0])
    data["quick_duration_s"] = bad
    _write_yaml(paths[0], data)
    with pytest.raises(ValueError):
        load(paths=paths)


def test_rejects_non_3_by_4_pack_and_bad_configuration_boundaries(
    tmp_path: Path,
) -> None:
    paths = _copy_configs(tmp_path)
    pack = _read_yaml(paths[1])
    pack_section = pack["pack"]
    assert isinstance(pack_section, dict)
    pack_section["rows"] = 2
    _write_yaml(paths[1], pack)
    with pytest.raises(ValueError, match="3x4"):
        load(paths=paths)


@pytest.mark.parametrize("config_index,section_name", [(1, "pack"), (2, "simulation")])
def test_rejects_unknown_fields_at_pack_and_simulation_boundaries(
    tmp_path: Path, config_index: int, section_name: str
) -> None:
    paths = _copy_configs(tmp_path)
    data = _read_yaml(paths[config_index])
    section = data[section_name]
    assert isinstance(section, dict)
    section["unexpected"] = 1
    _write_yaml(paths[config_index], data)
    with pytest.raises(ValueError, match="unexpected"):
        load(paths=paths)


def test_rejects_unknown_duplicate_missing_fields_ids_and_unreadable_files(
    tmp_path: Path,
) -> None:
    paths = _copy_configs(tmp_path)
    paths[0].write_text(
        BENCHMARK.read_text(encoding="utf-8") + "\nscenarios: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate YAML"):
        load(paths=paths)

    data = _read_yaml(BENCHMARK)
    data["unexpected"] = 1
    _write_yaml(paths[0], data)
    with pytest.raises(ValueError, match="unexpected"):
        load(paths=paths)

    data = _read_yaml(BENCHMARK)
    scenarios = data["scenarios"]
    assert isinstance(scenarios, dict)
    scenarios["other"] = copy.deepcopy(scenarios["normal"])
    _write_yaml(paths[0], data)
    with pytest.raises(ValueError, match="scenario"):
        load(paths=paths)

    del scenarios["normal"]
    _write_yaml(paths[0], data)
    with pytest.raises(ValueError, match="scenario"):
        load(paths=paths)

    with pytest.raises(FileNotFoundError, match="readable"):
        load(paths=(tmp_path / "missing.yaml", paths[1], paths[2]))


def test_yaml_inputs_reject_boolean_physical_values_and_invalid_faults(
    tmp_path: Path,
) -> None:
    paths = _copy_configs(tmp_path)
    pack = _read_yaml(paths[1])
    pack_data = pack["pack"]
    assert isinstance(pack_data, dict)
    pack_data["cell_mass_kg"] = True
    _write_yaml(paths[1], pack)
    with pytest.raises(ValueError, match="finite|cell_mass"):
        load(paths=paths)

    paths = _copy_configs(tmp_path / "retry")
    benchmark = _read_yaml(paths[0])
    scenario_data = benchmark["scenarios"]
    assert isinstance(scenario_data, dict)
    fault_case = scenario_data["resistance_fault"]
    assert isinstance(fault_case, dict)
    fault_config = fault_case["resistance_fault"]
    assert isinstance(fault_config, dict)
    fault_config["cell_index"] = 12
    _write_yaml(paths[0], benchmark)
    with pytest.raises(ValueError, match="cell_index|fault"):
        load(paths=paths)

    paths = _copy_configs(tmp_path / "late_fault")
    benchmark = _read_yaml(paths[0])
    scenario_data = benchmark["scenarios"]
    assert isinstance(scenario_data, dict)
    fault_case = scenario_data["resistance_fault"]
    assert isinstance(fault_case, dict)
    fault_config = fault_case["resistance_fault"]
    assert isinstance(fault_config, dict)
    fault_config["quick_start_time_s"] = 30.0
    _write_yaml(paths[0], benchmark)
    with pytest.raises(ValueError, match="within the selected benchmark duration"):
        load(paths=paths)


def test_yaml_syntax_and_exact_nested_fields_are_rejected(tmp_path: Path) -> None:
    paths = _copy_configs(tmp_path)
    paths[0].write_text("scenarios: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML"):
        load(paths=paths)

    paths = _copy_configs(tmp_path / "retry")
    data = _read_yaml(paths[0])
    scenario_data = data["scenarios"]
    assert isinstance(scenario_data, dict)
    normal_data = scenario_data["normal"]
    assert isinstance(normal_data, dict)
    normal_data["extra"] = 3
    _write_yaml(paths[0], data)
    with pytest.raises(ValueError, match="field"):
        load(paths=paths)


def test_metadata_values_are_immutable_scalars() -> None:
    case = load()["normal"]
    assert isinstance(case.scenario.metadata, Mapping)
    assert all(
        isinstance(value, (str, int, float, bool)) or value is None
        for value in case.scenario.metadata.values()
    )
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], case.scenario.metadata)[
            "profile_parameters"
        ] = {"peak": 120}
    assert json.dumps(dict(case.scenario.metadata), sort_keys=True)
