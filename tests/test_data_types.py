from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data.types import (
    DatasetSamplingConfig,
    DatasetScenario,
    TransitionRecord,
)


def make_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=2,
            cell_mass_kg=0.8,
            cell_specific_heat_capacity_j_kg_k=900.0,
            nominal_cell_resistance_ohm=0.015,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.5,
            ambient_thermal_resistance_k_w=8.0,
            cell_cooling_surface_area_m2=0.02,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.0,
            maximum_coolant_mass_flow_kg_s=0.1,
            base_heat_transfer_coefficient_w_m2_k=5.0,
            flow_heat_transfer_gain=500.0,
            flow_exponent=0.8,
        ),
        simulation=SimulationConfig(
            default_time_step_s=1.0,
            initial_soc_fraction=0.5,
            initial_cell_temperature_k=300.0,
            ambient_temperature_k=300.0,
            coolant_inlet_temperature_k=290.0,
            initial_current_a=0.0,
        ),
    )


def make_state(
    time_s: float,
    flow_kg_s: float = 0.02,
    temperatures_k: tuple[float, ...] = (300.0, 301.0),
) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=10.0,
        soc_fraction=0.5,
        ambient_temperature_k=300.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=flow_kg_s,
        cell_temperatures_k=np.array(temperatures_k),
        cell_resistances_ohm=np.full(len(temperatures_k), 0.01),
    )


def make_scenario(**overrides: Any) -> DatasetScenario:
    values: dict[str, Any] = {
        "trajectory_id": "trajectory_01",
        "environment_config": make_environment_config(),
        "current_profile": lambda _time_s: 10.0,
        "coolant_flow_profile": lambda _time_s: 0.02,
        "duration_s": 2.0,
        "time_step_s": 1.0,
    }
    return DatasetScenario(**(values | overrides))


def valid_sampling_values() -> dict[str, Any]:
    return {
        "duration_s": 120.0,
        "time_step_s": 1.0,
        "initial_soc_fraction_range": (0.2, 0.95),
        "initial_cell_temperature_k_range": (285.0, 315.0),
        "ambient_temperature_k_range": (283.0, 313.0),
        "coolant_inlet_temperature_k_range": (285.0, 300.0),
        "peak_current_a_range": (10.0, 250.0),
        "coolant_mass_flow_kg_s_range": (0.0, 0.1),
        "cell_resistance_scale_range": (0.8, 1.8),
        "neighbor_thermal_resistance_scale_range": (0.7, 1.5),
        "cooling_effectiveness_scale_range": (0.7, 1.4),
        "allowed_profile_names": ("constant", "pulse"),
    }


def make_sampling_config(**overrides: Any) -> DatasetSamplingConfig:
    return DatasetSamplingConfig(**(valid_sampling_values() | overrides))


def test_scenario_constructs_and_exposes_step_count() -> None:
    scenario = make_scenario(duration_s=0.3, time_step_s=0.1)

    assert scenario.trajectory_id == "trajectory_01"
    assert scenario.step_count == 3
    assert scenario.current_profile(0.0) == 10.0
    assert scenario.coolant_flow_profile(0.0) == 0.02
    assert dict(scenario.metadata) == {}


@pytest.mark.parametrize(
    "trajectory_id", ["", "_bad", "bad id", "bad/../id", "é", "a" * 65, 1, None]
)
def test_scenario_rejects_invalid_trajectory_id(trajectory_id: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(trajectory_id=trajectory_id)


@pytest.mark.parametrize(
    "duration_s", [0.0, -1.0, float("nan"), float("inf"), True, "2", 2.1]
)
def test_scenario_rejects_invalid_duration(duration_s: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(duration_s=duration_s)


@pytest.mark.parametrize(
    "time_step_s", [0.0, -1.0, float("nan"), float("inf"), True, "1"]
)
def test_scenario_rejects_invalid_timestep(time_step_s: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(time_step_s=time_step_s)


def test_scenario_requires_exact_environment_config_and_callable_profiles() -> None:
    with pytest.raises(ValueError, match="EnvironmentConfig"):
        make_scenario(environment_config=object())
    with pytest.raises(ValueError, match="callable"):
        make_scenario(current_profile=0.0)
    with pytest.raises(ValueError, match="callable"):
        make_scenario(coolant_flow_profile=None)

    class DerivedEnvironmentConfig(EnvironmentConfig):
        pass

    config = make_environment_config()
    derived_config = DerivedEnvironmentConfig(
        pack=config.pack,
        cooling=config.cooling,
        simulation=config.simulation,
    )
    with pytest.raises(ValueError, match="EnvironmentConfig"):
        make_scenario(environment_config=derived_config)


def test_scenario_metadata_is_copied_and_read_only() -> None:
    original = {"scenario": "nominal", "seed": 3}
    scenario = make_scenario(metadata=original)
    original["scenario"] = "mutated"

    assert scenario.metadata["scenario"] == "nominal"
    with pytest.raises(TypeError):
        scenario.metadata["new"] = "blocked"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        scenario.duration_s = 3.0  # type: ignore[misc]


def test_scenario_default_metadata_is_an_independent_immutable_mapping() -> None:
    first = make_scenario()
    second = make_scenario()

    assert first.metadata is not second.metadata
    with pytest.raises(TypeError):
        first.metadata["new"] = "blocked"  # type: ignore[index]


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"": "empty"},
        {1: "bad key"},
        {"x": object()},
        {"x": float("nan")},
        {"x": float("inf")},
    ],
)
def test_metadata_rejects_invalid_input(metadata: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(metadata=metadata)


def test_transition_constructs_and_checks_time_and_action_continuity() -> None:
    action = CoolingAction(0.03)
    record = TransitionRecord(
        trajectory_id="run-1",
        step_index=0,
        time_step_s=1.0,
        state=make_state(4.0),
        action=action,
        next_state=make_state(5.0, flow_kg_s=0.03),
    )

    assert record.step_index == 0
    assert record.state.time_s == 4.0
    assert record.next_state.time_s == 5.0
    assert record.action is action


@pytest.mark.parametrize("step_index", [-1, True, 1.0, "0"])
def test_transition_rejects_invalid_step_index(step_index: Any) -> None:
    with pytest.raises(ValueError):
        TransitionRecord(
            "run-1",
            step_index,
            1.0,
            make_state(0.0),
            CoolingAction(0.02),
            make_state(1.0, 0.02),
        )


def test_transition_rejects_bad_contracts_cell_counts_and_continuity() -> None:
    base = (
        "run-1",
        0,
        1.0,
        make_state(0.0),
        CoolingAction(0.02),
        make_state(1.0, 0.02),
    )
    with pytest.raises(ValueError, match="BatteryState"):
        TransitionRecord(base[0], base[1], base[2], object(), base[4], base[5])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CoolingAction"):
        TransitionRecord(base[0], base[1], base[2], base[3], object(), base[5])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cell counts"):
        TransitionRecord(
            base[0], base[1], base[2], base[3], base[4], make_state(1.0, 0.02, (300.0,))
        )
    with pytest.raises(ValueError, match="time"):
        TransitionRecord(
            base[0], base[1], base[2], base[3], base[4], make_state(1.1, 0.02)
        )
    with pytest.raises(ValueError, match="action"):
        TransitionRecord(
            base[0], base[1], base[2], base[3], base[4], make_state(1.0, 0.03)
        )


def test_transition_rejects_invalid_duration_and_metadata() -> None:
    with pytest.raises(ValueError):
        TransitionRecord(
            "run-1", 0, 0.0, make_state(0.0), CoolingAction(0.02), make_state(1.0, 0.02)
        )
    invalid_metadata: Any = None
    with pytest.raises(ValueError):
        TransitionRecord(
            "run-1",
            0,
            1.0,
            make_state(0.0),
            CoolingAction(0.02),
            make_state(1.0, 0.02),
            metadata=invalid_metadata,
        )


@pytest.mark.parametrize("trajectory_id", ["_invalid", "has space", "x" * 65])
def test_transition_rejects_invalid_trajectory_id(trajectory_id: str) -> None:
    with pytest.raises(ValueError):
        TransitionRecord(
            trajectory_id,
            0,
            1.0,
            make_state(0.0),
            CoolingAction(0.02),
            make_state(1.0, 0.02),
        )


def test_transition_metadata_is_independently_defaulted_and_frozen() -> None:
    args = (
        "run-1",
        0,
        1.0,
        make_state(0.0),
        CoolingAction(0.02),
        make_state(1.0, 0.02),
    )
    first = TransitionRecord(*args)
    second = TransitionRecord(*args)

    assert first.metadata is not second.metadata
    with pytest.raises(TypeError):
        first.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        first.step_index = 1  # type: ignore[misc]


def test_sampling_config_constructs_with_immutable_ranges() -> None:
    config = make_sampling_config()

    assert config.duration_s == 120.0
    assert config.time_step_s == 1.0
    assert config.allowed_profile_names == ("constant", "pulse")
    assert isinstance(config.coolant_mass_flow_kg_s_range, tuple)
    with pytest.raises(FrozenInstanceError):
        config.duration_s = 2.0  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.coolant_mass_flow_kg_s_range[0] = 0.01  # type: ignore[index]


def test_sampling_config_accepts_soc_range_inclusive_endpoints() -> None:
    config = make_sampling_config(initial_soc_fraction_range=(0.0, 1.0))

    assert config.initial_soc_fraction_range == (0.0, 1.0)


@pytest.mark.parametrize(
    "duration_s", [0.0, -1.0, float("nan"), float("inf"), True, 120.5]
)
def test_sampling_config_rejects_duration_and_nonmultiple(duration_s: Any) -> None:
    with pytest.raises(ValueError):
        make_sampling_config(duration_s=duration_s)


@pytest.mark.parametrize("time_step_s", [0.0, -1.0, float("nan"), float("inf"), True])
def test_sampling_config_rejects_invalid_timestep(time_step_s: Any) -> None:
    with pytest.raises(ValueError):
        make_sampling_config(time_step_s=time_step_s)


@pytest.mark.parametrize(
    "field",
    [
        "initial_soc_fraction_range",
        "initial_cell_temperature_k_range",
        "ambient_temperature_k_range",
        "coolant_inlet_temperature_k_range",
        "peak_current_a_range",
        "coolant_mass_flow_kg_s_range",
        "cell_resistance_scale_range",
        "neighbor_thermal_resistance_scale_range",
        "cooling_effectiveness_scale_range",
    ],
)
def test_sampling_config_rejects_malformed_or_descending_ranges(field: str) -> None:
    for invalid in (
        (1.0,),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, float("nan")),
        (0.0, float("inf")),
        [0.0, 1.0],
    ):
        with pytest.raises(ValueError):
            make_sampling_config(**{field: invalid})


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_soc_fraction_range", (-0.1, 0.5)),
        ("initial_soc_fraction_range", (0.0, 1.1)),
        ("initial_cell_temperature_k_range", (0.0, 300.0)),
        ("ambient_temperature_k_range", (-1.0, 300.0)),
        ("coolant_inlet_temperature_k_range", (0.0, 290.0)),
        ("peak_current_a_range", (-1.0, 20.0)),
        ("coolant_mass_flow_kg_s_range", (-0.1, 0.1)),
        ("cell_resistance_scale_range", (0.0, 1.0)),
        ("neighbor_thermal_resistance_scale_range", (0.0, 1.0)),
        ("cooling_effectiveness_scale_range", (0.0, 1.0)),
    ],
)
def test_sampling_config_rejects_out_of_domain_ranges(field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        make_sampling_config(**{field: value})


@pytest.mark.parametrize(
    "profiles", [(), ("unknown",), ("constant", "constant"), [], (1,)]
)
def test_sampling_config_rejects_invalid_profile_names(profiles: Any) -> None:
    with pytest.raises(ValueError):
        make_sampling_config(allowed_profile_names=profiles)


def test_sampling_config_yaml_uses_documented_prototype_ranges() -> None:
    raw = yaml.safe_load(Path("configs/dataset.yaml").read_text(encoding="utf-8"))
    ranges = raw["sampling"]
    values: dict[str, Any] = {
        key: tuple(value) if key.endswith("_range") else value
        for key, value in ranges.items()
    }
    values["allowed_profile_names"] = tuple(values["allowed_profile_names"])
    config = DatasetSamplingConfig(**values)

    assert config.duration_s == 120.0
    assert config.time_step_s == 1.0
    assert config.coolant_mass_flow_kg_s_range == (0.0, 0.1)
    assert set(config.allowed_profile_names) == {
        "constant",
        "pulse",
        "stop_and_go",
        "aggressive",
        "fast_charging",
    }
    assert "prototype assumptions, not manufacturer data" in Path(
        "configs/dataset.yaml"
    ).read_text(encoding="utf-8")


def test_dataset_contracts_do_not_import_experiments_models_or_control() -> None:
    tree = ast.parse(Path("src/thermotwin/data/types.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not any(
        name.startswith(
            ("thermotwin.experiments", "thermotwin.models", "thermotwin.control")
        )
        for name in imports
    )
