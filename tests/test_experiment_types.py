from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from thermotwin.core import (
    BatteryState,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.diagnostics.metrics import ExperimentMetrics
from thermotwin.experiments.types import (
    ExperimentRecord,
    ExperimentResult,
    ExperimentScenario,
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
    time_s: float, temperatures_k: tuple[float, ...] = (300.0, 301.0)
) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=300.0,
        coolant_inlet_temperature_k=290.0,
        coolant_mass_flow_kg_s=0.0,
        cell_temperatures_k=np.array(temperatures_k),
        cell_resistances_ohm=np.full(len(temperatures_k), 0.01),
    )


def make_record(time_s: float, power_w: float = 1.0) -> ExperimentRecord:
    return ExperimentRecord(state=make_state(time_s), pump_power_w=power_w)


def make_metrics() -> ExperimentMetrics:
    return ExperimentMetrics(301.0, 300.5, 1.0, 2.0, 0, 0.0)


def make_scenario(**overrides: Any) -> ExperimentScenario:
    values: dict[str, Any] = {
        "name": "nominal",
        "environment_config": make_environment_config(),
        "current_profile": lambda _time_s: 0.0,
        "duration_s": 2.0,
        "time_step_s": 1.0,
        "thermal_limit_k": 320.0,
        "pump_power_coefficient_w_s3_kg3": 100.0,
    }
    return ExperimentScenario(**(values | overrides))


def make_result(**overrides: Any) -> ExperimentResult:
    values: dict[str, Any] = {
        "scenario": make_scenario(),
        "controller_name": "threshold",
        "seed": 4,
        "records": (make_record(1.0), make_record(2.0)),
        "metrics": make_metrics(),
    }
    return ExperimentResult(**(values | overrides))


def test_scenario_constructs_and_exposes_deterministic_step_count() -> None:
    scenario = make_scenario(duration_s=0.3, time_step_s=0.1)

    assert scenario.step_count == 3
    assert scenario.name == "nominal"


def test_scenario_accepts_tight_floating_multiple_tolerance() -> None:
    scenario = make_scenario(duration_s=0.3 + 1e-13, time_step_s=0.1)

    assert scenario.step_count == 3


@pytest.mark.parametrize("name", ["", "   ", 12, None])
def test_scenario_rejects_invalid_name(name: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(name=name)


@pytest.mark.parametrize(
    "duration_s", [0.0, -1.0, float("nan"), float("inf"), True, "2"]
)
def test_scenario_rejects_invalid_duration(duration_s: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(duration_s=duration_s)


@pytest.mark.parametrize("time_step_s", [0.0, -1.0, float("nan"), float("inf"), True])
def test_scenario_rejects_invalid_step(time_step_s: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(time_step_s=time_step_s)


@pytest.mark.parametrize(
    "thermal_limit_k", [0.0, -1.0, float("nan"), float("inf"), True]
)
def test_scenario_rejects_invalid_thermal_limit(thermal_limit_k: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(thermal_limit_k=thermal_limit_k)


@pytest.mark.parametrize("coefficient", [-1.0, float("nan"), float("inf"), True])
def test_scenario_rejects_invalid_pump_coefficient(coefficient: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(pump_power_coefficient_w_s3_kg3=coefficient)


@pytest.mark.parametrize("duration_s", [0.0 + 1e-5, 2.1, 1.99])
def test_scenario_requires_exact_whole_number_of_steps(duration_s: float) -> None:
    with pytest.raises(ValueError, match="integer number of time steps"):
        make_scenario(duration_s=duration_s)


def test_scenario_requires_environment_contract_and_callable_profile() -> None:
    with pytest.raises(ValueError, match="EnvironmentConfig"):
        make_scenario(environment_config=object())
    with pytest.raises(ValueError, match="callable"):
        make_scenario(current_profile=0.0)


def test_record_constructs_and_is_frozen() -> None:
    record = make_record(1.0)

    assert record.state.time_s == 1.0
    with pytest.raises(FrozenInstanceError):
        record.pump_power_w = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("power", [-1.0, float("nan"), float("inf"), True, "1"])
def test_record_rejects_invalid_pump_power(power: Any) -> None:
    with pytest.raises(ValueError):
        ExperimentRecord(state=make_state(1.0), pump_power_w=power)


def test_record_rejects_non_battery_state() -> None:
    with pytest.raises(ValueError, match="BatteryState"):
        ExperimentRecord(state=object(), pump_power_w=0.0)  # type: ignore[arg-type]


def test_result_constructs_from_records_and_is_frozen() -> None:
    scenario = make_scenario(duration_s=4.0, time_step_s=2.0)
    result = make_result(
        scenario=scenario,
        records=(make_record(2.0), make_record(4.0)),
    )

    assert result.scenario is scenario
    assert result.scenario.environment_config == make_environment_config()
    assert result.scenario.environment_config is scenario.environment_config
    assert result.scenario.current_profile is scenario.current_profile
    assert result.scenario.duration_s == 4.0
    assert result.scenario_name == result.scenario.name
    assert result.scenario_name == "nominal"
    assert result.time_step_s == 2.0
    assert result.records[0].state.time_s == 2.0
    with pytest.raises(FrozenInstanceError):
        result.seed = 9  # type: ignore[misc]


def test_result_copies_records_to_nonempty_immutable_tuple() -> None:
    records = [make_record(1.0), make_record(2.0)]
    result = make_result(records=records)
    records.clear()

    assert isinstance(result.records, tuple)
    assert len(result.records) == 2
    with pytest.raises(AttributeError):
        result.records.append(make_record(3.0))  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["", "  ", None, 1])
def test_result_rejects_invalid_controller_name(name: Any) -> None:
    with pytest.raises(ValueError):
        make_result(controller_name=name)


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1"])
def test_result_rejects_invalid_seed(seed: Any) -> None:
    with pytest.raises(ValueError):
        make_result(seed=seed)


def test_result_rejects_non_scenario_provenance() -> None:
    with pytest.raises(ValueError):
        make_result(scenario=object())


def test_result_requires_the_scenario_contract_type_exactly() -> None:
    class DerivedScenario(ExperimentScenario):
        pass

    scenario = make_scenario()
    derived = DerivedScenario(
        name=scenario.name,
        environment_config=scenario.environment_config,
        current_profile=scenario.current_profile,
        duration_s=scenario.duration_s,
        time_step_s=scenario.time_step_s,
        thermal_limit_k=scenario.thermal_limit_k,
        pump_power_coefficient_w_s3_kg3=scenario.pump_power_coefficient_w_s3_kg3,
        metadata=scenario.metadata,
    )
    with pytest.raises(ValueError, match="ExperimentScenario"):
        make_result(scenario=derived)


def test_result_record_spacing_uses_scenario_time_step() -> None:
    scenario = make_scenario(duration_s=1.0, time_step_s=0.5)
    with pytest.raises(ValueError, match="time_step_s spacing"):
        make_result(scenario=scenario)


def test_result_rejects_empty_or_invalid_records() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        make_result(records=[])
    with pytest.raises(ValueError, match="ExperimentRecord"):
        make_result(records=[object()])


@pytest.mark.parametrize(
    "records",
    [
        (make_record(1.0), make_record(1.0)),
        (make_record(2.0), make_record(1.0)),
        (make_record(1.0), make_record(2.1)),
    ],
)
def test_result_rejects_bad_record_time_order_or_spacing(
    records: tuple[ExperimentRecord, ...],
) -> None:
    with pytest.raises(ValueError, match="time"):
        make_result(records=records)


def test_result_rejects_inconsistent_cell_count() -> None:
    with pytest.raises(ValueError, match="same cell count"):
        make_result(
            records=(
                make_record(1.0),
                ExperimentRecord(state=make_state(2.0, (300.0,)), pump_power_w=1.0),
            )
        )


def test_result_rejects_invalid_metrics_type() -> None:
    with pytest.raises(ValueError, match="ExperimentMetrics"):
        make_result(metrics=object())


def test_metadata_is_immutable_and_isolated_from_caller_mapping() -> None:
    original: dict[str, str | int | float | bool | None] = {"scenario": "nominal"}
    scenario = make_scenario(metadata=original)
    result = make_result(metadata=original)
    original["scenario"] = "changed"

    assert scenario.metadata["scenario"] == "nominal"
    assert result.metadata["scenario"] == "nominal"
    with pytest.raises(TypeError):
        scenario.metadata["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    "metadata",
    [
        {"": "empty key"},
        {1: "not a string key"},
        {"bad": object()},
        {"bad": float("nan")},
        {"bad": float("inf")},
        [],
    ],
)
def test_metadata_rejects_invalid_mapping_content(metadata: Any) -> None:
    with pytest.raises(ValueError):
        make_scenario(metadata=metadata)


def test_metadata_defaults_to_immutable_empty_mapping() -> None:
    scenario = make_scenario()
    result = make_result()

    assert dict(scenario.metadata) == {}
    assert dict(result.metadata) == {}
    with pytest.raises(TypeError):
        scenario.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        result.metadata["x"] = 1  # type: ignore[index]


def test_contracts_are_slotted() -> None:
    assert not hasattr(make_scenario(), "__dict__")
    assert not hasattr(make_record(1.0), "__dict__")
    assert not hasattr(make_result(), "__dict__")


def test_scenario_profile_annotation_is_callable_contract() -> None:
    scenario = make_scenario(current_profile=lambda _time_s: 1.0)

    assert scenario.current_profile(0.0) == 1.0
