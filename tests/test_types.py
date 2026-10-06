from __future__ import annotations

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


def make_pack(**overrides: Any) -> PackConfig:
    values: dict[str, Any] = {
        "rows": 3,
        "columns": 4,
        "cell_mass_kg": 0.8,
        "cell_specific_heat_capacity_j_kg_k": 900.0,
        "nominal_cell_resistance_ohm": 0.015,
        "cell_capacity_ah": 50.0,
        "neighbor_thermal_resistance_k_w": 1.5,
        "ambient_thermal_resistance_k_w": 8.0,
        "cell_cooling_surface_area_m2": 0.02,
    }
    return PackConfig(**(values | overrides))


def make_cooling(**overrides: Any) -> CoolingConfig:
    values: dict[str, Any] = {
        "minimum_coolant_mass_flow_kg_s": 0.0,
        "maximum_coolant_mass_flow_kg_s": 0.1,
        "base_heat_transfer_coefficient_w_m2_k": 5.0,
        "flow_heat_transfer_gain": 500.0,
        "flow_exponent": 0.8,
    }
    return CoolingConfig(**(values | overrides))


def make_simulation(**overrides: Any) -> SimulationConfig:
    values: dict[str, Any] = {
        "default_time_step_s": 1.0,
        "initial_soc_fraction": 0.8,
        "initial_cell_temperature_k": 298.15,
        "ambient_temperature_k": 298.15,
        "coolant_inlet_temperature_k": 293.15,
        "initial_current_a": 0.0,
    }
    return SimulationConfig(**(values | overrides))


def test_contracts_are_frozen_and_pack_exposes_cell_count() -> None:
    action = CoolingAction(0.02)
    pack = make_pack()

    assert pack.cell_count == 12
    with pytest.raises(FrozenInstanceError):
        action.coolant_mass_flow_kg_s = 0.03  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pack.rows = 2  # type: ignore[misc]


@pytest.mark.parametrize("flow", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_cooling_action_rejects_negative_or_non_finite_flow(flow: float) -> None:
    with pytest.raises(ValueError, match="coolant_mass_flow_kg_s"):
        CoolingAction(flow)


def test_cooling_action_accepts_zero_flow() -> None:
    assert CoolingAction(0.0).coolant_mass_flow_kg_s == 0.0


@pytest.mark.parametrize("field", ["rows", "columns"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_pack_rejects_non_positive_or_non_integer_dimensions(
    field: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=field):
        make_pack(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_mass_kg", 0.0),
        ("cell_specific_heat_capacity_j_kg_k", -1.0),
        ("nominal_cell_resistance_ohm", float("nan")),
        ("cell_capacity_ah", float("inf")),
        ("neighbor_thermal_resistance_k_w", 0.0),
        ("ambient_thermal_resistance_k_w", float("-inf")),
        ("cell_cooling_surface_area_m2", -0.1),
    ],
)
def test_pack_rejects_invalid_physical_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_pack(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_coolant_mass_flow_kg_s", -0.01),
        ("maximum_coolant_mass_flow_kg_s", float("nan")),
        ("base_heat_transfer_coefficient_w_m2_k", 0.0),
        ("flow_heat_transfer_gain", float("inf")),
        ("flow_exponent", -1.0),
    ],
)
def test_cooling_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_cooling(**{field: value})


def test_cooling_config_requires_minimum_flow_not_above_maximum() -> None:
    with pytest.raises(ValueError, match="minimum_coolant_mass_flow_kg_s"):
        make_cooling(
            minimum_coolant_mass_flow_kg_s=0.2,
            maximum_coolant_mass_flow_kg_s=0.1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("default_time_step_s", 0.0),
        ("initial_soc_fraction", -0.01),
        ("initial_soc_fraction", 1.01),
        ("initial_cell_temperature_k", 0.0),
        ("ambient_temperature_k", float("nan")),
        ("coolant_inlet_temperature_k", float("inf")),
        ("initial_current_a", float("-inf")),
    ],
)
def test_simulation_config_rejects_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_simulation(**{field: value})


def test_simulation_config_defaults_to_scalar_temperature_initialization() -> None:
    config = make_simulation()

    assert config.initial_cell_temperatures_k is None
    assert config.initial_cell_resistances_ohm is None


def test_simulation_config_normalizes_nonuniform_initial_temperatures() -> None:
    config = make_simulation(initial_cell_temperatures_k=(298, 299.5))

    temperatures = config.initial_cell_temperatures_k
    assert temperatures == (298.0, 299.5)
    assert temperatures is not None
    assert all(type(value) is float for value in temperatures)
    with pytest.raises(FrozenInstanceError):
        config.initial_cell_temperatures_k = (297.0, 298.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    "temperatures",
    [
        [],
        "298.15",
        (),
        (True,),
        ("298.15",),
        (1j,),
        (0.0,),
        (-1.0,),
        (float("nan"),),
        (float("inf"),),
        (-float("inf"),),
    ],
)
def test_simulation_config_rejects_invalid_nonuniform_initial_temperatures(
    temperatures: Any,
) -> None:
    with pytest.raises(ValueError, match="initial_cell_temperatures_k"):
        make_simulation(initial_cell_temperatures_k=temperatures)


def test_simulation_config_normalizes_nonuniform_initial_resistances() -> None:
    config = make_simulation(initial_cell_resistances_ohm=(0.015, np.float64(0.02)))

    resistances = config.initial_cell_resistances_ohm
    assert resistances == (0.015, 0.02)
    assert resistances is not None
    assert all(type(value) is float for value in resistances)
    with pytest.raises(FrozenInstanceError):
        config.initial_cell_resistances_ohm = (0.01, 0.02)  # type: ignore[misc]


@pytest.mark.parametrize(
    "resistances",
    [
        [],
        "0.015",
        (),
        (True,),
        ("0.015",),
        (1j,),
        (0.0,),
        (-0.01,),
        (float("nan"),),
        (float("inf"),),
        (-float("inf"),),
    ],
)
def test_simulation_config_rejects_invalid_nonuniform_initial_resistances(
    resistances: Any,
) -> None:
    with pytest.raises(ValueError, match="initial_cell_resistances_ohm"):
        make_simulation(initial_cell_resistances_ohm=resistances)


def make_state(
    temperatures: Any = None,
    resistances: Any = None,
    **overrides: Any,
) -> BatteryState:
    values: dict[str, Any] = {
        "time_s": 0.0,
        "current_a": 10.0,
        "soc_fraction": 0.8,
        "ambient_temperature_k": 298.15,
        "coolant_inlet_temperature_k": 293.15,
        "coolant_mass_flow_kg_s": 0.05,
        "cell_temperatures_k": (
            [298.15, 299.15] if temperatures is None else temperatures
        ),
        "cell_resistances_ohm": (
            [0.015, 0.016] if resistances is None else resistances
        ),
    }
    return BatteryState(**(values | overrides))


def test_battery_state_copies_inputs_to_read_only_float64_vectors() -> None:
    temperatures = np.array([298.15, 299.15], dtype=np.float32)
    resistances = [0.015, 0.016]
    state = make_state(temperatures, resistances)

    assert state.cell_temperatures_k.dtype == np.float64
    assert state.cell_resistances_ohm.dtype == np.float64
    assert state.cell_temperatures_k.ndim == state.cell_resistances_ohm.ndim == 1
    assert not np.shares_memory(state.cell_temperatures_k, temperatures)
    assert not state.cell_temperatures_k.flags.writeable
    assert not state.cell_resistances_ohm.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        state.cell_temperatures_k.setflags(write=True)
    stored_temperature = state.cell_temperatures_k[0]
    temperatures[0] = 280.0
    assert state.cell_temperatures_k[0] == stored_temperature
    with pytest.raises(ValueError, match="read-only"):
        state.cell_temperatures_k[0] = 280.0


@pytest.mark.parametrize(
    ("temperatures", "resistances"),
    [([], []), ([298.15], [0.015, 0.016]), ([[298.15]], [[0.015]])],
)
def test_battery_state_requires_equal_nonempty_one_dimensional_arrays(
    temperatures: Any, resistances: Any
) -> None:
    with pytest.raises(ValueError, match="cell_temperatures_k|cell_resistances_ohm"):
        make_state(temperatures, resistances)


@pytest.mark.parametrize(
    ("temperatures", "resistances"),
    [
        ([0.0], [0.015]),
        ([float("nan")], [0.015]),
        ([298.15], [0.0]),
        ([298.15], [float("inf")]),
    ],
)
def test_battery_state_requires_positive_finite_cell_values(
    temperatures: Any, resistances: Any
) -> None:
    with pytest.raises(ValueError, match="cell_temperatures_k|cell_resistances_ohm"):
        make_state(temperatures, resistances)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time_s", -0.01),
        ("time_s", float("nan")),
        ("current_a", float("inf")),
        ("soc_fraction", -0.01),
        ("soc_fraction", 1.01),
        ("ambient_temperature_k", 0.0),
        ("coolant_inlet_temperature_k", float("nan")),
        ("coolant_mass_flow_kg_s", -0.01),
    ],
)
def test_battery_state_rejects_invalid_scalar_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        make_state(**{field: value})


def test_environment_config_holds_the_frozen_contracts() -> None:
    environment = EnvironmentConfig(make_pack(), make_cooling(), make_simulation())

    assert environment.pack.cell_count == 12
    with pytest.raises(FrozenInstanceError):
        environment.pack = make_pack(rows=2)  # type: ignore[misc]


def test_pack_and_simulation_yaml_load_into_environment_config() -> None:
    root = Path(__file__).resolve().parents[1]
    pack_data = yaml.safe_load(
        (root / "configs" / "pack.yaml").read_text(encoding="utf-8")
    )
    simulation_data = yaml.safe_load(
        (root / "configs" / "simulation.yaml").read_text(encoding="utf-8")
    )

    config = EnvironmentConfig(
        pack=PackConfig(**pack_data["pack"]),
        cooling=CoolingConfig(**pack_data["cooling"]),
        simulation=SimulationConfig(**simulation_data["simulation"]),
    )

    assert config.pack.cell_count == 12
    assert config.pack.cell_mass_kg == 0.8
    assert config.pack.cell_specific_heat_capacity_j_kg_k == 900.0
    assert config.pack.nominal_cell_resistance_ohm == 0.015
    assert config.pack.cell_capacity_ah == 50.0
    assert config.pack.neighbor_thermal_resistance_k_w == 1.5
    assert config.pack.ambient_thermal_resistance_k_w == 8.0
    assert config.pack.cell_cooling_surface_area_m2 == 0.02
    assert config.cooling.minimum_coolant_mass_flow_kg_s == 0.0
    assert config.cooling.maximum_coolant_mass_flow_kg_s == 0.1
    assert config.cooling.base_heat_transfer_coefficient_w_m2_k == 5.0
    assert config.cooling.flow_heat_transfer_gain == 500.0
    assert config.cooling.flow_exponent == 0.8
    assert config.simulation.default_time_step_s == 1.0
    assert config.simulation.initial_soc_fraction == 0.8
    assert config.simulation.initial_cell_temperature_k == 298.15
    assert config.simulation.ambient_temperature_k == 298.15
    assert config.simulation.coolant_inlet_temperature_k == 293.15
    assert config.simulation.initial_current_a == 0.0
    assert config.simulation.initial_cell_temperatures_k is None
    assert config.simulation.initial_cell_resistances_ohm is None
