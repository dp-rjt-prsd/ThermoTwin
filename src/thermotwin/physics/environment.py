"""Authoritative battery-pack electrothermal simulation environment."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from thermotwin.core import (
    BatteryState,
    CoolingAction,
    CurrentProfile,
    EnvironmentConfig,
    ResistanceFaultConfig,
)
from thermotwin.physics.cell import (
    integrate_temperature_euler,
    joule_heating_w,
    temperature_rate_k_s,
)
from thermotwin.physics.cooling import (
    ambient_heat_exchange_w,
    cooling_heat_removal_w,
)
from thermotwin.physics.electrical import soc_after_coulomb_counting
from thermotwin.physics.pack import neighbor_conduction_w
from thermotwin.physics.topology import pack_edges


def _positive_finite_time_step(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("dt must be a finite positive number")
    try:
        step_s = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("dt must be a finite positive number") from error
    if not math.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("dt must be a finite positive number")
    return step_s


class BatteryEnvironment:
    """Integrate the configured pack model one explicit time step at a time."""

    def __init__(self, current_profile: CurrentProfile) -> None:
        if not callable(current_profile):
            raise ValueError("current_profile must be callable")
        self._current_profile = current_profile
        self._config: EnvironmentConfig | None = None
        self._state: BatteryState | None = None
        self._baseline_cell_resistances_ohm: tuple[float, ...] | None = None

    @property
    def state(self) -> BatteryState:
        """Return the current immutable state, or fail before reset."""
        if self._state is None:
            raise RuntimeError("environment must be reset before accessing state")
        return self._state

    def reset(self, config: EnvironmentConfig) -> BatteryState:
        """Create the initial state for a configured pack."""
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be an EnvironmentConfig")

        fault = config.resistance_fault
        if fault is not None and fault.cell_index >= config.pack.cell_count:
            raise ValueError("resistance_fault.cell_index is outside pack bounds")

        simulation = config.simulation
        initial_temperatures = simulation.initial_cell_temperatures_k
        if initial_temperatures is None:
            temperatures = np.full(
                config.pack.cell_count,
                simulation.initial_cell_temperature_k,
                dtype=np.float64,
            )
        else:
            if len(initial_temperatures) != config.pack.cell_count:
                raise ValueError(
                    "initial_cell_temperatures_k length must match pack.cell_count"
                )
            temperatures = np.asarray(initial_temperatures, dtype=np.float64)

        initial_resistances = simulation.initial_cell_resistances_ohm
        if initial_resistances is None:
            resistances = np.full(
                config.pack.cell_count,
                config.pack.nominal_cell_resistance_ohm,
                dtype=np.float64,
            )
        else:
            if len(initial_resistances) != config.pack.cell_count:
                raise ValueError(
                    "initial_cell_resistances_ohm length must match pack.cell_count"
                )
            resistances = np.asarray(initial_resistances, dtype=np.float64)
        baseline_resistances = tuple(float(value) for value in resistances)
        if fault is not None and fault.start_time_s == 0.0:
            resistances = np.array(baseline_resistances, dtype=np.float64)
            resistances[fault.cell_index] *= fault.resistance_multiplier
        state = BatteryState(
            time_s=0.0,
            current_a=self._current_profile(0.0),
            soc_fraction=simulation.initial_soc_fraction,
            ambient_temperature_k=simulation.ambient_temperature_k,
            coolant_inlet_temperature_k=simulation.coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=(config.cooling.minimum_coolant_mass_flow_kg_s),
            cell_temperatures_k=temperatures,
            cell_resistances_ohm=resistances,
        )
        self._config = config
        self._state = state
        self._baseline_cell_resistances_ohm = baseline_resistances
        return state

    def _resistances_at_time(self, time_s: float) -> np.ndarray:
        """Return schedule-derived resistances without mutating the baseline."""
        if self._config is None or self._baseline_cell_resistances_ohm is None:
            raise RuntimeError("environment must be reset before stepping")

        resistances = np.array(self._baseline_cell_resistances_ohm, dtype=np.float64)
        fault: ResistanceFaultConfig | None = self._config.resistance_fault
        if fault is not None and time_s >= fault.start_time_s:
            resistances[fault.cell_index] *= fault.resistance_multiplier
        return resistances

    def step(self, action: CoolingAction, dt: float) -> BatteryState:
        """Advance temperatures and SOC using the current interval's state."""
        if self._config is None or self._state is None:
            raise RuntimeError("environment must be reset before stepping")
        step_s = _positive_finite_time_step(dt)
        if not isinstance(action, CoolingAction):
            raise TypeError("action must be a CoolingAction")

        config = self._config
        current_state = self._state
        flow = action.coolant_mass_flow_kg_s
        if not (
            config.cooling.minimum_coolant_mass_flow_kg_s
            <= flow
            <= config.cooling.maximum_coolant_mass_flow_kg_s
        ):
            raise ValueError("coolant flow is outside configured limits")

        temperatures = current_state.cell_temperatures_k
        if config.resistance_fault is None:
            interval_resistances = current_state.cell_resistances_ohm
        else:
            interval_resistances = self._resistances_at_time(current_state.time_s)
        generated_heat = joule_heating_w(
            current_state.current_a,
            interval_resistances,
        )
        cooling_heat = cooling_heat_removal_w(
            temperatures,
            current_state.coolant_inlet_temperature_k,
            flow,
            config.pack.cell_cooling_surface_area_m2,
            config.cooling,
        )
        neighbor_heat = neighbor_conduction_w(
            temperatures,
            pack_edges(config.pack),
            config.pack.neighbor_thermal_resistance_k_w,
        )
        ambient_heat = ambient_heat_exchange_w(
            temperatures,
            current_state.ambient_temperature_k,
            config.pack.ambient_thermal_resistance_k_w,
        )
        net_heat = generated_heat - cooling_heat + neighbor_heat + ambient_heat
        rates = temperature_rate_k_s(
            net_heat,
            config.pack.cell_mass_kg,
            config.pack.cell_specific_heat_capacity_j_kg_k,
        )
        next_temperatures = integrate_temperature_euler(
            temperatures,
            rates,
            step_s,
        )
        next_time = current_state.time_s + step_s
        next_resistances = (
            current_state.cell_resistances_ohm
            if config.resistance_fault is None
            else self._resistances_at_time(next_time)
        )
        next_state = BatteryState(
            time_s=next_time,
            current_a=self._current_profile(next_time),
            soc_fraction=soc_after_coulomb_counting(
                current_state.soc_fraction,
                current_state.current_a,
                config.pack.cell_capacity_ah,
                step_s,
            ),
            ambient_temperature_k=current_state.ambient_temperature_k,
            coolant_inlet_temperature_k=current_state.coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=flow,
            cell_temperatures_k=next_temperatures,
            cell_resistances_ohm=next_resistances,
        )
        self._state = next_state
        return next_state
