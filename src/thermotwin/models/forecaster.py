"""Immutable, model-agnostic contracts for thermal forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from thermotwin.core.types import BatteryState, CoolingAction


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ForecastConditions:
    """Known exogenous conditions associated with a predicted state, in SI units.

    A condition at index ``i`` describes exogenous values on the predicted state
    after action ``i`` and is therefore an input to the next autoregressive step.
    When future conditions are omitted, a forecaster must persist the prior
    state's corresponding values.
    """

    current_a: float
    soc_fraction: float
    ambient_temperature_k: float
    coolant_inlet_temperature_k: float

    def __post_init__(self) -> None:
        current = _finite_scalar("current_a", self.current_a)
        soc = _finite_scalar("soc_fraction", self.soc_fraction)
        ambient = _finite_scalar("ambient_temperature_k", self.ambient_temperature_k)
        coolant = _finite_scalar(
            "coolant_inlet_temperature_k", self.coolant_inlet_temperature_k
        )
        if not 0.0 <= soc <= 1.0:
            raise ValueError("soc_fraction must be in [0, 1]")
        if ambient <= 0.0:
            raise ValueError("ambient_temperature_k must be positive")
        if coolant <= 0.0:
            raise ValueError("coolant_inlet_temperature_k must be positive")
        object.__setattr__(self, "current_a", current)
        object.__setattr__(self, "soc_fraction", soc)
        object.__setattr__(self, "ambient_temperature_k", ambient)
        object.__setattr__(self, "coolant_inlet_temperature_k", coolant)

    @classmethod
    def from_state(cls, state: BatteryState) -> ForecastConditions:
        """Copy exogenous forecast inputs from an existing battery state."""
        if not isinstance(state, BatteryState):
            raise TypeError("state must be a BatteryState")
        return cls(
            current_a=state.current_a,
            soc_fraction=state.soc_fraction,
            ambient_temperature_k=state.ambient_temperature_k,
            coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
        )


@dataclass(frozen=True, slots=True)
class ForecastRollout:
    """A validated sequence of post-step states predicted from one initial state."""

    initial_state: BatteryState
    actions: tuple[CoolingAction, ...]
    states: tuple[BatteryState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_state, BatteryState):
            raise ValueError("initial_state must be a BatteryState")
        if type(self.actions) is not tuple or not self.actions:
            raise ValueError("actions must be a nonempty tuple")
        if any(not isinstance(action, CoolingAction) for action in self.actions):
            raise ValueError("actions must contain CoolingAction values")
        if type(self.states) is not tuple or not self.states:
            raise ValueError("states must be a nonempty tuple")
        if len(self.actions) != len(self.states):
            raise ValueError("actions and states must have equal lengths")
        if any(not isinstance(state, BatteryState) for state in self.states):
            raise ValueError("states must contain BatteryState values")

        cell_count = int(self.initial_state.cell_temperatures_k.size)
        previous_time = float(self.initial_state.time_s)
        step_size: float | None = None
        for index, (action, state) in enumerate(
            zip(self.actions, self.states, strict=True)
        ):
            if state.cell_temperatures_k.size != cell_count:
                raise ValueError("all states must have a consistent cell count")
            if state.time_s <= previous_time:
                raise ValueError("state times must be strictly increasing")
            interval = float(state.time_s - previous_time)
            if step_size is None:
                step_size = interval
            elif not math.isclose(interval, step_size, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("state times must use one constant time step")
            if state.coolant_mass_flow_kg_s != action.coolant_mass_flow_kg_s:
                raise ValueError(
                    f"post-step state {index} coolant flow must equal its action"
                )
            previous_time = float(state.time_s)

    @property
    def time_step_s(self) -> float:
        """The common positive interval between successive states."""
        return float(self.states[0].time_s - self.initial_state.time_s)

    @property
    def step_count(self) -> int:
        return len(self.actions)

    @property
    def cell_count(self) -> int:
        return int(self.initial_state.cell_temperatures_k.size)

    @property
    def cell_temperatures_k(self) -> NDArray[np.float64]:
        """Return a defensive, immutable ``[steps, cells]`` temperature matrix."""
        values = np.stack(
            [state.cell_temperatures_k for state in self.states], axis=0
        ).astype(np.float64, copy=False)
        return np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)


@runtime_checkable
class ThermalForecaster(Protocol):
    """Model-agnostic one-step and autoregressive thermal prediction interface.

    ``future_conditions[i]`` describes the exogenous values on the predicted
    state after ``actions[i]`` and is used on the next autoregressive step.
    If the sequence is omitted, exogenous values persist from the prior state.
    """

    @property
    def time_step_s(self) -> float:
        """Fixed positive integration step used for predictions."""
        ...

    @property
    def cell_count(self) -> int:
        """Number of cells represented by predicted states."""
        ...

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        """Predict the next battery state for one cooling action."""
        ...

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: tuple[ForecastConditions, ...] | None = None,
    ) -> ForecastRollout:
        """Predict post-step states for an immutable sequence of actions."""
        ...
