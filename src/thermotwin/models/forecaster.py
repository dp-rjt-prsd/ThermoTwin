"""Immutable, model-agnostic contracts for thermal forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray

from thermotwin.core.types import BatteryState, CoolingAction
from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.mlp import ThermalMLP


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


def _finite_flow(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_time_step(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("time_step_s must be a finite positive real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("time_step_s must be finite") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    return result


class MLPThermalForecaster:
    """Adapt a normalized one-step ``ThermalMLP`` to physical state forecasts."""

    _model: ThermalMLP
    _normalization: NormalizationStats
    _time_step_s: float
    _minimum_coolant_flow_kg_s: float
    _maximum_coolant_flow_kg_s: float
    _cell_count: int
    _device: torch.device
    _dtype: torch.dtype

    def __init__(
        self,
        model: ThermalMLP,
        normalization: NormalizationStats,
        time_step_s: float,
        minimum_coolant_flow_kg_s: float,
        maximum_coolant_flow_kg_s: float,
    ) -> None:
        if not isinstance(model, ThermalMLP):
            raise TypeError("model must be a ThermalMLP")
        if not isinstance(normalization, NormalizationStats):
            raise TypeError("normalization must be NormalizationStats")
        normalized_step = _positive_time_step(time_step_s)

        minimum = _finite_flow("minimum_coolant_flow_kg_s", minimum_coolant_flow_kg_s)
        maximum = _finite_flow("maximum_coolant_flow_kg_s", maximum_coolant_flow_kg_s)
        if minimum > maximum:
            raise ValueError(
                "minimum coolant flow must not exceed maximum coolant flow"
            )

        metadata = model.architecture_metadata
        cells = metadata.output_dim
        if cells <= 0:
            raise ValueError("model output_dim must be a positive cell count")
        if metadata.input_dim != cells + 5:
            raise ValueError("model input_dim must equal output cells + 5")
        if normalization.feature_names != feature_names(cells):
            raise ValueError(
                "normalization feature_names do not match canonical schema"
            )
        if normalization.target_names != target_names(cells):
            raise ValueError("normalization target_names do not match canonical schema")
        if (
            normalization.feature_mean.shape != (cells + 5,)
            or normalization.feature_scale.shape != (cells + 5,)
            or normalization.target_mean.shape != (cells,)
            or normalization.target_scale.shape != (cells,)
        ):
            raise ValueError(
                "normalization statistic dimensions do not match model schema"
            )

        tensors = tuple(model.parameters()) + tuple(model.buffers())
        if not tensors:
            raise ValueError("model must have at least one parameter or buffer")
        device = tensors[0].device
        dtype = tensors[0].dtype
        if not dtype.is_floating_point:
            raise ValueError("model parameters and buffers must use a floating dtype")
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("model parameters and buffers must use one device")
        if any(
            tensor.dtype != dtype or not tensor.dtype.is_floating_point
            for tensor in tensors
        ):
            raise ValueError("model parameters and buffers must use one floating dtype")

        self._model = model
        self._normalization = normalization
        self._time_step_s = normalized_step
        self._minimum_coolant_flow_kg_s = minimum
        self._maximum_coolant_flow_kg_s = maximum
        self._cell_count = cells
        self._device = device
        self._dtype = dtype

    @property
    def time_step_s(self) -> float:
        return self._time_step_s

    @property
    def cell_count(self) -> int:
        return self._cell_count

    def _validate_state(self, state: BatteryState) -> None:
        if not isinstance(state, BatteryState):
            raise TypeError("state must be a BatteryState")
        if state.cell_temperatures_k.size != self._cell_count:
            raise ValueError(f"state cell count must be {self._cell_count}")

    def _validate_action(self, action: CoolingAction) -> None:
        if not isinstance(action, CoolingAction):
            raise TypeError("action must be a CoolingAction")
        flow = action.coolant_mass_flow_kg_s
        if (
            not self._minimum_coolant_flow_kg_s
            <= flow
            <= self._maximum_coolant_flow_kg_s
        ):
            raise ValueError("action coolant flow is outside configured flow bounds")

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        """Predict a post-action state using only the current state and action."""
        self._validate_state(state)
        self._validate_action(action)

        feature_values = [state.current_a, state.soc_fraction]
        feature_values.extend(float(value) for value in state.cell_temperatures_k)
        feature_values.extend(
            (
                state.ambient_temperature_k,
                state.coolant_inlet_temperature_k,
                action.coolant_mass_flow_kg_s,
            )
        )
        features = torch.tensor(
            [feature_values], device=self._device, dtype=self._dtype
        )
        feature_mean = torch.tensor(
            self._normalization.feature_mean.tolist(),
            device=self._device,
            dtype=self._dtype,
        )
        feature_scale = torch.tensor(
            self._normalization.feature_scale.tolist(),
            device=self._device,
            dtype=self._dtype,
        )
        target_mean = torch.tensor(
            self._normalization.target_mean.tolist(),
            device=self._device,
            dtype=self._dtype,
        )
        target_scale = torch.tensor(
            self._normalization.target_scale.tolist(),
            device=self._device,
            dtype=self._dtype,
        )
        normalized_features = (features - feature_mean) / feature_scale

        modules = tuple(self._model.modules())
        training_modes = tuple(module.training for module in modules)
        try:
            self._model.eval()
            with torch.inference_mode():
                normalized_delta = self._model(normalized_features)
        finally:
            for module, training in zip(modules, training_modes, strict=True):
                module.training = training

        if not isinstance(normalized_delta, torch.Tensor):
            raise TypeError("model output must be a torch.Tensor")
        if normalized_delta.shape != (1, self._cell_count):
            raise ValueError("model output must have shape (1, cell_count)")
        if not normalized_delta.is_floating_point():
            raise TypeError("model output must use a floating dtype")
        if (
            normalized_delta.device != self._device
            or normalized_delta.dtype != self._dtype
        ):
            raise ValueError("model output device and dtype must match the model")
        if not torch.isfinite(normalized_delta).all().item():
            raise ValueError("model output must contain only finite values")

        delta = normalized_delta[0] * target_scale + target_mean
        predicted_temperatures = (
            torch.tensor(
                state.cell_temperatures_k.tolist(),
                device=self._device,
                dtype=self._dtype,
            )
            + delta
        )
        if not torch.isfinite(predicted_temperatures).all().item():
            raise ValueError("predicted temperatures must be finite")
        if torch.any(predicted_temperatures <= 0.0).item():
            raise ValueError("predicted temperatures must be positive")
        result_time = state.time_s + self._time_step_s
        if not math.isfinite(result_time):
            raise ValueError("predicted time must be finite")

        return BatteryState(
            time_s=result_time,
            current_a=state.current_a,
            soc_fraction=state.soc_fraction,
            ambient_temperature_k=state.ambient_temperature_k,
            coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=action.coolant_mass_flow_kg_s,
            cell_temperatures_k=predicted_temperatures.to(
                device="cpu", dtype=torch.float64
            ).numpy(),
            cell_resistances_ohm=state.cell_resistances_ohm,
        )

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: tuple[ForecastConditions, ...] | None = None,
    ) -> ForecastRollout:
        """Predict sequential states, feeding each prediction into the next step."""
        self._validate_state(initial_state)
        if type(actions) is not tuple or not actions:
            raise ValueError("actions must be a nonempty tuple")
        if any(not isinstance(action, CoolingAction) for action in actions):
            raise ValueError("actions must contain CoolingAction values")
        for action in actions:
            self._validate_action(action)
        if future_conditions is not None:
            if type(future_conditions) is not tuple:
                raise ValueError("future_conditions must be a tuple or None")
            if len(future_conditions) != len(actions):
                raise ValueError("future_conditions length must match actions")
            if any(
                not isinstance(item, ForecastConditions) for item in future_conditions
            ):
                raise ValueError("future_conditions must contain ForecastConditions")

        states: list[BatteryState] = []
        current = initial_state
        for index, action in enumerate(actions):
            predicted = self.predict_next(current, action)
            if future_conditions is not None:
                condition = future_conditions[index]
                predicted = BatteryState(
                    time_s=predicted.time_s,
                    current_a=condition.current_a,
                    soc_fraction=condition.soc_fraction,
                    ambient_temperature_k=condition.ambient_temperature_k,
                    coolant_inlet_temperature_k=condition.coolant_inlet_temperature_k,
                    coolant_mass_flow_kg_s=predicted.coolant_mass_flow_kg_s,
                    cell_temperatures_k=predicted.cell_temperatures_k,
                    cell_resistances_ohm=predicted.cell_resistances_ohm,
                )
            states.append(predicted)
            current = predicted
        return ForecastRollout(initial_state, actions, tuple(states))
