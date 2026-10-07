"""Stateful service for interactive, authoritative thermal simulations."""

from __future__ import annotations

from thermotwin.control.base import Controller
from thermotwin.core import BatteryState, CoolingAction
from thermotwin.models.forecaster import ThermalForecaster
from thermotwin.physics.cell import joule_heating_w
from thermotwin.physics.cooling import (
    ambient_heat_exchange_w,
    cooling_heat_removal_w,
)
from thermotwin.physics.environment import BatteryEnvironment
from thermotwin.physics.pack import neighbor_conduction_w
from thermotwin.physics.topology import pack_edges
from thermotwin.visualization.live_contracts import (
    SimulationRequest,
    SimulationSnapshot,
)


class SimulationSession:
    """Own one interactive simulation state without coupling to Streamlit."""

    def __init__(
        self,
        request: SimulationRequest,
        controller: Controller,
        forecaster: ThermalForecaster | None = None,
        *,
        forecast_horizon_steps: int = 0,
    ) -> None:
        if not isinstance(request, SimulationRequest):
            raise TypeError("request must be a SimulationRequest")
        if not isinstance(controller, Controller):
            raise TypeError("controller must implement the Controller protocol")
        if forecaster is not None and not isinstance(forecaster, ThermalForecaster):
            raise TypeError("forecaster must satisfy the ThermalForecaster protocol")
        if isinstance(forecast_horizon_steps, bool) or forecast_horizon_steps < 0:
            raise ValueError("forecast_horizon_steps must be a nonnegative integer")
        self._request = request
        self._controller = controller
        self._forecaster = forecaster
        self._forecast_horizon_steps = int(forecast_horizon_steps)
        self._environment = BatteryEnvironment(request.current_profile)
        self._state: BatteryState | None = None
        self._snapshots: list[SimulationSnapshot] = []

    @property
    def request(self) -> SimulationRequest:
        """The immutable request used to configure this session."""
        return self._request

    @property
    def state(self) -> BatteryState | None:
        """The latest simulator state, or ``None`` before ``start``."""
        return self._state

    @property
    def snapshots(self) -> tuple[SimulationSnapshot, ...]:
        """Immutable view of completed post-step snapshots."""
        return tuple(self._snapshots)

    @property
    def complete(self) -> bool:
        """Whether the requested duration has been simulated."""
        return len(self._snapshots) >= self._request.step_count

    def start(self) -> BatteryState:
        """Reset the environment and controller and return the initial state."""
        self._controller.reset()
        self._state = self._environment.reset(self._request.environment_config)
        self._snapshots.clear()
        return self._state

    def reset(self) -> None:
        """Discard the active state and snapshots without starting a new run."""
        self._state = None
        self._snapshots.clear()

    def step(self) -> SimulationSnapshot:
        """Advance the authoritative simulator by one requested time step."""
        if self._state is None:
            raise RuntimeError("simulation session must be started before stepping")
        if self.complete:
            raise RuntimeError("simulation session is already complete")

        action = self._controller.act(self._state)
        if not isinstance(action, CoolingAction):
            raise TypeError("controller.act must return a CoolingAction")
        thermal_terms = self._thermal_terms(self._state, action)
        next_state = self._environment.step(action, self._request.time_step_s)
        forecast_states = self._forecast(next_state, action)
        snapshot = SimulationSnapshot(
            state=next_state,
            control_action=action,
            forecast_states=forecast_states,
            thermal_terms_w=thermal_terms,
            diagnostics={
                "scenario_name": self._request.scenario_name,
                "controller_name": self._request.controller_name,
                "model_name": self._request.model_name,
                "seed": self._request.seed,
                "step_index": len(self._snapshots) + 1,
            },
        )
        self._state = next_state
        self._snapshots.append(snapshot)
        return snapshot

    def _thermal_terms(
        self, state: BatteryState, action: CoolingAction
    ) -> dict[str, float]:
        config = self._request.environment_config
        generated = joule_heating_w(state.current_a, state.cell_resistances_ohm)
        cooling = cooling_heat_removal_w(
            state.cell_temperatures_k,
            state.coolant_inlet_temperature_k,
            action.coolant_mass_flow_kg_s,
            config.pack.cell_cooling_surface_area_m2,
            config.cooling,
        )
        conduction = neighbor_conduction_w(
            state.cell_temperatures_k,
            pack_edges(config.pack),
            config.pack.neighbor_thermal_resistance_k_w,
        )
        ambient = ambient_heat_exchange_w(
            state.cell_temperatures_k,
            state.ambient_temperature_k,
            config.pack.ambient_thermal_resistance_k_w,
        )
        return {
            "Q_gen_total_w": float(generated.sum()),
            "Q_cool_total_w": float(cooling.sum()),
            "Q_neighbor_net_w": float(conduction.sum()),
            "Q_ambient_total_w": float(ambient.sum()),
        }

    def run_to_completion(self) -> tuple[SimulationSnapshot, ...]:
        """Start if necessary and execute all remaining simulation steps."""
        if self._state is None:
            self.start()
        while not self.complete:
            self.step()
        return self.snapshots

    def _forecast(
        self, state: BatteryState, action: CoolingAction
    ) -> tuple[BatteryState, ...]:
        if self._forecaster is None or self._forecast_horizon_steps == 0:
            return ()
        if self._forecaster.time_step_s != self._request.time_step_s:
            raise ValueError("forecaster time_step_s must match request time_step_s")
        actions = (action,) * self._forecast_horizon_steps
        rollout = self._forecaster.rollout(state, actions)
        return rollout.states


__all__ = ["SimulationSession"]
