"""Shared controller protocol for simulation and experiment code."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from thermotwin.core import BatteryState, CoolingAction


@runtime_checkable
class Controller(Protocol):
    """Stateful or stateless controller accepted by ThermoTwin experiments."""

    def reset(self) -> None:
        """Restore controller state before starting a new trajectory."""

    def act(self, state: BatteryState) -> CoolingAction:
        """Choose a coolant command from the currently observed battery state."""


__all__ = ["Controller"]
