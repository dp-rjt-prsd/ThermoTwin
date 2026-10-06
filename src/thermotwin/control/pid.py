"""Bounded PID coolant control based on observed pack temperature."""

from __future__ import annotations

import math
from numbers import Real

from thermotwin.core import BatteryState, CoolingAction, CoolingConfig


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class PIDController:
    """Adjust coolant flow with bounded PID control of pack maximum temperature."""

    def __init__(
        self,
        cooling_config: CoolingConfig,
        target_temperature_k: float,
        kp_kg_s_k: float,
        ki_kg_s_k_s: float,
        kd_kg_s_k_per_s: float,
    ) -> None:
        target = _finite_real("target_temperature_k", target_temperature_k)
        if target <= 0.0:
            raise ValueError("target_temperature_k must be positive")

        gains = (
            ("kp_kg_s_k", kp_kg_s_k),
            ("ki_kg_s_k_s", ki_kg_s_k_s),
            ("kd_kg_s_k_per_s", kd_kg_s_k_per_s),
        )
        normalized_gains: list[float] = []
        for name, gain in gains:
            normalized = _finite_real(name, gain)
            if normalized < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            normalized_gains.append(normalized)

        self._cooling_config = cooling_config
        self._target_temperature_k = target
        self._kp = normalized_gains[0]
        self._ki = normalized_gains[1]
        self._kd = normalized_gains[2]
        self.reset()

    def reset(self) -> None:
        """Clear observation and accumulated integral state."""
        self._previous_time_s: float | None = None
        self._previous_error_k: float | None = None
        self._integral_error_k_s = 0.0

    def act(self, state: BatteryState) -> CoolingAction:
        """Return a legal flow based only on this observed battery state."""
        maximum_temperature_k = float(state.cell_temperatures_k.max())
        error_k = maximum_temperature_k - self._target_temperature_k
        proportional = self._kp * error_k
        integral = self._integral_error_k_s
        derivative = 0.0

        previous_time_s = self._previous_time_s
        previous_error_k = self._previous_error_k
        current_time_s = state.time_s
        if previous_time_s is not None:
            if current_time_s <= previous_time_s:
                raise ValueError("BatteryState.time_s must be strictly increasing")
            assert previous_error_k is not None
            elapsed_s = current_time_s - previous_time_s
            integral += error_k * elapsed_s
            derivative = (error_k - previous_error_k) / elapsed_s

        minimum_flow = self._cooling_config.minimum_coolant_mass_flow_kg_s
        maximum_flow = self._cooling_config.maximum_coolant_mass_flow_kg_s
        command = (
            minimum_flow + proportional + self._ki * integral + self._kd * derivative
        )
        flow = min(maximum_flow, max(minimum_flow, command))

        # Commit state only after validation and command construction succeed.
        self._previous_time_s = current_time_s
        self._previous_error_k = error_k
        self._integral_error_k_s = integral
        return CoolingAction(flow)


__all__ = ["PIDController"]
