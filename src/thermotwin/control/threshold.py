"""Deterministic four-stage reactive coolant-flow controller."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from thermotwin.core import BatteryState, CoolingAction, CoolingConfig


class ThresholdController:
    """Choose a fixed cooling stage from the observed maximum cell temperature."""

    __slots__ = ("_temperature_boundaries_k", "_stage_flows_kg_s")

    def __init__(
        self,
        cooling_config: CoolingConfig,
        temperature_boundaries_k: Sequence[float],
        coolant_mass_flows_kg_s: Sequence[float],
    ) -> None:
        if not isinstance(cooling_config, CoolingConfig):
            raise ValueError("cooling_config must be a CoolingConfig")

        boundaries = self._validated_values(
            "temperature_boundaries_k", temperature_boundaries_k, expected_length=3
        )
        if any(value <= 0.0 for value in boundaries):
            raise ValueError("temperature_boundaries_k must be positive")
        if any(
            boundaries[index] >= boundaries[index + 1]
            for index in range(len(boundaries) - 1)
        ):
            raise ValueError("temperature_boundaries_k must be strictly ascending")

        intermediate_flows = self._validated_values(
            "coolant_mass_flows_kg_s",
            coolant_mass_flows_kg_s,
            expected_length=2,
        )
        minimum_flow = cooling_config.minimum_coolant_mass_flow_kg_s
        maximum_flow = cooling_config.maximum_coolant_mass_flow_kg_s
        if any(
            flow < minimum_flow or flow > maximum_flow
            for flow in intermediate_flows
        ):
            raise ValueError("intermediate coolant flows must be within cooling bounds")
        if intermediate_flows[0] > intermediate_flows[1]:
            raise ValueError("intermediate coolant flows must be nondecreasing")

        self._temperature_boundaries_k = boundaries
        self._stage_flows_kg_s = (
            minimum_flow,
            intermediate_flows[0],
            intermediate_flows[1],
            maximum_flow,
        )

    @staticmethod
    def _validated_values(
        name: str,
        values: Sequence[object],
        *,
        expected_length: int,
    ) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must contain exactly {expected_length} numbers")
        try:
            normalized = tuple(values)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain exactly {expected_length} numbers"
            ) from error
        if len(normalized) != expected_length:
            raise ValueError(f"{name} must contain exactly {expected_length} numbers")
        finite_values: list[float] = []
        for value in normalized:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} values must be finite real numbers")
            try:
                finite_value = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    f"{name} values must be finite real numbers"
                ) from error
            if not math.isfinite(finite_value):
                raise ValueError(f"{name} values must be finite real numbers")
            finite_values.append(finite_value)
        return tuple(finite_values)

    def reset(self) -> None:
        """Do nothing; this controller has no state between observations."""

    def act(self, state: BatteryState) -> CoolingAction:
        """Return the stage selected by the current maximum cell temperature."""
        maximum_temperature_k = float(max(state.cell_temperatures_k))
        stage_index = sum(
            maximum_temperature_k >= boundary
            for boundary in self._temperature_boundaries_k
        )
        return CoolingAction(self._stage_flows_kg_s[stage_index])


__all__ = ["ThresholdController"]
