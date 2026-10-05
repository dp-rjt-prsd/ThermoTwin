"""Tests for the shared controller protocol and its illustrative config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from thermotwin.control.base import Controller
from thermotwin.core import BatteryState, CoolingAction


class StructuralController:
    """Minimal structural implementation used to check protocol membership."""

    def reset(self) -> None:
        """Reset the controller before a new trajectory."""

    def act(self, state: BatteryState) -> CoolingAction:
        """Return a legal-shaped cooling command for the supplied state."""
        del state
        return CoolingAction(0.0)


class MissingReset:
    """Controller-shaped object without the required reset method."""

    def act(self, state: BatteryState) -> CoolingAction:
        """Return a cooling command."""
        del state
        return CoolingAction(0.0)


class MissingAct:
    """Controller-shaped object without the required act method."""

    def reset(self) -> None:
        """Reset the controller."""


def test_controller_protocol_accepts_structural_implementation() -> None:
    controller = StructuralController()

    assert isinstance(controller, Controller)


def test_controller_protocol_rejects_object_missing_reset() -> None:
    assert not isinstance(MissingReset(), Controller)


def test_controller_protocol_rejects_object_missing_act() -> None:
    assert not isinstance(MissingAct(), Controller)


def test_controller_yaml_records_illustrative_si_assumptions() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "controller.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert isinstance(config, dict)
    thresholds: dict[str, Any] = config["threshold_controller"]
    boundaries = thresholds["temperature_boundaries_k"]
    assert boundaries == [303.15, 308.15, 313.15]
    assert boundaries == sorted(boundaries)
    stages = thresholds["stages"]
    assert [stage["name"] for stage in stages] == [
        "minimum",
        "low",
        "medium",
        "maximum",
    ]
    stage_flows = [stage["coolant_mass_flow_kg_s"] for stage in stages]
    assert stage_flows == [0.0, 0.025, 0.06, 0.1]
    assert len(stage_flows) == len(boundaries) + 1
    assert all(0.0 <= flow <= 0.1 for flow in stage_flows)

    pid: dict[str, Any] = config["pid_controller"]
    assert pid["target_temperature_k"] > 0.0
    assert all(
        pid[name] >= 0.0
        for name in ("kp_kg_s_k", "ki_kg_s_k_s", "kd_kg_s_k_per_s")
    )
    assert pid["minimum_coolant_mass_flow_kg_s"] == 0.0
    assert pid["maximum_coolant_mass_flow_kg_s"] == 0.1

    assert config["thermal_safety_limit_k"] == 313.15
    assert config["pump_power_coefficient_w_s3_kg3"] > 0.0
