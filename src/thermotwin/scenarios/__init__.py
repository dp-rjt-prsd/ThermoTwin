"""Public synthetic current-profile factories."""

from thermotwin.scenarios.charging import fast_charging_profile
from thermotwin.scenarios.drive_cycles import (
    aggressive_drive_profile,
    constant_current_profile,
    pulse_current_profile,
    stop_and_go_profile,
)

__all__ = [
    "aggressive_drive_profile",
    "constant_current_profile",
    "fast_charging_profile",
    "pulse_current_profile",
    "stop_and_go_profile",
]
