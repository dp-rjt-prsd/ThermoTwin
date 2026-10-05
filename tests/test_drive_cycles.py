from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from thermotwin.scenarios.drive_cycles import (
    aggressive_drive_profile,
    constant_current_profile,
    pulse_current_profile,
    stop_and_go_profile,
)


@pytest.mark.parametrize(
    ("time_s", "expected"),
    [(0.0, 8.0), (4.5, 8.0), (5.0, 0.0), (7.0, 0.0)],
)
def test_constant_profile_is_nonzero_until_half_open_duration(
    time_s: float, expected: float
) -> None:
    profile = constant_current_profile(8.0, 5.0)

    assert profile(time_s) == expected


def test_pulse_profile_has_half_open_pulse_and_ends_at_total_duration() -> None:
    profile = pulse_current_profile(2.0, 9.0, 3.0, 2.0, 8.0)

    times = (0.0, 2.999, 3.0, 4.999, 5.0, 7.999, 8.0)
    assert [profile(time_s) for time_s in times] == [
        2.0,
        2.0,
        9.0,
        9.0,
        2.0,
        2.0,
        0.0,
    ]


def test_stop_and_go_profile_repeats_four_equal_stages() -> None:
    profile = stop_and_go_profile(100.0, 8.0, 20.0)

    assert [profile(time_s) for time_s in (0.0, 1.99, 2.0, 4.0, 6.0, 7.999, 8.0)] == [
        0.0,
        0.0,
        100.0,
        40.0,
        -25.0,
        -25.0,
        0.0,
    ]
    assert profile(10.0) == 100.0
    assert profile(12.0) == 40.0


def test_aggressive_drive_profile_repeats_expected_stages() -> None:
    profile = aggressive_drive_profile(100.0, 20.0, 45.0)

    times = (0.0, 6.999, 7.0, 11.999, 12.0, 15.999, 16.0, 19.999)
    assert [profile(time_s) for time_s in times] == [
        100.0,
        100.0,
        75.0,
        75.0,
        90.0,
        90.0,
        -35.0,
        -35.0,
    ]
    assert profile(20.0) == 100.0
    assert profile(45.0) == 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: constant_current_profile(-3.0, 12.0),
        lambda: pulse_current_profile(-3.0, 7.0, 2.0, 3.0, 12.0),
        lambda: stop_and_go_profile(60.0, 8.0, 12.0),
        lambda: aggressive_drive_profile(60.0, 8.0, 12.0),
    ],
)
def test_profiles_are_repeatable_pure_callables(
    factory: Callable[[], Callable[[float], float]],
) -> None:
    first = factory()
    second = factory()
    times = [0.0, 2.0, 5.0, 8.0, 11.999, 12.0, 20.0]

    assert [first(time_s) for time_s in times] == [second(time_s) for time_s in times]


@pytest.mark.parametrize("time_s", [-1.0, True, "1", None, np.nan, np.inf, -np.inf])
def test_profiles_reject_invalid_query_times(time_s: Any) -> None:
    profile = constant_current_profile(4.0, 10.0)

    with pytest.raises(ValueError, match="time_s"):
        profile(time_s)


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: constant_current_profile(value, 10.0),
        lambda value: constant_current_profile(3.0, value),
        lambda value: pulse_current_profile(value, 4.0, 1.0, 2.0, 10.0),
        lambda value: pulse_current_profile(3.0, value, 1.0, 2.0, 10.0),
        lambda value: pulse_current_profile(3.0, 4.0, value, 2.0, 10.0),
        lambda value: pulse_current_profile(3.0, 4.0, 1.0, value, 10.0),
        lambda value: pulse_current_profile(3.0, 4.0, 1.0, 2.0, value),
        lambda value: stop_and_go_profile(value, 4.0, 10.0),
        lambda value: stop_and_go_profile(3.0, value, 10.0),
        lambda value: stop_and_go_profile(3.0, 4.0, value),
        lambda value: aggressive_drive_profile(value, 4.0, 10.0),
        lambda value: aggressive_drive_profile(3.0, value, 10.0),
        lambda value: aggressive_drive_profile(3.0, 4.0, value),
    ],
)
@pytest.mark.parametrize("value", [True, "1", None, np.nan, np.inf, -np.inf])
def test_factories_reject_invalid_numeric_inputs(
    factory: Callable[[Any], object], value: Any
) -> None:
    with pytest.raises(ValueError):
        factory(value)


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: constant_current_profile(1.0, value),
        lambda value: pulse_current_profile(1.0, 2.0, 0.0, value, 5.0),
        lambda value: pulse_current_profile(1.0, 2.0, 0.0, 1.0, value),
        lambda value: stop_and_go_profile(value, 1.0, 4.0),
        lambda value: stop_and_go_profile(1.0, value, 4.0),
        lambda value: stop_and_go_profile(1.0, 1.0, value),
        lambda value: aggressive_drive_profile(value, 1.0, 4.0),
        lambda value: aggressive_drive_profile(1.0, value, 4.0),
        lambda value: aggressive_drive_profile(1.0, 1.0, value),
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_factories_reject_invalid_durations_and_peak_currents(
    factory: Callable[[float], object], value: float
) -> None:
    with pytest.raises(ValueError):
        factory(value)


def test_pulse_requires_nonnegative_start_and_end_within_total_duration() -> None:
    with pytest.raises(ValueError, match="pulse_start_s"):
        pulse_current_profile(1.0, 2.0, -0.1, 1.0, 5.0)
    with pytest.raises(ValueError, match="pulse"):
        pulse_current_profile(1.0, 2.0, 4.0, 1.1, 5.0)


@pytest.mark.parametrize("factory", [stop_and_go_profile, aggressive_drive_profile])
def test_drive_profiles_require_cycle_no_longer_than_duration(
    factory: Callable[[float, float, float], Callable[[float], float]],
) -> None:
    with pytest.raises(ValueError, match="cycle_duration_s"):
        factory(10.0, 11.0, 10.0)
