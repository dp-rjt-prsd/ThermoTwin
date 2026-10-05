from __future__ import annotations

import math

import pytest

from thermotwin.scenarios.charging import fast_charging_profile


def test_fast_charging_profile_has_constant_current_then_linear_taper() -> None:
    profile = fast_charging_profile(100.0, 100.0, 0.7, 0.2)

    assert profile(0.0) == -100.0
    assert profile(69.999) == -100.0
    assert profile(70.0) == -100.0
    assert profile(85.0) == -60.0
    assert profile(99.999) == pytest.approx(-20.002666666666663)
    assert profile(100.0) == 0.0
    assert profile(120.0) == 0.0


def test_fast_charging_profile_is_repeatable() -> None:
    profile = fast_charging_profile(80.0, 240.0)
    sample_times = (0.0, 168.0, 204.0, 239.0, 240.0)

    assert tuple(profile(time_s) for time_s in sample_times) == tuple(
        profile(time_s) for time_s in sample_times
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (True, 100.0),
        ("100", 100.0),
        (math.nan, 100.0),
        (math.inf, 100.0),
        (0.0, 100.0),
        (-1.0, 100.0),
        (100.0, True),
        (100.0, "100"),
        (100.0, math.nan),
        (100.0, math.inf),
        (100.0, 0.0),
        (100.0, -1.0),
        (100.0, 100.0, True, 0.2),
        (100.0, 100.0, "0.7", 0.2),
        (100.0, 100.0, math.nan, 0.2),
        (100.0, 100.0, 1.0, 0.2),
        (100.0, 100.0, 0.0, 0.2),
        (100.0, 100.0, 0.7, True),
        (100.0, 100.0, 0.7, "0.2"),
        (100.0, 100.0, 0.7, math.inf),
        (100.0, 100.0, 0.7, -0.1),
        (100.0, 100.0, 0.7, 1.1),
    ],
)
def test_fast_charging_profile_rejects_invalid_factory_values(
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError):
        fast_charging_profile(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("time_s", [True, "1", math.nan, math.inf, -math.inf, -0.1])
def test_fast_charging_profile_rejects_invalid_time(time_s: object) -> None:
    profile = fast_charging_profile(100.0, 100.0)

    with pytest.raises(ValueError):
        profile(time_s)  # type: ignore[arg-type]
