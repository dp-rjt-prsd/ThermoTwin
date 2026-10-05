from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from thermotwin.physics.electrical import soc_after_coulomb_counting


def test_coulomb_counting_reduces_soc_by_analytical_discharge_delta() -> None:
    # 10 A for 1800 s consumes 5 Ah from a 50 Ah cell.
    assert soc_after_coulomb_counting(0.8, 10.0, 50.0, 1800.0) == pytest.approx(0.7)


def test_coulomb_counting_increases_soc_by_analytical_charge_delta() -> None:
    # Negative current denotes charging: 5 Ah added to a 50 Ah cell.
    assert soc_after_coulomb_counting(0.2, -10.0, 50.0, 1800.0) == pytest.approx(0.3)


def test_coulomb_counting_preserves_soc_at_zero_current() -> None:
    assert soc_after_coulomb_counting(0.37, 0.0, 50.0, 1800.0) == 0.37


def test_coulomb_counting_clamps_overdischarge_to_zero() -> None:
    assert soc_after_coulomb_counting(0.01, 10.0, 1.0, 3600.0) == 0.0


def test_coulomb_counting_clamps_overcharge_to_one() -> None:
    assert soc_after_coulomb_counting(0.99, -10.0, 1.0, 3600.0) == 1.0


@pytest.mark.parametrize("soc", [-0.01, 1.01, True, "0.5", None, np.nan, np.inf])
def test_coulomb_counting_rejects_invalid_initial_soc(soc: Any) -> None:
    with pytest.raises(ValueError, match="soc_fraction"):
        soc_after_coulomb_counting(soc, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("current", [True, "1", None, np.nan, np.inf, -np.inf])
def test_coulomb_counting_rejects_invalid_current(current: Any) -> None:
    with pytest.raises(ValueError, match="current_a"):
        soc_after_coulomb_counting(0.5, current, 1.0, 1.0)


@pytest.mark.parametrize("capacity", [0.0, -1.0, True, "1", None, np.nan, np.inf])
def test_coulomb_counting_rejects_invalid_capacity(capacity: Any) -> None:
    with pytest.raises(ValueError, match="cell_capacity_ah"):
        soc_after_coulomb_counting(0.5, 0.0, capacity, 1.0)


@pytest.mark.parametrize("dt", [0.0, -1.0, True, "1", None, np.nan, np.inf])
def test_coulomb_counting_rejects_invalid_time_step(dt: Any) -> None:
    with pytest.raises(ValueError, match="dt_s"):
        soc_after_coulomb_counting(0.5, 0.0, 1.0, dt)
