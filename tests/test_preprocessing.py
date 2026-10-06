from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from thermotwin.data.preprocessing import NormalizationStats, fit_normalization


def test_fit_uses_only_supplied_rows_and_round_trips() -> None:
    features = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    targets = np.array([[10.0], [12.0], [14.0]])
    stats = fit_normalization(features, targets, ("current", "soc"), ("delta",))

    np.testing.assert_allclose(stats.feature_mean, [3.0, 5.0])
    np.testing.assert_allclose(stats.feature_scale, [np.sqrt(8.0 / 3.0), 1.0])
    normalized = stats.transform_features(features)
    np.testing.assert_allclose(
        stats.inverse_transform_targets(stats.transform_targets(targets)), targets
    )
    np.testing.assert_allclose(normalized.mean(axis=0), [0.0, 0.0], atol=1e-14)
    assert not np.shares_memory(normalized, features)
    assert not stats.feature_mean.flags.writeable
    with pytest.raises(FrozenInstanceError):
        stats.feature_names = ("other", "soc")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("features", "targets", "feature_names", "target_names"),
    [
        (np.empty((0, 1)), np.ones((0, 1)), ("x",), ("y",)),
        (np.ones(2), np.ones((2, 1)), ("x",), ("y",)),
        (np.ones((2, 1)), np.ones((3, 1)), ("x",), ("y",)),
        (np.array([[np.nan]]), np.ones((1, 1)), ("x",), ("y",)),
        (np.ones((1, 1)), np.ones((1, 1)), ("",), ("y",)),
        (np.ones((1, 2)), np.ones((1, 1)), ("x", "x"), ("y",)),
    ],
)
def test_fit_rejects_invalid_inputs(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_normalization(features, targets, feature_names, target_names)


def test_stats_defensively_copies_and_validates_vectors() -> None:
    mean = np.array([1.0])
    stats = NormalizationStats(
        ("x",), ("y",), mean, np.array([2.0]), np.array([3.0]), np.array([4.0])
    )
    mean[0] = 99.0
    assert stats.feature_mean[0] == 1.0
    with pytest.raises(ValueError):
        NormalizationStats(
            ("x",),
            ("y",),
            np.array([[1.0]]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
        )


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_stats_reject_nonpositive_or_nonfinite_scales(scale: float) -> None:
    with pytest.raises(ValueError):
        NormalizationStats(
            ("x",),
            ("y",),
            np.array([0.0]),
            np.array([scale]),
            np.array([0.0]),
            np.array([1.0]),
        )


def test_near_constant_columns_use_unit_scale() -> None:
    stats = fit_normalization(
        np.array([[2.0], [2.0 + 1e-13]]),
        np.array([[4.0], [4.0 + 1e-13]]),
        ("x",),
        ("y",),
    )

    np.testing.assert_array_equal(stats.feature_scale, [1.0])
    np.testing.assert_array_equal(stats.target_scale, [1.0])


@pytest.mark.parametrize("bad", [[], np.ones((1, 2)), [[1.0, np.inf]]])
def test_transforms_reject_bad_shapes_and_nonfinite_data(bad: Any) -> None:
    stats = fit_normalization(np.ones((2, 1)), np.ones((2, 1)), ("x",), ("y",))
    with pytest.raises((TypeError, ValueError)):
        stats.transform_features(bad)


def test_metadata_round_trip_and_strict_reconstruction() -> None:
    stats = fit_normalization(
        np.array([[1.0], [3.0]]), np.array([[2.0], [6.0]]), ("x",), ("y",)
    )
    assert NormalizationStats.from_dict(stats.to_dict()).to_dict() == stats.to_dict()

    malformed = stats.to_dict()
    malformed["feature_mean"] = ["1"]
    with pytest.raises(ValueError):
        NormalizationStats.from_dict(malformed)
    with pytest.raises(ValueError):
        NormalizationStats.from_dict({**stats.to_dict(), "extra": 1})
