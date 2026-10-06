"""Train-fitted, framework-neutral normalization for model datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Population standard deviations at or below this absolute threshold are treated
# as constant columns and assigned a unit scale.
_SCALE_EPSILON = 1e-12
_METADATA_VERSION = 1
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "feature_names",
        "target_names",
        "feature_mean",
        "feature_scale",
        "target_mean",
        "target_scale",
    }
)


def _names(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a nonempty sequence of names")
    names = tuple(value)
    if not names or any(type(name) is not str or not name.strip() for name in names):
        raise ValueError(f"{label} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicates")
    return names


def _vector(value: object, label: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if value.ndim != 1:
        raise ValueError(f"{label} must be rank 1")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values")
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    result.setflags(write=False)
    return result


def _matrix(value: object, label: str, width: int) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{label} must have shape (rows, {width})")
    if value.shape[0] == 0:
        raise ValueError(f"{label} must have at least one row")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain real numeric values")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    """Immutable means and scales paired with their ordered column names."""

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    feature_mean: FloatArray
    feature_scale: FloatArray
    target_mean: FloatArray
    target_scale: FloatArray

    def __post_init__(self) -> None:
        feature_names = _names(self.feature_names, "feature_names")
        target_names = _names(self.target_names, "target_names")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "target_names", target_names)
        for prefix, names in (("feature", feature_names), ("target", target_names)):
            mean = _vector(getattr(self, f"{prefix}_mean"), f"{prefix}_mean")
            scale = _vector(getattr(self, f"{prefix}_scale"), f"{prefix}_scale")
            if mean.size != len(names) or scale.size != len(names):
                raise ValueError(f"{prefix} vectors must match the ordered names")
            if np.any(scale <= 0.0):
                raise ValueError(f"{prefix}_scale values must be positive")
            object.__setattr__(self, f"{prefix}_mean", mean)
            object.__setattr__(self, f"{prefix}_scale", scale)

    def transform_features(self, values: NDArray[Any]) -> FloatArray:
        """Return a fresh standardized feature matrix."""
        matrix = _matrix(values, "features", len(self.feature_names))
        return np.array((matrix - self.feature_mean) / self.feature_scale, copy=True)

    def transform_targets(self, values: NDArray[Any]) -> FloatArray:
        """Return a fresh standardized target matrix."""
        matrix = _matrix(values, "targets", len(self.target_names))
        return np.array((matrix - self.target_mean) / self.target_scale, copy=True)

    def inverse_transform_targets(self, values: NDArray[Any]) -> FloatArray:
        """Convert standardized target values back to their original units."""
        matrix = _matrix(values, "normalized targets", len(self.target_names))
        return np.array(matrix * self.target_scale + self.target_mean, copy=True)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible, independent metadata without writing files."""
        return {
            "schema_version": _METADATA_VERSION,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_scale": self.target_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, metadata: object) -> NormalizationStats:
        """Reconstruct stats from strict JSON-compatible metadata."""
        if type(metadata) is not dict or set(metadata) != _METADATA_KEYS:
            raise ValueError("normalization metadata has invalid keys or type")
        raw = cast(dict[str, object], metadata)
        version = raw["schema_version"]
        if type(version) is not int or version != _METADATA_VERSION:
            raise ValueError("unsupported normalization metadata schema_version")
        names: dict[str, tuple[str, ...]] = {}
        for key in ("feature_names", "target_names"):
            value = raw[key]
            if type(value) is not list:
                raise ValueError(f"{key} must be a JSON array")
            names[key] = _names(value, key)
        vectors: dict[str, FloatArray] = {}
        for key in ("feature_mean", "feature_scale", "target_mean", "target_scale"):
            value = raw[key]
            if type(value) is not list or any(
                isinstance(item, bool) or not isinstance(item, Real) for item in value
            ):
                raise ValueError(f"{key} must be a JSON array of real numbers")
            try:
                vector = np.asarray(value, dtype=np.float64)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(f"{key} contains invalid values") from error
            vectors[key] = vector
        return cls(
            names["feature_names"],
            names["target_names"],
            vectors["feature_mean"],
            vectors["feature_scale"],
            vectors["target_mean"],
            vectors["target_scale"],
        )


def fit_normalization(
    features: NDArray[Any],
    targets: NDArray[Any],
    feature_names: Sequence[str],
    target_names: Sequence[str],
) -> NormalizationStats:
    """Fit population statistics on exactly the supplied feature and target rows."""
    normalized_feature_names = _names(feature_names, "feature_names")
    normalized_target_names = _names(target_names, "target_names")
    feature_matrix = _matrix(features, "features", len(normalized_feature_names))
    target_matrix = _matrix(targets, "targets", len(normalized_target_names))
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise ValueError("features and targets must have matching row counts")
    feature_mean = feature_matrix.mean(axis=0, dtype=np.float64)
    target_mean = target_matrix.mean(axis=0, dtype=np.float64)
    feature_scale = feature_matrix.std(axis=0, ddof=0, dtype=np.float64)
    target_scale = target_matrix.std(axis=0, ddof=0, dtype=np.float64)
    feature_scale[feature_scale <= _SCALE_EPSILON] = 1.0
    target_scale[target_scale <= _SCALE_EPSILON] = 1.0
    if not all(
        np.isfinite(vector).all()
        for vector in (feature_mean, target_mean, feature_scale, target_scale)
    ):
        raise ValueError("fitted statistics must be finite")
    return NormalizationStats(
        normalized_feature_names,
        normalized_target_names,
        feature_mean,
        feature_scale,
        target_mean,
        target_scale,
    )
