"""Framework-neutral, immutable configuration for thermal surrogate training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Shape and activation for a small feed-forward temperature-delta model."""

    input_dim: int
    output_dim: int
    hidden_widths: tuple[int, ...]
    activation: Literal["gelu"] = "gelu"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "input_dim", _positive_integer("input_dim", self.input_dim)
        )
        object.__setattr__(
            self, "output_dim", _positive_integer("output_dim", self.output_dim)
        )
        widths = self.hidden_widths
        if not isinstance(widths, (tuple, list)) or not widths:
            raise ValueError(
                "hidden_widths must be a nonempty sequence of positive integers"
            )
        normalized = tuple(
            _positive_integer("hidden_widths entries", width) for width in widths
        )
        object.__setattr__(self, "hidden_widths", normalized)
        if self.activation != "gelu":
            raise ValueError("activation must be 'gelu'")


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Deterministic optimizer, stopping, batching, and device policy settings."""

    seed: int
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int
    min_improvement: float
    weight_decay: float
    device_policy: Literal["auto", "cpu", "cuda"]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_integer("seed", self.seed))
        for name in ("batch_size", "max_epochs", "patience"):
            object.__setattr__(self, name, _positive_integer(name, getattr(self, name)))
        for name in ("learning_rate", "min_improvement", "weight_decay"):
            value = _finite_real(name, getattr(self, name))
            if value < 0.0 or (name == "learning_rate" and value == 0.0):
                qualifier = "positive" if name == "learning_rate" else "nonnegative"
                raise ValueError(f"{name} must be {qualifier}")
            object.__setattr__(self, name, value)
        if self.patience > self.max_epochs:
            raise ValueError("patience must not exceed max_epochs")
        if self.device_policy not in ("auto", "cpu", "cuda"):
            raise ValueError("device_policy must be 'auto', 'cpu', or 'cuda'")


@dataclass(frozen=True, slots=True)
class PhysicsLossConfig:
    """Weight and watt scale for a normalized physics residual loss."""

    lambda_physics: float
    residual_scale_w: float

    def __post_init__(self) -> None:
        weight = _finite_real("lambda_physics", self.lambda_physics)
        if weight < 0.0:
            raise ValueError("lambda_physics must be nonnegative")
        object.__setattr__(self, "lambda_physics", weight)

        residual_scale = _finite_real("residual_scale_w", self.residual_scale_w)
        if residual_scale <= 0.0:
            raise ValueError("residual_scale_w must be positive")
        object.__setattr__(self, "residual_scale_w", residual_scale)
