"""Strict loaders for saved comparison-model artifacts."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, cast

import torch

from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.config import MLPConfig
from thermotwin.models.forecaster import MLPThermalForecaster
from thermotwin.models.mlp import ThermalMLP

ComparisonVariant = Literal["data_only", "physics_informed"]
_VARIANTS = frozenset(("data_only", "physics_informed"))


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON artifact contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON artifact contains nonfinite constant {value}")


def _validated_variant(value: object) -> ComparisonVariant:
    if type(value) is not str or value not in _VARIANTS:
        raise ValueError("variant must be 'data_only' or 'physics_informed'")
    return cast(ComparisonVariant, value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"could not read valid JSON artifact at {path}: {error}"
        ) from error
    if type(value) is not dict:
        raise ValueError(f"JSON artifact must be an object: {path}")
    return cast(dict[str, Any], value)


def _load_model_config(path: Path) -> MLPConfig:
    raw = _read_json(path)
    expected = {field.name for field in fields(MLPConfig)}
    if set(raw) != expected:
        raise ValueError(
            f"model_config.json fields do not match the MLPConfig schema: {path}"
        )
    if type(raw["hidden_widths"]) is not list:
        raise ValueError(f"model_config.json hidden_widths must be an array: {path}")
    try:
        return MLPConfig(**{**raw, "hidden_widths": tuple(raw["hidden_widths"])})
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"model_config.json is invalid at {path}: {error}") from error


def _load_normalization(path: Path) -> NormalizationStats:
    try:
        return NormalizationStats.from_dict(_read_json(path))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"normalization.json is invalid at {path}: {error}") from error


def _load_checkpoint(path: Path, config: MLPConfig) -> ThermalMLP:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(
            f"could not load tensor-only checkpoint at {path}: {error}"
        ) from error
    if (
        not isinstance(value, dict)
        or not value
        or not all(
            type(key) is str and bool(key) and isinstance(tensor, torch.Tensor)
            for key, tensor in value.items()
        )
    ):
        raise ValueError(
            f"checkpoint must contain only a nonempty tensor state_dict: {path}"
        )

    model = ThermalMLP(config)
    try:
        model.load_state_dict(value, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            f"checkpoint state_dict does not match model schema at {path}: {error}"
        ) from error
    return model


def load_comparison_forecaster(
    directory: str | Path,
    variant: ComparisonVariant,
    *,
    time_step_s: float,
    minimum_coolant_flow_kg_s: float,
    maximum_coolant_flow_kg_s: float,
) -> MLPThermalForecaster:
    """Load one explicitly selected comparison variant as a physical forecaster.

    Artifacts are limited to strict JSON metadata and tensor-only checkpoints;
    no serialized Python objects are accepted.
    """
    selected_variant = _validated_variant(variant)

    root = Path(directory)
    config = _load_model_config(root / "model_config.json")
    normalization = _load_normalization(root / "normalization.json")
    model = _load_checkpoint(root / f"{selected_variant}.pt", config)
    return MLPThermalForecaster(
        model,
        normalization,
        time_step_s,
        minimum_coolant_flow_kg_s,
        maximum_coolant_flow_kg_s,
    )
