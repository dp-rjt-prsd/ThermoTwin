"""Shared strict primitives for reading saved dashboard artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Collection, Mapping, Sequence
from numbers import Real
from pathlib import Path, PureWindowsPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def frozen_array(value: object, field: str, *, allow_nan: bool = False) -> FloatArray:
    def valid_numeric_input(item: object) -> bool:
        if isinstance(item, np.ndarray):
            return item.dtype.kind in "iuf"
        if isinstance(item, (list, tuple)):
            return all(valid_numeric_input(child) for child in item)
        return not isinstance(item, bool) and isinstance(item, Real)

    if not valid_numeric_input(value):
        raise ValueError(
            f"{field} must contain JSON numeric values, not strings or booleans"
        )
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a numeric array") from error
    if np.any(np.isinf(array)) or (not allow_nan and np.any(np.isnan(array))):
        raise ValueError(f"{field} must contain finite values")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(array.shape)


def finite(value: object, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return normalized


def nonnegative_integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative JSON integer")
    return value


def json_mapping(value: object, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field} must be a JSON object")
    return cast(dict[str, Any], value)


def exact_fields(
    value: Mapping[str, Any], expected: Collection[str], field: str
) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(f"{field} fields mismatch (missing={missing}, extra={extra})")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _finite_json_float(value: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("JSON number must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError("JSON number must be finite")
    return normalized


def csv_finite(value: str, field: str, *, minimum: float | None = None) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be numeric") from error
    return finite(normalized, field, minimum=minimum)


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"artifact {path}: invalid JSON: {error}") from error


def safe_artifact(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a nonempty relative path")
    win_path = PureWindowsPath(relative)
    path = Path(relative)
    if path.is_absolute() or win_path.is_absolute() or win_path.drive:
        raise ValueError(f"{field} must not be an absolute path")
    if ".." in path.parts or ".." in win_path.parts:
        raise ValueError(f"{field} must not contain traversal components")
    resolved_root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{field} resolves outside its artifact directory") from error
    return resolved


def read_csv(
    path: Path, expected: Collection[str], artifact: str
) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            headers = reader.fieldnames
            if headers is None or len(headers) != len(set(headers)):
                raise ValueError(f"{artifact} has missing or duplicate CSV columns")
            if set(headers) != set(expected):
                raise ValueError(f"{artifact} columns do not match required schema")
            rows: list[dict[str, str]] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"{artifact} row {row_number} has malformed columns"
                    )
                rows.append(cast(dict[str, str], row))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"artifact {artifact}: could not read CSV: {error}") from error
    if not rows:
        raise ValueError(f"{artifact} must contain at least one row")
    return rows


def number_column(
    rows: Sequence[Mapping[str, str]],
    field: str,
    *,
    minimum: float | None = None,
) -> FloatArray:
    values = [
        csv_finite(row[field], f"{field} at row {i}", minimum=minimum)
        for i, row in enumerate(rows)
    ]
    return frozen_array(values, field)


def float_matrix(
    rows: Sequence[Mapping[str, str]],
    names: Sequence[str],
    field: str,
    *,
    positive: bool = False,
) -> FloatArray:
    matrix = [
        [
            csv_finite(
                row[name],
                f"{field}.{name} row {row_index}",
                minimum=0.0 if positive else None,
            )
            for name in names
        ]
        for row_index, row in enumerate(rows)
    ]
    result = frozen_array(matrix, field)
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{field} values must be positive")
    return result


def metric_mapping(value: object, field: str) -> Mapping[str, float | int]:
    raw = json_mapping(value, field)
    metric_fields = (
        "peak_temperature_k",
        "mean_temperature_k",
        "maximum_temperature_spread_k",
        "pump_energy_j",
        "thermal_violation_count",
        "time_above_thermal_limit_s",
    )
    exact_fields(raw, metric_fields, field)
    normalized: dict[str, float | int] = {}
    for name in metric_fields:
        number = finite(raw[name], f"{field}.{name}", minimum=0.0)
        if name == "thermal_violation_count":
            if number != int(number):
                raise ValueError(f"{field}.{name} must be an integer")
            normalized[name] = int(number)
        else:
            normalized[name] = number
    return normalized


def recompute_metrics(
    cell_temperatures_k: FloatArray,
    pump_powers_w: FloatArray,
    time_step_s: float,
    thermal_limit_k: float,
) -> Mapping[str, float | int]:
    """Reproduce diagnostics.metrics formulas using only saved trace arrays."""
    if cell_temperatures_k.ndim != 2 or cell_temperatures_k.shape[1] != 12:
        raise ValueError("cell_temperatures_k must have shape (N, 12)")
    if cell_temperatures_k.shape[0] == 0 or pump_powers_w.shape != (
        cell_temperatures_k.shape[0],
    ):
        raise ValueError(
            "trace temperature and pump arrays must share a nonzero length"
        )
    time_step = finite(time_step_s, "time_step_s")
    thermal_limit = finite(thermal_limit_k, "thermal_limit_k")
    if time_step <= 0.0 or thermal_limit <= 0.0:
        raise ValueError("time_step_s and thermal_limit_k must be positive")
    if not np.all(np.isfinite(cell_temperatures_k)) or np.any(
        cell_temperatures_k <= 0.0
    ):
        raise ValueError("trace cell temperatures must be positive and finite")
    if not np.all(np.isfinite(pump_powers_w)) or np.any(pump_powers_w < 0.0):
        raise ValueError("trace pump powers must be finite and nonnegative")

    temperature_values = [
        float(value) for value in cell_temperatures_k.reshape(-1)
    ]
    peak_temperature = max(temperature_values)
    maximum_spread = 0.0
    violation_count = 0
    for row in cell_temperatures_k:
        row_minimum = float(np.min(row))
        row_maximum = float(np.max(row))
        maximum_spread = max(maximum_spread, row_maximum - row_minimum)
        if row_maximum > thermal_limit:
            violation_count += 1
    temperature_scale = max(temperature_values)
    mean_temperature = temperature_scale * (
        math.fsum(value / temperature_scale for value in temperature_values)
        / len(temperature_values)
    )
    pump_energy = math.fsum(float(power) * time_step for power in pump_powers_w)
    time_above = violation_count * time_step
    values = (
        peak_temperature,
        mean_temperature,
        maximum_spread,
        pump_energy,
        time_above,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recomputed trace metrics must be finite")
    return {
        "peak_temperature_k": peak_temperature,
        "mean_temperature_k": mean_temperature,
        "maximum_temperature_spread_k": maximum_spread,
        "pump_energy_j": pump_energy,
        "thermal_violation_count": violation_count,
        "time_above_thermal_limit_s": time_above,
    }


def parse_inline_json(value: str, field: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{field} is invalid JSON: {error}") from error
