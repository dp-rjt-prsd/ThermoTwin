"""Persist immutable experiment traces with their measured provenance."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os

    from thermotwin.experiments.types import ExperimentResult

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z", re.ASCII)
_SCHEMA_VERSION = "1.0"


def _json_ready(value: object) -> object:
    """Convert contract values, including NumPy numeric scalars, to JSON types."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _json_text(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _timeseries_text(result: ExperimentResult) -> str:
    cell_count = result.records[0].state.cell_temperatures_k.size
    fieldnames = [
        "time_s",
        "current_a",
        "soc_fraction",
        "ambient_temperature_k",
        "coolant_inlet_temperature_k",
        "coolant_mass_flow_kg_s",
        "pump_power_w",
        *(f"cell_temperature_{index}_k" for index in range(cell_count)),
        *(f"cell_resistance_{index}_ohm" for index in range(cell_count)),
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in result.records:
        state = record.state
        row: dict[str, float] = {
            "time_s": state.time_s,
            "current_a": state.current_a,
            "soc_fraction": state.soc_fraction,
            "ambient_temperature_k": state.ambient_temperature_k,
            "coolant_inlet_temperature_k": state.coolant_inlet_temperature_k,
            "coolant_mass_flow_kg_s": state.coolant_mass_flow_kg_s,
            "pump_power_w": record.pump_power_w,
        }
        row.update(
            {
                f"cell_temperature_{index}_k": float(temperature)
                for index, temperature in enumerate(state.cell_temperatures_k)
            }
        )
        row.update(
            {
                f"cell_resistance_{index}_ohm": float(resistance)
                for index, resistance in enumerate(state.cell_resistances_ohm)
            }
        )
        writer.writerow(row)
    return stream.getvalue()


def save_experiment_result(
    result: ExperimentResult,
    output_directory: str | os.PathLike[str],
    run_id: str,
) -> Path:
    """Write CSV, metrics, and provenance artifacts in a new run directory."""
    from thermotwin.experiments.types import ExperimentResult

    if type(result) is not ExperimentResult:
        raise TypeError("result must be exactly an ExperimentResult")
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must be 1-64 ASCII letters, digits, hyphens, or underscores "
            "and begin with a letter or digit"
        )

    scenario = result.scenario
    provenance = {
        "schema_version": _SCHEMA_VERSION,
        "scenario": {
            "name": scenario.name,
            "duration_s": scenario.duration_s,
            "time_step_s": scenario.time_step_s,
            "thermal_limit_k": scenario.thermal_limit_k,
            "pump_power_coefficient_w_s3_kg3": (
                scenario.pump_power_coefficient_w_s3_kg3
            ),
        },
        "environment_config": asdict(scenario.environment_config),
        "scenario_metadata": dict(scenario.metadata),
        "controller_class_name": result.controller_name,
        "seed": result.seed,
        "result_metadata": dict(result.metadata),
        "record_count": len(result.records),
    }

    # Prepare all content before creating a destination, so invalid/non-finite
    # serialized values cannot leave behind a partial run directory.
    timeseries_content = _timeseries_text(result)
    metrics_content = _json_text(asdict(result.metrics))
    provenance_content = _json_text(provenance)

    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / run_id
    run_directory.mkdir(exist_ok=False)
    (run_directory / "timeseries.csv").write_text(
        timeseries_content,
        encoding="utf-8",
        newline="",
    )
    (run_directory / "metrics.json").write_text(metrics_content, encoding="utf-8")
    (run_directory / "provenance.json").write_text(
        provenance_content,
        encoding="utf-8",
    )
    return run_directory
