from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thermotwin.core import (
    BatteryState,
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    ResistanceFaultConfig,
    SimulationConfig,
)
from thermotwin.diagnostics.metrics import ExperimentMetrics
from thermotwin.experiments.serialization import save_experiment_result
from thermotwin.experiments.types import (
    ExperimentRecord,
    ExperimentResult,
    ExperimentScenario,
)


def make_environment_config(
    resistance_fault: ResistanceFaultConfig | None = None,
) -> EnvironmentConfig:
    return EnvironmentConfig(
        pack=PackConfig(
            rows=1,
            columns=2,
            cell_mass_kg=0.8,
            cell_specific_heat_capacity_j_kg_k=900.0,
            nominal_cell_resistance_ohm=0.015,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.5,
            ambient_thermal_resistance_k_w=8.0,
            cell_cooling_surface_area_m2=0.02,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.0,
            maximum_coolant_mass_flow_kg_s=0.1,
            base_heat_transfer_coefficient_w_m2_k=5.0,
            flow_heat_transfer_gain=500.0,
            flow_exponent=0.8,
        ),
        simulation=SimulationConfig(
            default_time_step_s=1.0,
            initial_soc_fraction=0.5,
            initial_cell_temperature_k=300.0,
            ambient_temperature_k=302.0,
            coolant_inlet_temperature_k=290.0,
            initial_current_a=0.0,
            initial_cell_temperatures_k=(299.0, 300.0),
        ),
        resistance_fault=resistance_fault,
    )


def make_result(
    environment_config: EnvironmentConfig | None = None,
) -> ExperimentResult:
    scenario = ExperimentScenario(
        name="pulse-load",
        environment_config=(
            make_environment_config()
            if environment_config is None
            else environment_config
        ),
        current_profile=lambda _time_s: 0.0,
        duration_s=2.0,
        time_step_s=1.0,
        thermal_limit_k=320.0,
        pump_power_coefficient_w_s3_kg3=100.0,
        metadata={"profile": "two-pulse", "peak_current_a": 12.5},
    )
    records = tuple(
        ExperimentRecord(
            state=BatteryState(
                time_s=time_s,
                current_a=current,
                soc_fraction=soc,
                ambient_temperature_k=302.0,
                coolant_inlet_temperature_k=290.0,
                coolant_mass_flow_kg_s=flow,
                cell_temperatures_k=np.array(temperatures),
                cell_resistances_ohm=np.array(resistances),
            ),
            pump_power_w=pump,
        )
        for time_s, current, soc, flow, temperatures, resistances, pump in (
            (1.0, 5.0, 0.49, 0.02, (300.25, 301.5), (0.01, 0.02), 0.008),
            (2.0, 7.0, 0.48, 0.03, (300.75, 302.5), (0.011, 0.021), 0.027),
        )
    )
    return ExperimentResult(
        scenario=scenario,
        controller_name="PredictiveController",
        seed=17,
        records=records,
        metrics=ExperimentMetrics(302.5, 301.25, 1.75, 0.035, 0, 0.0),
        metadata={"model": "physics-informed", "trial": 3},
    )


def test_saves_csv_metrics_and_full_provenance(tmp_path: Path) -> None:
    result = make_result()
    run_directory = save_experiment_result(
        result,
        tmp_path / "nested" / "results",
        "run-1",
    )

    assert run_directory == tmp_path / "nested" / "results" / "run-1"
    with (
        (run_directory / "timeseries.csv").open(
            encoding="utf-8",
            newline="",
        ) as stream
    ):
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "time_s",
        "current_a",
        "soc_fraction",
        "ambient_temperature_k",
        "coolant_inlet_temperature_k",
        "coolant_mass_flow_kg_s",
        "pump_power_w",
        "cell_temperature_0_k",
        "cell_temperature_1_k",
        "cell_resistance_0_ohm",
        "cell_resistance_1_ohm",
    ]
    assert len(rows) == 2
    assert [float(row["time_s"]) for row in rows] == [1.0, 2.0]
    assert float(rows[0]["current_a"]) == 5.0
    assert float(rows[1]["soc_fraction"]) == 0.48
    assert float(rows[0]["coolant_mass_flow_kg_s"]) == 0.02
    assert float(rows[1]["pump_power_w"]) == 0.027
    assert float(rows[0]["cell_temperature_0_k"]) == 300.25
    assert float(rows[1]["cell_temperature_1_k"]) == 302.5
    assert float(rows[0]["cell_resistance_1_ohm"]) == 0.02

    metrics = json.loads((run_directory / "metrics.json").read_text(encoding="utf-8"))
    assert metrics == {
        "maximum_temperature_spread_k": 1.75,
        "mean_temperature_k": 301.25,
        "peak_temperature_k": 302.5,
        "pump_energy_j": 0.035,
        "thermal_violation_count": 0,
        "time_above_thermal_limit_s": 0.0,
    }
    provenance = json.loads(
        (run_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance == {
        "schema_version": "1.0",
        "scenario": {
            "name": "pulse-load",
            "duration_s": 2.0,
            "time_step_s": 1.0,
            "thermal_limit_k": 320.0,
            "pump_power_coefficient_w_s3_kg3": 100.0,
        },
        "environment_config": {
            "pack": {
                "rows": 1,
                "columns": 2,
                "cell_mass_kg": 0.8,
                "cell_specific_heat_capacity_j_kg_k": 900.0,
                "nominal_cell_resistance_ohm": 0.015,
                "cell_capacity_ah": 50.0,
                "neighbor_thermal_resistance_k_w": 1.5,
                "ambient_thermal_resistance_k_w": 8.0,
                "cell_cooling_surface_area_m2": 0.02,
            },
            "cooling": {
                "minimum_coolant_mass_flow_kg_s": 0.0,
                "maximum_coolant_mass_flow_kg_s": 0.1,
                "base_heat_transfer_coefficient_w_m2_k": 5.0,
                "flow_heat_transfer_gain": 500.0,
                "flow_exponent": 0.8,
            },
            "resistance_fault": None,
            "simulation": {
                "default_time_step_s": 1.0,
                "initial_soc_fraction": 0.5,
                "initial_cell_temperature_k": 300.0,
                "ambient_temperature_k": 302.0,
                "coolant_inlet_temperature_k": 290.0,
                "initial_current_a": 0.0,
                "initial_cell_temperatures_k": [299.0, 300.0],
                "initial_cell_resistances_ohm": None,
            },
        },
        "scenario_metadata": {"peak_current_a": 12.5, "profile": "two-pulse"},
        "controller_class_name": "PredictiveController",
        "seed": 17,
        "result_metadata": {"model": "physics-informed", "trial": 3},
        "record_count": 2,
    }


def test_saves_resistance_fault_configuration_in_provenance(
    tmp_path: Path,
) -> None:
    fault = ResistanceFaultConfig(
        cell_index=1,
        start_time_s=1.5,
        resistance_multiplier=2.5,
    )
    result = make_result(make_environment_config(resistance_fault=fault))
    run_directory = save_experiment_result(result, tmp_path, "fault-run")

    provenance = json.loads(
        (run_directory / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["environment_config"]["resistance_fault"] == {
        "cell_index": 1,
        "start_time_s": 1.5,
        "resistance_multiplier": 2.5,
    }


def test_artifact_content_is_deterministic_across_run_ids(tmp_path: Path) -> None:
    result = make_result()
    first = save_experiment_result(result, tmp_path, "first")
    second = save_experiment_result(result, tmp_path, "second")

    for filename in ("timeseries.csv", "metrics.json", "provenance.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "/absolute",
        " space",
        "x ",
        "é",
        "a.b",
        "a" * 65,
        True,
        1,
        None,
    ],
)
def test_rejects_unsafe_run_ids_before_writing(tmp_path: Path, run_id: Any) -> None:
    root = tmp_path / "not-created"

    with pytest.raises((TypeError, ValueError)):
        save_experiment_result(make_result(), root, run_id)

    assert not root.exists()


def test_rejects_wrong_result_type_before_writing(tmp_path: Path) -> None:
    root = tmp_path / "not-created"

    with pytest.raises(TypeError):
        save_experiment_result(object(), root, "valid")  # type: ignore[arg-type]

    assert not root.exists()


def test_existing_destination_is_not_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "results"
    destination = root / "run-1"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_experiment_result(make_result(), root, "run-1")

    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in destination.iterdir()) == ["keep.txt"]
