from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import cast

import pytest

from thermotwin.core import BatteryState, CoolingAction
from thermotwin.experiments.benchmark import (
    BenchmarkExecution,
    run_benchmark_matrix,
)
from thermotwin.experiments.benchmark_controllers import (
    BenchmarkControllerParameters,
)
from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkMode,
    ModelProvenance,
    ModelVariant,
    ScenarioId,
)
from thermotwin.models.forecaster import ForecastRollout, ThermalForecaster
from thermotwin.scenarios.benchmark import (
    BenchmarkScenarioCase,
    load_benchmark_scenarios,
)


class StubForecaster:
    cell_count = 12
    time_step_s = 1.0

    def __init__(self) -> None:
        self.initial_states: list[BatteryState] = []
        self.actions: list[tuple[CoolingAction, ...]] = []

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        return self._next(state, action)

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: object = None,
    ) -> ForecastRollout:
        assert future_conditions is None
        self.initial_states.append(initial_state)
        self.actions.append(actions)
        states: list[BatteryState] = []
        current = initial_state
        for action in actions:
            current = self._next(current, action)
            states.append(current)
        return ForecastRollout(initial_state, actions, tuple(states))

    def _next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        return BatteryState(
            time_s=state.time_s + self.time_step_s,
            current_a=state.current_a,
            soc_fraction=state.soc_fraction,
            ambient_temperature_k=state.ambient_temperature_k,
            coolant_inlet_temperature_k=state.coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=action.coolant_mass_flow_kg_s,
            cell_temperatures_k=state.cell_temperatures_k,
            cell_resistances_ohm=state.cell_resistances_ohm,
        )


def _parameters() -> BenchmarkControllerParameters:
    from thermotwin.control.predictive_types import PredictiveObjectiveWeights

    return BenchmarkControllerParameters(
        threshold_temperature_boundaries_k=(303.15, 308.15, 313.15),
        threshold_intermediate_flow_fractions=(0.25, 0.6),
        pid_target_temperature_k=308.15,
        pid_kp_kg_s_k=0.005,
        pid_ki_kg_s_k_s=0.0001,
        pid_kd_kg_s_k_per_s=0.0,
        predictive_horizon_steps=1,
        predictive_candidate_count=2,
        thermal_safety_limit_k=313.15,
        pump_power_coefficient_w_s3_kg3=100000.0,
        objective_weights=PredictiveObjectiveWeights(1.0, 1.0, 0.0001, 100.0),
    )


def _inputs(
    mode: BenchmarkMode = "quick",
) -> tuple[
    Mapping[ScenarioId, BenchmarkScenarioCase],
    dict[ModelVariant, ThermalForecaster],
    dict[ModelVariant, ModelProvenance],
]:
    root = Path(__file__).resolve().parents[1]
    cases = load_benchmark_scenarios(
        root / "configs/benchmark.yaml",
        root / "configs/pack.yaml",
        root / "configs/simulation.yaml",
        mode=mode,
        seed=107,
    )
    forecasters: dict[ModelVariant, ThermalForecaster] = {
        "data_only": StubForecaster(),
        "physics_informed": StubForecaster(),
    }

    def reference(name: str) -> ArtifactReference:
        return ArtifactReference(name, "a" * 64)

    provenance: dict[ModelVariant, ModelProvenance] = {
        key: ModelProvenance(
            model_variant=key,
            training_mode="full" if mode == "full" else "smoke",
            training_seed=17,
            checkpoint=reference(f"{key}/checkpoint.pt"),
            model_config=reference(f"{key}/config.json"),
            normalization=reference(f"{key}/normalization.json"),
            comparison_provenance=reference(f"{key}/comparison.json"),
            dataset_manifests=(reference(f"{key}/dataset.json"),),
        )
        for key in ("data_only", "physics_informed")
    }
    return cases, forecasters, provenance


def _execute(mode: BenchmarkMode = "quick") -> BenchmarkExecution:
    cases, forecasters, provenance = _inputs(mode)
    return run_benchmark_matrix(
        cases,
        _parameters(),
        forecasters,
        provenance,
        mode=mode,
    )


def test_executes_complete_canonical_fair_matrix_and_keeps_scenario_identity() -> None:
    cases, forecasters, provenance = _inputs()
    execution = run_benchmark_matrix(
        cases, _parameters(), forecasters, provenance, mode="quick"
    )
    assert len(execution.matrix.cells) == len(execution.results) == 24
    assert tuple(execution.results) == tuple(
        (scenario, variant) for scenario in SCENARIO_IDS for variant in VARIANT_IDS
    )
    assert tuple(
        (cell.scenario_id, cell.variant_id) for cell in execution.matrix.cells
    ) == tuple(execution.results)
    for scenario_id in SCENARIO_IDS:
        case = cases[scenario_id]
        scenario_cells = [
            cell for cell in execution.matrix.cells if cell.scenario_id == scenario_id
        ]
        assert {cell.seed for cell in scenario_cells} == {case.seed}
        assert {
            cell.scenario_configuration_sha256 for cell in scenario_cells
        } == {case.configuration_sha256}
        for variant_id in VARIANT_IDS:
            result = execution.results[(scenario_id, variant_id)]
            assert result.scenario is case.scenario
            assert result.seed == case.seed
            for record in result.records:
                flow = record.state.coolant_mass_flow_kg_s
                cooling = case.scenario.environment_config.cooling
                assert (
                    cooling.minimum_coolant_mass_flow_kg_s
                    <= flow
                    <= cooling.maximum_coolant_mass_flow_kg_s
                )
    for cell in execution.matrix.cells:
        if cell.variant_id == "predictive_data_only":
            assert cell.model_provenance is provenance["data_only"]
            assert cell.forecast_assumptions is not None
            assert cell.forecast_assumptions.uses_future_simulator_truth is False
        elif cell.variant_id == "predictive_physics_informed":
            assert cell.model_provenance is provenance["physics_informed"]
            assert cell.forecast_assumptions is not None
        else:
            assert cell.model_provenance is None
            assert cell.forecast_assumptions is None
        assert cell.fault_detection is None


def test_inputs_are_copied_and_execution_mapping_is_immutable() -> None:
    cases, forecasters, provenance = _inputs()
    mutable_cases = dict(cases)
    original_normal = mutable_cases["normal"]
    execution = run_benchmark_matrix(
        mutable_cases, _parameters(), forecasters, provenance, mode="quick"
    )
    mutable_cases.clear()
    forecasters.clear()
    provenance.clear()
    assert (
        execution.results["normal", "threshold"].scenario
        is original_normal.scenario
    )
    assert execution.matrix.cells[2].model_provenance is not None
    with pytest.raises(TypeError):
        cast(dict[object, object], execution.results)[("x", "y")] = object()
    with pytest.raises((FrozenInstanceError, AttributeError)):
        execution.matrix = execution.matrix  # type: ignore[misc]


def test_direct_execution_rejects_distinct_equivalent_scenario_objects() -> None:
    from thermotwin.experiments.types import ExperimentScenario

    execution = _execute()
    results = dict(execution.results)
    original = results["normal", "threshold"]
    equivalent_scenario = replace(original.scenario)
    assert type(equivalent_scenario) is ExperimentScenario
    assert equivalent_scenario == original.scenario
    assert equivalent_scenario is not original.scenario
    results["normal", "threshold"] = replace(
        original, scenario=equivalent_scenario
    )
    with pytest.raises(ValueError, match="identical scenario object"):
        BenchmarkExecution(matrix=execution.matrix, results=results)


def test_repeated_execution_is_deterministic_and_controllers_do_not_leak_state(
) -> None:
    first = _execute()
    second = _execute()
    assert first.matrix == second.matrix
    for key in first.results:
        left = first.results[key]
        right = second.results[key]
        assert left.metrics == right.metrics
        assert left.controller_name == right.controller_name
        assert [record.state.coolant_mass_flow_kg_s for record in left.records] == [
            record.state.coolant_mass_flow_kg_s for record in right.records
        ]


def test_full_mode_rejects_smoke_provenance() -> None:
    cases, forecasters, provenance = _inputs(mode="full")
    provenance = {
        key: replace(value, training_mode="smoke")
        for key, value in provenance.items()
    }
    with pytest.raises(ValueError, match="training_mode must be full"):
        run_benchmark_matrix(
            cases,
            _parameters(),
            forecasters,
            provenance,
            mode="full",
        )


@pytest.mark.parametrize("bad", [None, {}, {"normal": object()}])
def test_rejects_incomplete_or_wrong_case_mapping(bad: object) -> None:
    _, forecasters, provenance = _inputs()
    with pytest.raises((ValueError, TypeError)):
        run_benchmark_matrix(
            cast(Mapping[ScenarioId, BenchmarkScenarioCase], bad),
            _parameters(),
            forecasters,
            provenance,
            mode="quick",
        )


def test_rejects_unknown_case_key() -> None:
    cases, forecasters, provenance = _inputs()
    invalid_cases: dict[object, object] = {}
    for key, value in cases.items():
        invalid_cases[key] = value
    invalid_cases["unknown"] = invalid_cases.pop("normal")
    with pytest.raises(ValueError, match="keys mismatch"):
        run_benchmark_matrix(
            cast(Mapping[ScenarioId, BenchmarkScenarioCase], invalid_cases),
            _parameters(),
            forecasters,
            provenance,
            mode="quick",
        )


def test_rejects_wrong_and_unknown_forecaster_or_provenance_keys() -> None:
    cases, forecasters, provenance = _inputs()
    incomplete_forecasters: dict[object, object] = {
        "data_only": StubForecaster()
    }
    unknown_forecasters: dict[object, object] = {}
    for key, value in forecasters.items():
        unknown_forecasters[key] = value
    unknown_forecasters["unknown"] = StubForecaster()
    for bad_forecasters in (incomplete_forecasters, unknown_forecasters):
        with pytest.raises(ValueError, match="keys mismatch"):
            run_benchmark_matrix(
                cases,
                _parameters(),
                cast(Mapping[ModelVariant, ThermalForecaster], bad_forecasters),
                provenance,
                mode="quick",
            )
    incomplete_provenance: dict[object, object] = {
        "data_only": provenance["data_only"]
    }
    unknown_provenance: dict[object, object] = {}
    for provenance_key, provenance_value in provenance.items():
        unknown_provenance[provenance_key] = provenance_value
    unknown_provenance["unknown"] = provenance["data_only"]
    for bad_provenance in (incomplete_provenance, unknown_provenance):
        with pytest.raises(ValueError, match="keys mismatch"):
            run_benchmark_matrix(
                cases,
                _parameters(),
                forecasters,
                cast(Mapping[ModelVariant, ModelProvenance], bad_provenance),
                mode="quick",
            )
