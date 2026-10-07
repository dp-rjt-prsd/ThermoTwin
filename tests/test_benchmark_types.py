"""Validation tests for Phase 12's immutable benchmark contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from thermotwin.experiments.benchmark_types import (
    FORECAST_HORIZONS_S,
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkCellResult,
    BenchmarkMatrix,
    ControllerMetrics,
    FaultDetectionOutcome,
    ForecastAssumptions,
    ForecastHorizonMetrics,
    ModelEvaluationMetrics,
    ModelProvenance,
    ModelVariant,
    ScenarioId,
    TrainingMode,
    VariantId,
)

_DIGEST = "a" * 64


def _artifact(path: str = "models/model.pt") -> ArtifactReference:
    return ArtifactReference(path=path, sha256=_DIGEST)


def _provenance(
    model_variant: ModelVariant = "data_only",
    training_mode: TrainingMode = "full",
) -> ModelProvenance:
    return ModelProvenance(
        model_variant=model_variant,
        training_mode=training_mode,
        training_seed=4,
        checkpoint=_artifact("models/checkpoint.pt"),
        model_config=_artifact("configs/model.yaml"),
        normalization=_artifact("models/normalization.json"),
        comparison_provenance=_artifact("results/comparison.json"),
        dataset_manifests=(_artifact("data/train.json"),),
    )


def _controller_metrics() -> ControllerMetrics:
    return ControllerMetrics(
        peak_temperature_k=315.0,
        mean_temperature_k=310.0,
        maximum_temperature_spread_k=4.0,
        pump_energy_j=1200.0,
        thermal_violation_count=0,
        thermal_violation_duration_s=0.0,
    )


def _assumptions() -> ForecastAssumptions:
    return ForecastAssumptions(
        current_assumption="current_value_persistence",
        soc_assumption="current_value_persistence",
        ambient_temperature_assumption="current_value_persistence",
        coolant_inlet_temperature_assumption="current_value_persistence",
        action_handling="constant_candidate_sequence",
        uses_future_simulator_truth=False,
    )


def _cell(
    scenario_id: ScenarioId,
    variant_id: VariantId,
    *,
    seed: int = 9,
    digest: str = _DIGEST,
    provenance: ModelProvenance | None = None,
) -> BenchmarkCellResult:
    predictive = variant_id.startswith("predictive_")
    if predictive and provenance is None:
        model_variant: ModelVariant = (
            "data_only" if variant_id == "predictive_data_only" else "physics_informed"
        )
        provenance = _provenance(model_variant)
    return BenchmarkCellResult(
        mode="quick",
        scenario_id=scenario_id,
        variant_id=variant_id,
        seed=seed,
        scenario_configuration_sha256=digest,
        controller_metrics=_controller_metrics(),
        model_provenance=provenance,
        forecast_assumptions=_assumptions() if predictive else None,
        fault_detection=FaultDetectionOutcome(),
    )


def _matrix() -> BenchmarkMatrix:
    return BenchmarkMatrix(
        mode="quick",
        cells=tuple(
            _cell(scenario, variant)
            for scenario in SCENARIO_IDS
            for variant in VARIANT_IDS
        ),
    )


def test_exports_stable_ordered_benchmark_dimensions() -> None:
    assert SCENARIO_IDS == (
        "normal",
        "high_current",
        "hot_ambient",
        "weak_cooling",
        "fast_charging",
        "resistance_fault",
    )
    assert VARIANT_IDS == (
        "threshold",
        "pid",
        "predictive_data_only",
        "predictive_physics_informed",
    )
    assert FORECAST_HORIZONS_S == (30.0, 60.0, 120.0)


def test_artifact_reference_validates_path_and_sha256() -> None:
    assert _artifact().sha256 == _DIGEST
    for path in ("", "   "):
        with pytest.raises(ValueError, match="path"):
            ArtifactReference(path=path, sha256=_DIGEST)
    for digest in ("A" * 64, "a" * 63, "g" * 64):
        with pytest.raises(ValueError, match="sha256"):
            ArtifactReference(path="model.pt", sha256=digest)


def test_provenance_requires_unique_immutable_manifest_references() -> None:
    value = _provenance()
    assert value.dataset_manifests == (_artifact("data/train.json"),)
    with pytest.raises(FrozenInstanceError):
        value.training_seed = 8  # type: ignore[misc]
    with pytest.raises(ValueError, match="dataset_manifests"):
        ModelProvenance(
            model_variant="data_only",
            training_mode="full",
            training_seed=4,
            checkpoint=_artifact(),
            model_config=_artifact(),
            normalization=_artifact(),
            comparison_provenance=_artifact(),
            dataset_manifests=(_artifact(), _artifact()),
        )
    with pytest.raises(ValueError, match="training_seed"):
        _provenance().__class__(
            model_variant="data_only",
            training_mode="full",
            training_seed=True,
            checkpoint=_artifact(),
            model_config=_artifact(),
            normalization=_artifact(),
            comparison_provenance=_artifact(),
            dataset_manifests=(_artifact("data/train.json"),),
        )
    with pytest.raises(ValueError, match="dataset_manifests"):
        ModelProvenance(
            model_variant="data_only",
            training_mode="full",
            training_seed=4,
            checkpoint=_artifact(),
            model_config=_artifact(),
            normalization=_artifact(),
            comparison_provenance=_artifact(),
            dataset_manifests=(),
        )
    source_manifests = [_artifact("data/train.json")]
    provenance = ModelProvenance(
        model_variant="data_only",
        training_mode="full",
        training_seed=4,
        checkpoint=_artifact(),
        model_config=_artifact(),
        normalization=_artifact(),
        comparison_provenance=_artifact(),
        dataset_manifests=source_manifests,  # type: ignore[arg-type]
    )
    source_manifests.clear()
    assert len(provenance.dataset_manifests) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("current_assumption", "future_current"),
        ("soc_assumption", "future_soc"),
        ("ambient_temperature_assumption", "forecast_ambient"),
        ("coolant_inlet_temperature_assumption", "forecast_coolant"),
        ("action_handling", "optimized_per_step"),
        ("uses_future_simulator_truth", True),
    ],
)
def test_forecast_assumptions_fail_closed(field: str, value: object) -> None:
    arguments = {
        "current_assumption": "current_value_persistence",
        "soc_assumption": "current_value_persistence",
        "ambient_temperature_assumption": "current_value_persistence",
        "coolant_inlet_temperature_assumption": "current_value_persistence",
        "action_handling": "constant_candidate_sequence",
        "uses_future_simulator_truth": False,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        ForecastAssumptions(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("peak_temperature_k", True),
        ("peak_temperature_k", float("nan")),
        ("mean_temperature_k", 0.0),
        ("maximum_temperature_spread_k", -1.0),
        ("pump_energy_j", float("inf")),
        ("thermal_violation_count", True),
        ("thermal_violation_count", -1),
        ("thermal_violation_duration_s", -0.1),
    ],
)
def test_controller_metrics_reject_invalid_values(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "peak_temperature_k": 315.0,
        "mean_temperature_k": 310.0,
        "maximum_temperature_spread_k": 4.0,
        "pump_energy_j": 1200.0,
        "thermal_violation_count": 0,
        "thermal_violation_duration_s": 0.0,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        ControllerMetrics(**arguments)  # type: ignore[arg-type]


def test_controller_peak_temperature_cannot_be_below_mean() -> None:
    with pytest.raises(ValueError, match="peak_temperature_k"):
        ControllerMetrics(
            peak_temperature_k=309.0,
            mean_temperature_k=310.0,
            maximum_temperature_spread_k=0.0,
            pump_energy_j=1.0,
            thermal_violation_count=0,
            thermal_violation_duration_s=0.0,
        )


def test_model_evaluation_metrics_enforce_error_and_residual_ordering() -> None:
    valid_metrics = ModelEvaluationMetrics(
        model_variant="data_only",
        distribution="in_distribution",
        mean_absolute_error_k=1.0,
        root_mean_square_error_k=1.2,
        maximum_cell_error_k=2.0,
        mean_absolute_physics_residual_w=0.5,
        root_mean_square_physics_residual_w=0.6,
        maximum_absolute_physics_residual_w=1.0,
        sample_count=10,
    )
    assert valid_metrics.distribution == "in_distribution"
    with pytest.raises(ValueError, match="root_mean_square_error_k"):
        ModelEvaluationMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            mean_absolute_error_k=2.0,
            root_mean_square_error_k=1.0,
            maximum_cell_error_k=3.0,
            mean_absolute_physics_residual_w=1.0,
            root_mean_square_physics_residual_w=2.0,
            maximum_absolute_physics_residual_w=3.0,
            sample_count=10,
        )


def test_model_evaluation_metrics_reject_unknown_enums_and_invalid_counts() -> None:
    with pytest.raises(ValueError, match="model_variant"):
        ModelEvaluationMetrics(
            model_variant="unknown",  # type: ignore[arg-type]
            distribution="out_of_distribution",
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            mean_absolute_physics_residual_w=0.5,
            root_mean_square_physics_residual_w=0.6,
            maximum_absolute_physics_residual_w=1.0,
            sample_count=10,
        )


def test_model_evaluation_metrics_accepts_out_of_distribution_label() -> None:
    metrics = ModelEvaluationMetrics(
        model_variant="physics_informed",
        distribution="out_of_distribution",
        mean_absolute_error_k=1.0,
        root_mean_square_error_k=1.2,
        maximum_cell_error_k=2.0,
        mean_absolute_physics_residual_w=0.5,
        root_mean_square_physics_residual_w=0.6,
        maximum_absolute_physics_residual_w=1.0,
        sample_count=10,
    )
    assert metrics.distribution == "out_of_distribution"
    with pytest.raises(ValueError, match="sample_count"):
        ModelEvaluationMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            mean_absolute_physics_residual_w=0.5,
            root_mean_square_physics_residual_w=0.6,
            maximum_absolute_physics_residual_w=1.0,
            sample_count=True,
        )
    with pytest.raises(ValueError, match="mean_absolute_error_k"):
        ModelEvaluationMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            mean_absolute_error_k=float("nan"),
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            mean_absolute_physics_residual_w=0.5,
            root_mean_square_physics_residual_w=0.6,
            maximum_absolute_physics_residual_w=1.0,
            sample_count=10,
        )
    with pytest.raises(ValueError, match="maximum_cell_error_k"):
        ModelEvaluationMetrics(
            model_variant="data_only",
            distribution="out_of_distribution",
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=2.0,
            maximum_cell_error_k=1.5,
            mean_absolute_physics_residual_w=1.0,
            root_mean_square_physics_residual_w=1.2,
            maximum_absolute_physics_residual_w=2.0,
            sample_count=10,
        )
    with pytest.raises(ValueError, match="maximum_absolute_physics_residual_w"):
        ModelEvaluationMetrics(
            model_variant="physics_informed",
            distribution="out_of_distribution",
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            mean_absolute_physics_residual_w=2.0,
            root_mean_square_physics_residual_w=2.1,
            maximum_absolute_physics_residual_w=2.0,
            sample_count=10,
        )


def test_forecast_horizon_metrics_validate_allowed_horizons_and_bounds() -> None:
    value = ForecastHorizonMetrics(
        model_variant="data_only",
        distribution="in_distribution",
        horizon_s=30.0,
        mean_absolute_error_k=1.0,
        root_mean_square_error_k=1.2,
        maximum_cell_error_k=2.0,
        forecast_count=8,
    )
    assert value.horizon_s == 30.0
    assert value.distribution == "in_distribution"
    with pytest.raises(ValueError, match="horizon_s"):
        ForecastHorizonMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            horizon_s=45.0,
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            forecast_count=8,
        )
    with pytest.raises(ValueError, match="forecast_count"):
        ForecastHorizonMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            horizon_s=30.0,
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=2.0,
            forecast_count=True,
        )


def test_forecast_horizon_metrics_validate_rmse_and_maximum() -> None:
    with pytest.raises(ValueError, match="root_mean_square_error_k"):
        ForecastHorizonMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            horizon_s=60.0,
            mean_absolute_error_k=1.2,
            root_mean_square_error_k=1.0,
            maximum_cell_error_k=2.0,
            forecast_count=8,
        )


def test_forecast_horizon_metrics_accepts_out_of_distribution_label() -> None:
    metrics = ForecastHorizonMetrics(
        model_variant="physics_informed",
        distribution="out_of_distribution",
        horizon_s=120.0,
        mean_absolute_error_k=1.0,
        root_mean_square_error_k=1.2,
        maximum_cell_error_k=2.0,
        forecast_count=8,
    )
    assert metrics.distribution == "out_of_distribution"
    with pytest.raises(ValueError, match="maximum_cell_error_k"):
        ForecastHorizonMetrics(
            model_variant="data_only",
            distribution="in_distribution",
            horizon_s=120.0,
            mean_absolute_error_k=1.0,
            root_mean_square_error_k=1.2,
            maximum_cell_error_k=1.1,
            forecast_count=8,
        )


def test_fault_detection_contract_separates_absent_detection_and_validates_lead() -> (
    None
):
    assert FaultDetectionOutcome(threshold_crossing_time_s=12.0).lead_time_s is None
    outcome = FaultDetectionOutcome(
        suspected_cell_index=4,
        detection_time_s=7.0,
        threshold_crossing_time_s=12.0,
        lead_time_s=5.0,
    )
    assert outcome.suspected_cell_index == 4
    with pytest.raises(ValueError, match="lead_time_s"):
        FaultDetectionOutcome(
            suspected_cell_index=4,
            detection_time_s=7.0,
            threshold_crossing_time_s=12.0,
            lead_time_s=4.0,
        )
    with pytest.raises(ValueError, match="suspected_cell_index"):
        FaultDetectionOutcome(suspected_cell_index=12)
    with pytest.raises(ValueError, match="suspected_cell_index"):
        FaultDetectionOutcome(detection_time_s=2.0)
    with pytest.raises(ValueError, match="detection_time_s"):
        FaultDetectionOutcome(suspected_cell_index=1)
    with pytest.raises(ValueError, match="lead_time_s"):
        FaultDetectionOutcome(lead_time_s=2.0)
    with pytest.raises(ValueError, match="threshold_crossing_time_s"):
        FaultDetectionOutcome(
            suspected_cell_index=1,
            detection_time_s=13.0,
            threshold_crossing_time_s=12.0,
        )
    with pytest.raises(ValueError, match="suspected_cell_index"):
        FaultDetectionOutcome(suspected_cell_index=True, detection_time_s=2.0)
    with pytest.raises(ValueError, match="threshold_crossing_time_s"):
        FaultDetectionOutcome(threshold_crossing_time_s=float("inf"))
    with pytest.raises(ValueError, match="detection_time_s"):
        FaultDetectionOutcome(detection_time_s=-1.0)


def test_predictive_cell_requires_matching_provenance_and_assumptions() -> None:
    with pytest.raises(ValueError, match="model_provenance"):
        BenchmarkCellResult(
            mode="quick",
            scenario_id="normal",
            variant_id="predictive_data_only",
            seed=1,
            scenario_configuration_sha256=_DIGEST,
            controller_metrics=_controller_metrics(),
            model_provenance=_provenance("physics_informed"),
            forecast_assumptions=_assumptions(),
        )
    with pytest.raises(ValueError, match="forecast_assumptions"):
        BenchmarkCellResult(
            mode="quick",
            scenario_id="normal",
            variant_id="predictive_data_only",
            seed=1,
            scenario_configuration_sha256=_DIGEST,
            controller_metrics=_controller_metrics(),
            model_provenance=_provenance(),
        )


def test_reactive_cell_prohibits_model_provenance_and_assumptions() -> None:
    with pytest.raises(ValueError, match="model_provenance"):
        BenchmarkCellResult(
            mode="quick",
            scenario_id="normal",
            variant_id="threshold",
            seed=1,
            scenario_configuration_sha256=_DIGEST,
            controller_metrics=_controller_metrics(),
            model_provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="forecast_assumptions"):
        BenchmarkCellResult(
            mode="quick",
            scenario_id="normal",
            variant_id="threshold",
            seed=1,
            scenario_configuration_sha256=_DIGEST,
            controller_metrics=_controller_metrics(),
            forecast_assumptions=_assumptions(),
        )


def test_full_benchmark_rejects_smoke_model_provenance() -> None:
    with pytest.raises(ValueError, match="training_mode"):
        BenchmarkCellResult(
            mode="full",
            scenario_id="normal",
            variant_id="predictive_data_only",
            seed=1,
            scenario_configuration_sha256=_DIGEST,
            controller_metrics=_controller_metrics(),
            model_provenance=_provenance(training_mode="smoke"),
            forecast_assumptions=_assumptions(),
        )
    quick_result = BenchmarkCellResult(
        mode="quick",
        scenario_id="normal",
        variant_id="predictive_data_only",
        seed=1,
        scenario_configuration_sha256=_DIGEST,
        controller_metrics=_controller_metrics(),
        model_provenance=_provenance(training_mode="smoke"),
        forecast_assumptions=_assumptions(),
    )
    assert quick_result.model_provenance is not None


def test_benchmark_cell_rejects_unknown_ids_boolean_seed_and_bad_digest() -> None:
    base = {
        "mode": "quick",
        "scenario_id": "normal",
        "variant_id": "threshold",
        "seed": 1,
        "scenario_configuration_sha256": _DIGEST,
        "controller_metrics": _controller_metrics(),
    }
    for field, value in (
        ("scenario_id", "unknown"),
        ("variant_id", "unknown"),
    ):
        with pytest.raises(ValueError, match=field):
            BenchmarkCellResult(**(base | {field: value}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        BenchmarkCellResult(**(base | {"seed": True}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="scenario_configuration_sha256"):
        BenchmarkCellResult(
            **(base | {"scenario_configuration_sha256": "X" * 64})  # type: ignore[arg-type]
        )


def test_benchmark_matrix_accepts_complete_order_independent_matrix() -> None:
    matrix = _matrix()
    assert len(matrix.cells) == 24
    assert isinstance(matrix.cells, tuple)
    with pytest.raises(FrozenInstanceError):
        matrix.mode = "full"  # type: ignore[misc]
    source_cells = list(matrix.cells)
    copied_matrix = BenchmarkMatrix(
        mode="quick",
        cells=source_cells,  # type: ignore[arg-type]
    )
    source_cells.clear()
    assert len(copied_matrix.cells) == 24


def test_benchmark_matrix_reports_duplicate_before_missing_pair() -> None:
    cells = list(_matrix().cells)
    cells[-1] = cells[0]
    with pytest.raises(ValueError, match="duplicate"):
        BenchmarkMatrix(mode="quick", cells=tuple(cells))


def test_benchmark_matrix_requires_each_cell_and_consistent_mode_seed_and_digest() -> (
    None
):
    with pytest.raises(ValueError, match="missing"):
        BenchmarkMatrix(mode="quick", cells=_matrix().cells[:-1])
    cells = list(_matrix().cells)
    cells[1] = _cell("normal", "pid", seed=10)
    with pytest.raises(ValueError, match="seed"):
        BenchmarkMatrix(mode="quick", cells=tuple(cells))
    cells = list(_matrix().cells)
    cells[1] = _cell("normal", "pid", digest="b" * 64)
    with pytest.raises(ValueError, match="scenario_configuration_sha256"):
        BenchmarkMatrix(mode="quick", cells=tuple(cells))
    cells = list(_matrix().cells)
    cells[0] = BenchmarkCellResult(
        mode="full",
        scenario_id="normal",
        variant_id="threshold",
        seed=9,
        scenario_configuration_sha256=_DIGEST,
        controller_metrics=_controller_metrics(),
    )
    with pytest.raises(ValueError, match="mode"):
        BenchmarkMatrix(mode="quick", cells=tuple(cells))
