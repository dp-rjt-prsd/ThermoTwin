from __future__ import annotations

from dataclasses import replace

import pytest
from plotly.graph_objects import Figure  # type: ignore[import-untyped]

from thermotwin.experiments.benchmark_types import (
    SCENARIO_IDS,
    VARIANT_IDS,
    ArtifactReference,
    BenchmarkMode,
    ControllerMetrics,
    EvaluationDistribution,
    FaultDetectionOutcome,
    ModelEvaluationMetrics,
    ModelProvenance,
    ModelVariant,
    TrainingMode,
)
from thermotwin.experiments.final_evaluation import (
    EvidenceStatus,
    FinalEvaluation,
    FinalForecastMetrics,
    MetricComparison,
    ScenarioControllerMetrics,
    SourceArtifactDigest,
)
from thermotwin.visualization.benchmark_plots import (
    controller_matrix_figure,
    fault_timing_figure,
    forecast_error_figure,
    model_accuracy_figure,
    physics_residual_figure,
    thermal_energy_tradeoff_figure,
)

_MODEL_VARIANTS: tuple[ModelVariant, ...] = ("data_only", "physics_informed")
_DISTRIBUTIONS: tuple[EvaluationDistribution, ...] = (
    "in_distribution",
    "out_of_distribution",
)
_FORECAST_HORIZONS_S = (30.0, 60.0, 120.0)


def _controller_metrics(variant_index: int, scenario_index: int) -> ControllerMetrics:
    peak = 300.0 + scenario_index + variant_index
    return ControllerMetrics(
        peak_temperature_k=peak,
        mean_temperature_k=peak - 2.0,
        maximum_temperature_spread_k=float(variant_index + 1),
        pump_energy_j=float(scenario_index * 10 + variant_index),
        thermal_violation_count=variant_index,
        thermal_violation_duration_s=float(variant_index * 2),
    )


def _evaluation(
    *,
    mode: BenchmarkMode = "quick",
    fault: FaultDetectionOutcome | None = None,
) -> FinalEvaluation:
    status: EvidenceStatus = (
        "workflow_only" if mode == "quick" else "descriptive_synthetic_benchmark"
    )
    training_mode: TrainingMode = "smoke" if mode == "quick" else "full"

    def reference(label: str) -> ArtifactReference:
        return ArtifactReference(label, "a" * 64)

    provenance = tuple(
        ModelProvenance(
            model_variant=model,
            training_mode=training_mode,
            training_seed=13,
            checkpoint=reference(f"{model}/checkpoint"),
            model_config=reference(f"{model}/config"),
            normalization=reference(f"{model}/normalization"),
            comparison_provenance=reference(f"{model}/comparison"),
            dataset_manifests=(reference(f"{model}/dataset"),),
        )
        for model in _MODEL_VARIANTS
    )
    model_metrics = tuple(
        ModelEvaluationMetrics(
            model_variant=model,
            distribution=distribution,
            mean_absolute_error_k=1.0 + model_index + distribution_index,
            root_mean_square_error_k=2.0 + model_index + distribution_index,
            maximum_cell_error_k=3.0 + model_index + distribution_index,
            mean_absolute_physics_residual_w=4.0 + model_index + distribution_index,
            root_mean_square_physics_residual_w=5.0 + model_index + distribution_index,
            maximum_absolute_physics_residual_w=6.0 + model_index + distribution_index,
            sample_count=10,
        )
        for model_index, model in enumerate(_MODEL_VARIANTS)
        for distribution_index, distribution in enumerate(_DISTRIBUTIONS)
    )
    forecasts = tuple(
        FinalForecastMetrics(
            model_variant=model,
            distribution=distribution,
            horizon_s=horizon,
            mae_kelvin=1.0 + model_index + distribution_index + horizon / 100,
            rmse_kelvin=2.0 + model_index + distribution_index + horizon / 100,
            maximum_absolute_cell_error_kelvin=(
                3.0 + model_index + distribution_index + horizon / 100
            ),
            forecast_count=5,
            step_count=int(horizon),
        )
        for model_index, model in enumerate(_MODEL_VARIANTS)
        for distribution_index, distribution in enumerate(_DISTRIBUTIONS)
        for horizon in _FORECAST_HORIZONS_S
    )
    controllers = tuple(
        ScenarioControllerMetrics(
            scenario_id=scenario,
            variant_id=variant,
            metrics=_controller_metrics(variant_index, scenario_index),
        )
        for scenario_index, scenario in enumerate(SCENARIO_IDS)
        for variant_index, variant in enumerate(VARIANT_IDS)
    )
    return FinalEvaluation(
        evidence_status=status,
        benchmark_mode=mode,
        benchmark_seed=11,
        comparison_seed=13,
        id_dataset_seed=17,
        ood_dataset_seed=19,
        model_provenance=provenance,
        model_metrics=model_metrics,
        forecast_metrics=forecasts,
        controller_metrics=controllers,
        fault_detection=fault or FaultDetectionOutcome(),
        comparisons=(MetricComparison("model", "mae", 1.0, 1.0, "tie"),),
        source_artifacts=(SourceArtifactDigest("saved", "a" * 64),),
        limitations=("Synthetic scenarios; descriptive evidence only.",),
    )


def _assert_evidence_and_limitation(figure: Figure, mode: BenchmarkMode) -> None:
    title = str(figure.layout.title.text) + " " + " ".join(
        str(annotation.text) for annotation in figure.layout.annotations or ()
    )
    expected = "workflow_only" if mode == "quick" else "descriptive_synthetic_benchmark"
    assert expected in title
    assert "synthetic" in title.lower()
    assert "descriptive" in title.lower()
    assert not any(
        term in title.lower()
        for term in ("superior", "significant", "generalizes", "safe")
    )


def test_controller_matrix_uses_all_six_by_four_saved_metrics() -> None:
    evaluation = _evaluation()

    figure = controller_matrix_figure(evaluation)

    assert isinstance(figure, Figure)
    assert len(figure.data) == 24
    assert list(figure.data[0].x) == [
        "Normal",
        "High current",
        "Hot ambient",
        "Weak cooling",
        "Fast charging",
        "Resistance fault",
    ]
    assert list(figure.data[0].y) == [
        evaluation.controller_metrics[index].metrics.peak_temperature_k
        for index in range(0, 24, 4)
    ]
    assert figure.data[0].name == "Threshold"
    assert "pattern" in figure.data[0].marker
    _assert_evidence_and_limitation(figure, "quick")


def test_model_accuracy_and_physics_figures_use_saved_values() -> None:
    evaluation = _evaluation(mode="full")

    accuracy = model_accuracy_figure(evaluation)
    residuals = physics_residual_figure(evaluation)

    assert len(accuracy.data) == 6
    assert list(accuracy.data[0].y) == [
        evaluation.model_metrics[0].mean_absolute_error_k,
        evaluation.model_metrics[1].mean_absolute_error_k,
    ]
    assert len(residuals.data) == 6
    assert list(residuals.data[0].y) == [
        evaluation.model_metrics[0].mean_absolute_physics_residual_w,
        evaluation.model_metrics[1].mean_absolute_physics_residual_w,
    ]
    _assert_evidence_and_limitation(accuracy, "full")
    _assert_evidence_and_limitation(residuals, "full")


def test_forecast_horizon_figure_preserves_horizon_and_model_order() -> None:
    evaluation = _evaluation()

    figure = forecast_error_figure(evaluation)

    assert len(figure.data) == 12
    assert list(figure.data[0].x) == [30.0, 60.0, 120.0]
    expected = [
        item.mae_kelvin
        for item in evaluation.forecast_metrics
        if item.model_variant == "data_only"
        and item.distribution == "in_distribution"
    ]
    assert list(figure.data[0].y) == expected
    _assert_evidence_and_limitation(figure, "quick")


def test_tradeoff_and_fault_timing_show_saved_values_and_unavailable_state() -> None:
    evaluation = _evaluation()

    tradeoff = thermal_energy_tradeoff_figure(evaluation)
    fault = fault_timing_figure(evaluation)

    assert len(tradeoff.data) == 4
    assert list(tradeoff.data[0].x) == [
        evaluation.controller_metrics[index].metrics.pump_energy_j
        for index in range(0, 24, 4)
    ]
    assert list(tradeoff.data[0].y) == [
        evaluation.controller_metrics[index].metrics.peak_temperature_k
        for index in range(0, 24, 4)
    ]
    assert "Unavailable" in str(fault.data[0].cells.values)
    _assert_evidence_and_limitation(tradeoff, "quick")
    _assert_evidence_and_limitation(fault, "quick")


def test_fault_timing_preserves_detected_crossing_and_lead_times() -> None:
    evaluation = _evaluation(
        fault=FaultDetectionOutcome(
            suspected_cell_index=4,
            detection_time_s=7.5,
            threshold_crossing_time_s=10.0,
            lead_time_s=2.5,
        )
    )

    figure = fault_timing_figure(evaluation)

    assert list(figure.data[0].cells.values[1]) == [
        "C5",
        "7.5 s",
        "10.0 s",
        "2.5 s",
    ]


def test_all_builders_reject_wrong_input_type() -> None:
    builders = (
        controller_matrix_figure,
        model_accuracy_figure,
        forecast_error_figure,
        physics_residual_figure,
        thermal_energy_tradeoff_figure,
        fault_timing_figure,
    )
    for builder in builders:
        with pytest.raises(TypeError, match="FinalEvaluation"):
            builder(object())  # type: ignore[arg-type]


def test_exact_model_ties_and_permuted_records_keep_stable_traces() -> None:
    evaluation = _evaluation()
    data_id, data_ood = evaluation.model_metrics[:2]
    physics_id = replace(data_id, model_variant="physics_informed")
    physics_ood = replace(data_ood, model_variant="physics_informed")
    reordered = replace(
        evaluation,
        model_metrics=(
            physics_ood,
            physics_id,
            data_ood,
            data_id,
        ),
        forecast_metrics=tuple(reversed(evaluation.forecast_metrics)),
        controller_metrics=tuple(reversed(evaluation.controller_metrics)),
    )

    tied = model_accuracy_figure(reordered)
    first = model_accuracy_figure(evaluation)
    rollout = forecast_error_figure(reordered)

    assert tied.data[0].name == "Data-only"
    assert tied.data[1].name == "Physics-informed"
    assert list(tied.data[0].y) == list(tied.data[1].y)
    assert first.to_plotly_json() == model_accuracy_figure(evaluation).to_plotly_json()
    assert list(rollout.data[0].x) == [30.0, 60.0, 120.0]
    assert list(rollout.data[0].y) == [
        pytest.approx(1.3),
        pytest.approx(1.6),
        pytest.approx(2.2),
    ]
