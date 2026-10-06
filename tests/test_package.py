import tomllib
from pathlib import Path

import thermotwin.models as models
import thermotwin.visualization as visualization
from thermotwin import __version__
from thermotwin.data import (
    ModelDatasets,
    NormalizationStats,
    PhysicsDatasets,
    PhysicsTransitionDataset,
    ThermalTransitionDataset,
    TransitionArtifacts,
    build_physics_datasets,
    feature_names,
    fit_normalization,
    load_model_datasets,
    load_transition_artifacts,
    physics_context,
    physics_context_names,
    target_names,
)
from thermotwin.data.physics_dataset import (
    PhysicsDatasets as PhysicsDatasetsModule,
)
from thermotwin.data.physics_dataset import (
    PhysicsTransitionDataset as PhysicsTransitionDatasetModule,
)
from thermotwin.data.physics_dataset import (
    build_physics_datasets as build_physics_datasets_module,
)
from thermotwin.data.physics_dataset import (
    physics_context as physics_context_module,
)
from thermotwin.data.physics_dataset import (
    physics_context_names as physics_context_names_module,
)
from thermotwin.data.transition_artifacts import (
    TransitionArtifacts as TransitionArtifactsModule,
)
from thermotwin.data.transition_artifacts import (
    load_transition_artifacts as load_transition_artifacts_module,
)
from thermotwin.models import (
    EpochMetrics,
    ForecastConditions,
    ForecastEvaluation,
    ForecastHorizonMetrics,
    ForecastRollout,
    LossComponents,
    MLPArchitectureMetadata,
    MLPConfig,
    MLPThermalForecaster,
    ModelComparisonMetrics,
    PhysicsEpochMetrics,
    PhysicsLossConfig,
    PhysicsTrainingResult,
    RegressionMetrics,
    ThermalForecaster,
    ThermalMLP,
    TrainerConfig,
    TrainingResult,
    combined_thermal_loss,
    compute_regression_metrics,
    evaluate_forecast_horizons,
    evaluate_model,
    evaluate_physics_model,
    physics_residual_w,
    train_physics_aware_mlp,
    train_thermal_mlp,
)
from thermotwin.models.comparison import (
    ModelComparisonMetrics as ModelComparisonMetricsModule,
)
from thermotwin.models.comparison import (
    evaluate_physics_model as evaluate_physics_model_module,
)
from thermotwin.models.config import PhysicsLossConfig as PhysicsLossConfigModule
from thermotwin.models.forecast_evaluation import (
    ForecastEvaluation as ForecastEvaluationModule,
)
from thermotwin.models.forecast_evaluation import (
    ForecastHorizonMetrics as ForecastHorizonMetricsModule,
)
from thermotwin.models.forecast_evaluation import (
    evaluate_forecast_horizons as evaluate_forecast_horizons_module,
)
from thermotwin.models.forecaster import (
    ForecastConditions as ForecastConditionsModule,
)
from thermotwin.models.forecaster import (
    ForecastRollout as ForecastRolloutModule,
)
from thermotwin.models.forecaster import (
    MLPThermalForecaster as MLPThermalForecasterModule,
)
from thermotwin.models.forecaster import (
    ThermalForecaster as ThermalForecasterModule,
)
from thermotwin.models.losses import (
    LossComponents as LossComponentsModule,
)
from thermotwin.models.losses import (
    combined_thermal_loss as combined_thermal_loss_module,
)
from thermotwin.models.losses import (
    physics_residual_w as physics_residual_w_module,
)
from thermotwin.models.physics_train import (
    PhysicsEpochMetrics as PhysicsEpochMetricsModule,
)
from thermotwin.models.physics_train import (
    PhysicsTrainingResult as PhysicsTrainingResultModule,
)
from thermotwin.models.physics_train import (
    train_physics_aware_mlp as train_physics_aware_mlp_module,
)
from thermotwin.visualization import (
    controller_comparison_figure,
    coolant_flow_figure,
    fault_diagnostics_figure,
    forecast_comparison_figure,
    pack_heatmap_figure,
    temperature_timeseries_figure,
)
from thermotwin.visualization.data import (
    ControllerTrace as ControllerTraceModule,
)
from thermotwin.visualization.data import (
    DashboardData as DashboardDataModule,
)
from thermotwin.visualization.data import (
    FaultEvent as FaultEventModule,
)
from thermotwin.visualization.data import (
    FaultSummary as FaultSummaryModule,
)
from thermotwin.visualization.data import (
    FaultTrace as FaultTraceModule,
)
from thermotwin.visualization.data import (
    ForecastSnapshot as ForecastSnapshotModule,
)
from thermotwin.visualization.data import (
    load_dashboard_data as load_dashboard_data_module,
)
from thermotwin.visualization.plots import (
    controller_comparison_figure as controller_comparison_figure_module,
)
from thermotwin.visualization.plots import (
    coolant_flow_figure as coolant_flow_figure_module,
)
from thermotwin.visualization.plots import (
    fault_diagnostics_figure as fault_diagnostics_figure_module,
)
from thermotwin.visualization.plots import (
    forecast_comparison_figure as forecast_comparison_figure_module,
)
from thermotwin.visualization.plots import (
    pack_heatmap_figure as pack_heatmap_figure_module,
)
from thermotwin.visualization.plots import (
    temperature_timeseries_figure as temperature_timeseries_figure_module,
)


def test_package_exposes_initial_version() -> None:
    assert __version__ == "0.1.0"


def test_package_declares_dashboard_runtime_dependencies() -> None:
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with project_file.open("rb") as metadata_file:
        project_metadata = tomllib.load(metadata_file)["project"]

    assert "streamlit>=1.31" in project_metadata["dependencies"]
    assert "plotly>=5.20" in project_metadata["dependencies"]


def test_package_exposes_phase_six_data_api() -> None:
    assert NormalizationStats is not None
    assert callable(fit_normalization)
    assert feature_names(12)[0] == "state_current_a"
    assert target_names(12)[-1] == "delta_cell_temperature_11_k"
    assert ThermalTransitionDataset is not None
    assert ModelDatasets is not None
    assert callable(load_model_datasets)


def test_package_exposes_phase_six_model_api() -> None:
    config = MLPConfig(input_dim=17, output_dim=12, hidden_widths=(16,))
    model = ThermalMLP(config)

    assert model.architecture_metadata == MLPArchitectureMetadata(
        input_dim=17,
        output_dim=12,
        hidden_widths=(16,),
        activation="gelu",
    )
    assert TrainerConfig is not None
    assert TrainingResult is not None
    assert EpochMetrics is not None
    assert RegressionMetrics is not None
    assert callable(train_thermal_mlp)
    assert callable(compute_regression_metrics)
    assert callable(evaluate_model)


def test_package_exposes_phase_seven_data_api() -> None:
    import numpy as np

    assert TransitionArtifacts is TransitionArtifactsModule
    assert load_transition_artifacts is load_transition_artifacts_module
    assert PhysicsTransitionDataset is PhysicsTransitionDatasetModule
    assert PhysicsDatasets is PhysicsDatasetsModule
    assert build_physics_datasets is build_physics_datasets_module
    assert physics_context is physics_context_module
    assert physics_context_names is physics_context_names_module
    assert TransitionArtifacts is not None
    assert callable(load_transition_artifacts)
    assert physics_context_names(2) == (
        "time_step_s",
        "cell_thermal_capacity_j_k",
        "net_heat_cell_0_w",
        "net_heat_cell_1_w",
    )

    dataset = PhysicsTransitionDataset(
        features=np.zeros((1, 4), dtype=np.float32),
        targets=np.zeros((1, 2), dtype=np.float32),
        physics_contexts=np.ones((1, 4), dtype=np.float32),
        trajectory_ids=("trajectory-1",),
    )
    splits = PhysicsDatasets(dataset, dataset, dataset)
    assert len(dataset) == 1
    assert splits.train is dataset


def test_package_exposes_phase_eight_model_api() -> None:
    assert models.__all__ == [
        "MLPConfig",
        "TrainerConfig",
        "MLPArchitectureMetadata",
        "ThermalMLP",
        "EpochMetrics",
        "TrainingResult",
        "train_thermal_mlp",
        "RegressionMetrics",
        "compute_regression_metrics",
        "evaluate_model",
        "PhysicsLossConfig",
        "LossComponents",
        "physics_residual_w",
        "combined_thermal_loss",
        "PhysicsEpochMetrics",
        "PhysicsTrainingResult",
        "train_physics_aware_mlp",
        "ModelComparisonMetrics",
        "evaluate_physics_model",
        "ForecastConditions",
        "ForecastRollout",
        "ThermalForecaster",
        "MLPThermalForecaster",
        "ForecastHorizonMetrics",
        "ForecastEvaluation",
        "evaluate_forecast_horizons",
    ]
    assert PhysicsLossConfig is PhysicsLossConfigModule
    assert LossComponents is LossComponentsModule
    assert physics_residual_w is physics_residual_w_module
    assert combined_thermal_loss is combined_thermal_loss_module
    assert PhysicsEpochMetrics is PhysicsEpochMetricsModule
    assert PhysicsTrainingResult is PhysicsTrainingResultModule
    assert train_physics_aware_mlp is train_physics_aware_mlp_module
    assert ModelComparisonMetrics is ModelComparisonMetricsModule
    assert evaluate_physics_model is evaluate_physics_model_module
    assert ForecastConditions is ForecastConditionsModule
    assert ForecastRollout is ForecastRolloutModule
    assert ThermalForecaster is ThermalForecasterModule
    assert MLPThermalForecaster is MLPThermalForecasterModule
    assert ForecastHorizonMetrics is ForecastHorizonMetricsModule
    assert ForecastEvaluation is ForecastEvaluationModule
    assert evaluate_forecast_horizons is evaluate_forecast_horizons_module

    conditions = ForecastConditions(
        current_a=0.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=293.15,
    )
    assert conditions.soc_fraction == 0.5
    config = PhysicsLossConfig(lambda_physics=0.2, residual_scale_w=5.0)
    assert config.residual_scale_w == 5.0
    assert callable(physics_residual_w)
    assert callable(combined_thermal_loss)
    assert callable(train_physics_aware_mlp)
    assert callable(evaluate_physics_model)
    assert callable(evaluate_forecast_horizons)
    assert ThermalForecaster is not None
    assert MLPThermalForecaster is not None


def test_package_exposes_phase_eleven_visualization_api() -> None:
    assert visualization.__all__ == [
        "ControllerTrace",
        "DashboardData",
        "FaultEvent",
        "FaultSummary",
        "FaultTrace",
        "ForecastSnapshot",
        "load_dashboard_data",
        "controller_comparison_figure",
        "coolant_flow_figure",
        "fault_diagnostics_figure",
        "forecast_comparison_figure",
        "pack_heatmap_figure",
        "temperature_timeseries_figure",
    ]
    assert visualization.ControllerTrace is ControllerTraceModule
    assert visualization.DashboardData is DashboardDataModule
    assert visualization.FaultEvent is FaultEventModule
    assert visualization.FaultSummary is FaultSummaryModule
    assert visualization.FaultTrace is FaultTraceModule
    assert visualization.ForecastSnapshot is ForecastSnapshotModule
    assert visualization.load_dashboard_data is load_dashboard_data_module
    assert controller_comparison_figure is controller_comparison_figure_module
    assert coolant_flow_figure is coolant_flow_figure_module
    assert fault_diagnostics_figure is fault_diagnostics_figure_module
    assert forecast_comparison_figure is forecast_comparison_figure_module
    assert pack_heatmap_figure is pack_heatmap_figure_module
    assert temperature_timeseries_figure is temperature_timeseries_figure_module
    assert all(
        callable(builder)
        for builder in (
            controller_comparison_figure,
            coolant_flow_figure,
            fault_diagnostics_figure,
            forecast_comparison_figure,
            pack_heatmap_figure,
            temperature_timeseries_figure,
        )
    )
