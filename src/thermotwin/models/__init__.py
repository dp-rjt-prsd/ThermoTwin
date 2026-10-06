"""Public configuration, architecture, training, and evaluation APIs."""

from thermotwin.models.comparison import ModelComparisonMetrics, evaluate_physics_model
from thermotwin.models.config import MLPConfig, PhysicsLossConfig, TrainerConfig
from thermotwin.models.evaluation import (
    RegressionMetrics,
    compute_regression_metrics,
    evaluate_model,
)
from thermotwin.models.forecast_evaluation import (
    ForecastEvaluation,
    ForecastHorizonMetrics,
    evaluate_forecast_horizons,
)
from thermotwin.models.forecaster import (
    ForecastConditions,
    ForecastRollout,
    MLPThermalForecaster,
    ThermalForecaster,
)
from thermotwin.models.losses import (
    LossComponents,
    combined_thermal_loss,
    physics_residual_w,
)
from thermotwin.models.mlp import MLPArchitectureMetadata, ThermalMLP
from thermotwin.models.physics_train import (
    PhysicsEpochMetrics,
    PhysicsTrainingResult,
    train_physics_aware_mlp,
)
from thermotwin.models.train import (
    EpochMetrics,
    TrainingResult,
    train_thermal_mlp,
)

__all__ = [
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
