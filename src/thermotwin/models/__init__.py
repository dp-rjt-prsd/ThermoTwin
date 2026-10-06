"""Public configuration, architecture, training, and evaluation APIs."""

from thermotwin.models.config import MLPConfig, TrainerConfig
from thermotwin.models.evaluation import (
    RegressionMetrics,
    compute_regression_metrics,
    evaluate_model,
)
from thermotwin.models.mlp import MLPArchitectureMetadata, ThermalMLP
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
]
