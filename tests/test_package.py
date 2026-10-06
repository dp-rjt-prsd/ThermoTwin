from thermotwin import __version__
from thermotwin.data import (
    ModelDatasets,
    NormalizationStats,
    ThermalTransitionDataset,
    feature_names,
    fit_normalization,
    load_model_datasets,
    target_names,
)
from thermotwin.models import (
    EpochMetrics,
    MLPArchitectureMetadata,
    MLPConfig,
    RegressionMetrics,
    ThermalMLP,
    TrainerConfig,
    TrainingResult,
    compute_regression_metrics,
    evaluate_model,
    train_thermal_mlp,
)


def test_package_exposes_initial_version() -> None:
    assert __version__ == "0.1.0"


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
