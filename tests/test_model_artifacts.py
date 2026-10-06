from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from thermotwin.core.types import BatteryState, CoolingAction
from thermotwin.data.model_dataset import feature_names, target_names
from thermotwin.data.preprocessing import NormalizationStats
from thermotwin.models.artifacts import ComparisonVariant, load_comparison_forecaster
from thermotwin.models.config import MLPConfig
from thermotwin.models.forecaster import MLPThermalForecaster
from thermotwin.models.mlp import ThermalMLP


def _write_artifacts(directory: Path, variant: str = "data_only") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = MLPConfig(input_dim=8, output_dim=3, hidden_widths=(5,))
    (directory / "model_config.json").write_text(
        json.dumps(
            {
                "input_dim": config.input_dim,
                "output_dim": config.output_dim,
                "hidden_widths": list(config.hidden_widths),
                "activation": config.activation,
            }
        ),
        encoding="utf-8",
    )
    normalization = NormalizationStats(
        feature_names=feature_names(3),
        target_names=target_names(3),
        feature_mean=np.zeros(8),
        feature_scale=np.ones(8),
        target_mean=np.zeros(3),
        target_scale=np.ones(3),
    )
    (directory / "normalization.json").write_text(
        json.dumps(normalization.to_dict()), encoding="utf-8"
    )
    model = ThermalMLP(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(model.state_dict(), directory / f"{variant}.pt")


def _state() -> BatteryState:
    return BatteryState(
        time_s=0.0,
        current_a=12.0,
        soc_fraction=0.5,
        ambient_temperature_k=298.15,
        coolant_inlet_temperature_k=293.15,
        coolant_mass_flow_kg_s=0.05,
        cell_temperatures_k=np.full(3, 300.0),
        cell_resistances_ohm=np.full(3, 0.01),
    )


@pytest.mark.parametrize("variant", ["data_only", "physics_informed"])
def test_loads_each_comparison_variant_as_a_compatible_forecaster(
    tmp_path: Path, variant: str
) -> None:
    _write_artifacts(tmp_path, variant)

    forecaster = load_comparison_forecaster(
        tmp_path,
        cast(ComparisonVariant, variant),
        time_step_s=1.0,
        minimum_coolant_flow_kg_s=0.0,
        maximum_coolant_flow_kg_s=0.2,
    )

    assert isinstance(forecaster, MLPThermalForecaster)
    assert forecaster.time_step_s == 1.0
    assert forecaster.cell_count == 3
    predicted = forecaster.predict_next(_state(), CoolingAction(0.1))
    np.testing.assert_allclose(predicted.cell_temperatures_k, np.full(3, 300.0))


@pytest.mark.parametrize("variant", ["other", "", "DATA_ONLY", None])
def test_rejects_unknown_variant(tmp_path: Path, variant: object) -> None:
    with pytest.raises(ValueError, match="variant"):
        load_comparison_forecaster(
            tmp_path,
            cast(Any, variant),
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("model_config.json", '{"input_dim":8,"input_dim":8}', "duplicate"),
        ("model_config.json", '{"input_dim":NaN}', "nonfinite"),
        ("model_config.json", "{broken", "model_config.json"),
        ("normalization.json", '{"schema_version":Infinity}', "nonfinite"),
        (
            "normalization.json",
            '{"schema_version":1,"schema_version":1}',
            "duplicate",
        ),
    ],
)
def test_rejects_malformed_json_with_path_context(
    tmp_path: Path, filename: str, contents: str, message: str
) -> None:
    _write_artifacts(tmp_path)
    (tmp_path / filename).write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message) as error:
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )
    assert str(tmp_path / filename) in str(error.value)


@pytest.mark.parametrize(
    "missing", ["model_config.json", "normalization.json", "data_only.pt"]
)
def test_missing_artifact_error_names_path(tmp_path: Path, missing: str) -> None:
    _write_artifacts(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(ValueError) as error:
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )
    assert str(tmp_path / missing) in str(error.value)


def test_rejects_non_tensor_checkpoint(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    torch.save({"weights": [1, 2, 3]}, tmp_path / "data_only.pt")

    with pytest.raises(ValueError, match="tensor state_dict"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


def test_rejects_empty_checkpoint_key(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    torch.save({"": torch.ones(1)}, tmp_path / "data_only.pt")

    with pytest.raises(ValueError, match="tensor state_dict"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


def test_rejects_checkpoint_and_model_schema_mismatch(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    torch.save({"wrong.weight": torch.ones(1)}, tmp_path / "data_only.pt")

    with pytest.raises(ValueError, match="state_dict.*schema"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


def test_rejects_extra_model_config_schema_field(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    config_path = tmp_path / "model_config.json"
    config = cast(dict[str, object], json.loads(config_path.read_text()))
    config["unexpected"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="MLPConfig schema"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


def test_rejects_extra_normalization_schema_field(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    normalization_path = tmp_path / "normalization.json"
    normalization = cast(dict[str, object], json.loads(normalization_path.read_text()))
    normalization["unexpected"] = True
    normalization_path.write_text(json.dumps(normalization), encoding="utf-8")

    with pytest.raises(ValueError, match="normalization.json is invalid"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, -1.0),
        (0.2, 0.1),
        (float("nan"), 0.2),
    ],
)
def test_rejects_invalid_forecaster_flow_bounds(
    tmp_path: Path, bounds: tuple[float, float]
) -> None:
    _write_artifacts(tmp_path)

    with pytest.raises(ValueError):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=1.0,
            minimum_coolant_flow_kg_s=bounds[0],
            maximum_coolant_flow_kg_s=bounds[1],
        )


def test_rejects_invalid_forecaster_timestep(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    with pytest.raises(ValueError, match="time_step"):
        load_comparison_forecaster(
            tmp_path,
            "data_only",
            time_step_s=0.0,
            minimum_coolant_flow_kg_s=0.0,
            maximum_coolant_flow_kg_s=0.2,
        )
