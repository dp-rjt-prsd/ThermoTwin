from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from thermotwin.core.types import (
    CoolingConfig,
    EnvironmentConfig,
    PackConfig,
    SimulationConfig,
)
from thermotwin.data.sampling import sample_dataset_scenario
from thermotwin.data.types import (
    DatasetMetadataValue,
    DatasetSamplingConfig,
    DatasetScenario,
)


def _metadata_float(metadata: Mapping[str, DatasetMetadataValue], key: str) -> float:
    value = metadata[key]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


@pytest.fixture
def base_environment_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        pack=PackConfig(
            rows=3,
            columns=4,
            cell_mass_kg=0.8,
            cell_specific_heat_capacity_j_kg_k=900.0,
            nominal_cell_resistance_ohm=0.015,
            cell_capacity_ah=50.0,
            neighbor_thermal_resistance_k_w=1.5,
            ambient_thermal_resistance_k_w=8.0,
            cell_cooling_surface_area_m2=0.02,
        ),
        cooling=CoolingConfig(
            minimum_coolant_mass_flow_kg_s=0.005,
            maximum_coolant_mass_flow_kg_s=0.09,
            base_heat_transfer_coefficient_w_m2_k=5.0,
            flow_heat_transfer_gain=500.0,
            flow_exponent=0.8,
        ),
        simulation=SimulationConfig(
            default_time_step_s=1.0,
            initial_soc_fraction=0.8,
            initial_cell_temperature_k=298.15,
            ambient_temperature_k=298.15,
            coolant_inlet_temperature_k=293.15,
            initial_current_a=0.0,
        ),
    )


@pytest.fixture
def sampling_config() -> DatasetSamplingConfig:
    return DatasetSamplingConfig(
        duration_s=20.0,
        time_step_s=1.0,
        initial_soc_fraction_range=(0.2, 0.95),
        initial_cell_temperature_k_range=(285.0, 315.0),
        ambient_temperature_k_range=(283.0, 313.0),
        coolant_inlet_temperature_k_range=(285.0, 300.0),
        peak_current_a_range=(10.0, 250.0),
        coolant_mass_flow_kg_s_range=(0.01, 0.08),
        cell_resistance_scale_range=(0.8, 1.8),
        neighbor_thermal_resistance_scale_range=(0.7, 1.5),
        cooling_effectiveness_scale_range=(0.7, 1.4),
        allowed_profile_names=(
            "constant",
            "pulse",
            "stop_and_go",
            "aggressive",
            "fast_charging",
        ),
    )


def test_sampling_is_deterministic_per_seed_and_index(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    first = sample_dataset_scenario(base_environment_config, sampling_config, 18, 7)
    _ = sample_dataset_scenario(base_environment_config, sampling_config, 18, 2)
    again = sample_dataset_scenario(base_environment_config, sampling_config, 18, 7)

    assert first.trajectory_id == again.trajectory_id == "trajectory-000007"
    assert first.metadata["seed"] == 18
    assert first.metadata["trajectory_index"] == 7
    assert first.metadata == again.metadata
    assert first.environment_config == again.environment_config
    assert first.current_profile(0.0) == again.current_profile(0.0)
    assert first.current_profile(3.0) == again.current_profile(3.0)
    assert first.coolant_flow_profile(0.0) == again.coolant_flow_profile(99.0)


def test_sampling_does_not_read_or_mutate_global_rngs(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    np.random.seed(981)
    random.seed(732)
    numpy_before = repr(np.random.get_state())
    python_before = random.getstate()

    sample_dataset_scenario(base_environment_config, sampling_config, 4, 8)

    numpy_after = repr(np.random.get_state())
    python_after = random.getstate()
    assert numpy_before == numpy_after
    assert python_before == python_after


def test_indices_produce_different_sampled_values(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    first = sample_dataset_scenario(base_environment_config, sampling_config, 3, 0)
    second = sample_dataset_scenario(base_environment_config, sampling_config, 3, 1)

    assert first.environment_config != second.environment_config
    assert first.metadata != second.metadata


def test_samples_all_values_within_configured_ranges_and_pack_lengths(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    scenario = sample_dataset_scenario(base_environment_config, sampling_config, 22, 5)
    metadata = scenario.metadata
    config = scenario.environment_config
    temperatures = config.simulation.initial_cell_temperatures_k
    resistances = config.simulation.initial_cell_resistances_ohm

    assert temperatures is not None and len(temperatures) == config.pack.cell_count
    assert resistances is not None and len(resistances) == config.pack.cell_count
    for cell_index, resistance in enumerate(resistances):
        scale = _metadata_float(metadata, f"cell_resistance_scale_{cell_index}")
        assert resistance == pytest.approx(
            config.pack.nominal_cell_resistance_ohm * scale
        )
    assert all(
        sampling_config.initial_cell_temperature_k_range[0]
        <= value
        <= sampling_config.initial_cell_temperature_k_range[1]
        for value in temperatures
    )
    assert (
        sampling_config.initial_soc_fraction_range[0]
        <= config.simulation.initial_soc_fraction
        <= sampling_config.initial_soc_fraction_range[1]
    )
    assert (
        sampling_config.ambient_temperature_k_range[0]
        <= config.simulation.ambient_temperature_k
        <= sampling_config.ambient_temperature_k_range[1]
    )
    assert (
        sampling_config.coolant_inlet_temperature_k_range[0]
        <= config.simulation.coolant_inlet_temperature_k
        <= sampling_config.coolant_inlet_temperature_k_range[1]
    )
    assert (
        sampling_config.peak_current_a_range[0]
        <= _metadata_float(metadata, "peak_current_a")
        <= sampling_config.peak_current_a_range[1]
    )
    assert (
        sampling_config.coolant_mass_flow_kg_s_range[0]
        <= _metadata_float(metadata, "coolant_flow_kg_s")
        <= sampling_config.coolant_mass_flow_kg_s_range[1]
    )
    for cell_index in range(config.pack.cell_count):
        scale = _metadata_float(metadata, f"cell_resistance_scale_{cell_index}")
        assert sampling_config.cell_resistance_scale_range[0] <= scale
        assert scale <= sampling_config.cell_resistance_scale_range[1]
    assert (
        sampling_config.neighbor_thermal_resistance_scale_range[0]
        <= _metadata_float(metadata, "neighbor_thermal_resistance_scale")
        <= sampling_config.neighbor_thermal_resistance_scale_range[1]
    )
    assert (
        sampling_config.cooling_effectiveness_scale_range[0]
        <= _metadata_float(metadata, "cooling_effectiveness_scale")
        <= sampling_config.cooling_effectiveness_scale_range[1]
    )
    assert (
        config.cooling.minimum_coolant_mass_flow_kg_s
        <= _metadata_float(metadata, "coolant_flow_kg_s")
        <= config.cooling.maximum_coolant_mass_flow_kg_s
    )


def test_sampling_preserves_unsampled_base_fields_and_does_not_mutate_base(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    original = base_environment_config
    scenario = sample_dataset_scenario(base_environment_config, sampling_config, 15, 1)
    sampled = scenario.environment_config

    assert base_environment_config is original
    assert base_environment_config.pack == original.pack
    assert sampled.pack.rows == original.pack.rows
    assert sampled.pack.columns == original.pack.columns
    assert (
        sampled.pack.neighbor_thermal_resistance_k_w
        != original.pack.neighbor_thermal_resistance_k_w
    )
    assert sampled.cooling.flow_exponent == original.cooling.flow_exponent
    assert (
        sampled.cooling.base_heat_transfer_coefficient_w_m2_k
        != original.cooling.base_heat_transfer_coefficient_w_m2_k
    )
    assert (
        sampled.cooling.flow_heat_transfer_gain
        != original.cooling.flow_heat_transfer_gain
    )
    assert (
        sampled.simulation.default_time_step_s
        == original.simulation.default_time_step_s
    )
    assert sampled.simulation.initial_current_a == scenario.current_profile(0.0)


@pytest.mark.parametrize(
    ("profile_name", "expected_sign"),
    [
        ("constant", 1),
        ("pulse", 1),
        ("stop_and_go", 0),
        ("aggressive", 1),
        ("fast_charging", -1),
    ],
)
def test_each_profile_kind_uses_existing_factory_and_correct_current_sign(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
    profile_name: str,
    expected_sign: int,
) -> None:
    forced = replace(sampling_config, allowed_profile_names=(profile_name,))
    scenario = sample_dataset_scenario(base_environment_config, forced, 5, 1)
    current = scenario.current_profile(0.0)

    assert type(scenario) is DatasetScenario
    assert scenario.metadata["profile_name"] == profile_name
    if expected_sign == 1:
        assert current > 0.0
    elif expected_sign == -1:
        assert current < 0.0
    else:
        assert current == 0.0
    assert scenario.environment_config.simulation.initial_current_a == current


def test_profile_parameters_and_metadata_reproduce_sampled_callable(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    forced = replace(sampling_config, allowed_profile_names=("pulse",))
    scenario = sample_dataset_scenario(base_environment_config, forced, 9, 4)
    metadata = scenario.metadata

    assert metadata["profile_name"] == "pulse"
    assert metadata["pulse_start_s"] == pytest.approx(sampling_config.duration_s * 0.25)
    assert metadata["pulse_duration_s"] == pytest.approx(
        sampling_config.duration_s * 0.5
    )
    predicted_pulse_current = scenario.current_profile(
        _metadata_float(metadata, "pulse_start_s")
    )
    assert predicted_pulse_current == pytest.approx(
        _metadata_float(metadata, "peak_current_a")
    )
    assert scenario.current_profile(sampling_config.duration_s) == 0.0
    flow = _metadata_float(scenario.metadata, "coolant_flow_kg_s")
    assert scenario.coolant_flow_profile(0.0) == flow
    assert scenario.coolant_flow_profile(sampling_config.duration_s) == flow


@pytest.mark.parametrize(
    "seed,index",
    [(True, 0), (0, True), (-1, 0), (0, -1), (1.5, 0), (0, 2.5)],
)
def test_rejects_invalid_seed_or_trajectory_index(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
    seed: object,
    index: object,
) -> None:
    with pytest.raises(ValueError):
        sample_dataset_scenario(
            base_environment_config,
            sampling_config,
            seed,  # type: ignore[arg-type]
            index,  # type: ignore[arg-type]
        )


def test_rejects_subclasses_instead_of_exact_contract_types(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    class CustomEnvironmentConfig(EnvironmentConfig):
        pass

    with pytest.raises(ValueError, match="EnvironmentConfig"):
        sample_dataset_scenario(
            CustomEnvironmentConfig(
                base_environment_config.pack,
                base_environment_config.cooling,
                base_environment_config.simulation,
            ),
            sampling_config,
            0,
            0,
        )

    class CustomSamplingConfig(DatasetSamplingConfig):
        pass

    with pytest.raises(ValueError, match="DatasetSamplingConfig"):
        sample_dataset_scenario(
            base_environment_config,
            CustomSamplingConfig(
                sampling_config.duration_s,
                sampling_config.time_step_s,
                sampling_config.initial_soc_fraction_range,
                sampling_config.initial_cell_temperature_k_range,
                sampling_config.ambient_temperature_k_range,
                sampling_config.coolant_inlet_temperature_k_range,
                sampling_config.peak_current_a_range,
                sampling_config.coolant_mass_flow_kg_s_range,
                sampling_config.cell_resistance_scale_range,
                sampling_config.neighbor_thermal_resistance_scale_range,
                sampling_config.cooling_effectiveness_scale_range,
                sampling_config.allowed_profile_names,
            ),
            0,
            0,
        )


def test_rejects_requested_coolant_range_outside_base_legal_bounds(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
) -> None:
    invalid = replace(sampling_config, coolant_mass_flow_kg_s_range=(0.0, 0.1))
    with pytest.raises(ValueError, match="coolant"):
        sample_dataset_scenario(base_environment_config, invalid, 0, 0)


@pytest.mark.parametrize("duration,step", [(1.0, 1.0), (7.0, 1.0)])
@pytest.mark.parametrize("profile_name", ["pulse", "stop_and_go", "aggressive"])
def test_derived_profile_durations_are_valid_for_short_and_odd_horizons(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
    duration: float,
    step: float,
    profile_name: str,
) -> None:
    forced = replace(
        sampling_config,
        duration_s=duration,
        time_step_s=step,
        allowed_profile_names=(profile_name,),
    )
    scenario = sample_dataset_scenario(base_environment_config, forced, 2, 9)
    assert (
        scenario.current_profile(0.0)
        == scenario.environment_config.simulation.initial_current_a
    )
    assert scenario.step_count == round(duration / step)
