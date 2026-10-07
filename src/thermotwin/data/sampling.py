"""Deterministic per-trajectory sampling for synthetic dataset scenarios."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

import numpy as np

from thermotwin.core.types import EnvironmentConfig
from thermotwin.data.types import (
    CoolantFlowProfile,
    DatasetSamplingConfig,
    DatasetScenario,
)
from thermotwin.scenarios.charging import fast_charging_profile
from thermotwin.scenarios.drive_cycles import (
    aggressive_drive_profile,
    constant_current_profile,
    pulse_current_profile,
    stop_and_go_profile,
)


def _validate_index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def sample_dataset_scenario(
    base_environment_config: EnvironmentConfig,
    sampling_config: DatasetSamplingConfig,
    seed: int,
    trajectory_index: int,
) -> DatasetScenario:
    """Sample an immutable scenario with values stable for its seed and index."""
    if type(base_environment_config) is not EnvironmentConfig:
        raise ValueError("base_environment_config must be an EnvironmentConfig")
    if type(sampling_config) is not DatasetSamplingConfig:
        raise ValueError("sampling_config must be a DatasetSamplingConfig")
    seed_value = _validate_index("seed", seed)
    index_value = _validate_index("trajectory_index", trajectory_index)

    flow_lower, flow_upper = sampling_config.coolant_mass_flow_kg_s_range
    legal_flow_lower = base_environment_config.cooling.minimum_coolant_mass_flow_kg_s
    legal_flow_upper = base_environment_config.cooling.maximum_coolant_mass_flow_kg_s
    if flow_lower < legal_flow_lower or flow_upper > legal_flow_upper:
        raise ValueError(
            "coolant_mass_flow_kg_s_range must lie within base cooling legal bounds"
        )

    # Both coordinates seed an independent local generator; no global RNG state is used.
    rng = np.random.default_rng(np.random.SeedSequence([seed_value, index_value]))

    def sample_range(name: str) -> float:
        lower, upper = getattr(sampling_config, name)
        return float(rng.uniform(lower, upper))

    pack = base_environment_config.pack
    simulation = base_environment_config.simulation
    cell_temperatures = tuple(
        sample_range("initial_cell_temperature_k_range") for _ in range(pack.cell_count)
    )
    resistance_scales = tuple(
        sample_range("cell_resistance_scale_range") for _ in range(pack.cell_count)
    )
    neighbor_resistance_scale = sample_range("neighbor_thermal_resistance_scale_range")
    cooling_effectiveness_scale = sample_range("cooling_effectiveness_scale_range")
    soc = sample_range("initial_soc_fraction_range")
    ambient_temperature = sample_range("ambient_temperature_k_range")
    coolant_temperature = sample_range("coolant_inlet_temperature_k_range")
    peak_current = sample_range("peak_current_a_range")
    coolant_flow = sample_range("coolant_mass_flow_kg_s_range")
    profile_name = str(rng.choice(sampling_config.allowed_profile_names))

    duration = sampling_config.duration_s
    profile_parameters: dict[str, str | int | float | bool | None] = {
        "peak_current_a": peak_current,
        "profile_name": profile_name,
    }
    if profile_name == "constant":
        current_profile = constant_current_profile(peak_current, duration)
    elif profile_name == "pulse":
        pulse_start = duration * 0.25
        pulse_duration = duration * 0.5
        base_current = peak_current * 0.25
        profile_parameters.update(
            {
                "base_current_a": base_current,
                "pulse_start_s": pulse_start,
                "pulse_duration_s": pulse_duration,
            }
        )
        current_profile = pulse_current_profile(
            base_current_a=base_current,
            pulse_current_a=peak_current,
            pulse_start_s=pulse_start,
            pulse_duration_s=pulse_duration,
            total_duration_s=duration,
        )
    elif profile_name in ("stop_and_go", "aggressive"):
        # Half of the valid scenario duration is a valid cycle even for one-step runs.
        cycle_duration = duration * 0.5
        profile_parameters["cycle_duration_s"] = cycle_duration
        if profile_name == "stop_and_go":
            current_profile = stop_and_go_profile(
                peak_current, cycle_duration, duration
            )
        else:
            current_profile = aggressive_drive_profile(
                peak_current, cycle_duration, duration
            )
    else:
        profile_parameters["taper_start_fraction"] = 0.7
        profile_parameters["final_current_fraction"] = 0.2
        current_profile = fast_charging_profile(
            peak_current,
            duration,
            taper_start_fraction=0.7,
            final_current_fraction=0.2,
        )

    # The cooling model uses (base coefficient + flow gain * flow**exponent),
    # so scaling both coefficients applies one consistent effectiveness factor.
    sampled_pack = replace(
        pack,
        neighbor_thermal_resistance_k_w=(
            pack.neighbor_thermal_resistance_k_w * neighbor_resistance_scale
        ),
    )
    sampled_cooling = replace(
        base_environment_config.cooling,
        base_heat_transfer_coefficient_w_m2_k=(
            base_environment_config.cooling.base_heat_transfer_coefficient_w_m2_k
            * cooling_effectiveness_scale
        ),
        flow_heat_transfer_gain=(
            base_environment_config.cooling.flow_heat_transfer_gain
            * cooling_effectiveness_scale
        ),
    )
    sampled_simulation = replace(
        simulation,
        initial_soc_fraction=soc,
        ambient_temperature_k=ambient_temperature,
        coolant_inlet_temperature_k=coolant_temperature,
        initial_current_a=float(current_profile(0.0)),
        initial_cell_temperatures_k=cell_temperatures,
        initial_cell_resistances_ohm=tuple(
            pack.nominal_cell_resistance_ohm * scale for scale in resistance_scales
        ),
    )
    sampled_environment = replace(
        base_environment_config,
        pack=sampled_pack,
        cooling=sampled_cooling,
        simulation=sampled_simulation,
    )

    metadata: dict[str, str | int | float | bool | None] = {
        "seed": seed_value,
        "trajectory_index": index_value,
        "duration_s": duration,
        "profile_name": profile_name,
        "coolant_flow_kg_s": coolant_flow,
        "neighbor_thermal_resistance_scale": neighbor_resistance_scale,
        "cooling_effectiveness_scale": cooling_effectiveness_scale,
        "peak_current_a": peak_current,
        **profile_parameters,
    }
    metadata.update(
        {
            f"cell_resistance_scale_{cell_index}": scale
            for cell_index, scale in enumerate(resistance_scales)
        }
    )

    def constant_coolant_flow(_time_s: float) -> float:
        return coolant_flow

    coolant_flow_profile: CoolantFlowProfile = constant_coolant_flow
    segment = sampling_config.coolant_flow_segment_s
    if segment is not None:
        # Sampled after every other quantity so constant-flow scenarios keep the
        # exact values they had before segmented flow existed.
        segment_count = int(np.ceil(duration / segment))
        segment_flows = (coolant_flow,) + tuple(
            float(rng.uniform(flow_lower, flow_upper))
            for _ in range(segment_count - 1)
        )
        metadata["coolant_flow_segment_s"] = segment
        metadata["coolant_flow_segment_count"] = segment_count

        def segmented_coolant_flow(time_s: float) -> float:
            index = min(int(time_s // segment), segment_count - 1)
            return segment_flows[max(index, 0)]

        coolant_flow_profile = segmented_coolant_flow

    return DatasetScenario(
        trajectory_id=f"trajectory-{index_value:06d}",
        environment_config=sampled_environment,
        current_profile=current_profile,
        coolant_flow_profile=coolant_flow_profile,
        duration_s=duration,
        time_step_s=sampling_config.time_step_s,
        metadata=metadata,
    )
