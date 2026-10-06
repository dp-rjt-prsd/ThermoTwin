"""Generate immutable transitions through the authoritative battery simulator."""

from __future__ import annotations

from collections.abc import Sequence

from thermotwin.core import CoolingAction
from thermotwin.data.types import DatasetScenario, TransitionRecord
from thermotwin.physics.environment import BatteryEnvironment


def generate_trajectory(scenario: DatasetScenario) -> tuple[TransitionRecord, ...]:
    """Simulate one scenario and return its contiguous one-step transitions."""
    if type(scenario) is not DatasetScenario:
        raise TypeError("scenario must be exactly a DatasetScenario")

    environment = BatteryEnvironment(scenario.current_profile)
    state = environment.reset(scenario.environment_config)
    records: list[TransitionRecord] = []
    for step_index in range(scenario.step_count):
        flow = scenario.coolant_flow_profile(state.time_s)
        action = CoolingAction(flow)
        next_state = environment.step(action, scenario.time_step_s)
        records.append(
            TransitionRecord(
                trajectory_id=scenario.trajectory_id,
                step_index=step_index,
                time_step_s=scenario.time_step_s,
                state=state,
                action=action,
                next_state=next_state,
                metadata=scenario.metadata,
            )
        )
        state = next_state
    return tuple(records)


def generate_dataset(
    scenarios: Sequence[DatasetScenario],
) -> tuple[TransitionRecord, ...]:
    """Generate transitions in scenario order after validating the full input."""
    if not isinstance(scenarios, Sequence):
        raise TypeError("scenarios must be a nonempty Sequence of DatasetScenario")

    validated_scenarios = tuple(scenarios)
    if not validated_scenarios:
        raise ValueError("scenarios must be nonempty")

    trajectory_ids: set[str] = set()
    for scenario in validated_scenarios:
        if type(scenario) is not DatasetScenario:
            raise TypeError("every scenario must be exactly a DatasetScenario")
        if scenario.trajectory_id in trajectory_ids:
            raise ValueError("scenario trajectory IDs must be unique")
        trajectory_ids.add(scenario.trajectory_id)

    records: list[TransitionRecord] = []
    for scenario in validated_scenarios:
        records.extend(generate_trajectory(scenario))
    return tuple(records)
