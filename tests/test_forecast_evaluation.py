from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from thermotwin.core.types import BatteryState, CoolingAction
from thermotwin.data.types import TransitionRecord
from thermotwin.models.evaluation import RegressionMetrics
from thermotwin.models.forecast_evaluation import (
    ForecastEvaluation,
    ForecastHorizonMetrics,
    evaluate_forecast_horizons,
)
from thermotwin.models.forecaster import (
    ForecastConditions,
    ForecastRollout,
    ThermalForecaster,
)


def _state(
    time_s: float,
    current_a: float,
    soc_fraction: float,
    ambient_temperature_k: float,
    coolant_temperature_k: float,
    coolant_flow: float,
    temperatures: tuple[float, ...],
) -> BatteryState:
    return BatteryState(
        time_s=time_s,
        current_a=current_a,
        soc_fraction=soc_fraction,
        ambient_temperature_k=ambient_temperature_k,
        coolant_inlet_temperature_k=coolant_temperature_k,
        coolant_mass_flow_kg_s=coolant_flow,
        cell_temperatures_k=np.asarray(temperatures, dtype=np.float64),
        cell_resistances_ohm=np.full(len(temperatures), 0.01, dtype=np.float64),
    )


def _records(
    trajectory_id: str,
    count: int,
    *,
    base_temperature: float = 300.0,
    time_step_s: float = 1.0,
    observed_offsets: tuple[float, float] = (0.0, 0.0),
) -> tuple[TransitionRecord, ...]:
    records: list[TransitionRecord] = []
    for index in range(count):
        flow = 0.1 + index * 0.01
        start = _state(
            index * time_step_s,
            10.0 + index,
            0.8 - index * 0.001,
            295.0 + index,
            285.0 + index,
            0.0 if index == 0 else 0.1 + (index - 1) * 0.01,
            (base_temperature + index * 0.1, base_temperature + 1 + index * 0.1),
        )
        following = _state(
            (index + 1) * time_step_s,
            20.0 + index,
            0.7 - index * 0.001,
            305.0 + index,
            290.0 + index,
            flow,
            (
                base_temperature + (index + 1) * 0.1 + observed_offsets[0],
                base_temperature + 1 + (index + 1) * 0.1 + observed_offsets[1],
            ),
        )
        records.append(
            TransitionRecord(
                trajectory_id,
                index,
                time_step_s,
                start,
                CoolingAction(flow),
                following,
            )
        )
    # Ensure exact state/next-state continuity while retaining intentionally
    # varying post-action conditions at each recorded step.
    for index in range(1, len(records)):
        prior = records[index - 1]
        current = records[index]
        records[index] = TransitionRecord(
            trajectory_id,
            index,
            time_step_s,
            prior.next_state,
            current.action,
            current.next_state,
        )
    return tuple(records)


class RecordingForecaster:
    time_step_s = 1.0
    cell_count = 2

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                BatteryState,
                tuple[CoolingAction, ...],
                tuple[ForecastConditions, ...] | None,
            ]
        ] = []

    def predict_next(self, state: BatteryState, action: CoolingAction) -> BatteryState:
        raise NotImplementedError

    def rollout(
        self,
        initial_state: BatteryState,
        actions: tuple[CoolingAction, ...],
        future_conditions: tuple[ForecastConditions, ...] | None = None,
    ) -> ForecastRollout:
        self.calls.append((initial_state, actions, future_conditions))
        states: list[BatteryState] = []
        for index, action in enumerate(actions, start=1):
            condition = (
                future_conditions[index - 1]
                if future_conditions is not None
                else ForecastConditions.from_state(initial_state)
            )
            states.append(
                _state(
                    initial_state.time_s + index,
                    condition.current_a,
                    condition.soc_fraction,
                    condition.ambient_temperature_k,
                    condition.coolant_inlet_temperature_k,
                    action.coolant_mass_flow_kg_s,
                    (
                        float(initial_state.cell_temperatures_k[0]) + 0.1 * index,
                        float(initial_state.cell_temperatures_k[1]) + 0.1 * index,
                    ),
                )
            )
        return ForecastRollout(initial_state, actions, tuple(states))


def test_evaluates_exact_horizons_once_per_trajectory_with_physical_errors() -> None:
    first = _records("first", 120, observed_offsets=(1.0, -1.0))
    second = _records(
        "second", 120, base_temperature=310.0, observed_offsets=(3.0, 1.0)
    )
    forecaster = RecordingForecaster()

    result = evaluate_forecast_horizons(forecaster, first + second)

    assert [item.horizon_s for item in result.horizons] == [30.0, 60.0, 120.0]
    assert [item.step_count for item in result.horizons] == [30, 60, 120]
    assert [item.forecast_count for item in result.horizons] == [2, 2, 2]
    assert len(forecaster.calls) == 6
    assert [len(call[1]) for call in forecaster.calls] == [30, 30, 60, 60, 120, 120]
    assert all(
        call[0] is first[0].state or call[0] is second[0].state
        for call in forecaster.calls
    )

    for horizon in result.horizons:
        assert horizon.metrics.mae_kelvin == pytest.approx(1.5)
        assert horizon.metrics.rmse_kelvin == pytest.approx(np.sqrt(3.0))
        assert horizon.metrics.per_cell_mae_kelvin == pytest.approx((2.0, 1.0))
    serialized_horizons = cast(dict[str, object], result.to_dict()["horizons"])
    serialized_30 = cast(dict[str, object], serialized_horizons["30.0"])
    assert serialized_30["forecast_count"] == 2


def test_uses_only_recorded_actions_and_post_action_exogenous_conditions() -> None:
    records = _records("trajectory", 30)
    forecaster = RecordingForecaster()

    evaluate_forecast_horizons(forecaster, records, horizons_s=(30.0,))

    initial, actions, conditions = forecaster.calls[0]
    assert conditions is not None
    assert initial is records[0].state
    assert actions == tuple(record.action for record in records)
    assert conditions == tuple(
        ForecastConditions.from_state(record.next_state) for record in records
    )
    assert conditions[0].current_a == records[0].next_state.current_a
    assert conditions[0].current_a != records[0].state.current_a
    assert not hasattr(conditions[0], "cell_temperatures_k")


def test_metric_value_types_validate_consistency_and_are_json_friendly() -> None:
    metric = RegressionMetrics(0.0, 0.0, (0.0, 0.0), (0.0, 0.0), 0.0, 0)
    horizon = ForecastHorizonMetrics(30.0, 30, 2, metric)
    result = ForecastEvaluation((horizon,))

    assert horizon.to_dict()["metrics"] == metric.to_dict()
    serialized_horizons = cast(dict[str, object], result.to_dict()["horizons"])
    serialized_30 = cast(dict[str, object], serialized_horizons["30.0"])
    assert serialized_30["horizon_s"] == 30.0
    with pytest.raises((FrozenInstanceError, AttributeError)):
        horizon.step_count = 31  # type: ignore[misc]
    with pytest.raises(ValueError):
        ForecastEvaluation(())
    with pytest.raises(ValueError):
        ForecastEvaluation((horizon, horizon))
    with pytest.raises(ValueError):
        ForecastEvaluation((horizon, ForecastHorizonMetrics(60, 60, 3, metric)))


@pytest.mark.parametrize(
    "horizons",
    [
        (30.5,),
        (60.0, 30.0),
        (30.0, 30.0),
        [],
    ],
)
def test_rejects_fractional_duplicate_unsorted_or_mutable_horizons(
    horizons: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        evaluate_forecast_horizons(
            RecordingForecaster(),
            _records("trajectory", 120),
            horizons,  # type: ignore[arg-type]
        )


def test_rejects_horizon_not_integral_multiple_of_timestep() -> None:
    forecaster = RecordingForecaster()
    forecaster.time_step_s = 0.7
    with pytest.raises(ValueError, match="integer multiple"):
        evaluate_forecast_horizons(
            forecaster,
            _records("trajectory", 120, time_step_s=0.7),
            horizons_s=(30.0,),
        )


def test_rejects_insufficient_trajectory_records() -> None:
    with pytest.raises(ValueError, match="at least 30 records"):
        evaluate_forecast_horizons(
            RecordingForecaster(), _records("short", 29), horizons_s=(30.0,)
        )


def test_rejects_invalid_steps_timestep_and_discontinuous_state_chain() -> None:
    valid = _records("trajectory", 31)
    skipped = (valid[0],) + tuple(
        TransitionRecord(
            record.trajectory_id,
            record.step_index + 1,
            record.time_step_s,
            record.state,
            record.action,
            record.next_state,
        )
        for record in valid[1:]
    )
    with pytest.raises(ValueError, match="step indices"):
        evaluate_forecast_horizons(RecordingForecaster(), skipped, horizons_s=(30.0,))

    changed_dt = list(valid)
    changed_dt[1] = TransitionRecord(
        "trajectory",
        1,
        2.0,
        valid[1].state,
        valid[1].action,
        _state(3.0, 20.0, 0.7, 305.0, 290.0, 0.11, (300.2, 301.2)),
    )
    with pytest.raises(ValueError, match="timestep"):
        evaluate_forecast_horizons(
            RecordingForecaster(), tuple(changed_dt), horizons_s=(30.0,)
        )

    changed_state = _state(1.0, 999.0, 0.5, 300.0, 280.0, 0.1, (301.0, 302.0))
    broken = (
        valid[0],
        TransitionRecord(
            "trajectory", 1, 1.0, changed_state, valid[1].action, valid[1].next_state
        ),
    ) + valid[2:]
    with pytest.raises(ValueError, match="continuous"):
        evaluate_forecast_horizons(RecordingForecaster(), broken, horizons_s=(30.0,))


def test_rejects_wrong_cell_count_and_non_tuple_records() -> None:
    forecaster = RecordingForecaster()
    forecaster.cell_count = 3
    with pytest.raises(ValueError, match="cell count"):
        evaluate_forecast_horizons(
            forecaster, _records("trajectory", 30), horizons_s=(30.0,)
        )
    with pytest.raises((TypeError, ValueError), match="tuple"):
        evaluate_forecast_horizons(
            RecordingForecaster(),
            cast(tuple[TransitionRecord, ...], list(_records("trajectory", 30))),
            horizons_s=(30.0,),
        )


def test_rejects_object_that_does_not_conform_to_forecaster_protocol() -> None:
    with pytest.raises(TypeError, match="ThermalForecaster"):
        evaluate_forecast_horizons(
            cast(ThermalForecaster, object()),
            _records("trajectory", 30),
            horizons_s=(30.0,),
        )
