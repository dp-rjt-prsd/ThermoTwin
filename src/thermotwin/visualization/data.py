"""Public facade for strict, immutable dashboard artifact loading."""

from __future__ import annotations

from pathlib import Path

from thermotwin.visualization.contracts import (
    ControllerTrace,
    DashboardData,
    FaultEvent,
    FaultSummary,
    FaultTrace,
    ForecastSnapshot,
)
from thermotwin.visualization.fault_data import load_fault_data
from thermotwin.visualization.predictive_data import load_predictive_data


def load_dashboard_data(
    predictive_directory: str | Path, fault_directory: str | Path
) -> DashboardData:
    """Load, validate, and freeze predictive comparison and fault artifacts."""
    controller_traces, model_forecasts = load_predictive_data(predictive_directory)
    fault_trace, fault_summary = load_fault_data(fault_directory)
    return DashboardData(
        controller_traces=controller_traces,
        model_forecasts=model_forecasts,
        fault_trace=fault_trace,
        fault_summary=fault_summary,
    )


__all__ = [
    "ControllerTrace",
    "DashboardData",
    "FaultEvent",
    "FaultSummary",
    "FaultTrace",
    "ForecastSnapshot",
    "load_dashboard_data",
]
