# ThermoTwin

ThermoTwin is a prototype for predictive thermal management of a 12-cell
electric-vehicle battery pack. The physical simulator is the authoritative source
of battery state and thermal behavior; the digital twin is a separate learned model
that will estimate and forecast that behavior for proactive cooling decisions.
The learned model must not replace the simulator.

The current implementation target covers Phases 0-2: Python package and quality
tooling, shared state/configuration contracts, a minimal electrothermal simulator,
and deterministic drive and charging profiles. Predictive control, trained twins,
and dashboard functionality are later phases.

Python 3.11 or newer is required. Install the project and development tools with
`python -m pip install -e ".[dev]"`, then run `python -m pytest`.
