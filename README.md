# ThermoTwin

ThermoTwin is a prototype for predictive thermal management of a 12-cell
electric-vehicle battery pack. The physical simulator is the authoritative source
of battery state and thermal behavior; the digital twin is a separate learned model
that will estimate and forecast that behavior for proactive cooling decisions.
The learned model must not replace the simulator.

The current implementation covers Phases 0-5: Python package and quality tooling,
shared state/configuration contracts, a 12-cell electrothermal simulator,
deterministic drive and charging profiles, threshold and PID cooling controllers,
reproducible controller experiments, and trajectory-split synthetic dataset
generation. Predictive control, trained twins, fault detection, and dashboard
functionality are later phases.

Python 3.11 or newer is required. Install the project and development tools with
`python -m pip install -e ".[dev]"`, then run `python -m pytest`.

Run the accepted integration smoke paths with:

```bash
python scripts/run_baseline.py --smoke-test
python scripts/generate_dataset.py --smoke-test
```

Both commands accept `--output-directory` when artifacts should be retained at a
specific location. Without it, smoke mode writes beneath the operating system's
temporary directory so the repository remains clean.
