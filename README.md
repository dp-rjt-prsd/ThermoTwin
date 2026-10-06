# ThermoTwin

ThermoTwin is a prototype for predictive thermal management of a 12-cell
electric-vehicle battery pack. The physical simulator is the authoritative source
of battery state and thermal behavior; the digital twin is a separate learned model
that will estimate and forecast that behavior for proactive cooling decisions.
The learned model must not replace the simulator.

The current implementation covers Phases 0-6: Python package and quality tooling,
shared state/configuration contracts, a 12-cell electrothermal simulator,
deterministic drive and charging profiles, threshold and PID cooling controllers,
reproducible controller experiments, trajectory-split synthetic dataset
generation, and a trained pure data-driven thermal surrogate. The surrogate is a
small PyTorch Linear/GELU MLP that predicts one-step cell-temperature deltas from
the current, SOC, cell temperatures, ambient and coolant temperatures, and
coolant flow. Feature and target normalization is fitted on training data only;
training is seeded and saves a reloadable tensor-only checkpoint with its
configuration, normalization, history, evaluation, and provenance artifacts.
Evaluation errors are reported in kelvin. Physics-informed learning, rollout
forecasting, predictive control, time-varying fault detection, and the dashboard
remain later phases; the current model does not include a physics loss or enable
predictive control.

Python 3.11 or newer is required. Install the project and development tools with
`python -m pip install -e ".[dev]"`, then run `python -m pytest`.

Run the accepted integration smoke paths with:

```bash
python scripts/run_baseline.py --smoke-test
python scripts/generate_dataset.py --smoke-test
python scripts/train_model.py --smoke-test
```

The training command accepts optional `--dataset-directory` and
`--output-directory` arguments. The first selects an existing dataset containing
train/validation/test splits and a manifest; when omitted in smoke mode, a small
dataset is generated temporarily. The second selects a new directory for model
artifacts; when omitted in smoke mode, artifacts are written beneath a temporary
directory. Normal training uses `data/processed/dataset-42` by default and saves
to `results/model-training-42` unless these paths are supplied. The baseline and
dataset smoke commands also accept `--output-directory`; their smoke artifacts
otherwise go beneath the operating system's temporary directory.

Training smoke mode verifies the workflow and artifact generation. Its measured
smoke metrics are not a comparative benchmark or a claim of model improvement.
