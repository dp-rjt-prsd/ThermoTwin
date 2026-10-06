# ThermoTwin

ThermoTwin is a prototype for predictive thermal management of a 12-cell
electric-vehicle battery pack. The physical simulator is the authoritative source
of battery state and thermal behavior; the digital twin is a separate learned model
that estimates and forecasts that behavior for proactive cooling decisions.
The learned model must not replace the simulator.

Phases 0-11, including Checkpoint P11, are implemented and accepted. The system includes validated shared state and
configuration contracts, a 12-cell electrothermal simulator, deterministic drive
and charging profiles, threshold and PID cooling controllers, reproducible
controller experiments, and trajectory-split synthetic datasets. Its thermal
surrogates are small PyTorch Linear/GELU MLPs that predict one-step cell-temperature
deltas from current, SOC, cell temperatures, ambient and coolant temperatures, and
coolant flow. Training-only normalization, seeded training, and tensor-only model
checkpoints are shared by two comparable variants: a data-only model and a
physics-informed model.

The physics-informed training context is derived through the simulator's canonical
heat-generation, conduction, cooling, and ambient-exchange functions. A
differentiable combined data and physics loss retains gradients through model
predictions. The comparison workflow measures both variants on in-distribution and
shifted out-of-distribution data using unchanged in-distribution training
normalization. A model-agnostic `ThermalForecaster` supports one-step inference and
autoregressive rollouts without future cell-temperature leakage. Forecast
evaluation reports physical-unit errors at exact 30, 60, and 120 second horizons.

The current smoke evidence establishes that these workflows execute and produce
artifacts; it does not establish that the physics-informed model is superior. In
the current smoke forecast, error grows substantially at longer horizons, so the
smoke metrics must not be presented as a performance benchmark. The Phase 7-8
integration audit passed its repository quality gates and smoke workflows.

Phase 9 adds receding-horizon coolant-flow selection using model forecasts and
compares predictive data-only and physics-informed twins with reactive threshold
and PID controllers. Phase 10 adds simulator-owned resistance faults and
persistence-based detection of fault-induced thermal anomalies. Phase 11 loads
those saved predictive and fault artifacts into an interactive dashboard. The
dashboard is artifact-driven: it never runs simulation or model training.

The dashboard shows measured and horizon-selectable predicted 3 x 4 pack heatmaps,
temperature and coolant-flow histories, saved reactive-versus-predictive controller
metrics, data-only-versus-physics-informed forecasts, and fault-induced anomaly
evidence including the configured/suspected cell, anomaly score, predicted hotspot,
detection time, measured threshold crossing, and lead time. These synthetic
experiments support claims about forecast and controller comparisons only to the
extent shown in saved run artifacts. Smoke outputs validate workflow execution and
artifact integrity; they do not establish that either learned model or controller
is superior. Do not claim thermal-runaway detection or report performance values
that are not present in saved experiment results.

The Phase 9-11 integration audit passed: 1,265 pytest tests in 373.04s, full Ruff,
and strict mypy for 120 source files passed; fresh predictive and fault smoke
workflows passed; the strict artifact loader accepted four controller traces, two
model forecast sets, and a 30-step fault trace; and the documented dashboard
launcher health check returned HTTP 200 with `ok`. The predictive smoke again
produced identical metrics and zero cooling across all four variants, so it verifies
workflow integrity only. The fault smoke measured detection at 15 s, threshold
crossing at 26 s, 11 s lead, and peak temperature of 317.12793572865644 K. These
smoke results do not demonstrate controller or model superiority. Phase 12, the
benchmark suite, is next; quantitative superiority has not yet been demonstrated.

Python 3.11 or newer is required. Install the project and development tools with
`python -m pip install -e ".[dev]"`, then run `python -m pytest`.

Run the accepted integration smoke paths with:

```bash
python scripts/run_baseline.py --smoke-test
python scripts/generate_dataset.py --smoke-test
python scripts/train_model.py --smoke-test
python scripts/compare_models.py --smoke-test
python scripts/evaluate_forecasts.py --smoke-test
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

`scripts/compare_models.py` trains and evaluates matched data-only and
physics-informed models on in-distribution and shifted OOD datasets. Normal runs
accept `--id-dataset-directory`, `--ood-dataset-directory`, and
`--output-directory`; smoke mode creates temporary input datasets and accepts
`--output-directory`. Its new output directory contains both checkpoints,
configuration and normalization, training histories, ID/OOD metrics, and
provenance. Existing output directories are rejected rather than overwritten.

`scripts/evaluate_forecasts.py` evaluates both model variants on held-out
trajectory forecasts at 30, 60, and 120 seconds. Normal runs accept
`--comparison-directory`, `--dataset-directory`, and `--output-directory`; smoke
mode generates temporary comparison and evaluation inputs and accepts
`--output-directory`. Its output includes forecast metrics by horizon, copied
checkpoints and evaluation inputs, and provenance. Existing output directories
are rejected rather than overwritten. Both new smoke commands are workflow checks
only: their metrics are not evidence of comparative superiority, and the smoke
forecast error grows substantially with horizon.

Run the Phase 9-11 artifact workflow in order. Each selected predictive output
directory must be new; the fault workflow also requires that its
`fault-detection` output subdirectory does not already exist. Choose unused paths
for each run:

```bash
python scripts/run_predictive.py --smoke-test --output-directory results/predictive-smoke
python scripts/run_fault_detection.py --smoke-test --output-directory results/fault-smoke
python scripts/run_dashboard.py --predictive-directory results/predictive-smoke --fault-directory results/fault-smoke
```

The predictive smoke trains temporary comparison models when needed and writes a
`comparison_summary.json` with its controller artifacts. The fault smoke creates
its own temporary model comparison when needed and saves the fault run under
`results/fault-smoke/fault-detection/`. The dashboard launcher validates both
artifact roots and opens Streamlit with those paths; the dashboard then reads the
saved files only. Smoke measurements are workflow checks, not benchmark evidence
for controller superiority. Use saved non-smoke experiment outputs for any
quantitative comparison, and state the tested scenarios and metrics explicitly.
