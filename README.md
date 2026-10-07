# ThermoTwin

ThermoTwin is a prototype battery thermal management system for a simulated
12-cell EV pack. The physical simulator remains authoritative. Two learned twins
forecast cell temperatures: a data-only model and a physics-informed model. The
predictive controllers use those forecasts to choose coolant flow and are compared
with a reactive threshold controller and a PID controller.

Phases 0–12, including Checkpoint P12C, are implemented and audited. Phase 12
provides a deterministic six-scenario, four-variant evaluation and artifact-driven
figures. The saved release evaluation is a `quick` benchmark and is labeled
`workflow_only`; it does not establish model or controller superiority.

## Install

Use Python 3.11 or newer from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Ruff and strict mypy are also configured for the repository:

```bash
python -m ruff check .
python -m mypy src tests scripts
```

## Quick benchmark

Run the full six-scenario by four-variant matrix for its configured quick
duration (30 seconds per scenario):

```bash
python scripts/benchmark.py --quick --seed 1729 --output-directory results/benchmark-quick-1729
```

If no `--comparison-directory` is supplied, the command trains smoke models in
an operating-system temporary directory, removes those temporary inputs after
publishing the benchmark, and writes the bundle to the requested output. Quick
mode always reports `workflow_only`, including when you supply full-trained
models. Its purpose is to check the workflow and artifacts, not performance.
Every output directory in these instructions must be new and must not already
exist; commands reject existing outputs rather than overwrite them. Choose fresh
names for reruns.

Without `--quick`, `scripts/benchmark.py` runs the configured 600-second full
suite. Full mode requires `--comparison-directory` containing valid, full-trained
model provenance and rejects smoke-trained checkpoints before running scenarios.
For example, after creating `results/phase12-comparison` with the full workflow
below:

```bash
python scripts/benchmark.py --seed 1729 --comparison-directory results/phase12-comparison --output-directory results/benchmark-full-1729
```

## Small workflow-only release evaluation with full-trained models

The configured synthetic dataset trajectories are 120 seconds at a one-second
timestep. This release chain uses six trajectories for each dataset to reproduce
the small audit input scale. It trains full-mode comparison models, evaluates
30, 60, and 120 second forecasts, runs a quick benchmark, and uses the smoke fault
scenario. It is a workflow-only release evaluation, not a full statistical dataset
or a full-duration benchmark. Run these commands in order. All dataset and result
paths shown are examples and must not already exist. Each command creates a new
artifact directory.

```bash
python scripts/generate_dataset.py --seed 1729 --trajectory-count 6 --output-directory data/processed/phase12-id-1729
python scripts/generate_dataset.py --ood --seed 1730 --trajectory-count 6 --output-directory data/processed/phase12-ood-1730
python scripts/compare_models.py --id-dataset-directory data/processed/phase12-id-1729 --ood-dataset-directory data/processed/phase12-ood-1730 --output-directory results/phase12-comparison
python scripts/evaluate_forecasts.py --comparison-directory results/phase12-comparison --dataset-directory data/processed/phase12-id-1729 --output-directory results/phase12-forecast-id
python scripts/evaluate_forecasts.py --comparison-directory results/phase12-comparison --dataset-directory data/processed/phase12-ood-1730 --output-directory results/phase12-forecast-ood
python scripts/benchmark.py --quick --seed 1729 --comparison-directory results/phase12-comparison --output-directory results/phase12-benchmark-quick
python scripts/run_fault_detection.py --smoke-test --comparison-directory results/phase12-comparison --model-variant physics_informed --seed 1729 --output-directory results/phase12-fault
python scripts/final_evaluation.py --benchmark-directory results/phase12-benchmark-quick --comparison-directory results/phase12-comparison --id-forecast-directory results/phase12-forecast-id --ood-forecast-directory results/phase12-forecast-ood --fault-directory results/phase12-fault/fault-detection --output-directory results/phase12-final-evaluation
```

Each generated dataset contains six configured 120-second synthetic trajectories;
the generator's larger default trajectory count would create a different dataset
and would not reproduce this audit's input scale. The comparison trains both
variants in full mode on the ID dataset and evaluates them on ID and shifted OOD
data. Forecast evaluation requires every test trajectory to contain at least 120
seconds of transitions. The benchmark is quick/workflow-only despite the
full-trained models. The fault run uses `--smoke-test`, matching the saved fault
evidence. The final evaluation command reads and validates the
saved artifacts; it does not retrain models or rerun the benchmark. Its output
contains `evaluation/final_evaluation.json`, `manifest.json`, and six standalone
HTML figures under `figures/`.

To create a descriptive full-benchmark evaluation, replace `--quick` in the
benchmark command with full mode (omit that flag), use a new benchmark output
directory, and pass that directory as `--benchmark-directory` to the final
evaluation exporter. Full mode accepts only full-trained model artifacts. The
export inherits the benchmark's evidence status; it does not upgrade workflow-only
results merely because the checkpoints were trained in full mode.

## Saved release results

The locally generated and verified audit artifact at
`results/phase12-final-evaluation` is a quick, `workflow_only` evaluation using
full-trained models. Values below are descriptive summaries from that saved
synthetic run; the command chain above describes its small workflow and is not a
statistical study. Model errors are in kelvin;
physics residuals are in watts. Each forecast result contains one forecast per
distribution and horizon in this small release artifact.

| One-step test data | Data-only MAE / RMSE (K) | Physics-informed MAE / RMSE (K) |
|---|---:|---:|
| In distribution | 0.342359 / 0.443430 | 0.363450 / 0.461772 |
| Out of distribution | 3.290927 / 3.387537 | 3.167005 / 3.260634 |

| Mean absolute physics residual | Data-only (W) | Physics-informed (W) |
|---|---:|---:|
| In distribution | 246.499 | 261.684 |
| Out of distribution | 2369.467 | 2280.244 |

| 120-second rollout MAE | Data-only (K) | Physics-informed (K) |
|---|---:|---:|
| In distribution | 30.192248 | 32.958024 |
| Out of distribution | 241.878547 | 245.381359 |

The six-scenario controller summaries are means across the configured scenarios.
All four variants recorded 15 thermal-violation samples totaling 15 seconds in
the saved benchmark.

| Controller variant | Mean peak temperature (K) | Mean pump energy (J) |
|---|---:|---:|
| Reactive threshold | 306.986370 | 288.160417 |
| PID | 307.110424 | 50.239445 |
| Predictive with data-only twin | 307.042039 | 141.255697 |
| Predictive with physics-informed twin | 307.105105 | 33.256022 |

The saved fault case suspected cell index 5 (C6), detected at 15 seconds, crossed
the configured threshold at 26 seconds, and recorded an 11-second lead. This is
fault-induced thermal anomaly evidence, not thermal-runaway detection.

These values do not establish an overall winner, superiority, statistical
significance, safety certification, or real-world generalization. The 120-second
rollout errors are very large, especially OOD, and limit the forecasts' usefulness
at that horizon. The evaluation covers six configured synthetic scenarios only;
it is not a confidence interval or a population-level result. The fault artifact
does not contain checkpoint/configuration hashes or an evidence-mode field, so
the detector model cannot be cryptographically bound to the twin used in the
evaluation.

The `results/` directory is generated local output and is ignored by Git. These
sample artifacts and their figures are not committed with the source repository.

## Dashboard

The dashboard reads saved predictive and fault artifacts; it does not run
simulation or training. For a dashboard-specific pair of runs, create new output
directories using the same full comparison directory:

```bash
python scripts/run_predictive.py --comparison-directory results/phase12-comparison --seed 1729 --output-directory results/phase12-dashboard-predictive
python scripts/run_fault_detection.py --comparison-directory results/phase12-comparison --model-variant physics_informed --seed 1729 --output-directory results/phase12-dashboard-fault
python scripts/run_dashboard.py --predictive-directory results/phase12-dashboard-predictive --fault-directory results/phase12-dashboard-fault
```

`run_dashboard.py` validates the artifacts and sets
`THERMOTWIN_PREDICTIVE_RESULTS` and `THERMOTWIN_FAULT_RESULTS` for Streamlit.
The wrapper is the recommended launch method. To launch Streamlit directly from
PowerShell, set those variables to the predictive and fault artifact directories
first:

```powershell
$env:THERMOTWIN_PREDICTIVE_RESULTS = "results/phase12-dashboard-predictive"
$env:THERMOTWIN_FAULT_RESULTS = "results/phase12-dashboard-fault"
python -m streamlit run src/thermotwin/visualization/dashboard.py
```

The authoritative health endpoint is `http://localhost:8501/_stcore/health`; a
healthy server returns HTTP 200 with `ok`.

The export's self-contained Plotly figures are available without launching the
dashboard:

- `figures/controller_matrix.html`
- `figures/model_accuracy.html`
- `figures/forecast_errors.html`
- `figures/physics_residuals.html`
- `figures/thermal_energy_tradeoff.html`
- `figures/fault_timing.html`

## Audit status and limitations

The release audit passed 1,459 pytest tests in 385.55 seconds, repository Ruff,
and strict mypy across 138 source files. The final export was repeated
byte-identically across its eight published files. Dashboard checks included 21
focused tests and 10 later dashboard tests; Streamlit AppTest exercised 24
control combinations without an exception. The headless health endpoint returned
HTTP 200 with `ok`.

No real browser/DevTools tool was available for this audit. The narrow viewport
and accessibility tree were therefore not visually verified, so responsive and
accessibility risks remain. AppTest and the health endpoint confirm application
execution, not a complete browser accessibility review.

The actual-user audit also repaired checkpoint evidence being self-contained,
quick/full benchmark mode decoupling, real forecast CSV and configuration
provenance, strongest detector-event ranking, duplicate YAML key rejection,
dataset CLI error handling, and the final evaluation export CLI. Predictive
forecasts persist current exogenous inputs during rollout; future simulator truth
and fault labels are not supplied to the predictor or controller.
