# Implementation Plan: ThermoTwin Phases 0-12

## Overview

This plan records the implemented ThermoTwin foundation and accepted Phase 12 benchmark, saved-evidence aggregation, presentation figures, and release audit through Checkpoint P12C. The six-scenario result is descriptive synthetic evidence; the saved release export is quick-mode `workflow_only`, and it does not establish model/controller superiority, statistical significance, safety certification, or real-world generalization. `AGENTS.md` is authoritative where project documents conflict.

## Frozen Architecture Decisions

- Python 3.11+ with `src/` packaging, pytest, Ruff, and mypy.
- Physical quantities use explicit SI-unit suffixes in public names; temperatures are kelvin internally.
- `BatteryState` and configuration dataclasses are validated contracts. Cell arrays have shape `(rows * columns,)`.
- Positive current discharges the pack; negative current charges it. SOC is bounded to `[0, 1]`.
- A current profile is a pure callable from simulation time in seconds to current in amperes.
- `BatteryEnvironment` is the authoritative simulator and has no dependency on ML, control, diagnostics, or visualization.
- Environment thermal balance includes Joule heating, neighbor conduction, liquid cooling, and ambient exchange. Forward Euler is sufficient for the MVP, with explicit positive-`dt` validation.
- Pack adjacency is defined once in `physics/topology.py` and consumed by simulator physics.
- Controllers share a runtime-checkable `Controller` protocol with `reset()` and `act(state)` so stateful controllers can be reused fairly across experiments.
- Experiment traces contain one immutable post-step record per applied action. Pump energy is integrated over those records, and result serialization refuses to overwrite an existing run.
- Dataset samples are immutable `(state_t, action_t, state_t+1)` transitions. Dataset splitting is performed only on whole trajectory identifiers.
- Phase 5 may configure initial per-cell resistances, but time-varying fault injection and fault labels remain Phase 10 responsibilities.
- Phase 6 uses a small PyTorch MLP to predict one-step cell-temperature deltas. It does not include physics loss, multi-step rollout, or predictive control.
- Phase 6 training uses statistics fitted only on training rows, seeded mini-batches, validation-based best-weight restoration and early stopping, and a non-overwriting artifact directory. Test evaluation reports errors in kelvin.
- Model inputs are ordered explicitly as current, SOC, current cell temperatures, ambient temperature, coolant inlet temperature, and applied coolant flow.
- Feature and target normalization statistics are fitted only on training rows and persisted separately from tensor-only model weights.
- Training artifacts are non-overwriting and include configuration, seed, history, normalization, evaluation metrics, and a `state_dict` checkpoint.
- Phase 7 evaluates the simulator heat balance at the current state using the canonical NumPy physics functions, then combines that fixed physical target with the predicted temperature delta using PyTorch tensor operations. This reuses authoritative physics while preserving gradients through the prediction.
- Transition rows retain ordered raw simulator context and provenance so physics targets stay aligned with normalized training tensors; in-distribution training statistics are reused unchanged for OOD evaluation.
- The physics residual is scaled by an explicit characteristic power in watts before its squared loss is combined with data loss. Both components remain reportable during training.
- Data-only and physics-informed models use identical MLP architecture, initialization seed, splits, normalization, and optimizer settings; only the configured physics-loss weight differs.
- OOD evaluation reuses normalization fitted on the in-distribution training split and never refits statistics on OOD artifacts.
- Phase 8 forecasts autoregressively through a model-agnostic interface. Evaluation may supply recorded future exogenous conditions and actions, but never future cell temperatures; control defaults to persistence when future conditions are unavailable.
- Forecast horizons are exact integer multiples of the trained model timestep. Evaluation reports physical-unit errors for the required 30, 60, and 120 second horizons over held-out trajectories.
- Predictive control evaluates bounded constant-flow candidate sequences through the `ThermalForecaster` interface and applies only the first action. It cannot query simulator truth.
- The predictive objective reports thermal peak, temperature spread, cubic pump energy, and safety penalty components with configurable nonnegative weights.
- Resistance faults are immutable simulator configuration. The environment applies them at a configured cell and start time; controllers and anomaly detectors receive no fault label.
- Anomaly detection compares measured temperatures with one-step forecasts, requires persistence across samples, and records detection cell, score, time, predicted hotspot, and severity.
- Dashboard code consumes validated saved artifacts through visualization-layer loaders and contains no simulator or training logic.
- Phase 12 uses a deterministic six-scenario suite as a descriptive prototype evaluation. It does not establish statistical significance, CI coverage, real-world safety, or generalization beyond the configured synthetic scenarios.
- The threshold controller is the primary reactive baseline; PID is supplemental. Every scenario also runs predictive control with both the data-only and physics-informed twins against the identical scenario definition, seed, initial state, and simulator configuration.
- Weak cooling changes only the legal coolant-flow capacity; it does not secretly change thermal physics, controller weights, or any one variant's inputs.
- Full benchmark runs accept only validated, non-smoke model artifacts with recorded dataset, checkpoint, normalization, training-mode, seed, and configuration provenance. Quick mode remains workflow evidence only.
- Predictive benchmark rollouts disclose their persistence assumption for unknown future current, ambient temperature, coolant-inlet temperature, SOC inputs, and actions. No controller receives future simulator truth.

## Dependency DAG

```text
TASK-00 repository bootstrap
  -> TASK-01 shared state/config contracts
      -> TASK-02 canonical pack topology
      -> TASK-03 cell heat and thermal integration
      -> TASK-04 neighbor conduction
      -> TASK-05 cooling and ambient exchange
      -> TASK-06 SOC evolution
          -> TASK-07 battery environment integration
              -> TASK-08 deterministic drive profiles
              -> TASK-09 deterministic charging profile
                  -> TASK-10 Phase 0-2 integration audit
                      -> TASK-11 controller contract
                          -> TASK-12 threshold controller
                          -> TASK-13 PID controller
                          -> TASK-14 pump-energy model
                              -> TASK-15 Phase 3 integration
                                  -> TASK-16 experiment metrics
                                      -> TASK-17 experiment contracts
                                          -> TASK-18 experiment runner
                                          -> TASK-19 result serialization
                                              -> TASK-20 Phase 4 integration
                                                  -> TASK-21 initial resistance contract
                                                      -> TASK-22 dataset contracts
                                                          -> TASK-23 parameter sampler
                                                          -> TASK-24 trajectory generator
                                                          -> TASK-25 trajectory split and persistence
                                                              -> TASK-26 dataset CLI integration
                                                                  -> TASK-27 Phase 3-5 audit
                                                                      -> TASK-28 ML dependencies and training config
                                                                          -> TASK-29 normalization contract
                                                                              -> TASK-30 dataset loader
                                                                                  -> TASK-31 thermal MLP
                                                                                      -> TASK-32 deterministic trainer
                                                                                          -> TASK-33 model evaluation
                                                                                              -> TASK-34 training CLI integration
                                                                                                  -> TASK-35 Phase 6 audit
                                                                                                      +-> TASK-36 physics-loss configuration --+
                                                                                                      +-> TASK-37 transition/provenance contracts +
                                                                                                                                        -> TASK-38 canonical physics contexts
                                                                                                                                            -> TASK-39 differentiable residual and combined loss
                                                                                                                                                -> TASK-40 matched data-only/physics-informed training
                                                                                                                                                    -> TASK-41 measured ID/OOD comparison
                                                                                                                                                        -> TASK-42 forecaster contract and MLP adapter
                                                                                                                                                            -> TASK-43 autoregressive rollout
                                                                                                                                                                -> TASK-44 exact-horizon evaluation
                                                                                                                                                                    -> TASK-45 Phase 7-8 integration audit
                                                                                                                                                                        +-> TASK-46 predictive-control contracts --+
                                                                                                                                                                        +-> TASK-47 resistance-fault contract ------+--> TASK-49 Phase A audit
                                                                                                                                                                        +-> TASK-48 dashboard dependencies ----------+        |
                                                                                                                                                                                                                          +-> TASK-50 predictive objective -> TASK-51 predictive controller -> TASK-55 predictive comparison
                                                                                                                                                                                                                          +-> TASK-52 simulator fault integration -> TASK-53 persistent anomaly detection -> TASK-54 lead-time metric -> TASK-56 fault workflow
                                                                                                                                                                                                                                                                            TASK-55 + TASK-56 -> TASK-57 dashboard artifact model -> TASK-58 plots -> TASK-59 Streamlit dashboard -> TASK-60 Phase 9-11 audit
                                                                                                                                                                                                                                                                                                                                                                                                    |
                                                                                                                                                                                                                                                                                                                                                                                                    v
TASK-61 empty-split regression -> TASK-62 common snapshot contract -> TASK-63 dashboard source pairing -> TASK-64 accessibility/truth separation -> TASK-65 benchmark contracts
                                                                                                                                                                                     TASK-65 -> TASK-66 six scenarios --+
                                                                                                                                                                                     TASK-65 -> TASK-67 controller factory +--> TASK-68 suite execution/persistence -> TASK-69 benchmark CLI -> TASK-70 final aggregation -> TASK-71 presentation figures -> TASK-72 release audit/docs
```

Tasks with disjoint files may run concurrently only after their shared dependency is accepted. No concurrent task may modify the same file.

## Task List

### Phase 0: Foundation

- TASK-00: Bootstrap package and quality tooling.
- TASK-01: Define validated state, action, and configuration contracts.
- TASK-02: Implement deterministic rectangular pack topology.
- Checkpoint P0: package import, focused tests, full pytest, Ruff, and mypy pass.

### Phase 1: Minimal Electrothermal Simulator

- TASK-03: Implement Joule heating and isolated-cell thermal integration.
- TASK-04: Implement conservative neighbor conduction.
- TASK-05: Implement flow-dependent cooling and ambient heat exchange.
- TASK-06: Implement signed-current coulomb counting.
- TASK-07: Integrate physics into `BatteryEnvironment` and run deterministic physical sanity checks.
- Checkpoint P1: full suite and deterministic simulation pass; zero-current equilibrium, heating, cooling, conduction, SOC, action bounds, and determinism are covered.

### Phase 2: Scenario Generation

- TASK-08: Implement constant, pulse, stop-and-go, and aggressive drive profiles.
- TASK-09: Implement deterministic fast-charging profile.
- TASK-10: Run the Phase 0-2 integration audit and update task status.
- Checkpoint P2: full suite, Ruff, mypy, and a profile-driven simulator smoke run pass.

### Phase 3: Baseline BTMS

- TASK-11: Freeze the controller protocol and illustrative controller configuration.
- TASK-12: Implement the four-stage threshold controller with exact boundary tests.
- TASK-13: Implement a resettable, bounded PID controller targeting pack maximum temperature.
- TASK-14: Implement the configurable cubic pump-power and energy approximation.
- TASK-15: Reconcile control-package exports and run a closed-loop controller comparison smoke test.
- Checkpoint P3: threshold and PID actions always remain legal; reset restores PID determinism; pump calculations have explicit SI units; full quality gates pass.

### Phase 4: Experiment Framework

- TASK-16: Implement experiment metric calculation over post-step records.
- TASK-17: Freeze validated immutable experiment scenario, record, and result contracts.
- TASK-18: Implement `run_experiment(scenario, controller, seed)` using only public simulator/controller interfaces.
- TASK-19: Persist traces, metrics, and provenance as CSV/JSON in a new non-overwriting run directory.
- TASK-20: Add the baseline smoke script and verify identical-scenario fairness across controllers.
- Checkpoint P4: repeated runs are deterministic; controller comparison shares the same scenario and initial state; saved artifacts round-trip without invented values.

### Phase 5: Synthetic Dataset Generation

- TASK-21: Extend initial-condition contracts with optional validated per-cell resistances.
- TASK-22: Freeze immutable dataset scenario and transition-record contracts plus explicit sampling ranges.
- TASK-23: Implement deterministic parameter sampling for state, load, cooling, resistance, coupling, and cooling effectiveness.
- TASK-24: Generate authoritative simulator transitions as `(state_t, action_t, state_t+1)` records.
- TASK-25: Split exclusively by trajectory and persist flat train/validation/test CSV files with a manifest.
- TASK-26: Add `scripts/generate_dataset.py --smoke-test` and verify reproducible artifact generation.
- TASK-27: Run the Phase 3-5 repository audit and update task status.
- Checkpoint P5: no trajectory crosses split boundaries; smoke generation is deterministic; full pytest, Ruff, mypy, baseline, and dataset smoke commands pass.

### Phase 6: Data-Driven Thermal Surrogate

- TASK-28: Add PyTorch/pandas dependencies and explicit training configuration.
- TASK-29: Implement immutable train-fitted normalization statistics.
- TASK-30: Load and validate split CSV artifacts into normalized tensor datasets without future-state leakage.
- TASK-31: Implement the configurable Linear/GELU thermal delta MLP.
- TASK-32: Implement seeded mini-batch training, validation, early stopping, device selection, and tensor-only checkpointing.
- TASK-33: Evaluate physical-unit MAE, RMSE, per-cell error, and worst-cell error.
- TASK-34: Add `scripts/train_model.py --smoke-test` and persist reproducible training artifacts.
- TASK-35: Run the Phase 6 repository audit and update documentation/status.
- Checkpoint P6: repeated CPU smoke training is deterministic; the saved checkpoint reloads; evaluation is calculated in kelvin; full pytest, Ruff, mypy, and all smoke commands pass.

Phase 6 completion evidence: 875 pytest tests passed; Ruff and strict mypy checks passed; baseline, dataset-generation, and model-training smoke commands passed. The training smoke path verified artifact production and kelvin evaluation; its metrics are smoke verification only, not a comparative performance claim.

### Phase 7: Physics-Informed Learning

- TASK-36: Define validated physics-loss configuration with explicit dimensionless weight and watt residual scale.
- TASK-37: Preserve ordered raw transition/provenance context aligned with normalized tensors and reuse unchanged train-fitted normalization for OOD data.
- TASK-38: Compose each transition's heat-balance context through canonical simulator physics functions.
- TASK-39: Compute a differentiable physics residual and combined data/physics loss, retaining gradients through predictions and testing physical signs.
- TASK-40: Train matched data-only and physics-informed MLP variants with the same initialization, data, normalization, and optimizer settings.
- TASK-41: Run and persist measured in-distribution and shifted OOD comparisons, including physical-unit prediction and physics-consistency metrics.
- Checkpoint P7: both variants train from the same initialization and data contract; residual gradients and physical signs are verified; ID/OOD metrics come only from saved outputs.

### Phase 8: Forecast Engine

- TASK-42: Define the model-agnostic `ThermalForecaster` contract and MLP adapter for physical-unit one-step predictions.
- TASK-43: Roll out autoregressively with explicit action/exogenous-condition inputs, persistence defaults, and no future-temperature leakage.
- TASK-44: Measure physical-unit forecast error across held-out trajectories at exact 30, 60, and 120 second horizons.
- TASK-45: Reconcile exports, scripts, and documentation, then run the Phase 7-8 repository audit.
- Checkpoint P8: data-only and physics-informed checkpoints can be loaded behind the same forecaster interface; required-horizon evaluations, full quality gates, and smoke workflows pass.

Phase 7-8 completion evidence: 1,000 pytest tests passed; Ruff and strict mypy checks passed; baseline, dataset-generation, Phase 6 training, data-vs-physics comparison, and exact-horizon forecast smoke commands passed. The smoke comparison did not demonstrate physics-informed superiority, and the smoke forecast error increased substantially from 30 to 120 seconds; these results validate execution and artifact integrity only.

### Phase 9: Predictive Controller

- TASK-46: Freeze validated predictive-controller configuration and immutable candidate-evaluation contracts.
- TASK-49: Audit Phase A contracts and repository quality gates before control logic begins.
- TASK-50: Implement the configurable predictive objective and component accounting.
- TASK-51: Implement bounded receding-horizon candidate evaluation and first-action selection over `ThermalForecaster`.
- TASK-55: Add a reproducible threshold/PID/predictive closed-loop comparison workflow and saved measured artifacts.
- Checkpoint P9: predictive actions stay within limits, use forecasts rather than simulator truth, and are compared fairly with reactive baselines.

### Phase 10: Fault Injection and Detection

- TASK-47: Add immutable resistance-fault configuration without exposing labels to controllers.
- TASK-52: Apply scheduled resistance multiplication inside `BatteryEnvironment` with exact boundary and reset behavior.
- TASK-53: Implement rolling measured-minus-forecast residual statistics with persistent per-cell anomaly decisions.
- TASK-54: Implement threshold-crossing lead-time measurement with explicit unavailable-result behavior.
- TASK-56: Add a reproducible fault-and-detection workflow that persists fault provenance separately from inference inputs.
- Checkpoint P10: the simulator owns the fault; detection is persistence-based; cell, score, time, severity, and measured lead time are recorded.

### Phase 11: Dashboard

- TASK-48: Declare supported Plotly and Streamlit runtime dependencies.
- TASK-57: Load and validate experiment, forecast, and fault artifacts into immutable dashboard-facing data.
- TASK-58: Build reusable Plotly heatmap, temperature/cooling time-series, and comparison figures.
- TASK-59: Integrate metric cards, horizon-selectable predicted heatmap, controller/model comparison, and fault panel in Streamlit without simulation logic.
- TASK-60: Reconcile exports and documentation, then run the Phase 9-11 repository audit and dashboard smoke verification.
- Checkpoint P11: the dashboard renders from saved artifacts and the full suite, Ruff, strict mypy, workflows, and UI smoke path pass.

Phase 9-11 Phase A contract evidence: 1,096 pytest tests passed; Ruff and strict mypy passed. Predictive-control contracts, backward-compatible resistance-fault provenance, and dashboard dependency declarations are accepted. Downstream logic remains gated on these contracts.

Phase 9-10 completion evidence: 1,220 pytest tests passed; Ruff and strict mypy passed; the predictive comparison and fault-detection smoke workflows completed with saved artifacts. The predictive smoke produced identical metrics and zero cooling across all four variants and therefore supports workflow integrity only, not controller superiority. The fresh Phase 9-11 fault smoke measured detection at 15 s, threshold crossing at 26 s, 11 s lead time, and peak temperature of 317.12793572865644 K for its configured synthetic resistance fault.

Phase 9-11 integration audit evidence (TASK-60): full pytest passed with 1,265 tests in 373.04 s; full Ruff passed; strict mypy passed for 120 source files; fresh predictive and fault smoke workflows passed. The strict artifact loader accepted four controller traces, two model forecast sets, and a 30-step fault trace with a saved 11 s lead. The documented dashboard launcher health check returned HTTP 200 with `ok`. The predictive smoke again produced identical metrics and zero cooling across all four variants, so it demonstrates workflow integrity only and does not establish controller superiority. The fault smoke measured detection at 15 s, threshold crossing at 26 s, 11 s lead, and peak temperature of 317.12793572865644 K. No controller or model superiority is demonstrated by this audit. The dashboard health path reads saved artifacts without starting simulation or training.

### Phase 12: Benchmark Suite and Final Evaluation

Phase 12 implements confirmed-evidence repairs, a deterministic six-scenario by four-variant benchmark, saved-evidence aggregation, figures, and a release audit. The result is descriptive synthetic evidence; the saved quick benchmark is workflow-only. Every implementation handoff retains its file whitelist.

- TASK-61: Fix the empty forecast-evaluation split regression before indexing a transition.
  - Likely file scope: `scripts/evaluate_forecasts.py`; `tests/test_forecast_script.py`.
  - Acceptance: an empty test split produces a stable, actionable CLI error without `IndexError`; nonempty behavior and non-overwrite guarantees remain unchanged.
  - Verification: `python -m pytest tests/test_forecast_script.py`.
  - Dependencies: TASK-60.
- TASK-62: Freeze a common-source/common-action forecast snapshot contract.
  - Likely file scope: `scripts/run_predictive.py`; `tests/test_predictive_experiment.py`.
  - Acceptance: data-only and physics-informed snapshots use the exact same explicitly identified measured source state, source time, and legal coolant action at every 30/60/120-second horizon; serialization validates and records that common provenance instead of pairing each twin with its own closed-loop terminal state.
  - Verification: `python -m pytest tests/test_predictive_experiment.py`.
  - Dependencies: TASK-61.
- TASK-63: Pair dashboard measured and predicted heatmaps to the same saved source.
  - Likely file scope: `src/thermotwin/visualization/predictive_data.py`; `src/thermotwin/visualization/contracts.py`; `tests/test_dashboard_data.py`.
  - Acceptance: artifact loading rejects cross-model source/action mismatches, exposes the shared source as immutable view data, and prevents the UI from comparing a selected controller's unrelated final state with a model forecast.
  - Verification: `python -m pytest tests/test_dashboard_data.py`.
  - Dependencies: TASK-62.
- TASK-64: Repair dashboard accessibility and separate inference output from hidden fault truth.
  - Likely file scope: `src/thermotwin/visualization/dashboard.py`; `src/thermotwin/visualization/plots.py`; `tests/test_dashboard.py`; `tests/test_visualization_plots.py`.
  - Acceptance: the measured heatmap is the selected forecast's shared source; labels state source/action/horizon provenance; plots and controls have meaningful accessible labels and do not depend on color alone; the operator-facing suspected-cell panel shows detector output or unavailable, never the configured fault cell as a substitute; configured fault truth is isolated and explicitly labeled as evaluation-only where needed.
  - Verification: `python -m pytest tests/test_dashboard.py tests/test_visualization_plots.py`; run the documented headless dashboard health check.
  - Dependencies: TASK-63.
- TASK-65: Define immutable benchmark and result contracts.
  - Likely file scope: `src/thermotwin/experiments/benchmark_types.py`; `tests/test_benchmark_types.py`.
  - Acceptance: contracts validate scenario identity, seed, variant identity, model/dataset/checkpoint provenance, smoke/full training mode, persistence assumptions, controller metrics, forecast metrics, physics residuals, and optional fault lead time; invalid/missing/nonfinite values and duplicate matrix cells fail clearly; full-mode inputs reject smoke-trained models.
  - Verification: `python -m pytest tests/test_benchmark_types.py`.
  - Dependencies: TASK-64.
- TASK-66: Implement the deterministic six-scenario factory and configuration.
  - Likely file scope: `configs/benchmark.yaml`; `src/thermotwin/scenarios/benchmark.py`; `tests/test_benchmark_scenarios.py`.
  - Acceptance: normal operation, high current, hot ambient, weak cooling, fast charging, and cell-resistance fault scenarios are deterministic and physically valid; weak cooling changes only legal flow capacity; a scenario is constructed once and reused unchanged by all four variants.
  - Verification: `python -m pytest tests/test_benchmark_scenarios.py`.
  - Dependencies: TASK-65.
- TASK-67: Implement the four-variant controller factory and leakage regression tests.
  - Likely file scope: `src/thermotwin/experiments/benchmark_controllers.py`; `tests/test_benchmark_controllers.py`.
  - Acceptance: the matrix contains threshold as the primary reactive baseline, PID as supplemental, predictive plus data-only twin, and predictive plus physics-informed twin; all receive the same public state and scenario while predictive variants use only their forecaster and disclosed future-exogenous persistence, never simulator truth or fault labels.
  - Verification: `python -m pytest tests/test_benchmark_controllers.py`.
  - Dependencies: TASK-65.
- Checkpoint P12A: TASK-61 through TASK-67 focused tests, Ruff, strict mypy, and diff review pass; confirmed evidence/UI defects and all benchmark contracts are accepted before suite execution.
- TASK-68: Execute and persist the complete scenario/variant matrix.
  - Likely file scope: `src/thermotwin/experiments/benchmark.py`; `src/thermotwin/experiments/benchmark_serialization.py`; `tests/test_benchmark.py`; `tests/test_benchmark_serialization.py`.
  - Acceptance: every one of the six scenarios runs all four variants from the identical scenario, seed, initial state, duration, timestep, limits, and simulator parameters; each cell persists non-overwriting trace, metrics, controller/model identity, configuration, seed, and provenance; interruption or an incomplete matrix cannot masquerade as a complete suite.
  - Verification: `python -m pytest tests/test_benchmark.py tests/test_benchmark_serialization.py`.
  - Dependencies: TASK-66, TASK-67, Checkpoint P12A.
- TASK-69: Add benchmark CLI quick/full workflows with user-grade error handling.
  - Likely file scope: `scripts/benchmark.py`; `tests/test_benchmark_script.py`.
  - Acceptance: `--quick` deterministically exercises all six scenarios and four variants but is labeled workflow-only evidence; full mode requires validated non-smoke model artifacts and rejects missing/incompatible provenance; invalid paths, empty splits, existing outputs, and malformed artifacts return concise actionable stderr without a traceback.
  - Verification: `python -m pytest tests/test_benchmark_script.py`; `python scripts/benchmark.py --quick --output-directory <new-temporary-directory>`.
  - Dependencies: TASK-68.
  - Accepted: 7 focused CLI tests passed; scoped Ruff and strict mypy passed. A standalone quick run generated and reloaded all 24 cells. Identical quick reruns with the same model artifacts produced matching manifests and cell artifacts, and full mode rejected smoke-trained provenance before scenario execution. Quick evidence is labeled `workflow_only`; it demonstrates workflow integrity and makes no model or controller superiority claim.
- Checkpoint P12B: a fresh quick 6-by-4 matrix is complete, deterministic, reloadable, and provenance-valid; full-mode smoke-artifact rejection is demonstrated before final aggregation.
  - Accepted: the standalone quick bundle reloaded with 24 cells; deterministic reruns matched saved manifests and cell artifacts; full-mode smoke rejection passed. This is workflow-only evidence and does not establish comparative performance.
- TASK-70: Aggregate the saved final evaluation without unsupported claims.
  - Likely file scope: `src/thermotwin/experiments/final_evaluation.py`; `tests/test_final_evaluation.py`.
  - Acceptance: aggregation reports model ID/OOD MAE, RMSE, maximum cell error, 30/60/120-second rollout error, physics residual, controller peak/mean/spread temperature, pump energy, safety count/duration, and fault detection lead time from saved artifacts only; missing matrix cells or incompatible provenance fail; results describe configured runs and make no CI, significance, generalization, or superiority claim unless measured values support it.
  - Verification: `python -m pytest tests/test_final_evaluation.py`.
  - Dependencies: TASK-69, Checkpoint P12B.
  - Accepted: aggregation consumes the persisted comparison, ID/OOD forecast, benchmark, and fault evidence and rejects incomplete or inconsistent artifacts. The final export records exact values and provenance without selecting an overall winner. The saved evaluation and bundle manifest were repeated byte-identically across all eight published files. The referenced local example is generated under ignored `results/` and is not committed.
- TASK-71: Generate presentation figures directly from saved evaluation artifacts.
  - Likely file scope: `src/thermotwin/visualization/benchmark_plots.py`; `tests/test_benchmark_plots.py`.
  - Acceptance: figures cover the four-variant controller matrix, model ID/OOD and horizon errors, physics residuals, scenario thermal/energy tradeoffs, and fault timing; units, scenario/model labels, and quick/full evidence status are visible and accessible; no plotted number is recomputed from hidden inputs.
  - Verification: `python -m pytest tests/test_benchmark_plots.py`.
  - Dependencies: TASK-70.
  - Accepted: six standalone Plotly HTML figures are generated from the saved final evaluation and included in the integrity-checked export. Repeated export matched byte-for-byte across its eight files.
- TASK-72: Perform an actual-user, browser, CLI, architecture, and release audit and reconcile documentation.
  - Likely file scope: `README.md`; `PLAN.md`; `tasks/plan.md`; `tasks/todo.md`; `scripts/benchmark.py`; `tests/test_benchmark_script.py`; `src/thermotwin/visualization/dashboard.py`; `tests/test_dashboard.py`. Any newly discovered defect outside this sequential scope requires a separately scoped repair before release.
  - Acceptance: a clean user follows README commands from environment setup through quick benchmark, artifact loading, and dashboard interaction; browser inspection covers the 3x4 measured/predicted heatmaps, horizons, controller/model comparisons, cooling trace, responsive layout, accessibility, console errors, and hidden-truth separation; CLI failures are actionable and traceback-free; dependency direction, type/lint/test gates, deterministic reruns, full-mode provenance rejection, non-overwrite behavior, saved numbers, and claims are audited; docs report only generated evidence and disclose predictive exogenous persistence and synthetic/descriptive limitations.
  - Verification: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests scripts`; all documented smoke/quick commands; headless dashboard health plus browser/AppTest journey; `git diff --check`.
  - Dependencies: TASK-71.
  - Accepted: user instructions now document installation, actual CLI options, non-overwriting output paths, the ID/OOD-to-export artifact chain, figures, dashboard variables/launch, and the authoritative health endpoint. The release audit fixed self-contained checkpoint evidence, quick/full separation, CSV/config provenance, strongest detector-event ranking, duplicate YAML rejection, dataset CLI errors, and final export CLI behavior. Quality gates passed: 1,459 pytest tests in 385.55 seconds, repository Ruff, and strict mypy across 138 source files. Dashboard verification passed 21 focused tests and 10 later repaired-dashboard tests; AppTest exercised 24 combinations without exception, and headless health returned HTTP 200 `ok`. No real browser/DevTools was available, so narrow viewport behavior and the accessibility tree were not visually verified; responsive/accessibility risk remains. Saved values and limitations are documented without a superiority claim.
- Checkpoint P12C: final release audit accepted.
  - Acceptance: all Phase 12 jobs are accepted, actual-user instructions match current CLIs, quality gates and available dashboard checks pass, saved outputs support published values, and limitations are explicit.
  - Accepted: TASK-61 through TASK-72 are accepted. The generated quick-mode release export is workflow-only, covers the configured synthetic cases, and carries no claim of overall superiority, statistical significance, safety certification, or real-world generalization. Browser/DevTools visual verification is unavailable and is disclosed; generated `results/` outputs remain ignored and uncommitted.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Unit ambiguity | High | Explicit suffixes and boundary validation |
| Sign mistakes in heat flow | High | Conservation and direction tests |
| Numerically implausible Euler steps | High | Positive timestep checks and integration sanity test |
| Scenario nondeterminism | Medium | Pure time-indexed functions and repeatability tests |
| Physics duplication later in ML | Medium | Keep equations in small canonical physics functions |
| Worker collisions | High | Exact file whitelists and sequential shared-file reconciliation |
| Stateful-controller carryover | High | Require `reset()` and test repeated-run determinism |
| Metric ambiguity | High | Name units and define violations as post-step samples plus elapsed seconds |
| Artifact overwrite/path escape | High | Validate run IDs and reject existing destinations |
| Temporal leakage | High | Split immutable transitions by trajectory ID only |
| Sampling drift | Medium | Seed per trajectory index and persist sampled metadata |
| Train/test normalization leakage | High | Fit statistics only from the training split and verify with deliberately shifted validation/test data |
| Opaque or unsafe checkpoints | High | Save tensor-only `state_dict`; keep model/normalization/provenance in JSON and load with `weights_only=True` |
| Training nondeterminism | Medium | Seed Python/NumPy/PyTorch, use an explicit DataLoader generator, and force CPU in smoke mode |
| Misleading normalized metrics | High | Denormalize deltas and report evaluation errors in kelvin |
| Physics equation drift | High | Precompute heat-balance targets exclusively through canonical simulator physics functions |
| Residual scale dominating training | High | Configure an explicit watt scale and report data and physics components separately |
| Unfair model comparison | High | Hold architecture, initialization, split, normalization, and optimizer fixed while varying only physics weight |
| OOD normalization leakage | High | Apply in-distribution train statistics unchanged to OOD artifacts |
| Autoregressive truth leakage | High | Future temperatures are targets only; rollout receives at most exogenous conditions and actions |
| Horizon/timestep ambiguity | Medium | Require exact integer horizon multiples and report horizons in seconds and steps |
| Predictive-controller truth leakage | High | Depend only on `ThermalForecaster` and the currently observed state; test with a fake forecaster |
| Objective scale imbalance | High | Expose component values and configurable weights in explicit SI-derived units |
| Fault-label leakage | High | Keep fault configuration in simulator/scenario provenance, never detector or controller input |
| Single-sample false alarm | High | Require configurable rolling calibration and consecutive detections |
| Dashboard logic drift | Medium | Load immutable saved artifacts and keep simulation/training outside the UI module |
| Empty or undersized evaluation split | High | Validate before indexing and return an actionable CLI error |
| Mismatched forecast comparison source | High | Require both twins to share one persisted source state and action |
| Benchmark scenario drift | High | Instantiate once and reuse the immutable scenario across all four variants |
| Smoke-model performance claim | High | Persist training mode and reject smoke artifacts in full benchmark mode |
| Future-exogenous leakage | High | Use and disclose persistence for unavailable future conditions; prohibit simulator truth |
| Hidden fault truth in operator UI | High | Separate evaluation provenance from inference-facing suspected-cell output |

## Future Work Beyond Phase 12

- Any statistical generalization study requires repeated independently sampled scenario families and uncertainty analysis beyond this deterministic descriptive benchmark.
