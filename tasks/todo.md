# ThermoTwin Phase 0-11 Tasks

## Phase 0

- [x] TASK-00 Bootstrap package and quality tooling.
  - Acceptance: Python package imports; pytest discovers a smoke test; `.gitignore`, metadata, dependency groups, Ruff, and mypy configuration exist.
  - Verify: `python -m pytest tests/test_package.py`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: none.
- [x] TASK-01 Define validated shared contracts and default YAML configuration.
  - Acceptance: explicit-unit state/action/config dataclasses validate shapes, ranges, and flow bounds; defaults describe a 3x4 pack.
  - Verify: `python -m pytest tests/test_types.py`.
  - Dependencies: TASK-00.
- [x] TASK-01A Add optional validated per-cell initial temperatures for public environment setup.
  - Acceptance: scalar initialization remains the default; tuple override is immutable, positive, finite, and length-checked later by the environment against pack size.
  - Verify: `python -m pytest tests/test_types.py`.
  - Dependencies: TASK-01.
- [x] TASK-02 Implement canonical rectangular topology.
  - Acceptance: 3x4 grid yields 12 cells and 17 unique undirected edges in deterministic order; invalid dimensions fail.
  - Verify: `python -m pytest tests/test_topology.py`.
  - Dependencies: TASK-01.
- [x] Checkpoint P0: full pytest, Ruff, and mypy pass.

## Phase 1

- [x] TASK-03 Implement Joule heating and isolated-cell thermal integration.
  - Acceptance: zero current, current-squared scaling, resistance scaling, and analytical temperature update are tested.
  - Verify: `python -m pytest tests/test_cell.py`.
  - Dependencies: TASK-01.
- [x] TASK-04 Implement conservative neighbor conduction.
  - Acceptance: equal temperatures exchange no heat; heat flows hot to cold; net pairwise energy is zero.
  - Verify: `python -m pytest tests/test_pack.py`.
  - Dependencies: TASK-02.
- [x] TASK-05 Implement flow-dependent cooling and ambient exchange.
  - Acceptance: coefficient and heat removal increase monotonically with legal flow; signs reverse correctly across coolant/ambient temperatures.
  - Verify: `python -m pytest tests/test_cooling.py`.
  - Dependencies: TASK-01.
- [x] TASK-06 Implement signed-current coulomb counting.
  - Acceptance: discharge lowers SOC, charge raises SOC, zero current preserves SOC, and bounds are respected.
  - Verify: `python -m pytest tests/test_electrical.py`.
  - Dependencies: TASK-01.
- [x] TASK-07 Integrate `BatteryEnvironment`.
  - Acceptance: reset/step satisfy frozen contracts; all heat terms and SOC are integrated; action and timestep bounds are enforced; deterministic runs match.
  - Verify: `python -m pytest tests/test_environment.py`.
  - Dependencies: TASK-03, TASK-04, TASK-05, TASK-06.
- [x] Checkpoint P1: full pytest, Ruff, mypy, and deterministic physical sanity run pass.

## Phase 2

- [x] TASK-08 Implement deterministic drive profiles.
  - Acceptance: constant, pulse, stop-and-go, and aggressive profiles obey duration/current contracts and are repeatable.
  - Verify: `python -m pytest tests/test_drive_cycles.py`.
  - Dependencies: TASK-07.
- [x] TASK-09 Implement deterministic fast-charging profile.
  - Acceptance: charging current follows the negative-current convention, tapers safely, obeys duration, and is repeatable.
  - Verify: `python -m pytest tests/test_charging.py`.
  - Dependencies: TASK-07.
- [x] TASK-10 Perform integration audit.
  - Acceptance: full suite, Ruff, mypy, and a profile-driven simulator smoke run pass; no dependency-direction violations are found.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: TASK-08, TASK-09.
- [x] Checkpoint P2: Phase 0-2 accepted and ready for baseline-controller work.

## Phase 3

- [x] TASK-11 Freeze the controller protocol and controller YAML.
  - Acceptance: runtime-checkable controllers expose `reset()` and `act(BatteryState) -> CoolingAction`; illustrative thresholds, PID gains, flow limits, safety temperature, and pump coefficient are explicit configuration.
  - Verify: `python -m pytest tests/test_control_base.py`.
  - Dependencies: TASK-10.
- [x] TASK-12 Implement the threshold controller.
  - Acceptance: four deterministic cooling stages implement the documented 30/35/40 C boundaries and reject invalid thresholds/flows.
  - Verify: `python -m pytest tests/test_threshold_controller.py`.
  - Dependencies: TASK-11.
- [x] TASK-13 Implement the PID controller.
  - Acceptance: maximum pack temperature drives bounded proportional/integral/derivative action; nonmonotonic state time is rejected; reset restores initial behavior.
  - Verify: `python -m pytest tests/test_pid_controller.py`.
  - Dependencies: TASK-11.
- [x] TASK-14 Implement pump power and energy.
  - Acceptance: `P = k * flow^3` and `E = P * dt` are validated, monotonic, and unit-tested.
  - Verify: `python -m pytest tests/test_control_objective.py`.
  - Dependencies: TASK-11.
- [x] TASK-15 Reconcile Phase 3 exports and integration.
  - Acceptance: both controllers satisfy the protocol, stay within the configured flow interval, and execute deterministic closed-loop simulations.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: TASK-12, TASK-13, TASK-14.
- [x] Checkpoint P3: baseline BTMS accepted and ready for experiment-framework work.

## Phase 4

- [x] TASK-16 Implement experiment metrics.
  - Acceptance: peak/mean temperature, maximum spread, pump energy, violating sample count, and time above limit are calculated from explicit post-step records.
  - Verify: `python -m pytest tests/test_metrics.py`.
  - Dependencies: TASK-15.
- [x] TASK-17 Freeze experiment contracts.
  - Acceptance: scenario, post-step record, and result types are immutable, validate durations/units/provenance, and carry structured metrics.
  - Verify: `python -m pytest tests/test_experiment_types.py`.
  - Dependencies: TASK-16.
- [x] TASK-18 Implement the experiment runner.
  - Acceptance: the runner resets controller/environment, applies one action per exact timestep, records post-step truth, computes metrics, and repeats deterministically.
  - Verify: `python -m pytest tests/test_experiment_runner.py`.
  - Dependencies: TASK-17.
- [x] TASK-19 Implement non-overwriting result serialization.
  - Acceptance: validated run IDs produce trace CSV, metrics JSON, and provenance JSON; existing destinations and path-like identifiers are rejected.
  - Verify: `python -m pytest tests/test_experiment_serialization.py`.
  - Dependencies: TASK-17.
- [x] TASK-20 Add baseline experiment integration.
  - Acceptance: `scripts/run_baseline.py` compares controllers under the same scenario, seed, initial state, and duration and emits only measured results.
  - Verify: `python scripts/run_baseline.py --smoke-test` plus full quality gates.
  - Dependencies: TASK-18, TASK-19.
- [x] Checkpoint P4: reproducible controller experiments and artifacts accepted.

## Phase 5

- [x] TASK-21 Add optional per-cell initial resistance configuration.
  - Acceptance: scalar nominal resistance remains default; optional positive finite tuples are immutable and pack-length checked by reset.
  - Verify: `python -m pytest tests/test_types.py tests/test_environment.py`.
  - Dependencies: TASK-20.
- [x] TASK-22 Freeze dataset and sampling contracts.
  - Acceptance: scenario and transition records preserve state/action/next-state provenance; sampling ranges are explicit, validated, and represented in YAML.
  - Verify: `python -m pytest tests/test_data_types.py`.
  - Dependencies: TASK-21.
- [x] TASK-23 Implement the deterministic parameter sampler.
  - Acceptance: sampling varies initial state, current profile, flow, resistance, thermal coupling, and cooling effectiveness within declared ranges and is stable per seed/index.
  - Verify: `python -m pytest tests/test_sampling.py`.
  - Dependencies: TASK-22.
- [x] TASK-24 Implement trajectory generation.
  - Acceptance: every transition is produced by `BatteryEnvironment`, is contiguous, uses the sampled action, and carries trajectory metadata without simulator-truth leakage into controllers.
  - Verify: `python -m pytest tests/test_data_generate.py`.
  - Dependencies: TASK-22.
- [x] TASK-25 Implement trajectory-level split and persistence.
  - Acceptance: train/validation/test trajectory IDs are disjoint, seeded splits are deterministic, flattened rows retain all 12 cell states, and files never silently overwrite.
  - Verify: `python -m pytest tests/test_dataset.py`.
  - Dependencies: TASK-22.
- [x] TASK-26 Add dataset-generation CLI integration.
  - Acceptance: smoke mode generates a small deterministic three-way dataset and manifest under a caller-selected output directory.
  - Verify: `python scripts/generate_dataset.py --smoke-test`.
  - Dependencies: TASK-23, TASK-24, TASK-25.
- [x] TASK-27 Perform the Phase 3-5 integration audit.
  - Acceptance: all focused and repository quality gates pass; baseline and dataset smoke paths execute; dependency direction and result integrity are reviewed.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`; both smoke scripts.
  - Dependencies: TASK-26.
- [x] Checkpoint P5: Phase 3-5 accepted and ready for data-driven model work.

## Phase 6

- [x] TASK-28 Add ML dependencies and training configuration.
  - Acceptance: supported PyTorch and pandas runtime dependencies are declared; model/training hyperparameters, seed, device policy, and smoke settings are explicit YAML values.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: TASK-27.
- [x] TASK-29 Implement train-fitted normalization statistics.
  - Acceptance: immutable feature/target statistics validate shapes and finiteness, constant columns use a safe unit scale, transforms round-trip, and JSON metadata round-trips.
  - Verify: `python -m pytest tests/test_preprocessing.py`.
  - Dependencies: TASK-28.
- [x] TASK-30 Implement the model dataset loader.
  - Acceptance: manifest/CSV boundaries are validated, feature and target order is explicit, normalization is fitted only on train rows, trajectories remain disjoint, and no next-state fields enter model features.
  - Verify: `python -m pytest tests/test_model_dataset.py`.
  - Dependencies: TASK-29.
- [x] Checkpoint P6A: preprocessing and loading focused tests plus repository quality gates pass.
- [x] TASK-31 Implement the thermal delta MLP.
  - Acceptance: configurable Linear/GELU layers map `[batch, 17]` to `[batch, 12]`, reject invalid dimensions, preserve gradients, and expose serializable architecture metadata.
  - Verify: `python -m pytest tests/test_mlp.py`.
  - Dependencies: TASK-30.
- [x] TASK-32 Implement deterministic training and checkpointing.
  - Acceptance: seeded DataLoader training records train/validation loss, restores the best validation weights, supports early stopping/device selection, and writes a tensor-only checkpoint without overwriting.
  - Verify: `python -m pytest tests/test_model_train.py`.
  - Dependencies: TASK-31.
- [x] TASK-33 Implement physical-unit evaluation.
  - Acceptance: MAE, RMSE, per-cell MAE/RMSE, maximum absolute error, and worst-cell index are computed after target denormalization and reject incompatible shapes.
  - Verify: `python -m pytest tests/test_model_evaluation.py`.
  - Dependencies: TASK-31.
- [x] Checkpoint P6B: model, trainer, and evaluator focused tests plus repository quality gates pass.
- [x] TASK-34 Add training CLI integration.
  - Acceptance: `scripts/train_model.py --smoke-test` can generate/load a small dataset, trains on CPU, evaluates the test split, and persists configuration/history/normalization/metrics/provenance plus reloadable weights in a new output directory.
  - Verify: `python -m pytest tests/test_train_model_script.py`; `python scripts/train_model.py --smoke-test`.
  - Dependencies: TASK-32, TASK-33.
- [x] TASK-35 Perform the Phase 6 integration audit.
  - Acceptance: all focused and repository quality gates pass; all smoke paths execute; deterministic reruns match; documentation reflects Phase 6 without claiming comparative performance.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`; baseline, dataset, and model-training smoke scripts.
  - Dependencies: TASK-34.
- [x] Checkpoint P6: pure data-driven model training and evaluation accepted and ready for physics-informed learning.
  - Audit evidence: 875 pytest tests passed; Ruff and strict mypy passed; baseline, dataset, and model-training smoke commands passed. Smoke metrics verify the workflow only and are not a comparative performance claim.

## Phase 7

- [x] TASK-36 Define physics-loss configuration.
  - Acceptance: finite nonnegative physics weight and positive watt residual scale are explicit in immutable config and YAML, including smoke values.
  - Verify: `python -m pytest tests/test_model_config.py`.
  - Dependencies: TASK-35.
- [x] TASK-37 Add ordered transition/provenance contracts and reusable normalization.
  - Acceptance: CSV rows reconstruct exact immutable transitions aligned with normalized tensors; generation provenance is validated; externally supplied training statistics are reused unchanged for OOD data.
  - Verify: `python -m pytest tests/test_model_dataset.py tests/test_transition_artifacts.py`.
  - Dependencies: TASK-35.
- [x] Checkpoint P7A: Phase A contracts pass focused tests, Ruff, strict mypy, and diff review before physics logic begins.
- [x] TASK-38 Compute canonical physics heat-balance targets.
  - Acceptance: Joule heat, cooling, neighbor conduction, and ambient exchange are composed from existing physics functions with sampled scenario parameters and aligned to every transition row.
  - Verify: `python -m pytest tests/test_physics_dataset.py`.
  - Dependencies: TASK-36, TASK-37.
- [x] TASK-39 Implement differentiable physics residual and combined loss.
  - Acceptance: equilibrium, incorrect derivative, resistance, cooling, conduction, scaling, and backward-gradient behavior are tested without NumPy conversion of predictions.
  - Verify: `python -m pytest tests/test_model_losses.py`.
  - Dependencies: TASK-38.
- [x] TASK-40 Implement comparable dual-model training.
  - Acceptance: one deterministic trainer produces data-only and physics-informed models with identical architecture/initialization/data/optimizer settings and reports both loss components.
  - Verify: `python -m pytest tests/test_physics_train.py`.
  - Dependencies: TASK-39.
- [x] TASK-41 Add measured ID/OOD comparison workflow.
  - Acceptance: both variants are evaluated on ID and shifted OOD trajectories using only ID train normalization; artifacts include configuration, seeds, checkpoints, component histories, and physical-unit/physics-consistency metrics.
  - Verify: `python -m pytest tests/test_compare_models_script.py`; `python scripts/compare_models.py --smoke-test`.
  - Dependencies: TASK-40.
- [x] Checkpoint P7: physics-informed learning comparison accepted without assuming which model wins.

## Phase 8

- [x] TASK-42 Freeze the forecaster contract and implement single-step inference.
  - Acceptance: a runtime-checkable model-agnostic interface predicts physical next-cell temperatures from `BatteryState` and legal `CoolingAction` using persisted normalization and explicit model timestep.
  - Verify: `python -m pytest tests/test_forecaster.py`.
  - Dependencies: TASK-41.
- [x] TASK-43 Implement autoregressive rollout.
  - Acceptance: rollout feeds predicted temperatures back into later steps, accepts explicit future exogenous conditions/actions, supports persistence defaults, and never consumes future true temperatures.
  - Verify: `python -m pytest tests/test_forecaster.py`.
  - Dependencies: TASK-42.
- [x] TASK-44 Evaluate exact 30/60/120-second horizons.
  - Acceptance: horizon/timestep compatibility is validated and MAE/RMSE/maximum cell error are measured by horizon over whole held-out trajectories for either trained model variant.
  - Verify: `python -m pytest tests/test_forecast_evaluation.py tests/test_forecast_script.py`; `python scripts/evaluate_forecasts.py --smoke-test`.
  - Dependencies: TASK-43.
- [x] TASK-45 Perform the Phase 7-8 integration audit.
  - Acceptance: exports/docs are reconciled; full pytest, Ruff, strict mypy, comparison smoke, and forecast smoke pass; measured artifacts are reviewed for plausibility and no unsupported superiority claim is made.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests scripts`; both new smoke scripts.
  - Dependencies: TASK-44.
- [x] Checkpoint P8: physics-informed models and required-horizon forecast engine accepted for predictive-controller work.
  - Audit evidence: 1,000 pytest tests passed; Ruff and strict mypy passed; baseline, dataset, Phase 6 training, Phase 7 comparison, and Phase 8 forecast smoke commands passed. Smoke results are workflow evidence only: they did not establish physics-informed superiority and showed substantial autoregressive error growth with horizon.

## Phase 9

- [x] TASK-46 Freeze predictive-controller configuration and candidate-evaluation contracts.
  - Acceptance: horizon, candidate count, legal flow bounds, safety limit, pump coefficient, and objective weights are immutable and validated; candidate results expose each cost component.
  - Verify: `python -m pytest tests/test_predictive_types.py`.
  - Dependencies: TASK-45.
- [x] TASK-49 Audit all Phase A contracts before logic work.
  - Acceptance: focused tests, Ruff, strict mypy, and diff review pass with no downstream implementation added prematurely.
  - Verify: `python -m pytest tests/test_predictive_types.py tests/test_types.py`; `python -m ruff check .`; `python -m mypy src tests scripts`.
  - Dependencies: TASK-46, TASK-47, TASK-48.
- [x] TASK-50 Implement the predictive objective.
  - Acceptance: peak temperature, spread, horizon pump energy, and safety penalty are calculated and exposed separately; nonfinite inputs fail.
  - Verify: `python -m pytest tests/test_predictive_objective.py`.
  - Dependencies: TASK-49.
- [x] TASK-51 Implement the receding-horizon predictive controller.
  - Acceptance: `np.linspace` candidates are rolled out through `ThermalForecaster`; minimum-cost ties are deterministic; only the first legal action is returned; reset is deterministic.
  - Verify: `python -m pytest tests/test_predictive_controller.py`.
  - Dependencies: TASK-50.
- [x] TASK-55 Add predictive closed-loop comparison artifacts.
  - Acceptance: threshold, PID, and predictive controllers run on an identical scenario; only measured metrics are saved and existing output paths are not overwritten.
  - Verify: `python -m pytest tests/test_predictive_experiment.py`; `python scripts/run_predictive.py --smoke-test`.
  - Dependencies: TASK-51.
- [x] Checkpoint P9: predictive cooling is accepted against reactive baselines.

## Phase 10

- [x] TASK-47 Add the resistance-fault configuration contract.
  - Acceptance: optional cell index, nonnegative start time, and multiplier greater than one are immutable, serializable, and validated without changing fault-free defaults.
  - Verify: `python -m pytest tests/test_types.py`.
  - Dependencies: TASK-45.
- [x] TASK-52 Integrate scheduled resistance faults into the simulator.
  - Acceptance: the selected cell resistance changes once at the exact schedule boundary, affects Joule heating, reset restores the schedule, and other cells remain unchanged.
  - Verify: `python -m pytest tests/test_environment_faults.py`.
  - Dependencies: TASK-49.
- [x] TASK-53 Implement persistent rolling thermal anomaly detection.
  - Acceptance: residuals equal measured minus forecast temperatures; rolling scores are per-cell; isolated spikes do not trigger; persistent abnormal residuals report cell, score, time, predicted hotspot, and severity.
  - Verify: `python -m pytest tests/test_anomaly.py`.
  - Dependencies: TASK-52.
- [x] TASK-54 Implement fault lead-time measurement.
  - Acceptance: threshold time minus detection time is reported from measured events, including zero/negative lead and no-crossing cases without invented values.
  - Verify: `python -m pytest tests/test_fault_metrics.py`.
  - Dependencies: TASK-53.
- [x] TASK-56 Add the saved fault-detection workflow.
  - Acceptance: inference receives no fault label; fault provenance, residual trace, detection event, threshold crossing, and lead time are persisted separately and reproducibly.
  - Verify: `python -m pytest tests/test_fault_experiment.py`; `python scripts/run_fault_detection.py --smoke-test`.
  - Dependencies: TASK-54.
- [x] Checkpoint P10: simulator-owned fault injection and persistent anomaly detection are accepted.

## Phase 11

- [x] TASK-48 Declare dashboard dependencies.
  - Acceptance: supported Plotly and Streamlit runtime constraints are declared without weakening existing quality tooling.
  - Verify: `python -m pytest tests/test_package.py`.
  - Dependencies: TASK-45.
- [x] TASK-57 Implement dashboard artifact loading and immutable view data.
  - Acceptance: controller/model/fault artifacts are validated, missing or malformed data fails clearly, and 12-cell topology is preserved.
  - Verify: `python -m pytest tests/test_dashboard_data.py`.
  - Dependencies: TASK-55, TASK-56.
- [x] TASK-58 Implement reusable Plotly figures.
  - Acceptance: 3x4 measured/predicted heatmaps, temperature and coolant-flow traces, thresholds, and comparison figures are produced without Streamlit or simulator imports.
  - Verify: `python -m pytest tests/test_visualization_plots.py`.
  - Dependencies: TASK-57.
- [x] TASK-59 Integrate the Streamlit dashboard.
  - Acceptance: the dashboard renders metric cards, selectable prediction horizon, controller/model comparisons, and fault details exclusively from loaded artifacts.
  - Verify: `python -m pytest tests/test_dashboard.py`; dashboard headless smoke command.
  - Dependencies: TASK-58.
- [x] TASK-60 Perform the Phase 9-11 integration audit.
  - Acceptance: exports/docs are reconciled; full pytest, Ruff, strict mypy, predictive/fault workflows, and dashboard smoke pass; quantitative claims are limited to saved results.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests scripts`; all Phase 9-11 smoke commands.
  - Audit evidence: full pytest passed (1,265 passed in 373.04s); full Ruff passed; strict mypy passed for 120 source files; fresh predictive and fault smoke workflows passed; strict artifact loader accepted 4 controller traces, 2 model forecast sets, and a 30-step fault trace with a saved 11 s lead; documented dashboard launcher health returned HTTP 200 `ok`. Predictive smoke again produced identical metrics and zero cooling across all four variants, demonstrating workflow integrity only. Fault smoke measured detection at 15 s, threshold crossing at 26 s, lead 11 s, and peak 317.12793572865644 K. No controller or model superiority is demonstrated.
  - Dependencies: TASK-59.
- [x] Checkpoint P11: predictive control, fault diagnostics, and artifact-driven dashboard accepted for benchmark-suite work.
  - Next: Phase 12 benchmark suite; quantitative superiority has not yet been demonstrated.

## Phase 12

Phase 12 is a deterministic, descriptive six-scenario benchmark, not evidence of CI coverage, statistical significance, production safety, or generalization beyond the configured synthetic runs. Threshold is the primary reactive baseline and PID is supplemental. Every scenario must run threshold, PID, predictive/data-only, and predictive/physics-informed variants from the identical scenario. Predictive control must disclose exogenous persistence and must not consume future simulator truth.

- [x] TASK-61 Fix the empty forecast-evaluation split regression.
  - Scope: `scripts/evaluate_forecasts.py`; `tests/test_forecast_script.py`.
  - Acceptance: an empty test split fails before indexing with a concise actionable CLI error, while valid evaluation and non-overwrite behavior remain unchanged.
  - Verify: `python -m pytest tests/test_forecast_script.py`.
  - Dependencies: TASK-60.
  - Accepted: regression first reproduced the uncaught `IndexError`; the guard now rejects an empty test split before indexing. Focused verification passed with 5 tests, Ruff, and strict mypy.
- [x] TASK-62 Freeze a common-source/common-action forecast snapshot contract.
  - Scope: `scripts/run_predictive.py`; `tests/test_predictive_experiment.py`.
  - Acceptance: both model variants use and persist the exact same identified measured source state/time and legal action for all required horizons; mismatches are rejected.
  - Verify: `python -m pytest tests/test_predictive_experiment.py`.
  - Dependencies: TASK-61.
  - Accepted: schema version 2 stores one threshold-run terminal source and action for both twins; distinct predictive terminal states are covered by regression tests. Focused verification passed with 6 tests, Ruff, and strict mypy.
- [x] TASK-63 Enforce dashboard source pairing.
  - Scope: `src/thermotwin/visualization/predictive_data.py`; `src/thermotwin/visualization/contracts.py`; `tests/test_dashboard_data.py`.
  - Acceptance: loading rejects cross-model source/action mismatches and exposes one immutable common source so a model prediction cannot be paired with an unrelated controller terminal state.
  - Verify: `python -m pytest tests/test_dashboard_data.py`.
  - Dependencies: TASK-62.
  - Accepted: the schema-v2 loader validates one threshold source/action and rejects mixed or legacy fields; immutable snapshots share that source while retaining distinct predictions. Verification passed with 30 tests, Ruff, strict mypy, and a fresh writer-to-loader smoke integration.
- [x] TASK-64 Repair dashboard accessibility and hidden-truth separation.
  - Scope: `src/thermotwin/visualization/dashboard.py`; `src/thermotwin/visualization/plots.py`; `tests/test_dashboard.py`; `tests/test_visualization_plots.py`.
  - Acceptance: measured and predicted heatmaps share saved source provenance; accessible labels and non-color cues are present; operator-facing suspected-cell output comes only from detector evidence or says unavailable; configured fault truth is isolated as evaluation-only.
  - Verify: `python -m pytest tests/test_dashboard.py tests/test_visualization_plots.py`; documented headless dashboard health check.
  - Dependencies: TASK-63.
  - Accepted: the measured heatmap uses the shared threshold source; provenance is visible; detector events alone drive the operator panel; injected truth is evaluation-only; labels have high-contrast backgrounds and all cell traces have non-color cues. Verification passed with 21 tests, Ruff, strict mypy, and a fresh HTTP 200 `ok` dashboard health check.
- [x] TASK-65 Define immutable benchmark/result contracts.
  - Scope: `src/thermotwin/experiments/benchmark_types.py`; `tests/test_benchmark_types.py`.
  - Acceptance: contracts validate scenario/variant identity, seed, model and dataset provenance, training mode, persistence assumptions, controller/forecast/physics/fault metrics, finite values, and matrix uniqueness; full mode rejects smoke-trained model artifacts.
  - Verify: `python -m pytest tests/test_benchmark_types.py`.
  - Dependencies: TASK-64.
  - Accepted: frozen contracts cover the 6x4 matrix, artifact/model provenance, no-future-truth assumptions, controller/model/forecast/physics/fault metrics, and full-mode smoke rejection. Review aligned distribution labels with saved artifacts and tightened detector/thermal invariants. Verification passed with 32 tests, Ruff, and strict mypy.
- [x] TASK-66 Implement deterministic six-scenario configuration and factory.
  - Scope: `configs/benchmark.yaml`; `src/thermotwin/scenarios/benchmark.py`; `tests/test_benchmark_scenarios.py`.
  - Acceptance: normal, high-current, hot-ambient, weak-cooling, fast-charging, and resistance-fault scenarios are reproducible and valid; weak cooling reduces legal flow capacity only; one immutable scenario is shared by all variants.
  - Verify: `python -m pytest tests/test_benchmark_scenarios.py`.
  - Dependencies: TASK-65.
  - Accepted: strict immutable quick/full cases cover all six scenarios with canonical seed-independent configuration digests and isolated ambient, cooling-capacity, charging, load, and resistance-fault changes. The shared pump coefficient remains `100000.0`. Verification passed with 25 tests, Ruff, and strict mypy.
- [x] TASK-67 Implement the four-variant controller factory and leakage regression.
  - Scope: `src/thermotwin/experiments/benchmark_controllers.py`; `tests/test_benchmark_controllers.py`.
  - Acceptance: threshold primary, PID supplemental, predictive/data-only, and predictive/physics-informed variants receive identical public state/scenario inputs; predictive variants use disclosed persistence through the forecaster and receive no future truth or fault label.
  - Verify: `python -m pytest tests/test_benchmark_controllers.py`.
  - Dependencies: TASK-65.
  - Accepted: the immutable factory returns fresh threshold, PID, data-only predictive, and physics-informed predictive controllers with scenario-derived limits and explicit no-future-truth persistence assumptions. Leakage/bounds/freshness validation passed with 26 tests, Ruff, and strict mypy.
- [ ] Checkpoint P12A: stabilization and benchmark contracts accepted before suite execution.
  - Acceptance: TASK-61 through TASK-67 focused tests, Ruff, strict mypy, and diff review pass; the two disjoint TASK-66/TASK-67 scopes are integrated without contract drift.
  - Verify: focused tests above; `python -m ruff check .`; `python -m mypy src tests scripts`.
  - Dependencies: TASK-66, TASK-67.
- [ ] TASK-68 Execute and persist the full 6-by-4 benchmark matrix.
  - Scope: `src/thermotwin/experiments/benchmark.py`; `src/thermotwin/experiments/benchmark_serialization.py`; `tests/test_benchmark.py`; `tests/test_benchmark_serialization.py`.
  - Acceptance: all four variants run each identical scenario/seed/state/configuration; non-overwriting artifacts retain traces, metrics, identity, configuration, seed, model/dataset/checkpoint provenance, and completeness state; partial output cannot appear complete.
  - Verify: `python -m pytest tests/test_benchmark.py tests/test_benchmark_serialization.py`.
  - Dependencies: TASK-66, TASK-67, Checkpoint P12A.
- [ ] TASK-69 Add quick/full benchmark CLI workflows.
  - Scope: `scripts/benchmark.py`; `tests/test_benchmark_script.py`.
  - Acceptance: quick mode runs the full 6-by-4 matrix and labels it workflow-only; full mode requires validated non-smoke model provenance; malformed/missing paths, empty data, incompatibility, and existing outputs produce concise actionable stderr without traceback.
  - Verify: `python -m pytest tests/test_benchmark_script.py`; `python scripts/benchmark.py --quick --output <new-temporary-directory>`.
  - Dependencies: TASK-68.
- [ ] Checkpoint P12B: benchmark execution and persistence accepted.
  - Acceptance: a fresh quick matrix has 24 complete, deterministic, reloadable, provenance-valid cells and full mode demonstrably rejects smoke-trained artifacts.
  - Verify: rerun TASK-69 verification against fresh temporary outputs and compare manifests/metrics.
  - Dependencies: TASK-69.
- [ ] TASK-70 Aggregate final evaluation artifacts.
  - Scope: `src/thermotwin/experiments/final_evaluation.py`; `tests/test_final_evaluation.py`.
  - Acceptance: saved-only aggregation reports ID/OOD MAE/RMSE/max error, 30/60/120-second errors, physics residual, controller peak/mean/spread, pump energy, safety count/duration, and fault lead time; incomplete/incompatible evidence fails; conclusions stay descriptive unless results support more.
  - Verify: `python -m pytest tests/test_final_evaluation.py`.
  - Dependencies: TASK-69, Checkpoint P12B.
- [ ] TASK-71 Generate accessible presentation figures from saved evaluation data.
  - Scope: `src/thermotwin/visualization/benchmark_plots.py`; `tests/test_benchmark_plots.py`.
  - Acceptance: figures cover controller matrix, ID/OOD/horizon errors, physics residual, thermal-energy tradeoffs, and fault timing with visible units, identities, evidence mode, and non-color cues; no hidden input or fabricated number is used.
  - Verify: `python -m pytest tests/test_benchmark_plots.py`.
  - Dependencies: TASK-70.
- [ ] TASK-72 Perform the actual-user/browser/release audit and documentation reconciliation.
  - Scope: `README.md`; `PLAN.md`; `tasks/plan.md`; `tasks/todo.md`; `scripts/benchmark.py`; `tests/test_benchmark_script.py`; `src/thermotwin/visualization/dashboard.py`; `tests/test_dashboard.py`. Any defect outside this sequential scope requires a separately whitelisted repair task.
  - Acceptance: clean setup-to-quick-benchmark-to-dashboard instructions work; browser/AppTest inspection covers all required views, responsive/accessibility behavior, console health, source pairing, and hidden-truth separation; CLI errors are actionable and traceback-free; full pytest/Ruff/mypy, deterministic runs, non-overwrite/provenance checks, artifact numbers, architecture, and claims are audited; docs disclose persistence and descriptive synthetic limitations.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests scripts`; all documented smoke/quick commands; dashboard health and browser/AppTest journey; `git diff --check`.
  - Dependencies: TASK-71.
- [ ] Checkpoint P12C: final release audit accepted.
  - Acceptance: all Phase 12 jobs are accepted, the actual-user journey and full gates pass, fresh saved outputs support every published number, and no critical defect or unsupported claim remains.
  - Verify: repeat TASK-72 verification from a clean artifact directory and review the final diff/evidence bundle.
  - Dependencies: TASK-72.
