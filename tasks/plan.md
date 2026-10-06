# Implementation Plan: ThermoTwin Phases 0-8

## Overview

This plan records the implemented foundation through the reproducible baseline-controller experiment, leakage-safe synthetic transition dataset, data-only and physics-informed thermal surrogates, and model-agnostic multi-step forecast engine for Phases 0-8. Phases 7-8 and their orchestrator integration audit are complete. `AGENTS.md` is authoritative where project documents conflict.

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

## Future Phases Beyond This Implementation

- Phase 9 adds predictive control, including candidate coolant-action evaluation and cost aggregation.
- Later fault work adds residual-based anomaly calibration and detection persistence; the dashboard follows in a later phase.
