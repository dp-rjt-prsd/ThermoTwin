# ThermoTwin Phase 0-5 Tasks

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
- [ ] TASK-15 Reconcile Phase 3 exports and integration.
  - Acceptance: both controllers satisfy the protocol, stay within the configured flow interval, and execute deterministic closed-loop simulations.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: TASK-12, TASK-13, TASK-14.
- [ ] Checkpoint P3: baseline BTMS accepted and ready for experiment-framework work.

## Phase 4

- [ ] TASK-16 Implement experiment metrics.
  - Acceptance: peak/mean temperature, maximum spread, pump energy, violating sample count, and time above limit are calculated from explicit post-step records.
  - Verify: `python -m pytest tests/test_metrics.py`.
  - Dependencies: TASK-15.
- [ ] TASK-17 Freeze experiment contracts.
  - Acceptance: scenario, post-step record, and result types are immutable, validate durations/units/provenance, and carry structured metrics.
  - Verify: `python -m pytest tests/test_experiment_types.py`.
  - Dependencies: TASK-16.
- [ ] TASK-18 Implement the experiment runner.
  - Acceptance: the runner resets controller/environment, applies one action per exact timestep, records post-step truth, computes metrics, and repeats deterministically.
  - Verify: `python -m pytest tests/test_experiment_runner.py`.
  - Dependencies: TASK-17.
- [ ] TASK-19 Implement non-overwriting result serialization.
  - Acceptance: validated run IDs produce trace CSV, metrics JSON, and provenance JSON; existing destinations and path-like identifiers are rejected.
  - Verify: `python -m pytest tests/test_experiment_serialization.py`.
  - Dependencies: TASK-17.
- [ ] TASK-20 Add baseline experiment integration.
  - Acceptance: `scripts/run_baseline.py` compares controllers under the same scenario, seed, initial state, and duration and emits only measured results.
  - Verify: `python scripts/run_baseline.py --smoke-test` plus full quality gates.
  - Dependencies: TASK-18, TASK-19.
- [ ] Checkpoint P4: reproducible controller experiments and artifacts accepted.

## Phase 5

- [ ] TASK-21 Add optional per-cell initial resistance configuration.
  - Acceptance: scalar nominal resistance remains default; optional positive finite tuples are immutable and pack-length checked by reset.
  - Verify: `python -m pytest tests/test_types.py tests/test_environment.py`.
  - Dependencies: TASK-20.
- [ ] TASK-22 Freeze dataset and sampling contracts.
  - Acceptance: scenario and transition records preserve state/action/next-state provenance; sampling ranges are explicit, validated, and represented in YAML.
  - Verify: `python -m pytest tests/test_data_types.py`.
  - Dependencies: TASK-21.
- [ ] TASK-23 Implement the deterministic parameter sampler.
  - Acceptance: sampling varies initial state, current profile, flow, resistance, thermal coupling, and cooling effectiveness within declared ranges and is stable per seed/index.
  - Verify: `python -m pytest tests/test_sampling.py`.
  - Dependencies: TASK-22.
- [ ] TASK-24 Implement trajectory generation.
  - Acceptance: every transition is produced by `BatteryEnvironment`, is contiguous, uses the sampled action, and carries trajectory metadata without simulator-truth leakage into controllers.
  - Verify: `python -m pytest tests/test_data_generate.py`.
  - Dependencies: TASK-22.
- [ ] TASK-25 Implement trajectory-level split and persistence.
  - Acceptance: train/validation/test trajectory IDs are disjoint, seeded splits are deterministic, flattened rows retain all 12 cell states, and files never silently overwrite.
  - Verify: `python -m pytest tests/test_dataset.py`.
  - Dependencies: TASK-22.
- [ ] TASK-26 Add dataset-generation CLI integration.
  - Acceptance: smoke mode generates a small deterministic three-way dataset and manifest under a caller-selected output directory.
  - Verify: `python scripts/generate_dataset.py --smoke-test`.
  - Dependencies: TASK-23, TASK-24, TASK-25.
- [ ] TASK-27 Perform the Phase 3-5 integration audit.
  - Acceptance: all focused and repository quality gates pass; baseline and dataset smoke paths execute; dependency direction and result integrity are reviewed.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`; both smoke scripts.
  - Dependencies: TASK-26.
- [ ] Checkpoint P5: Phase 3-5 accepted and ready for data-driven model work.
