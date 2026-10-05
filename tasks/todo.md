# ThermoTwin Phase 0-2 Tasks

## Phase 0

- [x] TASK-00 Bootstrap package and quality tooling.
  - Acceptance: Python package imports; pytest discovers a smoke test; `.gitignore`, metadata, dependency groups, Ruff, and mypy configuration exist.
  - Verify: `python -m pytest tests/test_package.py`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: none.
- [x] TASK-01 Define validated shared contracts and default YAML configuration.
  - Acceptance: explicit-unit state/action/config dataclasses validate shapes, ranges, and flow bounds; defaults describe a 3x4 pack.
  - Verify: `python -m pytest tests/test_types.py`.
  - Dependencies: TASK-00.
- [x] TASK-02 Implement canonical rectangular topology.
  - Acceptance: 3x4 grid yields 12 cells and 17 unique undirected edges in deterministic order; invalid dimensions fail.
  - Verify: `python -m pytest tests/test_topology.py`.
  - Dependencies: TASK-01.
- [x] Checkpoint P0: full pytest, Ruff, and mypy pass.

## Phase 1

- [ ] TASK-03 Implement Joule heating and isolated-cell thermal integration.
  - Acceptance: zero current, current-squared scaling, resistance scaling, and analytical temperature update are tested.
  - Verify: `python -m pytest tests/test_cell.py`.
  - Dependencies: TASK-01.
- [ ] TASK-04 Implement conservative neighbor conduction.
  - Acceptance: equal temperatures exchange no heat; heat flows hot to cold; net pairwise energy is zero.
  - Verify: `python -m pytest tests/test_pack.py`.
  - Dependencies: TASK-02.
- [ ] TASK-05 Implement flow-dependent cooling and ambient exchange.
  - Acceptance: coefficient and heat removal increase monotonically with legal flow; signs reverse correctly across coolant/ambient temperatures.
  - Verify: `python -m pytest tests/test_cooling.py`.
  - Dependencies: TASK-01.
- [ ] TASK-06 Implement signed-current coulomb counting.
  - Acceptance: discharge lowers SOC, charge raises SOC, zero current preserves SOC, and bounds are respected.
  - Verify: `python -m pytest tests/test_electrical.py`.
  - Dependencies: TASK-01.
- [ ] TASK-07 Integrate `BatteryEnvironment`.
  - Acceptance: reset/step satisfy frozen contracts; all heat terms and SOC are integrated; action and timestep bounds are enforced; deterministic runs match.
  - Verify: `python -m pytest tests/test_environment.py`.
  - Dependencies: TASK-03, TASK-04, TASK-05, TASK-06.
- [ ] Checkpoint P1: full pytest, Ruff, mypy, and deterministic physical sanity run pass.

## Phase 2

- [ ] TASK-08 Implement deterministic drive profiles.
  - Acceptance: constant, pulse, stop-and-go, and aggressive profiles obey duration/current contracts and are repeatable.
  - Verify: `python -m pytest tests/test_drive_cycles.py`.
  - Dependencies: TASK-07.
- [ ] TASK-09 Implement deterministic fast-charging profile.
  - Acceptance: charging current follows the negative-current convention, tapers safely, obeys duration, and is repeatable.
  - Verify: `python -m pytest tests/test_charging.py`.
  - Dependencies: TASK-07.
- [ ] TASK-10 Perform integration audit.
  - Acceptance: full suite, Ruff, mypy, and a profile-driven simulator smoke run pass; no dependency-direction violations are found.
  - Verify: `python -m pytest`; `python -m ruff check .`; `python -m mypy src tests`.
  - Dependencies: TASK-08, TASK-09.
- [ ] Checkpoint P2: Phase 0-2 accepted and ready for baseline-controller work.
