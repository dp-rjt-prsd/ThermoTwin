# Implementation Plan: ThermoTwin Phases 0-2

## Overview

Build the tested foundation, shared contracts, minimal 12-cell electrothermal simulator, and deterministic current-profile generators required by `PLAN.md` Phases 0 through 2. `AGENTS.md` is authoritative where project documents conflict.

## Frozen Architecture Decisions

- Python 3.11+ with `src/` packaging, pytest, Ruff, and mypy.
- Physical quantities use explicit SI-unit suffixes in public names; temperatures are kelvin internally.
- `BatteryState` and configuration dataclasses are validated contracts. Cell arrays have shape `(rows * columns,)`.
- Positive current discharges the pack; negative current charges it. SOC is bounded to `[0, 1]`.
- A current profile is a pure callable from simulation time in seconds to current in amperes.
- `BatteryEnvironment` is the authoritative simulator and has no dependency on ML, control, diagnostics, or visualization.
- Environment thermal balance includes Joule heating, neighbor conduction, liquid cooling, and ambient exchange. Forward Euler is sufficient for the MVP, with explicit positive-`dt` validation.
- Pack adjacency is defined once in `physics/topology.py` and consumed by simulator physics.

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

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Unit ambiguity | High | Explicit suffixes and boundary validation |
| Sign mistakes in heat flow | High | Conservation and direction tests |
| Numerically implausible Euler steps | High | Positive timestep checks and integration sanity test |
| Scenario nondeterminism | Medium | Pure time-indexed functions and repeatability tests |
| Physics duplication later in ML | Medium | Keep equations in small canonical physics functions |
| Worker collisions | High | Exact file whitelists and sequential shared-file reconciliation |

## Open Questions Deferred Beyond Phase 2

- Final YAML loading/override strategy for experiment and training configuration.
- Differentiable backend strategy for reusing canonical equations in PyTorch physics loss.
- Controller and forecaster interfaces beyond the already documented contracts.
