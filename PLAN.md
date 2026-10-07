# ThermoTwin — Orchestrator–Worker Implementation Plan

## 1. Development Model

ThermoTwin will be implemented using an **Orchestrator–Worker agent pattern** inside the Codex extension for VS Code.

### Orchestrator

**Model:** Sol 5.6

Sol is responsible for:

- understanding the user goal
- reading project-level context
- inspecting the current codebase
- maintaining architectural consistency
- decomposing work into isolated jobs
- selecting the files required for each job
- delegating implementation work
- reviewing returned diffs
- integrating work across modules
- running integration tests
- deciding the next task

Sol should perform reasoning and coordination.

Sol should avoid implementing large features itself unless:

- the task is extremely small
- integration requires a tiny patch
- a worker repeatedly fails
- the task cannot be reasonably isolated

---

## 2. Worker

**Model:** Luna 6

Luna acts as an execution-focused coding worker.

Each Luna worker receives:

1. one clearly defined task
2. only the relevant project context
3. explicit files it may inspect
4. explicit files it may modify
5. required interfaces
6. expected behavior
7. verification instructions

Luna must:

- inspect the supplied code
- implement only the requested behavior
- avoid unrelated refactors
- run the specified tests/lint/type checks
- report the files modified
- report verification results
- report assumptions or unresolved issues

Luna should not independently redesign the architecture.

If an architectural issue is discovered, Luna reports it to Sol instead of introducing a large redesign.

---

# 3. Core Principle

The workflow is:

```text
Goal
  ↓
Sol inspects repository
  ↓
Sol designs next implementation step
  ↓
Sol decomposes it into isolated jobs
  ↓
Luna workers implement jobs
  ↓
Workers test their changes
  ↓
Sol reviews diffs
  ↓
Sol integrates
  ↓
Sol runs cross-module validation
  ↓
Sol determines next jobs
```

Do not ask multiple workers to modify the same file simultaneously unless absolutely necessary.

Prefer dependency-aware sequential delegation over merge-heavy parallel work.

---

# 4. Orchestrator Loop

For every development iteration, Sol should follow this cycle.

## Step 1 — Inspect

Read:

- `AGENTS.md`
- `PLAN.md`
- `ARCHITECTURE.md`
- relevant configs
- relevant interfaces
- existing implementation
- current tests

Do not assume that earlier planned components were implemented exactly as designed.

The codebase is the source of truth.

---

## Step 2 — Determine Current State

Identify:

```text
DONE
IN PROGRESS
MISSING
BROKEN
BLOCKED
```

Determine the smallest useful vertical increment that can be completed next.

Examples:

```text
BatteryCell thermal equation
Pack topology
Cooling model
Environment.step()
ThresholdController
Experiment runner
Dataset generator
ML dataset class
Thermal MLP
Forecast rollout
Predictive controller
Dashboard heatmap
```

---

## Step 3 — Design Interfaces First

Before delegating implementation across multiple modules, Sol should freeze required interfaces.

Example:

```python
class Controller:
    def act(self, state: BatteryState) -> CoolingAction:
        ...
```

and:

```python
class BatteryEnvironment:
    def reset(self, config) -> BatteryState:
        ...

    def step(
        self,
        action: CoolingAction,
        dt: float,
    ) -> BatteryState:
        ...
```

Workers should implement against stable contracts rather than inventing incompatible APIs independently.

---

## Step 4 — Build Worker Jobs

Each job should ideally satisfy all of the following:

- modifies 1–3 tightly related files
- has one clear purpose
- can be tested independently
- takes an existing interface as input
- produces an explicit artifact or behavior
- does not require broad architectural decisions

Good:

```text
Implement neighbor conduction in src/thermotwin/physics/pack.py
and add unit tests in tests/test_pack.py.
```

Bad:

```text
Implement the battery simulator.
```

Good:

```text
Implement ThresholdController using the existing Controller
interface and add boundary-condition tests.
```

Bad:

```text
Implement all thermal management algorithms.
```

---

# 5. Worker Job Template

Sol should formulate jobs approximately like this:

```text
TASK
Implement <specific behavior>.

CONTEXT
<only information required to understand this task>

FILES TO READ
- ...

FILES YOU MAY MODIFY
- ...

DO NOT MODIFY
- ...

REQUIRED INTERFACES
- ...

BEHAVIOR
- ...

EDGE CASES
- ...

VERIFICATION
Run:
<test command>

DONE WHEN
<clear acceptance criteria>

REPORT
Return:
1. summary
2. files changed
3. tests run
4. results
5. assumptions/issues
```

The worker should not receive the entire repository unless necessary.

---

# 6. Parallelism Rules

Parallel workers may be used when their changes are independent.

Good parallelization:

```text
Worker A
Pack topology

Worker B
Drive-cycle generation

Worker C
Metrics utilities
```

assuming none modify shared files.

Avoid:

```text
Worker A -> environment.py
Worker B -> environment.py
Worker C -> environment.py
```

Sol should prefer:

```text
A + B + C in parallel
        ↓
Sol review
        ↓
integration job
```

---

# 7. Diff Review Protocol

After each worker completes a job, Sol reviews the diff.

Check:

### Correctness

Does it implement the requested behavior?

### Scope

Did the worker modify unrelated code?

### Architecture

Does it obey `ARCHITECTURE.md`?

### API consistency

Does it preserve existing interfaces?

### Physical correctness

For physics code:

- units correct?
- signs correct?
- energy balance sensible?
- state updates plausible?

### Numerical correctness

Check:

- timestep usage
- vector dimensions
- broadcasting
- normalization
- numerical stability

### Testing

Were meaningful tests added?

Did they actually exercise the new behavior?

### Maintainability

Avoid:

- hidden constants
- giant functions
- circular imports
- UI logic in simulation code
- physics code inside controllers

Sol should reject or patch changes that violate these constraints.

---

# 8. Verification Hierarchy

Workers verify their own isolated tasks.

Sol verifies integrated behavior.

## Worker-level verification

Examples:

```bash
pytest tests/test_cell.py
```

```bash
pytest tests/test_pack.py
```

```bash
pytest tests/test_controllers.py
```

Potentially:

```bash
ruff check <modified files>
```

and type checks if configured.

---

## Orchestrator-level verification

After integration:

```bash
pytest
```

Then run at least one deterministic simulation.

Example:

```bash
python scripts/run_baseline.py
```

For ML stages:

```bash
python scripts/train_model.py --smoke-test
```

For integration:

```bash
python scripts/benchmark.py --quick
```

Sol must not consider a phase complete solely because individual worker tests pass.

---

# 9. Development Strategy

Development follows **vertical slices**, not isolated subsystem completion.

The preferred sequence is:

```text
Minimal physics
    ↓
Executable simulation
    ↓
Baseline controller
    ↓
Experiment runner
    ↓
Dataset generation
    ↓
Learned surrogate
    ↓
Forecast rollout
    ↓
Predictive controller
    ↓
Fault detection
    ↓
Dashboard
```

At every stage, the project should remain runnable.

---

# 10. Phase 0 — Repository Bootstrap

## Goal

Create the minimal project structure and stable interfaces.

### Sol responsibilities

Define:

- package structure
- core dataclasses
- module boundaries
- config structure
- test setup

### Suggested jobs

#### Worker 0A — Project skeleton

Create:

```text
src/thermotwin/
tests/
configs/
scripts/
```

Configure Python package and testing.

#### Worker 0B — Core data structures

Implement:

```python
BatteryState
CoolingAction
PackConfig
SimulationConfig
```

#### Worker 0C — Pack topology

Implement deterministic 3×4 battery adjacency.

### Integration checkpoint

Sol verifies imports and runs:

```bash
pytest
```

---

# 11. Phase 1 — Minimal Electrothermal Simulator

## Goal

Produce physically plausible battery temperature trajectories.

Build incrementally.

---

## Job 1A — Cell heat generation

Implement:

\[
Q_{gen}=I^2R
\]

Tests:

- zero current produces zero Joule heating
- doubled current produces 4× heating
- increased resistance increases heating linearly

---

## Job 1B — Cell thermal dynamics

Implement:

\[
mC_p\dot T=Q
\]

for an isolated cell.

Test expected temperature increase analytically.

---

## Job 1C — Neighbor conduction

Implement:

\[
Q_{ij}=\frac{T_j-T_i}{R_{th,ij}}
\]

Tests:

- equal-temperature cells exchange no heat
- heat flows hot → cold
- pairwise energy exchange is equal and opposite

---

## Job 1D — Cooling model

Implement:

\[
Q_{cool}=h(\dot m)A(T-T_c)
\]

Requirements:

- configurable
- monotonically stronger with coolant flow
- isolated from pack logic

Tests verify increased flow increases cooling.

---

## Job 1E — SOC model

Implement coulomb counting:

\[
SOC_{t+\Delta t}
=
SOC_t-\frac{I\Delta t}{3600C_{Ah}}
\]

Test charging/discharging convention.

---

## Job 1F — Battery environment

Combine:

```text
heat generation
+
conduction
+
cooling
+
SOC
```

into:

```python
BatteryEnvironment.step()
```

### Sol integration test

Run:

```text
constant current
constant ambient
constant coolant flow
```

Verify temperatures evolve sensibly.

---

# 12. Phase 2 — Scenario Generation

## Goal

Produce reusable drive and charging profiles.

Potential parallel jobs:

### Worker 2A

Constant and pulse current profiles.

### Worker 2B

Stop-and-go and aggressive synthetic drive profiles.

### Worker 2C

Fast-charging profile.

Workers should return pure functions/data and avoid simulator coupling.

---

# 13. Phase 3 — Baseline BTMS

Do not introduce AI before this phase passes.

## Job 3A — Controller interface

If not already frozen:

```python
class Controller:
    def act(self, state: BatteryState) -> CoolingAction:
        ...
```

---

## Job 3B — Threshold controller

Implement deterministic cooling stages based on \(T_{max}\).

---

## Job 3C — PID controller

Optional but preferred.

PID should target maximum pack temperature.

---

## Job 3D — Pump-energy model

Implement:

\[
P_{pump}=k_p\dot m^3
\]

as a configurable approximation.

---

# 14. Phase 4 — Experiment Framework

## Goal

Make every controller testable under exactly the same scenario.

Create:

```python
run_experiment(
    scenario,
    controller,
    seed,
)
```

Return structured metrics and time series.

Potential worker decomposition:

### Worker 4A

Experiment runner.

### Worker 4B

Metric calculation.

### Worker 4C

Result serialization.

Metrics:

- peak temperature
- mean temperature
- maximum \(\Delta T\)
- pump energy
- thermal-limit violations
- time above thermal limit

### Sol integration checkpoint

Run:

```text
same scenario
+
same initial state
+
different controllers
```

and verify comparison fairness.

---

# 15. Phase 5 — Synthetic Dataset Generation

Do this only after Sol accepts the physical simulator.

## Job 5A — Parameter sampler

Randomize:

- initial SOC
- initial temperature
- ambient temperature
- current profiles
- coolant flow
- resistance
- thermal coupling
- cooling effectiveness

---

## Job 5B — Dataset generator

Generate trajectories using the simulator.

Output:

```text
state_t
action_t
state_t+1
trajectory_id
scenario metadata
```

---

## Job 5C — Dataset split

Split by:

```text
trajectory
```

not by individual rows.

Required:

```text
train
validation
test
```

Avoid leakage between temporally adjacent samples.

---

# 16. Phase 6 — Data-Driven Thermal Surrogate

Start simple.

Do not permit workers to introduce GNNs, PINNs, transformers, or RL yet.

## Job 6A — Dataset loader

Return normalized model features and targets.

---

## Job 6B — Thermal MLP

Input:

\[
[I,SOC,T_1...T_{12},T_{amb},T_c,\dot m]
\]

Output:

\[
\Delta T_1...\Delta T_{12}
\]

Prefer residual prediction:

\[
\hat T_{t+1}=T_t+\Delta\hat T
\]

---

## Job 6C — Training loop

Requirements:

- seeded training
- train/validation metrics
- checkpointing
- early stopping if appropriate
- device selection
- smoke-test configuration

---

## Job 6D — Evaluation

Report:

- MAE
- RMSE
- per-cell error
- worst-cell error

Sol reviews model behavior before continuing.

---

# 17. Phase 7 — Physics-Informed Learning

Only begin once pure data-driven forecasting works.

## Job 7A — Physics residual

Implement:

\[
r_i =
mC_p
\frac{\hat T_i-T_i}{\Delta t}
-
(
Q_{gen,i}
-Q_{cool,i}
+Q_{neighbor,i}
)
\]

Reuse physics functions where possible.

Do not duplicate physical equations in multiple locations.

---

## Job 7B — Combined loss

Implement:

\[
L=L_{data}+\lambda L_{physics}
\]

Make \(\lambda\) configurable.

---

## Job 7C — Comparative experiment

Train:

```text
Model A: data loss only
Model B: data + physics loss
```

Compare in-distribution and out-of-distribution performance.

Sol must not assume physics-informed training performs better.

Use measured results.

---

# 18. Phase 8 — Forecast Engine

Create abstraction:

```python
class ThermalForecaster:
    def predict_next(...):
        ...

    def rollout(...):
        ...
```

## Job 8A

Single-step prediction wrapper.

## Job 8B

Autoregressive rollout.

## Job 8C

Forecast evaluation.

Evaluate:

```text
30 s
60 s
120 s
```

Track error versus horizon.

---

# 19. Phase 9 — Predictive Controller

Do not start with RL.

Implement a simple receding-horizon optimizer.

## Job 9A — Objective

\[
J=
w_TT_{max}
+w_\Delta\Delta T
+w_PE_{pump}
+w_SP_{unsafe}
\]

All weights configurable.

---

## Job 9B — Candidate-action rollout

Generate candidate coolant flows:

```python
np.linspace(min_flow, max_flow, N)
```

Use the digital twin to forecast each.

---

## Job 9C — Action selection

Return:

\[
u^*=\arg\min J
\]

---

## Job 9D — Closed-loop controller

At every control interval:

```text
measure
↓
forecast
↓
optimize
↓
apply first action
↓
repeat
```

### Sol integration checkpoint

Compare against threshold/PID baseline.

---

# 20. Phase 10 — Fault Injection

Faults belong inside the simulator.

## Job 10A — Resistance fault

Allow:

\[
R_k\rightarrow\alpha R_k
\]

with configurable:

- cell
- start time
- severity

---

## Job 10B — Anomaly residual

Calculate:

\[
e_i=
T_{measured,i}
-
T_{expected,i}
\]

---

## Job 10C — Persistent anomaly detection

Use rolling statistics or another lightweight detector.

Output:

```text
cell
anomaly score
time detected
severity
```

---

## Job 10D — Lead-time metric

Calculate:

\[
t_{threshold}-t_{detection}
\]

This becomes a key demo metric.

---

# 21. Phase 11 — Dashboard

Dashboard work begins only after the experiment runner works.

Potential parallel jobs:

### Worker 11A

3×4 thermal heatmap.

### Worker 11B

Temperature and cooling time-series plots.

### Worker 11C

Metric cards.

### Worker 11D

Fault control panel.

### Worker 11E

Baseline-vs-ThermoTwin comparison panel.

Sol integrates these into one Streamlit application.

UI should consume existing experiment interfaces.

The UI must not contain simulation logic.

---

# 22. Phase 12 — Benchmark Suite

Status: implemented and audited at Checkpoint P12C. The deterministic benchmark
is a descriptive evaluation of six configured synthetic scenarios. The saved
release evaluation uses quick mode and is labeled `workflow_only`; it does not
establish model/controller superiority, statistical significance, safety
certification, or real-world generalization. Full mode requires non-smoke model
provenance. Quick mode remains workflow-only even when supplied full-trained
models.

The accepted scenario configuration covers:

```text
Normal operation
High-current operation
Hot ambient
Weak cooling
Fast charging
Cell resistance fault
```

For every scenario, run all four variants from the same scenario and seed:

```text
Reactive threshold controller + physical simulator
PID controller + physical simulator
Predictive controller + data-only thermal twin
Predictive controller + physics-informed thermal twin
```

Record metrics and provenance in a complete, reloadable artifact bundle. During
forecast rollouts, current exogenous inputs persist into the future; future
simulator truth and fault labels are not available to the controller.

---

# 23. Final Evaluation

Final evaluation aggregates saved model, forecast, controller, and fault artifacts
and exports standalone figures. The saved release figures and descriptive values
are documented in `README.md`; generated `results/` artifacts are local outputs
and are not committed. The observed 120-second forecast errors are very large,
especially OOD. Do not infer an overall winner from individual metrics.
Repeated independently sampled scenario families and uncertainty analysis remain
future work beyond Phase 12.

The four variants are compared on peak and mean temperature, maximum temperature
spread, pump energy, and thermal-violation count and duration. Model evaluation
reports ID/OOD one-step error and physics residual; forecast evaluation reports
30/60/120-second error; fault evaluation reports detection time and lead time.

The fault artifact reports suspected cell and lead time separately from injected
fault truth. Its detector model is not cryptographically bound to a checkpoint
because the saved fault artifact lacks checkpoint/configuration hashes and an
evidence-mode field. This remains a documented limitation.

---

# 24. Codex Execution Rules

When working through the VS Code Codex extension, Sol should treat the editor repository as the canonical workspace.

Before each delegation:

1. inspect git status
2. inspect relevant files
3. verify no conflicting worker task is modifying them
4. define acceptance criteria
5. delegate

After each worker:

1. inspect returned diff
2. inspect changed files directly
3. run required tests
4. fix or reject bad changes
5. commit/stabilize logically complete work before beginning dependent work

Avoid accumulating many unreviewed agent changes.

---

# 25. Context Discipline

Sol should not dump the entire repository into every Luna worker.

Give workers only what they need.

For example, a worker implementing the cooling model may need:

```text
ARCHITECTURE cooling section
PackConfig definition
existing physics interfaces
cooling.py
test conventions
```

It does not need:

```text
dashboard code
training pipeline
anomaly detector
README presentation material
```

This reduces distraction and architectural drift.

---

# 26. Worker Isolation

Every delegated job should specify:

```text
Allowed files
Read-only context files
Forbidden files
```

Example:

```text
Allowed:
src/thermotwin/physics/cooling.py
tests/test_cooling.py

Read-only:
src/thermotwin/core/types.py
configs/pack.yaml

Do not modify:
environment.py
pack.py
controllers/*
```

This is the default delegation style.

---

# 27. When Sol Should Not Parallelize

Do not parallelize tasks when:

- one establishes an API used by the other
- both modify the same module
- physical assumptions are still unsettled
- an upstream implementation has not been validated
- tests for the shared subsystem are failing

Example:

```text
Do NOT simultaneously implement:
BatteryEnvironment
PredictiveController
```

if the state/action interfaces are not yet stable.

---

# 28. When Sol Should Parallelize

Good examples:

After `BatteryState` is frozen:

```text
Worker A → topology
Worker B → current profiles
Worker C → metric utilities
```

After experiment interfaces are frozen:

```text
Worker A → heatmap
Worker B → comparison table
Worker C → anomaly panel
```

After dataset schema is frozen:

```text
Worker A → loader
Worker B → normalization utilities
Worker C → evaluation metrics
```

---

# 29. Failure Protocol

If a Luna worker fails:

### First failure

Sol inspects:

- error
- assumptions
- test failure
- interface mismatch

Then refine the task and retry.

### Repeated failure

Reduce job scope.

Example:

Instead of:

```text
Implement predictive controller
```

split into:

```text
Implement objective function
```

then:

```text
Implement candidate rollout
```

then:

```text
Implement action selection
```

### Architectural conflict

Sol resolves it directly before delegating further work.

Workers must not independently reconcile competing architectural interpretations.

---

# 30. Definition of Done for a Worker Job

A worker job is not complete merely because code was written.

It must satisfy:

```text
requested behavior implemented
+
scope respected
+
tests added or updated
+
verification executed
+
results reported
```

Sol decides final acceptance.

---

# 31. Definition of Done for a Phase

A phase is complete only when:

1. all constituent jobs are accepted
2. entire test suite passes
3. at least one integration path executes
4. public interfaces are stable enough for the next phase
5. implementation matches architecture
6. no critical TODO blocks dependent work

---

# 32. MVP Stop Condition

If hackathon time becomes constrained, Sol should prioritize this exact chain:

Physics-informed learning is part of the mandatory MVP path.

The required MVP sequence is:

Battery simulator
    ↓
Threshold/PID baseline
    ↓
Experiment framework
    ↓
Synthetic dataset
    ↓
Data-driven ML thermal predictor
    ↓
Physics-informed thermal predictor
    ↓
120-second rollout
    ↓
Predictive coolant controller
    ↓
Baseline comparison
    ↓
Dashboard

Optional after this:

fault detection
GNN
RL
advanced CFD
aging model


A complete simple system is better than several incomplete advanced systems.

---

# 33. Stretch-Goal Delegation

Only after MVP passes.

Possible independent branches:

```text
Branch A
GNN thermal surrogate

Branch B
Uncertainty estimation

Branch C
Sparse sensor reconstruction

Branch D
Battery aging model

Branch E
Real drive-cycle import

Branch F
Continuous MPC optimization
```

Sol should benchmark each against the existing MVP rather than replacing working components blindly.

---

# 34. Git Discipline

Prefer small logical changes.

Suggested progression:

```text
feat: add core battery state models
feat: implement cell thermal dynamics
feat: add pack thermal conduction
feat: add liquid cooling model
feat: add battery simulation environment
feat: add threshold thermal controller
feat: add experiment runner
feat: add synthetic dataset generator
feat: add thermal surrogate
feat: add forecast rollout
feat: add predictive controller
feat: add anomaly detection
feat: add dashboard
```

A worker's changes should ideally correspond to one coherent logical diff.

Sol remains responsible for repository-wide coherence.

---

# 35. First Codex Execution Sequence

When development begins, Sol should execute approximately:

```text
1. Read AGENTS.md
2. Read PLAN.md
3. Read ARCHITECTURE.md
4. Inspect repository
5. Determine existing implementation
6. Freeze core interfaces
7. Delegate repository/bootstrap work if needed
8. Delegate BatteryState/config implementation
9. Review
10. Delegate thermal equations
11. Review + test
12. Delegate cooling model
13. Review + test
14. Delegate pack/environment integration
15. Run full simulation test
16. Implement baseline
17. Run first complete experiment
```

No ML implementation should start before step 17 produces believable thermal trajectories.

---

# 36. Final Operating Principle

Sol is responsible for **thinking globally**.

Luna is responsible for **executing locally**.

The intended pattern is:

```text
SOL
Understand
Architect
Decompose
Delegate
Review
Integrate
Validate
Decide
        |
        v
LUNA
Inspect scoped context
Implement one job
Test
Report
```

The orchestrator should continuously minimize the size and ambiguity of worker tasks.

For ThermoTwin, the most important constraint is:

\[
\boxed{
\text{Correct physics}
\rightarrow
\text{reliable simulator}
\rightarrow
\text{trustworthy data}
\rightarrow
\text{useful AI}
}
\]

No agent should skip this dependency chain.
