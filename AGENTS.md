# ThermoTwin — Agent Instructions

## 1. Project Goal

ThermoTwin is a hackathon prototype of an AI-assisted Battery Thermal Management System for an electric-vehicle battery pack.

The system must demonstrate that a predictive controller using a **physics-informed thermal digital twin** can outperform a conventional reactive thermal controller.

GitHub Link: https://github.com/dp-rjt-prsd/ThermoTwin.git

The central comparison is:

### Reactive BTMS

- reacts to measured temperature
- uses threshold or PID-based cooling
- does not explicitly predict future thermal state

### ThermoTwin

- estimates the current thermal state
- predicts future cell-level temperatures
- incorporates physical thermal constraints during learning
- detects developing hotspots and abnormal heat generation
- proactively selects coolant flow
- attempts to minimize peak temperature, temperature non-uniformity, and cooling energy

The project must not degrade into a generic battery-temperature forecasting demo.

The intended contribution is:

> A physics-informed thermal digital twin that predicts future pack temperatures and uses those predictions to proactively control cooling.

---

# 2. Development Method

ThermoTwin is developed using an **Orchestrator–Worker agent architecture** inside the Codex extension for VS Code.

Two agent roles exist.

---

# 3. Orchestrator

**Model:** Sol 5.6

Sol is the primary reasoning and coordination agent.

Sol is responsible for:

- understanding the user's goal
- reading project-level documentation
- inspecting the current repository
- understanding existing implementation state
- maintaining architectural coherence
- resolving ambiguous technical decisions
- decomposing work into small isolated jobs
- selecting relevant context for each worker
- spawning Luna workers
- reviewing worker changes
- validating worker claims
- integrating modules
- running repository-level tests
- identifying regressions
- deciding the next implementation step

Sol should think globally.

Sol should not delegate architectural responsibility to Luna.

---

# 4. Worker

**Model:** Luna 6

Luna is an execution-focused coding worker.

Each Luna worker should receive:

1. exactly one clearly defined task
2. the minimum context needed to execute it
3. explicit files to inspect
4. explicit files it may modify
5. required interfaces
6. expected behavior
7. edge cases
8. verification commands
9. completion criteria

Luna should:

- inspect the supplied context
- implement only the requested change
- avoid unrelated refactoring
- preserve existing interfaces
- add or update tests where appropriate
- run the requested tests
- report exactly what changed
- report test results
- report assumptions and unresolved issues

Luna should not independently redesign the repository.

If Luna encounters an architectural problem, it should report the problem to Sol.

---

# 5. Core Agent Workflow

The development loop is:

```text
User Goal
    ↓
Sol inspects repository
    ↓
Sol determines current state
    ↓
Sol designs next implementation increment
    ↓
Sol freezes any required interfaces
    ↓
Sol decomposes increment into isolated jobs
    ↓
Luna workers execute jobs
    ↓
Workers run local verification
    ↓
Sol reviews diffs
    ↓
Sol runs integration verification
    ↓
Sol accepts / rejects / repairs changes
    ↓
Sol determines next increment
```

Sol is responsible for ensuring that locally correct worker changes form a globally correct system.

---

# 6. Context Discipline

Do not provide Luna with unnecessary repository context.

A worker implementing the cooling model may need:

```text
src/thermotwin/physics/cooling.py
src/thermotwin/core/types.py
configs/pack.yaml
tests/test_cooling.py
```

It probably does not need:

```text
dashboard code
model-training code
fault detection
README presentation material
```

Keep worker prompts narrow.

This reduces:

- token waste
- accidental refactoring
- architectural drift
- duplicated logic
- conflicting edits

---

# 7. Worker Isolation

Each delegated job should explicitly define:

```text
FILES TO READ

FILES YOU MAY MODIFY

FILES YOU MUST NOT MODIFY
```

Example:

```text
FILES TO READ
- src/thermotwin/core/types.py
- src/thermotwin/physics/pack.py

FILES YOU MAY MODIFY
- src/thermotwin/physics/cooling.py
- tests/test_cooling.py

DO NOT MODIFY
- src/thermotwin/physics/environment.py
- src/thermotwin/control/*
- src/thermotwin/models/*
```

Luna must respect these boundaries.

---

# 8. Worker Task Template

Sol should structure delegated jobs approximately as:

```text
TASK

Implement <specific behavior>.

CONTEXT

<minimal technical context>

FILES TO READ

- ...

FILES YOU MAY MODIFY

- ...

DO NOT MODIFY

- ...

REQUIRED INTERFACES

- ...

EXPECTED BEHAVIOR

- ...

EDGE CASES

- ...

VERIFICATION

Run:
<commands>

DONE WHEN

<acceptance criteria>

REPORT

Return:
1. summary
2. files changed
3. tests added/changed
4. tests executed
5. result
6. assumptions/issues
```

---

# 9. Task Granularity

Worker tasks should be small enough to reason about independently.

Good:

```text
Implement neighbor conduction in pack.py and add unit tests.
```

Good:

```text
Implement the physics residual calculation used by the
physics-informed loss.
```

Good:

```text
Implement candidate coolant-flow evaluation using the existing
ThermalForecaster interface.
```

Bad:

```text
Implement the battery simulator.
```

Bad:

```text
Implement the AI.
```

Bad:

```text
Finish the controller.
```

Whenever a task feels broad, Sol should decompose it further.

---

# 10. Parallelism

Parallel Luna workers are encouraged only for truly independent work.

Good:

```text
Worker A → drive-cycle generators

Worker B → experiment metrics

Worker C → pack-topology helper
```

assuming none modify overlapping files.

Avoid:

```text
Worker A → models/losses.py

Worker B → models/losses.py

Worker C → models/losses.py
```

Sol should minimize concurrent edits to shared modules.

---

# 11. Dependency-Aware Execution

Do not parallelize dependent work.

Example:

```text
Physics interfaces
        ↓
Simulator
        ↓
Dataset schema
        ↓
Model inputs
        ↓
Physics-informed loss
        ↓
Forecaster
        ↓
Predictive controller
```

Do not implement downstream modules against unstable upstream interfaces.

---

# 12. Diff Review Protocol

After each worker returns, Sol must inspect the actual changes.

Sol should check:

### Scope

Did Luna modify only the requested files?

### Correctness

Does the implementation satisfy the task?

### Interface compatibility

Does it preserve the agreed API?

### Test quality

Do tests actually verify behavior rather than merely exercise code?

### Physics

For thermal code:

- are SI units consistent?
- are signs correct?
- does heat flow in the correct direction?
- is conservation approximately respected?
- are numerical values plausible?

### ML

For model code:

- are tensor shapes correct?
- is normalization consistent?
- are train/test boundaries clean?
- is gradient flow preserved?
- is physics loss differentiable?

### Control

For controller code:

- are action limits respected?
- is future information leaked?
- is optimization using the learned model rather than simulator truth?

### Architecture

Does the change preserve separation between:

```text
physics
models
control
experiments
diagnostics
visualization
```

Sol should reject worker changes that violate architecture even if their isolated tests pass.

---

# 13. Verification Hierarchy

## Worker verification

Workers run targeted tests.

Example:

```bash
pytest tests/test_cooling.py
```

or:

```bash
pytest tests/test_losses.py
```

or:

```bash
pytest tests/test_predictive_controller.py
```

Workers may also run:

```bash
ruff check <modified files>
```

if configured.

---

## Sol verification

Sol runs broader validation after integration.

At minimum:

```bash
pytest
```

Where appropriate:

```bash
python scripts/run_baseline.py
```

```bash
python scripts/generate_dataset.py --smoke-test
```

```bash
python scripts/train_model.py --smoke-test
```

```bash
python scripts/benchmark.py --quick
```

A phase is not complete just because worker-level tests pass.

---

# 14. Primary Success Criteria

The final system must demonstrate:

1. simulated 12-cell battery pack
2. time-varying electrical load
3. cell-level heat generation
4. cell-to-cell thermal coupling
5. liquid cooling
6. conventional baseline controller
7. data-driven thermal surrogate
8. physics-informed thermal surrogate
9. future temperature prediction
10. predictive coolant-flow optimization
11. fault/hotspot injection
12. anomaly detection
13. quantitative comparison
14. interactive visualization

At minimum report:

- maximum pack temperature
- mean pack temperature
- temperature spread

\[
\Delta T=T_{max}-T_{min}
\]

- cooling energy
- thermal violations
- forecast MAE/RMSE
- fault detection lead time
- physics-informed vs purely data-driven model performance

---

# 15. Project Scope

## Required

- synthetic battery thermal simulation
- lumped electrothermal modeling
- 12-cell battery pack
- liquid cold plate abstraction
- variable coolant flow
- synthetic current/charging profiles
- data-driven model baseline
- physics-informed ML
- multi-step thermal forecasting
- predictive control
- rule/PID baseline
- anomaly detection
- interactive dashboard
- reproducible experiments

## Not required for MVP

Do not block the project on:

- full electrochemistry
- COMSOL
- ANSYS
- full 3D CFD
- real EV battery hardware
- physical thermal-runaway testing
- reinforcement learning
- large transformers
- neural operators
- production BMS firmware

---

# 16. Reference Physical System

Assume a battery pack with 12 cells arranged:

```text
C1   C2   C3   C4
C5   C6   C7   C8
C9   C10  C11  C12
```

Adjacent cells exchange heat.

The pack sits on an abstract liquid cold plate.

The controllable action is coolant flow:

\[
u(t)=\dot m_{coolant}(t)
\]

The simulator should support configurable minimum and maximum coolant flow.

---

# 17. Physical Model

For cell \(i\):

\[
m_iC_{p,i}\frac{dT_i}{dt}
=
Q_{gen,i}
-
Q_{cool,i}
+
Q_{neighbor,i}
+
Q_{ambient,i}
\]

Initial irreversible heat model:

\[
Q_{gen,i}=I^2R_i
\]

Neighbor heat transfer:

\[
Q_{neighbor,i}
=
\sum_{j\in N(i)}
\frac{T_j-T_i}{R_{th,ij}}
\]

Cooling:

\[
Q_{cool,i}
=
h_i(\dot m)A_i(T_i-T_c)
\]

The simulator is the authoritative environment.

The neural network must not replace the physical simulator.

---

# 18. Simulator Responsibility

The simulator owns:

- SOC
- cell temperatures
- electrical heat generation
- thermal conduction
- cooling
- ambient heat exchange
- faults
- simulation time

Recommended interface:

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

---

# 19. Battery State

Recommended:

```python
@dataclass
class BatteryState:
    time: float
    current: float
    soc: float

    ambient_temperature: float
    coolant_inlet_temperature: float
    coolant_flow: float

    cell_temperatures: np.ndarray
    cell_resistances: np.ndarray
```

All physical quantities should use SI units internally unless explicitly documented otherwise.

---

# 20. Pack Topology

Represent cell adjacency explicitly.

Example:

```text
0 ---- 1 ---- 2 ---- 3
|      |      |      |
4 ---- 5 ---- 6 ---- 7
|      |      |      |
8 ---- 9 ----10 ----11
```

This topology should be reusable by:

- physical conduction model
- physics-informed residual
- future GNN implementation

Do not duplicate topology definitions.

---

# 21. Synthetic Dataset

Generate trajectories by varying:

- current profile
- C-rate
- SOC
- initial temperatures
- ambient temperature
- coolant temperature
- coolant flow
- cell resistance
- thermal coupling
- cooling effectiveness
- fault cell
- fault severity

Include:

- normal driving
- aggressive driving
- pulse loads
- sustained high load
- fast charging
- hot ambient conditions
- weak cooling
- abnormal resistance faults

Dataset splits must occur at trajectory/scenario level.

Do not randomly split individual adjacent timesteps between train and test.

---

# 22. Data-Driven Model Baseline

Before introducing the physics term, train a pure data-driven model.

Input:

\[
x_t=
[
I,
SOC,
T_1...T_{12},
T_{ambient},
T_{coolant},
\dot m
]
\]

Output:

\[
\Delta T_1...\Delta T_{12}
\]

Preferred formulation:

\[
\hat T_{t+1}
=
T_t+\Delta\hat T
\]

Recommended initial architecture:

```text
Input
 ↓
Linear
 ↓
GELU
 ↓
Linear
 ↓
GELU
 ↓
Linear
 ↓
12 temperature deltas
```

Keep the model small.

The purpose of the pure data-driven model is to create a controlled baseline for evaluating the physics-informed model.

---

# 23. Physics-Informed Learning Is Required

Physics-informed learning is a **core project requirement**, not a stretch goal.

ThermoTwin should contain at least two trained variants:

```text
Model A
Data-driven model

Model B
Physics-informed model
```

The models should otherwise be comparable where practical.

---

# 24. Physics-Informed Loss

The preferred training objective is:

\[
L=
L_{data}
+
\lambda_{physics}L_{physics}
\]

Data loss:

\[
L_{data}
=
\frac{1}{N}
\sum_i
(\hat T_i-T_i)^2
\]

Physics residual:

\[
r_i=
m_iC_{p,i}
\frac{\hat T_i-T_i}{\Delta t}
-
(
Q_{gen,i}
-
Q_{cool,i}
+
Q_{neighbor,i}
+
Q_{ambient,i}
)
\]

Physics loss:

\[
L_{physics}
=
\frac{1}{N}
\sum_i r_i^2
\]

Use normalization/scaling where needed.

The physics term must remain differentiable.

Do not accidentally detach tensors or convert them to NumPy inside the training path.

---

# 25. Physics Code Reuse

The physics-informed loss should reuse existing physical-model functions where technically sensible.

Do not maintain separate versions of:

```text
heat generation
conduction
cooling
```

inside:

```text
physics/
```

and:

```text
models/losses.py
```

unless there is a clear reason.

Sol should design a shared differentiable representation or closely aligned implementation.

Duplication creates silent divergence between simulator physics and learning physics.

---

# 26. Physics-Informed Model Evaluation

The physics-informed model must be compared against the purely data-driven model.

Evaluate:

### In-distribution

- normal validation trajectories
- normal coolant ranges
- familiar temperatures

### Out-of-distribution

- hotter ambient
- higher current
- unusual cooling rates
- changed resistance
- faulted cell
- unseen load profiles

Report:

- MAE
- RMSE
- maximum cell error
- long-horizon rollout error
- physical residual
- unstable/unphysical predictions

Do not claim that physics-informed learning is better unless results support it.

Even if predictive accuracy is similar, reduced physical inconsistency or improved OOD stability is still relevant if demonstrated quantitatively.

---

# 27. Physics-Informed Learning Task Decomposition

Sol should not delegate "implement physics-informed learning" as one job.

Recommended decomposition:

### Worker A

Implement differentiable heat-generation term.

### Worker B

Implement differentiable cooling and neighbor-conduction residuals.

### Worker C

Implement `physics_residual(...)`.

### Worker D

Implement combined loss:

```python
loss = data_loss + lambda_physics * physics_loss
```

### Worker E

Add unit tests for residual behavior.

### Worker F

Integrate loss into training configuration.

### Worker G

Implement data-vs-physics-informed comparison experiment.

Sol reviews all mathematical assumptions before integration.

---

# 28. Physics Loss Tests

At minimum test:

### Equilibrium

For a physically balanced state, residual should be near zero.

### Wrong temperature derivative

Artificially incorrect \(\Delta T\) should increase residual.

### Increased resistance

At fixed current, increasing resistance should increase expected heat generation.

### Cooling

Increasing coolant flow should change the cooling term in the correct direction.

### Conduction

Neighbor heat-transfer signs should be correct.

### Differentiability

Calling:

```python
loss.backward()
```

must populate gradients for model parameters.

---

# 29. Forecasting

ThermoTwin must perform multi-step forecasting.

Required target horizons:

```text
30 seconds
60 seconds
120 seconds
```

Optional:

```text
300 seconds
```

Autoregressive rollout is acceptable.

Track error growth with forecast horizon.

---

# 30. Forecast Interface

The controller should not directly depend on model implementation.

Use an abstraction such as:

```python
class ThermalForecaster:
    def predict_next(
        self,
        state,
        action,
    ):
        ...

    def rollout(
        self,
        state,
        action_sequence,
        horizon,
    ):
        ...
```

The forecaster may internally use:

- data-driven MLP
- physics-informed MLP
- future GNN
- ensemble

The controller should not care.

---

# 31. Baseline Controller

Implement a conventional controller before predictive control.

Minimum:

```text
Tmax < 30°C
→ minimum cooling

30–35°C
→ low cooling

35–40°C
→ medium cooling

Tmax >= 40°C
→ maximum cooling
```

Preferably also implement PID.

This gives ThermoTwin a meaningful benchmark.

---

# 32. Predictive Controller

Do not start with reinforcement learning.

Use receding-horizon candidate evaluation.

At each control interval:

1. read current state
2. generate candidate coolant flows
3. forecast each candidate using ThermoTwin
4. calculate cost
5. choose best action
6. apply only first action
7. repeat

Objective:

\[
J=
w_TT_{max}
+
w_\Delta\Delta T
+
w_PP_{pump}
+
w_SP_{safety}
\]

Safety violations should receive a large penalty.

---

# 33. Pump Energy

Approximate initially:

\[
P_{pump}\propto\dot m^3
\]

or:

\[
P_{pump}=k_p\dot m^3
\]

Document the approximation.

Do not present this as a manufacturer-specific pump curve unless such data is actually used.

---

# 34. Fault Injection

Primary fault:

```text
increase internal resistance of one cell
```

Example:

\[
R_k\rightarrow\alpha R_k
\]

with:

\[
\alpha>1
\]

Because:

\[
Q=I^2R
\]

this produces abnormal heat generation.

The digital twin must not receive the fault label during inference.

---

# 35. Anomaly Detection

Compare observed and expected temperatures:

\[
e_i(t)
=
T_{observed,i}(t)
-
T_{predicted,i}(t)
\]

Use rolling residual statistics.

Possible anomaly score:

\[
A_i=
\frac{|e_i-\mu_i|}{\sigma_i+\epsilon}
\]

Avoid triggering on single noisy samples.

Record:

- cell ID
- score
- detection time
- predicted hotspot temperature
- severity

---

# 36. Fault Claims

Do not claim:

```text
thermal runaway detection
```

unless the simulator actually models thermal runaway dynamics.

Prefer:

```text
abnormal thermal behavior detection
```

```text
hotspot precursor detection
```

```text
fault-induced thermal anomaly detection
```

---

# 37. Experimental Integrity

Never fabricate performance improvements.

Every reported number must come from saved experiment output.

Store:

- configuration
- random seed
- controller
- model checkpoint
- scenario
- time series
- metrics

under:

```text
results/
```

Where possible, presentation plots should be generated directly from experiment artifacts.

---

# 38. Required Comparison Matrix

The final experiment suite should ideally compare:

```text
Reactive Controller + Physical Simulator

Predictive Controller + Data-Driven Twin

Predictive Controller + Physics-Informed Twin
```

This is stronger than only comparing:

```text
Reactive vs AI
```

because it shows whether the physics-informed contribution actually matters.

---

# 39. Key Metrics

Report:

### Thermal safety

\[
T_{max}
\]

### Uniformity

\[
\Delta T=T_{max}-T_{min}
\]

### Energy

\[
E_{pump}
\]

### Safety violations

number and duration

### Forecast performance

MAE / RMSE

### Physics consistency

mean physics residual

### Fault performance

detection lead time

---

# 40. Dashboard

Recommended:

- Streamlit
- Plotly

Required views:

### Current pack heatmap

3 × 4 grid.

### Predicted heatmap

Configurable future horizon.

### Thermal time series

Show:

- cell temperatures
- Tmax
- mean temperature
- thermal threshold

### Cooling control

Show coolant-flow command.

### Model comparison

Allow comparison of:

```text
Data-Driven Twin

Physics-Informed Twin
```

### Controller comparison

Show:

```text
Reactive BTMS

ThermoTwin BTMS
```

### Fault panel

Display:

- suspected cell
- anomaly score
- predicted hotspot
- estimated threshold crossing

---

# 41. Software Engineering Rules

Use:

- Python 3.11+
- NumPy
- SciPy
- Pandas
- PyTorch
- Plotly
- Streamlit
- pytest

Prefer:

- type hints
- dataclasses
- pure functions for reusable physics
- explicit configuration

Avoid:

- giant modules
- hidden global state
- magic physical constants
- duplicated equations
- UI logic inside physics code
- simulator truth leaking into controller decisions

---

# 42. Configuration

Keep physical and training parameters outside implementation logic.

Example:

```text
configs/
    pack.yaml
    simulation.yaml
    training.yaml
    controller.yaml
```

Important parameters include:

```text
cell mass
heat capacity
cell resistance
capacity
thermal resistance
cooling coefficients
coolant flow limits
ambient temperature
thermal safety limit
physics-loss weight
forecast horizon
controller cost weights
```

---

# 43. Testing Requirements

At minimum:

### Zero-current test

With no temperature gradient and:

\[
I=0
\]

the pack should not spontaneously heat.

### Joule heating test

\[
Q\propto I^2
\]

### Resistance test

\[
Q\propto R
\]

### Cooling test

Higher flow should increase heat removal.

### Conduction test

Heat should move hot → cold.

### Energy sanity test

Temperature evolution should remain physically plausible.

### Determinism test

Identical configuration + seed should reproduce results.

### Controller action test

Actions must remain within legal coolant-flow limits.

### Physics-loss test

Residual behaves correctly and remains differentiable.

### Rollout test

Forecast output shape and state evolution remain valid.

---

# 44. Repository Separation

Recommended structure:

```text
src/thermotwin/

    physics/
        cell.py
        pack.py
        cooling.py
        topology.py
        environment.py

    scenarios/
        drive_cycles.py
        charging.py
        faults.py

    data/
        generate.py
        dataset.py
        preprocessing.py

    models/
        mlp.py
        losses.py
        forecaster.py
        train.py

    control/
        base.py
        threshold.py
        pid.py
        predictive.py
        objective.py

    diagnostics/
        anomaly.py
        metrics.py

    experiments/
        runner.py
        benchmark.py

    visualization/
        heatmap.py
        plots.py
        dashboard.py
```

---

# 45. Dependency Direction

Prefer:

```text
visualization
      |
experiments
      |
control ------> models
  |              |
  +----------> physics
                 |
               config
```

The physics package should not depend on:

- PyTorch training loops
- controllers
- Streamlit
- visualization

Where physics-informed learning requires differentiable equations, structure shared physics carefully instead of introducing circular dependencies.

Sol is responsible for enforcing this.

---

# 46. Development Priority

Use this order:

```text
core interfaces
    ↓
thermal physics
    ↓
battery simulator
    ↓
baseline controller
    ↓
experiment runner
    ↓
synthetic dataset
    ↓
data-driven thermal model
    ↓
physics-informed thermal model
    ↓
multi-step forecaster
    ↓
predictive controller
    ↓
fault detection
    ↓
dashboard
    ↓
benchmark suite
```

Physics-informed learning is explicitly part of the main implementation path.

---

# 47. MVP Definition

The minimum acceptable ThermoTwin demo contains:

```text
12-cell simulator
+
reactive controller
+
synthetic dataset
+
data-driven model
+
physics-informed model
+
120-second temperature forecast
+
predictive coolant controller
+
quantitative benchmark
+
dashboard
```

If time becomes constrained, reduce model sophistication or UI polish before removing physics-informed learning.

---

# 48. Features to Cut Before Physics-Informed Learning

If hackathon time runs short, cut in this order:

1. GNN
2. reinforcement learning
3. CFD validation
4. battery aging
5. real-world drive-cycle import
6. sophisticated anomaly detection
7. advanced coolant modeling
8. UI polish

Do not remove the data-driven vs physics-informed comparison unless absolutely necessary.

---

# 49. Stretch Goals

After the core system works:

1. Graph Neural Network
2. uncertainty-aware forecasting
3. ensemble models
4. continuous MPC optimization
5. sparse sensor reconstruction
6. battery aging objective
7. fast-charging optimization
8. neural ODE
9. CFD-generated supervision
10. hardware-in-the-loop
11. real thermal sensor input
12. reinforcement learning

---

# 50. Git Discipline

Prefer small coherent diffs.

Examples:

```text
feat: add battery state and configuration types

feat: implement cell joule heating

feat: add pack thermal conduction

feat: implement liquid cooling model

feat: add simulation environment

feat: add reactive controller

feat: add experiment runner

feat: generate synthetic thermal trajectories

feat: train data-driven thermal surrogate

feat: add differentiable physics residual

feat: train physics-informed thermal surrogate

feat: add thermal forecast rollout

feat: add predictive cooling controller

feat: add thermal anomaly detection

feat: add thermotwin dashboard
```

Do not accumulate large unreviewed agent diffs.

---

# 51. Failure Protocol

If a Luna worker fails:

## First attempt

Sol inspects:

- error trace
- modified files
- assumptions
- interface mismatch
- failing tests

Then refine and retry.

## Repeated failure

Reduce task scope.

Example:

Instead of:

```text
Implement physics-informed training
```

use:

```text
Implement physics residual function
```

then:

```text
Add residual unit tests
```

then:

```text
Implement combined loss
```

then:

```text
Integrate combined loss into training loop
```

---

# 52. Definition of Done — Worker Job

A worker task is complete only when:

```text
requested behavior implemented
+
scope respected
+
tests added/updated
+
verification executed
+
results reported
```

Sol performs final acceptance.

---

# 53. Definition of Done — Phase

A phase is complete only when:

1. all relevant worker jobs are accepted
2. test suite passes
3. integration path executes successfully
4. interfaces are stable enough for downstream work
5. results are physically plausible
6. architecture remains consistent
7. no critical unresolved issue blocks the next phase

---

# 54. Orchestrator Operating Principle

Sol should continuously ask:

```text
What is the smallest independently testable job
that advances the critical path?
```

Sol thinks globally.

Luna executes locally.

The intended relationship is:

```text
SOL
Understand
Reason
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
Implement
Test
Report
```

---

# 55. Scientific Operating Principle

The project dependency chain is:

\[
\boxed{
\text{Correct Physics}
\rightarrow
\text{Reliable Simulator}
\rightarrow
\text{Trustworthy Dataset}
\rightarrow
\text{Data-Driven Baseline}
\rightarrow
\text{Physics-Informed Twin}
\rightarrow
\text{Predictive Control}
}
\]

Agents must not skip this chain.

The physics-informed model is not decorative.

It must be trained, evaluated, and compared against the data-driven baseline.

The final ThermoTwin claim is:

> A physics-informed digital twin learns the thermal dynamics of an EV battery pack, forecasts future cell-level temperatures, and enables proactive cooling decisions that balance thermal safety, temperature uniformity, and cooling energy.