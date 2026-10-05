# ThermoTwin — System Architecture

## 1. System Overview

ThermoTwin consists of four conceptually separate systems:

```text
PHYSICAL SYSTEM
      ↓
DIGITAL TWIN
      ↓
PREDICTIVE CONTROLLER
      ↓
VISUALIZATION / DIAGNOSTICS
```

During development, the physical battery is replaced by an electrothermal simulator.

The key architectural requirement is that the **simulator and digital twin remain separate**.

The simulator represents ground truth.

The digital twin attempts to learn/predict that ground truth.

---

# 2. High-Level Architecture

```text
                       DRIVE / CHARGE PROFILE
                                |
                                v
                    +-----------------------+
                    |  Battery Environment  |
                    |                       |
                    | Electrothermal Model  |
                    +-----------+-----------+
                                |
                                | measurements
                                v
                    +-----------------------+
                    |   State Estimator /   |
                    |     Data Pipeline     |
                    +-----------+-----------+
                                |
                                v
                    +-----------------------+
                    |      ThermoTwin       |
                    |                       |
                    | Physics-Informed ML   |
                    | Thermal Surrogate     |
                    +-----------+-----------+
                                |
                     future thermal states
                                |
                                v
                    +-----------------------+
                    | Predictive Controller |
                    |                       |
                    | Candidate Actions     |
                    | Rollout + Optimizer   |
                    +-----------+-----------+
                                |
                         coolant flow
                                |
                                v
                    +-----------------------+
                    |       Pump /          |
                    | Cooling System Model  |
                    +-----------+-----------+
                                |
                                +----------+
                                           |
                                           v
                                   Battery Environment
```

The entire loop repeats continuously.

---

# 3. Control Loop

At time \(t\):

```text
Battery Simulator
      |
      | state(t)
      v
ThermoTwin
      |
      | predicted T(t:t+H)
      v
Predictive Optimizer
      |
      | optimal coolant flow
      v
Battery Simulator
      |
      | state(t+1)
      +-------------------->
```

Mathematically:

\[
x_{t+1}=f_{physical}(x_t,u_t)
\]

The digital twin approximates:

\[
\hat x_{t+1}=f_\theta(x_t,u_t)
\]

The controller calculates:

\[
u_t^*
=
\arg\min_u
J(
f_\theta(x_t,u)
)
\]

---

# 4. Physical Simulator

## Responsibilities

The simulator owns:

- battery state
- SOC
- cell temperatures
- heat generation
- thermal conduction
- cooling
- ambient interaction
- faults
- simulation time

The simulator must not depend on the ML model.

Interface:

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

# 5. Battery State

Recommended representation:

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

Shape:

```text
cell_temperatures -> (12,)
cell_resistances  -> (12,)
```

---

# 6. Pack Topology

Physical layout:

```text
  0 ---- 1 ---- 2 ---- 3
  |      |      |      |
  4 ---- 5 ---- 6 ---- 7
  |      |      |      |
  8 ---- 9 ----10 ----11
```

Edges represent thermal conduction.

Example:

```python
edges = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),

    (0, 4), (4, 8),
    (1, 5), (5, 9),
    (2, 6), (6, 10),
    (3, 7), (7, 11),
]
```

This representation makes future migration to a GNN straightforward.

---

# 7. Thermal Dynamics

For each cell:

\[
mC_p\dot T_i
=
Q_{gen,i}
-
Q_{cool,i}
+
Q_{neighbor,i}
+
Q_{ambient,i}
\]

## Electrical heating

MVP:

\[
Q_{gen,i}=I^2R_i
\]

Later:

\[
Q_{gen}
=
I(V-U_{OCV})
-
IT\frac{\partial U_{OCV}}{\partial T}
\]

may be introduced for more realistic irreversible + reversible heat generation.

---

# 8. Cooling Model

The cold plate is abstracted through an effective heat-transfer coefficient:

\[
Q_{cool,i}
=
h(\dot m)A(T_i-T_c)
\]

Initially:

\[
h(\dot m)
=
h_0+k\dot m^\alpha
\]

where constants are configurable.

This model deliberately abstracts:

- channel geometry
- local fluid velocity
- Reynolds number
- Nusselt number
- detailed CFD

The abstraction should be clearly documented in the presentation.

---

# 9. Coolant Temperature

MVP:

Assume constant coolant inlet temperature.

Improved model:

Track coolant heating as it travels through the pack.

Conceptually:

```text
Tin
 ↓
Cells 1–4
 ↓
Cells 5–8
 ↓
Cells 9–12
 ↓
Tout
```

Approximate:

\[
\dot m C_{p,coolant}
(T_{out}-T_{in})
=
Q_{removed}
\]

This creates realistic spatial cooling non-uniformity.

---

# 10. Digital Twin Architecture

## Input

```text
current
SOC
ambient temperature
coolant inlet temperature
coolant flow
T1
T2
...
T12
```

Optional:

```text
previous current
previous coolant flow
cell resistance estimates
SOH
```

## Model

MVP:

```text
                STATE
                  |
                  v
           +-------------+
           | Linear 128  |
           +-------------+
                  |
                GELU
                  |
           +-------------+
           | Linear 256  |
           +-------------+
                  |
                GELU
                  |
           +-------------+
           | Linear 128  |
           +-------------+
                  |
                GELU
                  |
           +-------------+
           |  Linear 12  |
           +-------------+
                  |
                  v
            dT1 ... dT12
```

Residual prediction:

\[
\hat T_{t+1}=T_t+\Delta\hat T
\]

is preferred over predicting absolute temperatures.

---

# 11. Physics-Informed Loss

```text
                   Prediction
                       |
             +---------+---------+
             |                   |
             v                   v
        Data Loss          Physics Residual
             |                   |
             +---------+---------+
                       |
                       v
                    Total Loss
```

\[
L=
L_{data}
+
\lambda_{physics}L_{physics}
\]

This creates a hybrid:

```text
DATA
 +
PHYSICS
 =
THERMOTWIN
```

---

# 12. Forecast Engine

The controller should not directly call the raw PyTorch model.

Create an abstraction:

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

This permits replacement of:

```text
MLP
```

with:

```text
GNN
Neural ODE
Physics model
Ensemble
```

without rewriting the controller.

---

# 13. Controller Architecture

Define common interface:

```python
class Controller:
    def act(self, state) -> CoolingAction:
        ...
```

Implement:

```text
ThresholdController
PIDController
PredictiveController
```

This allows identical experiments using different controllers.

---

# 14. Predictive Controller

```text
                       Current State
                             |
                             v
                  Generate Candidate Flows
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
       Flow 1             Flow 2             Flow N
          |                  |                  |
          v                  v                  v
      Twin Rollout       Twin Rollout       Twin Rollout
          |                  |                  |
          v                  v                  v
        Cost 1             Cost 2             Cost N
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                          argmin
                             |
                             v
                     Selected Flow
```

Objective:

\[
J =
w_TT_{max}
+
w_\Delta(T_{max}-T_{min})
+
w_PP_{pump}
+
w_SS
\]

Safety penalty:

\[
S=
\max(0,T_{max}-T_{limit})^2
\]

Set \(w_S\) sufficiently high that unsafe solutions are strongly discouraged.

---

# 15. Pump Model

Use a simple approximate relationship:

\[
P_{pump}
=
k_p\dot m^3
\]

Then:

\[
E_{pump}
=
\int P_{pump}dt
\]

The exact relationship is an engineering approximation for MVP and must remain configurable.

---

# 16. Fault Architecture

Fault injection belongs to the physical simulator.

```text
FaultInjector
      |
      v
BatteryEnvironment
      |
      v
Measurements
      |
      v
ThermoTwin
      |
      v
AnomalyDetector
```

Do NOT tell the digital twin directly which cell is faulty during inference.

Otherwise fault detection becomes meaningless.

Example:

\[
R_7\rightarrow1.6R_7
\]

causes:

\[
Q_7=I^2(1.6R_7)
\]

while ThermoTwin initially expects nominal behavior.

---

# 17. Anomaly Detection

For each cell:

\[
e_i(t)
=
T_{measured,i}(t)
-
T_{predicted,i}(t)
\]

Maintain rolling statistics.

Possible anomaly score:

\[
A_i
=
\frac{|e_i-\mu_i|}{\sigma_i+\epsilon}
\]

Trigger only after persistent abnormal values to avoid noisy alerts.

Output:

```python
@dataclass
class ThermalAnomaly:
    cell_id: int
    score: float
    detected_at: float
    predicted_temperature: float
    severity: str
```

---

# 18. Data Flow

```text
              SCENARIO CONFIG
                    |
                    v
             Current Profile
                    |
                    v
        +-----------------------+
        | Battery Environment   |
        +-----------------------+
                    |
             BatteryState
                    |
          +---------+----------+
          |                    |
          v                    v
      Logger             Controller
                               |
                     +---------+---------+
                     |                   |
                     v                   v
                ThermoTwin           Baseline
                     |
                     v
                 Forecast
                     |
                     v
                 Optimizer
                     |
                     v
                CoolingAction
                     |
                     +------------+
                                  |
                                  v
                         Battery Environment
```

---

# 19. Proposed Repository Structure

```text
thermotwin/
│
├── AGENTS.md
├── PLAN.md
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── pack.yaml
│   ├── simulation.yaml
│   ├── training.yaml
│   └── controller.yaml
│
├── src/
│   └── thermotwin/
│       │
│       ├── physics/
│       │   ├── cell.py
│       │   ├── pack.py
│       │   ├── cooling.py
│       │   ├── topology.py
│       │   └── environment.py
│       │
│       ├── scenarios/
│       │   ├── drive_cycles.py
│       │   ├── charging.py
│       │   └── faults.py
│       │
│       ├── data/
│       │   ├── generate.py
│       │   ├── dataset.py
│       │   └── preprocessing.py
│       │
│       ├── models/
│       │   ├── mlp.py
│       │   ├── losses.py
│       │   ├── forecaster.py
│       │   └── train.py
│       │
│       ├── control/
│       │   ├── base.py
│       │   ├── threshold.py
│       │   ├── pid.py
│       │   ├── predictive.py
│       │   └── objective.py
│       │
│       ├── diagnostics/
│       │   ├── anomaly.py
│       │   └── metrics.py
│       │
│       ├── experiments/
│       │   ├── runner.py
│       │   └── benchmark.py
│       │
│       └── visualization/
│           ├── heatmap.py
│           ├── plots.py
│           └── dashboard.py
│
├── scripts/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── run_baseline.py
│   ├── run_thermotwin.py
│   └── benchmark.py
│
├── tests/
│   ├── test_cell.py
│   ├── test_pack.py
│   ├── test_cooling.py
│   ├── test_environment.py
│   ├── test_forecaster.py
│   └── test_controllers.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
│
└── results/
```

---

# 20. Dependency Direction

Maintain:

```text
visualization
      |
experiments
      |
controllers ----> models
      |             |
      +-------> physics
                    |
                  config
```

The physics package should have no dependency on:

- PyTorch
- controllers
- Streamlit
- visualization

The ML package may consume physics functions for physics-informed losses.

The UI must never contain physical calculations.

---

# 21. Configuration

Physical parameters should live in configuration.

Example:

```yaml
pack:
  rows: 3
  columns: 4
  cells: 12

cell:
  mass_kg: ...
  heat_capacity_j_kgk: ...
  resistance_ohm: ...
  capacity_ah: ...

cooling:
  min_flow: ...
  max_flow: ...
  inlet_temperature_c: ...

safety:
  warning_temperature_c: ...
  maximum_temperature_c: ...
```

Values should be sourced/documented before final presentation.

Do not invent precise cell specifications and present them as manufacturer data.

---

# 22. Experiment Architecture

Experiments must be reproducible:

```python
result = run_experiment(
    scenario=scenario,
    controller=controller,
    seed=42,
)
```

Return:

```python
ExperimentResult(
    timeseries=...,
    peak_temperature=...,
    max_temperature_spread=...,
    pump_energy=...,
    thermal_violations=...,
)
```

This same interface should power:

- tests
- benchmark scripts
- plots
- dashboard
- presentation metrics

---

# 23. Final Demo Architecture

```text
                  THERMOTWIN DASHBOARD
                           |
            +--------------+--------------+
            |                             |
            v                             v
     Current Heatmap              Future Heatmap
            |                             |
            +--------------+--------------+
                           |
                           v
                    Thermal Forecast
                           |
                           v
              +-------------------------+
              | Predictive Optimization |
              +-------------------------+
                           |
                           v
                  Recommended Flow
                           |
                           v
              +-------------------------+
              |  Battery Pack Simulator |
              +-------------------------+
                           |
                           v
                    Sensor State
                           |
             +-------------+-------------+
             |                           |
             v                           v
       ThermoTwin                 Anomaly Detector
             |                           |
             v                           v
       Future State                 Fault Warning
```

---

# 24. Future GNN Architecture

The existing pack topology deliberately maps naturally onto a graph.

Node:

```text
battery cell
```

Node features:

```text
temperature
SOC
resistance
heat generation
cooling exposure
```

Edges:

```text
thermal conduction
```

A future model could perform:

\[
H^{(l+1)}
=
GNN(H^{(l)},E)
\]

and predict cell-wise thermal evolution.

This should replace the MLP only after the complete MLP-based architecture works.

---

# 25. Core Separation of Concerns

The most important architectural boundary is:

```text
GROUND TRUTH ≠ DIGITAL TWIN
```

The simulator generates what physically happens.

ThermoTwin predicts what will happen.

The controller makes decisions using ThermoTwin.

The simulator determines whether those decisions were actually good.

Therefore:

\[
\boxed{
Simulator
\rightarrow
Measurements
\rightarrow
ThermoTwin
\rightarrow
Controller
\rightarrow
Simulator
}
\]

That closed loop is the core of the project.