# ThermoTwin

**A physics-informed digital twin for predictive EV battery cooling.**

ThermoTwin is a computer model of a 12-cell electric-vehicle battery pack sitting
on a liquid cold plate. It shows how a "digital twin" — a fast learned copy of the
pack's thermal behaviour — can predict cell temperatures a few seconds ahead, and
use that prediction to decide how much coolant to pump *before* the pack gets hot,
instead of reacting after it already has.

Everything runs on a laptop. No battery hardware is needed: a physics simulator
plays the role of the real pack.

---

## What is inside

| Part | What it does |
|---|---|
| **Battery pack simulator** | 12 cells in a 3 × 4 grid. Each cell heats up from current (Joule heating, I²R), passes heat to its neighbours, loses heat to the air, and is cooled by a liquid cold plate. This is the "real" pack. |
| **Two digital twins** | Small neural networks that learn to predict the next cell temperatures. One learns from data only; the other is also penalised when it breaks the energy balance (physics-informed). |
| **Four cooling controllers** | *Threshold* and *PID* react to the measured temperature. Two *predictive* controllers try five coolant flow rates on the twin, look 12 seconds ahead, and pick the best trade-off between temperature, uniformity and pump energy. |
| **Fault detector** | Compares measured and twin-predicted temperatures to spot a cell that is heating abnormally (for example, from increased internal resistance). |
| **Dashboard** | An interactive control-room screen for exploring all of the above. |

The cooling loop:

```
pack simulator ──measurements──► digital twin ──predicted temperatures──► controller
       ▲                                                                       │
       └──────────────────────────── coolant flow ◄──────────────────────────┘
```

---

## How to run it

You need **Python 3.11 or newer** and **Git**. The first install can take
several minutes because it downloads PyTorch (the neural-network library).

> Commands are shown for **Windows PowerShell**. On macOS or Linux, use `python3`
> instead of `python` where needed, and activate the environment with
> `source .venv/bin/activate`.

### Step 1 — Download and install (once)

```powershell
git clone https://github.com/dp-rjt-prsd/ThermoTwin.git
cd ThermoTwin
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

This creates a private Python environment in the `.venv` folder and installs
everything ThermoTwin needs. Next time, you only need `cd ThermoTwin` and
`.venv\Scripts\Activate.ps1`.

### Step 2 — Create the results the dashboard shows (once, about 2 minutes)

The repository holds the code, not the generated results. These commands build
them: create training data, train both twins, test them, compare the controllers,
and run a fault case. Run them **in order**, from the `ThermoTwin` folder:

```powershell
python scripts/generate_dataset.py --seed 1729 --trajectory-count 6 --output-directory data/processed/id
python scripts/generate_dataset.py --ood --seed 1730 --trajectory-count 6 --output-directory data/processed/ood
python scripts/compare_models.py --id-dataset-directory data/processed/id --ood-dataset-directory data/processed/ood --output-directory results/phase12-final-comparison
python scripts/evaluate_forecasts.py --comparison-directory results/phase12-final-comparison --dataset-directory data/processed/id --output-directory results/forecast-id
python scripts/evaluate_forecasts.py --comparison-directory results/phase12-final-comparison --dataset-directory data/processed/ood --output-directory results/forecast-ood
python scripts/benchmark.py --quick --seed 1729 --comparison-directory results/phase12-final-comparison --output-directory results/benchmark
python scripts/run_fault_detection.py --smoke-test --comparison-directory results/phase12-final-comparison --model-variant physics_informed --seed 1729 --output-directory results/fault
python scripts/final_evaluation.py --benchmark-directory results/benchmark --comparison-directory results/phase12-final-comparison --id-forecast-directory results/forecast-id --ood-forecast-directory results/forecast-ood --fault-directory results/fault/fault-detection --output-directory results/phase12-final-evaluation
python scripts/run_predictive.py --comparison-directory results/phase12-final-comparison --seed 1729 --output-directory results/predictive
```

What each line does, in plain words:

| # | Step | Result |
|---|---|---|
| 1–2 | Simulate 6 normal operating runs (driving and charging) and 6 harsher "stress" runs (hotter, higher current, higher cell resistance) | Training and test data |
| 3 | Train both twins on the normal data and test them on both | The two trained twins |
| 4–5 | Check how well each twin predicts 30, 60 and 120 seconds ahead | Forecast accuracy |
| 6 | Run all four controllers through six driving scenarios | Controller comparison |
| 7 | Triple one cell's resistance and see whether the detector catches it | Fault-detection case |
| 8 | Collect everything into one summary with charts | Final evaluation |
| 9 | Record one short run of each controller for the dashboard | Dashboard time series |

The same seeds always give the same numbers.

### Step 3 — Open the dashboard

```powershell
python scripts/run_dashboard.py --predictive-directory results/predictive --fault-directory results/fault --evaluation-directory results/phase12-final-evaluation
```

Then open **http://localhost:8501** in your browser. Stop the dashboard with
`Ctrl + C` in the terminal.

The dashboard has four tabs:

- **Guided Demo** — a 7-step story from cell heat to proactive cooling. Use
  **Next ▶** to walk through; each step shows its own chart and equation.
- **Live Thermal Lab** — set the current, temperatures, controller and an optional
  cell fault, press **Run simulation**, then **▶ Play** to watch the pack heat up
  and the coolant respond.
- **Controllers & Twins** — compare the four controllers across six scenarios, and
  see how accurate each twin is.
- **Fault Lab** — what the operator sees when a cell starts overheating, kept
  separate from the "answer key" of which cell was actually faulted.

**Physics & Formulas** (top bar) explains every equation used.

### If something goes wrong

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python 3.11+ and tick "Add to PATH", or try `py` instead of `python`. |
| Activating `.venv` is blocked in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again. |
| A step says the output folder **already exists** | The scripts never overwrite results. Delete that folder in `results/` (or `data/processed/`) and rerun the step. |
| Port 8501 is busy | Close the other dashboard, or run `$env:STREAMLIT_SERVER_PORT=8502` before Step 3 and open `http://localhost:8502`. |
| Live Lab says a twin could not be loaded | Step 2, line 3 must save to exactly `results/phase12-final-comparison`. |

---

## Results so far

From the Step 2 run (six short synthetic scenarios, 30 seconds each). These are
**descriptive results from a small simulated study**, not proof of real-world
performance.

**Cooling controllers** — average over six scenarios:

| Controller | Average peak cell temperature | Average pump energy |
|---|---:|---:|
| Reactive threshold | 33.84 °C | 288.2 J |
| Reactive PID | 33.96 °C | 50.2 J |
| Predictive, data-only twin | 33.89 °C | 141.3 J |
| Predictive, physics-informed twin | 33.96 °C | **33.3 J** |

The physics-informed predictive controller used about **88 % less pump energy**
than the threshold controller, with peak temperatures within 0.12 °C. All four
controllers spent the same 15 s above the 40 °C limit in the high-current scenario.

**Digital twins** — average prediction error one second ahead:

| Test data | Data-only twin | Physics-informed twin |
|---|---:|---:|
| Similar to training | 0.34 °C | 0.36 °C |
| Harsher than training | 3.29 °C | **3.17 °C** |

On the harsher conditions, the physics-informed twin's worst single-cell error was
16.6 % lower (4.98 °C vs 5.97 °C).

**Fault detection** — a cell with triple resistance (C6) was flagged at 15 s and
crossed 40 °C at 26 s: **11 seconds of warning**, without the detector being told
which cell was faulty.

**Known limits**

- Predictions are good a few seconds ahead but drift badly over 60–120 seconds
  (average errors of about 25–30 °C on normal data, far more on the harsher
  data), so long-range forecasts are not yet reliable.
- The study is small: six training runs and 30-second scenarios. It shows the
  method works end to end; it does not establish statistical significance.
- This detects abnormal heating from a resistance fault. It does **not** model or
  detect thermal runaway.
- The pump energy uses a simple cubic approximation (power ∝ flow³), not a
  manufacturer's pump curve.

---

## Project layout

```
configs/      pack, cooling, controller, training and scenario settings (YAML)
src/thermotwin/
  physics/        cell heating, conduction, cooling, pack simulator
  models/         the two digital twins, training and physics-informed loss
  control/        threshold, PID and predictive controllers
  diagnostics/    fault detector and performance metrics
  experiments/    benchmark and evaluation runs
  visualization/  the dashboard
scripts/      the commands used in Steps 2 and 3
tests/        automated checks
```

All physical values (cell mass 0.8 kg, heat capacity 900 J/kg·K, resistance
15 mΩ, coolant flow 0–0.1 kg/s, 40 °C limit, …) live in `configs/` and are
illustrative, not taken from a specific manufacturer's cell.

---

## For developers

```powershell
python -m pytest                    # run all automated tests
python -m ruff check .              # code style
python -m mypy src tests scripts    # type checks
```

- **Longer benchmark:** run `scripts/benchmark.py` without `--quick` (600 s per
  scenario). It needs fully trained twins from Step 2, line 3.
- **Standalone charts:** `results/phase12-final-evaluation/figures/` contains six
  interactive HTML charts that open in any browser without the dashboard.
- **Launching Streamlit directly** (without `run_dashboard.py`): set
  `THERMOTWIN_PREDICTIVE_RESULTS`, `THERMOTWIN_FAULT_RESULTS` and optionally
  `THERMOTWIN_EVALUATION_RESULTS`, then run
  `python -m streamlit run src/thermotwin/visualization/dashboard.py`.
