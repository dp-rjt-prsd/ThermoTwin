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
| **Four cooling controllers** | *Threshold* and *PID* react to the measured temperature. Two *predictive* controllers try five coolant flow rates on the twin, look 12 seconds ahead, and pick the cheapest option that keeps the pack below a 33 °C comfort target and well clear of the 40 °C limit. |
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

### Step 2 — Create the results the dashboard shows (once, about 10 minutes)

The repository holds the code, not the generated results. One command builds
them all, from the `ThermoTwin` folder:

```powershell
.\scripts\build_all_results.ps1
```

It runs nine steps and prints each one as it goes:

| # | Step | Result |
|---|---|---|
| 1–2 | Simulate 60 normal operating runs (driving and charging) and 12 harsher "stress" runs (hotter, higher current, higher cell resistance). Coolant flow changes every 10 s within each run. | Training and test data |
| 3 | Train both twins on the normal data and test them on both | The two trained twins |
| 4–5 | Check how well each twin predicts 30, 60 and 120 seconds ahead | Forecast accuracy |
| 6 | Run all four controllers through six 10-minute scenarios | Controller comparison |
| 7 | Triple one cell's resistance at 60 s and see whether the detector catches it | Fault-detection case |
| 8 | Collect everything into one summary with charts | Final evaluation |
| 9 | Record one short run of each controller for the dashboard | Dashboard time series |

The same seeds always give the same numbers. On macOS or Linux, run the nine
`python scripts/...` lines listed inside `scripts/build_all_results.ps1` one by one.

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
| Step 2 says an output folder **already exists** | The scripts never overwrite results. Delete (or rename) the old `results/` and `data/processed/` folders and run Step 2 again. |
| Port 8501 is busy | Close the other dashboard, or run `$env:STREAMLIT_SERVER_PORT=8502` before Step 3 and open `http://localhost:8502`. |
| Live Lab says a twin could not be loaded | Step 2 must have finished; the twins live in `results/phase12-final-comparison`. |

---

## Results so far

From the Step 2 run: six simulated 10-minute scenarios (normal driving, high
current, hot ambient, weak cooling, fast charging, and a cell resistance fault).
These are **descriptive results from a simulated study**, not proof of real-world
performance.

**Cooling controllers** — six scenarios:

| Controller | Average peak cell temperature | Hottest moment | Time above 40 °C (all scenarios) | Average pump energy |
|---|---:|---:|---:|---:|
| Reactive threshold | 33.0 °C | 40.3 °C | 272 s | 6.8 kJ |
| Reactive PID | 38.5 °C | 43.9 °C | 368 s | 9.0 kJ |
| Predictive, data-only twin | 32.3 °C | 38.5 °C | **0 s** | 13.1 kJ |
| Predictive, physics-informed twin | 32.5 °C | 39.3 °C | **0 s** | 12.0 kJ |

The predictive controllers are the only ones that **never crossed the 40 °C
limit**. In the high-current scenario the threshold controller spent 272 s above
it and PID 185 s, because they only react once the pack is already hot. The
predictive controllers start cooling early instead. In easy scenarios (normal
driving, hot ambient, weak cooling) all controllers used similar, small amounts of
pump energy. In the hard ones the predictive controllers spent **more** energy
(up to about 50 kJ against 33 kJ) to stay safe, which is the trade-off.

**Digital twins** — prediction error:

| Test data | Horizon | Data-only twin | Physics-informed twin |
|---|---|---:|---:|
| Similar to training | 1 s | 0.031 °C | 0.037 °C |
| Similar to training | 30 s / 120 s | 1.2 / 2.9 °C | 1.4 / 3.5 °C |
| Harsher than training | 1 s | 0.172 °C | **0.166 °C** |
| Harsher than training | 30 s / 120 s | 5.3 / 11.4 °C | **5.0** / 12.5 °C |

Both twins correctly predict that more coolant flow means a cooler pack, and get
the size of that effect within about 15–20 % of the simulator. The
physics-informed twin is about as accurate as the data-only one: slightly better
one step ahead on harsher data, slightly worse on long forecasts. **It is not
clearly better overall.**

**Fault detection** — a cell (C6) whose resistance triples at 60 s was flagged
at 63 s and crossed 40 °C at 102 s: **39 seconds of warning**, with no false
alarms on other cells, and without the detector being told which cell was faulty.

**Known limits**

- Forecasts are good up to about 30 seconds ahead on familiar conditions; on
  harsher conditions 120-second forecasts are off by about 11–13 °C.
- In hard scenarios the predictive controllers keep the pack safe by pumping
  more, so they use more pump energy than the reactive ones there.
- The detector uses a 0.05 °C noise floor (about one sensor resolution); the
  simulation itself has no sensor noise.
- This detects abnormal heating from a resistance fault. It does **not** model or
  detect thermal runaway.
- The pump energy uses a simple cubic approximation (power ∝ flow³), not a
  manufacturer's pump curve.

### What changed after the first submission

The first version had three problems, found and fixed:

1. **The cell resistance (15 mΩ) and cold plate were unrealistic.** Even full
   coolant flow could not hold the pack below 70 °C. Resistance is now 3 mΩ and
   the cold plate reaches about 330 W/m²K at full flow, so cooling actually
   matters.
2. **The twins had learned the cooling effect backwards.** Coolant flow was
   constant within each training run, and only 6 runs were used, so the twins
   could not tell what flow does. Flow now changes every 10 s and 60 runs are
   used.
3. **The predictive controller over-cooled or under-cooled.** It now has a
   33 °C comfort target, so it spends pump energy only when the twin forecasts the
   pack heading above it.

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
3 mΩ, coolant flow 0–0.1 kg/s, 40 °C limit, …) live in `configs/` and are
illustrative, not taken from a specific manufacturer's cell.

---

## For developers

```powershell
python -m pytest                    # run all automated tests
python -m ruff check .              # code style
python -m mypy src tests scripts    # type checks
```

- **Quick benchmark:** `scripts/benchmark.py --quick` runs 30-second scenarios
  for a fast workflow check; Step 2 uses the full 600-second version.
- **Standalone charts:** `results/phase12-final-evaluation/figures/` contains six
  interactive HTML charts that open in any browser without the dashboard.
- **Launching Streamlit directly** (without `run_dashboard.py`): set
  `THERMOTWIN_PREDICTIVE_RESULTS`, `THERMOTWIN_FAULT_RESULTS` and optionally
  `THERMOTWIN_EVALUATION_RESULTS`, then run
  `python -m streamlit run src/thermotwin/visualization/dashboard.py`.
