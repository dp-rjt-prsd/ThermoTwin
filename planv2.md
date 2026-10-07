# ThermoTwin V2 Judge-Focused Website Plan

## 1. Status and Authority

This document is the authoritative roadmap for the ThermoTwin V2 website.
The completed Phase 0-12 implementation history remains in `tasks/plan.md` and
`tasks/todo.md`. Cross-session execution evidence for this roadmap is recorded
in `handoff.md`.

Phase status:

- [x] Phase A — Superdesign and shared contracts *(implemented; type-check gate pending)*
- [x] Phase B — Simulation services and isolated UI slices
- [x] Phase C — Central dashboard integration
- [ ] Phase D — Presentation hardening and end-to-end validation *(browser/release gate pending)*

A phase may be checked only after Sol reviews the actual diff and its phase
verification gate passes. Worker completion reports alone are not sufficient.

## 2. Product Objective

Transform the current artifact-backed Streamlit dashboard into a visually
polished thermal-control workspace for a mechanical-engineering judging panel.
The default experience must explain ThermoTwin through physical heat flow,
pack-level visuals, controller behavior, and measured experiment evidence.

The presentation must preserve the central project claim:

> A physics-informed thermal digital twin forecasts future cell-level
> temperatures and enables proactive coolant-flow decisions that balance
> thermal safety, temperature uniformity, and cooling energy.

The website must not become a generic ML dashboard. Machine-learning details
are secondary to the physical system, equations, and observable behavior.

## 3. Judge-Focused Design Principles

1. Lead with the 12-cell battery pack, not model architecture.
2. Explain cause and effect visually: current creates heat, heat moves between
   cells, coolant removes heat, and controllers trade safety against pump work.
3. Pair every major visualization with the applicable formula and plain-language
   mechanical interpretation.
4. Show Celsius prominently while retaining Kelvin and SI units in details and
   hover data.
5. Use saved or newly simulated evidence only. Never hard-code performance
   improvements or claim that one controller or model wins when the selected
   artifacts do not support that conclusion.
6. Keep injected-fault truth separate from information available to the
   controller and operator during inference.
7. Make the primary judge journey understandable without requiring knowledge of
   neural networks, training losses, or tensor operations.

## 4. Information Architecture

### 4.1 Guided Demo — Default View

The opening view presents a controlled visual story:

1. Electrical load produces irreversible cell heat.
2. Heat spreads through the 3×4 pack topology.
3. The cold plate removes heat using controllable coolant flow.
4. A reactive controller responds to measured temperature.
5. ThermoTwin forecasts future temperatures and evaluates cooling actions.
6. An increased-resistance fault creates localized abnormal heating.
7. Persistent forecast residuals identify the developing anomaly.

Each step includes:

- a synchronized 3×4 measured or predicted heatmap;
- one concise mechanical explanation;
- the active equation with current values substituted when available;
- units, signs, and physical interpretation;
- an optional engineering-detail expander;
- playback, pause, reset, speed, and timeline controls.

Guided presets include normal driving, aggressive load, hot ambient, weak
cooling, and a single-cell resistance fault. Presets must be deterministic.

### 4.2 Live Thermal Lab

The live workspace provides:

- scenario and current-profile selection;
- initial SOC and temperature settings;
- ambient and coolant-inlet temperatures;
- controller and model selection;
- legal coolant-flow controls;
- optional resistance-fault cell, start time, and multiplier;
- measured and forecast 3×4 heatmaps;
- cell, maximum, mean, and safety-limit temperature histories;
- coolant command, pump power, and cumulative pump energy;
- 30, 60, and 120-second prediction horizons;
- reset and reproducible export.

The simulation service owns the state. Streamlit owns only view state and user
interaction state.

### 4.3 Cooling Comparison

Compare synchronized runs for:

- threshold reactive control;
- PID control;
- predictive control with the data-driven twin;
- predictive control with the physics-informed twin.

Show peak temperature, mean temperature, maximum spread, pump energy, thermal
violation count, and time above the thermal limit. Metric deltas must identify
the selected reference run and must never imply statistical significance from a
single trajectory.

### 4.4 Fault Injection Lab

Provide controls for fault cell, resistance multiplier, injection time, current
profile, and scenario conditions. Show:

- the physical \(I^2R\) consequence;
- observed-versus-expected temperature residuals;
- rolling anomaly scores and persistence state;
- suspected cell and detection time;
- predicted hotspot temperature;
- threshold-crossing time and detection lead time when defined;
- evaluation-only injected-fault truth in a clearly separated panel.

Use the terms “abnormal thermal behavior,” “hotspot precursor,” or
“fault-induced thermal anomaly.” Do not claim thermal-runaway detection.

### 4.5 Saved Evidence

Retain the current immutable artifact-viewing workflow. The view must expose:

- selected run and scenario;
- controller and model variant;
- configuration and seed;
- checkpoint and normalization provenance;
- measured metrics and forecast horizons;
- accessible saved figures;
- CSV and JSON exports;
- artifact validation errors with actionable guidance.

## 5. Visual Design System

Use a dark engineering-control-room style:

- graphite/navy page background;
- slightly elevated instrument panels with restrained borders;
- high-contrast off-white primary text and muted blue-gray secondary text;
- cyan for coolant paths and commanded flow;
- amber for warnings and red-orange for safety violations;
- a perceptually uniform cold-to-hot scale with numeric temperature labels;
- solid surfaces, restrained corner radii, minimal shadows, and no decorative
  purple gradients;
- Inter or Segoe UI for interface text and monospaced numerals for measurements;
- consistent spacing and typography tokens rather than arbitrary local styles.

Alerts must include icons, labels, or patterns and must not rely on color alone.
The initial design target is 1366×768 and 1440×900 judge displays, followed by
1024, 768, and 320-pixel responsive layouts.

## 6. Physics and Formula Experience

A global **Physics & Formulas** drawer is available from every view. Contextual
formula controls open the relevant entry directly.

Every formula entry contains:

- rendered LaTeX;
- a plain-language mechanical explanation;
- symbol definitions and SI units;
- sign convention;
- live substituted values when the active snapshot contains the required terms;
- a small energy-flow or variable relationship visual;
- an implementation-provenance reference;
- an explicit approximation note where applicable.

### 6.1 Formula Catalog

| Context | Formula |
|---|---|
| Cell energy balance | \(m_iC_{p,i}\frac{dT_i}{dt}=Q_{gen,i}-Q_{cool,i}+Q_{neighbor,i}+Q_{ambient,i}\) |
| Joule heating | \(Q_{gen,i}=I^2R_i\) |
| Neighbor conduction | \(Q_{neighbor,i}=\sum_{j\in N(i)}\frac{T_j-T_i}{R_{th,ij}}\) |
| Coolant coefficient | \(h(\dot m)=h_0+g\dot m^n\) |
| Coolant heat removal | \(Q_{cool,i}=h(\dot m)A_i(T_i-T_c)\) |
| Ambient exchange | \(Q_{ambient,i}=\frac{T_{ambient}-T_i}{R_{ambient}}\) |
| Euler temperature step | \(T_{i,t+1}=T_{i,t}+\Delta t\,\dot T_i\) |
| Coulomb-counted SOC | \(SOC_{t+1}=SOC_t-\frac{I\Delta t}{3600C_{Ah}}\) |
| Thermal forecast | \(\hat T_{t+1}=T_t+\Delta\hat T\) |
| Physics residual | \(r_i=m_iC_{p,i}\frac{\Delta\hat T_i}{\Delta t}-Q_{net,i}\) |
| Training objective | \(L=L_{data}+\lambda_{physics}L_{physics}\) |
| Pump model | \(P_{pump}=k_p\dot m^3,\quad E_{pump}=\sum_t P_{pump,t}\Delta t\) |
| Predictive objective | \(J=w_TT_{max}+w_\Delta\Delta T+w_PE_{pump}+w_SP_{safety}\) |
| Temperature spread | \(\Delta T=T_{max}-T_{min}\) |
| Mean absolute error | \(MAE=\frac{1}{N}\sum|\hat T-T|\) |
| Root mean squared error | \(RMSE=\sqrt{\frac{1}{N}\sum(\hat T-T)^2}\) |
| Anomaly residual | \(e_i=T_{observed,i}-T_{predicted,i}\) |
| Anomaly score | \(A_i=\frac{|e_i-\mu_i|}{\sigma_i+\epsilon}\) |
| Detection lead time | \(t_{lead}=t_{threshold}-t_{detection}\) |

The threshold controller displays its temperature-to-flow piecewise rule. The
PID view displays the implemented clipped proportional-integral-derivative
equation with its error definition. Formula presentation must describe the
cubic pump expression as an MVP approximation, not a manufacturer pump curve.

## 7. Architecture and Shared Interfaces

### 7.1 Dependency Direction

```text
Streamlit views
      |
presentation/view models
      |
simulation service ---- saved artifact loaders
      |
control --> models --> shared physics
      |
diagnostics and experiment metrics
```

The UI must not reimplement physical equations, controller objectives, anomaly
scores, or experiment metrics. Formula evaluation adapters call existing shared
functions and return display-ready terms.

### 7.2 Contracts Frozen in Phase A

- `SimulationRequest`: selected scenario, initial conditions, controller/model,
  duration, timestep, seed, and optional fault request.
- `SimulationSnapshot`: current `BatteryState`, forecast data, thermal terms,
  control decision, diagnostics, and provenance.
- `ControllerComparison`: synchronized run identifiers, metrics, reference run,
  and evidence notes.
- `FormulaSpec`: stable ID, title, LaTeX, explanation, symbols, units,
  approximation notes, and implementation provenance.
- `FormulaEvaluation`: formula ID, snapshot time, substituted terms, result, and
  validity or unavailability reason.
- `PresentationStep`: title, explanation, active visual, formula IDs, and preset
  transition.

Interfaces remain immutable where practical. The simulator, controller,
forecaster, and diagnostic packages remain independent of Streamlit.

## 8. Implementation Phases

### Phase A — Superdesign and Shared Contracts

Tasks:

1. Authenticate the Superdesign CLI and initialize the required repository
   design context.
2. Reproduce the current dashboard as a baseline design.
3. Define the dark control-room design system.
4. Create two controlled branches: cinematic guided demo and dense instrument
   panel. Keep fonts, colors, spacing, and component styles fixed.
5. Select one design direction and generate the remaining page flow from it.
6. Freeze the shared contracts and formula registry schema.

Acceptance criteria:

- [ ] Superdesign project, draft, and canvas references are recorded.
- [ ] The approved design covers all five website views and responsive states.
- [ ] Shared DTOs and formula metadata are documented and type checked.
- [ ] No downstream UI task depends on an unstable contract.

Verification gate:

- Review design fidelity and judge comprehension on the canvas.
- Run focused contract tests, Ruff, and strict mypy.
- Sol reviews the diff before checking Phase A complete.

### Phase B — Simulation Services and Isolated UI Slices

Tasks may run in parallel only after Phase A and only with non-overlapping file
whitelists:

1. Deterministic simulation session service: start, step, pause, reset, preset,
   and export.
2. Formula registry and live term evaluation using shared physical functions.
3. Guided-demo presentation components.
4. Live heatmap and thermal-history components.
5. Controller-comparison figures.
6. Fault-injection and anomaly panels.
7. Saved-evidence and provenance panels.

Acceptance criteria:

- [ ] Repeated configuration and seed produce identical traces.
- [ ] All coolant actions remain within configured limits.
- [ ] Formula values match the authoritative implementations.
- [ ] UI slices accept DTOs and do not import simulator internals directly.
- [ ] Loading, empty, invalid, and unavailable-model states are represented.

Verification gate:

- Run focused service, formula, visualization, and determinism tests.
- Run Ruff and strict mypy on modified scopes.
- Review every worker diff for scope, physics, leakage, and interface safety.
- Run the repository test suite before checking Phase B complete.

### Phase C — Central Dashboard Integration

Tasks:

1. Assemble navigation and shared Streamlit session state.
2. Make Guided Demo the default view.
3. Connect controls, services, formulas, artifacts, and synchronized charts.
4. Add deterministic judge presets and model/checkpoint availability handling.
5. Preserve the current saved-results workflow and truth separation.

Acceptance criteria:

- [ ] A user can complete the full guided journey without entering file paths.
- [ ] Live, comparison, fault, and saved-evidence views share consistent state
  without leaking simulator truth into inference.
- [ ] Every major visualization opens its applicable formula entry.
- [ ] Unsupported comparisons display neutral evidence language.
- [ ] Existing saved Phase 12 artifacts still load and validate.

Verification gate:

- Run dashboard integration and artifact-loader tests.
- Launch the dashboard with final predictive and fault artifacts.
- Complete normal, aggressive, hot-ambient, weak-cooling, and fault journeys.
- Run the full quality suite before checking Phase C complete.

### Phase D — Presentation Hardening and End-to-End Validation

Tasks:

1. Cache validated artifacts, checkpoints, and stable plot data.
2. Add polished playback, timeline, fullscreen, reset, and export behavior.
3. Complete responsive, keyboard, contrast, and non-color-state improvements.
4. Perform actual-user, projector, browser-console, and scientific-claim audits.
5. Reconcile the README and launch instructions after behavior is verified.

Acceptance criteria:

- [ ] The judge journey works at 1366×768 and 1440×900 without clipping.
- [ ] The interface remains usable at 1024, 768, and 320-pixel widths.
- [ ] Keyboard navigation reaches all interactive controls.
- [ ] Text contrast meets WCAG AA and charts do not rely only on color.
- [ ] Exports include configuration, seed, controller, model, and scenario.
- [ ] No console error, false performance claim, or unresolved critical defect
  remains.

Verification gate:

- Run Ruff, strict mypy, and the complete pytest suite.
- Run release smoke workflows and launch the dashboard from documented commands.
- Complete the judge journey using release artifacts in a real browser.
- Verify formulas, units, artifact provenance, and scientific wording.
- Sol performs the final diff and release audit before checking Phase D complete.

## 9. End-to-End Acceptance Journey

A first-time mechanical-engineering judge must be able to:

1. Understand the pack layout and energy balance without ML knowledge.
2. Run the guided demonstration without locating artifact directories.
3. Identify the hottest cell and understand the relevant heat terms.
4. Compare reactive and predictive coolant commands on synchronized plots.
5. Inspect 30, 60, and 120-second forecasts.
6. Inject a resistance fault and connect it to increased \(I^2R\) heating.
7. Observe persistent anomaly detection without seeing the hidden fault label in
   the operator view.
8. Open the relevant equation from every major visualization.
9. Trace every reported result to saved configuration and provenance.

## 10. Scientific Integrity and Safety

- The simulator remains the authoritative environment.
- Predictive control uses forecaster outputs, never future simulator truth.
- Fault labels remain evaluation-only during inference.
- Comparisons use common initial states and scenarios.
- Dataset and model evaluation retain trajectory-level split boundaries.
- All displayed measurements and claims come from saved or newly generated
  output.
- The UI reports uncertainty or missing evidence instead of manufacturing a
  successful result.
- The website does not claim physical hardware validation, CFD fidelity, a
  manufacturer pump curve, or thermal-runaway detection.

## 11. Handoff Update Protocol

At the start of a phase:

1. Set its status in `handoff.md` to `in progress`.
2. Record prerequisites and assigned task scopes.

After the phase verification gate passes:

1. Check the phase in this document.
2. Append a dated completion entry to `handoff.md`.
3. Record changed files, frozen interfaces, commands, exact results, design
   references, scientific review, known issues, and the next phase.

If verification fails, leave the phase unchecked and record the failure and
recovery task. Never rewrite an older completion entry silently; append a dated
amendment.

## 12. Assumptions

- Streamlit remains the frontend framework; a React/API rewrite is out of scope.
- The primary release is a reliable local judge demonstration. Public hosting is
  a separate deployment increment.
- Live behavior uses deterministic simulator steps and Streamlit session state.
- Existing physics, controller, model, diagnostic, and experiment code remains
  authoritative.
- The current Phase 12 artifacts are the starting evidence set.
- Model sophistication will not be increased merely for visual presentation.
