"""Judge-facing presentation metadata and visual styling for the V2 dashboard."""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from plotly.graph_objects import Figure  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class FormulaSpec:
    """One contextual equation with mechanical interpretation and provenance."""

    identifier: str
    title: str
    latex: str
    explanation: str
    symbols: tuple[str, ...]
    provenance: str
    approximation: str | None = None


@dataclass(frozen=True, slots=True)
class StoryStep:
    """One judge-guided explanation step."""

    number: int
    title: str
    summary: str
    formula_ids: tuple[str, ...]
    accent: str


FORMULAS: Mapping[str, FormulaSpec] = MappingProxyType(
    {
        "energy_balance": FormulaSpec(
            "energy_balance",
            "Cell energy balance",
            r"m_i C_{p,i}\frac{dT_i}{dt}=Q_{gen,i}-Q_{cool,i}+Q_{neighbor,i}+Q_{ambient,i}",
            "Stored thermal energy changes because cells generate heat, exchange "
            "heat with neighbors and ambient air, and reject heat to the cold plate.",
            (
                "mᵢ — cell mass (kg)",
                "Cₚ,ᵢ — specific heat capacity (J/kg·K)",
                "Tᵢ — cell temperature (K)",
                "Q — signed heat-transfer rate (W)",
            ),
            "thermotwin.physics.environment.BatteryEnvironment.step",
        ),
        "joule": FormulaSpec(
            "joule",
            "Irreversible Joule heating",
            r"Q_{gen,i}=I^2R_i",
            "Electrical current produces heat in proportion to resistance and to "
            "the square of current. A resistance fault therefore creates a local hotspot.",
            ("I — signed pack current (A)", "Rᵢ — cell resistance (Ω)"),
            "thermotwin.physics.cell.joule_heating_w",
        ),
        "conduction": FormulaSpec(
            "conduction",
            "Neighbor conduction",
            r"Q_{neighbor,i}=\sum_{j\in N(i)}\frac{T_j-T_i}{R_{th,ij}}",
            "Heat flows from hotter neighboring cells toward colder cells through "
            "the reusable 3×4 pack topology.",
            (
                "N(i) — cells adjacent to cell i",
                "Rₜₕ,ᵢⱼ — pair thermal resistance (K/W)",
            ),
            "thermotwin.physics.pack.neighbor_conduction_w",
        ),
        "cooling_coefficient": FormulaSpec(
            "cooling_coefficient",
            "Flow-dependent heat-transfer coefficient",
            r"h(\dot m)=h_0+g\dot m^n",
            "Increasing coolant mass flow increases the effective cold-plate heat "
            "transfer coefficient within configured bounds.",
            (
                "ṁ — coolant mass flow (kg/s)",
                "h — heat-transfer coefficient (W/m²·K)",
            ),
            "thermotwin.physics.cooling.heat_transfer_coefficient_w_m2_k",
        ),
        "cooling": FormulaSpec(
            "cooling",
            "Cold-plate heat removal",
            r"Q_{cool,i}=h(\dot m)A_i(T_i-T_c)",
            "The liquid cold plate removes more heat when a cell is hotter than "
            "the coolant and when coolant flow raises the transfer coefficient.",
            (
                "Aᵢ — cooled cell surface area (m²)",
                "T꜀ — coolant inlet temperature (K)",
            ),
            "thermotwin.physics.cooling.cooling_heat_removal_w",
        ),
        "ambient": FormulaSpec(
            "ambient",
            "Ambient heat exchange",
            r"Q_{ambient,i}=\frac{T_{ambient}-T_i}{R_{ambient}}",
            "Ambient air heats a cooler cell or removes heat from a hotter cell; "
            "the sign follows the temperature difference.",
            ("Rambient — ambient thermal resistance (K/W)",),
            "thermotwin.physics.cooling.ambient_heat_exchange_w",
        ),
        "integration": FormulaSpec(
            "integration",
            "Temperature integration",
            r"T_{i,t+1}=T_{i,t}+\Delta t\,\dot T_i",
            "The simulator advances every cell temperature using an explicit Euler step.",
            ("Δt — simulation time step (s)", "Ṫᵢ — temperature rate (K/s)"),
            "thermotwin.physics.cell.integrate_temperature_euler",
        ),
        "soc": FormulaSpec(
            "soc",
            "Coulomb-counted state of charge",
            r"SOC_{t+1}=SOC_t-\frac{I\Delta t}{3600C_{Ah}}",
            "Positive current discharges the pack; negative current charges it.",
            ("Cₐₕ — cell capacity (Ah)",),
            "thermotwin.physics.electrical.soc_after_coulomb_counting",
        ),
        "forecast": FormulaSpec(
            "forecast",
            "Thermal forecast update",
            r"\hat T_{t+1}=T_t+\Delta\hat T",
            "The compact surrogate predicts a temperature change, which is added "
            "to the measured state and rolled forward autoregressively.",
            ("T̂ — predicted cell temperature (K)",),
            "thermotwin.models.forecaster.MLPThermalForecaster",
        ),
        "physics_residual": FormulaSpec(
            "physics_residual",
            "Physics residual",
            r"r_i=m_iC_{p,i}\frac{\Delta\hat T_i}{\Delta t}-Q_{net,i}",
            "A prediction is physically inconsistent when its implied energy "
            "change disagrees with the modeled net heat rate.",
            ("rᵢ — energy-balance residual (W)",),
            "thermotwin.models.losses.physics_residual_w",
        ),
        "training_loss": FormulaSpec(
            "training_loss",
            "Physics-informed training objective",
            r"L=L_{data}+\lambda_{physics}L_{physics}",
            "The physics-informed twin balances agreement with training data "
            "against agreement with the thermal energy balance.",
            ("λphysics — configured physics-loss weight",),
            "thermotwin.models.losses.combined_thermal_loss",
        ),
        "pump": FormulaSpec(
            "pump",
            "Cooling effort",
            r"P_{pump}=k_p\dot m^3,\qquad E_{pump}=\sum_tP_{pump,t}\Delta t",
            "Higher coolant flow can reduce temperature but incurs rapidly rising "
            "pump power, creating the controller tradeoff.",
            ("Ppump — approximate pump power (W)", "Epump — pump energy (J)"),
            "thermotwin.control.objective.pump_power_w",
            "MVP cubic approximation; not a manufacturer-specific pump curve.",
        ),
        "predictive_cost": FormulaSpec(
            "predictive_cost",
            "Predictive cooling objective",
            r"J=w_TT_{max}+w_\Delta\Delta T+w_PE_{pump}+w_SP_{safety}",
            "Candidate coolant actions are ranked by peak temperature, pack "
            "uniformity, cooling energy, and a large safety penalty.",
            ("w — configured nonnegative objective weights",),
            "thermotwin.control.objective.evaluate_predictive_candidate",
        ),
        "threshold": FormulaSpec(
            "threshold",
            "Reactive threshold controller",
            r"\dot m=\begin{cases}\dot m_{min},&T_{max}<30^\circ C\\\dot m_{low},&30\le T_{max}<35^\circ C\\\dot m_{medium},&35\le T_{max}<40^\circ C\\\dot m_{max},&T_{max}\ge40^\circ C\end{cases}",
            "The conventional baseline reacts to the maximum temperature that "
            "has already been measured; it does not forecast the future pack state.",
            ("ṁ — commanded coolant mass flow (kg/s)",),
            "thermotwin.control.threshold.ThresholdController.act",
        ),
        "pid": FormulaSpec(
            "pid",
            "Reactive PID controller",
            r"\dot m=\operatorname{clip}\!\left(K_pe+K_i\int e\,dt+K_d\frac{de}{dt},\dot m_{min},\dot m_{max}\right)",
            "PID continuously adjusts flow from measured temperature error while "
            "respecting the configured coolant-flow limits.",
            ("e — measured maximum-temperature error (K)",),
            "thermotwin.control.pid.PIDController.act",
        ),
        "spread": FormulaSpec(
            "spread",
            "Pack temperature spread",
            r"\Delta T=T_{max}-T_{min}",
            "A smaller spread means the pack is more thermally uniform.",
            ("ΔT — instantaneous or maximum observed spread (K or °C difference)",),
            "thermotwin.diagnostics.metrics.calculate_experiment_metrics",
        ),
        "mae": FormulaSpec(
            "mae",
            "Mean absolute forecast error",
            r"MAE=\frac{1}{N}\sum_{n=1}^{N}|\hat T_n-T_n|",
            "MAE reports the typical absolute temperature-prediction error.",
            ("N — evaluated cell-time predictions",),
            "thermotwin.models.evaluation.compute_regression_metrics",
        ),
        "rmse": FormulaSpec(
            "rmse",
            "Root mean squared forecast error",
            r"RMSE=\sqrt{\frac{1}{N}\sum_{n=1}^{N}(\hat T_n-T_n)^2}",
            "RMSE gives larger forecast errors more influence than MAE.",
            ("T̂ₙ − Tₙ — prediction error (K)",),
            "thermotwin.models.evaluation.compute_regression_metrics",
        ),
        "anomaly_residual": FormulaSpec(
            "anomaly_residual",
            "Thermal forecast residual",
            r"e_i=T_{observed,i}-T_{predicted,i}",
            "A fault can make measured temperature depart persistently from the "
            "digital twin's expected temperature.",
            ("eᵢ — observed-minus-predicted temperature (K)",),
            "thermotwin.diagnostics.anomaly.calculate_temperature_residual",
        ),
        "anomaly_score": FormulaSpec(
            "anomaly_score",
            "Rolling anomaly score",
            r"A_i=\frac{|e_i-\mu_i|}{\sigma_i+\epsilon}",
            "The detector normalizes each residual against its recent baseline and "
            "requires persistent threshold exceedance before emitting an event.",
            ("μᵢ, σᵢ — rolling residual mean and deviation",),
            "thermotwin.diagnostics.anomaly.ThermalAnomalyDetector.update",
        ),
        "lead_time": FormulaSpec(
            "lead_time",
            "Fault-detection lead time",
            r"t_{lead}=t_{threshold}-t_{detection}",
            "Positive lead time means the anomaly was detected before the measured "
            "temperature crossed the configured thermal limit.",
            ("t — saved experiment time (s)",),
            "thermotwin.diagnostics.fault_metrics.calculate_detection_lead_time",
        ),
    }
)


STORY_STEPS: tuple[StoryStep, ...] = (
    StoryStep(
        1,
        "Energy enters",
        "Current creates cell-level I²R heat.",
        ("joule",),
        "warning",
    ),
    StoryStep(
        2,
        "Heat moves",
        "Neighbor conduction redistributes the hotspot.",
        ("conduction", "energy_balance"),
        "danger",
    ),
    StoryStep(
        3,
        "Cooling responds",
        "The cold plate removes heat through controlled flow.",
        ("cooling_coefficient", "cooling"),
        "coolant",
    ),
    StoryStep(
        4,
        "Twin forecasts",
        "Both twins project the 12-cell thermal state forward.",
        ("forecast", "physics_residual"),
        "forecast",
    ),
    StoryStep(
        5,
        "Controller decides",
        "Candidate actions balance safety, uniformity, and energy.",
        ("threshold", "pid", "predictive_cost", "pump", "spread"),
        "coolant",
    ),
    StoryStep(
        6,
        "Fault emerges",
        "Extra resistance creates abnormal localized heat.",
        ("joule", "anomaly_residual", "anomaly_score"),
        "danger",
    ),
    StoryStep(
        7,
        "Evidence closes the loop",
        "Saved results quantify forecast, thermal, energy, and fault behavior.",
        ("mae", "rmse", "lead_time"),
        "forecast",
    ),
)


GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
  --tt-bg: #071018;
  --tt-surface: #0D1822;
  --tt-raised: #132431;
  --tt-interactive: #18313E;
  --tt-border: #29404D;
  --tt-text: #F3F7F9;
  --tt-muted: #A6B8C3;
  --tt-faint: #78909C;
  --tt-coolant: #27C2D1;
  --tt-forecast: #62D6A7;
  --tt-warning: #F4B942;
  --tt-danger: #F06449;
  --tt-focus: #8BDFF0;
  --tt-mono: "JetBrains Mono", Consolas, monospace;
}
html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"] { font-family: Inter, "Segoe UI", Arial, sans-serif; }
.stApp { background: var(--tt-bg); color: var(--tt-text); }
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
.stAppDeployButton, [data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 100% !important; padding: 0 1.25rem 2.5rem !important; }
p, label, .stCaption { color: var(--tt-muted); }
code { font-family: var(--tt-mono); }
.stApp [data-testid="stMarkdownContainer"]:has(> div[class^="tt-"]) { margin-bottom: 0; }
.tt-icon { display: inline-block; vertical-align: -2px; margin-right: .4rem; }
.tt-tone-muted { color: var(--tt-faint); } .tt-tone-coolant { color: var(--tt-coolant); }
.tt-tone-forecast { color: var(--tt-forecast); } .tt-tone-warning { color: var(--tt-warning); }
.tt-tone-danger { color: var(--tt-danger); } .tt-tone-primary { color: var(--tt-text); }

/* ---------- top navigation ---------- */
.st-key-tt_header { position: sticky; top: 0; z-index: 990; background: var(--tt-surface);
  border-bottom: 1px solid var(--tt-border); margin: 0 -1.25rem 1.1rem; padding: .45rem 1.25rem; width: calc(100% + 2.5rem) !important; }
.tt-brand { display: flex; align-items: center; gap: .1rem; font-weight: 700; font-size: 1.1rem;
  letter-spacing: -0.01em; color: var(--tt-text); white-space: nowrap; }
.st-key-tt_nav div[role="radiogroup"] { gap: .25rem; flex-wrap: nowrap; }
.st-key-tt_nav label[data-testid="stRadioOption"] { background: transparent; border: 0; border-radius: 6px;
  padding: .38rem .8rem; margin: 0; transition: background .15s, color .15s; }
.st-key-tt_nav label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-tt_nav label[data-testid="stRadioOption"] p { color: var(--tt-muted); font-weight: 500; font-size: .9rem; white-space: nowrap; }
.st-key-tt_nav label[data-testid="stRadioOption"]:hover { background: var(--tt-raised); }
.st-key-tt_nav label[data-testid="stRadioOption"]:hover p { color: var(--tt-text); }
.st-key-tt_nav label[data-selected="true"] { background: var(--tt-interactive) !important; }
.st-key-tt_nav label[data-selected="true"] p { color: var(--tt-focus) !important; }
.st-key-tt_header [data-testid="stPopover"] button { background: transparent; border: 1px solid transparent; color: var(--tt-muted); }
.st-key-tt_header [data-testid="stPopover"] button:hover { background: var(--tt-raised); color: var(--tt-text); border-color: var(--tt-border); }
[data-testid="stPopoverBody"] { background: var(--tt-surface); border: 1px solid var(--tt-border); }

/* ---------- rails ---------- */
.st-key-tt_rail_left, .st-key-tt_rail_right { background: var(--tt-surface); border: 1px solid var(--tt-border);
  border-radius: 10px; padding: 1rem; min-height: calc(100vh - 6.5rem); gap: .6rem; }
.tt-rail-eyebrow { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--tt-faint); }
.tt-rail-heading { font-size: 1rem; font-weight: 600; color: var(--tt-text); margin: .15rem 0 .6rem;
  padding-bottom: .8rem; border-bottom: 1px solid var(--tt-border); }
.tt-rail-title { display: flex; align-items: center; gap: .5rem; font-size: .72rem; font-weight: 600;
  letter-spacing: .08em; text-transform: uppercase; color: var(--tt-text); margin: .1rem 0 .3rem; }
.tt-rail-sub { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--tt-faint); margin: .8rem 0 .1rem; }
.tt-pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--tt-focus); animation: tt-pulse 1.6s ease-in-out infinite; }
@keyframes tt-pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

/* story steps */
.st-key-tt_rail_left { counter-reset: tt-step; }
.tt-progress { height: 4px; background: var(--tt-border); border-radius: 2px; margin: .5rem 0 .3rem; overflow: hidden; }
.tt-progress > span { display: block; height: 100%; background: var(--tt-coolant); transition: width .3s ease; }
[class*="st-key-tt_step_"] { counter-increment: tt-step; }
[class*="st-key-tt_step_"] button { width: 100%; justify-content: flex-start; text-align: left; background: transparent !important;
  border: 1px solid transparent !important; box-shadow: none !important; border-radius: 8px; padding: .65rem .7rem .3rem 2.6rem; position: relative; min-height: 0; }
[class*="st-key-tt_step_"] button p { color: var(--tt-muted); font-weight: 500; font-size: .92rem; text-align: left; }
[class*="st-key-tt_step_"] button > div { justify-content: flex-start; width: 100%; }
[class*="st-key-tt_step_"] [data-testid="stMarkdownContainer"] { text-align: left; }
[class*="st-key-tt_step_"] button::before { content: counter(tt-step); position: absolute; left: .7rem; top: .62rem; width: 1.4rem; height: 1.4rem;
  border-radius: 50%; border: 1px solid var(--tt-border); color: var(--tt-faint); font-size: .72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }
[class*="st-key-tt_step_"] button:hover { background: var(--tt-raised); }
[class*="st-key-tt_step_"] button:hover p { color: var(--tt-text); }
[class*="st-key-tt_step_"] .tt-step-desc { font-size: .76rem; color: var(--tt-faint); margin: -.15rem 0 0 2.6rem; padding: 0 .4rem .55rem 0; line-height: 1.35; }
[class*="st-key-tt_step_on_"] { background: var(--tt-interactive); border: 1px solid var(--tt-border); border-radius: 8px; }
[class*="st-key-tt_step_on_"] button p { color: var(--tt-text); }
[class*="st-key-tt_step_on_"] button::before { background: rgba(139,223,240,.2); border-color: transparent; color: var(--tt-focus); }
[class*="st-key-tt_step_on_"] .tt-step-desc { color: var(--tt-muted); }
[class*="st-key-tt_step_"] { gap: 0 !important; }

/* ---------- headings and cards ---------- */
.tt-heading h1 { font-size: 1.75rem !important; line-height: 1.2; font-weight: 700; letter-spacing: -0.02em; color: var(--tt-text); padding: 0 0 .35rem !important; margin: 0; }
.tt-heading p { color: var(--tt-muted); max-width: 64rem; line-height: 1.55; margin: 0 0 .4rem; }
.tt-disclaimer { color: var(--tt-faint); font-style: italic; }
[class*="st-key-tt_card_"] { background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: 8px;
  overflow: hidden; padding-bottom: .8rem; gap: .55rem; }
[class*="st-key-tt_card_"] > [data-testid="stElementContainer"]:not(:first-child),
[class*="st-key-tt_card_"] > [data-testid="stHorizontalBlock"],
[class*="st-key-tt_card_"] > [data-testid="stLayoutWrapper"] { padding-left: 1rem; padding-right: 1rem; }
.tt-card-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; background: var(--tt-raised);
  border-bottom: 1px solid var(--tt-border); padding: .65rem 1rem; }
.tt-card-label { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; display: flex; align-items: center; }
.tt-card-meta { font-family: var(--tt-mono); font-size: .75rem; }

/* ---------- pack tiles ---------- */
.tt-pack { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; padding: .4rem 0 .2rem; }
.tt-pack-flow { grid-template-columns: 1fr 3.4rem 1fr 3.4rem 1fr 3.4rem 1fr; grid-template-rows: auto 1.7rem auto 1.7rem auto; gap: 0; }
.tt-tile { border: 1px solid; border-radius: 6px; min-height: 5.2rem; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: .1rem; }
.tt-tile-forecast { border-style: dashed; }
.tt-tile-forecast .tt-tile-value { color: var(--tt-forecast); }
.tt-tile-hot { box-shadow: 0 0 16px rgba(240,100,73,.35); outline: 2px solid rgba(243,247,249,.85); outline-offset: 1px; }
.tt-tile-id { font-size: .72rem; color: var(--tt-faint); }
.tt-tile-value { font-family: var(--tt-mono); font-size: 1rem; font-weight: 500; color: var(--tt-text); }
.tt-tile-value small { font-size: .7rem; color: var(--tt-muted); }
.tt-tile-sub { font-family: var(--tt-mono); font-size: .66rem; color: var(--tt-faint); }
.tt-conn { display: flex; align-items: center; justify-content: center; flex-direction: column; font-family: var(--tt-mono);
  font-size: .64rem; color: var(--tt-faint); }
.tt-conn-on b { color: var(--tt-warning); font-size: .95rem; line-height: 1; }
.tt-conn-on span { color: var(--tt-text); }
.tt-legend { display: flex; align-items: center; gap: .6rem; font-family: var(--tt-mono); font-size: .7rem; color: var(--tt-faint); margin-top: .5rem; flex-wrap: wrap; }
.tt-legend-bar { height: 6px; flex: 1; min-width: 6rem; border-radius: 3px; }
.tt-legend-note { width: 100%; font-family: Inter, sans-serif; border-top: 3px solid rgba(39,194,209,.55); padding-top: .3rem; text-align: center; }

/* ---------- rail widgets ---------- */
.tt-settings { background: var(--tt-raised); border: 1px solid var(--tt-border); border-radius: 8px; padding: .8rem .9rem; display: grid; gap: .7rem; }
.tt-setting label { display: block; font-size: .66rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; color: var(--tt-faint); margin-bottom: .15rem; }
.tt-setting div { font-size: .9rem; font-weight: 500; color: var(--tt-text); }
.tt-kpi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
.tt-kpi { background: var(--tt-raised); border: 1px solid var(--tt-border); border-radius: 8px; padding: .65rem .75rem; min-width: 0; }
.tt-kpi-label { font-size: .66rem; text-transform: uppercase; letter-spacing: .08em; color: var(--tt-faint); margin-bottom: .2rem; }
.tt-kpi-value { font-family: var(--tt-mono); font-size: 1.1rem; color: var(--tt-text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tt-kpi-value small { font-size: .7rem; color: var(--tt-faint); }
.tt-flow { background: var(--tt-raised); border: 1px solid var(--tt-border); border-radius: 8px; padding: .7rem .8rem; display: flex; justify-content: space-between; align-items: center; }
.tt-flow-sub { font-family: var(--tt-mono); font-size: .72rem; color: var(--tt-faint); }
.tt-alert { border-radius: 8px; padding: .75rem .8rem; font-size: .88rem; color: var(--tt-text); display: grid; gap: .2rem; }
.tt-alert div:not(.tt-alert-head) { color: var(--tt-muted); font-size: .8rem; }
.tt-alert div:nth-child(2) { color: var(--tt-text); font-size: .9rem; }
.tt-alert b { font-family: var(--tt-mono); color: var(--tt-text); }
.tt-alert-head { display: flex; justify-content: space-between; font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .2rem; }
.tt-alert-danger { background: rgba(240,100,73,.1); border: 1px solid rgba(240,100,73,.3); }
.tt-alert-danger .tt-alert-head { color: var(--tt-danger); }
.tt-alert-ok { background: rgba(98,214,167,.08); border: 1px solid rgba(98,214,167,.3); }
.tt-alert-ok .tt-alert-head { color: var(--tt-forecast); }
.tt-alert-score { font-family: var(--tt-mono); text-transform: none; letter-spacing: 0; }

/* ---------- notices, chips, truth ---------- */
.tt-notice { background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: 8px; padding: .75rem .9rem; color: var(--tt-muted); font-size: .88rem; line-height: 1.5; height: 100%; }
.tt-notice b { color: var(--tt-text); }
.tt-notice-title { font-size: .68rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem; display: flex; align-items: center; }
.tt-notice-coolant .tt-notice-title { color: var(--tt-coolant); } .tt-notice-forecast .tt-notice-title { color: var(--tt-forecast); }
.tt-notice-warning .tt-notice-title { color: var(--tt-warning); } .tt-notice-danger .tt-notice-title { color: var(--tt-danger); }
.tt-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 .4rem; }
.tt-chip { border: 1px solid var(--tt-border); background: var(--tt-surface); border-radius: 6px; padding: .25rem .6rem; color: var(--tt-faint); font-size: .76rem; font-family: var(--tt-mono); }
.tt-chip b { color: var(--tt-text); font-weight: 500; }
.tt-truth-strip { display: flex; align-items: center; gap: .8rem; margin: 1rem 0 .7rem; }
.tt-truth-label { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--tt-danger);
  background: rgba(240,100,73,.1); border: 1px solid rgba(240,100,73,.2); padding: .25rem .55rem; border-radius: 4px; display: flex; align-items: center; white-space: nowrap; }
.tt-truth-line { height: 1px; flex: 1; background: var(--tt-border); }
.tt-truth-chips { display: flex; flex-wrap: wrap; gap: .7rem; }
.tt-chip-truth { background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: 6px; padding: .45rem .75rem; font-family: var(--tt-mono); font-size: .76rem; color: var(--tt-muted); }

/* ---------- streamlit widgets ---------- */
div[role="radiogroup"] { gap: .3rem; }
div[role="radiogroup"] label { background: var(--tt-raised); border: 1px solid var(--tt-border); border-radius: 6px; padding: .3rem .65rem; margin: 0; }
div[role="radiogroup"] label[data-selected="true"] { background: var(--tt-interactive); border-color: var(--tt-coolant); }
div[role="radiogroup"] label[data-selected="true"] p { color: var(--tt-text); }
.stButton > button, .stDownloadButton > button { border: 1px solid var(--tt-border); background: var(--tt-raised); color: var(--tt-text); border-radius: 6px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--tt-coolant); color: var(--tt-coolant); }
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] { background: var(--tt-coolant) !important; border-color: var(--tt-coolant) !important; color: #04121A !important; }
button[data-testid="stBaseButton-primary"] p, button[data-testid="stBaseButton-primaryFormSubmit"] p { color: #04121A !important; font-weight: 700; }
button[data-testid="stBaseButton-primary"]:disabled { opacity: .45; }
div[data-testid="stForm"] { border: 0; padding: 0; }
div[data-testid="stExpander"] { border: 1px solid var(--tt-border); border-radius: 8px; background: var(--tt-surface); }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: var(--tt-raised); border-color: var(--tt-border); }
div[data-testid="stDataFrame"] { border: 1px solid var(--tt-border); border-radius: 6px; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; padding-top: .6rem !important; }
.stAlert { border-radius: 8px; border: 1px solid var(--tt-border); }
hr { border-color: var(--tt-border); }
@media (max-width: 1100px) {
  .st-key-tt_rail_left, .st-key-tt_rail_right { min-height: 0; }
  .tt-pack-flow { grid-template-columns: 1fr 2.4rem 1fr 2.4rem 1fr 2.4rem 1fr; }
}
@media (max-width: 768px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 0 .75rem 2rem !important; }
  .st-key-tt_header { position: static; margin: 0 -.75rem 1rem; padding: .45rem .75rem; }
  .st-key-tt_nav div[role="radiogroup"] { flex-wrap: wrap; }
  .tt-tile { min-height: 4rem; }
  .tt-tile-sub { display: none; }
}
</style>
"""


def apply_plot_theme(figure: Figure) -> Figure:
    """Apply the V2 dark instrument theme to an existing Plotly figure."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color="#A6B8C3"),
        title_font=dict(color="#F3F7F9", size=17),
        legend=dict(bgcolor="rgba(7,16,24,.75)", bordercolor="#29404D"),
        hoverlabel=dict(bgcolor="#132431", bordercolor="#29404D", font_color="#F3F7F9"),
        margin=dict(l=55, r=30, t=68, b=52),
    )
    figure.update_xaxes(gridcolor="#1D3441", zerolinecolor="#29404D")
    figure.update_yaxes(gridcolor="#1D3441", zerolinecolor="#29404D")
    for annotation in figure.layout.annotations:
        if getattr(annotation, "bgcolor", None) == "#FFFFFF":
            annotation.bgcolor = "#132431"
            annotation.bordercolor = "#29404D"
            annotation.font.color = "#F3F7F9"
    return figure


__all__ = [
    "FORMULAS",
    "GLOBAL_CSS",
    "STORY_STEPS",
    "FormulaSpec",
    "StoryStep",
    "apply_plot_theme",
]
