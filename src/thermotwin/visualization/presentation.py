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
:root {
  --tt-bg: #071018;
  --tt-surface: #0D1822;
  --tt-raised: #132431;
  --tt-interactive: #18313E;
  --tt-border: #29404D;
  --tt-text: #F3F7F9;
  --tt-muted: #A6B8C3;
  --tt-coolant: #27C2D1;
  --tt-forecast: #62D6A7;
  --tt-warning: #F4B942;
  --tt-danger: #F06449;
}
.stApp { background: var(--tt-bg); color: var(--tt-text); }
[data-testid="stHeader"] { background: rgba(7, 16, 24, .92); border-bottom: 1px solid var(--tt-border); }
[data-testid="stSidebar"] { background: var(--tt-surface); border-right: 1px solid var(--tt-border); }
[data-testid="stSidebar"] * { color: var(--tt-text); }
.block-container { max-width: 1540px; padding-top: 1.15rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
p, label, .stCaption { color: var(--tt-muted); }
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(24,49,62,.76), rgba(13,24,34,.96));
  border: 1px solid var(--tt-border); border-radius: 8px; padding: .85rem 1rem;
}
div[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; }
div[data-testid="stMetricValue"] { font-family: "JetBrains Mono", Consolas, monospace; color: var(--tt-text); }
.tt-hero { background: linear-gradient(110deg, rgba(19,36,49,.98), rgba(13,24,34,.92)); border: 1px solid var(--tt-border); border-radius: 10px; padding: 1.25rem 1.4rem; margin-bottom: 1rem; }
.tt-eyebrow { color: var(--tt-coolant); font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.tt-title { color: var(--tt-text); font-size: clamp(1.55rem, 2.5vw, 2.35rem); font-weight: 760; margin: .3rem 0; }
.tt-copy { color: var(--tt-muted); max-width: 920px; line-height: 1.55; }
.tt-badge { display: inline-flex; align-items: center; gap: .35rem; border: 1px solid var(--tt-border); background: var(--tt-raised); border-radius: 999px; padding: .25rem .6rem; margin: .25rem .3rem .1rem 0; color: var(--tt-muted); font-size: .73rem; }
.tt-panel { background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: 8px; padding: 1rem 1.05rem; margin: .5rem 0 1rem; }
.tt-panel-title { color: var(--tt-text); font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: .35rem; }
.tt-panel-copy { color: var(--tt-muted); line-height: 1.5; }
.tt-warning { border-left: 3px solid var(--tt-warning); }
.tt-danger { border-left: 3px solid var(--tt-danger); }
.tt-coolant { border-left: 3px solid var(--tt-coolant); }
.tt-forecast { border-left: 3px solid var(--tt-forecast); }
.tt-truth { background: rgba(240,100,73,.08); border: 1px solid rgba(240,100,73,.35); border-radius: 8px; padding: 1rem; }
div[role="radiogroup"] { gap: .25rem; }
div[role="radiogroup"] label { background: var(--tt-surface); border: 1px solid var(--tt-border); border-radius: 6px; padding: .42rem .7rem; }
div[role="radiogroup"] label:has(input:checked) { background: var(--tt-interactive); border-color: var(--tt-coolant); }
.stButton > button, .stDownloadButton > button { border: 1px solid var(--tt-border); background: var(--tt-raised); color: var(--tt-text); border-radius: 6px; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--tt-coolant); color: var(--tt-coolant); }
div[data-testid="stExpander"] { border: 1px solid var(--tt-border); border-radius: 8px; background: var(--tt-surface); }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { background: var(--tt-raised); border-color: var(--tt-border); }
.stAlert { border-radius: 8px; border: 1px solid var(--tt-border); }
hr { border-color: var(--tt-border); }
@media (max-width: 768px) {
  .block-container { padding-left: .8rem; padding-right: .8rem; }
  .tt-hero { padding: 1rem; }
}
</style>
"""


def apply_plot_theme(figure: Figure) -> Figure:
    """Apply the V2 dark instrument theme to an existing Plotly figure."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1822",
        plot_bgcolor="#0D1822",
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
