"""HTML building blocks for the ThermoTwin control-room layout.

Streamlit renders these strings with ``st.markdown(..., unsafe_allow_html=True)``.
They carry no behaviour: every interactive control stays a Streamlit widget.
All values passed in come from validated artifacts or the live simulator.
"""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import numpy as np

from thermotwin.visualization.judge_figures import THERMAL_SCALE

# Lucide-style inline icons (stroke icons, 24×24 viewBox).
_ICON_PATHS = {
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
    '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "sigma": '<path d="M18 7V4H6l6 8-6 8h12v-3"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "waves": '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
    '<path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>'
    '<path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/>',
    "alert": '<path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/>'
    '<path d="M12 9v4M12 17h.01"/>',
    "forward": '<path d="m13 19 9-7-9-7v14zM2 19l9-7-9-7v14z"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.4-.5-2-1-3-1.1-2.1-.2-4 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.2.4-2.3 1-3.2.3 1.1 1 2.2 2.5 2.7z"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
}


def icon(name: str, *, size: int = 14, color: str = "currentColor") -> str:
    """Inline SVG icon markup."""
    return (
        f'<svg class="tt-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICON_PATHS[name]}</svg>"
    )


def _scale_rgb(position: float) -> tuple[int, int, int]:
    position = min(max(position, 0.0), 1.0)
    stops = [(stop, color) for stop, color in THERMAL_SCALE]
    for (left, left_color), (right, right_color) in zip(stops, stops[1:], strict=False):
        if position <= right:
            fraction = 0.0 if right == left else (position - left) / (right - left)
            a = [int(left_color[i : i + 2], 16) for i in (1, 3, 5)]
            b = [int(right_color[i : i + 2], 16) for i in (1, 3, 5)]
            r, g, bl = (round(a[k] + fraction * (b[k] - a[k])) for k in range(3))
            return r, g, bl
    last = stops[-1][1]
    return int(last[1:3], 16), int(last[3:5], 16), int(last[5:7], 16)


def brand() -> str:
    return (
        '<div class="tt-brand">'
        f"{icon('cpu', size=18, color='#8BDFF0')}<span>ThermoTwin</span></div>"
    )


def page_heading(title: str, copy: str, disclaimer: str | None = None) -> str:
    note = (
        f' <span class="tt-disclaimer">{escape(disclaimer)}</span>'
        if disclaimer
        else ""
    )
    return (
        f'<div class="tt-heading"><h1>{escape(title)}</h1>'
        f"<p>{escape(copy)}{note}</p></div>"
    )


def card_header(
    label: str, meta: str = "", *, icon_name: str | None = None, tone: str = "muted"
) -> str:
    symbol = icon(icon_name) if icon_name else ""
    meta_markup = (
        f'<span class="tt-card-meta tt-tone-{tone}">{escape(meta)}</span>'
        if meta
        else ""
    )
    return (
        f'<div class="tt-card-head"><span class="tt-card-label tt-tone-{tone}">'
        f"{symbol}{escape(label)}</span>{meta_markup}</div>"
    )


def pack_tiles(
    values: Sequence[float] | np.ndarray,
    *,
    value_range: tuple[float, float],
    unit: str,
    decimals: int = 1,
    secondary: Sequence[str] | None = None,
    highlight: int | None = None,
    forecast: bool = False,
    flows: Sequence[tuple[int, int, float]] | None = None,
) -> str:
    """The 3×4 pack as tinted tiles; optional conduction connectors between cells."""
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.shape != (12,):
        raise ValueError("pack_tiles requires exactly 12 cell values")
    low, high = value_range
    span = high - low if high > low else 1.0
    connectors: dict[tuple[int, int], tuple[int, float]] = {}
    for hot, cold, watts in flows or ():
        connectors[(min(hot, cold), max(hot, cold))] = (hot, watts)

    def tile(index: int) -> str:
        red, green, blue = _scale_rgb((data[index] - low) / span)
        classes = "tt-tile" + (" tt-tile-forecast" if forecast else "")
        if index == highlight:
            classes += " tt-tile-hot"
        sub = (
            f'<span class="tt-tile-sub">{escape(secondary[index])}</span>'
            if secondary is not None
            else ""
        )
        return (
            f'<div class="{classes}" style="background:rgba({red},{green},{blue},.24);'
            f'border-color:rgba({red},{green},{blue},.65)">'
            f'<span class="tt-tile-id">C{index + 1}</span>'
            f'<span class="tt-tile-value">{data[index]:.{decimals}f}'
            f"<small> {escape(unit)}</small></span>{sub}</div>"
        )

    def connector(left: int, right: int, horizontal: bool) -> str:
        found = connectors.get((left, right))
        if found is None:
            return '<div class="tt-conn"></div>'
        hot, watts = found
        if horizontal:
            arrow = "→" if hot == left else "←"
        else:
            arrow = "↓" if hot == left else "↑"
        return f'<div class="tt-conn tt-conn-on"><b>{arrow}</b><span>{watts:.1f} W</span></div>'

    if not connectors:
        body = "".join(tile(index) for index in range(12))
        return f'<div class="tt-pack">{body}</div>'
    cells: list[str] = []
    for row in range(3):
        for column in range(4):
            index = row * 4 + column
            cells.append(tile(index))
            if column < 3:
                cells.append(connector(index, index + 1, True))
        if row < 2:
            for column in range(4):
                index = row * 4 + column
                cells.append(connector(index, index + 4, False))
                if column < 3:
                    cells.append('<div class="tt-conn"></div>')
    return f'<div class="tt-pack tt-pack-flow">{"".join(cells)}</div>'


def scale_legend(value_range: tuple[float, float], unit: str) -> str:
    low, high = value_range
    stops = ", ".join(f"{color} {stop * 100:.0f}%" for stop, color in THERMAL_SCALE)
    return (
        '<div class="tt-legend">'
        f"<span>{low:.0f} {escape(unit)}</span>"
        f'<div class="tt-legend-bar" style="background:linear-gradient(90deg,{stops})"></div>'
        f"<span>{high:.0f} {escape(unit)}</span>"
        '<span class="tt-legend-note">liquid cold plate under all cells</span></div>'
    )


def kpi(label: str, value: str, unit: str = "", *, tone: str = "") -> str:
    unit_markup = f"<small>{escape(unit)}</small>" if unit else ""
    tone_class = f" tt-tone-{tone}" if tone else ""
    return (
        f'<div class="tt-kpi"><div class="tt-kpi-label">{escape(label)}</div>'
        f'<div class="tt-kpi-value{tone_class}">{escape(value)} {unit_markup}</div></div>'
    )


def kpi_grid(items: Sequence[str]) -> str:
    return f'<div class="tt-kpi-grid">{"".join(items)}</div>'


def rail_title(label: str, *, live: bool = False) -> str:
    dot = '<span class="tt-pulse"></span>' if live else ""
    return f'<div class="tt-rail-title">{dot}{escape(label)}</div>'


def setting_block(rows: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="tt-setting"><label>{escape(label)}</label><div>{escape(value)}</div></div>'
        for label, value in rows
    )
    return f'<div class="tt-settings">{body}</div>'


def flow_card(flow_kg_s: float, pump_w: float | None = None) -> str:
    pump = (
        f'<div class="tt-flow-sub">pump {pump_w:.1f} W</div>'
        if pump_w is not None
        else ""
    )
    return (
        '<div class="tt-flow"><div><div class="tt-kpi-label">Coolant flow (ṁ)</div>'
        f'<div class="tt-kpi-value tt-tone-coolant">{flow_kg_s:.4f} <small>kg/s</small></div>'
        f"{pump}</div>{icon('waves', size=26, color='rgba(39,194,209,.55)')}</div>"
    )


def anomaly_card(
    title: str, score: str | None, lines: Sequence[str], *, tone: str = "danger"
) -> str:
    score_markup = (
        f'<span class="tt-alert-score">Score: {escape(score)}</span>' if score else ""
    )
    body = "".join(f"<div>{line}</div>" for line in lines)
    return (
        f'<div class="tt-alert tt-alert-{tone}"><div class="tt-alert-head">'
        f"<span>{escape(title)}</span>{score_markup}</div>{body}</div>"
    )


def truth_strip(chips: Sequence[tuple[str, str, str]]) -> str:
    """Evaluation-only ground truth: (label, value, tone) chips."""
    body = "".join(
        f'<div class="tt-chip-truth">{escape(label)}: '
        f'<span class="tt-tone-{tone}">{escape(value)}</span></div>'
        for label, value, tone in chips
    )
    return (
        '<div class="tt-truth-strip"><div class="tt-truth-label">'
        f'{icon("alert")}Evaluation-only ground truth</div><div class="tt-truth-line"></div></div>'
        f'<div class="tt-truth-chips">{body}</div>'
    )


def notice(title: str, copy: str, tone: str = "coolant") -> str:
    return (
        f'<div class="tt-notice tt-notice-{tone}"><div class="tt-notice-title">'
        f"{escape(title)}</div><div>{copy}</div></div>"
    )


def chips(items: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f'<span class="tt-chip">{escape(label)} <b>{escape(value)}</b></span>'
        for label, value in items
    )
    return f'<div class="tt-chips">{body}</div>'


__all__ = [
    "anomaly_card",
    "brand",
    "card_header",
    "chips",
    "flow_card",
    "icon",
    "kpi",
    "kpi_grid",
    "notice",
    "pack_tiles",
    "page_heading",
    "rail_title",
    "scale_legend",
    "setting_block",
    "truth_strip",
]
