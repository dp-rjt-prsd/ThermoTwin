from __future__ import annotations

import numpy as np
import pytest

from thermotwin.visualization.ui_components import (
    anomaly_card,
    card_header,
    kpi,
    pack_tiles,
    truth_strip,
)


def test_pack_tiles_render_twelve_labelled_cells_in_row_major_order() -> None:
    markup = pack_tiles(np.arange(12.0), value_range=(0.0, 11.0), unit="°C")

    assert markup.count('<div class="tt-tile') == 12
    assert markup.index(">C1<") < markup.index(">C4<") < markup.index(">C12<")
    assert "<small> °C</small>" in markup


def test_pack_tiles_mark_forecast_and_hot_cells() -> None:
    markup = pack_tiles(
        [30.0] * 12, value_range=(25.0, 35.0), unit="°C", highlight=5, forecast=True
    )

    assert markup.count("tt-tile tt-tile-forecast") == 12
    assert markup.count("tt-tile-hot") == 1


def test_pack_tiles_draw_conduction_connectors_from_hot_to_cold() -> None:
    markup = pack_tiles(
        [30.0] * 12,
        value_range=(25.0, 35.0),
        unit="°C",
        flows=[(5, 6, 2.0), (5, 1, 1.5)],
    )

    assert "tt-pack-flow" in markup
    assert "<b>→</b><span>2.0 W</span>" in markup
    assert "<b>↑</b><span>1.5 W</span>" in markup


def test_pack_tiles_reject_wrong_cell_count() -> None:
    with pytest.raises(ValueError):
        pack_tiles([1.0] * 11, value_range=(0.0, 1.0), unit="W")


def test_text_components_escape_user_visible_strings() -> None:
    assert "&lt;b&gt;" in card_header("<b>x</b>")
    assert "&lt;script&gt;" in kpi("<script>", "1")
    assert "&lt;i&gt;" in truth_strip([("<i>", "C1", "primary")])
    assert "Score: 3.00" in anomaly_card(
        "Anomaly", "3.00", ["Suspected cell: <b>C1</b>"]
    )
