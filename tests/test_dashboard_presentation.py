from __future__ import annotations

from thermotwin.visualization.presentation import FORMULAS, STORY_STEPS


def test_formula_catalog_covers_every_contextual_story_equation() -> None:
    referenced = {formula_id for step in STORY_STEPS for formula_id in step.formula_ids}

    assert referenced <= FORMULAS.keys()
    assert {step.number for step in STORY_STEPS} == set(range(1, 8))
    assert all(spec.latex and spec.explanation for spec in FORMULAS.values())
    assert all(spec.provenance.startswith("thermotwin.") for spec in FORMULAS.values())


def test_formula_catalog_labels_approximations_and_avoids_runaway_claims() -> None:
    assert "approximation" in (FORMULAS["pump"].approximation or "").lower()
    combined_copy = " ".join(
        f"{spec.title} {spec.explanation} {spec.approximation or ''}"
        for spec in FORMULAS.values()
    ).lower()

    assert "thermal-runaway detection" not in combined_copy
