from pathlib import Path

import numpy as np
import pytest

from src.model import cv, explain, inference
from src.model.data import FORECAST_EDITION, StudyData, load_study_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables() -> tuple:
    study = load_study_data(ROOT)
    f1 = inference.final_fit(study, cv.PL_SPECS["M1"])
    f2 = inference.final_fit(study, cv.PL_SPECS["M2"])
    return (explain.contribution_table(study, f1, FORECAST_EDITION), explain.contribution_table(study, f2, FORECAST_EDITION), f1)


def test_contributions_sum_to_utility_and_match_matrix_product(tables: tuple) -> None:
    t1, _, fit = tables
    cols = explain.feature_columns(t1)
    np.testing.assert_allclose(t1[cols].sum(axis=1), t1["utility"], atol=1e-12)
    x = fit.matrices[FORECAST_EDITION]
    by_key = t1.set_index("player_key")
    assert np.isclose(by_key["utility"].sum(), (x @ fit.beta).sum())


def test_first_pick_probabilities_sum_to_one_and_follow_rank(tables: tuple) -> None:
    t1, _, _ = tables
    assert t1["first_pick_probability"].sum() == pytest.approx(1.0)
    assert t1["first_pick_probability"].is_monotonic_decreasing
    assert t1["rank_by_utility"].tolist() == list(range(1, 31))


def test_pair_gap_terms_sum_to_utility_difference(tables: tuple) -> None:
    t1, _, _ = tables
    a, b = t1.iloc[0]["player_name"], t1.iloc[5]["player_name"]
    gap = explain.explain_pair(t1, a, b)
    assert gap["gap"].sum() == pytest.approx(t1.iloc[0]["utility"] - t1.iloc[5]["utility"])
    assert gap["gap"].abs().is_monotonic_decreasing


def test_shift_table_splits_utility_change_exactly(tables: tuple) -> None:
    t1, t2, _ = tables
    shifts = explain.shift_table(t1, t2)
    delta_cols = [c for c in shifts.columns if c.startswith("delta_")]
    np.testing.assert_allclose(shifts[delta_cols].sum(axis=1), shifts["utility_change"], atol=1e-10)
    assert {"delta_P1_post_form", "delta_P2_post_attention", "delta_P3_final_spike"} <= set(delta_cols)


def test_relative_contributions_are_centred(tables: tuple) -> None:
    t1, _, _ = tables
    rel = explain.relative_contributions(t1)
    np.testing.assert_allclose(rel.mean(), 0.0, atol=1e-12)
