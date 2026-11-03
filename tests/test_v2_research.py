from pathlib import Path

import numpy as np
import pytest

from src.model import research_v2 as rs
from src.model.data import load_study_t2_v2, load_study_v2
from src.model.features_v2 import BLOCKS_V2, m1_columns_v2

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-9


def test_blocks_partition_the_m1_columns() -> None:
    columns = [c for block in BLOCKS_V2.values() for c in block]
    assert sorted(columns) == sorted(m1_columns_v2())


def test_shapley_values_split_the_total_exactly() -> None:
    names = ("a", "b", "c")
    values = {frozenset(): 0.0, frozenset("a"): 0.3, frozenset("b"): 0.2, frozenset("c"): 0.0, frozenset("ab"): 0.6,
              frozenset("ac"): 0.3, frozenset("bc"): 0.2, frozenset("abc"): 0.7}
    shapley = rs.shapley_values(values, names)
    assert sum(shapley.values()) == pytest.approx(0.7, abs=TOLERANCE)
    # c adds 0 as first or second member and 0.1 as the last member of 2 of the 6 orderings: 0.1 x 2/6
    assert shapley["c"] == pytest.approx(0.1 / 3, abs=TOLERANCE)


def test_a_block_that_adds_nothing_has_zero_shapley_value() -> None:
    names = ("a", "b")
    values = {frozenset(): 0.0, frozenset("a"): 0.4, frozenset("b"): 0.0, frozenset("ab"): 0.4}
    assert rs.shapley_values(values, names)["b"] == pytest.approx(0.0, abs=TOLERANCE)


def test_points_percentile_target_is_standardised() -> None:
    target = rs.points_percentile_target(np.array([10.0, 5.0, 5.0, 1.0, 0.0, 0.0]))
    assert target.mean() == pytest.approx(0.0, abs=TOLERANCE) and target.std(ddof=1) == pytest.approx(1.0)
    assert target[0] == target.max() and target[3] > target[4]


def test_competition_rank_gives_ties_the_best_rank() -> None:
    np.testing.assert_array_equal(rs.competition_rank_of(np.array([10.0, 5.0, 5.0, 1.0, 0.0])), [1, 2, 2, 4, 5])


def test_permutation_test_detects_a_strong_fame_gap_and_not_noise() -> None:
    rng = np.random.default_rng(3)
    editions = np.repeat(np.arange(8), 10)
    high = np.tile(np.array([True] * 5 + [False] * 5), 8)
    strong_ranks = np.where(high, rng.uniform(1, 5, 80), rng.uniform(8, 12, 80))
    _, p_strong, _ = rs.permutation_pvalue(strong_ranks, high, editions, 2000, 7)
    noise_ranks = rng.uniform(1, 12, 80)
    _, p_noise, _ = rs.permutation_pvalue(noise_ranks, high, editions, 2000, 7)
    assert p_strong < 0.01 and p_noise > 0.05


def test_permutation_keeps_labels_inside_each_edition() -> None:
    editions = np.array([0, 0, 0, 1, 1, 1])
    high = np.array([True, False, False, True, True, False])
    ranks = np.array([1.0, 5.0, 6.0, 2.0, 3.0, 9.0])
    observed, _, _ = rs.permutation_pvalue(ranks, high, editions, 50, 1)
    assert observed == pytest.approx(ranks[~high].mean() - ranks[high].mean())


def test_post_window_study_has_constant_p1_in_2025() -> None:
    study = load_study_t2_v2(ROOT)
    assert study.frame_t2.groupby("edition").P1_post_form.std()[2025] == 0.0
    assert study.labelled_editions == (2022, 2023, 2024, 2025)
    assert "M2" in study.features.model_columns


def test_tournament_columns_flag_world_cup_players() -> None:
    frame = load_study_v2(ROOT).frame_t1
    assert frame[frame.edition == 2025].is_major_tournament.sum() == 0
    assert frame[frame.edition == 2018].is_world_cup.sum() > 20
    assert (frame.is_world_cup <= frame.is_major_tournament).all()
