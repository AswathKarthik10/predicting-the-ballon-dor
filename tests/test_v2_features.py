import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import baselines, cv
from src.model.cv import edition_metrics
from src.model.data import TARGET_COLS, StudyData, load_study_v2
from src.model.features_v2 import (FORMAT_INTERACTIONS, RESIDUALISER_REGRESSORS, V2_CORE_FEATURES, attack_raw_npga,
                                   individual_awards_raw, market_value_at_t1)
from src.model.models_extra import N_BANDS, NO_VOTE_BAND, rank_bands
from src.model.validation_v2 import apply_selection_rule

ROOT = Path(__file__).resolve().parents[1]
FEATURES_V2_SOURCE = ROOT / "src" / "model" / "features_v2.py"
HELD_OUT = 2021
TRAIN = [2018, 2019, 2022, 2025]
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def study() -> StudyData:
    return load_study_v2(ROOT)


def with_scrambled_targets(study: StudyData, edition: int) -> StudyData:
    targets = study.targets.copy()
    mask = (study.frame_t1["edition"] == edition).to_numpy()
    rng = np.random.default_rng(1)
    for col in ("bdo_points", "vote_share"):
        targets.loc[mask, col] = rng.permutation(targets.loc[mask, col].to_numpy())
    return replace(study, targets=targets)


def test_feature_module_never_mentions_targets_or_odds() -> None:
    source = FEATURES_V2_SOURCE.read_text()
    for name in TARGET_COLS:
        assert not re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", source), f"features_v2.py must not reference {name}"
    assert "odds" not in source.lower() and "implied_prob" not in source.lower()


def test_individual_awards_counts_runner_up_golden_ball_as_half() -> None:
    df = pd.DataFrame({"major_individual_awards_n": [1, 0, 2, 0], "league_player_of_season": [True, False, False, False],
                       "uefa_player_award": [False, False, True, False], "award_wc_golden_ball": [1, 2, 3, 0]})
    np.testing.assert_allclose(individual_awards_raw(df).to_numpy(), [2.0, 0.5, 3.5, 0.0], atol=TOLERANCE)


def test_retired_player_market_value_falls_back_to_window_start() -> None:
    df = pd.DataFrame({"tm_value_eur_t1": [0, 40_000_000], "tm_value_eur_window_start": [15_000_000.0, 38_000_000.0]})
    np.testing.assert_allclose(market_value_at_t1(df).to_numpy(), [15_000_000.0, 40_000_000.0])


def test_attack_uses_club_value_when_international_is_missing() -> None:
    df = pd.DataFrame({"club_npga_p90_opp_adj": [0.8, 0.6], "intl_npga_p90_opp_adj": [np.nan, 1.0],
                       "club_minutes": [3000.0, 3000.0], "intl_minutes": [100.0, 1000.0]})
    np.testing.assert_allclose(attack_raw_npga(df).to_numpy(), [0.8, (0.6 * 3000 + 1.0 * 1000) / 4000])


def test_core_features_are_finite_standardised_and_goalkeepers_have_zero_attack(study: StudyData) -> None:
    frame = study.frame_t1
    columns = [c for c in V2_CORE_FEATURES if c != "F10_season_attention"]
    assert np.isfinite(frame[columns].to_numpy(dtype=float)).all()
    assert (frame.loc[frame["is_goalkeeper"] == 1, "F1_attack"] == 0).all()
    continuous = [c for c in columns if c != "F8_defending_winner"]
    means = frame.groupby("edition")[continuous].mean().abs().to_numpy()
    assert (means < 1e-9).all()


def test_residualiser_regressors_are_f1_to_f7_and_awards() -> None:
    assert RESIDUALISER_REGRESSORS == V2_CORE_FEATURES[:7] + ("F11_individual_awards",)


def test_format_interactions_vanish_in_season_editions(study: StudyData) -> None:
    matrix = baselines.design_matrices(study, "t1", list(study.features.model_columns["M1-F"]), False, [2022, 2023], [2017, 2024])
    n_main = len(study.features.model_columns["M1"])
    assert np.any(matrix[2017][:, n_main:] != 0)
    assert np.all(matrix[2024][:, n_main:] == 0)
    assert len(FORMAT_INTERACTIONS) == matrix[2017].shape[1] - n_main


def test_training_matrices_do_not_depend_on_the_held_out_edition(study: StudyData) -> None:
    columns = list(study.features.model_columns["M1"])
    before = baselines.design_matrices(study, "t1", columns, False, TRAIN, TRAIN + [HELD_OUT])
    altered_frame = study.frame_t1.copy()
    held = altered_frame["edition"] == HELD_OUT
    altered_frame.loc[held, "season_attention_raw"] = altered_frame.loc[held, "season_attention_raw"] * -7 + 4
    altered_frame.loc[held, "F7_prior_standing"] += 3
    after = baselines.design_matrices(replace(study, frame_t1=altered_frame), "t1", columns, False, TRAIN, TRAIN + [HELD_OUT])
    for edition in TRAIN:
        np.testing.assert_array_equal(before[edition], after[edition])


@pytest.mark.parametrize("name", ["M1", "M1-G", "M1-F", "B1", "L1", "O1", "D1"])
def test_predictions_for_held_out_edition_do_not_depend_on_its_targets(study: StudyData, name: str) -> None:
    before, _ = cv.predict_model(study, name, TRAIN, [HELD_OUT])
    after, _ = cv.predict_model(with_scrambled_targets(study, HELD_OUT), name, TRAIN, [HELD_OUT])
    np.testing.assert_allclose(before[HELD_OUT], after[HELD_OUT], rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize("name", ["L1", "O1", "D1"])
def test_extra_models_beat_the_uniform_reference_on_a_training_like_edition(study: StudyData, name: str) -> None:
    utilities, _ = cv.predict_model(study, name, [e for e in study.labelled_editions if e != 2024], [2024])
    points = baselines.points_of(study, 2024)
    uniform = edition_metrics(np.zeros(len(points)), points)["heldout_ll"]
    assert edition_metrics(utilities[2024], points)["heldout_ll"] > uniform


def test_rank_bands_on_known_case() -> None:
    points = np.array([30.0, 20.0, 20.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    bands = rank_bands(points)
    # winner=0; tied 2nd-3rd=1; ranks 4..10 (the 4th to the 10th place)=2; 11th and below with votes=3; no votes=4
    assert bands[0] == 0 and bands[1] == bands[2] == 1
    assert list(bands[3:9]) == [2] * 6
    assert bands[9] == 2 and bands[10] == bands[11] == 3 and bands[12] == NO_VOTE_BAND == N_BANDS - 1


def test_top10_accuracy_and_ndcg_on_known_case() -> None:
    points = np.arange(20, 0, -1, dtype=float)
    perfect = edition_metrics(points.copy(), points)
    assert perfect["top10_acc"] == 1.0 and perfect["ndcg10"] == pytest.approx(1.0)
    reversed_ = edition_metrics(-points, points)
    discount = 1.0 / np.log2(np.arange(2, 12))
    expected = (np.arange(1, 11) * discount).sum() / (np.arange(20, 10, -1) * discount).sum()  # worst ten players ranked 1st to 10th
    assert reversed_["top10_acc"] == 0.0 and reversed_["ndcg10"] == pytest.approx(expected)


def test_selection_prefers_the_simplest_pl_model_within_the_margin() -> None:
    selection = apply_selection_rule({"M1": -2.30, "M1-G": -2.31, "M1-F": -2.29, "X1": -2.40, "B1": -2.50, "U0": -2.7})
    assert selection.published == "M1"
    assert selection.reported_alongside == "X1"


def test_selection_switches_to_the_richer_model_when_it_wins_clearly_and_logs_a_non_pl_winner() -> None:
    selection = apply_selection_rule({"M1": -2.40, "M1-G": -2.41, "M1-F": -2.30, "L1": -2.20, "X1": -2.5, "U0": -2.7})
    assert selection.published == "M1-F"
    assert selection.reported_alongside == "L1"
    assert any("rule 5.4(3)" in note for note in selection.notes)
