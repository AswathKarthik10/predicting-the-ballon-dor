from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src.model import ballots_v2 as b2
from src.model import cv, odds_v2, robustness_v2 as rb
from src.model import forecast_ballots as fb
from src.model import forecast_v2 as f2
from src.model.data import load_study_data, load_study_v2
from src.model.features_v2 import m1_columns_v2
from src.model.research_v2 import pl_spec_for

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CAPTURES = {(2016, "2016-10-24"), (2018, "2018-09-25"), (2019, "2019-10-21"), (2021, "2021-10-28"), (2024, "2024-07-16"),
                     (2024, "2024-08-28"), (2025, "2025-08-06")}
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def study():
    return load_study_v2(ROOT)


def test_usable_captures_follow_the_spec_rules(study) -> None:
    odds = odds_v2.read_odds(ROOT)
    captures = odds_v2.usable_captures(odds, odds_v2.t1_dates(ROOT), study.labelled_editions)
    assert {(int(e), str(d)) for e, d in zip(captures.edition, captures.captured_at)} == EXPECTED_CAPTURES
    assert not set(captures.capture_phase) & {"after_deadline"}
    assert 2026 not in set(captures.edition)
    assert captures.is_nominee.all()


def test_restricted_loglik_is_hand_checked_with_a_zero_point_player() -> None:
    utility = np.array([0.2, -0.4, 0.9])
    points = np.array([5.0, 0.0, 2.0])  # player 1 is quoted but received no points
    expected = (utility[0] - logsumexp(utility)) + (utility[2] - logsumexp(utility[[1, 2]]))
    value, n_voted = odds_v2.loglik_on_quoted(utility, points)
    assert n_voted == 2 and value == pytest.approx(expected, abs=TOLERANCE)


def test_market_and_model_loglik_agree_when_they_hold_the_same_probabilities(study) -> None:
    edition = 2018
    frame = study.frame_t1[study.frame_t1.edition == edition].reset_index(drop=True)
    quoted = frame.player_key.iloc[:4]
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    captures = pd.DataFrame({"edition": edition, "captured_at": "2018-09-25", "capture_phase": "before_t1", "days_before_t1": 38,
                             "player_key": quoted.to_numpy(), "implied_prob": probs})
    utilities = {edition: np.zeros(30)}
    utilities[edition][:4] = np.log(probs)
    row = odds_v2.capture_rows(study, captures, utilities).iloc[0]
    assert row.market_loglik == pytest.approx(row.model_loglik, abs=TOLERANCE)


def test_forecast_refuses_to_run_when_a_result_file_exists(tmp_path: Path) -> None:
    f2.refuse_if_result_present(tmp_path)
    (tmp_path / "data" / "holdout").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        f2.refuse_if_result_present(tmp_path)


def test_forecast_refuses_a_result_file_anywhere_in_the_repo(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "bdo_2026_result.csv").write_text("rank,player\n")
    with pytest.raises(RuntimeError):
        f2.refuse_if_result_present(tmp_path)


def test_the_repo_has_no_result_file_today() -> None:
    f2.refuse_if_result_present(ROOT)


def test_2026_juror_inputs_from_the_v2_tables_match_the_v1_inputs(study) -> None:
    new = f2.inputs_2026_v2(ROOT, study)
    old = fb.inputs_2026(ROOT, load_study_data(ROOT))
    assert new.n_jurors == old.n_jurors == 100
    np.testing.assert_array_equal(new.nation_match, old.nation_match)
    np.testing.assert_array_equal(new.confed_match, old.confed_match)
    np.testing.assert_array_equal(new.league_match, old.league_match)


def test_era_alphas_cover_every_labelled_player_edition() -> None:
    ballots = b2.load_ballots_v2(ROOT)
    alphas, season_fit = f2.era_alpha_table(ballots)
    assert len(alphas) == 270 and set(alphas.edition) == set(b2.BALLOT_EDITIONS_V2)
    assert season_fit.editions == (2022, 2023, 2024, 2025)
    assert season_fit.beta[0] == pytest.approx(2.3109540708, abs=1e-6)


def test_compare_with_v1_keeps_one_row_per_player() -> None:
    v2 = pd.DataFrame({"player_key": ["a", "b"], "player_name": ["A", "B"], "p_win": [.6, .4], "p_top3": [.9, .8], "expected_rank": [1.4, 2.0],
                       "forecast_rank": [1, 2]})
    v1 = pd.DataFrame({"player_key": ["a", "b"], "p_win": [.5, .5], "p_top3": [.9, .9], "expected_rank": [1.5, 1.6], "forecast_rank": [1, 2]})
    out = f2.compare_with_v1(v2, v1)
    assert len(out) == 2 and out.p_win_change.iloc[0] == pytest.approx(0.1)


def test_robustness_runs_cover_every_listed_variation(study) -> None:
    runs = {r.run_id: r for r in rb.build_runs(ROOT, study)}
    assert set(runs) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7a", "R7b", "R8", "R9"}
    assert runs["R1"].study.labelled_editions == (2022, 2023, 2024, 2025)
    assert 2016 not in runs["R2"].study.labelled_editions and len(runs["R2"].study.labelled_editions) == 8
    assert runs["R3"].study.labelled_editions == (2016, 2017, 2018, 2019, 2021)
    assert runs["R4"].study.labelled_editions == (2021, 2022, 2023, 2024, 2025)
    assert "F2_form_rating" not in runs["R5"].columns and "ea_rating_z" in runs["R6"].columns
    assert "F12_market_value" not in runs["R7a"].columns
    assert not {"F9_fame", "F10_season_attention"} & set(runs["R8"].columns)


def test_xg_and_league_poty_variants_change_the_inputs(study) -> None:
    runs = {r.run_id: r for r in rb.build_runs(ROOT, study)}
    base = study.frame_t1
    xg = runs["R4"].study.frame_t1
    assert np.isfinite(xg[["F1_attack", "F11_individual_awards"]].to_numpy()).all()
    poty = runs["R7b"].study.frame_t1
    assert not np.allclose(poty.F11_individual_awards, base.F11_individual_awards)


def test_a_format_main_effect_cannot_change_a_plackett_luce_fit(study) -> None:
    runs = {r.run_id: r for r in rb.build_runs(ROOT, study)}
    train = [2018, 2019, 2022, 2024]
    plain, _ = cv.predict_pl(study, pl_spec_for("M1", m1_columns_v2()), train, [2025])
    fixed, _ = cv.predict_pl(runs["R9"].study, pl_spec_for("M1", runs["R9"].columns), train, [2025])
    np.testing.assert_allclose(plain[2025], fixed[2025], atol=1e-5)
