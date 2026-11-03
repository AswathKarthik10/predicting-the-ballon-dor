from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.model import ballots as bl
from src.model import ballots_v2 as b2
from src.model import jury_v2 as jv

ROOT = Path(__file__).resolve().parents[1]
V1_HOME_BETA = 2.3109540708040397
V1_CONFED_BETA = 0.3663374288795643
V1_LEAGUE_BETA = 0.47806985501276095
FD_STEP = 1e-6
FD_TOL = 1e-4
N_ROWS_EXPECTED = 37_770


@pytest.fixture(scope="module")
def ballots() -> pd.DataFrame:
    return b2.load_ballots_v2(ROOT)


@pytest.fixture(scope="module")
def attention() -> pd.DataFrame:
    return b2.read_tables_v2(ROOT).attention


def test_ballot_table_covers_nine_editions_and_all_votes(ballots: pd.DataFrame) -> None:
    assert len(ballots) == N_ROWS_EXPECTED
    assert tuple(sorted(ballots.edition.unique())) == b2.BALLOT_EDITIONS_V2


def test_2016_ballots_use_the_three_pick_scheme(ballots: pd.DataFrame) -> None:
    picks = ballots[(ballots.edition == 2016) & (ballots.points > 0)].groupby("juror_country_key").points.apply(lambda s: tuple(sorted(s, reverse=True)))
    assert set(picks) == {b2.SCHEME_3}


def test_ballot_validation_catches_a_changed_vote() -> None:
    jury = b2.read_tables_v2(ROOT)
    from src.model.data import T1_FILE, V2_DIR
    from src.model.jurors import load_juror_inputs_from

    table = bl.build_ballot_table(jury.tables, load_juror_inputs_from(ROOT / V2_DIR / T1_FILE))
    table.loc[table.index[table.points > 0][0], "points"] += 1
    with pytest.raises(ValueError):
        b2.validate_ballots_v2(table, jury.tables)


def test_v1_coefficients_are_reproduced_on_2022_2025(ballots: pd.DataFrame) -> None:
    data = jv.build_data(ballots, jv.SEASON_ERA, b2.BASE_INDICATORS)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    np.testing.assert_allclose(fit.beta, [V1_HOME_BETA, V1_CONFED_BETA, V1_LEAGUE_BETA], atol=1e-6)


def test_home_era_interaction_is_zero_before_the_reform(ballots: pd.DataFrame) -> None:
    before = ballots[ballots.edition < b2.FIRST_FIFA_TOP100_EDITION]
    assert (before[b2.HOME_ERA_COLUMN] == 0).all() and before.home.sum() > 0
    after = ballots[ballots.edition >= b2.FIRST_FIFA_TOP100_EDITION]
    assert (after[b2.HOME_ERA_COLUMN] == after.home).all()


def test_home_split_columns_partition_home(ballots: pd.DataFrame) -> None:
    parts = ballots[["home_uefa", "home_conmebol", "home_other"]].sum(axis=1)
    assert (parts == ballots.home).all()
    assert (ballots.home_top_tier + ballots.home_not_top_tier == ballots.home).all()


def test_top_tier_marks_three_nominees_per_edition(ballots: pd.DataFrame) -> None:
    per_edition = ballots[ballots.top_tier == 1].groupby("edition").player_key.nunique()
    assert (per_edition >= b2.TOP_TIER_PLACES).all() and (per_edition <= b2.TOP_TIER_PLACES + 1).all()


def test_local_attention_is_standardised_within_covered_cells(attention: pd.DataFrame) -> None:
    local = b2.local_attention_table(attention, "wiki_country_share_in")
    means = local.groupby(["edition", "juror_country_key"]).local_attention.mean().abs()
    sizes = local.groupby(["edition", "juror_country_key"]).size()
    assert (means < 1e-9).all()
    assert (sizes == 30).all()


def test_only_cells_with_enough_nominees_are_covered(attention: pd.DataFrame) -> None:
    local = b2.local_attention_table(attention, "wiki_country_share_in")
    present = attention.groupby(["edition", "country_key"]).wiki_country_share_in.count()
    covered = local[["edition", "juror_country_key"]].drop_duplicates().set_index(["edition", "juror_country_key"]).index
    assert (present.loc[covered] >= b2.MIN_COVERED_NOMINEES).all()
    assert (present.drop(covered) < b2.MIN_COVERED_NOMINEES).all()


def test_gradient_is_correct_with_four_indicators() -> None:
    rng = np.random.default_rng(7)
    n_ballots, n_players, n_indicators, k = 6, 7, 4, 3
    picks = np.array([rng.permutation(n_players)[:k] for _ in range(n_ballots)])
    data = [bl.EditionBallots(edition=2016, player_keys=tuple(f"p{i}" for i in range(n_players)), juror_keys=tuple(f"j{i}" for i in range(n_ballots)),
                              indicators=rng.integers(0, 2, size=(n_ballots, n_players, n_indicators)).astype(float), picks=picks)]
    theta = rng.normal(scale=0.3, size=n_players + n_indicators)
    value, grad = bl.penalised_negative_loglik(theta, data, 0.01)
    numeric = approx_fprime(theta, lambda t: bl.penalised_negative_loglik(t, data, 0.01)[0], FD_STEP)
    np.testing.assert_allclose(grad, numeric, atol=FD_TOL)
    hess = bl.penalised_hessian(theta, data, 0.01)
    assert hess.shape == (n_players + n_indicators,) * 2
    assert np.isfinite(value)


def test_home_after_reform_adds_the_two_coefficients() -> None:
    indicators = (*b2.BASE_INDICATORS, b2.HOME_ERA_COLUMN)
    fit = bl.BiasFit(editions=(2016,), alpha={2016: np.zeros(2)}, beta=np.array([2.0, 0.3, 0.4, -0.5]), covariance=np.eye(4) * 0.01, log_likelihood=0.0)
    boot = np.tile(fit.beta, (50, 1)) + np.linspace(-0.1, 0.1, 50)[:, None]
    table = jv.home_after_reform(fit, boot, indicators)
    assert table.beta.iloc[0] == pytest.approx(1.5)
    assert table.laplace_hi90.iloc[0] - table.laplace_lo90.iloc[0] > 0
