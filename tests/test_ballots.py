from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import ballots as bl

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 7
FD_STEP = 1e-6
FD_TOL = 1e-5
N_JURORS = 100
N_PLAYERS = 30
TRUE_BETA = np.array([1.5, 0.4, 0.5])
SE_MULTIPLE = 2.0
HOME_PROB, CONFED_PROB, LEAGUE_PROB = 0.05, 0.2, 0.05

needs_jury_data = pytest.mark.skipif(not (ROOT / bl.JURY_DIR / bl.VOTES_FILE).exists(), reason="data/jury/ not populated")


@pytest.fixture(scope="module")
def real_ballots() -> pd.DataFrame:
    return bl.load_ballots(ROOT)


def synthetic_edition(rng: np.random.Generator, edition: int, alpha: np.ndarray, beta: np.ndarray) -> bl.EditionBallots:
    home = (rng.random((N_JURORS, N_PLAYERS)) < HOME_PROB).astype(float)
    confed = ((rng.random((N_JURORS, N_PLAYERS)) < CONFED_PROB) & (home == 0)).astype(float)
    league = (rng.random((N_JURORS, N_PLAYERS)) < LEAGUE_PROB).astype(float)
    indicators = np.stack([home, confed, league], axis=2)
    value = alpha[None, :] + indicators @ beta + rng.gumbel(size=(N_JURORS, N_PLAYERS))
    k = len(bl.SCHEME_BY_EDITION[edition])
    picks = np.argsort(-value, axis=1)[:, :k]
    return bl.EditionBallots(edition=edition, player_keys=tuple(f"p{i}" for i in range(N_PLAYERS)),
                             juror_keys=tuple(f"j{i}" for i in range(N_JURORS)), indicators=indicators, picks=picks)


def synthetic_data(seed: int) -> list[bl.EditionBallots]:
    rng = np.random.default_rng(seed)
    return [synthetic_edition(rng, e, rng.normal(scale=1.0, size=N_PLAYERS), TRUE_BETA) for e in bl.BALLOT_EDITIONS]


@needs_jury_data
@pytest.mark.parametrize("edition", bl.BALLOT_EDITIONS)
def test_ballot_points_reproduce_official_totals(real_ballots: pd.DataFrame, edition: int) -> None:
    results = bl.read_jury_tables(ROOT).results
    summed = real_ballots[real_ballots.edition == edition].groupby("player_key").points.sum()
    official = results[results.edition == edition].set_index("player_key").points
    pd.testing.assert_series_equal(summed.sort_index().astype(float), official.sort_index().astype(float), check_names=False)


@needs_jury_data
def test_home_and_confed_never_both_one(real_ballots: pd.DataFrame) -> None:
    assert not ((real_ballots.home == 1) & (real_ballots.confed == 1)).any()


@needs_jury_data
def test_checks_raise_on_corrupted_ballots(real_ballots: pd.DataFrame) -> None:
    broken = real_ballots.copy()
    broken.loc[broken.index[broken.points > 0][0], "points"] += 1
    tables = bl.read_jury_tables(ROOT)
    with pytest.raises(ValueError):
        bl.validate_ballots(broken, tables)


@needs_jury_data
def test_pick_rank_is_null_exactly_when_unpicked(real_ballots: pd.DataFrame) -> None:
    assert (real_ballots.pick_rank.isna() == (real_ballots.points == 0)).all()


def test_gradient_matches_finite_differences() -> None:
    data = synthetic_data(RNG_SEED)
    theta = np.random.default_rng(RNG_SEED).normal(scale=0.3, size=len(bl.zero_theta(data)))
    _, grad = bl.penalised_negative_loglik(theta, data, bl.RIDGE_ALPHA)
    numeric = np.zeros_like(theta)
    for i in range(len(theta)):
        step = np.zeros_like(theta)
        step[i] = FD_STEP
        plus = bl.penalised_negative_loglik(theta + step, data, bl.RIDGE_ALPHA)[0]
        minus = bl.penalised_negative_loglik(theta - step, data, bl.RIDGE_ALPHA)[0]
        numeric[i] = (plus - minus) / (2 * FD_STEP)
    np.testing.assert_allclose(grad, numeric, atol=FD_TOL * max(1.0, np.abs(grad).max()))


def test_hessian_matches_finite_differences_of_gradient() -> None:
    data = synthetic_data(RNG_SEED)
    theta = np.random.default_rng(RNG_SEED).normal(scale=0.3, size=len(bl.zero_theta(data)))
    hess = bl.penalised_hessian(theta, data, bl.RIDGE_ALPHA)
    for i in list(range(0, len(theta), 17)) + [len(theta) - 1]:
        step = np.zeros_like(theta)
        step[i] = FD_STEP
        numeric = (bl.penalised_negative_loglik(theta + step, data, bl.RIDGE_ALPHA)[1]
                   - bl.penalised_negative_loglik(theta - step, data, bl.RIDGE_ALPHA)[1]) / (2 * FD_STEP)
        np.testing.assert_allclose(hess[i], numeric, atol=FD_TOL * max(1.0, np.abs(hess[i]).max()))


def test_fit_recovers_known_beta_within_two_standard_errors() -> None:
    data = synthetic_data(RNG_SEED)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    se = np.sqrt(np.diag(fit.covariance))
    assert (np.abs(fit.beta - TRUE_BETA) < SE_MULTIPLE * se).all(), (fit.beta, se)


def test_resampling_keeps_ballots_whole_and_within_edition() -> None:
    data = synthetic_data(RNG_SEED)[0]
    resampled = bl.resample_ballots(data, np.random.default_rng(RNG_SEED))
    original_rows = {(tuple(i.ravel()), tuple(p)) for i, p in zip(data.indicators, data.picks)}
    assert all((tuple(i.ravel()), tuple(p)) in original_rows for i, p in zip(resampled.indicators, resampled.picks))
    assert len(resampled.juror_keys) == len(data.juror_keys)


# ---- real jury, simulation and 2026 target guard -------------------------------------------------------------------

from src.model import baselines, cv, data as study_data, forecast_ballots, inference, jurors, link, simulate  # noqa: E402

EXPECTED_2026_CONFEDERATIONS = {"UEFA": 41, "CAF": 20, "AFC": 17, "CONCACAF": 11, "CONMEBOL": 10, "OFC": 1}
POINTS_5_PICK, POINTS_10_PICK = 16, 67
GUARDED_TARGET_COLUMNS = ("bdo_points", "bdo_rank", "vote_share")
BALLOT_MODULES = ("ballots.py", "ballots_report.py", "link.py", "forecast_ballots.py", "validation_ballots.py", "figures_ballots.py")
SIM_SIZE = 200


def synthetic_inputs(n_jurors: int, n_players: int, rng: np.random.Generator) -> jurors.SimInputs:
    nation = rng.random((n_jurors, n_players)) < HOME_PROB
    confed = (rng.random((n_jurors, n_players)) < CONFED_PROB) & ~nation
    return jurors.SimInputs(nation_match=nation, confed_match=confed, league_match=rng.random((n_jurors, n_players)) < LEAGUE_PROB,
                            n_jurors=n_jurors)


@needs_jury_data
def test_real_jury_2026_has_the_published_confederation_counts() -> None:
    jury = jurors.real_jury(ROOT, 2026)
    counts = pd.Series([j.confederation for j in jury]).value_counts().to_dict()
    assert len(jury) == 100
    assert counts == EXPECTED_2026_CONFEDERATIONS


@needs_jury_data
@pytest.mark.parametrize("edition", bl.BALLOT_EDITIONS)
def test_real_jury_for_past_editions_is_the_voting_nations(edition: int) -> None:
    assert len(jurors.real_jury(ROOT, edition)) == bl.BALLOTS_BY_EDITION[edition]


@needs_jury_data
def test_real_sim_inputs_keep_confed_exclusive_of_home() -> None:
    inputs = jurors.build_real_sim_inputs(jurors.real_jury(ROOT, 2026), jurors.player_attributes(ROOT, 2026))
    assert not (inputs.nation_match & inputs.confed_match).any()
    assert inputs.nation_match.any() and inputs.confed_match.any() and inputs.league_match.any()


@pytest.mark.parametrize("scheme, per_juror", [(bl.SCHEME_5, POINTS_5_PICK), (bl.SCHEME_10, POINTS_10_PICK)])
def test_scheme_awards_fixed_points_per_juror(scheme: tuple[int, ...], per_juror: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    inputs = synthetic_inputs(N_JURORS, N_PLAYERS, rng)
    config = simulate.BallotSimConfig(k=1.0, sigma_shared=1.0, scheme=scheme, n_dropped=0)
    result = simulate.simulate_ballots(rng.normal(size=(SIM_SIZE, N_PLAYERS)), np.tile(TRUE_BETA, (SIM_SIZE, 1)), inputs, config, rng)
    np.testing.assert_array_equal(result.points.sum(axis=1), np.full(SIM_SIZE, N_JURORS * per_juror))
    assert result.points_per_ballot == per_juror


def test_dropping_jurors_removes_their_points() -> None:
    rng = np.random.default_rng(RNG_SEED)
    inputs = synthetic_inputs(N_JURORS, N_PLAYERS, rng)
    config = simulate.BallotSimConfig(k=1.0, sigma_shared=1.0, scheme=bl.SCHEME_10, n_dropped=5)
    result = simulate.simulate_ballots(rng.normal(size=(SIM_SIZE, N_PLAYERS)), np.tile(TRUE_BETA, (SIM_SIZE, 1)), inputs, config, rng)
    np.testing.assert_array_equal(result.points.sum(axis=1), np.full(SIM_SIZE, 95 * POINTS_10_PICK))
    assert result.n_jurors == 95


def test_simulated_home_bonus_matches_the_fitted_model() -> None:
    """Ballots simulated with a known beta and no shared shock are recovered by the ballot PL fit (simulator and model agree)."""
    rng = np.random.default_rng(RNG_SEED)
    alpha = rng.normal(size=N_PLAYERS)
    inputs = synthetic_inputs(2000, N_PLAYERS, rng)
    config = simulate.BallotSimConfig(k=1.0, sigma_shared=0.0, scheme=bl.SCHEME_10, n_dropped=0)
    tally = simulate.simulate_ballots(np.tile(alpha, (1, 1)), TRUE_BETA[None, :], inputs, config, rng)
    assert tally.home_first_share[0] > HOME_PROB  # a home bonus lifts the compatriot-first share above the unbiased baseline


@pytest.mark.parametrize("filename", BALLOT_MODULES)
def test_ballot_modules_never_name_the_target_columns(filename: str) -> None:
    source = (ROOT / "src" / "model" / filename).read_text()
    assert not any(column in source for column in GUARDED_TARGET_COLUMNS)


def test_2026_forecast_never_reads_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    study = study_data.load_study_data(ROOT)
    assert study.targets.loc[study.frame_t1.edition == study_data.FORECAST_EDITION].isna().all().all()

    original = baselines.points_of

    def guarded(study_arg: study_data.StudyData, edition: int) -> np.ndarray:
        if edition == study_data.FORECAST_EDITION:
            raise AssertionError(f"targets read for edition {edition}")
        return original(study_arg, edition)

    monkeypatch.setattr(baselines, "points_of", guarded)
    fit = inference.final_fit(study, cv.PL_SPECS["M1"])
    params = forecast_ballots.BallotParams(beta=TRUE_BETA, beta_covariance=np.eye(3) * 0.01,
                                           link_fit=link.LinkFit("T1", bl.BALLOT_EDITIONS, 1.5, 1.5, 0.5, 120))
    if (ROOT / bl.JURY_DIR / bl.VOTES_FILE).exists():
        table = forecast_ballots.forecast_table_ballots(study, fit, forecast_ballots.inputs_2026(ROOT, study), params, 0, SIM_SIZE, RNG_SEED)
        assert abs(table.p_win.sum() - 1.0) < 1e-9
