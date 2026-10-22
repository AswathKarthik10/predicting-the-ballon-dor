from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import jurors, simulate
from src.model.jurors import BiasParams, SimInputs

ROOT = Path(__file__).resolve().parents[1]
NO_BIAS = BiasParams(b_home=0.0, b_confed=0.0, b_league=0.0)


@pytest.fixture(scope="module")
def juror_inputs() -> pd.DataFrame:
    return jurors.load_juror_inputs(ROOT)


def tiny_inputs(n_jurors: int, n_players: int) -> SimInputs:
    off = np.zeros((n_jurors, n_players), dtype=bool)
    return SimInputs(nation_match=off.copy(), confed_match=off.copy(), league_match=off.copy(), n_jurors=n_jurors)


@pytest.mark.parametrize("edition", [2022, 2023, 2024, 2025, 2026])
def test_panel_reproduces_the_data_columns_for_every_edition(juror_inputs: pd.DataFrame, edition: int) -> None:
    players = jurors.edition_players(juror_inputs, edition)
    sim_inputs = jurors.build_sim_inputs(players, jurors.JURORS_BY_EDITION[edition])
    assert sim_inputs.n_jurors == jurors.JURORS_BY_EDITION[edition]
    np.testing.assert_array_equal(sim_inputs.nation_match.sum(axis=0), players["jurors_same_nation"].to_numpy())
    np.testing.assert_array_equal(sim_inputs.confed_match.sum(axis=0), players["jurors_same_confed"].to_numpy())
    np.testing.assert_array_equal(sim_inputs.league_match.sum(axis=0), players["jurors_league_country"].to_numpy())


def test_2026_panel_composition(juror_inputs: pd.DataFrame) -> None:
    players = jurors.edition_players(juror_inputs, 2026)
    panel = jurors.build_jurors(players, 100)
    assert len(panel) == 100
    by_confed = pd.Series([j.confederation for j in panel]).value_counts().to_dict()
    assert by_confed[jurors.UEFA] == 41 and by_confed[jurors.CAF] == 20 and by_confed[jurors.CONMEBOL] == 10
    assert by_confed[jurors.CONCACAF] == 11 and by_confed[jurors.AFC] == 1 and by_confed[jurors.UNASSIGNED] == 17


def test_unknown_nationality_or_league_raises() -> None:
    with pytest.raises(KeyError):
        jurors.player_confederation("Atlantis")
    with pytest.raises(KeyError):
        jurors.league_country(999)


def test_points_per_simulation_equal_jurors_times_67() -> None:
    rng = np.random.default_rng(7)
    strengths = rng.normal(size=(300, 30))
    config = simulate.SimConfig(tau_shared=1.0, tau_juror=1.0, bias=NO_BIAS)
    result = simulate.simulate(strengths, tiny_inputs(100, 30), config, rng)
    np.testing.assert_allclose(result.points.sum(axis=1), 100 * 67)
    assert (np.sort(result.ranks, axis=1) == np.arange(1, 31)).all()


def test_no_noise_no_bias_reproduces_the_strength_ordering() -> None:
    strengths = np.tile(np.linspace(2, -2, 12), (50, 1))
    config = simulate.SimConfig(tau_shared=0.0, tau_juror=1e-9, bias=NO_BIAS)
    result = simulate.simulate(strengths, tiny_inputs(20, 12), config, np.random.default_rng(7))
    assert (result.ranks[:, :10] == np.arange(1, 11)).all()  # the ten ballot places follow the strengths exactly
    assert (np.sort(result.ranks[:, 10:], axis=1) == [11, 12]).all()  # zero-point players share the last places
    summary = simulate.summarise(result)
    assert summary["p_win"].iloc[0] == 1.0 and summary["p_win"].sum() == pytest.approx(1.0)


def test_win_probabilities_sum_to_one_and_are_reproducible() -> None:
    strengths = np.random.default_rng(1).normal(size=(400, 30))
    config = simulate.SimConfig(tau_shared=1.0, tau_juror=1.0, bias=NO_BIAS)
    a = simulate.simulate(strengths, tiny_inputs(100, 30), config, np.random.default_rng(7))
    b = simulate.simulate(strengths, tiny_inputs(100, 30), config, np.random.default_rng(7))
    np.testing.assert_array_equal(a.ranks, b.ranks)
    assert simulate.summarise(a)["p_win"].sum() == pytest.approx(1.0)
    assert simulate.summarise(a)["expected_vote_share"].sum() == pytest.approx(1.0)


def test_home_bias_moves_a_juror_to_their_compatriot() -> None:
    inputs = tiny_inputs(1, 12)
    inputs.nation_match[0, 7] = True
    config = simulate.SimConfig(tau_shared=0.0, tau_juror=1e-9, bias=BiasParams(100.0, 0.0, 0.0))
    result = simulate.simulate(np.zeros((5, 12)), inputs, config, np.random.default_rng(7))
    assert (result.points[:, 7] == 15).all()
    assert (result.ranks[:, 7] == 1).all()


def test_tie_break_uses_first_then_second_place_votes() -> None:
    points = np.array([[30.0, 30.0, 30.0, 10.0]])
    first = np.array([[1.0, 2.0, 2.0, 0.0]])
    second = np.array([[5.0, 1.0, 3.0, 0.0]])
    ranks = simulate.rank_players(points, first, second, np.zeros((1, 4)))
    assert ranks.tolist() == [[3, 2, 1, 4]]  # player 2 (2 firsts, 3 seconds) > player 1 (2, 1) > player 0 (1 first)


def test_concentration_statistics_on_a_known_vector() -> None:
    stats = simulate.concentration_stats(np.array([50.0, 30.0, 20.0, 0.0]))
    assert stats["winner_share"] == pytest.approx(0.5)
    assert stats["n_with_points"] == 3.0
    assert stats["herfindahl"] == pytest.approx(0.25 + 0.09 + 0.04)
