import numpy as np
import pytest
from scipy.special import logsumexp

from src.model.pl import (build_ranking, edition_gradient, edition_hessian_of_loglik, edition_loglik, fit_pl,
                          fit_temperature, penalised_hessian, penalised_negative_loglik)

RNG_SEED = 7
FD_STEP = 1e-6
FD_TOL = 1e-5


def synthetic_edition(rng: np.random.Generator, n_players: int, beta: np.ndarray, n_voted: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(n_players, len(beta)))
    gumbel = rng.gumbel(size=n_players)
    order = np.argsort(-(x @ beta + gumbel))
    points = np.zeros(n_players)
    for place, player in enumerate(order[:n_voted]):
        points[player] = n_voted - place
    return x, points


def test_ranking_excludes_zero_point_players_and_groups_ties() -> None:
    points = np.array([0.0, 5.0, 3.0, 3.0, 0.0, 1.0])
    ranking = build_ranking(points)
    assert ranking.n_voted == 4
    assert ranking.group_sizes == (1, 2, 1)
    assert ranking.group_starts == (0, 1, 3)
    assert set(ranking.order[:4]) == {1, 2, 3, 5}


def test_loglik_matches_hand_calculation_with_zero_points_and_ties() -> None:
    u = np.array([0.5, 1.0, -0.2, 0.3, 0.9])
    points = np.array([0.0, 6.0, 2.0, 2.0, 0.0])  # winner 1; tie between 2 and 3; players 0 and 4 got nothing
    expected = u[1] - logsumexp(u)  # winner chosen from all five
    remaining = u[[0, 2, 3, 4]]  # tie group chosen from everyone except the winner (zero-point players stay in)
    expected += u[2] + u[3] - 2 * logsumexp(remaining)
    assert edition_loglik(u, build_ranking(points)) == pytest.approx(expected, abs=1e-12)


def test_zero_point_players_are_never_ordered_among_themselves() -> None:
    rng = np.random.default_rng(RNG_SEED)
    x = rng.normal(size=(8, 3))
    beta = np.array([0.4, -0.3, 0.8])
    points = np.array([5.0, 4.0, 3.0, 0, 0, 0, 0, 0])
    base = edition_loglik(x @ beta, build_ranking(points))
    swapped_x = x.copy()
    swapped_x[[3, 7]] = swapped_x[[7, 3]]  # swapping two zero-point players cannot change the likelihood
    assert edition_loglik(swapped_x @ beta, build_ranking(points)) == pytest.approx(base, abs=1e-12)
    # the likelihood also ignores any point label order beyond the voted players
    assert build_ranking(points).n_voted == 3


def test_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(RNG_SEED)
    beta = rng.normal(size=4)
    xs, rankings = [], []
    for _ in range(3):
        x, points = synthetic_edition(rng, 30, np.array([0.8, -0.5, 0.3, 0.0]), 10)
        points[np.argsort(-points)[3:5]] = points[np.argsort(-points)[3]]  # inject a tie
        xs.append(x)
        rankings.append(build_ranking(points))
    lam = 0.7
    _, analytic = penalised_negative_loglik(beta, xs, rankings, lam)
    numeric = np.zeros_like(beta)
    for k in range(len(beta)):
        step = np.zeros_like(beta)
        step[k] = FD_STEP
        up, _ = penalised_negative_loglik(beta + step, xs, rankings, lam)
        down, _ = penalised_negative_loglik(beta - step, xs, rankings, lam)
        numeric[k] = (up - down) / (2 * FD_STEP)
    assert np.max(np.abs(analytic - numeric)) < FD_TOL


def test_hessian_matches_finite_differences_of_gradient() -> None:
    rng = np.random.default_rng(RNG_SEED)
    beta = rng.normal(size=3)
    x, points = synthetic_edition(rng, 30, np.array([0.6, -0.4, 0.2]), 10)
    ranking = build_ranking(points)
    analytic = edition_hessian_of_loglik(x, beta, ranking)
    numeric = np.zeros((3, 3))
    for k in range(3):
        step = np.zeros(3)
        step[k] = FD_STEP
        numeric[:, k] = (edition_gradient(x, beta + step, ranking) - edition_gradient(x, beta - step, ranking)) / (2 * FD_STEP)
    assert np.max(np.abs(analytic - numeric)) < 1e-4
    full = penalised_hessian([x], [ranking], beta, 0.5)
    assert np.all(np.linalg.eigvalsh(full) > 0)


def test_recovers_known_beta_on_synthetic_rankings() -> None:
    rng = np.random.default_rng(RNG_SEED)
    true_beta = np.array([1.0, -0.6, 0.4, 0.0, 0.8])
    xs, rankings = [], []
    for _ in range(300):
        x, points = synthetic_edition(rng, 30, true_beta, 10)
        xs.append(x)
        rankings.append(build_ranking(points))
    estimate = fit_pl(xs, rankings, 1e-6)
    assert np.max(np.abs(estimate - true_beta)) < 0.1


def test_ridge_shrinks_towards_zero() -> None:
    rng = np.random.default_rng(RNG_SEED)
    true_beta = np.array([1.0, -0.6, 0.4])
    xs, rankings = [], []
    for _ in range(10):
        x, points = synthetic_edition(rng, 30, true_beta, 10)
        xs.append(x)
        rankings.append(build_ranking(points))
    norms = [np.linalg.norm(fit_pl(xs, rankings, lam)) for lam in (0.01, 1.0, 100.0)]
    assert norms[0] > norms[1] > norms[2]
    assert norms[2] < 0.5 * norms[0]


def test_zero_features_give_the_uniform_likelihood() -> None:
    points = np.array([6.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    ranking = build_ranking(points)
    n = len(points)
    expected = -sum(np.log(n - k) for k in range(5))
    assert edition_loglik(np.zeros(n), ranking) == pytest.approx(expected, abs=1e-12)


def test_temperature_recovers_scale() -> None:
    rng = np.random.default_rng(RNG_SEED)
    scores, rankings = [], []
    for _ in range(200):
        s = rng.normal(size=30) * 2.0
        order = np.argsort(-(s / 2.0 + rng.gumbel(size=30)))  # data generated with T = 2
        points = np.zeros(30)
        for place, player in enumerate(order[:10]):
            points[player] = 10 - place
        scores.append(s)
        rankings.append(build_ranking(points))
    assert fit_temperature(scores, rankings) == pytest.approx(2.0, rel=0.1)
