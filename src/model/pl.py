"""Plackett-Luce ranking model with a ridge penalty (spec section 4.1).

Voted players are ordered by points; tied voted players form one group (Breslow approximation); zero-point
nominees appear only in the "not yet chosen" denominator and are never ordered among themselves.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

MAX_ITER = 500
LOG_TEMPERATURE_BOUNDS = (-6.0, 6.0)


@dataclass(frozen=True)
class Ranking:
    order: np.ndarray  # player indices sorted by points, highest first
    group_starts: tuple[int, ...]  # position in `order` where each tie group starts
    group_sizes: tuple[int, ...]
    n_voted: int


def build_ranking(points: np.ndarray) -> Ranking:
    """Build the group structure from points only. Zero-point players are excluded from the groups."""
    if np.isnan(points).any():
        raise ValueError("points contain NaN")
    order = np.argsort(-points, kind="stable")
    sorted_points = points[order]
    starts: list[int] = []
    sizes: list[int] = []
    position = 0
    while position < len(sorted_points) and sorted_points[position] > 0:
        size = int((sorted_points == sorted_points[position]).sum())
        starts.append(position)
        sizes.append(size)
        position += size
    return Ranking(order=order, group_starts=tuple(starts), group_sizes=tuple(sizes), n_voted=int(sum(sizes)))


def edition_loglik(utility: np.ndarray, ranking: Ranking) -> float:
    u = utility[ranking.order]
    total = 0.0
    for start, size in zip(ranking.group_starts, ranking.group_sizes):
        total += u[start:start + size].sum() - size * logsumexp(u[start:])
    return float(total)


def edition_loglik_per_voted(utility: np.ndarray, ranking: Ranking) -> float:
    return edition_loglik(utility, ranking) / ranking.n_voted


def edition_gradient(x: np.ndarray, beta: np.ndarray, ranking: Ranking) -> np.ndarray:
    xo = x[ranking.order]
    u = xo @ beta
    grad = np.zeros_like(beta)
    for start, size in zip(ranking.group_starts, ranking.group_sizes):
        weights = np.exp(u[start:] - logsumexp(u[start:]))
        grad += xo[start:start + size].sum(axis=0) - size * (weights @ xo[start:])
    return grad


def edition_hessian_of_loglik(x: np.ndarray, beta: np.ndarray, ranking: Ranking) -> np.ndarray:
    xo = x[ranking.order]
    u = xo @ beta
    hess = np.zeros((len(beta), len(beta)))
    for start, size in zip(ranking.group_starts, ranking.group_sizes):
        weights = np.exp(u[start:] - logsumexp(u[start:]))
        mean = weights @ xo[start:]
        second = (xo[start:] * weights[:, None]).T @ xo[start:]
        hess -= size * (second - np.outer(mean, mean))
    return hess


def penalised_negative_loglik(beta: np.ndarray, xs: list[np.ndarray], rankings: list[Ranking], lam: float) -> tuple[float, np.ndarray]:
    value = lam * float(beta @ beta)
    grad = 2.0 * lam * beta
    for x, ranking in zip(xs, rankings):
        value -= edition_loglik(x @ beta, ranking)
        grad = grad - edition_gradient(x, beta, ranking)
    return value, grad


def fit_pl(xs: list[np.ndarray], rankings: list[Ranking], lam: float) -> np.ndarray:
    """Maximise sum of edition log-likelihoods minus lam * ||beta||^2 (each edition counts equally)."""
    k = xs[0].shape[1]
    result = minimize(penalised_negative_loglik, np.zeros(k), args=(xs, rankings, lam), jac=True,
                      method="L-BFGS-B", options={"maxiter": MAX_ITER})
    if not result.success:
        raise RuntimeError(f"PL fit did not converge (lam={lam}): {result.message}")
    return result.x


def penalised_hessian(xs: list[np.ndarray], rankings: list[Ranking], beta: np.ndarray, lam: float) -> np.ndarray:
    """Hessian of the penalised NEGATIVE log-likelihood, used for the Laplace approximation."""
    hess = 2.0 * lam * np.eye(len(beta))
    for x, ranking in zip(xs, rankings):
        hess -= edition_hessian_of_loglik(x, beta, ranking)
    return hess


def fit_temperature(scores: list[np.ndarray], rankings: list[Ranking]) -> float:
    """Maximum-likelihood temperature T so that utility = score / T, over the given editions."""
    def negative(log_t: float) -> float:
        t = np.exp(log_t)
        return -sum(edition_loglik(s / t, r) for s, r in zip(scores, rankings))

    result = minimize_scalar(negative, bounds=LOG_TEMPERATURE_BOUNDS, method="bounded")
    return float(np.exp(result.x))
