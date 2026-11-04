"""Lasso and elastic-net Plackett-Luce (Experiment 4, spec sections 4-5).

Minimises  -sum_editions loglik(beta) + l1 * ||beta||_1 + l2 * ||beta||^2  by accelerated proximal gradient (FISTA with
backtracking and adaptive restart). Elastic net uses l1 = lam * mix and l2 = lam * (1 - mix); lasso is mix = 1.
Convergence is judged on the objective, as in glmnet-style solvers: at the weak end of the penalty grid the problem is
close to separable and the coefficients keep drifting while the objective is flat. A fit that hits the iteration cap is
returned with `converged=False` so the search log can report it.
"""

from dataclasses import dataclass

import numpy as np

from src.model.pl import Ranking
from src.model.v4_pl_fast import SortedEdition, penalised_negative_loglik, sort_edition

MAX_ITERATIONS = 20000
OBJECTIVE_TOLERANCE = 1e-8
INITIAL_LIPSCHITZ = 1.0
BACKTRACK_FACTOR = 2.0
MAX_BACKTRACKS = 60
LASSO_MIX = 1.0


@dataclass(frozen=True)
class PenalisedFit:
    beta: np.ndarray
    converged: bool


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def sort_editions(xs: list[np.ndarray], rankings: list[Ranking]) -> list[SortedEdition]:
    return [sort_edition(x, ranking) for x, ranking in zip(xs, rankings)]


def fit_elastic_net_sorted(editions: list[SortedEdition], l1: float, l2: float, start: np.ndarray) -> PenalisedFit:
    beta = start.copy()
    momentum_point = start.copy()
    step_scale = 1.0
    lipschitz = INITIAL_LIPSCHITZ
    previous_objective = np.inf
    for _ in range(MAX_ITERATIONS):
        value, grad = penalised_negative_loglik(momentum_point, editions, l2)
        for _ in range(MAX_BACKTRACKS):
            candidate = soft_threshold(momentum_point - grad / lipschitz, l1 / lipschitz)
            diff = candidate - momentum_point
            candidate_value, _ = penalised_negative_loglik(candidate, editions, l2)
            if candidate_value <= value + grad @ diff + 0.5 * lipschitz * diff @ diff + 1e-12:
                break
            lipschitz *= BACKTRACK_FACTOR
        else:
            raise RuntimeError(f"PL elastic-net backtracking failed (l1={l1}, l2={l2})")
        if (momentum_point - candidate) @ (candidate - beta) > 0:  # adaptive restart of the momentum
            step_scale = 1.0
        next_scale = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * step_scale ** 2))
        momentum_point = candidate + ((step_scale - 1.0) / next_scale) * (candidate - beta)
        beta, step_scale = candidate, next_scale
        objective = candidate_value + l1 * float(np.abs(candidate).sum())
        if abs(previous_objective - objective) < OBJECTIVE_TOLERANCE * max(1.0, abs(objective)):
            return PenalisedFit(beta=beta, converged=True)
        previous_objective = objective
    return PenalisedFit(beta=beta, converged=False)


def fit_pl_lasso(editions: list[SortedEdition], lam: float, start: np.ndarray) -> PenalisedFit:
    return fit_elastic_net_sorted(editions, lam * LASSO_MIX, 0.0, start)


def fit_pl_mixed(editions: list[SortedEdition], lam: float, mix: float, start: np.ndarray) -> PenalisedFit:
    return fit_elastic_net_sorted(editions, lam * mix, lam * (1.0 - mix), start)
