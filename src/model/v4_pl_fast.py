"""Vectorised Plackett-Luce loss and gradient for the penalised v4 fits.

Same model as `pl.py` (tie groups by Breslow, zero-point nominees only in the denominator), but the suffix log-sum-exp is
computed with one accumulate instead of a Python loop over groups, which is what makes the lasso and stability-selection
searches affordable. `pl.py` stays the reference; the tests check both agree.
"""

from dataclasses import dataclass

import numpy as np

from src.model.pl import Ranking


@dataclass(frozen=True)
class SortedEdition:
    x: np.ndarray  # rows in ranking order
    starts: np.ndarray
    sizes: np.ndarray
    group_of_row: np.ndarray  # index of the tie group each ordered row belongs to, -1 for zero-point rows


def sort_edition(x: np.ndarray, ranking: Ranking) -> SortedEdition:
    n = len(ranking.order)
    group_of_row = np.full(n, -1)
    for g, (start, size) in enumerate(zip(ranking.group_starts, ranking.group_sizes)):
        group_of_row[start:start + size] = g
    return SortedEdition(x=x[ranking.order], starts=np.array(ranking.group_starts), sizes=np.array(ranking.group_sizes),
                         group_of_row=group_of_row)


def loglik_and_gradient(edition: SortedEdition, beta: np.ndarray) -> tuple[float, np.ndarray]:
    u = edition.x @ beta
    log_suffix = np.logaddexp.accumulate(u[::-1])[::-1]
    voted = edition.group_of_row >= 0
    value = float(u[voted].sum() - (edition.sizes * log_suffix[edition.starts]).sum())
    at_or_after_start = np.arange(len(u))[None, :] >= edition.starts[:, None]
    weights = np.exp(np.where(at_or_after_start, u[None, :] - log_suffix[edition.starts][:, None], -np.inf))
    coefficient = edition.sizes @ weights
    grad = edition.x[voted].sum(axis=0) - coefficient @ edition.x
    return value, grad


def penalised_negative_loglik(beta: np.ndarray, editions: list[SortedEdition], l2: float) -> tuple[float, np.ndarray]:
    value = l2 * float(beta @ beta)
    grad = 2.0 * l2 * beta
    for edition in editions:
        ll, g = loglik_and_gradient(edition, beta)
        value -= ll
        grad = grad - g
    return value, grad
