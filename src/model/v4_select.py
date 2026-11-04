"""Experiment 4 feature selectors FS0-FS5 (spec section 4).

Every selector sees only the training editions it is given: preprocessed matrices and their points. It returns the sorted
indices of the selected columns.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from lightgbm import LGBMRanker
from scipy.stats import rankdata

from src.model.baselines import relevance_labels
from src.model.pl import Ranking, build_ranking
from src.model.v4_pl_penalised import fit_pl_lasso, sort_editions

SEED = 7
STABILITY_RESAMPLES = 100
STABILITY_FRACTION = 0.5
STABILITY_PENALTY = 1.0
CORRELATION_PRUNE_THRESHOLD = 0.9
SELECTOR_TREE_PARAMS = dict(n_estimators=100, learning_rate=0.05, num_leaves=4, min_child_samples=5, reg_lambda=5.0,
                            importance_type="gain", random_state=SEED, n_jobs=1, verbose=-1, deterministic=True,
                            force_row_wise=True)
NOT_USED = 0
TOP_K_GRID = (5, 10, 15, 25)
LAMBDA_FS2 = tuple(float(v) for v in np.logspace(-3, 1, 13))
PI_GRID = (0.6, 0.75, 0.9)


class SelectorKind(str, Enum):
    NONE = "FS0"
    FILTER = "FS1"
    LASSO = "FS2"
    STABILITY = "FS3"
    TREE = "FS4"
    PRUNED_FILTER = "FS5"


@dataclass(frozen=True)
class SelectorSpec:
    kind: SelectorKind
    k: int
    lam: float
    pi: float

    @property
    def name(self) -> str:
        return f"{self.kind.value}(k={self.k},lam={self.lam},pi={self.pi})"


@dataclass(frozen=True)
class TrainSet:
    """Preprocessed training editions: one matrix, one points vector and one ranking per edition."""
    editions: tuple[int, ...]
    xs: list[np.ndarray]
    points: list[np.ndarray]
    rankings: list[Ranking]


def make_train_set(editions: tuple[int, ...], xs: list[np.ndarray], points: list[np.ndarray]) -> TrainSet:
    return TrainSet(editions=editions, xs=xs, points=points, rankings=[build_ranking(p) for p in points])


def standardised_ranks(x: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 0, x)
    centred = ranks - ranks.mean(axis=0)
    scale = np.sqrt((centred ** 2).sum(axis=0))
    return np.where(scale > 0, centred / np.where(scale > 0, scale, 1.0), 0.0)


def filter_scores(train: TrainSet) -> np.ndarray:
    """Absolute within-edition Spearman with points, averaged over training editions (the FS1 ranking score)."""
    per_edition = [np.abs(standardised_ranks(x).T @ standardised_ranks(p[:, None])[:, 0]) for x, p in zip(train.xs, train.points)]
    return np.mean(per_edition, axis=0)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:k])


def select_filter(train: TrainSet, k: int) -> np.ndarray:
    return top_k(filter_scores(train), k)


def pairwise_spearman(train: TrainSet) -> np.ndarray:
    """Within-edition column-by-column Spearman, averaged over training editions."""
    per_edition = [standardised_ranks(x).T @ standardised_ranks(x) for x in train.xs]
    return np.mean(per_edition, axis=0)


def select_pruned_filter(train: TrainSet, k: int) -> np.ndarray:
    scores = filter_scores(train)
    correlation = np.abs(pairwise_spearman(train))
    kept: list[int] = []
    for col in np.argsort(-scores, kind="stable"):
        if all(correlation[col, other] <= CORRELATION_PRUNE_THRESHOLD for other in kept):
            kept.append(int(col))
    return np.sort(np.array(kept[:k], dtype=int))


def select_lasso(train: TrainSet, lam: float) -> np.ndarray:
    fit = fit_pl_lasso(sort_editions(train.xs, train.rankings), lam, np.zeros(train.xs[0].shape[1]))
    return np.flatnonzero(fit.beta)


def half_sample_train(train: TrainSet, rng: np.random.Generator) -> TrainSet:
    keep = [np.sort(rng.choice(len(p), size=int(len(p) * STABILITY_FRACTION), replace=False)) for p in train.points]
    return make_train_set(train.editions, [x[rows] for x, rows in zip(train.xs, keep)], [p[rows] for p, rows in zip(train.points, keep)])


def stability_frequencies(train: TrainSet) -> np.ndarray:
    """Share of half-samples (players resampled within edition) in which the lasso keeps each column."""
    rng = np.random.default_rng(SEED)
    n_columns = train.xs[0].shape[1]
    counts = np.zeros(n_columns)
    for _ in range(STABILITY_RESAMPLES):
        sample = half_sample_train(train, rng)
        fit = fit_pl_lasso(sort_editions(sample.xs, sample.rankings), STABILITY_PENALTY, np.zeros(n_columns))
        counts += fit.beta != 0
    return counts / STABILITY_RESAMPLES


def select_stability(frequencies: np.ndarray, pi: float) -> np.ndarray:
    return np.flatnonzero(frequencies >= pi)


def select_tree(train: TrainSet, k: int) -> np.ndarray:
    x = np.vstack(train.xs)
    labels = np.concatenate([relevance_labels(p) for p in train.points])
    model = LGBMRanker(**SELECTOR_TREE_PARAMS).fit(x, labels, group=[len(p) for p in train.points])
    gain = model.booster_.feature_importance(importance_type="gain")
    chosen = top_k(gain, k)
    return chosen[gain[chosen] > 0]


def select_features(train: TrainSet, spec: SelectorSpec, frequencies: np.ndarray) -> np.ndarray:
    """`frequencies` are the stability-selection frequencies for this training set (cached by the caller; FS3 only)."""
    n_columns = train.xs[0].shape[1]
    if spec.kind is SelectorKind.NONE:
        return np.arange(n_columns)
    if spec.kind is SelectorKind.FILTER:
        return select_filter(train, spec.k)
    if spec.kind is SelectorKind.LASSO:
        return select_lasso(train, spec.lam)
    if spec.kind is SelectorKind.STABILITY:
        return select_stability(frequencies, spec.pi)
    if spec.kind is SelectorKind.TREE:
        return select_tree(train, spec.k)
    return select_pruned_filter(train, spec.k)
