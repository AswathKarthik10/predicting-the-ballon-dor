"""Extra comparison models for the ten-edition study (experiments spec section 4): L1, O1 and D1.

Like B0-B2 and X1 they return raw scores; cv.py turns them into Plackett-Luce utilities with a fitted temperature."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from statsmodels.miscmodels.ordinal_model import OrderedModel

from src.model.baselines import SEED, SHARE_FLOOR, m1_matrices, points_of, relevance_labels, vote_share_of
from src.model.data import StudyData

LGBM_PARAMS: dict[str, object] = {"objective": "lambdarank", "num_leaves": 4, "n_estimators": 150, "learning_rate": 0.05,
                                  "min_child_samples": 10, "reg_lambda": 5, "random_state": SEED, "verbose": -1,
                                  "n_jobs": 1, "deterministic": True, "force_row_wise": True}
TREE_PARAMS: dict[str, object] = {"max_depth": 3, "min_samples_leaf": 8, "random_state": SEED}
ORDINAL_MAX_ITER = 500
# Rank bands: winner, 2nd-3rd, 4th-10th, 11th or lower with votes, no votes
BAND_UPPER_RANKS: tuple[int, ...] = (1, 3, 10)
N_BANDS = len(BAND_UPPER_RANKS) + 2
NO_VOTE_BAND = N_BANDS - 1


def rank_bands(points: np.ndarray) -> np.ndarray:
    """Band index per player from competition ranks (ties share the best rank); zero-point players get the last band."""
    ranks = 1 + (points[None, :] > points[:, None]).sum(axis=1)
    bands = np.searchsorted(np.array(BAND_UPPER_RANKS), ranks, side="left")
    return np.where(points > 0, bands, NO_VOTE_BAND)


def lgbm_scores(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """L1: LightGBM LambdaMART with editions as query groups. Needs lightgbm and libomp."""
    from lightgbm import LGBMRanker

    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([relevance_labels(points_of(study, ed)) for ed in train_editions])
    groups = [len(mats[ed]) for ed in train_editions]
    model = LGBMRanker(**LGBM_PARAMS).fit(x_train, y_train, group=groups)
    return {ed: model.predict(mats[ed]) for ed in apply_editions}


def ordinal_scores(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """O1: ordered logit on the rank band; score = minus the expected band (a better finish scores higher)."""
    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([rank_bands(points_of(study, ed)) for ed in train_editions])
    fitted = OrderedModel(y_train, x_train, distr="logit").fit(method="bfgs", disp=False, maxiter=ORDINAL_MAX_ITER)
    bands = np.arange(N_BANDS)
    return {ed: -(fitted.model.predict(fitted.params, exog=mats[ed]) @ bands) for ed in apply_editions}


def tree_scores(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """D1: shallow regression tree on vote share; score = log of the (floored) predicted share."""
    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([vote_share_of(study, ed) for ed in train_editions])
    model = DecisionTreeRegressor(**TREE_PARAMS).fit(x_train, y_train)
    return {ed: np.log(np.maximum(model.predict(mats[ed]), SHARE_FLOOR)) for ed in apply_editions}
