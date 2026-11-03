"""Comparison models B0, B1, B2 and X1 (spec section 4). Each returns raw scores for the apply editions; the
temperature that turns scores into Plackett-Luce utilities is fitted in cv.py."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from src.model.data import StudyData

SEED = 7
SHARE_FLOOR = 1e-4
PROBABILITY_FLOOR = 1e-6
RF_PARAMS: dict[str, object] = {"n_estimators": 500, "max_depth": 3, "min_samples_leaf": 3,
                                "class_weight": "balanced", "random_state": SEED}
XGB_PARAMS: dict[str, object] = {"objective": "rank:pairwise", "max_depth": 2, "n_estimators": 100,
                                 "learning_rate": 0.05, "subsample": 0.8, "reg_lambda": 5, "random_state": SEED}
B0_TITLE_WEIGHT = 2.0
B0_HONOR_WEIGHT = 1.0
B0_STANDING_WEIGHT = 0.5


def edition_mask(study: StudyData, edition: int) -> np.ndarray:
    return (study.frame_t1["edition"] == edition).to_numpy()


def points_of(study: StudyData, edition: int) -> np.ndarray:
    return study.targets.loc[edition_mask(study, edition), "bdo_points"].to_numpy(dtype=float)


def vote_share_of(study: StudyData, edition: int) -> np.ndarray:
    return study.targets.loc[edition_mask(study, edition), "vote_share"].to_numpy(dtype=float)


def design_matrices(study: StudyData, snapshot: str, columns: list[str], include_post: bool,
                    train_editions: list[int], edition_list: list[int]) -> dict[int, np.ndarray]:
    """Feature matrices per edition. Residualisation is fitted on the training editions only."""
    frame = study.frame_t1 if snapshot == "t1" else study.frame_t2
    train = frame[frame["edition"].isin(train_editions)]
    params = study.features.fit_params(train, include_post)
    matrix = study.features.assemble(frame, params, include_post)[columns]
    return {ed: matrix[(frame["edition"] == ed).to_numpy()].to_numpy(dtype=float) for ed in edition_list}


def m1_matrices(study: StudyData, train_editions: list[int], edition_list: list[int]) -> dict[int, np.ndarray]:
    return design_matrices(study, "t1", list(study.features.base_columns), False, train_editions, edition_list)


def naive_scores(study: StudyData, apply_editions: list[int]) -> dict[int, np.ndarray]:
    """B0: 2 x major titles + other honors + 0.5 x prior standing, raw values."""
    frame = study.frame_t1
    score = (B0_TITLE_WEIGHT * frame["major_titles_raw"] + B0_HONOR_WEIGHT * frame["other_honors_raw"]
             + B0_STANDING_WEIGHT * frame["prior_standing_raw"])
    return {ed: score[edition_mask(study, ed)].to_numpy(dtype=float) for ed in apply_editions}


def ridge_scores(study: StudyData, alpha: float, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """B1: ridge regression on vote share; score = log of the predicted share (floored)."""
    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([vote_share_of(study, ed) for ed in train_editions])
    model = Ridge(alpha=alpha).fit(x_train, y_train)
    return {ed: np.log(np.maximum(model.predict(mats[ed]), SHARE_FLOOR)) for ed in apply_editions}


def winner_labels(points: np.ndarray) -> np.ndarray:
    return (points == points.max()).astype(int)


def forest_probabilities(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """B2: raw (un-normalised) win probabilities."""
    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([winner_labels(points_of(study, ed)) for ed in train_editions])
    model = RandomForestClassifier(**RF_PARAMS).fit(x_train, y_train)
    return {ed: model.predict_proba(mats[ed])[:, 1] for ed in apply_editions}


def forest_scores(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    raw = forest_probabilities(study, train_editions, apply_editions)
    return {ed: np.log(p / p.sum() + PROBABILITY_FLOOR) for ed, p in raw.items()}


def relevance_labels(points: np.ndarray) -> np.ndarray:
    """Relevance = dense rank of the points among voted players (1 = fewest points); zero-point players = 0."""
    labels = np.zeros(len(points), dtype=int)
    voted = points > 0
    levels = np.unique(points[voted])
    labels[voted] = np.searchsorted(levels, points[voted]) + 1
    return labels


def xgb_scores(study: StudyData, train_editions: list[int], apply_editions: list[int]) -> dict[int, np.ndarray]:
    """X1: LambdaMART-style pairwise ranker; editions are query groups. Needs xgboost and libomp."""
    from xgboost import XGBRanker

    mats = m1_matrices(study, train_editions, sorted(set(train_editions) | set(apply_editions)))
    x_train = np.vstack([mats[ed] for ed in train_editions])
    y_train = np.concatenate([relevance_labels(points_of(study, ed)) for ed in train_editions])
    qid = np.concatenate([np.full(len(mats[ed]), ed) for ed in train_editions])
    model = XGBRanker(**XGB_PARAMS).fit(x_train, y_train, qid=qid)
    return {ed: model.predict(mats[ed]) for ed in apply_editions}

