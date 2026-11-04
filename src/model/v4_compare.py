"""Experiment 4 comparators on the same four outer folds (spec section 6.3): v3 E3-S, v1 M1 and the uniform reference U0.

Utilities are recomputed with the frozen v1-v3 code and checked against the saved metrics, then aligned to the v4 panel
row order by player_key so the paired bootstrap resamples the same players for every model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import baselines, cv, experiment3 as e3
from src.model.data import V2_SPEC, StudyData, load_study_data, load_study_v2_1
from src.model.features_v2 import V2_FEATURES
from src.model.pl import edition_loglik_per_voted, fit_pl
from src.model.run_v2_1 import v2_1_options

FOLDS: tuple[int, ...] = e3.SEASON_FOLDS
LABEL_V4, LABEL_V3, LABEL_V1, LABEL_U0 = "v4", "E3-S", "v1-M1", "U0"
SAVED_V1_METRICS = Path("results") / "metrics_by_fold.csv"
SAVED_V3_METRICS = Path("results") / "v3" / "metrics_by_fold_v3.csv"
REPRODUCTION_TOLERANCE = 1e-6
PRIMARY = "heldout_ll"


def keys_of(study: StudyData, edition: int) -> np.ndarray:
    frame = study.frame_t1
    return frame.loc[baselines.edition_mask(study, edition), "player_key"].to_numpy()


def align_to(utility: np.ndarray, keys: np.ndarray, target_keys: np.ndarray) -> np.ndarray:
    if set(keys) != set(target_keys):
        raise ValueError(f"player sets differ between studies: {sorted(set(keys) ^ set(target_keys))}")
    position = {key: i for i, key in enumerate(keys)}
    return utility[[position[key] for key in target_keys]]


def metric_rows(label: str, fold: int, utility: np.ndarray, points: np.ndarray) -> list[dict[str, object]]:
    return [{"run": label, "model": "PL", "fold": fold, "metric": metric, "value": value}
            for metric, value in cv.edition_metrics(utility, points).items()]


def check_reproduces(rows: pd.DataFrame, saved: pd.DataFrame, label: str) -> None:
    merged = rows[rows.metric == PRIMARY].merge(saved, on="fold", suffixes=("", "_saved"))
    gap = float((merged["value"] - merged["value_saved"]).abs().max())
    if gap >= REPRODUCTION_TOLERANCE:
        raise ValueError(f"{label} does not reproduce its saved held-out log-likelihood: max abs gap {gap}")


def v3_utilities(root: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], StudyData]:
    """E3-S M1 utilities per fold in the study's own row order, and the inner-CV score of the tuned model per fold."""
    study = load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES)
    result = e3.run_one(study, e3.RUN_SEASON, "M1", FOLDS)
    saved = pd.read_csv(root / SAVED_V3_METRICS)
    saved = saved[(saved.run == e3.RUN_SEASON) & (saved.model == "M1") & (saved.metric == PRIMARY)][["fold", "value"]]
    check_reproduces(result.metrics[result.metrics.metric == PRIMARY], saved, "E3-S M1")
    utilities = {fold: result.utilities[(e3.RUN_SEASON, "M1", fold)] for fold in FOLDS}
    inner = {fold: best_inner_score(study, [ed for ed in FOLDS if ed != fold]) for fold in FOLDS}
    return utilities, inner, study


def best_inner_score(study: StudyData, train: list[int]) -> float:
    """Mean inner leave-one-edition-out log-likelihood at the penalty `cv.tune_lambda` picks (same loop, score instead of penalty)."""
    spec = cv.pl_spec(study.features, "M1")
    rankings = {ed: cv.ranking_of(study, ed) for ed in train}
    scores = np.zeros((len(train), len(cv.LAMBDA_GRID)))
    for i, held in enumerate(train):
        inner_train = [ed for ed in train if ed != held]
        mats = baselines.design_matrices(study, spec.snapshot, list(spec.columns), spec.include_post, inner_train, train)
        for j, lam in enumerate(cv.LAMBDA_GRID):
            beta = fit_pl([mats[ed] for ed in inner_train], [rankings[ed] for ed in inner_train], lam)
            scores[i, j] = edition_loglik_per_voted(mats[held] @ beta, rankings[held])
    return float(scores.mean(axis=0).max())


def v1_utilities(root: Path) -> tuple[dict[int, np.ndarray], StudyData]:
    study = load_study_data(root)
    utilities: dict[int, np.ndarray] = {}
    for fold in FOLDS:
        train = [ed for ed in study.labelled_editions if ed != fold]
        scores, _ = cv.predict_model(study, "M1", train, [fold])
        utilities[fold] = scores[fold]
    saved = pd.read_csv(root / SAVED_V1_METRICS, comment="#")
    saved = saved[(saved.model == "M1") & (saved.metric == PRIMARY)][["fold", "value"]]
    rows = pd.DataFrame([r for fold in FOLDS for r in metric_rows(LABEL_V1, fold, utilities[fold], baselines.points_of(study, fold))])
    check_reproduces(rows, saved, "v1 M1")
    return utilities, study
