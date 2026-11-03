"""Leave-one-edition-out validation, nested tuning, temperature calibration and metrics (spec section 5)."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.model import baselines, models_extra
from src.model.data import StudyData
from src.model.features import V1_FEATURES, FeatureSet
from src.model.pl import Ranking, build_ranking, edition_loglik, edition_loglik_per_voted, fit_pl, fit_temperature

LAMBDA_GRID = np.logspace(-2, 2, 13)
TOP_N_OVERLAP = 3
MODEL_ORDER: tuple[str, ...] = ("U0", "B0", "B1", "B2", "X1", "L1", "O1", "D1", "M1", "M1-G", "M1-F", "M2")
PL_MODELS: tuple[str, ...] = ("M1", "M1-G", "M1-F", "M2")
METRIC_NAMES: tuple[str, ...] = ("heldout_ll", "spearman", "pairwise_acc", "top1_hit", "top3_overlap", "winner_rank", "top10_acc", "ndcg10")
TOP_N_ACCURACY = 10
POST_WINDOW_MODELS: tuple[str, ...] = ("M2",)


@dataclass(frozen=True)
class PlSpec:
    name: str
    snapshot: str
    columns: tuple[str, ...]
    include_post: bool


def pl_spec(features: FeatureSet, name: str) -> PlSpec:
    include_post = name in POST_WINDOW_MODELS
    return PlSpec(name, "t2" if include_post else "t1", features.model_columns[name], include_post)


PL_SPECS: dict[str, PlSpec] = {name: pl_spec(V1_FEATURES, name) for name in V1_FEATURES.model_columns}


@dataclass(frozen=True)
class FitInfo:
    model: str
    beta: np.ndarray | None
    lam: float | None
    temperature: float | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CvResults:
    metrics: pd.DataFrame  # model, fold, metric, value
    heldout_utilities: dict[tuple[str, int], np.ndarray]  # (model, fold) to the held-out utilities, for the bootstrap
    coefficients: pd.DataFrame  # model, fold, feature, beta, lambda
    tuning: pd.DataFrame  # model, fold, selected hyper-parameter and temperature
    diagnostics: pd.DataFrame  # extra numbers, e.g. random forest probability sums


def ranking_of(study: StudyData, edition: int) -> Ranking:
    return build_ranking(baselines.points_of(study, edition))


def tune_lambda(study: StudyData, spec: PlSpec, train: list[int]) -> float:
    """Penalty chosen by inner leave-one-edition-out over the training editions (residualiser refit in every inner fold)."""
    columns = list(spec.columns)
    rankings = {ed: ranking_of(study, ed) for ed in train}
    inner = np.zeros((len(train), len(LAMBDA_GRID)))
    for i, held in enumerate(train):
        inner_train = [ed for ed in train if ed != held]
        mats = baselines.design_matrices(study, spec.snapshot, columns, spec.include_post, inner_train, train)
        for j, lam in enumerate(LAMBDA_GRID):
            beta = fit_pl([mats[ed] for ed in inner_train], [rankings[ed] for ed in inner_train], lam)
            inner[i, j] = edition_loglik_per_voted(mats[held] @ beta, rankings[held])
    return float(LAMBDA_GRID[int(np.argmax(inner.mean(axis=0)))])


def predict_pl(study: StudyData, spec: PlSpec, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    """Fit a Plackett-Luce model on the training editions (lambda chosen by inner leave-one-edition-out) and
    return utilities for the apply editions."""
    columns = list(spec.columns)
    rankings = {ed: ranking_of(study, ed) for ed in train}
    lam_best = tune_lambda(study, spec, train)
    mats = baselines.design_matrices(study, spec.snapshot, columns, spec.include_post, train, sorted(set(train) | set(apply)))
    beta = fit_pl([mats[ed] for ed in train], [rankings[ed] for ed in train], lam_best)
    utilities = {ed: mats[ed] @ beta for ed in apply}
    return utilities, FitInfo(spec.name, beta, lam_best, None, tuple(columns))


ScoreFn = Callable[[StudyData, list[int], list[int]], dict[int, np.ndarray]]


def calibrate_and_predict(study: StudyData, score_fn: ScoreFn, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], float]:
    """Fit temperature T on out-of-sample scores from an inner leave-one-edition-out loop (training editions only),
    then return score / T for the apply editions."""
    oof = {held: score_fn(study, [ed for ed in train if ed != held], [held])[held] for held in train}
    temperature = fit_temperature([oof[ed] for ed in train], [ranking_of(study, ed) for ed in train])
    final = score_fn(study, train, apply)
    return {ed: s / temperature for ed, s in final.items()}, temperature


def tune_ridge_alpha(study: StudyData, train: list[int]) -> float:
    inner = np.zeros((len(train), len(LAMBDA_GRID)))
    for i, held in enumerate(train):
        inner_train = [ed for ed in train if ed != held]
        for j, alpha in enumerate(LAMBDA_GRID):
            scores = baselines.ridge_scores(study, alpha, inner_train, inner_train + [held])
            temperature = fit_temperature([scores[ed] for ed in inner_train], [ranking_of(study, ed) for ed in inner_train])
            inner[i, j] = edition_loglik_per_voted(scores[held] / temperature, ranking_of(study, held))
    return float(LAMBDA_GRID[int(np.argmax(inner.mean(axis=0)))])


def predict_b0(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    scores = baselines.naive_scores(study, sorted(set(train) | set(apply)))
    temperature = fit_temperature([scores[ed] for ed in train], [ranking_of(study, ed) for ed in train])
    return {ed: scores[ed] / temperature for ed in apply}, FitInfo("B0", None, None, temperature, ())


def predict_b1(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    alpha = tune_ridge_alpha(study, train)

    def score_fn(s: StudyData, tr: list[int], ap: list[int]) -> dict[int, np.ndarray]:
        return baselines.ridge_scores(s, alpha, tr, ap)

    utilities, temperature = calibrate_and_predict(study, score_fn, train, apply)
    return utilities, FitInfo("B1", None, alpha, temperature, ())


def predict_b2(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    utilities, temperature = calibrate_and_predict(study, baselines.forest_scores, train, apply)
    return utilities, FitInfo("B2", None, None, temperature, ())


def predict_x1(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    utilities, temperature = calibrate_and_predict(study, baselines.xgb_scores, train, apply)
    return utilities, FitInfo("X1", None, None, temperature, ())


def predict_l1(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    utilities, temperature = calibrate_and_predict(study, models_extra.lgbm_scores, train, apply)
    return utilities, FitInfo("L1", None, None, temperature, ())


def predict_o1(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    utilities, temperature = calibrate_and_predict(study, models_extra.ordinal_scores, train, apply)
    return utilities, FitInfo("O1", None, None, temperature, ())


def predict_d1(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    utilities, temperature = calibrate_and_predict(study, models_extra.tree_scores, train, apply)
    return utilities, FitInfo("D1", None, None, temperature, ())


def predict_uniform(study: StudyData, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    n = {ed: int(baselines.edition_mask(study, ed).sum()) for ed in apply}
    return {ed: np.zeros(n[ed]) for ed in apply}, FitInfo("U0", None, None, None, ())


def predict_model(study: StudyData, name: str, train: list[int], apply: list[int]) -> tuple[dict[int, np.ndarray], FitInfo]:
    if name in study.features.model_columns:
        return predict_pl(study, pl_spec(study.features, name), train, apply)
    dispatch = {"U0": predict_uniform, "B0": predict_b0, "B1": predict_b1, "B2": predict_b2, "X1": predict_x1,
                "L1": predict_l1, "O1": predict_o1, "D1": predict_d1}
    return dispatch[name](study, train, apply)


def edition_metrics(utility: np.ndarray, points: np.ndarray) -> dict[str, float]:
    ranking = build_ranking(points)
    winner = int(np.argmax(points))
    pair_i, pair_j = np.triu_indices(len(points), k=1)
    differs = points[pair_i] != points[pair_j]
    higher_points_first = points[pair_i] > points[pair_j]
    util_diff = utility[pair_i] - utility[pair_j]
    correct = np.where(util_diff == 0, 0.5, ((util_diff > 0) == higher_points_first).astype(float))
    actual_top = set(np.argsort(-points, kind="stable")[:TOP_N_OVERLAP])
    predicted_top = set(np.argsort(-utility, kind="stable")[:TOP_N_OVERLAP])
    has_order = np.ptp(utility) > 0  # a constant utility (the uniform reference) has no meaningful top or rank
    rho = spearmanr(utility, points)[0] if has_order else np.nan
    predicted_order = np.argsort(-utility, kind="stable")
    actual_order = np.argsort(-points, kind="stable")
    n_top = min(TOP_N_ACCURACY, len(points))
    discount = 1.0 / np.log2(np.arange(2, n_top + 2))
    dcg = float((points[predicted_order[:n_top]] * discount).sum())
    ideal = float((points[actual_order[:n_top]] * discount).sum())
    top10_overlap = len(set(predicted_order[:n_top]) & set(actual_order[:n_top]))
    return {
        "heldout_ll": edition_loglik_per_voted(utility, ranking),
        "spearman": float(rho),
        "pairwise_acc": float(correct[differs].mean()),
        "top1_hit": float(int(np.argmax(utility)) == winner) if has_order else np.nan,
        "top3_overlap": float(len(actual_top & predicted_top)) if has_order else np.nan,
        "winner_rank": float(1 + (utility > utility[winner]).sum()) if has_order else np.nan,
        "top10_acc": float(top10_overlap / TOP_N_ACCURACY) if has_order else np.nan,
        "ndcg10": dcg / ideal if has_order else np.nan,
    }


def run_cross_validation(study: StudyData, model_names: list[str]) -> CvResults:
    metric_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    diag_rows: list[dict[str, object]] = []
    heldout: dict[tuple[str, int], np.ndarray] = {}
    labelled = list(study.labelled_editions)
    for held in labelled:
        train = [ed for ed in labelled if ed != held]
        points = baselines.points_of(study, held)
        for name in model_names:
            utilities, info = predict_model(study, name, train, [held])
            heldout[(name, held)] = utilities[held]
            for metric, value in edition_metrics(utilities[held], points).items():
                metric_rows.append({"model": name, "fold": held, "metric": metric, "value": value})
            tuning_rows.append({"model": name, "fold": held, "lambda_or_alpha": info.lam, "temperature": info.temperature})
            if info.beta is not None:
                for feature, beta in zip(info.columns, info.beta):
                    coef_rows.append({"model": name, "fold": held, "feature": feature, "beta": beta, "lambda": info.lam})
        if "B2" in model_names:
            raw = baselines.forest_probabilities(study, train, [held])[held]
            diag_rows.append({"fold": held, "diagnostic": "rf_raw_win_probability_sum", "value": float(raw.sum())})
    return CvResults(metrics=pd.DataFrame(metric_rows), heldout_utilities=heldout, coefficients=pd.DataFrame(coef_rows),
                     tuning=pd.DataFrame(tuning_rows), diagnostics=pd.DataFrame(diag_rows))


def summarise_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot_table(index=["model", "metric"], columns="fold", values="value")
    wide.columns = [f"fold_{c}" for c in wide.columns]
    wide.insert(0, "mean", metrics.groupby(["model", "metric"])["value"].mean())
    return wide.reset_index()


SELECTION_MARGIN = 0.02


@dataclass(frozen=True)
class Selection:
    published: str
    reported_alongside: str
    notes: tuple[str, ...]
    mean_ll: dict[str, float]


def apply_selection_rule(summary: pd.DataFrame) -> Selection:
    """Spec rule 5.4: publish the better of M1 and M2 (M1 unless M2 is better by at least 0.02 per voted player)."""
    ll = summary[summary["metric"] == "heldout_ll"].set_index("model")["mean"].to_dict()
    notes: list[str] = []
    published = "M2" if ll["M2"] - ll["M1"] >= SELECTION_MARGIN else "M1"
    other = "M1" if published == "M2" else "M2"
    gap = ll["M2"] - ll["M1"]
    notes.append(f"M2 minus M1 mean held-out log-likelihood = {gap:+.4f} (margin {SELECTION_MARGIN}); published model = {published}.")
    best_pl = max(ll["M1"], ll["M2"])
    for rival in ("B1", "B2", "X1"):
        if rival in ll and ll[rival] > best_pl:
            notes.append(f"{rival} beats both M1 and M2 ({ll[rival]:.4f} vs {best_pl:.4f}); publish {published} anyway per rule 5.4(3), "
                         f"report {rival} alongside and log in deviations.md.")
    if "X1" in ll and ll["X1"] > ll["M1"]:
        notes.append("X1 beats M1: inspect X1's splits and test the matching interaction in PL as exploratory analysis (rule 5.4(4)).")
    return Selection(published=published, reported_alongside=other, notes=tuple(notes), mean_ll=ll)
