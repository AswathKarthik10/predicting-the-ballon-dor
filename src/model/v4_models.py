"""Experiment 4 model menu (spec section 5).

PL models give utilities directly. Tree models give scores that are converted to PL utilities with a temperature fitted by
maximum likelihood on out-of-fold training scores (leave-one-edition-out over the training editions only), as in v1.
A model with no selected columns returns zero utilities (the uniform reference).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from lightgbm import LGBMRanker
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRanker

from src.model.baselines import relevance_labels
from src.model.pl import fit_temperature
from src.model.v4_pl_fast import penalised_negative_loglik, sort_edition
from src.model.v4_pl_penalised import fit_pl_lasso, fit_pl_mixed, sort_editions
from src.model.v4_select import TrainSet, make_train_set

SEED = 7
LAMBDA_GRID_RIDGE = tuple(float(v) for v in np.logspace(-2, 2, 13))
LAMBDA_GRID_L1 = tuple(float(v) for v in np.logspace(-3, 1, 13))
MIX_GRID = (0.25, 0.5, 0.75)
LGB_LEAVES = (4, 8)
LGB_ESTIMATORS = (100, 300)
LGB_MIN_LEAF = (5, 10)
XGB_DEPTH = (2, 3)
XGB_ESTIMATORS = (100, 300)
RF_DEPTH = (3, 5)
RF_MIN_LEAF = (3, 8)
LEARNING_RATE = 0.05
TREE_L2 = 5.0
XGB_SUBSAMPLE = 0.8
RF_TREES = 500
MAX_LGB_LABEL = 30
NOT_USED = 0
RIDGE_MAX_ITER = 500


class ModelKind(str, Enum):
    RIDGE = "P-R"
    LASSO = "P-L"
    ELASTIC = "P-EN"
    LGB = "LGB"
    XGB = "XGB"
    RF = "RF"


@dataclass(frozen=True)
class ModelSpec:
    kind: ModelKind
    lam: float
    mix: float
    size: int  # num_leaves (LGB) or max_depth (XGB, RF)
    n_estimators: int
    min_leaf: int

    @property
    def name(self) -> str:
        return f"{self.kind.value}(lam={self.lam},mix={self.mix},size={self.size},n={self.n_estimators},leaf={self.min_leaf})"


@dataclass(frozen=True)
class Prediction:
    utilities: list[np.ndarray]
    temperature: float  # 1.0 for PL models
    converged: bool


def model_grid() -> list[ModelSpec]:
    specs = [ModelSpec(ModelKind.RIDGE, lam, NOT_USED, NOT_USED, NOT_USED, NOT_USED) for lam in LAMBDA_GRID_RIDGE]
    specs += [ModelSpec(ModelKind.LASSO, lam, NOT_USED, NOT_USED, NOT_USED, NOT_USED) for lam in LAMBDA_GRID_L1]
    specs += [ModelSpec(ModelKind.ELASTIC, lam, mix, NOT_USED, NOT_USED, NOT_USED) for lam in LAMBDA_GRID_L1 for mix in MIX_GRID]
    specs += [ModelSpec(ModelKind.LGB, NOT_USED, NOT_USED, leaves, n, leaf)
              for leaves in LGB_LEAVES for n in LGB_ESTIMATORS for leaf in LGB_MIN_LEAF]
    specs += [ModelSpec(ModelKind.XGB, NOT_USED, NOT_USED, depth, n, NOT_USED) for depth in XGB_DEPTH for n in XGB_ESTIMATORS]
    specs += [ModelSpec(ModelKind.RF, NOT_USED, NOT_USED, depth, RF_TREES, leaf) for depth in RF_DEPTH for leaf in RF_MIN_LEAF]
    return specs


def fit_ridge(train: TrainSet, lam: float) -> np.ndarray:
    editions = [sort_edition(x, r) for x, r in zip(train.xs, train.rankings)]
    k = train.xs[0].shape[1]
    result = minimize(penalised_negative_loglik, np.zeros(k), args=(editions, lam), jac=True, method="L-BFGS-B",
                      options={"maxiter": RIDGE_MAX_ITER})
    if not result.success:
        raise RuntimeError(f"ridge PL fit did not converge (lam={lam}): {result.message}")
    return result.x


def percentile_target(points: np.ndarray) -> np.ndarray:
    return (rankdata(points) - 1.0) / max(len(points) - 1, 1)


def fit_tree_scorer(spec: ModelSpec, train: TrainSet):
    x = np.vstack(train.xs)
    if spec.kind is ModelKind.RF:
        target = np.concatenate([percentile_target(p) for p in train.points])
        return RandomForestRegressor(n_estimators=spec.n_estimators, max_depth=spec.size, min_samples_leaf=spec.min_leaf,
                                     random_state=SEED, n_jobs=1).fit(x, target).predict
    labels = np.concatenate([relevance_labels(p) for p in train.points])
    if labels.max() > MAX_LGB_LABEL:
        raise ValueError(f"relevance label {labels.max()} exceeds the LightGBM default label gain table")
    if spec.kind is ModelKind.LGB:
        model = LGBMRanker(n_estimators=spec.n_estimators, learning_rate=LEARNING_RATE, num_leaves=spec.size,
                           min_child_samples=spec.min_leaf, reg_lambda=TREE_L2, random_state=SEED, n_jobs=1, verbose=-1,
                           deterministic=True, force_row_wise=True)
        return model.fit(x, labels, group=[len(p) for p in train.points]).predict
    qid = np.concatenate([np.full(len(p), i) for i, p in enumerate(train.points)])
    model = XGBRanker(objective="rank:pairwise", max_depth=spec.size, n_estimators=spec.n_estimators, learning_rate=LEARNING_RATE,
                      subsample=XGB_SUBSAMPLE, reg_lambda=TREE_L2, random_state=SEED, n_jobs=1)
    return model.fit(x, labels, qid=qid).predict


def subset_train(train: TrainSet, columns: np.ndarray, drop_edition: int) -> TrainSet:
    keep = [i for i, ed in enumerate(train.editions) if ed != drop_edition]
    return make_train_set(tuple(train.editions[i] for i in keep), [train.xs[i][:, columns] for i in keep],
                          [train.points[i] for i in keep])


def tree_temperature(spec: ModelSpec, train: TrainSet, columns: np.ndarray) -> float:
    """Temperature from out-of-fold scores: leave one training edition out, fit on the rest, score the left-out edition."""
    scores = [fit_tree_scorer(spec, subset_train(train, columns, ed))(train.xs[i][:, columns]) for i, ed in enumerate(train.editions)]
    return fit_temperature(scores, train.rankings)


@dataclass(frozen=True)
class PlFit:
    beta: np.ndarray
    converged: bool


def fit_pl_beta(spec: ModelSpec, train: TrainSet) -> PlFit:
    """Coefficients of a PL model (ridge, lasso or elastic net) on the given matrices, all columns used."""
    start = np.zeros(train.xs[0].shape[1])
    if spec.kind is ModelKind.RIDGE:
        return PlFit(fit_ridge(train, spec.lam), True)
    editions = sort_editions(train.xs, train.rankings)
    fit = fit_pl_lasso(editions, spec.lam, start) if spec.kind is ModelKind.LASSO else fit_pl_mixed(editions, spec.lam, spec.mix, start)
    return PlFit(fit.beta, fit.converged)


def predict_pl(spec: ModelSpec, train: TrainSet, columns: np.ndarray, apply_xs: list[np.ndarray]) -> Prediction:
    narrowed = make_train_set(train.editions, [x[:, columns] for x in train.xs], train.points)
    fit = fit_pl_beta(spec, narrowed)
    return Prediction(utilities=[x[:, columns] @ fit.beta for x in apply_xs], temperature=1.0, converged=fit.converged)


def predict_tree(spec: ModelSpec, train: TrainSet, columns: np.ndarray, apply_xs: list[np.ndarray]) -> Prediction:
    narrowed = make_train_set(train.editions, [x[:, columns] for x in train.xs], train.points)
    temperature = tree_temperature(spec, train, columns)
    scorer = fit_tree_scorer(spec, narrowed)
    return Prediction(utilities=[scorer(x[:, columns]) / temperature for x in apply_xs], temperature=temperature, converged=True)


def predict_utilities(spec: ModelSpec, train: TrainSet, columns: np.ndarray, apply_xs: list[np.ndarray]) -> Prediction:
    if len(columns) == 0:
        return Prediction(utilities=[np.zeros(len(x)) for x in apply_xs], temperature=1.0, converged=True)
    if spec.kind in (ModelKind.RIDGE, ModelKind.LASSO, ModelKind.ELASTIC):
        return predict_pl(spec, train, columns, apply_xs)
    return predict_tree(spec, train, columns, apply_xs)
