"""Final Plackett-Luce fit, parameter uncertainty and indicator importance (spec sections 6.1-6.2)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import baselines, cv
from src.model.cv import PlSpec
from src.model.data import StudyData
from src.model.pl import Ranking, build_ranking, fit_pl, penalised_hessian

SEED = 7
N_BOOTSTRAP = 500
Z_90 = 1.6448536269514722


@dataclass(frozen=True)
class FinalFit:
    spec: PlSpec
    beta: np.ndarray
    lam: float
    matrices: dict[int, np.ndarray]  # training editions and 2026
    rankings: dict[int, Ranking]  # training editions only
    covariance: np.ndarray  # inverse of the penalised Hessian

    @property
    def columns(self) -> list[str]:
        return list(self.spec.columns)


def final_fit(study: StudyData, spec: PlSpec) -> FinalFit:
    """Fit on all labelled editions; lambda by leave-one-edition-out over them."""
    labelled = list(study.labelled_editions)
    lam = cv.tune_lambda(study, spec, labelled)
    editions = labelled + [study.forecast_edition]
    mats = baselines.design_matrices(study, spec.snapshot, list(spec.columns), spec.include_post, labelled, editions)
    rankings = {ed: cv.ranking_of(study, ed) for ed in labelled}
    xs = [mats[ed] for ed in labelled]
    rks = [rankings[ed] for ed in labelled]
    beta = fit_pl(xs, rks, lam)
    covariance = np.linalg.inv(penalised_hessian(xs, rks, beta, lam))
    return FinalFit(spec=spec, beta=beta, lam=lam, matrices=mats, rankings=rankings, covariance=covariance)


def laplace_draws(fit: FinalFit, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(fit.beta, fit.covariance, size=n_draws)


def laplace_summary(fit: FinalFit) -> pd.DataFrame:
    se = np.sqrt(np.diag(fit.covariance))
    return pd.DataFrame({"feature": fit.columns, "beta": fit.beta, "se_laplace": se,
                         "laplace_lo90": fit.beta - Z_90 * se, "laplace_hi90": fit.beta + Z_90 * se})


def bootstrap_betas(fit: FinalFit, study: StudyData, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Within-edition player bootstrap at the final lambda and fixed features (clarification S3)."""
    points = {ed: baselines.points_of(study, ed) for ed in study.labelled_editions}
    draws = np.empty((n_boot, len(fit.beta)))
    for b in range(n_boot):
        xs: list[np.ndarray] = []
        rks: list[Ranking] = []
        for ed in study.labelled_editions:
            idx = rng.integers(0, len(points[ed]), len(points[ed]))
            xs.append(fit.matrices[ed][idx])
            rks.append(build_ranking(points[ed][idx]))
        draws[b] = fit_pl(xs, rks, fit.lam)
    return draws


def coefficient_table(fit: FinalFit, study: StudyData, fold_coefs: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    table = laplace_summary(fit)
    boot = bootstrap_betas(fit, study, n_boot, rng)
    table["bootstrap_lo90"] = np.percentile(boot, 5, axis=0)
    table["bootstrap_hi90"] = np.percentile(boot, 95, axis=0)
    table["bootstrap_sd"] = boot.std(axis=0, ddof=1)
    folds = fold_coefs[fold_coefs["model"] == fit.spec.name].pivot(index="feature", columns="fold", values="beta")
    table["same_sign_folds"] = [int((np.sign(folds.loc[f]) == np.sign(b)).sum()) for f, b in zip(table["feature"], table["beta"])]
    table["lambda"] = fit.lam
    contrib_train = np.vstack([fit.matrices[ed] for ed in study.labelled_editions]) * fit.beta
    contrib_2026 = fit.matrices[study.forecast_edition] * fit.beta
    table["mean_abs_contribution_train"] = np.abs(contrib_train).mean(axis=0)
    table["mean_abs_contribution_2026"] = np.abs(contrib_2026).mean(axis=0)
    return table.sort_values("mean_abs_contribution_train", ascending=False).reset_index(drop=True)


def comparison_forecast(study: StudyData, model_names: list[str]) -> pd.DataFrame:
    """2026 predicted rank under every model, for transparency (U0 is not a model of interest)."""
    frame = study.frame_t1[study.frame_t1["edition"] == study.forecast_edition][["player_key", "player_name"]].reset_index(drop=True)
    for name in model_names:
        utilities, _ = cv.predict_model(study, name, list(study.labelled_editions), [study.forecast_edition])
        frame[f"rank_{name}"] = pd.Series(utilities[study.forecast_edition]).rank(ascending=False, method="first").astype(int)
    return frame
