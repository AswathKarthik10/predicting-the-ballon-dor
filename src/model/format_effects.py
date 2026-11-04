"""Experiment 3, section 3: did the drivers of the vote change with the format? Reads no 2026 outcome.

3.1 fits M1 at one fixed penalty on the season editions and on the calendar editions and compares coefficients.
3.2 adds `feature x is_season` interaction blocks to a pooled nine-edition M1, one block at a time."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.model import baselines, cv, inference, research_v2 as rs, validation_v2
from src.model import experiment3 as e3
from src.model.data import StudyData
from src.model.features import FeatureParams, FeatureSet
from src.model.features_v2 import m1_columns_v2
from src.model.inference import Z_90
from src.model.pl import Ranking, build_ranking, fit_pl, penalised_hessian

SEED = 7
N_BOOT = 500
INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT = validation_v2.INTERVAL_LOW_PCT, validation_v2.INTERVAL_HIGH_PCT
SEASON_SUFFIX = "_x_season"
INTERACTION_BLOCKS: dict[str, tuple[str, ...]] = {
    "F9-F10 fame and attention": ("F9_fame", "F10_season_attention"),
    "F11 awards": ("F11_individual_awards",),
    "F4 major titles": ("F4_major_titles",),
    "F1 attack": ("F1_attack",),
}


@dataclass(frozen=True)
class FixedFit:
    beta: np.ndarray
    covariance: np.ndarray
    matrices: dict[int, np.ndarray]
    rankings: dict[int, Ranking]


def median_e3a_lambda(tuning: pd.DataFrame) -> float:
    """Median M1 penalty over the nine E3-A folds (spec 3.1)."""
    chosen = tuning[(tuning.run == e3.RUN_ALL) & (tuning.model == "M1")].lambda_or_alpha
    if len(chosen) != 9:
        raise ValueError(f"expected 9 E3-A M1 penalties, found {len(chosen)}")
    return float(chosen.median())


def fixed_lambda_fit(study: StudyData, editions: tuple[int, ...], columns: tuple[str, ...], lam: float) -> FixedFit:
    """M1 on the given editions at a fixed penalty; the residualiser is fitted on those editions only."""
    mats = baselines.design_matrices(study, "t1", list(columns), False, list(editions), list(editions))
    rankings = {ed: cv.ranking_of(study, ed) for ed in editions}
    xs, rks = [mats[ed] for ed in editions], [rankings[ed] for ed in editions]
    beta = fit_pl(xs, rks, lam)
    return FixedFit(beta, np.linalg.inv(penalised_hessian(xs, rks, beta, lam)), mats, rankings)


def bootstrap_fixed(study: StudyData, fit: FixedFit, editions: tuple[int, ...], lam: float, n_boot: int, seed: int) -> np.ndarray:
    """Within-edition player bootstrap at fixed features and penalty."""
    rng = np.random.default_rng(seed)
    points = {ed: baselines.points_of(study, ed) for ed in editions}
    draws = np.empty((n_boot, len(fit.beta)))
    for b in range(n_boot):
        xs, rks = [], []
        for ed in editions:
            idx = rng.integers(0, len(points[ed]), len(points[ed]))
            xs.append(fit.matrices[ed][idx])
            rks.append(build_ranking(points[ed][idx]))
        draws[b] = fit_pl(xs, rks, lam)
    return draws


def coefficients_by_format(study: StudyData, lam: float) -> pd.DataFrame:
    columns = m1_columns_v2()
    season = fixed_lambda_fit(study, e3.SEASON_FOLDS, columns, lam)
    calendar = fixed_lambda_fit(study, e3.CALENDAR_FOLDS, columns, lam)
    boot_s = bootstrap_fixed(study, season, e3.SEASON_FOLDS, lam, N_BOOT, SEED)
    boot_c = bootstrap_fixed(study, calendar, e3.CALENDAR_FOLDS, lam, N_BOOT, SEED)
    diff_draws = boot_s - boot_c  # the two sets of editions are disjoint, so the draws are independent
    se_s, se_c = np.sqrt(np.diag(season.covariance)), np.sqrt(np.diag(calendar.covariance))
    lo = np.percentile(diff_draws, INTERVAL_LOW_PCT, axis=0)
    hi = np.percentile(diff_draws, INTERVAL_HIGH_PCT, axis=0)
    return pd.DataFrame({
        "feature": columns, "lambda": lam,
        "beta_season": season.beta, "laplace_lo90_season": season.beta - Z_90 * se_s, "laplace_hi90_season": season.beta + Z_90 * se_s,
        "boot_lo90_season": np.percentile(boot_s, INTERVAL_LOW_PCT, axis=0), "boot_hi90_season": np.percentile(boot_s, INTERVAL_HIGH_PCT, axis=0),
        "beta_calendar": calendar.beta, "laplace_lo90_calendar": calendar.beta - Z_90 * se_c, "laplace_hi90_calendar": calendar.beta + Z_90 * se_c,
        "boot_lo90_calendar": np.percentile(boot_c, INTERVAL_LOW_PCT, axis=0), "boot_hi90_calendar": np.percentile(boot_c, INTERVAL_HIGH_PCT, axis=0),
        "diff_season_minus_calendar": season.beta - calendar.beta, "diff_lo90": lo, "diff_hi90": hi,
        "diff_excludes_zero": (lo > 0) | (hi < 0),
    })


def with_season_interactions(study: StudyData, sources: tuple[str, ...]) -> tuple[StudyData, tuple[str, ...]]:
    """Study whose M1 matrix also carries `source x is_season` columns; returns the new M1 column tuple."""
    base = study.features
    new_columns = tuple(f"{s}{SEASON_SUFFIX}" for s in sources)

    def assemble(frame: pd.DataFrame, params: FeatureParams, include_post: bool) -> pd.DataFrame:
        out = base.assemble(frame, params, include_post)
        for source, column in zip(sources, new_columns):
            out[column] = out[source] * (1.0 - frame["is_calendar"])
        return out

    columns = base.model_columns["M1"] + new_columns
    features = FeatureSet(name=f"{base.name}+season_interactions", fit_params=base.fit_params, assemble=assemble,
                          base_columns=columns, model_columns={"M1": columns})
    return replace(study, features=features), columns


def season_loeo_utilities(study: StudyData, columns: tuple[str, ...]) -> dict[int, np.ndarray]:
    """Held-out utilities on 2022-2025, trained on the eight other editions (the E3-A training sets)."""
    spec = rs.pl_spec_for("M1", columns)
    labelled = study.labelled_editions
    return {held: cv.predict_pl(study, spec, e3.train_set(e3.RUN_ALL, held, labelled), [held])[0][held] for held in e3.SEASON_FOLDS}


def interaction_tables(study: StudyData) -> pd.DataFrame:
    base_utilities = season_loeo_utilities(study, study.features.model_columns["M1"])
    points = {f: baselines.points_of(study, f) for f in e3.SEASON_FOLDS}
    rows: list[pd.DataFrame] = []
    for block, sources in INTERACTION_BLOCKS.items():
        variant, columns = with_season_interactions(study, sources)
        coefs = rs.final_coefficients(variant, columns, "M1", N_BOOT, SEED).table
        coefs = coefs[coefs.feature.str.endswith(SEASON_SUFFIX)].copy()
        utilities = season_loeo_utilities(variant, columns)
        folds = list(e3.SEASON_FOLDS)
        draws = validation_v2.bootstrap_fold_metrics(
            {("new", f): utilities[f] for f in folds} | {("base", f): base_utilities[f] for f in folds}, points, ["new", "base"], folds,
            validation_v2.N_BOOTSTRAP, validation_v2.BOOTSTRAP_SEED)
        diff = np.nanmean(draws[("new", e3.PRIMARY_METRIC)], axis=1) - np.nanmean(draws[("base", e3.PRIMARY_METRIC)], axis=1)
        per_fold = [cv.edition_metrics(utilities[f], points[f])[e3.PRIMARY_METRIC] - cv.edition_metrics(base_utilities[f], points[f])[e3.PRIMARY_METRIC]
                    for f in folds]
        coefs.insert(0, "block", block)
        coefs["delta_heldout_ll_2022_25"] = float(np.mean(per_fold))
        coefs["delta_lo90"] = float(np.percentile(diff, INTERVAL_LOW_PCT))
        coefs["delta_hi90"] = float(np.percentile(diff, INTERVAL_HIGH_PCT))
        coefs["folds_better"] = int(sum(d > 0 for d in per_fold))
        coefs["delta_by_fold"] = " ".join(f"{f}:{d:+.4f}" for f, d in zip(folds, per_fold))
        coefs["excludes_zero"] = (coefs.bootstrap_lo90 > 0) | (coefs.bootstrap_hi90 < 0)
        rows.append(coefs)
        print(f"interaction block {block!r} done", flush=True)
    return pd.concat(rows, ignore_index=True)
