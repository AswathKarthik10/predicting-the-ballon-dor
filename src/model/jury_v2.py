"""Jury experiments on the 2016-2025 ballots (experiments spec sections 7.1-7.3)."""

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import ballots_report as br
from src.model.ballots_v2 import (BALLOT_EDITIONS_V2, BASE_INDICATORS, HOME_ERA_COLUMN, LOCAL_ATTENTION_COLUMN, attach_local_attention,
                                  coverage_table, local_attention_table, scheme_for)

CALENDAR_ERA: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021)
SEASON_ERA: tuple[int, ...] = (2022, 2023, 2024, 2025)
INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT = 5, 95
RESULT_COLUMNS: tuple[str, ...] = ("spec", "editions", "indicator", "beta", "se_laplace", "laplace_lo90", "laplace_hi90", "boot_lo90",
                                   "boot_hi90", "boot_se", "n_pairs", "n_flagged_ballots", "low_evidence")
ATTENTION_VARIANTS: dict[str, tuple[str, tuple[int, ...]]] = {
    "wiki share, 3,500 views": ("wiki_country_share_in", BALLOT_EDITIONS_V2),
    "wiki share, 1,000 views": ("wiki_country_share_in_r1000", tuple(e for e in BALLOT_EDITIONS_V2 if e >= 2018)),
    "news share": ("news_country_share_in", BALLOT_EDITIONS_V2),
}


def edition_label(editions: tuple[int, ...]) -> str:
    return f"{editions[0]}-{editions[-1]} ({len(editions)})" if len(editions) > 1 else str(editions[0])


def indicator_evidence(ballots: pd.DataFrame, editions: tuple[int, ...], indicators: tuple[str, ...]) -> pd.DataFrame:
    sub = ballots[ballots.edition.isin(editions)]
    rows = []
    for name in indicators:
        flagged = sub[sub[name] != 0]
        rows.append({"indicator": name, "n_flagged_ballots": len(flagged[["edition", "juror_country_key"]].drop_duplicates()), "n_pairs": len(flagged)})
    return pd.DataFrame(rows)


def build_data(ballots: pd.DataFrame, editions: tuple[int, ...], indicators: tuple[str, ...]) -> list[bl.EditionBallots]:
    return [bl.edition_ballots_for(ballots, e, indicators, len(scheme_for(e))) for e in editions]


def fit_spec(ballots: pd.DataFrame, spec: str, editions: tuple[int, ...], indicators: tuple[str, ...], n_boot: int
             ) -> tuple[pd.DataFrame, bl.BiasFit, np.ndarray]:
    """Ballot-level Plackett-Luce with the given indicator columns; coefficients with Laplace and ballot-bootstrap intervals."""
    data = build_data(ballots, editions, indicators)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    boot = bl.bootstrap_beta(data, bl.RIDGE_ALPHA, fit, n_boot, bl.BOOTSTRAP_SEED)
    se = np.sqrt(np.diag(fit.covariance))
    lower, upper = np.percentile(boot, [INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT], axis=0)
    table = pd.DataFrame({"spec": spec, "editions": edition_label(editions), "indicator": indicators, "beta": fit.beta, "se_laplace": se,
                          "laplace_lo90": fit.beta - bl.Z_90 * se, "laplace_hi90": fit.beta + bl.Z_90 * se,
                          "boot_lo90": lower, "boot_hi90": upper, "boot_se": boot.std(axis=0, ddof=1)})
    table = table.merge(indicator_evidence(ballots, editions, indicators), on="indicator", how="left")
    table["low_evidence"] = table.n_pairs < bl.MIN_INDICATOR_PAIRS
    return table[list(RESULT_COLUMNS)], fit, boot


def home_after_reform(fit: bl.BiasFit, boot: np.ndarray, indicators: tuple[str, ...]) -> pd.DataFrame:
    """Home effect in 2022-2025 = home + home x era2022, with intervals from the paired bootstrap draws."""
    i, j = indicators.index("home"), indicators.index(HOME_ERA_COLUMN)
    draws = boot[:, i] + boot[:, j]
    variance = fit.covariance[i, i] + fit.covariance[j, j] + 2 * fit.covariance[i, j]
    se = float(np.sqrt(variance))
    beta = float(fit.beta[i] + fit.beta[j])
    return pd.DataFrame([{"spec": "era interaction", "editions": edition_label(BALLOT_EDITIONS_V2), "indicator": "home (2022-2025 jury)",
                          "beta": beta, "se_laplace": se, "laplace_lo90": beta - bl.Z_90 * se, "laplace_hi90": beta + bl.Z_90 * se,
                          "boot_lo90": float(np.percentile(draws, INTERVAL_LOW_PCT)), "boot_hi90": float(np.percentile(draws, INTERVAL_HIGH_PCT)),
                          "boot_se": float(draws.std(ddof=1)), "n_pairs": np.nan, "n_flagged_ballots": np.nan, "low_evidence": False}])


def descriptive_by_era(ballots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, editions in (("2016-2021", CALENDAR_ERA), ("2022-2025", SEASON_ERA), ("all nine", BALLOT_EDITIONS_V2)):
        summary = br.descriptive_summary(ballots[ballots.edition.isin(editions)])
        summary.insert(0, "era", label)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def places_gained(fit: bl.BiasFit, boot: np.ndarray, label: str) -> pd.DataFrame:
    table = br.rank_shift_table(fit, boot)
    table.insert(0, "spec", label)
    return table


def local_attention_fits(ballots: pd.DataFrame, attention: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does attention inside a juror's own country predict that juror's ranking beyond the player's global strength?
    Fitted on covered juror-editions only; reports the other coefficients with and without local attention, and coverage."""
    results: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    for variant, (column, editions) in ATTENTION_VARIANTS.items():
        local = local_attention_table(attention, column)
        local = local[local.edition.isin(editions)]
        covered = attach_local_attention(ballots[ballots.edition.isin(editions)], local)
        cov = coverage_table(ballots[ballots.edition.isin(editions)], local)
        cov.insert(0, "variant", variant)
        coverage.append(cov)
        base_table, _, _ = fit_spec(covered, f"{variant}: covered jurors, without local attention", editions, BASE_INDICATORS, n_boot)
        with_table, _, _ = fit_spec(covered, f"{variant}: covered jurors, with local attention", editions, (*BASE_INDICATORS, LOCAL_ATTENTION_COLUMN), n_boot)
        results += [base_table, with_table]
    return pd.concat(results, ignore_index=True), pd.concat(coverage, ignore_index=True)
