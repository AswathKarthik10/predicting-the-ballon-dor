"""Bias estimates table, plain-language translation and headers for the ballot-based outputs (spec section 3)."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model.data import T1_FILE, sha256_of
from src.model.report import git_commit

SPEC_MAIN, SPEC_10PICK, SPEC_BY_EDITION = "main", "10pick_only", "by_edition"
TEN_PICK_EDITIONS: tuple[int, ...] = (2024, 2025)
NEUTRAL_RANK = 5
SHIFT_SIMS_PER_EDITION = 2000
SHIFT_BOOT_DRAWS = 100
SHIFT_SEED = 7


def jury_header_lines(root: Path) -> list[str]:
    hashes = [f"# sha256_{name.removesuffix('.parquet')}={sha256_of(root / bl.JURY_DIR / name)}" for name in bl.JURY_FILES]
    return [f"# sha256_t1={sha256_of(root / T1_FILE)}", *hashes, f"# git_commit={git_commit(root)}", f"# seed={bl.BOOTSTRAP_SEED}"]


def indicator_evidence(ballots: pd.DataFrame, editions: tuple[int, ...]) -> pd.DataFrame:
    """Per indicator: ballots with at least one flagged nominee, and flagged player-juror pairs."""
    sub = ballots[ballots.edition.isin(editions)]
    rows = []
    for name in bl.INDICATORS:
        flagged = sub[sub[name] == 1]
        rows.append({"indicator": name, "n_ballots": len(flagged[["edition", "juror_country_key"]].drop_duplicates()),
                     "n_pairs": len(flagged)})
    return pd.DataFrame(rows)


def estimate_rows(spec: str, edition_label: str, fit: bl.BiasFit, boot: np.ndarray, evidence: pd.DataFrame) -> pd.DataFrame:
    se = np.sqrt(np.diag(fit.covariance))
    lower, upper = np.percentile(boot, [5, 95], axis=0)
    out = pd.DataFrame({"spec": spec, "editions": edition_label, "indicator": bl.INDICATORS, "beta": fit.beta, "se_laplace": se,
                        "laplace_lo90": fit.beta - bl.Z_90 * se, "laplace_hi90": fit.beta + bl.Z_90 * se,
                        "boot_lo90": lower, "boot_hi90": upper, "boot_se": boot.std(axis=0, ddof=1)})
    out = out.merge(evidence, on="indicator", how="left")
    out["low_evidence"] = out.n_pairs < bl.MIN_INDICATOR_PAIRS
    return out


def fit_spec(ballots: pd.DataFrame, spec: str, editions: tuple[int, ...], n_boot: int) -> tuple[pd.DataFrame, bl.BiasFit, np.ndarray]:
    data = bl.build_edition_ballots(ballots, editions)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    boot = bl.bootstrap_beta(data, bl.RIDGE_ALPHA, fit, n_boot, bl.BOOTSTRAP_SEED)
    label = "-".join(str(e) for e in editions) if len(editions) > 1 else str(editions[0])
    return estimate_rows(spec, label, fit, boot, indicator_evidence(ballots, editions)), fit, boot


def estimate_all_specs(ballots: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, bl.BiasFit, np.ndarray]:
    main_rows, main_fit, main_boot = fit_spec(ballots, SPEC_MAIN, bl.BALLOT_EDITIONS, n_boot)
    tenpick_rows, _, _ = fit_spec(ballots, SPEC_10PICK, TEN_PICK_EDITIONS, n_boot)
    edition_rows = [fit_spec(ballots, SPEC_BY_EDITION, (e,), n_boot)[0] for e in bl.BALLOT_EDITIONS]
    return pd.concat([main_rows, tenpick_rows, *edition_rows], ignore_index=True), main_fit, main_boot


def mean_rank_shift(alpha: np.ndarray, beta_k: float, n_sims: int, rng: np.random.Generator) -> float:
    """Mean places gained by one nominee given +beta_k, among draws where a neutral juror would rank him NEUTRAL_RANK-th.

    alpha is held fixed. Each draw gives every nominee Gumbel noise; the shifted nominee is the one whose neutral rank is 5.
    """
    n_players = len(alpha)
    noise = rng.gumbel(size=(n_sims, n_players))
    neutral = alpha[None, :] + noise
    neutral_rank = (-neutral).argsort(axis=1).argsort(axis=1) + 1
    shifted_rank = np.empty_like(neutral_rank)
    shifted_value = neutral + beta_k
    # Rank of each player if only that player gets the bonus: count the others strictly above his shifted value.
    for i in range(n_players):
        others = np.delete(neutral, i, axis=1)
        shifted_rank[:, i] = 1 + (others > shifted_value[:, [i]]).sum(axis=1)
    at_target = neutral_rank == NEUTRAL_RANK
    return float((neutral_rank[at_target] - shifted_rank[at_target]).mean())


def rank_shift_for_beta(fit: bl.BiasFit, beta: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    """Mean places gained for each indicator, averaged over editions (fresh generator per call, same seed)."""
    shifts = np.zeros(bl.N_BETA)
    for k in range(bl.N_BETA):
        rng = np.random.default_rng(seed)
        shifts[k] = np.mean([mean_rank_shift(alpha, beta[k], n_sims, rng) for alpha in fit.alpha.values()])
    return shifts


def rank_shift_table(fit: bl.BiasFit, boot: np.ndarray) -> pd.DataFrame:
    point = rank_shift_for_beta(fit, fit.beta, SHIFT_SIMS_PER_EDITION, SHIFT_SEED)
    draws = np.array([rank_shift_for_beta(fit, b, SHIFT_SIMS_PER_EDITION // 4, SHIFT_SEED) for b in boot[:SHIFT_BOOT_DRAWS]])
    lower, upper = np.percentile(draws, [5, 95], axis=0)
    return pd.DataFrame({"indicator": bl.INDICATORS, "beta": fit.beta, "places_gained": point, "lo90": lower, "hi90": upper})


def first_place_shares(ballots: pd.DataFrame, flag: str) -> pd.DataFrame:
    """For nominees with at least one flagged juror: share of flagged jurors ranking him 1st vs share of all other jurors."""
    top = ballots.assign(first=(ballots.pick_rank == 1).astype(int))
    rows = []
    for (edition, player), group in top.groupby(["edition", "player_key"]):
        flagged, rest = group[group[flag] == 1], group[group[flag] == 0]
        if len(flagged) == 0:
            continue
        rows.append({"edition": edition, "player_key": player, "n_flagged": len(flagged),
                     "flagged_first": flagged["first"].mean(), "others_first": rest["first"].mean()})
    return pd.DataFrame(rows)


def descriptive_summary(ballots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag in bl.INDICATORS:
        pairs = first_place_shares(ballots, flag)
        weights = pairs.n_flagged
        rows.append({"indicator": flag, "n_nominees": len(pairs), "n_flagged_jurors": int(weights.sum()),
                     "share_flagged_rank_first": float(np.average(pairs.flagged_first, weights=weights)),
                     "share_others_rank_first": float(np.average(pairs.others_first, weights=weights))})
    return pd.DataFrame(rows)


def write_bias_summary(estimates: pd.DataFrame, shifts: pd.DataFrame, descriptive: pd.DataFrame, header: list[str], path: Path) -> None:
    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
        return "\n".join(lines)

    main = estimates[estimates.spec == SPEC_MAIN][["indicator", "beta", "se_laplace", "boot_lo90", "boot_hi90", "n_ballots", "n_pairs"]]
    text = "\n".join([
        "# Juror nationality effects estimated from real ballots (2022-2025)", "",
        *[line.replace("# ", "<!-- ", 1) + " -->" for line in header], "",
        "These coefficients describe how ballots differ by juror nationality *after* controlling for each nominee's overall standing "
        "that year. They are not a causal estimate of juror bias.", "",
        "## Coefficients (log-odds scale, main fit)", "", md(main), "",
        f"## What they mean in places (a nominee a neutral juror would rank {NEUTRAL_RANK}th)", "",
        "Places gained on one juror's ballot when the juror is a compatriot / same-continent / same league country. "
        "Simulated from the fitted model with each nominee's strength held fixed; interval is the 90% bootstrap range.", "", md(shifts), "",
        "## Descriptive check, no model", "",
        "Share of flagged jurors who ranked the nominee 1st, against the share of all other jurors who did, "
        "pooled over nominees with at least one flagged juror (weighted by number of flagged jurors).", "", md(descriptive), ""])
    path.write_text(text)
