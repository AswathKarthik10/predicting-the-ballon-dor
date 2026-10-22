"""Leave-one-edition-out validation of the ballot-based simulation against the placeholder one (spec section 6)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import ballots_report as br
from src.model import baselines, jurors, link, simulate
from src.model.data import StudyData

VALIDATION_SIMS = 10_000
SEED = 7
CALIBRATION_FILE = Path("results") / "calibration_tau_juror.csv"
TAU_SHARED_PLACEHOLDER = 1.0
WORSE_EDITIONS_TO_REJECT = 3
MODEL_NEW, MODEL_PLACEHOLDER = "ballots", "placeholder"
CONCENTRATION_STATS: tuple[str, ...] = ("winner_share", "n_with_points", "herfindahl")


@dataclass(frozen=True)
class HeldOutParams:
    edition: int
    beta: np.ndarray
    beta_covariance: np.ndarray
    link_fit: link.LinkFit


def placeholder_tau_juror(root: Path) -> float:
    table = pd.read_csv(root / CALIBRATION_FILE)
    return float(table.groupby("tau_juror").squared_relative_error.sum().idxmin())


def fit_without_edition(ballot_table: pd.DataFrame, strengths: pd.DataFrame, snapshot: str, held_out: int) -> HeldOutParams:
    editions = tuple(e for e in bl.BALLOT_EDITIONS if e != held_out)
    data = bl.build_edition_ballots(ballot_table, editions)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    link_fit = link.fit_link(link.alpha_by_player(fit, data), strengths, snapshot, editions)
    return HeldOutParams(edition=held_out, beta=fit.beta, beta_covariance=fit.covariance, link_fit=link_fit)


def ballot_simulation(root: Path, edition: int, utilities: np.ndarray, params: HeldOutParams, n_sims: int, seed: int) -> simulate.SimResult:
    rng = np.random.default_rng(seed)
    inputs = jurors.build_real_sim_inputs(jurors.real_jury(root, edition), jurors.player_attributes(root, edition))
    betas = rng.multivariate_normal(params.beta, params.beta_covariance, size=n_sims)
    config = simulate.BallotSimConfig(k=params.link_fit.k, sigma_shared=params.link_fit.sigma_shared,
                                      scheme=bl.SCHEME_BY_EDITION[edition], n_dropped=0)
    return simulate.simulate_ballots(np.tile(utilities, (n_sims, 1)), betas, inputs, config, rng)


def placeholder_simulation(root: Path, edition: int, utilities: np.ndarray, tau_juror: float, n_sims: int, seed: int) -> simulate.SimResult:
    """Original placeholder simulation (old reconstructed panel, 0.8/0.2/0.2), run under the edition's true points scheme."""
    players = jurors.edition_players(jurors.load_juror_inputs(root), edition)
    inputs = jurors.build_sim_inputs(players, jurors.JURORS_BY_EDITION[edition])
    config = simulate.SimConfig(tau_shared=TAU_SHARED_PLACEHOLDER, tau_juror=tau_juror, bias=jurors.PLACEHOLDER_BIAS)
    scheme = np.array(bl.SCHEME_BY_EDITION[edition], dtype=float)
    return simulate.simulate_scheme(np.tile(utilities, (n_sims, 1)), inputs, config, scheme, np.random.default_rng(seed))


def edition_row(model: str, snapshot: str, edition: int, result: simulate.SimResult, actual_points: np.ndarray,
                actual_concentration: dict[str, float], actual_home_share: float) -> dict[str, object]:
    winner = int(np.argmax(actual_points))
    p_win = float((result.ranks[:, winner] == 1).mean())
    sim_conc = simulate.simulated_concentration(result)
    row: dict[str, object] = {"model": model, "snapshot": snapshot, "edition": edition, "p_actual_winner": p_win,
                              "log_p_actual_winner": float(np.log(max(p_win, 1.0 / len(result.ranks)))),
                              "winner_expected_rank": float(result.ranks[:, winner].mean()),
                              "home_first_share_sim": float(np.nanmean(result.home_first_share)),
                              "home_first_share_actual": actual_home_share}
    for stat in CONCENTRATION_STATS:
        row[f"{stat}_sim"] = sim_conc[stat]
        row[f"{stat}_actual"] = actual_concentration[stat]
    return row


def actual_home_first_share(ballot_table: pd.DataFrame, edition: int) -> float:
    pairs = br.first_place_shares(ballot_table[ballot_table.edition == edition], "home")
    return float(np.average(pairs.flagged_first, weights=pairs.n_flagged))


def validate(root: Path, study: StudyData, ballot_table: pd.DataFrame, strengths_by_snapshot: dict[str, pd.DataFrame],
             n_sims: int) -> pd.DataFrame:
    tau_juror = placeholder_tau_juror(root)
    rows: list[dict[str, object]] = []
    for snapshot, strengths in strengths_by_snapshot.items():
        for edition in bl.BALLOT_EDITIONS:
            keys = study.frame_t1.loc[study.frame_t1.edition == edition, "player_key"].to_numpy()
            utilities = strengths[strengths.edition == edition].set_index("player_key").loc[keys, "u"].to_numpy()
            actual = baselines.points_of(study, edition)
            conc = simulate.concentration_stats(actual)
            home = actual_home_first_share(ballot_table, edition)
            params = fit_without_edition(ballot_table, strengths, snapshot, edition)
            new = ballot_simulation(root, edition, utilities, params, n_sims, SEED)
            old = placeholder_simulation(root, edition, utilities, tau_juror, n_sims, SEED)
            rows.append({**edition_row(MODEL_NEW, snapshot, edition, new, actual, conc, home), "k": params.link_fit.k,
                         "sigma_shared": params.link_fit.sigma_shared, "beta_home": params.beta[0], "beta_confed": params.beta[1],
                         "beta_league": params.beta[2]})
            rows.append(edition_row(MODEL_PLACEHOLDER, snapshot, edition, old, actual, conc, home))
    return pd.DataFrame(rows)


def gate_decision(table: pd.DataFrame, snapshot: str) -> tuple[int, bool]:
    """Number of editions where the ballot simulation has a lower log score than the placeholder, and whether that rejects it."""
    sub = table[table.snapshot == snapshot].pivot(index="edition", columns="model", values="log_p_actual_winner")
    worse = int((sub[MODEL_NEW] < sub[MODEL_PLACEHOLDER]).sum())
    return worse, worse >= WORSE_EDITIONS_TO_REJECT


def side_by_side(table: pd.DataFrame, snapshot: str, columns: tuple[str, ...]) -> pd.DataFrame:
    sub = table[table.snapshot == snapshot]
    out = sub.pivot(index="edition", columns="model", values=list(columns))
    out.columns = [f"{metric} [{model}]" for metric, model in out.columns]
    return out


def table_markdown(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [f"{v:.3f}" for v in row]) + " |")
    return "\n".join(lines)


def write_validation_report(table: pd.DataFrame, header: list[str], path: Path) -> None:
    parts = ["# Validation of the ballot-based juror simulation on 2022-2025", "",
             *[line.replace("# ", "<!-- ", 1) + " -->" for line in header], "",
             "Each edition is simulated with out-of-fold model strengths, that edition's real jury and points scheme, and bias, link "
             "and noise parameters estimated from the other three editions' ballots. The placeholder column is the original simulation "
             "(reconstructed panel, b = 0.8/0.2/0.2, tau_juror from the 2024-25 concentration calibration) run under the same "
             "points scheme. With four editions each metric is a handful of observations, so differences are suggestive, not conclusive.", ""]
    for snapshot in sorted(table.snapshot.unique()):
        sub = table[table.snapshot == snapshot]
        totals = sub.groupby("model").log_p_actual_winner.sum()
        worse, rejected = gate_decision(table, snapshot)
        verdict = ("REJECTED: the ballot simulation is worse on log score in " if rejected else "Accepted: the ballot simulation is worse on log score in ")
        parts += [f"## {snapshot} ({link.MODEL_BY_SNAPSHOT[snapshot]})", "",
                  f"Sum of log P(actual winner): ballots {totals[MODEL_NEW]:.3f}, placeholder {totals[MODEL_PLACEHOLDER]:.3f}. "
                  f"{verdict}{worse} of 4 editions (rule: reject at {WORSE_EDITIONS_TO_REJECT} or more).", "",
                  "### Winner probability and expected rank", "",
                  table_markdown(side_by_side(table, snapshot, ("p_actual_winner", "log_p_actual_winner", "winner_expected_rank"))), "",
                  "### Concentration (simulated mean vs actual)", ""]
        for stat in CONCENTRATION_STATS:
            wide = side_by_side(table, snapshot, (f"{stat}_sim",))
            wide[f"{stat} [actual]"] = sub[sub.model == MODEL_NEW].set_index("edition")[f"{stat}_actual"]
            parts += [table_markdown(wide), ""]
        home = side_by_side(table, snapshot, ("home_first_share_sim",))
        home["home_first_share [actual]"] = sub[sub.model == MODEL_NEW].set_index("edition").home_first_share_actual
        parts += ["### Home-vote check: share of compatriot jurors ranking their own nominee 1st", "", table_markdown(home), ""]
    parts += ["## Caveats", "",
              "- The placeholder tau_juror was calibrated on the 2024-25 concentration statistics, so the placeholder is in-sample for those editions.",
              "- Out-of-fold strengths for the other editions used to fit the link were produced by models that saw the held-out edition's "
              "results; the held-out edition's own ballots and strengths never enter its parameters.", ""]
    path.write_text("\n".join(parts))
