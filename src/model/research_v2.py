"""Research experiments on the ten-edition study (experiments spec sections 6.2-6.5).

Everything here works on labelled editions only; 2026 is never read."""

from dataclasses import dataclass, replace
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from src.model import baselines, cv, inference
from src.model.cv import PlSpec
from src.model.data import StudyData
from src.model.features import apply_residualiser, fit_residualiser
from src.model.features_v2 import BLOCKS_V2, POSITION_FEATURES, TOURNAMENT_INTERACTION, m1_columns_v2
from src.model.models_extra import LGBM_PARAMS

SEED = 7
RIDGE_ALPHA_SHARES = 10.0
N_BOOT_SHARES = 300
INTERVAL_LOW_PCT, INTERVAL_HIGH_PCT = 5, 95
WORSE_FOLDS_REQUIRED = 6  # spec 6.2: a block matters if dropping it worsens log-likelihood in at least 6 of 9 folds
FOOTBALL_TOP_N = 10
N_PERMUTATIONS = 10_000
CALENDAR_ERA_FOLDS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021)
SEASON_ERA_FOLDS: tuple[int, ...] = (2022, 2023, 2024, 2025)
TOURNAMENT_EDITIONS: tuple[int, ...] = (2016, 2018, 2019, 2021, 2023, 2024)
WORLD_CUP_EDITIONS: tuple[int, ...] = (2018, 2023)
FOOTBALL_BLOCKS: tuple[str, ...] = ("Season output", "Big stage", "Trophies and team", "Individual awards", "Prior standing", "Position")
FOOTBALL_COLUMNS_FOR_FAME: tuple[str, ...] = ("F1_attack", "F2_form_rating", "F3_big_stage", "F4_major_titles", "F5_other_honors",
                                              "F6_team_strength", "F7_prior_standing", "F8_defending_winner", "F11_individual_awards",
                                              *POSITION_FEATURES)


def columns_of(blocks: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for b in blocks for c in BLOCKS_V2[b])


def pl_spec_for(name: str, columns: tuple[str, ...]) -> PlSpec:
    return PlSpec(name, "t1", columns, False)


def competition_rank_of(points: np.ndarray) -> np.ndarray:
    """1 + number of players with strictly more points (ties share the best rank)."""
    return 1.0 + (points[None, :] > points[:, None]).sum(axis=1)


# ---------------------------------------------------------------------------------------------------------------------
# 6.2 Variance shares: leave-one-edition-out ridge R2 of the within-edition points percentile, split by Shapley values.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShareInputs:
    folds: tuple[int, ...]
    column_names: tuple[str, ...]
    fold_matrices: dict[int, dict[int, np.ndarray]]  # held-out edition -> edition -> matrix (residualiser fitted without the held-out edition)
    target: dict[int, np.ndarray]  # edition -> within-edition z-scored points percentile


def points_percentile_target(points: np.ndarray) -> np.ndarray:
    percentile = pd.Series(points).rank(pct=True).to_numpy()
    return (percentile - percentile.mean()) / percentile.std(ddof=1)


def build_share_inputs(study: StudyData) -> ShareInputs:
    folds = study.labelled_editions
    columns = tuple(study.features.base_columns)
    fold_matrices: dict[int, dict[int, np.ndarray]] = {}
    for held in folds:
        train = [e for e in folds if e != held]
        fold_matrices[held] = baselines.design_matrices(study, "t1", list(columns), False, train, list(folds))
    target = {e: points_percentile_target(baselines.points_of(study, e)) for e in folds}
    return ShareInputs(folds=folds, column_names=columns, fold_matrices=fold_matrices, target=target)


def ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def loeo_r2(inputs: ShareInputs, column_index: list[int], rows: dict[int, np.ndarray], alpha: float) -> float:
    """Pooled out-of-sample R2 (uncentred; the target is already standardised) for a column subset."""
    if not column_index:
        return 0.0
    sse = 0.0
    total = 0.0
    for held in inputs.folds:
        train = [e for e in inputs.folds if e != held]
        mats = inputs.fold_matrices[held]
        x_train = np.vstack([mats[e][rows[e]][:, column_index] for e in train])
        y_train = np.concatenate([inputs.target[e][rows[e]] for e in train])
        x_test = mats[held][rows[held]][:, column_index]
        y_test = inputs.target[held][rows[held]]
        sse += float(((y_test - x_test @ ridge_solve(x_train, y_train, alpha)) ** 2).sum())
        total += float((y_test ** 2).sum())
    return 1.0 - sse / total


def all_subset_values(inputs: ShareInputs, block_names: tuple[str, ...], rows: dict[int, np.ndarray], alpha: float) -> dict[frozenset, float]:
    position = {c: i for i, c in enumerate(inputs.column_names)}
    block_index = {b: [position[c] for c in BLOCKS_V2[b]] for b in block_names}
    values: dict[frozenset, float] = {}
    for size in range(len(block_names) + 1):
        for subset in combinations(block_names, size):
            index = sorted(i for b in subset for i in block_index[b])
            values[frozenset(subset)] = loeo_r2(inputs, index, rows, alpha)
    return values


def shapley_values(values: dict[frozenset, float], block_names: tuple[str, ...]) -> dict[str, float]:
    n = len(block_names)
    out: dict[str, float] = {}
    for block in block_names:
        others = [b for b in block_names if b != block]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                total += weight * (values[frozenset(subset) | {block}] - values[frozenset(subset)])
        out[block] = total
    return out


def variance_shares(inputs: ShareInputs, n_boot: int, seed: int) -> pd.DataFrame:
    """Shares of out-of-sample R2 by block, with 90% bootstrap intervals (players resampled within edition)."""
    names = tuple(BLOCKS_V2)
    full = frozenset(names)
    identity = {e: np.arange(len(inputs.target[e])) for e in inputs.folds}
    values = all_subset_values(inputs, names, identity, RIDGE_ALPHA_SHARES)
    shapley = shapley_values(values, names)
    total = sum(shapley.values())
    rng = np.random.default_rng(seed)
    boot_shares = np.zeros((n_boot, len(names)))
    boot_total = np.zeros(n_boot)
    for b in range(n_boot):
        rows = {e: rng.integers(0, len(identity[e]), len(identity[e])) for e in inputs.folds}
        draw = all_subset_values(inputs, names, rows, RIDGE_ALPHA_SHARES)
        sh = shapley_values(draw, names)
        boot_shares[b] = [sh[n] / sum(sh.values()) for n in names]
        boot_total[b] = draw[full]
    rows_out: list[dict[str, object]] = []
    for i, name in enumerate(names):
        rows_out.append({"block": name, "r2_alone": values[frozenset({name})], "share": shapley[name] / total,
                         "share_lo90": float(np.percentile(boot_shares[:, i], INTERVAL_LOW_PCT)),
                         "share_hi90": float(np.percentile(boot_shares[:, i], INTERVAL_HIGH_PCT)),
                         "unique_r2": values[full] - values[full - {name}]})
    table = pd.DataFrame(rows_out).sort_values("share", ascending=False).reset_index(drop=True)
    table["total_oos_r2"] = values[full]
    table["total_oos_r2_lo90"] = float(np.percentile(boot_total, INTERVAL_LOW_PCT))
    table["total_oos_r2_hi90"] = float(np.percentile(boot_total, INTERVAL_HIGH_PCT))
    return table


# ---------------------------------------------------------------------------------------------------------------------
# 6.2 Ablations: refit M1 without one block at a time.
# ---------------------------------------------------------------------------------------------------------------------

def loeo_predictions(study: StudyData, spec: PlSpec) -> dict[int, np.ndarray]:
    """Held-out utilities of one Plackett-Luce specification, leave-one-edition-out with nested lambda tuning."""
    folds = list(study.labelled_editions)
    out: dict[int, np.ndarray] = {}
    for held in folds:
        utilities, _ = cv.predict_pl(study, spec, [e for e in folds if e != held], [held])
        out[held] = utilities[held]
    return out


def fold_metrics(study: StudyData, utilities: dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for fold, u in utilities.items():
        metrics = cv.edition_metrics(u, baselines.points_of(study, fold))
        rows.append({"fold": fold, **metrics})
    return pd.DataFrame(rows).set_index("fold")


def ablation_table(study: StudyData, reference: pd.DataFrame) -> pd.DataFrame:
    """`reference` is M1's per-fold metrics (index fold). One row per block dropped."""
    full_columns = m1_columns_v2()
    rows: list[dict[str, object]] = []
    for block, dropped in BLOCKS_V2.items():
        columns = tuple(c for c in full_columns if c not in dropped)
        metrics = fold_metrics(study, loeo_predictions(study, pl_spec_for(f"M1-minus-{block}", columns)))
        delta = metrics["heldout_ll"] - reference["heldout_ll"]
        rows.append({"block": block, "mean_delta_ll": float(delta.mean()), "folds_worse": int((delta < 0).sum()),
                     "matters": bool((delta < 0).sum() >= WORSE_FOLDS_REQUIRED),
                     "mean_delta_winner_rank": float((metrics["winner_rank"] - reference["winner_rank"]).mean()),
                     **{f"delta_ll_{fold}": float(v) for fold, v in delta.items()}})
    return pd.DataFrame(rows).sort_values("mean_delta_ll").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# 6.2 (supplement) TreeSHAP on L1.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapResult:
    table: pd.DataFrame
    values: np.ndarray  # rows x features
    features: np.ndarray  # rows x features
    columns: tuple[str, ...]


def shap_l1(study: StudyData) -> ShapResult:
    """L1 fitted on all labelled editions; SHAP values are LightGBM's exact tree contributions (in-sample, descriptive)."""
    from lightgbm import LGBMRanker

    folds = list(study.labelled_editions)
    columns = tuple(study.features.base_columns)
    mats = baselines.design_matrices(study, "t1", list(columns), False, folds, folds)
    x = np.vstack([mats[e] for e in folds])
    y = np.concatenate([baselines.relevance_labels(baselines.points_of(study, e)) for e in folds])
    model = LGBMRanker(**LGBM_PARAMS).fit(x, y, group=[len(mats[e]) for e in folds])
    contributions = model.predict(x, pred_contrib=True)[:, :-1]
    table = pd.DataFrame({"feature": columns, "mean_abs_shap": np.abs(contributions).mean(axis=0)})
    return ShapResult(table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True), contributions, x, columns)


# ---------------------------------------------------------------------------------------------------------------------
# 6.3 The fame finding.
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FamePlayers:
    frame: pd.DataFrame  # edition, player, predicted rank, fame residual, actual rank


def football_only_fame_players(study: StudyData) -> FamePlayers:
    """Football-only PL in leave-one-edition-out; for each held-out player the predicted rank, the fame residualised on
    football features in the training fold, and the actual finish."""
    folds = list(study.labelled_editions)
    columns = tuple(c for c in m1_columns_v2() if c in columns_of(FOOTBALL_BLOCKS))
    spec = pl_spec_for("football-only", columns)
    frame = study.frame_t1
    parts: list[pd.DataFrame] = []
    for held in folds:
        train_eds = [e for e in folds if e != held]
        utilities, _ = cv.predict_pl(study, spec, train_eds, [held])
        model = fit_residualiser(frame[frame.edition.isin(train_eds)], "F9_fame", FOOTBALL_COLUMNS_FOR_FAME)
        rows = frame[frame.edition == held]
        points = baselines.points_of(study, held)
        part = pd.DataFrame({"edition": held, "player_name": rows.player_name.to_numpy(),
                             "predicted_rank": pd.Series(utilities[held]).rank(ascending=False, method="first").to_numpy(),
                             "fame_residual": apply_residualiser(rows, model, "F9_fame").to_numpy(),
                             "actual_rank": competition_rank_of(points)})
        parts.append(part)
    return FamePlayers(pd.concat(parts, ignore_index=True))


def permutation_pvalue(ranks: np.ndarray, high: np.ndarray, editions: np.ndarray, n_perm: int, seed: int) -> tuple[float, float, float]:
    """Observed gap in mean actual rank (low-fame minus high-fame), one-sided and two-sided permutation p-values.
    Labels are shuffled within edition."""
    def gap(labels: np.ndarray) -> float:
        return float(ranks[~labels].mean() - ranks[labels].mean())

    observed = gap(high)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(editions == e) for e in np.unique(editions)]
    gaps = np.empty(n_perm)
    for k in range(n_perm):
        shuffled = high.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(high[idx])
        gaps[k] = gap(shuffled)
    return observed, float((gaps >= observed).mean()), float((np.abs(gaps) >= abs(observed)).mean())


def fame_test_table(players: FamePlayers, n_perm: int, seed: int) -> pd.DataFrame:
    selected = players.frame[players.frame.predicted_rank <= FOOTBALL_TOP_N].copy()
    selected["high_fame"] = selected.fame_residual >= 0
    eras = {"all nine editions": selected.edition.unique(), "2016-2021": CALENDAR_ERA_FOLDS, "2022-2025": SEASON_ERA_FOLDS}
    rows: list[dict[str, object]] = []
    for label, members in eras.items():
        sub = selected[selected.edition.isin(members)]
        if sub.high_fame.nunique() < 2:
            continue
        observed, p_one, p_two = permutation_pvalue(sub.actual_rank.to_numpy(), sub.high_fame.to_numpy(), sub.edition.to_numpy(), n_perm, seed)
        for name, flag in (("fame at or above expected", True), ("fame below expected", False)):
            group = sub[sub.high_fame == flag]
            rows.append({"era": label, "group": name, "n": len(group), "mean_actual_rank": float(group.actual_rank.mean()),
                         "share_top10_finish": float((group.actual_rank <= FOOTBALL_TOP_N).mean()),
                         "gap_low_minus_high": observed, "p_one_sided": p_one, "p_two_sided": p_two})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CoefficientResult:
    table: pd.DataFrame
    lam: float


def final_coefficients(study: StudyData, columns: tuple[str, ...], name: str, n_boot: int, seed: int) -> CoefficientResult:
    """Final fit of a PL specification on all labelled editions, with Laplace and within-edition bootstrap intervals."""
    fit = inference.final_fit(study, pl_spec_for(name, columns))
    table = inference.laplace_summary(fit)
    boot = inference.bootstrap_betas(fit, study, n_boot, np.random.default_rng(seed))
    table["bootstrap_lo90"] = np.percentile(boot, INTERVAL_LOW_PCT, axis=0)
    table["bootstrap_hi90"] = np.percentile(boot, INTERVAL_HIGH_PCT, axis=0)
    table.insert(0, "model", name)
    table["lambda"] = fit.lam
    table["editions"] = f"{min(study.labelled_editions)}-{max(study.labelled_editions)} ({len(study.labelled_editions)})"
    return CoefficientResult(table, fit.lam)


def attention_first_time_test(study: StudyData, n_boot: int, seed: int) -> pd.DataFrame:
    columns = m1_columns_v2() + ("season_attention_x_first_time",)
    table = final_coefficients(study, columns, "M1+attention_x_first_time", n_boot, seed).table
    return table[table.feature.isin(["season_attention_x_first_time", "F10_season_attention"])].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# 6.4 Post-window information (2022-2025, T2).
# ---------------------------------------------------------------------------------------------------------------------

def post_window_test(study_t2: StudyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """M2 versus M1 in the four-fold design. Returns per-fold rows and the P1-P3 fold coefficients."""
    results = cv.run_cross_validation(study_t2, ["M1", "M2"])
    ll = results.metrics[results.metrics.metric == "heldout_ll"].pivot(index="fold", columns="model", values="value")
    ll["m2_minus_m1"] = ll["M2"] - ll["M1"]
    coefficients = results.coefficients
    post = coefficients[(coefficients.model == "M2") & (coefficients.feature.str.startswith("P"))]
    wide = post.pivot(index="fold", columns="feature", values="beta")
    per_fold = ll.join(wide).reset_index()
    return per_fold, post


# ---------------------------------------------------------------------------------------------------------------------
# 6.5 Tournament check.
# ---------------------------------------------------------------------------------------------------------------------

TOURNAMENT_MODELS: dict[str, tuple[str, ...]] = {
    "M1 without F11": tuple(c for c in m1_columns_v2() if c != "F11_individual_awards"),
    "M1": m1_columns_v2(),
    "M1 with tournament interaction": m1_columns_v2() + (TOURNAMENT_INTERACTION,),
}


def tournament_residuals(study: StudyData, predictions: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    """Held-out residual = actual rank minus predicted rank, for players who played a major tournament in a tournament edition."""
    frame = study.frame_t1
    rows: list[pd.DataFrame] = []
    for model, utilities in predictions.items():
        for edition in TOURNAMENT_EDITIONS:
            mask = ((frame.edition == edition) & (frame.is_major_tournament == 1)).to_numpy()
            in_edition = (frame.edition == edition).to_numpy()
            predicted = pd.Series(utilities[edition]).rank(ascending=False, method="first").to_numpy()
            actual = competition_rank_of(baselines.points_of(study, edition))
            played = mask[in_edition]
            rows.append(pd.DataFrame({"model": model, "edition": edition, "player_name": frame.player_name[in_edition].to_numpy()[played],
                                      "is_world_cup": frame.is_world_cup[in_edition].to_numpy()[played],
                                      "predicted_rank": predicted[played], "actual_rank": actual[played],
                                      "residual": actual[played] - predicted[played]}))
    return pd.concat(rows, ignore_index=True)


def tournament_summary(residuals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in residuals.groupby("model", sort=False):
        for label, sub in (("all tournament editions", group), ("World Cup editions (2018, 2023)", group[group.edition.isin(WORLD_CUP_EDITIONS)]),
                           ("Euro and Copa editions", group[~group.edition.isin(WORLD_CUP_EDITIONS)])):
            rows.append({"model": model, "subset": label, "n_players": len(sub), "mean_residual": float(sub.residual.mean()),
                         "mean_abs_residual": float(sub.residual.abs().mean())})
    return pd.DataFrame(rows)


def golden_ball_coefficients(study: StudyData, n_boot: int, seed: int) -> pd.DataFrame:
    """F11 and a separate golden-ball term estimated with two World Cups (2016-2025) and with one (2022-2025, the v1 situation)."""
    columns = m1_columns_v2() + ("golden_ball_score",)
    pieces: list[pd.DataFrame] = []
    for label, editions in (("two World Cups (2018, 2023)", study.labelled_editions), ("one World Cup (2023)", SEASON_ERA_FOLDS)):
        restricted = replace(study, labelled_editions=tuple(editions))
        table = final_coefficients(restricted, columns, "M1+golden_ball", n_boot, seed).table
        table["world_cups"] = label
        pieces.append(table[table.feature.isin(["F11_individual_awards", "golden_ball_score"])])
    return pd.concat(pieces, ignore_index=True)


def tournament_interaction_coefficient(study: StudyData, n_boot: int, seed: int) -> pd.DataFrame:
    table = final_coefficients(study, m1_columns_v2() + (TOURNAMENT_INTERACTION,), "M1+tourn_ko_x_world_cup", n_boot, seed).table
    return table[table.feature.isin([TOURNAMENT_INTERACTION, "F3_big_stage"])].reset_index(drop=True)
