"""Stage 4, research tests on the ten-edition study (experiments spec sections 6.2-6.5).

Reads no 2026 output. Every table is written as soon as it is computed."""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model import research_v2 as rs  # noqa: E402
from src.model.data import load_study_t2_v2, load_study_v2  # noqa: E402
from src.model.features_v2 import m1_columns_v2  # noqa: E402

OUT_DIR = Path("results") / "v2"
FIG_DIR = Path("figures") / "v2"
N_BOOT_COEF = 500
SEED = 7
SHAP_TOP_N = 14


def log(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


def plot_shares(shares: pd.DataFrame, ablations: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    order = shares.iloc[::-1]
    axes[0].barh(order.block, order.share * 100, xerr=[(order.share - order.share_lo90) * 100, (order.share_hi90 - order.share) * 100], capsize=3)
    axes[0].set_xlabel("share of out-of-sample R2 (%), 90% bootstrap interval")
    axes[0].set_title("Variance shares by block")
    abl = ablations.iloc[::-1]
    axes[1].barh(abl.block, abl.mean_delta_ll, color=["#c0392b" if m else "#7f8c8d" for m in abl.matters])
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("change in held-out log-likelihood when the block is dropped\n(red: worse in at least 6 of 9 folds)")
    axes[1].set_title("Ablations")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fame(players: rs.FamePlayers, path: Path) -> None:
    selected = players.frame[players.frame.predicted_rank <= rs.FOOTBALL_TOP_N].copy()
    selected["high_fame"] = selected.fame_residual >= 0
    eras = {"all nine editions": selected, "2016-2021": selected[selected.edition.isin(rs.CALENDAR_ERA_FOLDS)],
            "2022-2025": selected[selected.edition.isin(rs.SEASON_ERA_FOLDS)]}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.35
    for i, (label, sub) in enumerate(eras.items()):
        high, low = sub[sub.high_fame].actual_rank, sub[~sub.high_fame].actual_rank
        ax.bar(i - width / 2, high.mean(), width, color="#2c7fb8", label="fame at or above expected" if i == 0 else None)
        ax.bar(i + width / 2, low.mean(), width, color="#d95f0e", label="fame below expected" if i == 0 else None)
        ax.text(i - width / 2, high.mean() + 0.2, f"n={len(high)}", ha="center", fontsize=8)
        ax.text(i + width / 2, low.mean() + 0.2, f"n={len(low)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(eras)), list(eras))
    ax.set_ylabel("mean actual finishing rank (lower is better)")
    ax.set_title("Players the football model puts in its top 10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shap(result: rs.ShapResult, path: Path) -> None:
    order = result.table.feature.tolist()[:SHAP_TOP_N][::-1]
    index = {c: i for i, c in enumerate(result.columns)}
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for row, feature in enumerate(order):
        j = index[feature]
        values, raw = result.values[:, j], result.features[:, j]
        span = np.ptp(raw) or 1.0
        ax.scatter(values, row + rng.uniform(-0.3, 0.3, len(values)), c=(raw - raw.min()) / span, cmap="coolwarm", s=8, alpha=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("SHAP value (L1 ranker score)")
    ax.set_title("L1 TreeSHAP, all nine labelled editions (colour: feature value)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(root: Path) -> None:
    started = time.time()
    out, fig = root / OUT_DIR, root / FIG_DIR
    fig.mkdir(parents=True, exist_ok=True)
    study = load_study_v2(root)

    inputs = rs.build_share_inputs(study)
    shares = rs.variance_shares(inputs, rs.N_BOOT_SHARES, SEED)
    shares.to_csv(out / "variance_shares.csv", index=False)
    log("variance shares done", started)

    m1_by_fold = pd.read_csv(out / "metrics_by_fold.csv")
    m1 = m1_by_fold[m1_by_fold.model == "M1"].pivot(index="fold", columns="metric", values="value")
    ablations = rs.ablation_table(study, m1)
    ablations.to_csv(out / "ablations.csv", index=False)
    plot_shares(shares, ablations, fig / "fig1_shares.pdf")
    log("ablations done", started)

    shap = rs.shap_l1(study)
    shap.table.to_csv(out / "shap_L1.csv", index=False)
    plot_shap(shap, fig / "shap_L1_beeswarm.pdf")
    log("SHAP done", started)

    players = rs.football_only_fame_players(study)
    players.frame.to_csv(out / "fame_players.csv", index=False)
    fame = rs.fame_test_table(players, rs.N_PERMUTATIONS, SEED)
    fame.to_csv(out / "fame_test.csv", index=False)
    plot_fame(players, fig / "fig2_fame.pdf")
    rs.attention_first_time_test(study, N_BOOT_COEF, SEED).to_csv(out / "attention_first_time.csv", index=False)
    log("fame test done", started)

    study_t2 = load_study_t2_v2(root)
    per_fold, _ = rs.post_window_test(study_t2)
    per_fold.to_csv(out / "post_window_test.csv", index=False)
    log("post-window test done", started)

    predictions = {name: rs.loeo_predictions(study, rs.pl_spec_for(name, columns)) for name, columns in rs.TOURNAMENT_MODELS.items()}
    residuals = rs.tournament_residuals(study, predictions)
    residuals.to_csv(out / "tournament_residuals.csv", index=False)
    rs.tournament_summary(residuals).to_csv(out / "tournament_check.csv", index=False)
    pd.concat([rs.tournament_interaction_coefficient(study, N_BOOT_COEF, SEED), rs.golden_ball_coefficients(study, N_BOOT_COEF, SEED)],
              ignore_index=True).to_csv(out / "tournament_coefficients.csv", index=False)
    log("tournament check done", started)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
