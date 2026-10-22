"""Figures for the ballot-based juror simulation (spec section 7)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model import ballots as bl

TOP_N = 10
COLOR_ORIGINAL = "#4C78A8"
COLOR_BALLOTS = "#F58518"
COLOR_TENPICK = "#54A24B"
FIGURE_DPI = 150


def plot_win_probability(comparison: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True)
    for ax, snapshot in zip(axes, sorted(comparison.snapshot.unique())):
        top = comparison[comparison.snapshot == snapshot].nlargest(TOP_N, "p_win_ballots").iloc[::-1]
        y = np.arange(len(top))
        ax.barh(y - 0.2, top.p_win_original, height=0.4, color=COLOR_ORIGINAL, label="Original (placeholder jury)")
        ax.barh(y + 0.2, top.p_win_ballots, height=0.4, color=COLOR_BALLOTS, label="Ballot-based (real jury)")
        ax.set_yticks(y)
        ax.set_yticklabels(top.player_name)
        ax.set_title(f"{snapshot}: probability of winning the 2026 Ballon d'Or")
        ax.set_xlabel("P(win)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_bias(estimates: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    specs = [("main", "All four editions (2022-25)", COLOR_BALLOTS, -0.12), ("10pick_only", "10-pick editions only (2024-25)", COLOR_TENPICK, 0.12)]
    positions = np.arange(len(bl.INDICATORS))
    for spec, label, color, offset in specs:
        rows = estimates[estimates.spec == spec].set_index("indicator").loc[list(bl.INDICATORS)]
        ax.errorbar(rows.beta, positions + offset, xerr=[rows.beta - rows.boot_lo90, rows.boot_hi90 - rows.beta], fmt="o", color=color,
                    capsize=3, label=label)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_yticks(positions)
    ax.set_yticklabels(["Same country (home)", "Same confederation", "Same league country"])
    ax.invert_yaxis()
    ax.set_xlabel("Shift in ballot utility (Gumbel scale), 90% bootstrap interval")
    ax.set_title("How juror nationality shifts a ballot")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)
