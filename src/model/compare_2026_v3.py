"""Experiment 3, section 4: EXPLORATORY comparison of the v1, v2 (locked), v2.1 and v3 forecasts with the published 2026 top ten.

Runs after the forecasts are written. The published top ten is a constant here, not a file in data/holdout, so the lock-in guard in
forecast_v2.refuse_if_result_present stays intact for the forecast code."""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PUBLISHED_TOP10: tuple[str, ...] = ("Kane", "Mbappé", "Yamal", "Rodri", "Messi", "Kvaratskhelia", "Dembélé", "Haaland", "Olise", "Ruiz")
TOP_N_SHORT, TOP_N_LONG = 3, 10
FORECAST_FILES: dict[str, Path] = {
    "v1": Path("results") / "forecast_2026_T1_ballots.csv",
    "v2 (locked)": Path("results") / "v2" / "forecast_2026_v2_T1.csv",
    "v2.1": Path("results") / "v2_1" / "forecast_2026_v2_1_T1.csv",
    "v3 (season only)": Path("results") / "v3" / "forecast_2026_v3_T1.csv",
}
WINNER = PUBLISHED_TOP10[0]


def fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn").lower()


def actual_rank_by_player(table: pd.DataFrame) -> pd.Series:
    """Actual rank (1-10) for each player_key in the published top ten; every published name must match exactly one nominee."""
    tokens = table.player_name.map(lambda n: set(fold_accents(n).split()))
    ranks: dict[str, int] = {}
    for rank, short in enumerate(PUBLISHED_TOP10, start=1):
        hits = table.index[tokens.map(lambda t: fold_accents(short) in t)]
        if len(hits) != 1:
            raise ValueError(f"published name {short!r} matched {len(hits)} nominees: {list(table.loc[hits, 'player_name'])}")
        ranks[table.loc[hits[0], "player_key"]] = rank
    return pd.Series(ranks)


def compare_one(version: str, table: pd.DataFrame) -> dict[str, object]:
    actual = actual_rank_by_player(table)
    by_rank = table.sort_values("forecast_rank")
    winner_key = actual.index[actual == 1][0]
    actual_top3 = set(actual.index[actual <= TOP_N_SHORT])
    in_actual = table.set_index("player_key").loc[actual.index]
    rho = spearmanr(in_actual.forecast_rank.to_numpy(), actual.to_numpy())[0]
    return {"version": version, "pick": by_rank.player_name.iloc[0], "pick_p_win": float(by_rank.p_win.iloc[0]),
            f"{WINNER}_predicted_rank": int(in_actual.loc[winner_key, "forecast_rank"]), f"{WINNER}_p_win": float(in_actual.loc[winner_key, "p_win"]),
            "top3_overlap": len(set(by_rank.player_key.head(TOP_N_SHORT)) & actual_top3),
            "top10_overlap": len(set(by_rank.player_key.head(TOP_N_LONG)) & set(actual.index)),
            "spearman_on_actual_top10": float(rho)}


def main(root: Path) -> None:
    rows = [compare_one(v, pd.read_csv(root / path, comment="#")) for v, path in FORECAST_FILES.items()]
    out = pd.DataFrame(rows)
    out.to_csv(root / "results" / "v3" / "comparison_2026_all_versions.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
