"""Per-player contributions and plain-language explanations of the model's ranking.

Utility u_i = sum_k beta_k * x_ik. Every feature is a within-edition z-score (flags aside), so a contribution is
relative to the average nominee of that edition. Contributions describe the model's arithmetic, not causes.
"""

import numpy as np
import pandas as pd

from src.model.data import StudyData
from src.model.inference import FinalFit

FEATURE_LABELS: dict[str, str] = {
    "F1_attack": "attacking output (npxG+xA per 90)",
    "F2_form_rating": "match ratings",
    "F3_big_stage": "big-stage output (UCL knockout, tournament, finals)",
    "F4_major_titles": "major titles (UCL / World Cup / continental)",
    "F5_other_honors": "other honors (league, cups, awards)",
    "F6_team_strength": "team strength (club rating, UCL and tournament stage)",
    "F7_prior_standing": "prior Ballon d'Or standing",
    "F8_defending_winner": "being the defending winner",
    "F9_fame": "baseline fame (pre-season Wikipedia attention)",
    "F10_season_attention": "season attention beyond what performance explains",
    "is_defender": "playing as a defender",
    "is_goalkeeper": "playing as a goalkeeper",
    "P1_post_form": "post-window form",
    "P2_post_attention": "post-window attention change",
    "P3_final_spike": "attention spike in the last 14 days",
}
FRAMING = ("These are the model's arithmetic, not causes. Every feature is a within-edition z-score, so a contribution is "
           "measured against the average 2026 nominee. The explanation is read-only: nothing here is used to tune or change the model.")


def contribution_table(study: StudyData, fit: FinalFit, edition: int) -> pd.DataFrame:
    """One row per nominee: beta_k * x_ik for each feature, their sum (the utility), rank and the plain
    Plackett-Luce first-pick probability (before any juror model)."""
    rows = study.frame_t1[study.frame_t1["edition"] == edition][["player_key", "player_name"]].reset_index(drop=True)
    contrib = pd.DataFrame(fit.matrices[edition] * fit.beta, columns=fit.columns)
    out = pd.concat([rows, contrib], axis=1)
    out["utility"] = contrib.sum(axis=1)
    exp_u = np.exp(out["utility"] - out["utility"].max())
    out["first_pick_probability"] = exp_u / exp_u.sum()
    out["rank_by_utility"] = out["utility"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank_by_utility").reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in ("player_key", "player_name", "utility", "first_pick_probability", "rank_by_utility")]


def shift_table(t1: pd.DataFrame, t2: pd.DataFrame) -> pd.DataFrame:
    """T1 to T2 change per player: rank, utility, first-pick probability, and the utility change split by feature."""
    a = t1.set_index("player_key")
    b = t2.set_index("player_key")
    shared = [c for c in feature_columns(t2) if c in feature_columns(t1)]
    post_only = [c for c in feature_columns(t2) if c not in feature_columns(t1)]
    out = pd.DataFrame({"player_name": a["player_name"], "rank_T1": a["rank_by_utility"], "rank_T2": b["rank_by_utility"],
                        "utility_T1": a["utility"], "utility_T2": b["utility"],
                        "pwin_T1": a["first_pick_probability"], "pwin_T2": b["first_pick_probability"]})
    out["rank_change"] = out["rank_T1"] - out["rank_T2"]  # positive = moves up in T2
    out["utility_change"] = out["utility_T2"] - out["utility_T1"]
    for col in shared:
        out[f"delta_{col}"] = b[col] - a[col]
    for col in post_only:
        out[f"delta_{col}"] = b[col]
    return out.sort_values("rank_T2").reset_index(drop=True)


def explain_pair(table: pd.DataFrame, higher: str, lower: str) -> pd.DataFrame:
    """Which features account for the utility gap between two players (rows sum to the gap)."""
    cols = feature_columns(table)
    a = table[table["player_name"] == higher]
    b = table[table["player_name"] == lower]
    if len(a) != 1 or len(b) != 1:
        raise ValueError(f"expected exactly one row each for {higher!r} and {lower!r}")
    diff = pd.DataFrame({"feature": cols, f"{higher}": a[cols].iloc[0].to_numpy(), f"{lower}": b[cols].iloc[0].to_numpy()})
    diff["gap"] = diff[higher] - diff[lower]
    return diff.reindex(diff["gap"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def relative_contributions(table: pd.DataFrame) -> pd.DataFrame:
    """Each contribution minus the mean contribution of the pool (the average nominee)."""
    cols = feature_columns(table)
    return table[cols] - table[cols].mean()


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def player_explanation(table: pd.DataFrame, rel: pd.DataFrame, position: int, n_terms: int) -> str:
    row = table.iloc[position]
    contrib = rel.iloc[position].sort_values()
    boosts = contrib[contrib > 0].sort_values(ascending=False).head(n_terms)
    drags = contrib[contrib < 0].head(n_terms)
    boost_text = "; ".join(f"{label(f)} ({v:+.2f})" for f, v in boosts.items()) or "none"
    drag_text = "; ".join(f"{label(f)} ({v:+.2f})" for f, v in drags.items()) or "none"
    return (f"**#{int(row['rank_by_utility'])} {row['player_name']}**: utility {row['utility']:+.2f} "
            f"(plain first-pick probability {row['first_pick_probability']:.1%}). "
            f"Pushing up, relative to the average nominee: {boost_text}. Pulling down: {drag_text}.")


def explanations_markdown(t1: pd.DataFrame, t2: pd.DataFrame, shifts: pd.DataFrame, header: list[str], top_n: int, n_terms: int) -> str:
    lines = ["# Why the model ranked 2026 this way", "", *header, "", FRAMING, ""]
    for name, table, model in (("T1 (M1, in-window information)", t1, "M1"), ("T2 (M2, with post-window information)", t2, "M2")):
        rel = relative_contributions(table)
        lines += [f"## {name}", ""]
        lines += [f"- {player_explanation(table, rel, i, n_terms)}" for i in range(top_n)]
        lines += [""]
        gap = explain_pair(table, table.iloc[0]["player_name"], table.iloc[1]["player_name"])
        top_gap = gap.head(4)
        lines += [f"Why #1 {table.iloc[0]['player_name']} is above #2 {table.iloc[1]['player_name']} in {model}: "
                  + "; ".join(f"{label(r['feature'])} ({r['gap']:+.2f})" for _, r in top_gap.iterrows())
                  + f" (total gap {gap['gap'].sum():+.2f} utility).", ""]
    movers = shifts.reindex(shifts["rank_change"].abs().sort_values(ascending=False).index).head(5)
    lines += ["## Biggest movers between T1 and T2", ""]
    for _, r in movers.iterrows():
        delta_cols = [c for c in shifts.columns if c.startswith("delta_")]
        parts = pd.Series({c[6:]: r[c] for c in delta_cols})
        top = parts.reindex(parts.abs().sort_values(ascending=False).index).head(3)
        lines.append(f"- {r['player_name']}: rank {int(r['rank_T1'])} -> {int(r['rank_T2'])}; utility change {r['utility_change']:+.2f}, mainly "
                     + "; ".join(f"{label(f)} ({v:+.2f})" for f, v in top.items()) + ".")
    lines += [""]
    return "\n".join(lines)

