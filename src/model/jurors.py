"""Juror panel construction from the dataset columns (spec 6.3 fallback; see deviations.md S1, S2, S8).

No individual ballots or juror nation lists exist, so the panel is rebuilt from `jurors_same_nation`,
`jurors_same_confed` and `jurors_league_country`. The bias parameters are PLACEHOLDERS, not estimates.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.data import T1_FILE

UEFA, CONMEBOL, CAF, AFC, CONCACAF = "UEFA", "CONMEBOL", "CAF", "AFC", "CONCACAF"
UNASSIGNED = "UNASSIGNED"
CONFEDERATION: dict[str, str] = {
    **{n: UEFA for n in ("Belgium", "Croatia", "England", "France", "Georgia", "Germany", "Italy", "Netherlands", "Norway",
                         "Poland", "Portugal", "Scotland", "Serbia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine")},
    **{n: CONMEBOL for n in ("Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay")},
    **{n: CAF for n in ("Algeria", "Cameroon", "Egypt", "Guinea", "Ivory Coast", "Morocco", "Nigeria", "Senegal")},
    **{n: AFC for n in ("South Korea", "Saudi Arabia")},
    **{n: CONCACAF for n in ("Mexico", "USA")},
}
LEAGUE_COUNTRY: dict[int, str] = {47: "England", 53: "France", 54: "Germany", 55: "Italy", 57: "Netherlands", 61: "Portugal",
                                  87: "Spain", 130: "USA", 536: "Saudi Arabia"}
JURORS_BY_EDITION: dict[int, int] = {2022: 93, 2023: 92, 2024: 99, 2025: 100, 2026: 100}
JUROR_COLUMNS: tuple[str, ...] = ("edition", "player_key", "nationality", "league_key", "jurors_same_nation",
                                  "jurors_same_confed", "jurors_league_country")
PLACEHOLDER_LABEL = "PLACEHOLDER (no juror ballots available; not estimated)"


@dataclass(frozen=True)
class BiasParams:
    b_home: float
    b_confed: float
    b_league: float


PLACEHOLDER_BIAS = BiasParams(b_home=0.8, b_confed=0.2, b_league=0.2)


@dataclass(frozen=True)
class Juror:
    nation: str | None  # None = a juror whose nation is not a nominee or league nation
    confederation: str


@dataclass(frozen=True)
class SimInputs:
    nation_match: np.ndarray  # jurors x players, bool
    confed_match: np.ndarray
    league_match: np.ndarray
    n_jurors: int


def load_juror_inputs(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / T1_FILE, usecols=list(JUROR_COLUMNS))


def player_confederation(nationality: str) -> str:
    if nationality not in CONFEDERATION:
        raise KeyError(f"no confederation for nationality {nationality!r}")
    return CONFEDERATION[nationality]


def league_country(league_key: int) -> str:
    if league_key not in LEAGUE_COUNTRY:
        raise KeyError(f"no country for league id {league_key}")
    return LEAGUE_COUNTRY[league_key]


def confederation_counts(players: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, set[int]] = {}
    for nationality, count in zip(players["nationality"], players["jurors_same_confed"]):
        counts.setdefault(player_confederation(nationality), set()).add(int(count))
    ambiguous = {c: v for c, v in counts.items() if len(v) > 1}
    if ambiguous:
        raise ValueError(f"inconsistent juror counts per confederation: {ambiguous}")
    return {c: next(iter(v)) for c, v in counts.items()}


def build_jurors(players: pd.DataFrame, n_jurors: int) -> list[Juror]:
    """One juror per nominee nationality and per league country (their nations are named); the rest of each
    confederation's jurors are anonymous; the shortfall to n_jurors is unassigned."""
    counts = confederation_counts(players)
    nominee_nations = sorted(set(players["nationality"]))
    league_nations = sorted({league_country(int(k)) for k in players["league_key"]})
    named = sorted(set(nominee_nations) | set(league_nations))
    jurors: list[Juror] = []
    for confed in sorted(set(counts) | {player_confederation(n) for n in named}):
        members = [n for n in named if player_confederation(n) == confed]
        jurors += [Juror(nation=n, confederation=confed) for n in members]
        if confed in counts:
            extra = counts[confed] - len(members)
            if extra < 0:
                raise ValueError(f"{confed}: {len(members)} named jurors exceed the reported count {counts[confed]}")
            jurors += [Juror(nation=None, confederation=confed) for _ in range(extra)]
    shortfall = n_jurors - len(jurors)
    if shortfall < 0:
        raise ValueError(f"constructed {len(jurors)} jurors, more than the panel size {n_jurors}")
    return jurors + [Juror(nation=None, confederation=UNASSIGNED) for _ in range(shortfall)]


def build_sim_inputs(players: pd.DataFrame, n_jurors: int) -> SimInputs:
    jurors = build_jurors(players, n_jurors)
    nationality = players["nationality"].to_numpy()
    player_confed = np.array([player_confederation(n) for n in nationality])
    player_league = np.array([league_country(int(k)) for k in players["league_key"]])
    juror_nation = np.array([j.nation if j.nation is not None else "" for j in jurors])
    juror_confed = np.array([j.confederation for j in jurors])
    return SimInputs(
        nation_match=juror_nation[:, None] == nationality[None, :],
        confed_match=juror_confed[:, None] == player_confed[None, :],
        league_match=juror_nation[:, None] == player_league[None, :],
        n_jurors=len(jurors),
    )


def edition_players(juror_inputs: pd.DataFrame, edition: int) -> pd.DataFrame:
    return juror_inputs[juror_inputs["edition"] == edition].reset_index(drop=True)
