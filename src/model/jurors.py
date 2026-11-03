"""Juror panels for the simulation (spec 6.3).

`build_jurors` / `build_sim_inputs` rebuild a panel from summary columns and use PLACEHOLDER bias values; they were the
first run, written under the mistaken belief that no ballots existed (deviations.md S1, S8, B1). Ballots and the real
juror nation lists do exist (data/jury/): `real_jury` and `build_real_sim_inputs` use the real panel, and the bias values
for that panel are estimated from the ballots in `src/model/ballots.py`.
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
JURY_DIR = Path("data") / "jury"
VOTES_FILE = "fact_bdo_country_votes.parquet"
NATIONS_FILE = "fact_juror_nations.parquet"
COUNTRY_FILE = "dim_country.parquet"
PLAYER_FILE = "dim_player.parquet"
RESULTS_FILE = "fact_bdo_results.parquet"
LEAGUE_FILE = "dim_league_context.parquet"
CLUB_FILE = "dim_club.parquet"
JURY_FILES: tuple[str, ...] = (VOTES_FILE, NATIONS_FILE, COUNTRY_FILE, PLAYER_FILE, RESULTS_FILE, LEAGUE_FILE, CLUB_FILE)
# league_key 130 (MLS) is absent from dim_league_context; mapped by hand (deviation B3).
MANUAL_LEAGUE_COUNTRY: dict[str, str] = {"130": "c_usa"}
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


def load_juror_inputs_from(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(JUROR_COLUMNS))


def load_juror_inputs(root: Path) -> pd.DataFrame:
    return load_juror_inputs_from(root / T1_FILE)


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


@dataclass(frozen=True)
class RealJuror:
    country_key: str
    confederation: str


@dataclass(frozen=True)
class PlayerAttributes:
    player_keys: tuple[str, ...]  # snapshot row order for the edition
    nationality_country: tuple[str, ...]
    confederation: tuple[str, ...]
    league_country: tuple[str, ...]


def real_jury(root: Path, edition: int) -> list[RealJuror]:
    """The edition's juror nations from fact_juror_nations, sorted by country key. Past editions keep only the nations that
    voted (93/92/99/100); 2026 has `voted` unknown, so all 100 nations are returned."""
    nations = pd.read_parquet(root / JURY_DIR / NATIONS_FILE)
    rows = nations[(nations["edition"] == edition) & (nations["voted"] != False)].sort_values("country_key")  # noqa: E712
    return [RealJuror(country_key=c, confederation=f) for c, f in zip(rows["country_key"], rows["confederation"])]


def league_country_map(leagues: pd.DataFrame) -> dict[str, str]:
    """league_key (string) -> country_key. Rows with a null league_key (2 of 295) are dropped; the rest map one-to-one."""
    pairs = leagues[["league_key", "country_key"]].dropna(subset=["league_key"]).drop_duplicates()
    if pairs["league_key"].duplicated().any():
        raise ValueError("a league_key maps to more than one country in dim_league_context")
    return {**dict(zip(pairs["league_key"], pairs["country_key"])), **MANUAL_LEAGUE_COUNTRY}


def player_attributes(root: Path, edition: int) -> PlayerAttributes:
    """Nationality country, confederation and league country of an edition's nominees, in snapshot row order."""
    rows = edition_players(load_juror_inputs(root), edition)
    players = pd.read_parquet(root / JURY_DIR / PLAYER_FILE).set_index("player_key")["nationality_country_key"]
    confed = pd.read_parquet(root / JURY_DIR / COUNTRY_FILE).set_index("country_key")["confederation"]
    league_map = league_country_map(pd.read_parquet(root / JURY_DIR / LEAGUE_FILE))
    nationality = rows["player_key"].map(players)
    league = rows["league_key"].astype(int).astype(str).map(league_map)
    if nationality.isna().any() or league.isna().any():
        raise ValueError(f"edition {edition}: nominees without a nationality or league country")
    return PlayerAttributes(player_keys=tuple(rows["player_key"]), nationality_country=tuple(nationality),
                            confederation=tuple(nationality.map(confed)), league_country=tuple(league))


def build_real_sim_inputs(jury: list[RealJuror], players: PlayerAttributes) -> SimInputs:
    """Match matrices for the real panel. confed_match excludes home, as in the ballot model (spec section 2)."""
    juror_country = np.array([j.country_key for j in jury])
    juror_confed = np.array([j.confederation for j in jury])
    nation_match = juror_country[:, None] == np.array(players.nationality_country)[None, :]
    same_confed = juror_confed[:, None] == np.array(players.confederation)[None, :]
    return SimInputs(nation_match=nation_match, confed_match=same_confed & ~nation_match,
                     league_match=juror_country[:, None] == np.array(players.league_country)[None, :], n_jurors=len(jury))
