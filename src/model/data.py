"""Loading, integrity checks and the study data container.

This is the only module besides the later modelling code that touches the outcome columns. They are
kept in a separate frame so the feature code never sees them.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.model.awards_v2_1 import EVENTS_FILE, attach_award_columns
from src.model.features import EDITION_COL, V1_FEATURES, FeatureSet, build_base_features, build_post_features
from src.model.features_v2 import (MAIN_OPTIONS, V2_FEATURES, V2_POST_FEATURES, FeatureOptions, build_base_features_with,
                                   build_post_features_v2)

T1_FILE = "player_snapshot_t1.csv"
T2_FILE = "player_snapshot_t2.csv"
V2_DIR = Path("data") / "v2"
V2_1_DIR = Path("data") / "v2_1"
KEY_COLS: tuple[str, ...] = ("edition", "player_key")
TARGET_COLS: tuple[str, ...] = ("bdo_points", "bdo_rank", "vote_share")
SNAPSHOT_COLS: tuple[str, ...] = ("snapshot", "snapshot_date")
IDENTITY_COLS: tuple[str, ...] = ("edition", "player_key", "player_name", "nationality", "position_group")
FORECAST_EDITION = 2026
TRAIN_EDITIONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
NOMINEES_PER_EDITION = 30


@dataclass(frozen=True)
class StudySpec:
    """Which editions a study contains. The forecast edition is the one with null targets."""
    t1_editions: tuple[int, ...]
    t2_editions: tuple[int, ...]
    forecast_edition: int

    @property
    def labelled_editions(self) -> tuple[int, ...]:
        return tuple(ed for ed in self.t1_editions if ed != self.forecast_edition)


V1_SPEC = StudySpec(t1_editions=(2022, 2023, 2024, 2025, 2026), t2_editions=(2022, 2023, 2024, 2025, 2026), forecast_edition=FORECAST_EDITION)
V2_T1_EDITIONS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026)
V2_SPEC = StudySpec(t1_editions=V2_T1_EDITIONS, t2_editions=V1_SPEC.t2_editions, forecast_edition=FORECAST_EDITION)


@dataclass(frozen=True)
class EditionFacts:
    """Edition facts from experiments spec section 2. `zero_point_nominees` is None for the forecast edition."""
    eligibility_format: str
    ballot_picks: int
    points_per_ballot: int
    jurors: int | None
    zero_point_nominees: int | None


CALENDAR = "calendar_year"
SEASON = "season"
POINTS_3_PICKS = 9
POINTS_5_PICKS = 16
POINTS_10_PICKS = 67
EDITION_FACTS: dict[int, EditionFacts] = {
    2016: EditionFacts(CALENDAR, 3, POINTS_3_PICKS, 173, 11),
    2017: EditionFacts(CALENDAR, 5, POINTS_5_PICKS, 176, 2),
    2018: EditionFacts(CALENDAR, 5, POINTS_5_PICKS, 180, 2),
    2019: EditionFacts(CALENDAR, 5, POINTS_5_PICKS, 176, 3),
    2021: EditionFacts(CALENDAR, 5, POINTS_5_PICKS, 170, 2),
    2022: EditionFacts(SEASON, 5, POINTS_5_PICKS, 93, 11),
    2023: EditionFacts(SEASON, 5, POINTS_5_PICKS, 92, 5),
    2024: EditionFacts(SEASON, 10, POINTS_10_PICKS, 99, 2),
    2025: EditionFacts(SEASON, 10, POINTS_10_PICKS, 100, 0),
    2026: EditionFacts(SEASON, 10, POINTS_10_PICKS, None, None),
}


@dataclass(frozen=True)
class StudyData:
    frame_t1: pd.DataFrame  # identity + base features, no outcome columns
    frame_t2: pd.DataFrame | None  # frame_t1 plus the post-window features; None when the study has no T2 rows
    targets: pd.DataFrame  # outcome columns, same index as the frames
    sha256_t1: str
    sha256_t2: str
    labelled_editions: tuple[int, ...]
    forecast_edition: int
    features: FeatureSet


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_edition_facts(t1: pd.DataFrame) -> None:
    """Assert the per-edition facts of spec section 2 against the data: format, picks, jurors, point sums, zero-point nominees."""
    for edition, group in t1.groupby("edition"):
        facts = EDITION_FACTS[int(edition)]
        if (group["eligibility_format"] != facts.eligibility_format).any():
            raise ValueError(f"{edition}: eligibility_format is not {facts.eligibility_format}")
        if (group["ballot_picks"] != facts.ballot_picks).any():
            raise ValueError(f"{edition}: ballot_picks is not {facts.ballot_picks}")
        if facts.jurors is None:
            continue
        if (group["jurors_voted"] != facts.jurors).any():
            raise ValueError(f"{edition}: jurors_voted is not {facts.jurors}")
        total = float(group["bdo_points"].sum())
        expected = facts.jurors * facts.points_per_ballot
        if total != expected:
            raise ValueError(f"{edition}: points sum to {total}, expected {facts.jurors} jurors x {facts.points_per_ballot} = {expected}")
        zero = int((group["bdo_points"] == 0).sum())
        if zero != facts.zero_point_nominees:
            raise ValueError(f"{edition}: {zero} zero-point nominees, expected {facts.zero_point_nominees}")


def check_t1(t1: pd.DataFrame, spec: StudySpec) -> None:
    if set(t1[EDITION_COL].unique()) != set(spec.t1_editions):
        raise ValueError("unexpected set of editions")
    if len(t1) != NOMINEES_PER_EDITION * len(spec.t1_editions):
        raise ValueError(f"expected {NOMINEES_PER_EDITION} rows per edition, got {len(t1)} (T1)")
    if t1.duplicated(list(KEY_COLS)).any():
        raise ValueError("T1: duplicate (edition, player_key)")
    forecast = t1[t1[EDITION_COL] == spec.forecast_edition]
    if not forecast[list(TARGET_COLS)].isna().all().all():
        raise ValueError(f"{spec.forecast_edition} target columns must be entirely null")
    labelled = t1[t1[EDITION_COL] != spec.forecast_edition]
    if labelled[list(TARGET_COLS)].isna().any().any():
        raise ValueError("labelled editions must have no null targets")


def check_snapshots(t1: pd.DataFrame, t2: pd.DataFrame, spec: StudySpec) -> None:
    check_t1(t1, spec)
    if set(t2[EDITION_COL].unique()) != set(spec.t2_editions):
        raise ValueError("unexpected set of editions")
    if len(t2) != NOMINEES_PER_EDITION * len(spec.t2_editions):
        raise ValueError(f"expected {NOMINEES_PER_EDITION} rows per edition, got {len(t2)} (T2)")
    if t2.duplicated(list(KEY_COLS)).any():
        raise ValueError("T2: duplicate (edition, player_key)")
    t1_overlap = t1[t1[EDITION_COL].isin(spec.t2_editions)].reset_index(drop=True)
    if not (t1_overlap[list(KEY_COLS)].to_numpy() == t2[list(KEY_COLS)].to_numpy()).all():
        raise ValueError("T1 and T2 rows are not in the same order")
    shared = [c for c in t1.columns if c not in SNAPSHOT_COLS and c in t2.columns]
    if not t1_overlap[shared].equals(t2[shared]):
        raise ValueError("in-window columns differ between T1 and T2")
    forecast = t2[t2[EDITION_COL] == spec.forecast_edition]
    if not forecast[list(TARGET_COLS)].isna().all().all():
        raise ValueError(f"{spec.forecast_edition} target columns must be entirely null in T2")


def load_study(path_t1: Path, path_t2: Path, spec: StudySpec) -> StudyData:
    """Load a study whose T1 and T2 cover the same editions (v1 and the bridge run).

    T1 rows outside `spec.t1_editions` are dropped, so the expanded file can be restricted to 2022-2026."""
    if spec.t1_editions != spec.t2_editions:
        raise ValueError("load_study needs identical T1 and T2 editions; the ten-edition study has its own loader")
    t1 = pd.read_csv(path_t1)
    t1 = t1[t1[EDITION_COL].isin(spec.t1_editions)].reset_index(drop=True)
    t2 = pd.read_csv(path_t2)
    check_snapshots(t1, t2, spec)
    identity = t1[list(IDENTITY_COLS)]
    base = build_base_features(t1)
    post = build_post_features(t2)
    frame_t1 = pd.concat([identity, base.drop(columns=[EDITION_COL])], axis=1)
    frame_t2 = pd.concat([frame_t1, post], axis=1)
    targets = t1[list(TARGET_COLS)].copy()
    return StudyData(frame_t1=frame_t1, frame_t2=frame_t2, targets=targets,
                     sha256_t1=sha256_of(path_t1), sha256_t2=sha256_of(path_t2),
                     labelled_editions=spec.labelled_editions, forecast_edition=spec.forecast_edition, features=V1_FEATURES)


def load_study_data(root: Path) -> StudyData:
    """The v1 study: the original 150-row snapshots."""
    return load_study(root / T1_FILE, root / T2_FILE, V1_SPEC)


def load_bridge_study(root: Path) -> StudyData:
    """The bridge study: v1 design (v1 features, 2022-2026) on the 2022-2026 rows of the expanded files."""
    return load_study(root / V2_DIR / T1_FILE, root / V2_DIR / T2_FILE, V1_SPEC)


def load_study_v2(root: Path) -> StudyData:
    """The ten-edition study (T1 only): v2 features on 2016-2026. M2 is not available here (T2 covers 2022-2026 only)."""
    return load_study_v2_variant(root, V2_SPEC, MAIN_OPTIONS, V2_FEATURES)


def load_study_v2_variant(root: Path, spec: StudySpec, options: FeatureOptions, features: FeatureSet) -> StudyData:
    """The T1-only study restricted to `spec.t1_editions`, with swappable source columns (robustness runs)."""
    return load_study_with_t1_columns(root, spec, options, features, lambda t1: t1)


def load_study_v2_1(root: Path, spec: StudySpec, options: FeatureOptions, features: FeatureSet) -> StudyData:
    """Like the variant loader, with the raw v2.1 award score attached to the T1 rows (options.awards must be set)."""
    if options.awards is None:
        raise ValueError("load_study_v2_1 needs options.awards")
    events = pd.read_parquet(root / V2_1_DIR / EVENTS_FILE)
    return load_study_with_t1_columns(root, spec, options, features, lambda t1: attach_award_columns(t1, events, options.awards))


def load_study_with_t1_columns(root: Path, spec: StudySpec, options: FeatureOptions, features: FeatureSet,
                               prepare: Callable[[pd.DataFrame], pd.DataFrame]) -> StudyData:
    path_t1 = root / V2_DIR / T1_FILE
    t1 = pd.read_csv(path_t1)
    t1 = t1[t1[EDITION_COL].isin(spec.t1_editions)].reset_index(drop=True)
    check_t1(t1, spec)
    check_edition_facts(t1)
    identity = t1[list(IDENTITY_COLS)]
    base = build_base_features_with(prepare(t1), options)
    frame_t1 = pd.concat([identity, base.drop(columns=[EDITION_COL])], axis=1)
    return StudyData(frame_t1=frame_t1, frame_t2=None, targets=t1[list(TARGET_COLS)].copy(), sha256_t1=sha256_of(path_t1),
                     sha256_t2="", labelled_editions=spec.labelled_editions, forecast_edition=spec.forecast_edition,
                     features=features)


def load_study_t2_v2(root: Path) -> StudyData:
    """The 2022-2026 study with v2 features and the v2 post-window block (P1 from non-penalty goals + assists), for M2."""
    path_t1, path_t2 = root / V2_DIR / T1_FILE, root / V2_DIR / T2_FILE
    t1 = pd.read_csv(path_t1)
    t1 = t1[t1[EDITION_COL].isin(V1_SPEC.t1_editions)].reset_index(drop=True)
    t2 = pd.read_csv(path_t2)
    check_snapshots(t1, t2, V1_SPEC)
    check_edition_facts(t1)
    identity = t1[list(IDENTITY_COLS)]
    base = build_base_features_with(t1, MAIN_OPTIONS)
    post = build_post_features_v2(t2)
    frame_t1 = pd.concat([identity, base.drop(columns=[EDITION_COL])], axis=1)
    frame_t2 = pd.concat([frame_t1, post], axis=1)
    return StudyData(frame_t1=frame_t1, frame_t2=frame_t2, targets=t1[list(TARGET_COLS)].copy(), sha256_t1=sha256_of(path_t1),
                     sha256_t2=sha256_of(path_t2), labelled_editions=V1_SPEC.labelled_editions,
                     forecast_edition=V1_SPEC.forecast_edition, features=V2_POST_FEATURES)
