"""Loading, integrity checks and the study data container.

This is the only module besides the later modelling code that touches the outcome columns. They are
kept in a separate frame so the feature code never sees them.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.model.features import EDITION_COL, WINDOW_END, build_base_features, build_post_features

T1_FILE = "player_snapshot_t1.csv"
T2_FILE = "player_snapshot_t2.csv"
KEY_COLS: tuple[str, ...] = ("edition", "player_key")
TARGET_COLS: tuple[str, ...] = ("bdo_points", "bdo_rank", "vote_share")
SNAPSHOT_COLS: tuple[str, ...] = ("snapshot", "snapshot_date")
IDENTITY_COLS: tuple[str, ...] = ("edition", "player_key", "player_name", "nationality", "position_group")
FORECAST_EDITION = 2026
TRAIN_EDITIONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
EXPECTED_ROWS = 150


@dataclass(frozen=True)
class StudyData:
    frame_t1: pd.DataFrame  # identity + base features, no outcome columns
    frame_t2: pd.DataFrame  # frame_t1 plus the post-window features
    targets: pd.DataFrame  # outcome columns, same index as the frames
    sha256_t1: str
    sha256_t2: str


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_snapshots(t1: pd.DataFrame, t2: pd.DataFrame) -> None:
    if len(t1) != EXPECTED_ROWS or len(t2) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows per snapshot, got {len(t1)} and {len(t2)}")
    for name, df in (("T1", t1), ("T2", t2)):
        if df.duplicated(list(KEY_COLS)).any():
            raise ValueError(f"{name}: duplicate (edition, player_key)")
    if not (t1[list(KEY_COLS)].to_numpy() == t2[list(KEY_COLS)].to_numpy()).all():
        raise ValueError("T1 and T2 rows are not in the same order")
    shared = [c for c in t1.columns if c not in SNAPSHOT_COLS]
    if not t1[shared].equals(t2[shared]):
        raise ValueError("in-window columns differ between T1 and T2")
    forecast = t1[t1[EDITION_COL] == FORECAST_EDITION]
    if not forecast[list(TARGET_COLS)].isna().all().all():
        raise ValueError("2026 target columns must be entirely null")
    labelled = t1[t1[EDITION_COL] != FORECAST_EDITION]
    if labelled[list(TARGET_COLS)].isna().any().any():
        raise ValueError("labelled editions must have no null targets")
    if set(t1[EDITION_COL].unique()) != set(WINDOW_END):
        raise ValueError("unexpected set of editions")


def load_study_data(root: Path) -> StudyData:
    path_t1, path_t2 = root / T1_FILE, root / T2_FILE
    t1 = pd.read_csv(path_t1)
    t2 = pd.read_csv(path_t2)
    check_snapshots(t1, t2)
    identity = t1[list(IDENTITY_COLS)]
    base = build_base_features(t1)
    post = build_post_features(t2)
    frame_t1 = pd.concat([identity, base.drop(columns=[EDITION_COL])], axis=1)
    frame_t2 = pd.concat([frame_t1, post], axis=1)
    targets = t1[list(TARGET_COLS)].copy()
    return StudyData(frame_t1=frame_t1, frame_t2=frame_t2, targets=targets,
                     sha256_t1=sha256_of(path_t1), sha256_t2=sha256_of(path_t2))
