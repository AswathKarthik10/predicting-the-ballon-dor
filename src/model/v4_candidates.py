"""Experiment 4 candidate features and mechanical encoding (spec sections 2.1-2.3).

Every column of the T1 snapshot is a candidate except those excluded by the rules in section 2.2. No composites, ratios
or ordinal maps are built. Imputation is not done here: missing values stay NaN and get a `<col>_missing` indicator;
the training-fold median is applied inside the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

T1_PATH = Path("data") / "v2" / "player_snapshot_t1.csv"
EDITIONS = (2022, 2023, 2024, 2025, 2026)
LABELLED_EDITIONS = (2022, 2023, 2024, 2025)
EDITION_COL = "edition"
POINTS_COL = "bdo_points"

MIN_CATEGORY_ROWS = 5
OTHER_LEVEL = "other"
MISSING_SUFFIX = "_missing"

IDENTIFIER_COLS = ("edition", "snapshot", "player_key", "player_name", "club_key", "club_team_id", "league_key")
TARGET_COLS = ("bdo_points", "bdo_rank", "vote_share")
DATE_COLS = ("snapshot_date", "tm_value_date", "ea_update_date", "built_at", "pipeline_version")
PROVENANCE_COLS = ("honors_source", "tourn_individual_award")
EDITION_CONSTANT_COLS = ("eligibility_format", "ballot_picks", "jurors_voted")
CATEGORICAL_COLS = ("position_group", "nationality", "tourn_name", "ucl_stage_reached", "nt_tourn_stage_reached")

REASON_IDENTIFIER = "identifier or label"
REASON_TARGET = "target"
REASON_DATE = "date or build metadata"
REASON_PROVENANCE = "provenance text"
REASON_EDITION_CONSTANT = "constant within every edition"
REASON_ZERO_VARIANCE = "zero variance across all rows"
REASON_KEPT = "kept"

KIND_NUMERIC = "numeric"
KIND_BOOLEAN = "boolean"
KIND_ONE_HOT = "one_hot"
KIND_MISSING = "missing_indicator"


@dataclass(frozen=True)
class Candidates:
    features: pd.DataFrame  # encoded candidate columns for all rows (NaN kept), index aligned to `rows`
    rows: pd.DataFrame  # edition, player_key, bdo_points per row
    table: pd.DataFrame  # one row per raw column and per encoded column (written to candidates.csv)
    dropped_categoricals: tuple[str, ...]


def load_raw_t1(root: Path) -> pd.DataFrame:
    t1 = pd.read_csv(root / T1_PATH)
    return t1[t1[EDITION_COL].isin(EDITIONS)].reset_index(drop=True)


def listed_exclusions() -> dict[str, str]:
    reasons: dict[str, str] = {}
    for cols, reason in ((IDENTIFIER_COLS, REASON_IDENTIFIER), (TARGET_COLS, REASON_TARGET), (DATE_COLS, REASON_DATE),
                         (PROVENANCE_COLS, REASON_PROVENANCE), (EDITION_CONSTANT_COLS, REASON_EDITION_CONSTANT)):
        reasons.update({col: reason for col in cols})
    return reasons


def constant_within_every_edition(column: pd.Series, edition: pd.Series) -> bool:
    return bool((column.groupby(edition).nunique(dropna=False) <= 1).all())


def exclusion_reasons(t1: pd.DataFrame) -> dict[str, str]:
    """Reason per excluded column: the listed ones plus columns found constant within every edition or across all rows."""
    reasons = listed_exclusions()
    for col in t1.columns:
        if col in reasons:
            continue
        if t1[col].nunique(dropna=False) <= 1:
            reasons[col] = REASON_ZERO_VARIANCE
        elif constant_within_every_edition(t1[col], t1[EDITION_COL]):
            reasons[col] = REASON_EDITION_CONSTANT
    return reasons


def is_boolean_like(column: pd.Series) -> bool:
    if column.dtype == bool:
        return True
    non_null = column.dropna()
    return column.dtype == object and len(non_null) > 0 and non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all()


def merged_levels(column: pd.Series, labelled: pd.Series) -> pd.Series:
    """Levels with fewer than MIN_CATEGORY_ROWS labelled rows are merged into OTHER_LEVEL; missing stays missing."""
    counts = column[labelled].value_counts()
    keep = set(counts[counts >= MIN_CATEGORY_ROWS].index)
    return column.where(column.isna() | column.isin(keep), OTHER_LEVEL)


def one_hot_columns(column: pd.Series, labelled: pd.Series) -> pd.DataFrame:
    """One-hot of the merged levels. Empty frame when no level has enough labelled rows (the column is dropped)."""
    counts = column[labelled].value_counts()
    if not (counts >= MIN_CATEGORY_ROWS).any():
        return pd.DataFrame(index=column.index)
    merged = merged_levels(column, labelled)
    dummies = pd.get_dummies(merged, prefix=str(column.name), dtype=float)
    dummies[merged.isna()] = 0.0
    return dummies


def encode_column(column: pd.Series, labelled: pd.Series) -> tuple[pd.DataFrame, str]:
    if column.name in CATEGORICAL_COLS:
        return one_hot_columns(column, labelled), KIND_ONE_HOT
    if is_boolean_like(column):
        return column.astype(float).to_frame(), KIND_BOOLEAN
    if not pd.api.types.is_numeric_dtype(column):
        raise ValueError(f"column {column.name!r} has dtype {column.dtype} and is not a listed categorical")
    return column.astype(float).to_frame(), KIND_NUMERIC


def missing_indicator(column: pd.Series) -> pd.Series:
    return column.isna().astype(float).rename(f"{column.name}{MISSING_SUFFIX}")


def drop_constant_labelled(features: pd.DataFrame, labelled: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """One-hot or indicator columns that are constant on the labelled rows carry no signal; drop and report them."""
    constant = [col for col in features.columns if features.loc[labelled, col].nunique(dropna=False) <= 1]
    return features.drop(columns=constant), constant


def table_row(name: str, source: str, kind: str, status: str, reason: str, missing_share: float, n_unique: int) -> dict[str, object]:
    return {"column": name, "source_column": source, "kind": kind, "status": status, "reason": reason,
            "missing_share": round(missing_share, 4), "n_unique": n_unique}


def build_candidates(t1: pd.DataFrame) -> Candidates:
    reasons = exclusion_reasons(t1)
    labelled = t1[EDITION_COL].isin(LABELLED_EDITIONS)
    blocks: list[pd.DataFrame] = []
    origin: dict[str, tuple[str, str]] = {}  # encoded name -> (source column, kind)
    table: list[dict[str, object]] = []
    dropped_categoricals: list[str] = []
    for col in t1.columns:
        missing_share = float(t1[col].isna().mean())
        n_unique = int(t1[col].nunique(dropna=True))
        if col in reasons:
            table.append(table_row(col, col, "raw", "excluded", reasons[col], missing_share, n_unique))
            continue
        encoded, kind = encode_column(t1[col], labelled)
        if encoded.shape[1] == 0:
            dropped_categoricals.append(col)
            table.append(table_row(col, col, kind, "excluded", f"no category with {MIN_CATEGORY_ROWS}+ labelled rows", missing_share, n_unique))
            continue
        blocks.append(encoded)
        origin.update({name: (col, kind) for name in encoded.columns})
        if missing_share > 0:
            indicator = missing_indicator(t1[col])
            blocks.append(indicator.to_frame())
            origin[str(indicator.name)] = (col, KIND_MISSING)
    features, constant = drop_constant_labelled(pd.concat(blocks, axis=1), labelled)
    for name in features.columns:
        source, kind = origin[name]
        table.append(table_row(name, source, kind, "candidate", REASON_KEPT, float(features[name].isna().mean()), int(features[name].nunique())))
    for name in constant:
        table.append(table_row(name, origin[name][0], origin[name][1], "excluded", "constant on labelled rows", 0.0, 1))
    rows = t1[[EDITION_COL, "player_key", POINTS_COL]].copy()
    return Candidates(features=features, rows=rows, table=pd.DataFrame(table), dropped_categoricals=tuple(dropped_categoricals))


def write_candidates(candidates: Candidates, path: Path) -> None:
    candidates.table.to_csv(path, index=False)
