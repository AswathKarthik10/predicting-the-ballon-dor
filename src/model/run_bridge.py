"""Bridge run (experiments spec section 14, step 2): the v1 design on the 2022-2026 rows of the expanded files.

Two variants isolate the effect of the three columns the dataset revised after v1:
`as_revised` uses the expanded file as delivered; `v1_columns_restored` puts the v1 values of those columns back and
must reproduce results/metrics_summary.csv exactly.
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.model import cv
from src.model.data import T1_FILE, T2_FILE, V1_SPEC, V2_DIR, StudyData, load_study
from src.model.features import EDITION_COL

REVISED_COLUMNS: tuple[str, ...] = ("career_bdo_podiums_before", "club_trophies_n", "won_domestic_cup")
KEY_COLUMNS: tuple[str, ...] = (EDITION_COL, "player_key")
BRIDGE_MODELS: list[str] = ["U0", "B0", "B1", "B2", "X1", "M1", "M1-G", "M2"]
V1_SUMMARY = Path("results") / "metrics_summary.csv"
OUT_DIR = Path("results") / "v2"
VARIANT_AS_REVISED = "as_revised"
VARIANT_RESTORED = "v1_columns_restored"


def revised_column_changes(root: Path) -> pd.DataFrame:
    """One row per (edition, player) whose revised columns differ from v1."""
    v1 = pd.read_csv(root / T1_FILE)
    v2 = pd.read_csv(root / V2_DIR / T1_FILE)
    v2 = v2[v2[EDITION_COL].isin(V1_SPEC.t1_editions)].reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for column in REVISED_COLUMNS:
        differs = v1[column].astype(float) != v2[column].astype(float)
        changed = v2.loc[differs, [EDITION_COL, "player_name"]].copy()
        changed["column"] = column
        changed["v1_value"] = v1.loc[differs, column].to_numpy()
        changed["expanded_value"] = v2.loc[differs, column].to_numpy()
        rows.append(changed)
    return pd.concat(rows, ignore_index=True)


def restored_study(root: Path, scratch: Path) -> StudyData:
    """The bridge study with the revised columns taken back from the v1 snapshots."""
    v1 = pd.read_csv(root / T1_FILE)
    v2 = pd.read_csv(root / V2_DIR / T1_FILE)
    v2 = v2[v2[EDITION_COL].isin(V1_SPEC.t1_editions)].reset_index(drop=True)
    if not (v1[list(KEY_COLUMNS)].to_numpy() == v2[list(KEY_COLUMNS)].to_numpy()).all():
        raise ValueError("v1 and expanded snapshots are not row-aligned")
    v2[list(REVISED_COLUMNS)] = v1[list(REVISED_COLUMNS)]
    path_t1 = scratch / T1_FILE
    v2.to_csv(path_t1, index=False)
    t2 = pd.read_csv(root / V2_DIR / T2_FILE)
    v1_t2 = pd.read_csv(root / T2_FILE)
    t2[list(REVISED_COLUMNS)] = v1_t2[list(REVISED_COLUMNS)]
    path_t2 = scratch / T2_FILE
    t2.to_csv(path_t2, index=False)
    return load_study(path_t1, path_t2, V1_SPEC)


def run_variant(study: StudyData, variant: str) -> cv.CvResults:
    results = cv.run_cross_validation(study, BRIDGE_MODELS)
    results.metrics.insert(0, "variant", variant)
    return results


def compare_with_v1(root: Path, summary: pd.DataFrame) -> pd.DataFrame:
    v1 = pd.read_csv(root / V1_SUMMARY, comment="#")
    keep = ["model", "metric", "mean"]
    merged = summary[keep].merge(v1[keep], on=["model", "metric"], suffixes=("_bridge", "_v1"))
    merged["difference"] = merged["mean_bridge"] - merged["mean_v1"]
    return merged


def main(root: Path) -> None:
    OUT = root / OUT_DIR
    OUT.mkdir(parents=True, exist_ok=True)
    revised = load_study(root / V2_DIR / T1_FILE, root / V2_DIR / T2_FILE, V1_SPEC)
    revised_results = run_variant(revised, VARIANT_AS_REVISED)
    with tempfile.TemporaryDirectory() as scratch:
        restored_results = run_variant(restored_study(root, Path(scratch)), VARIANT_RESTORED)
    metrics = pd.concat([revised_results.metrics, restored_results.metrics], ignore_index=True)
    metrics.to_csv(OUT / "bridge_metrics_by_fold.csv", index=False)
    comparisons: list[pd.DataFrame] = []
    for variant, part in metrics.groupby("variant"):
        summary = cv.summarise_metrics(part.drop(columns="variant"))
        comparison = compare_with_v1(root, summary)
        comparison.insert(0, "variant", variant)
        comparisons.append(comparison)
    pd.concat(comparisons, ignore_index=True).to_csv(OUT / "bridge_vs_v1.csv", index=False)
    revised_column_changes(root).to_csv(OUT / "bridge_revised_columns.csv", index=False)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
