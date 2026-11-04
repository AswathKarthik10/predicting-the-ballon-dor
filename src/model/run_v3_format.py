"""Experiment 3, stage 2: coefficients by format and pooled format interactions (spec section 3). Writes results/v3/."""

import time
from pathlib import Path

import pandas as pd

from src.model import format_effects as fe
from src.model.data import V2_SPEC, load_study_v2_1
from src.model.features_v2 import V2_FEATURES
from src.model.run_v2_1 import v2_1_options

OUT_DIR = Path("results") / "v3"


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    study = load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES)
    lam = fe.median_e3a_lambda(pd.read_csv(out / "tuning_by_fold_v3.csv"))
    print(f"fixed lambda (median over E3-A folds) = {lam}", flush=True)
    fe.coefficients_by_format(study, lam).to_csv(out / "coefficients_by_format.csv", index=False)
    print(f"coefficients by format done in {time.time() - started:.0f}s", flush=True)
    fe.interaction_tables(study).to_csv(out / "format_interactions.csv", index=False)
    print(f"interactions done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
