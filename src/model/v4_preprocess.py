"""Experiment 4 preprocessing options (spec section 3).

Everything learned from data (medians, skew flags, global moments) is fitted on the training rows only. The within-edition
scalings use only the rows of the edition being transformed, so they never see another edition.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import rankdata, skew

SKEW_THRESHOLD = 1.0
ZERO_SPREAD_FLOOR = 1e-12


class Scaling(str, Enum):
    WITHIN_Z = "within_edition_z"
    GLOBAL_Z = "global_z"
    WITHIN_PCT = "within_edition_pct"


@dataclass(frozen=True)
class PreprocessOption:
    scaling: Scaling
    log_skewed: bool

    @property
    def name(self) -> str:
        return f"{self.scaling.value}|{'log1p' if self.log_skewed else 'raw'}"


PREPROCESS_OPTIONS: tuple[PreprocessOption, ...] = tuple(
    PreprocessOption(scaling, log_skewed) for scaling in Scaling for log_skewed in (False, True))


@dataclass(frozen=True)
class Preprocessor:
    option: PreprocessOption
    medians: np.ndarray
    log_columns: np.ndarray  # boolean mask of columns that get log1p
    mean: np.ndarray  # global moments after imputation and log (used by GLOBAL_Z only)
    std: np.ndarray


def impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), medians, x)


def log_mask_of(imputed_train: np.ndarray) -> np.ndarray:
    no_negatives = (imputed_train >= 0).all(axis=0)
    spread = imputed_train.std(axis=0) > ZERO_SPREAD_FLOOR
    skewed = np.zeros(imputed_train.shape[1], dtype=bool)
    skewed[spread] = skew(imputed_train[:, spread], axis=0) > SKEW_THRESHOLD
    return skewed & no_negatives


def apply_log(x: np.ndarray, log_columns: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[:, log_columns] = np.log1p(np.clip(x[:, log_columns], 0.0, None))
    return out


def fit_preprocessor(x_train: np.ndarray, option: PreprocessOption) -> Preprocessor:
    medians = np.nanmedian(x_train, axis=0)
    imputed = impute(x_train, medians)
    log_columns = log_mask_of(imputed) if option.log_skewed else np.zeros(x_train.shape[1], dtype=bool)
    logged = apply_log(imputed, log_columns)
    return Preprocessor(option=option, medians=medians, log_columns=log_columns, mean=logged.mean(axis=0), std=logged.std(axis=0))


def z_score(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe = np.where(std > ZERO_SPREAD_FLOOR, std, 1.0)
    return np.where(std > ZERO_SPREAD_FLOOR, (x - mean) / safe, 0.0)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Column-wise rank scaled to [0, 1] with ties averaged."""
    n = x.shape[0]
    if n < 2:
        return np.zeros_like(x)
    ranks = np.apply_along_axis(rankdata, 0, x)
    return (ranks - 1.0) / (n - 1)


def scale_within_edition(x: np.ndarray, edition: np.ndarray, scaling: Scaling) -> np.ndarray:
    out = np.zeros_like(x)
    for ed in np.unique(edition):
        rows = edition == ed
        block = x[rows]
        if scaling is Scaling.WITHIN_Z:
            out[rows] = z_score(block, block.mean(axis=0), block.std(axis=0))
            continue
        ranked = percentile_rank(block)
        constant = block.std(axis=0) <= ZERO_SPREAD_FLOOR
        ranked[:, constant] = 0.0
        out[rows] = ranked
    return out


def transform(prep: Preprocessor, x: np.ndarray, edition: np.ndarray) -> np.ndarray:
    logged = apply_log(impute(x, prep.medians), prep.log_columns)
    if prep.option.scaling is Scaling.GLOBAL_Z:
        return z_score(logged, prep.mean, prep.std)
    return scale_within_edition(logged, edition, prep.option.scaling)
