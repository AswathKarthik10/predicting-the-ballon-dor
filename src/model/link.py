"""Link between published-model strengths u and ballot-level juror utilities alpha (spec section 4).

alpha_(i,e) = c_e + k * u_(i,e) + r_(i,e), one common slope k and a free intercept per edition. The residual SD is the
race-level uncertainty shared by all jurors in an edition.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import forecast
from src.model.data import StudyData

MODEL_BY_SNAPSHOT: dict[str, str] = {"T1": "M1", "T2": "M2"}


@dataclass(frozen=True)
class LinkFit:
    snapshot: str
    editions: tuple[int, ...]
    k: float
    sigma_shared: float
    r2_within: float
    n_players: int


def alpha_by_player(fit: bl.BiasFit, data: list[bl.EditionBallots]) -> pd.DataFrame:
    rows = [pd.DataFrame({"edition": d.edition, "player_key": d.player_keys, "alpha": fit.alpha[d.edition]}) for d in data]
    return pd.concat(rows, ignore_index=True)


def oof_strengths(study: StudyData, snapshot: str, editions: tuple[int, ...]) -> pd.DataFrame:
    """Out-of-fold model strengths (the model never saw that edition) per player-edition."""
    model = MODEL_BY_SNAPSHOT[snapshot]
    frames = []
    for edition in editions:
        keys = study.frame_t1.loc[study.frame_t1.edition == edition, "player_key"].to_numpy()
        frames.append(pd.DataFrame({"edition": edition, "player_key": keys, "u": forecast.oof_utilities_for(study, model, edition)}))
    return pd.concat(frames, ignore_index=True)


def fit_link(alphas: pd.DataFrame, strengths: pd.DataFrame, snapshot: str, editions: tuple[int, ...]) -> LinkFit:
    joined = alphas.merge(strengths, on=["edition", "player_key"], how="inner", validate="1:1")
    joined = joined[joined.edition.isin(editions)]
    if len(joined) != len(alphas[alphas.edition.isin(editions)]):
        raise ValueError("alpha rows without a model strength for some player-editions")
    centred_u = joined.u - joined.groupby("edition").u.transform("mean")
    centred_a = joined.alpha - joined.groupby("edition").alpha.transform("mean")
    k = float((centred_u * centred_a).sum() / (centred_u ** 2).sum())
    residual = centred_a - k * centred_u
    dof = len(joined) - len(editions) - 1
    return LinkFit(snapshot=snapshot, editions=editions, k=k, sigma_shared=float(np.sqrt((residual ** 2).sum() / dof)),
                   r2_within=float(1.0 - (residual ** 2).sum() / (centred_a ** 2).sum()), n_players=len(joined))


def link_table(fits: list[LinkFit]) -> pd.DataFrame:
    return pd.DataFrame([{"snapshot": f.snapshot, "model": MODEL_BY_SNAPSHOT[f.snapshot], "editions": "-".join(map(str, f.editions)),
                          "k": f.k, "sigma_shared": f.sigma_shared, "r2_within_edition": f.r2_within, "n_player_editions": f.n_players}
                         for f in fits])
