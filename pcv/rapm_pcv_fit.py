"""
Ridge RAPM for PCV/60 stint targets: separate **offensive** and **defensive** models, two rows
per 5v5 stint (focal = home, then away), *sample_weight* = *duration_sec*,
*GroupKFold* on *NHL_game_id* for *alpha*.

*Offense*: y = the focal side's *PCV/60*; X: five +1s for the focal *offensive* skaters; last column
*focal is game home* (1 / 0).

*Defense*: y = **minus** *opponent* *PCV/60*; X: five +1s for the focal *defending* skaters; same
*focal* column. The negative target makes **higher (positive) Ridge coefficients = less opponent
PCV/60* (i.e. better defense), matching an intuitive "defense is good" sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold

# CV grid (GroupKFold): 1k–50k for a wide low end; 100k/250k so defense (or offense) need not sit on 50k.
DEFAULT_ALPHAS: tuple = (
    1_000,
    2_500,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    250_000,
)
DEFAULT_CV: int = 5


def _skater_ids_as_list(x: Any) -> list:
    """Stint rows may store id lists as list/tuple or ndarray (e.g. from Parquet)."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)


@dataclass(frozen=True)
class RapmFitResult:
    name: str
    alpha: float
    model: Ridge
    player_id_order: list[int]
    n_rows: int
    alpha_hits_grid_max: bool = False  # True if CV picked max(alphas) — consider wider grid or more games


def stint_frame_all_player_ids(st: pd.DataFrame) -> set[int]:
    s: set[int] = set()
    for a, b in zip(st["home_skater_ids"], st["away_skater_ids"]):
        for p in a:
            s.add(int(p))
        for p in b:
            s.add(int(p))
    return s


def player_stint_5v5_toi_sec(stints: pd.DataFrame) -> dict[int, float]:
    """
    Total 5v5 *duration_sec* a skater was on the ice across all stint rows (in regression sample only).
    Matches Natural Stat Trick–style 5v5 TOI in this pipeline’s definition (non-EN stints, ≥*min* sec).
    """
    from collections import defaultdict

    toi: dict[int, float] = defaultdict(float)
    for row in stints.itertuples():
        d = float(getattr(row, "duration_sec", 0.0) or 0.0)
        if d <= 0.0:
            continue
        for p in _skater_ids_as_list(getattr(row, "home_skater_ids", None)):
            toi[int(p)] += d
        for p in _skater_ids_as_list(getattr(row, "away_skater_ids", None)):
            toi[int(p)] += d
    return dict(toi)


def build_rapm_mats(
    stints: pd.DataFrame,
    *,
    all_player_ids: list[int],
    sparse_mats: bool = True,
) -> dict[str, Any]:
    """
    Two rows per stint, 5 nonzeros in player block + 1 in *focal is home* column = 6.
    *all_player_ids* = sorted list for stable columns.
    """
    def _row_float(row: Any, *cols: str, default: float = 0.0) -> float:
        for col in cols:
            if hasattr(row, col):
                try:
                    v = float(getattr(row, col))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    return v
        return float(default)

    p2c = {p: i for i, p in enumerate(all_player_ids)}
    n_p = len(all_player_ids)
    p_focal_home = n_p
    n_s = len(stints)
    n_r = 2 * n_s
    p_cols = n_p + 1
    w = np.zeros(n_r, dtype=np.float64)
    g = np.zeros(n_r, dtype=np.int64)
    y_of = np.zeros(n_r, dtype=np.float64)
    y_df = np.zeros(n_r, dtype=np.float64)
    ridx = 0
    for row in stints.itertuples():
        gid = int(getattr(row, "NHL_game_id"))
        d = float(getattr(row, "duration_sec"))
        h60 = _row_float(row, "home_pcv_per_60_context_adj", "home_pcv_per_60")
        a60 = _row_float(row, "away_pcv_per_60_context_adj", "away_pcv_per_60")
        h_def_direct = _row_float(row, "home_direct_defense_pcv_per60")
        a_def_direct = _row_float(row, "away_direct_defense_pcv_per60")
        w[ridx] = d
        g[ridx] = gid
        y_of[ridx] = h60
        y_df[ridx] = -a60 + h_def_direct  # suppress opponent PTV + direct defensive events
        ridx += 1
        w[ridx] = d
        g[ridx] = gid
        y_of[ridx] = a60
        y_df[ridx] = -h60 + a_def_direct
        ridx += 1
    ridx = 0
    rows_o: list = []
    cols_o: list = []
    d_o: list = []
    rows_d: list = []
    cols_d: list = []
    d_d: list = []
    for row in stints.itertuples():
        hs = [p2c[int(p)] for p in getattr(row, "home_skater_ids")]
        ajs = [p2c[int(p)] for p in getattr(row, "away_skater_ids")]
        # offense row0: home skaters
        for c in hs:
            rows_o.append(ridx)
            cols_o.append(c)
            d_o.append(1.0)
        rows_o.append(ridx)
        cols_o.append(p_focal_home)
        d_o.append(1.0)
        ridx += 1
        for c in ajs:
            rows_o.append(ridx)
            cols_o.append(c)
            d_o.append(1.0)
        rows_o.append(ridx)
        cols_o.append(p_focal_home)
        d_o.append(0.0)
        ridx += 1
    ridx = 0
    for row in stints.itertuples():
        hs = [p2c[int(p)] for p in getattr(row, "home_skater_ids")]
        ajs = [p2c[int(p)] for p in getattr(row, "away_skater_ids")]
        for c in hs:
            rows_d.append(ridx)
            cols_d.append(c)
            d_d.append(1.0)
        rows_d.append(ridx)
        cols_d.append(p_focal_home)
        d_d.append(1.0)
        ridx += 1
        for c in ajs:
            rows_d.append(ridx)
            cols_d.append(c)
            d_d.append(1.0)
        rows_d.append(ridx)
        cols_d.append(p_focal_home)
        d_d.append(0.0)
        ridx += 1
    if not sparse_mats:
        Xo = np.zeros((n_r, p_cols), dtype=np.float64)
        ro = np.asarray(rows_o, int)
        co = np.asarray(cols_o, int)
        dvo = np.asarray(d_o, float)
        Xo[ro, co] = dvo
        Xd = np.zeros((n_r, p_cols), dtype=np.float64)
        rd = np.asarray(rows_d, int)
        cd = np.asarray(cols_d, int)
        dvd = np.asarray(d_d, float)
        Xd[rd, cd] = dvd
    else:
        Xo = sparse.coo_matrix(
            (np.asarray(d_o, float), (np.asarray(rows_o, int), np.asarray(cols_o, int))),
            shape=(n_r, p_cols),
        ).tocsc()
        Xd = sparse.coo_matrix(
            (np.asarray(d_d, float), (np.asarray(rows_d, int), np.asarray(cols_d, int))),
            shape=(n_r, p_cols),
        ).tocsc()
    return {
        "y_off": y_of,
        "X_off": Xo,
        "y_def": y_df,
        "X_def": Xd,
        "w": w,
        "g": g,
        "n_players": n_p,
    }


def choose_alpha_grouped_ridge(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    g: np.ndarray,
    alphas: tuple = DEFAULT_ALPHAS,
    n_splits: int = DEFAULT_CV,
) -> tuple[float, bool]:
    """
    Returns (best_alpha, hits_grid_max). *hits_grid_max* is True when the chosen α equals
    max(*alphas*) — common on small samples; extend the grid or do not trust coefficients for
    leaderboards until CV sits in the interior on a full-season run.
    """
    a_list = tuple(float(a) for a in alphas)
    a_max = max(a_list)
    n_u = int(len(np.unique(g)))
    n_s = len(y)
    default_alpha = float(a_list[len(a_list) // 2])
    if n_s < 4 or n_u < 2:
        return default_alpha, False
    n_sp = min(n_splits, n_u)
    if n_sp < 2:
        return default_alpha, False
    cv = GroupKFold(n_splits=n_sp)
    reg = GridSearchCV(
        Ridge(fit_intercept=True, max_iter=10000),
        param_grid={"alpha": list(a_list)},
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=None,
    )
    reg.fit(x, y, sample_weight=w, groups=g)
    best = float(reg.best_params_["alpha"])
    at_max = abs(best - a_max) <= 1e-6 * max(a_max, 1.0)
    return best, at_max


def fit_rapm_pair(
    stints: pd.DataFrame,
    all_player_ids: list[int] | None = None,
    *,
    alphas: tuple = DEFAULT_ALPHAS,
) -> tuple[RapmFitResult, RapmFitResult, pd.DataFrame, pd.DataFrame]:
    a_grid = tuple(float(x) for x in alphas)
    if stints.empty:
        empty = pd.DataFrame(columns=["player_id", "coef"])
        return (
            RapmFitResult("off", float("nan"), Ridge(), [], 0, False),
            RapmFitResult("def", float("nan"), Ridge(), [], 0, False),
            empty,
            empty,
        )
    pids: list[int]
    if all_player_ids is not None:
        pids = list(all_player_ids)
    else:
        pids = sorted(int(x) for x in stint_frame_all_player_ids(stints))
    m = build_rapm_mats(stints, all_player_ids=pids, sparse_mats=True)
    y0, yd = m["y_off"], m["y_def"]
    xo, xd = m["X_off"], m["X_def"]
    w, g = m["w"], m["g"]
    n_p = m["n_players"]

    ao, o_max = choose_alpha_grouped_ridge(y0, xo, w, g, alphas=a_grid)
    ad, d_max = choose_alpha_grouped_ridge(yd, xd, w, g, alphas=a_grid)
    mo = Ridge(alpha=ao, fit_intercept=True, max_iter=10000)
    md = Ridge(alpha=ad, fit_intercept=True, max_iter=10000)
    mo.fit(xo, y0, sample_weight=w)
    md.fit(xd, yd, sample_weight=w)
    c_o = np.asarray(mo.coef_).ravel()[:n_p]
    c_d = np.asarray(md.coef_).ravel()[:n_p]
    t_o = RapmFitResult("off", ao, mo, pids, 2 * len(stints), o_max)
    t_d = RapmFitResult("def", ad, md, pids, 2 * len(stints), d_max)
    df_o = pd.DataFrame({"player_id": pids, "offensive_rapm_pcv": c_o})
    df_d = pd.DataFrame({"player_id": pids, "defensive_rapm_pcv": c_d})
    return t_o, t_d, df_o, df_d


def _ridge_coef_player_block(mo: Ridge, n_p: int) -> np.ndarray:
    return np.asarray(mo.coef_).ravel()[:n_p]


def fit_rapm_pair_fixed_alphas(
    stints: pd.DataFrame,
    all_player_ids: list[int],
    alpha_off: float,
    alpha_def: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge on *stints* with fixed (α_off, α_def) — for bootstrap refits, no CV."""
    if stints.empty or not all_player_ids:
        return np.array([]), np.array([])
    m = build_rapm_mats(stints, all_player_ids=all_player_ids, sparse_mats=True)
    y0, yd = m["y_off"], m["y_def"]
    xo, xd = m["X_off"], m["X_def"]
    w = m["w"]
    n_p = m["n_players"]
    mo = Ridge(alpha=float(alpha_off), fit_intercept=True, max_iter=10000)
    md = Ridge(alpha=float(alpha_def), fit_intercept=True, max_iter=10000)
    mo.fit(xo, y0, sample_weight=w)
    md.fit(xd, yd, sample_weight=w)
    return _ridge_coef_player_block(mo, n_p), _ridge_coef_player_block(md, n_p)


def bootstrap_rapm_pcv_ses(
    stints: pd.DataFrame,
    t_o: RapmFitResult,
    t_d: RapmFitResult,
    n_bootstrap: int,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stint resampling with replacement; refit with **fixed** (α_off, α_def) from the main *fit_rapm_pair*.
    Returns standard deviations of per-player coeff across *n_bootstrap* replicates.
    """
    pids = t_o.player_id_order
    n_p = len(pids)
    if n_bootstrap < 1 or stints.empty or n_p < 1:
        return np.full(n_p, np.nan), np.full(n_p, np.nan)
    rng = rng or np.random.default_rng()
    n_s = int(len(stints))
    if n_s < 2:
        return np.full(n_p, np.nan), np.full(n_p, np.nan)
    acc_o: list[np.ndarray] = []
    acc_d: list[np.ndarray] = []
    idx0 = np.arange(n_s, dtype=int)
    ao, ad = t_o.alpha, t_d.alpha
    for _ in range(int(n_bootstrap)):
        samp = rng.choice(idx0, size=n_s, replace=True)
        st_b = stints.iloc[samp].reset_index(drop=True)
        co, cd = fit_rapm_pair_fixed_alphas(st_b, pids, ao, ad)
        if co.size == n_p:
            acc_o.append(co)
        if cd.size == n_p:
            acc_d.append(cd)
    if not acc_o or not acc_d:
        return np.full(n_p, np.nan), np.full(n_p, np.nan)
    a_o = np.vstack(acc_o)
    a_d = np.vstack(acc_d)
    if a_o.shape[0] < 2:
        se_o = np.full(n_p, np.nan)
    else:
        se_o = a_o.std(axis=0, ddof=1)
    if a_d.shape[0] < 2:
        se_d = np.full(n_p, np.nan)
    else:
        se_d = a_d.std(axis=0, ddof=1)
    return se_o, se_d


def bootstrap_rapm_pcv_ses_for_player_ids(
    stints: pd.DataFrame,
    t_o: RapmFitResult,
    t_d: RapmFitResult,
    n_bootstrap: int,
    player_ids: list[int] | set[int] | np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Same resampling as :func:`bootstrap_rapm_pcv_ses`, but return SE **only** for *player_ids*
    (same compute cost as full bootstrap; use to publish SEs for a top-k list only).
    """
    want = {int(p) for p in player_ids}
    pids = t_o.player_id_order
    idx = [i for i, p in enumerate(pids) if int(p) in want]
    if not idx or n_bootstrap < 1 or stints.empty:
        return {}, {}
    se_o, se_d = bootstrap_rapm_pcv_ses(stints, t_o, t_d, n_bootstrap, rng=rng)
    omap = {int(pids[i]): float(se_o[i]) for i in idx if not np.isnan(se_o[i])}
    dmap = {int(pids[i]): float(se_d[i]) for i in idx if not np.isnan(se_d[i])}
    return omap, dmap


__all__ = [
    "DEFAULT_ALPHAS",
    "RapmFitResult",
    "stint_frame_all_player_ids",
    "player_stint_5v5_toi_sec",
    "build_rapm_mats",
    "fit_rapm_pair",
    "fit_rapm_pair_fixed_alphas",
    "bootstrap_rapm_pcv_ses",
    "bootstrap_rapm_pcv_ses_for_player_ids",
    "choose_alpha_grouped_ridge",
]
