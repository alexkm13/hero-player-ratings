#!/usr/bin/env python3
"""
Optimize card overall category weights (F/D) against process-based impact targets.

The hero overall on cards is a weighted blend of nine 5v5 category z-scores
(see build_styled_all_in_one_cards.py). This script searches weight vectors on
the simplex and scores them with walk-forward season-level correlation.

Targets (pick via --target):
  - eh_xgpm: EH regular-season RAPM xG±/60 (default; clearest external impact)
  - impact: EH + PTV same-season composite (matches tune_all_in_one_8cat impact objective)
  - blend: 0.65 * eh_xgpm_z + 0.35 * impact_z (within season/position)

Output JSON is compatible with build_styled_all_in_one_cards.py --overall-weights-json.
After adopting new weights, set CARD_OVERALL_WEIGHT_MULT to 1.0 for all labels
(or re-run this tuner and apply role-specific priors in _draw_weights only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from all_in_one_category_specs import CAT_COLS
from forward_slot_utils import assign_forward_slot_column
from tune_all_in_one_8cat_model import (
    SAME_SEASON_IMPACT_FEATURE_COLS,
    _build_same_season_impact_anchor,
    _corr,
    _to_num,
)

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Reliability weighting: downweight noisy target seasons
# ---------------------------------------------------------------------------
RELIABILITY_COLS = (
    "ptv_roll3_n_seasons_in_window",
    "ptv_roll3_rapm_net_possession_impact",
    "eh_rapm_Team",
    "a3z_team",
)

# PTV defensive process metrics for alternative target
PTV_DEFENSIVE_COLS = (
    "ptv_season_rapm_threat_suppression",
    "ptv_season_rapm_pre_shot_pressure",
    "ptv_season_rapm_possession_defense",
    "ptv_coalesced_rapm_threat_suppression",
    "ptv_coalesced_rapm_possession_defense",
)

# Experience-based segment shrinkage weights
EXPERIENCE_COLS = ("ptv_roll3_n_seasons_in_window",)


def _compute_reliability_weights(
    df: pd.DataFrame,
    *,
    toi_floor: float = 0.3,
    team_change_weight: float = 0.5,
    rookie_weight: float = 0.6,
    spike_weight: float = 0.4,
    spike_threshold_sd: float = 1.5,
    experience_boost: bool = False,
) -> pd.Series:
    """
    Compute per-row reliability weights for training.

    Downweights:
    - Low next-season TOI relative to current (role loss, injury)
    - Team changes (context shift)
    - Rookies/2nd-year players (high volatility)
    - One-season spikes above prior 3Y (unsustainable)

    If experience_boost=True, also upweights veterans with 3+ seasons.

    Returns weights in [0, 1] where 1 = fully reliable.
    """
    w = pd.Series(1.0, index=df.index)

    # Work with a sorted copy that preserves original index for alignment
    work = df[["player_id", "season_id", "_toi_min"]].copy()
    work["_team"] = df.get("eh_rapm_Team", pd.Series("", index=df.index)).fillna(
        df.get("a3z_team", pd.Series("", index=df.index))
    )
    work = work.sort_values(["player_id", "season_id"])

    # Compute next-season values within each player
    work["_next_toi"] = work.groupby("player_id")["_toi_min"].shift(-1)
    work["_next_team"] = work.groupby("player_id")["_team"].shift(-1)

    # Align back to original index
    work = work.reindex(df.index)

    # 1. TOI continuity: next_toi / current_toi
    curr_toi = _to_num(df["_toi_min"])
    next_toi = work["_next_toi"]
    toi_ratio = (next_toi / curr_toi.replace(0, np.nan)).clip(0, 2)
    # If next TOI < 50% of current, weight = toi_floor
    toi_weight = np.where(
        toi_ratio < 0.5,
        float(toi_floor),
        np.where(toi_ratio < 1.0, 0.5 + 0.5 * toi_ratio, 1.0)
    )
    # If no next season data, keep weight=1 (eval seasons)
    toi_weight = np.where(next_toi.isna(), 1.0, toi_weight)
    w = w * toi_weight

    # 2. Team change detection
    team_changed = (work["_team"].astype(str) != work["_next_team"].astype(str)) & work["_next_team"].notna()
    w = np.where(team_changed, w * float(team_change_weight), w)

    # 3. Rookie/2nd-year detection (fewer than 2 prior seasons)
    n_prior = _to_num(df.get("ptv_roll3_n_seasons_in_window", pd.Series(3, index=df.index)))
    is_early_career = n_prior < 2
    w = np.where(is_early_career, w * float(rookie_weight), w)

    # 4. One-season spike detection
    #    Current impact z-score >> prior 3Y impact suggests unsustainable
    curr_impact = _to_num(df.get("impact_z", pd.Series(0, index=df.index)))
    prior_impact = _to_num(df.get("ptv_roll3_rapm_net_possession_impact", pd.Series(0, index=df.index)))
    # Z-score the prior impact within season/position for comparison
    prior_z = df.copy()
    prior_z["_prior"] = prior_impact
    prior_impact_z = _z_by_group(prior_z, "_prior", ["season_id", "position_group"])
    spike = (curr_impact - prior_impact_z).abs() > float(spike_threshold_sd)
    # Only penalize upward spikes (career years), not drops
    upward_spike = spike & (curr_impact > prior_impact_z)
    w = np.where(upward_spike, w * float(spike_weight), w)

    # 5. Experience boost: upweight veterans with established patterns
    if experience_boost:
        # Players with 3 seasons get full weight, 2 seasons get 0.85, 1 season gets 0.7
        exp_factor = np.where(
            n_prior >= 3, 1.0,
            np.where(n_prior >= 2, 0.85, 0.7)
        )
        w = w * exp_factor

    # Ensure weights are in [0.1, 1.0]
    w = pd.Series(w, index=df.index).clip(0.1, 1.0)
    return w


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 3:
        return float("nan")

    x, y, w = x[mask], y[mask], w[mask]
    w = w / w.sum()  # Normalize weights

    mx = np.sum(w * x)
    my = np.sum(w * y)

    dx = x - mx
    dy = y - my

    cov_xy = np.sum(w * dx * dy)
    var_x = np.sum(w * dx * dx)
    var_y = np.sum(w * dy * dy)

    if var_x <= 0 or var_y <= 0:
        return float("nan")

    return float(cov_xy / np.sqrt(var_x * var_y))


# Nine categories shown on player cards (5v5 only; ST excluded from overall).
CARD_CAT_COLS = [
    "cat_shot_creation_z",
    "cat_transition_offense_z",
    "cat_shot_defense_z",
    "cat_transition_defense_z",
    "cat_finishing_z",
    "cat_playmaking_z",
    "cat_forecheck_z",
    "cat_physicality_z",
    "cat_skating_z",
]

ROLE_PRIORS: dict[str, np.ndarray] = {
    "F": np.array(
        [
            0.27,
            0.10,
            0.19,
            0.09,
            0.09,
            0.07,
            0.08,
            0.03,
            0.08,
        ],
        dtype=float,
    ),
    "D": np.array(
        [
            0.21,
            0.02,
            0.31,
            0.16,
            0.15,
            0.06,
            0.01,
            0.06,
            0.03,
        ],
        dtype=float,
    ),
    # Centers: offense-first priors (skill + finishing); defense moderated in search.
    "C": np.array(
        [
            0.28,  # shot_creation
            0.12,  # transition_offense
            0.10,  # shot_defense
            0.08,  # transition_defense
            0.12,  # finishing
            0.12,  # playmaking
            0.08,  # forecheck
            0.02,  # physicality
            0.08,  # skating
        ],
        dtype=float,
    ),
    # Wings: offense-first priors (skill + transition); keep defense light in search.
    "W": np.array(
        [
            0.30,  # shot_creation
            0.14,  # transition_offense
            0.06,  # shot_defense
            0.03,  # transition_defense
            0.08,  # finishing
            0.18,  # playmaking
            0.10,  # forecheck
            0.03,  # physicality
            0.08,  # skating
        ],
        dtype=float,
    ),
}

# Center slot: relaxed constraints to allow defense weight where data supports it.
# Data shows shot_defense r=0.647 for centers - strongest predictor.
CENTER_SLOT_MAX_SHOT_DEFENSE = 0.30  # was 0.14
CENTER_SLOT_MAX_DEFENSE_SHARE = 0.40  # was 0.18
CENTER_SLOT_MAX_FORECHECK = 0.15  # was 0.10
CENTER_SLOT_MIN_OFFENSE_SHARE = 0.40  # was 0.58

# Wing slot: relaxed constraints to allow defense weight where data supports it.
# Data shows shot_defense r=0.633 for wings - strongest predictor.
WING_SLOT_MAX_SHOT_DEFENSE = 0.25  # was 0.08
WING_SLOT_MAX_DEFENSE_SHARE = 0.35  # was 0.11
WING_SLOT_MAX_PHYSICALITY = 0.12  # was 0.07
WING_SLOT_MIN_OFFENSE_SHARE = 0.45  # SC + TO + PM + FIN (was 0.62)
_IDX_SC, _IDX_TO, _IDX_SD, _IDX_TD, _IDX_FIN, _IDX_PM, _IDX_FC, _IDX_PHY = 0, 1, 2, 3, 4, 5, 6, 7


def _center_weight_allowed(w: np.ndarray) -> bool:
    w = np.asarray(w, dtype=float)
    if w[_IDX_SD] > float(CENTER_SLOT_MAX_SHOT_DEFENSE) + 1e-9:
        return False
    if (w[_IDX_SD] + w[_IDX_TD]) > float(CENTER_SLOT_MAX_DEFENSE_SHARE) + 1e-9:
        return False
    if w[_IDX_FC] > float(CENTER_SLOT_MAX_FORECHECK) + 1e-9:
        return False
    if (w[_IDX_SC] + w[_IDX_TO] + w[_IDX_PM] + w[_IDX_FIN]) < float(CENTER_SLOT_MIN_OFFENSE_SHARE) - 1e-9:
        return False
    return True


def _wing_weight_allowed(w: np.ndarray) -> bool:
    w = np.asarray(w, dtype=float)
    if w[_IDX_SD] > float(WING_SLOT_MAX_SHOT_DEFENSE) + 1e-9:
        return False
    if (w[_IDX_SD] + w[_IDX_TD]) > float(WING_SLOT_MAX_DEFENSE_SHARE) + 1e-9:
        return False
    if w[_IDX_PHY] > float(WING_SLOT_MAX_PHYSICALITY) + 1e-9:
        return False
    if (w[_IDX_SC] + w[_IDX_TO] + w[_IDX_PM] + w[_IDX_FIN]) < float(WING_SLOT_MIN_OFFENSE_SHARE) - 1e-9:
        return False
    return True


def _z_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    x = _to_num(df[col])
    mu = x.groupby([df[c] for c in group_cols], dropna=False).transform("mean")
    sd = x.groupby([df[c] for c in group_cols], dropna=False).transform("std")
    return (x - mu) / sd.replace(0.0, np.nan)


def _load_panel(
    scores_path: Path,
    feature_path: Path,
    *,
    min_toi_min: float,
) -> pd.DataFrame:
    d = pd.read_csv(scores_path, low_memory=False)
    need_adj = [f"{c}__adj" for c in CARD_CAT_COLS]
    miss = [c for c in need_adj if c not in d.columns]
    if miss:
        raise SystemExit(f"Missing adjusted category columns in scores: {miss[:5]}")

    fs_cols = list(
        dict.fromkeys(
            [
                "player_id",
                "season_id",
                "eh_rapm_TOI",
                "eh_rapm_xG±/60",
                "eh_rapm_Position",
                "eh_std_FOW/60",
                *SAME_SEASON_IMPACT_FEATURE_COLS,
                *RELIABILITY_COLS,
                *PTV_DEFENSIVE_COLS,
            ]
        )
    )
    fs = pd.read_csv(feature_path, low_memory=False)
    fs = fs.loc[:, ~fs.columns.duplicated()].copy()
    fs_cols = [c for c in fs_cols if c in fs.columns]
    fs = fs[fs_cols].drop_duplicates(["player_id", "season_id"], keep="last")
    overlap = [c for c in fs.columns if c not in {"player_id", "season_id"} and c in d.columns]
    if overlap:
        d = d.drop(columns=overlap)
    d = d.merge(fs, on=["player_id", "season_id"], how="left")

    toi = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi = toi.where(toi.notna(), _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index))))
    d["_toi_min"] = toi
    d = d[d["_toi_min"] >= float(min_toi_min)].copy()

    if "eh_rapm_xG±/60" not in d.columns:
        raise SystemExit("Feature table missing eh_rapm_xG±/60")
    tmp = d.copy()
    tmp["_eh_xgpm"] = pd.to_numeric(d["eh_rapm_xG±/60"], errors="coerce")
    d["eh_xgpm_z"] = _z_by_group(tmp, "_eh_xgpm", ["season_id", "position_group"])
    d["impact_z"] = _build_same_season_impact_anchor(
        d,
        xgf_w=0.20,
        xga_w=0.25,
        xgs_w=0.45,
        gs_w=0.10,
    )
    d["forward_slot"] = assign_forward_slot_column(d)
    return d


def _build_target(d: pd.DataFrame, target: str) -> pd.Series:
    t = str(target).lower().strip()
    if t == "eh_xgpm":
        return _to_num(d["eh_xgpm_z"])
    if t == "impact":
        return _to_num(d["impact_z"])
    if t == "blend":
        eh = _to_num(d["eh_xgpm_z"])
        imp = _to_num(d["impact_z"])
        m = eh.notna() & imp.notna()
        out = eh.copy()
        out.loc[m] = 0.65 * eh.loc[m] + 0.35 * imp.loc[m]
        return out
    if t == "defense_blend":
        # For D: blend impact with PTV defensive process metrics
        # For F: use standard impact
        imp = _to_num(d["impact_z"])
        out = imp.copy()

        # Get PTV defensive metrics for D
        is_d = d["position_group"].astype(str).str.upper().eq("D")
        if is_d.any():
            # Z-score PTV metrics within season
            threat_supp = _to_num(d.get("ptv_coalesced_rapm_threat_suppression", pd.Series(np.nan, index=d.index)))
            poss_def = _to_num(d.get("ptv_coalesced_rapm_possession_defense", pd.Series(np.nan, index=d.index)))

            # Z-score within season/position
            def z_group(s: pd.Series) -> pd.Series:
                return _z_by_group(d.assign(_x=s), "_x", ["season_id", "position_group"])

            threat_z = z_group(threat_supp)
            poss_z = z_group(poss_def)

            # Blend: 50% impact + 25% threat suppression + 25% possession defense
            d_mask = is_d & threat_z.notna() & poss_z.notna() & imp.notna()
            out.loc[d_mask] = (
                0.50 * imp.loc[d_mask]
                + 0.25 * (-threat_z.loc[d_mask])  # negative because lower threat = better defense
                + 0.25 * (-poss_z.loc[d_mask])  # negative because lower = better defense
            )
        return out
    raise SystemExit(f"Unknown --target {target!r}")


def _score_weights(
    d: pd.DataFrame,
    w: np.ndarray,
    *,
    target: pd.Series,
    eval_seasons: set[int] | None,
    reliability_weights: pd.Series | None = None,
    prior: np.ndarray | None = None,
    regularization: float = 0.0,
    huber_delta: float | None = None,
) -> float:
    """
    Score a weight vector against target.

    Args:
        prior: If provided with regularization > 0, penalize deviation from prior
        regularization: L2 penalty strength for deviation from prior (0 = no penalty)
        huber_delta: If provided, use Huber-style robust correlation (winsorize residuals)
    """
    mat = d[[f"{c}__adj" for c in CARD_CAT_COLS]].to_numpy(dtype=float)
    pred = np.nansum(mat * w[None, :], axis=1)
    dd = d.assign(_pred=pred, _y=target)
    if reliability_weights is not None:
        dd = dd.assign(_rel_w=reliability_weights)
    if eval_seasons is not None:
        dd = dd[dd["season_id"].isin(eval_seasons)].copy()

    # Optional: Huber-style outlier handling - winsorize extreme residuals
    if huber_delta is not None and huber_delta > 0:
        y_arr = dd["_y"].to_numpy(dtype=float)
        pred_arr = dd["_pred"].to_numpy(dtype=float)
        valid = np.isfinite(y_arr) & np.isfinite(pred_arr)
        if valid.sum() > 10:
            resid = y_arr[valid] - pred_arr[valid]
            resid_std = np.std(resid)
            if resid_std > 0:
                clip_val = float(huber_delta) * resid_std
                y_clipped = np.clip(y_arr, pred_arr - clip_val, pred_arr + clip_val)
                dd = dd.assign(_y=y_clipped)

    season_scores: list[float] = []
    for _, g in dd.groupby("season_id", dropna=False):
        if reliability_weights is not None and "_rel_w" in g.columns:
            r = _weighted_corr(
                g["_pred"].to_numpy(float),
                g["_y"].to_numpy(float),
                g["_rel_w"].to_numpy(float),
            )
        else:
            r = _corr(g["_pred"].to_numpy(float), g["_y"].to_numpy(float))
        if np.isfinite(r):
            season_scores.append(float(r))
    if not season_scores:
        return float("nan")

    base_score = float(np.mean(season_scores))

    # L2 regularization penalty toward prior
    if prior is not None and regularization > 0:
        prior_norm = prior / float(np.sum(prior)) if np.sum(prior) > 0 else prior
        w_norm = w / float(np.sum(w)) if np.sum(w) > 0 else w
        deviation = np.sum((w_norm - prior_norm) ** 2)
        penalty = float(regularization) * deviation
        return base_score - penalty

    return base_score


def _sample_weights(
    rng: np.random.Generator,
    role: str,
    *,
    n_draws: int,
    dirichlet_alpha: float,
) -> np.ndarray:
    n = len(CARD_CAT_COLS)
    alpha = float(max(dirichlet_alpha, 0.05))
    draws = rng.dirichlet(alpha=np.full(n, alpha), size=int(n_draws))
    prior = ROLE_PRIORS.get(role)
    if prior is not None and len(prior) == n:
        p = prior / float(prior.sum())
        extra = 8 if str(role).upper() == "W" else (6 if str(role).upper() == "C" else 1)
        draws = np.vstack([draws] + [p] * int(extra))
    return draws


def _tune_forward_slot(
    d: pd.DataFrame,
    slot: str,
    *,
    target: pd.Series,
    train_seasons: set[int],
    eval_seasons: set[int],
    n_draws: int,
    seed: int,
    dirichlet_alpha: float,
    reliability_weights: pd.Series | None = None,
    regularization: float = 0.0,
    huber_delta: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    sub = d[
        d["position_group"].astype(str).str.upper().eq("F")
        & d["forward_slot"].astype(str).str.upper().eq(str(slot).upper())
    ].copy()
    sub_rel_w = reliability_weights.loc[sub.index] if reliability_weights is not None else None
    rng = np.random.default_rng(int(seed) + (17 if slot == "C" else 23))
    cands = _sample_weights(rng, slot, n_draws=n_draws, dirichlet_alpha=dirichlet_alpha)

    slot_u = str(slot).upper()
    prior = ROLE_PRIORS.get(slot_u)
    if prior is not None and len(prior) == len(CARD_CAT_COLS):
        prior_norm = prior / float(prior.sum())
        best_w = prior_norm.copy()
    else:
        prior_norm = None
        best_w = cands[0] / float(np.sum(cands[0]))
    best_train = -np.inf
    for w in cands:
        w = w / float(np.sum(w))
        if slot_u == "C" and not _center_weight_allowed(w):
            continue
        if slot_u == "W" and not _wing_weight_allowed(w):
            continue
        obj = _score_weights(
            sub, w, target=target, eval_seasons=train_seasons,
            reliability_weights=sub_rel_w,
            prior=prior_norm, regularization=regularization, huber_delta=huber_delta,
        )
        if np.isfinite(obj) and obj > best_train:
            best_train = float(obj)
            best_w = w.copy()
    if not np.isfinite(best_train) and prior_norm is not None:
        best_w = prior_norm.copy()
        best_train = _score_weights(
            sub, best_w, target=target, eval_seasons=train_seasons,
            reliability_weights=sub_rel_w,
            prior=prior_norm, regularization=regularization, huber_delta=huber_delta,
        )

    full = {c: 0.0 for c in CAT_COLS}
    for i, c in enumerate(CARD_CAT_COLS):
        full[c] = float(best_w[i])
    s = float(sum(full[c] for c in CARD_CAT_COLS))
    if s > 0:
        for c in CARD_CAT_COLS:
            full[c] /= s

    # Eval metrics computed without regularization for fair comparison
    metrics = {
        "n_rows": int(len(sub)),
        "train_r": float(best_train),
        "eval_r": _score_weights(sub, best_w, target=target, eval_seasons=eval_seasons),
        "eval_r_eh": _score_weights(
            sub,
            best_w,
            target=_to_num(sub["eh_xgpm_z"]),
            eval_seasons=eval_seasons,
        ),
        "eval_r_impact": _score_weights(
            sub,
            best_w,
            target=_to_num(sub["impact_z"]),
            eval_seasons=eval_seasons,
        ),
    }
    return full, metrics


def _tune_role(
    d: pd.DataFrame,
    role: str,
    *,
    target: pd.Series,
    train_seasons: set[int],
    eval_seasons: set[int],
    n_draws: int,
    seed: int,
    dirichlet_alpha: float,
    reliability_weights: pd.Series | None = None,
    regularization: float = 0.0,
    huber_delta: float | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    sub = d[d["position_group"].astype(str).str.upper().eq(role)].copy()
    sub_rel_w = reliability_weights.loc[sub.index] if reliability_weights is not None else None
    rng = np.random.default_rng(int(seed) + (11 if role == "D" else 0))
    cands = _sample_weights(rng, role, n_draws=n_draws, dirichlet_alpha=dirichlet_alpha)

    # Get prior for regularization
    prior = ROLE_PRIORS.get(role)
    if prior is not None:
        prior = prior / float(prior.sum())

    best_w = cands[0].copy()
    best_train = -np.inf
    for w in cands:
        w = w / float(np.sum(w))
        obj = _score_weights(
            sub, w, target=target, eval_seasons=train_seasons,
            reliability_weights=sub_rel_w,
            prior=prior, regularization=regularization, huber_delta=huber_delta,
        )
        if np.isfinite(obj) and obj > best_train:
            best_train = float(obj)
            best_w = w.copy()

    full = {c: 0.0 for c in CAT_COLS}
    for i, c in enumerate(CARD_CAT_COLS):
        full[c] = float(best_w[i])
    s = float(sum(full[c] for c in CARD_CAT_COLS))
    if s > 0:
        for c in CARD_CAT_COLS:
            full[c] /= s

    # Eval metrics computed without regularization for fair comparison
    metrics = {
        "train_r": float(best_train),
        "eval_r": _score_weights(sub, best_w, target=target, eval_seasons=eval_seasons),
        "eval_r_eh": _score_weights(
            sub,
            best_w,
            target=_to_num(sub["eh_xgpm_z"]),
            eval_seasons=eval_seasons,
        ),
        "eval_r_impact": _score_weights(
            sub,
            best_w,
            target=_to_num(sub["impact_z"]),
            eval_seasons=eval_seasons,
        ),
        "pooled_eval_r": _score_weights(sub, best_w, target=target, eval_seasons=eval_seasons),
    }
    return full, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores_tuned.csv")
    ap.add_argument("--features", type=Path, default=ROOT / "data/pcv/player_season_feature_table_all_in_one.csv")
    ap.add_argument("--out-json", type=Path, default=ROOT / "data/pcv/all_in_one_card_overall_weights.json")
    ap.add_argument("--baseline-json", type=Path, default=ROOT / "data/pcv/all_in_one_8cat_impact_weights.json")
    ap.add_argument("--target", choices=["eh_xgpm", "impact", "blend", "defense_blend"], default="eh_xgpm")
    ap.add_argument("--n-draws", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dirichlet-alpha", type=float, default=1.4, help="Higher = search closer to uniform.")
    ap.add_argument("--min-toi-min", type=float, default=200.0)
    ap.add_argument(
        "--train-seasons",
        type=str,
        default="20162017,20172018,20182019,20192020,20202021,20212022,20222023",
    )
    ap.add_argument("--eval-seasons", type=str, default="20232024,20242025")
    ap.add_argument(
        "--tune-forward-slots",
        action="store_true",
        help="Also tune weights_C and weights_W (EH position slot) against the target.",
    )
    ap.add_argument(
        "--merge-impact-json",
        type=Path,
        default=None,
        help="Write merged weights into this impact JSON (keeps meta; updates weights_C/W).",
    )
    ap.add_argument(
        "--wing-slot-target",
        choices=["eh_xgpm", "impact", "blend"],
        default="blend",
        help="Objective for weights_W slot tuning (default: blend = 0.65*xG + 0.35*impact).",
    )
    ap.add_argument(
        "--center-slot-target",
        choices=["eh_xgpm", "impact", "blend"],
        default="blend",
        help="Objective for weights_C slot tuning (default: blend).",
    )
    ap.add_argument(
        "--reliability-weights",
        action="store_true",
        help="Enable reliability weighting to downweight noisy targets (team changes, rookies, spikes).",
    )
    ap.add_argument(
        "--exclude-unreliable",
        action="store_true",
        help="Exclude unreliable rows from training instead of downweighting.",
    )
    ap.add_argument(
        "--reliability-threshold",
        type=float,
        default=0.6,
        help="Exclude rows with reliability weight below this threshold (default: 0.6).",
    )
    ap.add_argument(
        "--toi-floor",
        type=float,
        default=0.3,
        help="Weight floor for low next-season TOI (default: 0.3).",
    )
    ap.add_argument(
        "--team-change-weight",
        type=float,
        default=0.5,
        help="Weight multiplier for team changes (default: 0.5).",
    )
    ap.add_argument(
        "--rookie-weight",
        type=float,
        default=0.6,
        help="Weight multiplier for rookies/2nd-year players (default: 0.6).",
    )
    ap.add_argument(
        "--spike-weight",
        type=float,
        default=0.4,
        help="Weight multiplier for one-season spikes (default: 0.4).",
    )
    ap.add_argument(
        "--regularization",
        type=float,
        default=0.0,
        help="L2 regularization strength toward role priors (0 = none, try 0.1-0.5 for D).",
    )
    ap.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        help="Huber loss delta for outlier-robust training (in std devs, e.g. 2.0).",
    )
    ap.add_argument(
        "--experience-boost",
        action="store_true",
        help="Upweight veterans with 3+ seasons for segment-specific shrinkage.",
    )
    args = ap.parse_args()

    train_seasons = {int(x.strip()) for x in str(args.train_seasons).split(",") if x.strip()}
    eval_seasons = {int(x.strip()) for x in str(args.eval_seasons).split(",") if x.strip()}

    d = _load_panel(args.scores, args.features, min_toi_min=float(args.min_toi_min))
    d["season_id"] = pd.to_numeric(d["season_id"], errors="coerce").astype(int)
    y = _build_target(d, str(args.target))

    # Compute reliability weights if enabled
    rel_weights: pd.Series | None = None
    if bool(args.reliability_weights) or bool(args.exclude_unreliable) or bool(args.experience_boost):
        rel_weights = _compute_reliability_weights(
            d,
            toi_floor=float(args.toi_floor),
            team_change_weight=float(args.team_change_weight),
            rookie_weight=float(args.rookie_weight),
            spike_weight=float(args.spike_weight),
            experience_boost=bool(args.experience_boost),
        )
        avg_w = float(rel_weights.mean())
        low_w = float((rel_weights < 0.5).sum())
        print(f"Reliability weights: mean={avg_w:.3f}, n_downweighted={low_w} ({100*low_w/len(rel_weights):.1f}%)", flush=True)

    # Optionally exclude unreliable rows from training
    d_train = d
    y_train = y
    if bool(args.exclude_unreliable) and rel_weights is not None:
        threshold = float(args.reliability_threshold)
        reliable_mask = rel_weights >= threshold
        # Only exclude from training seasons, keep all for eval
        train_mask = d["season_id"].isin(train_seasons)
        exclude_mask = train_mask & ~reliable_mask
        d_train = d[~exclude_mask].copy()
        y_train = y.loc[d_train.index]
        rel_weights = rel_weights.loc[d_train.index] if rel_weights is not None else None
        n_excluded = int(exclude_mask.sum())
        print(f"Excluded {n_excluded} unreliable training rows (threshold={threshold})", flush=True)
        print(f"Training data: {len(d_train)} rows (was {len(d)})", flush=True)
        # Don't use reliability weights if we're excluding - they've served their purpose
        rel_weights = None

    w_f, m_f = _tune_role(
        d_train,
        "F",
        target=y_train,
        train_seasons=train_seasons,
        eval_seasons=eval_seasons,
        n_draws=int(args.n_draws),
        seed=int(args.seed),
        dirichlet_alpha=float(args.dirichlet_alpha),
        reliability_weights=rel_weights,
        regularization=float(args.regularization),
        huber_delta=args.huber_delta,
    )
    w_d, m_d = _tune_role(
        d_train,
        "D",
        target=y_train,
        train_seasons=train_seasons,
        eval_seasons=eval_seasons,
        n_draws=int(args.n_draws),
        seed=int(args.seed) + 99,
        dirichlet_alpha=float(args.dirichlet_alpha),
        reliability_weights=rel_weights,
        regularization=float(args.regularization),
        huber_delta=args.huber_delta,
    )

    baseline_metrics: dict[str, object] = {}
    if args.baseline_json.is_file():
        base = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        bw_f = np.array([float(base.get("weights_F", {}).get(c, 0.0)) for c in CARD_CAT_COLS], dtype=float)
        bw_d = np.array([float(base.get("weights_D", {}).get(c, 0.0)) for c in CARD_CAT_COLS], dtype=float)
        if float(bw_f.sum()) > 0:
            bw_f /= float(bw_f.sum())
        if float(bw_d.sum()) > 0:
            bw_d /= float(bw_d.sum())
        sub_f = d[d["position_group"].astype(str).str.upper().eq("F")]
        sub_d = d[d["position_group"].astype(str).str.upper().eq("D")]
        baseline_metrics = {
            "F": {
                "eval_r": _score_weights(sub_f, bw_f, target=y, eval_seasons=eval_seasons),
                "eval_r_eh": _score_weights(sub_f, bw_f, target=_to_num(sub_f["eh_xgpm_z"]), eval_seasons=eval_seasons),
            },
            "D": {
                "eval_r": _score_weights(sub_d, bw_d, target=y, eval_seasons=eval_seasons),
                "eval_r_eh": _score_weights(sub_d, bw_d, target=_to_num(sub_d["eh_xgpm_z"]), eval_seasons=eval_seasons),
            },
        }

    slot_metrics: dict[str, dict[str, float]] = {}
    w_c: dict[str, float] | None = None
    w_w: dict[str, float] | None = None
    if bool(args.tune_forward_slots):
        y_center = _build_target(d_train, str(args.center_slot_target))
        y_wing = _build_target(d_train, str(args.wing_slot_target))
        w_c, m_c = _tune_forward_slot(
            d_train,
            "C",
            target=y_center,
            train_seasons=train_seasons,
            eval_seasons=eval_seasons,
            n_draws=int(args.n_draws),
            seed=int(args.seed) + 31,
            dirichlet_alpha=float(args.dirichlet_alpha),
            reliability_weights=rel_weights,
            regularization=float(args.regularization),
            huber_delta=args.huber_delta,
        )
        w_w, m_w = _tune_forward_slot(
            d_train,
            "W",
            target=y_wing,
            train_seasons=train_seasons,
            eval_seasons=eval_seasons,
            n_draws=int(args.n_draws),
            seed=int(args.seed) + 47,
            dirichlet_alpha=float(args.dirichlet_alpha),
            reliability_weights=rel_weights,
            regularization=float(args.regularization),
            huber_delta=args.huber_delta,
        )
        slot_metrics = {
            "C": {**m_c, "target": str(args.center_slot_target)},
            "W": {**m_w, "target": str(args.wing_slot_target)},
        }
        print(
            f"C eval_r={m_c.get('eval_r')} eh={m_c.get('eval_r_eh')} n={m_c.get('n_rows')}  "
            f"W eval_r={m_w.get('eval_r')} eh={m_w.get('eval_r_eh')} n={m_w.get('n_rows')}",
            flush=True,
        )
        print("Weights C (tuned):")
        for c in CARD_CAT_COLS:
            print(f"  {c}: {w_c[c]:.4f}", flush=True)
        print("Weights W (tuned):")
        for c in CARD_CAT_COLS:
            print(f"  {c}: {w_w[c]:.4f}", flush=True)

    payload = {
        "meta": {
            "target": str(args.target),
            "n_draws": int(args.n_draws),
            "train_seasons": sorted(train_seasons),
            "eval_seasons": sorted(eval_seasons),
            "card_categories": CARD_CAT_COLS,
            "tune_forward_slots": bool(args.tune_forward_slots),
            "note": "Use with build_styled --overall-weights-json; prefer CARD_OVERALL_WEIGHT_MULT=1.0 when applying.",
        },
        "metrics": {"F": m_f, "D": m_d, "baseline": baseline_metrics, "slots": slot_metrics},
        "weights_F": {c: float(w_f[c]) for c in CAT_COLS},
        "weights_D": {c: float(w_d[c]) for c in CAT_COLS},
    }
    if w_c is not None and w_w is not None:
        payload["weights_C"] = {c: float(w_c[c]) for c in CAT_COLS}
        payload["weights_W"] = {c: float(w_w[c]) for c in CAT_COLS}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}", flush=True)
    print(f"F eval_r={m_f.get('eval_r')}  D eval_r={m_d.get('eval_r')}", flush=True)
    if baseline_metrics:
        print(f"Baseline F eval_r={baseline_metrics['F']['eval_r']}  D={baseline_metrics['D']['eval_r']}", flush=True)

    if args.merge_impact_json is not None and w_c is not None and w_w is not None:
        merge_path = Path(args.merge_impact_json)
        base = json.loads(merge_path.read_text(encoding="utf-8")) if merge_path.is_file() else {}
        meta = base.get("meta", {}) if isinstance(base.get("meta"), dict) else {}
        slot_note = meta.get("objective", {}) if isinstance(meta.get("objective"), dict) else {}
        if not isinstance(slot_note, dict):
            slot_note = {}
        slot_note["forward_slot_tune"] = {
            "target": str(args.target),
            "wing_slot_target": str(args.wing_slot_target),
            "center_slot_target": str(args.center_slot_target),
            "eval_seasons": sorted(eval_seasons),
            "metrics": slot_metrics,
        }
        meta["objective"] = slot_note
        base["meta"] = meta
        base["weights_C"] = payload["weights_C"]
        base["weights_W"] = payload["weights_W"]
        merge_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"Merged weights_C/W into {merge_path}", flush=True)


if __name__ == "__main__":
    main()
