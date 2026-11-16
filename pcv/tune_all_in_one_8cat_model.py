#!/usr/bin/env python3
"""
Objective tuning for 11-category all-in-one model (includes finishing: G vs ixG, SH% vs xSH%).

Features:
- Role-specific category weights (F and D)
- Reliability shrinkage by TOI
- Noisy-category caps (special teams, penalty value)
- Objective tuning against outcome anchors:
  xGF/60, xGA/60 (flipped), xG share, goal share

Input is the existing category score table (cat_*_z columns).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from all_in_one_category_specs import CAT_COLS, NOISY_CAT_COLS as NOISY_COLS
from hero_evidence_exposure import EXPOSURE_SOURCE_COLUMNS, exposure_proxies, preserve_tracking

ROOT = Path(__file__).resolve().parent


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, low_memory=False)
FIXED_BENCHMARK_TARGET_WEIGHTS = {
    "xGF_per60": 0.35,
    "xGA_per60_flipped": 0.35,
    "onice_xg_share": 0.20,
    "onice_goal_share": 0.10,
}

# Reliability floors for weak/noisy categories to avoid overfitting low-sample spikes.
MIN_K_FLOOR_BY_ROLE_CATEGORY: dict[str, dict[str, float]] = {
    "F": {
        # weak / noisy components: force higher reliability thresholds
        "cat_shot_defense_z": 150.0,
        "cat_transition_defense_z": 1800.0,
        "cat_finishing_z": 2200.0,
        "cat_power_play_z": 2200.0,
        "cat_penalty_kill_z": 2200.0,
        "cat_physicality_z": 1300.0,
        "cat_skating_z": 1600.0,
        "cat_forecheck_z": 900.0,
    },
    "D": {
        "cat_transition_defense_z": 1800.0,
        "cat_finishing_z": 2200.0,
        "cat_power_play_z": 2200.0,
        "cat_penalty_kill_z": 2200.0,
        "cat_physicality_z": 1100.0,
        "cat_skating_z": 1200.0,
        "cat_forecheck_z": 900.0,
        "cat_shot_defense_z": 1000.0,
    },
}


@dataclass(frozen=True)
class TuneConfig:
    min_5v5_share: float
    cap_noisy: float
    k_general: float
    k_noisy: float
    n_samples: int
    seed: int


def _parse_k_candidates(s: str) -> list[float]:
    vals: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        if np.isfinite(v) and v > 0.0:
            vals.append(v)
    out = sorted(set(vals))
    if not out:
        out = [150.0, 250.0, 400.0, 550.0, 700.0, 900.0, 1200.0]
    return out


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dedupe_player_season(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    need = {"player_id", "season_id"}
    if not need.issubset(df.columns):
        return df
    out = df.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["season_id"] = pd.to_numeric(out["season_id"], errors="coerce").astype("Int64")
    out = out[out["player_id"].notna() & out["season_id"].notna()].copy()
    out["player_id"] = out["player_id"].astype(int)
    out["season_id"] = out["season_id"].astype(int)
    before = len(out)
    out = out.drop_duplicates(["player_id", "season_id"], keep="last")
    dropped = before - len(out)
    if dropped > 0:
        print(f"{label}: dropped duplicate player-season rows={dropped}", flush=True)
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return float("nan")
    x = a[m]
    y = b[m]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# Card Overall construction used by the published ~0.65 persistence number:
# recency-weighted 3yr of single-season HERO percentiles, shrink toward 50,
# then 30% pull toward the latest season. Display post-hocs are excluded.
CARD_RECENCY_DECAY = 0.40
CARD_RECENT_BLEND = 0.30
CARD_SHRINK_K = 1200.0
CARD_SHRINK_TOI_CUT = 2500.0
CARD_ONE_SEASON_SHRINK_SCALE = 0.70
CARD_GP_MIN = 40
CARD_YOY_HOLDOUT_LEFT_SEASON_DEFAULT = 20232024


@dataclass(frozen=True)
class CardYoyWindows:
    player_id: np.ndarray
    season_id: np.ndarray
    win_idx: np.ndarray
    win_w: np.ndarray
    shrink_w: np.ndarray
    pair_left: np.ndarray
    pair_right: np.ndarray
    pair_left_season: np.ndarray


def _pct_by_season_numpy(values: np.ndarray, season_id: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=float)
    for sid in np.unique(season_id):
        m = season_id == sid
        x = values[m]
        finite = np.isfinite(x)
        n = int(finite.sum())
        if n < 2:
            continue
        tmp = np.full(x.shape[0], np.nan, dtype=float)
        tmp[finite] = pd.Series(x[finite]).rank(method="average", pct=True).to_numpy(dtype=float) * 100.0
        out[m] = tmp
    return out


def _build_card_yoy_windows(
    player_id: np.ndarray,
    season_id: np.ndarray,
    toi: np.ndarray,
    gp: np.ndarray,
    *,
    decay: float = CARD_RECENCY_DECAY,
    shrink_k: float = CARD_SHRINK_K,
    shrink_toi_cut: float = CARD_SHRINK_TOI_CUT,
    one_season_scale: float = CARD_ONE_SEASON_SHRINK_SCALE,
    gp_min: int = CARD_GP_MIN,
) -> CardYoyWindows:
    n = int(len(player_id))
    win_idx = np.full((n, 3), -1, dtype=int)
    win_w = np.zeros((n, 3), dtype=float)
    shrink_w = np.zeros(n, dtype=float)
    pid = player_id.astype(np.int64, copy=False)
    sid = season_id.astype(np.int64, copy=False)
    toi_v = np.asarray(toi, dtype=float)
    gp_v = np.asarray(gp, dtype=float)
    order = np.lexsort((sid, pid))
    i = 0
    while i < n:
        j = i + 1
        while j < n and pid[order[j]] == pid[order[i]]:
            j += 1
        block = order[i:j]
        block_sid = sid[block]
        block_toi = toi_v[block]
        for loc, row in enumerate(block):
            ages = (int(block_sid[loc]) - block_sid.astype(int)) // 10001
            keep = (ages >= 0) & (ages <= 2)
            if not bool(keep.any()):
                continue
            k_idx = np.flatnonzero(keep)
            rec = np.power(float(decay), ages[k_idx].astype(float))
            ww = np.clip(block_toi[k_idx], 0.0, None) * rec
            take = min(3, int(k_idx.size))
            win_idx[row, :take] = block[k_idx][:take]
            win_w[row, :take] = ww[:take]
            toi3 = float(np.sum(np.clip(block_toi[k_idx], 0.0, None)))
            sw = 1.0
            if toi3 < float(shrink_toi_cut):
                sw = toi3 / (toi3 + float(shrink_k)) if toi3 > 0.0 else 0.0
                if int(k_idx.size) <= 1:
                    sw *= float(one_season_scale)
            shrink_w[row] = float(np.clip(sw, 0.0, 1.0))
        i = j
    pair_left: list[int] = []
    pair_right: list[int] = []
    pair_left_season: list[int] = []
    pos = { (int(pid[k]), int(sid[k])): k for k in range(n) }
    for k in range(n):
        nxt = pos.get((int(pid[k]), int(sid[k]) + 10001))
        if nxt is None:
            continue
        if not (np.isfinite(gp_v[k]) and np.isfinite(gp_v[nxt])):
            continue
        if float(gp_v[k]) < float(gp_min) or float(gp_v[nxt]) < float(gp_min):
            continue
        pair_left.append(k)
        pair_right.append(int(nxt))
        pair_left_season.append(int(sid[k]))
    return CardYoyWindows(
        player_id=pid,
        season_id=sid,
        win_idx=win_idx,
        win_w=win_w,
        shrink_w=shrink_w,
        pair_left=np.asarray(pair_left, dtype=int),
        pair_right=np.asarray(pair_right, dtype=int),
        pair_left_season=np.asarray(pair_left_season, dtype=int),
    )


def _card_from_season_pct(
    season_pct: np.ndarray,
    windows: CardYoyWindows,
    *,
    recent_blend: float = CARD_RECENT_BLEND,
) -> np.ndarray:
    pct = np.asarray(season_pct, dtype=float)
    idx = windows.win_idx
    valid = idx >= 0
    gathered = np.where(valid, pct[np.where(valid, idx, 0)], np.nan)
    finite = np.isfinite(gathered)
    w = np.where(finite, windows.win_w, 0.0)
    den = w.sum(axis=1)
    num = np.nansum(np.where(finite, gathered * windows.win_w, 0.0), axis=1)
    base = np.divide(num, den, out=np.full(pct.shape[0], np.nan), where=den > 0.0)
    shrunk = 50.0 + windows.shrink_w * (base - 50.0)
    blend = float(np.clip(float(recent_blend), 0.0, 1.0))
    card = (1.0 - blend) * shrunk + blend * pct
    return card


def _card_yoy_r2(
    season_score: np.ndarray,
    windows: CardYoyWindows,
    *,
    eval_seasons: list[int] | None = None,
    recent_blend: float = CARD_RECENT_BLEND,
) -> float:
    if windows.pair_left.size < 8:
        return float("nan")
    pct = _pct_by_season_numpy(np.asarray(season_score, dtype=float), windows.season_id)
    card = _card_from_season_pct(pct, windows, recent_blend=recent_blend)
    a = card[windows.pair_left]
    b = card[windows.pair_right]
    m = np.isfinite(a) & np.isfinite(b)
    if eval_seasons:
        m &= np.isin(windows.pair_left_season, np.asarray([int(x) for x in eval_seasons], dtype=int))
    r = _corr(a[m], b[m])
    if not np.isfinite(r):
        return float("nan")
    return float(r * r)


def _role_toi_gp(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0).to_numpy(dtype=float)
    gp_std = _to_num(d.get("eh_std_GP", pd.Series(np.nan, index=d.index)))
    gp_rapm = _to_num(d.get("eh_rapm_GP", pd.Series(np.nan, index=d.index)))
    gp = gp_std.where(gp_std.notna(), gp_rapm).to_numpy(dtype=float)
    return toi, gp


def _z_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    x = _to_num(df[col])
    g = df[group_cols]
    mu = x.groupby([g[c] for c in group_cols], dropna=False).transform("mean")
    sd = x.groupby([g[c] for c in group_cols], dropna=False).transform("std")
    return (x - mu) / sd.replace(0.0, np.nan)


def _pct_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    x = _to_num(df[col])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        sub = x.loc[idx]
        m = sub.notna()
        if int(m.sum()) < 2:
            continue
        out.loc[sub[m].index] = (sub[m].rank(method="average", pct=True) * 100.0).to_numpy()
    return out


def _resolve_walkforward_eval_seasons(
    d: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_seasons: int,
) -> list[int]:
    ss = (
        pd.to_numeric(d.get("season_id", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )
    out: list[int] = []
    for sid in ss:
        if sid < int(start_season) or sid > int(end_season):
            continue
        n_prior = sum(1 for x in ss if x < sid and x >= int(start_season))
        if n_prior >= int(min_train_seasons):
            out.append(int(sid))
    return out


def _apply_shrink_and_caps(
    df: pd.DataFrame,
    cfg: TuneConfig,
    *,
    k_map_by_role: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    d = df.copy()
    # Reliability minutes: prefer EH EV TOI when available (more complete player-season coverage),
    # fallback to A3Z tracked 5v5 TOI.
    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0)
    rel_general = toi / (toi + float(cfg.k_general))
    rel_noisy = toi / (toi + float(cfg.k_noisy))
    pos = d["position_group"].astype(str).str.upper().str.strip()

    for c in CAT_COLS:
        v = _to_num(d.get(c, pd.Series(np.nan, index=d.index)))
        if c in NOISY_COLS:
            v = v.clip(lower=-float(cfg.cap_noisy), upper=float(cfg.cap_noisy))
        if k_map_by_role is None:
            d[f"{c}__adj"] = v * (rel_noisy if c in NOISY_COLS else rel_general)
            continue

        k_f = float(k_map_by_role.get("F", {}).get(c, cfg.k_noisy if c in NOISY_COLS else cfg.k_general))
        k_d = float(k_map_by_role.get("D", {}).get(c, cfg.k_noisy if c in NOISY_COLS else cfg.k_general))
        rel_f = toi / (toi + max(k_f, 1e-9))
        rel_d = toi / (toi + max(k_d, 1e-9))
        rel = np.where(pos.eq("D"), rel_d, rel_f)
        d[f"{c}__adj"] = v * rel
    return d


def _apply_component_reliability_refinement(
    df: pd.DataFrame,
    *,
    enabled: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Event-exposure EB stabilization for weaker components.

    This is intentionally conservative: it nudges noisy component z-scores toward
    season/position means when event exposure is small.
    """
    info: dict[str, object] = {"enabled": bool(enabled), "applied": False, "components": {}}
    if not enabled:
        return df, info

    d = df.copy()
    group_cols = ["season_id", "position_group"]
    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0).clip(lower=0.0)

    preserve_tracking(d)
    evidence = exposure_proxies(d)

    # (k, blend, exposure proxy)
    recipes: dict[str, tuple[float, float, pd.Series]] = {
        "cat_finishing_z": (180.0, 0.60, evidence["finishing_shots"]),
        "cat_physicality_z": (
            120.0,
            0.55,
            evidence["physicality_events"],
        ),
        "cat_skating_z": (90.0, 0.40, (toi / 60.0).clip(lower=1.0)),
        "cat_playmaking_z": (140.0, 0.25, evidence["playmaking_assists"]),
        "cat_transition_defense_z": (220.0, 0.20, evidence["transition_defense_retrievals"]),
    }

    applied_any = False
    for col, (k, blend, n_proxy) in recipes.items():
        if col not in d.columns:
            continue
        base = _to_num(d[col])
        observed = n_proxy.notna() & base.notna()
        if not observed.any():
            info["components"][col] = {"k": float(k), "blend": float(blend),
                "observed_rows": 0, "unknown_exposure_rows": int(base.notna().sum()), "applied": False}
            continue
        eb = _eb_shrink_by_group(d.assign(_tmp=base), "_tmp", n_proxy, k=float(k), group_cols=group_cols)
        out = ((1.0 - float(blend)) * base + float(blend) * eb).where(observed, base)
        out = _z_by_group(d.assign(_tmp=out), "_tmp", group_cols)
        d[col] = out
        adj_col = f"{col}__adj"
        if adj_col in d.columns:
            old_adj = _to_num(d[adj_col])
            ratio = old_adj / base.where(base.abs() > 1e-9, np.nan)
            d[adj_col] = out * ratio.fillna(1.0)
        info["components"][col] = {"k": float(k), "blend": float(blend),
            "observed_rows": int(observed.sum()), "unknown_exposure_rows": int((base.notna() & ~observed).sum()),
            "applied": True}
        applied_any = True

    info["applied"] = bool(applied_any)
    return d, info


def _pct_by_season_pos(series: pd.Series, season: pd.Series, pos: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype=float)
    x = _to_num(series)
    tmp = pd.DataFrame({"season_id": season, "position_group": pos, "_x": x})
    for _, idx in tmp.groupby(["season_id", "position_group"], dropna=False).groups.items():
        s = x.loc[idx]
        m = s.notna()
        if int(m.sum()) < 2:
            continue
        out.loc[s[m].index] = (s[m].rank(method="average", pct=True) * 100.0).to_numpy()
    return out


def _apply_forward_calibration_policies(
    df: pd.DataFrame,
    *,
    defense_floor_pct: float,
    defense_penalty_strength: float,
    defense_penalty_power: float,
    spike_reg_strength: float,
    spike_toi_k: float,
) -> pd.DataFrame:
    d = df.copy()
    pos = d["position_group"].astype(str).str.upper().str.strip()
    is_f = pos.eq("F")
    if not is_f.any():
        return d

    # Policy 2: nonlinear downside penalty for weak forward defense.
    def_col = "cat_shot_defense_z__adj" if "cat_shot_defense_z__adj" in d.columns else "cat_5v5_defense_z__adj"
    if def_col in d.columns and float(defense_penalty_strength) > 0.0 and float(defense_floor_pct) > 0.0:
        def_pct = _pct_by_season_pos(d[def_col], d["season_id"], pos)
        floor = float(defense_floor_pct)
        gap = ((floor - pd.to_numeric(def_pct, errors="coerce")) / floor).clip(lower=0.0)
        pen = float(defense_penalty_strength) * np.power(gap.to_numpy(float), float(max(defense_penalty_power, 1.0)))
        pen_s = pd.Series(pen, index=d.index, dtype=float).fillna(0.0)
        d["fwd_defense_penalty"] = 0.0
        d.loc[is_f, "fwd_defense_penalty"] = pen_s.loc[is_f]
        d.loc[is_f, def_col] = pd.to_numeric(d.loc[is_f, def_col], errors="coerce") - d.loc[is_f, "fwd_defense_penalty"]
    else:
        d["fwd_defense_penalty"] = 0.0

    # Policy 3: regress one-season spikes in forward offense/process more strongly.
    if float(spike_reg_strength) > 0.0:
        toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
        toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
        toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0).clip(lower=0.0)
        spike_cols = [
            "cat_shot_creation_z__adj",
            "cat_transition_offense_z__adj",
        ]
        for c in spike_cols:
            if c not in d.columns:
                continue
            dd = d[["player_id", "season_id", c]].copy()
            dd["player_id"] = pd.to_numeric(dd["player_id"], errors="coerce")
            dd["season_id"] = pd.to_numeric(dd["season_id"], errors="coerce")
            dd[c] = pd.to_numeric(dd[c], errors="coerce")
            dd = dd.sort_values(["player_id", "season_id"], na_position="last")
            prior = dd.groupby("player_id")[c].transform(lambda s: s.shift(1).expanding().mean())
            seasons_to_date = dd.groupby("player_id").cumcount() + 1

            prior = prior.fillna(0.0)
            excess = (dd[c] - prior).clip(lower=0.0)
            exp_factor = ((3.0 - seasons_to_date.astype(float)) / 2.0).clip(lower=0.0, upper=1.0)
            toi_factor = (float(spike_toi_k) / (toi.loc[dd.index] + float(spike_toi_k))).clip(lower=0.0, upper=1.0)
            reg = float(spike_reg_strength) * exp_factor * toi_factor * excess
            reg = reg.where(is_f.loc[dd.index], 0.0).fillna(0.0)

            d[f"{c}__spike_reg"] = 0.0
            d.loc[dd.index, f"{c}__spike_reg"] = reg.to_numpy(float)
            d.loc[dd.index, c] = dd[c].to_numpy(float) - reg.to_numpy(float)
    return d


def _ridge_residualize(y: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Residualize ``y`` on ``x`` with simple ridge; returns NaN where unavailable."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
        return out
    m = np.isfinite(y) & np.isfinite(x).all(axis=1)
    n = int(m.sum())
    p = int(x.shape[1])
    if n < max(25, 6 * p):
        return out

    xv = x[m]
    yv = y[m]
    mu_x = np.mean(xv, axis=0)
    sd_x = np.std(xv, axis=0)
    keep = np.isfinite(sd_x) & (sd_x > 1e-4)
    if not keep.any():
        return out
    xv = (xv[:, keep] - mu_x[keep]) / sd_x[keep]
    xv = np.clip(xv, -8.0, 8.0)
    y_mu = float(np.mean(yv))
    yv = yv - y_mu

    xtx = xv.T @ xv
    xty = xv.T @ yv
    lam = max(float(alpha), 1e-9)
    try:
        beta = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0], dtype=float), xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx + lam * np.eye(xtx.shape[0], dtype=float), xty, rcond=None)[0]
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    beta = np.clip(beta, -25.0, 25.0)

    xa = (x[:, keep] - mu_x[keep]) / sd_x[keep]
    xa = np.clip(xa, -8.0, 8.0)
    m_all = np.isfinite(xa).all(axis=1) & np.isfinite(y)
    if not m_all.any():
        return out
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        pred = y_mu + np.sum(xa[m_all] * beta[None, :], axis=1)
    pred = np.nan_to_num(pred, nan=y_mu, posinf=y_mu, neginf=y_mu)
    out[m_all] = y[m_all] - pred
    return out


def _apply_possession_deoverlap(
    df: pd.DataFrame,
    *,
    enabled: bool,
    alpha: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Remove overlap from possession-process by residualizing against overlapping categories
    within each season/position group, then re-standardize to keep category scale stable.
    """
    d = df.copy()
    info = {"enabled": bool(enabled), "n_groups": 0.0, "n_rows_residualized": 0.0}
    if not enabled:
        return d, info

    y_col = "cat_possession_process_z__adj"
    if y_col not in d.columns:
        info["enabled"] = False
        return d, info

    x_cols = [
        "cat_transition_offense_z__adj",
        "cat_shot_defense_z__adj",
        "cat_transition_defense_z__adj",
        "cat_shot_creation_z__adj",
        "cat_playmaking_z__adj",
    ]
    x_cols = [c for c in x_cols if c in d.columns]
    if not x_cols:
        info["enabled"] = False
        return d, info

    y_new = pd.Series(np.nan, index=d.index, dtype=float)
    for _, idx in d.groupby(["season_id", "position_group"], dropna=False).groups.items():
        sub = d.loc[idx]
        y = _to_num(sub[y_col]).to_numpy(float)
        x = np.column_stack([_to_num(sub[c]).to_numpy(float) for c in x_cols])
        resid = _ridge_residualize(y, x, alpha=float(alpha))
        m = np.isfinite(resid)
        if m.any():
            y_new.loc[sub.index[m]] = resid[m]
            info["n_rows_residualized"] += float(m.sum())
            info["n_groups"] += 1.0

    d[f"{y_col}__pre_deoverlap"] = _to_num(d[y_col])
    d.loc[y_new.notna(), y_col] = y_new.loc[y_new.notna()]
    d[y_col] = _z_by_group(d, y_col, ["season_id", "position_group"])
    return d, info


def _apply_defense_collinearity_control(
    df: pd.DataFrame,
    *,
    enabled: bool,
    alpha: float,
    residual_weight: float,
    min_abs_corr: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Reduce double-counting between forward defensive categories by residualizing
    Transition Defense against Shot Defense within each season/position group.
    """
    d = df.copy()
    info = {
        "enabled": bool(enabled),
        "n_groups": 0.0,
        "n_rows_residualized": 0.0,
        "residual_weight": float(residual_weight),
        "min_abs_corr": float(min_abs_corr),
        "n_groups_skipped_low_corr": 0.0,
    }
    if not enabled:
        return d, info
    y_col = "cat_transition_defense_z__adj"
    x_col = "cat_shot_defense_z__adj"
    if y_col not in d.columns or x_col not in d.columns:
        info["enabled"] = False
        return d, info

    y_new = pd.Series(np.nan, index=d.index, dtype=float)
    grp = d["position_group"].astype(str).str.upper().str.strip()
    d = d.assign(__pos_grp=grp)
    rw = float(np.clip(float(residual_weight), 0.0, 1.0))
    min_corr = float(max(0.0, min(0.999, float(min_abs_corr))))
    for (_, pos), idx in d.groupby(["season_id", "__pos_grp"], dropna=False).groups.items():
        if str(pos) != "F":
            continue
        sub = d.loc[idx]
        y = _to_num(sub[y_col]).to_numpy(float)
        x = _to_num(sub[x_col]).to_numpy(float).reshape(-1, 1)
        m_corr = np.isfinite(y) & np.isfinite(x[:, 0])
        if int(m_corr.sum()) < 30:
            continue
        sx = np.std(x[m_corr, 0])
        sy = np.std(y[m_corr])
        if float(sx) <= 1e-9 or float(sy) <= 1e-9:
            continue
        corr = float(np.corrcoef(y[m_corr], x[m_corr, 0])[0, 1])
        if not np.isfinite(corr) or abs(corr) < min_corr:
            info["n_groups_skipped_low_corr"] += 1.0
            continue
        resid = _ridge_residualize(y, x, alpha=float(alpha))
        m = np.isfinite(resid)
        if m.any():
            # Keep part of original transition-defense signal so we do not
            # over-strip true defense information.
            y_adj = (1.0 - rw) * y[m] + rw * resid[m]
            y_new.loc[sub.index[m]] = y_adj
            info["n_rows_residualized"] += float(m.sum())
            info["n_groups"] += 1.0

    d[f"{y_col}__pre_collinearity"] = _to_num(d[y_col])
    d.loc[y_new.notna(), y_col] = y_new.loc[y_new.notna()]
    d[y_col] = _z_by_group(d, y_col, ["season_id", "position_group"])
    d = d.drop(columns=["__pos_grp"], errors="ignore")
    return d, info


def _weighted_row_score(mat: np.ndarray, w: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mat)
    num = np.nansum(np.where(finite, mat * w[None, :], 0.0), axis=1)
    den = np.sum(np.where(finite, w[None, :], 0.0), axis=1)
    out = np.full(mat.shape[0], np.nan, dtype=float)
    m = den > 0.0
    out[m] = num[m] / den[m]
    return out


def _objective_for_role(
    d: pd.DataFrame,
    w: np.ndarray,
    adj_cols: list[str],
    *,
    objective_target: pd.Series | None = None,
    objective_score: str = "corr",
    eval_seasons: list[int] | None = None,
    card_windows: CardYoyWindows | None = None,
) -> float:
    if d.empty:
        return float("nan")
    mat = d[adj_cols].to_numpy(dtype=float)
    score = _weighted_row_score(mat, w)
    if card_windows is not None:
        return _card_yoy_r2(score, card_windows, eval_seasons=eval_seasons)

    if eval_seasons:
        seasons = [int(x) for x in eval_seasons]
    else:
        seasons = sorted(pd.to_numeric(d["season_id"], errors="coerce").dropna().astype(int).unique())
    season_scores: list[float] = []

    for sid in seasons:
        s = d[pd.to_numeric(d["season_id"], errors="coerce").astype("Int64") == sid].copy()
        idx = s.index.to_numpy()
        sc = score[d.index.get_indexer(idx)]

        if objective_target is not None:
            y = _to_num(objective_target.reindex(idx)).to_numpy(dtype=float)
            if str(objective_score).lower() == "logloss":
                m = np.isfinite(sc) & np.isfinite(y)
                if int(m.sum()) < 20:
                    continue
                ys = y[m]
                # Require binary labels; if only one class appears, skip that season.
                cls = np.unique(ys)
                if not set(cls.tolist()).issubset({0.0, 1.0}) or cls.size < 2:
                    continue
                xs = sc[m]
                mu = float(np.nanmean(xs))
                sd = float(np.nanstd(xs))
                if not np.isfinite(sd) or sd < 1e-9:
                    continue
                z = (xs - mu) / sd
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                ll = -np.mean(ys * np.log(p) + (1.0 - ys) * np.log(1.0 - p))
                if np.isfinite(ll):
                    season_scores.append(float(-ll))
            else:
                r = _corr(sc, y)
                if np.isfinite(r):
                    season_scores.append(float(r))
            continue

        r_xgf = _corr(sc, _to_num(s["yoy_xGF_per60"]).to_numpy(dtype=float))
        r_xga = _corr(sc, -_to_num(s["yoy_xGA_per60"]).to_numpy(dtype=float))
        r_xgs = _corr(sc, _to_num(s["yoy_onice_xg_share"]).to_numpy(dtype=float))
        r_goal = _corr(sc, _to_num(s["yoy_onice_goal_share"]).to_numpy(dtype=float))

        vals = np.array([r_xgf, r_xga, r_xgs, r_goal], dtype=float)
        wts = np.array([0.35, 0.35, 0.20, 0.10], dtype=float)
        m = np.isfinite(vals)
        if not m.any():
            continue
        sv = float(np.sum(vals[m] * wts[m]) / np.sum(wts[m]))
        season_scores.append(sv)

    if not season_scores:
        return float("nan")
    return float(np.mean(season_scores))


SAME_SEASON_IMPACT_FEATURE_COLS = (
    "eh_rapm_xG±/60",
    "eh_rapm_xGF/60",
    "eh_rapm_xGA/60",
    "ptv_season_rapm_net_possession_impact",
    "ptv_roll3_rapm_net_possession_impact",
    "ptv_pooled_rapm_net_possession_impact",
    "ptv_season_rapm_possession_offense",
    "ptv_roll3_rapm_possession_offense",
    "ptv_pooled_rapm_possession_offense",
    "ptv_season_rapm_possession_defense",
    "ptv_roll3_rapm_possession_defense",
    "ptv_pooled_rapm_possession_defense",
    "ptv_season_rapm_shot_threat",
    "ptv_roll3_rapm_shot_threat",
    "ptv_pooled_rapm_shot_threat",
)


def _coalesce_num(df: pd.DataFrame, *cols: str) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in cols:
        s = _to_num(df.get(c, pd.Series(np.nan, index=df.index)))
        out = out.where(out.notna(), s)
    return out


def _build_same_season_impact_anchor(
    df: pd.DataFrame,
    *,
    xgf_w: float = 0.20,
    xga_w: float = 0.25,
    xgs_w: float = 0.45,
    gs_w: float = 0.10,
) -> pd.Series:
    """Process-based current-season impact: EH RAPM + PTV coalesced (season → roll3 → pooled)."""
    xgf = _coalesce_num(
        df,
        "eh_rapm_xGF/60",
        "ptv_season_rapm_possession_offense",
        "ptv_roll3_rapm_possession_offense",
        "ptv_pooled_rapm_possession_offense",
    )
    xga_raw = _coalesce_num(
        df,
        "eh_rapm_xGA/60",
        "ptv_season_rapm_possession_defense",
        "ptv_roll3_rapm_possession_defense",
        "ptv_pooled_rapm_possession_defense",
    )
    xga = -xga_raw
    xgs = _coalesce_num(
        df,
        "eh_rapm_xG±/60",
        "ptv_season_rapm_net_possession_impact",
        "ptv_roll3_rapm_net_possession_impact",
        "ptv_pooled_rapm_net_possession_impact",
    )
    gs = _coalesce_num(
        df,
        "ptv_season_rapm_shot_threat",
        "ptv_roll3_rapm_shot_threat",
        "ptv_pooled_rapm_shot_threat",
    )
    tmp = pd.DataFrame(
        {
            "season_id": df["season_id"],
            "position_group": df["position_group"],
            "_xgf": xgf,
            "_xga": xga,
            "_xgs": xgs,
            "_gs": gs,
        },
        index=df.index,
    )
    zxgf = _z_by_group(tmp, "_xgf", ["season_id", "position_group"])
    zxga = _z_by_group(tmp, "_xga", ["season_id", "position_group"])
    zxgs = _z_by_group(tmp, "_xgs", ["season_id", "position_group"])
    zgs = _z_by_group(tmp, "_gs", ["season_id", "position_group"])
    w = np.asarray([xgf_w, xga_w, xgs_w, gs_w], dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if float(np.sum(np.abs(w))) <= 1e-9:
        w = np.asarray([0.20, 0.25, 0.45, 0.10], dtype=float)
    w = w / float(np.sum(np.abs(w)))
    return float(w[0]) * zxgf + float(w[1]) * zxga + float(w[2]) * zxgs + float(w[3]) * zgs


def _build_objective_target(
    df: pd.DataFrame,
    *,
    mode: str = "anchor",
    anchor_type: str = "yoy",
    residual_baseline_col: str = "overall_8cat_z",
    residual_alpha: float = 2.0,
    xgf_w: float = 0.35,
    xga_w: float = 0.35,
    xgs_w: float = 0.20,
    gs_w: float = 0.10,
    fallback_next_overall: bool = True,
    anchor_next_overall_blend: float = 0.75,
    fallback_source_col: str = "overall_8cat_z",
) -> pd.Series:
    """
    Build a row-level composite target used for category-level k selection.
    Uses the same anchor blend as the main objective.
    """
    if str(anchor_type).lower() == "same_season_impact":
        anchor = _build_same_season_impact_anchor(
            df,
            xgf_w=float(xgf_w),
            xga_w=float(xga_w),
            xgs_w=float(xgs_w),
            gs_w=float(gs_w),
        )
        if str(mode).lower() != "residual":
            return anchor
        b = _to_num(df.get(residual_baseline_col, pd.Series(np.nan, index=df.index)))
        out = anchor.copy()
        grp = pd.DataFrame(
            {
                "season_id": pd.to_numeric(df["season_id"], errors="coerce"),
                "position_group": df["position_group"].astype(str).str.upper().str.strip(),
                "_y": anchor,
                "_b": b,
            },
            index=df.index,
        )
        for _, g in grp.groupby(["season_id", "position_group"], dropna=False):
            m = g["_y"].notna() & g["_b"].notna()
            if int(m.sum()) < 20:
                continue
            idx = g.index[m]
            yv = g.loc[idx, "_y"].to_numpy(dtype=float)
            xv = g.loc[idx, "_b"].to_numpy(dtype=float).reshape(-1, 1)
            out.loc[idx] = _ridge_residualize(yv, xv, alpha=float(residual_alpha))
        tmp2 = pd.DataFrame(
            {"season_id": df["season_id"], "position_group": df["position_group"], "_res": out},
            index=df.index,
        )
        return _z_by_group(tmp2, "_res", ["season_id", "position_group"])

    xgf = _to_num(df.get("yoy_xGF_per60", pd.Series(np.nan, index=df.index)))
    xga = -_to_num(df.get("yoy_xGA_per60", pd.Series(np.nan, index=df.index)))
    xgs = _to_num(df.get("yoy_onice_xg_share", pd.Series(np.nan, index=df.index)))
    gs = _to_num(df.get("yoy_onice_goal_share", pd.Series(np.nan, index=df.index)))

    tmp = pd.DataFrame(
        {
            "season_id": df["season_id"],
            "position_group": df["position_group"],
            "_xgf": xgf,
            "_xga": xga,
            "_xgs": xgs,
            "_gs": gs,
        }
    )
    zxgf = _z_by_group(tmp, "_xgf", ["season_id", "position_group"])
    zxga = _z_by_group(tmp, "_xga", ["season_id", "position_group"])
    zxgs = _z_by_group(tmp, "_xgs", ["season_id", "position_group"])
    zgs = _z_by_group(tmp, "_gs", ["season_id", "position_group"])
    w = np.asarray([xgf_w, xga_w, xgs_w, gs_w], dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if float(np.sum(np.abs(w))) <= 1e-9:
        w = np.asarray([0.35, 0.35, 0.20, 0.10], dtype=float)
    w = w / float(np.sum(np.abs(w)))
    anchor = float(w[0]) * zxgf + float(w[1]) * zxga + float(w[2]) * zxgs + float(w[3]) * zgs

    target = anchor.copy()
    if bool(fallback_next_overall) and fallback_source_col in df.columns:
        dd = pd.DataFrame(
            {
                "player_id": pd.to_numeric(df.get("player_id", pd.Series(np.nan, index=df.index)), errors="coerce"),
                "season_id": pd.to_numeric(df.get("season_id", pd.Series(np.nan, index=df.index)), errors="coerce"),
                "position_group": df.get("position_group", pd.Series("", index=df.index)).astype(str).str.upper().str.strip(),
                "_base": _to_num(df.get(fallback_source_col, pd.Series(np.nan, index=df.index))),
            },
            index=df.index,
        )
        dd = dd[dd["player_id"].notna() & dd["season_id"].notna()].copy()
        if not dd.empty:
            dd["player_id"] = dd["player_id"].astype(int)
            dd["season_id"] = dd["season_id"].astype(int)
            dd = dd.sort_values(["player_id", "season_id"], kind="mergesort")
            dd["_next"] = dd.groupby("player_id", dropna=False)["_base"].shift(-1)
            dd["_next_z"] = _z_by_group(dd, "_next", ["season_id", "position_group"])
            nxt = dd["_next_z"].reindex(df.index)
            b = float(np.clip(float(anchor_next_overall_blend), 0.0, 1.0))
            # Blend when both available; otherwise use whichever signal exists.
            both = target.notna() & nxt.notna()
            target = target.where(target.notna(), nxt)
            target = target.where(~both, b * target + (1.0 - b) * nxt)

    if str(mode).lower() != "residual":
        return target

    b = _to_num(df.get(residual_baseline_col, pd.Series(np.nan, index=df.index)))
    out = target.copy()
    grp = pd.DataFrame(
        {
            "season_id": pd.to_numeric(df["season_id"], errors="coerce"),
            "position_group": df["position_group"].astype(str).str.upper().str.strip(),
            "_y": anchor,
            "_b": b,
        },
        index=df.index,
    )
    for _, g in grp.groupby(["season_id", "position_group"], dropna=False):
        m = g["_y"].notna() & g["_b"].notna()
        if int(m.sum()) < 20:
            continue
        idx = g.index[m]
        yv = g.loc[idx, "_y"].to_numpy(dtype=float)
        xv = g.loc[idx, "_b"].to_numpy(dtype=float).reshape(-1, 1)
        out.loc[idx] = _ridge_residualize(yv, xv, alpha=float(residual_alpha))
    tmp2 = pd.DataFrame({"season_id": df["season_id"], "position_group": df["position_group"], "_res": out}, index=df.index)
    return _z_by_group(tmp2, "_res", ["season_id", "position_group"])


def _build_next_year_binary_target(
    df: pd.DataFrame,
    *,
    source_col: str = "overall_8cat_z",
) -> pd.Series:
    """
    Binary YoY target for log-loss tuning.
    Label=1 when next-season value of source_col is above role+season mean (z>=0).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if source_col not in df.columns:
        return out
    d = pd.DataFrame(
        {
            "player_id": pd.to_numeric(df.get("player_id", pd.Series(np.nan, index=df.index)), errors="coerce"),
            "season_id": pd.to_numeric(df.get("season_id", pd.Series(np.nan, index=df.index)), errors="coerce"),
            "position_group": df.get("position_group", pd.Series("", index=df.index)).astype(str).str.upper().str.strip(),
            "_base": _to_num(df.get(source_col, pd.Series(np.nan, index=df.index))),
        },
        index=df.index,
    )
    d = d[d["player_id"].notna() & d["season_id"].notna()].copy()
    if d.empty:
        return out
    d["player_id"] = d["player_id"].astype(int)
    d["season_id"] = d["season_id"].astype(int)
    d = d.sort_values(["player_id", "season_id"], kind="mergesort")
    d["_next"] = d.groupby("player_id", dropna=False)["_base"].shift(-1)
    d["_next_z"] = _z_by_group(d.assign(_x=d["_next"]), "_x", ["season_id", "position_group"])
    m = d["_next_z"].notna()
    out.loc[d.index[m]] = (pd.to_numeric(d.loc[m, "_next_z"], errors="coerce") >= 0.0).astype(float).to_numpy(float)
    return out


def _build_fixed_benchmark_target(df: pd.DataFrame) -> pd.Series:
    """
    Locked benchmark target for apples-to-apples model comparisons across runs.
    """
    xgf = _to_num(df.get("yoy_xGF_per60", pd.Series(np.nan, index=df.index)))
    xga_inv = -_to_num(df.get("yoy_xGA_per60", pd.Series(np.nan, index=df.index)))
    xgs = _to_num(df.get("yoy_onice_xg_share", pd.Series(np.nan, index=df.index)))
    gs = _to_num(df.get("yoy_onice_goal_share", pd.Series(np.nan, index=df.index)))
    tmp = pd.DataFrame(
        {
            "season_id": df.get("season_id"),
            "position_group": df.get("position_group"),
            "_xgf": xgf,
            "_xga_inv": xga_inv,
            "_xgs": xgs,
            "_gs": gs,
        },
        index=df.index,
    )
    zxgf = _z_by_group(tmp, "_xgf", ["season_id", "position_group"])
    zxga = _z_by_group(tmp, "_xga_inv", ["season_id", "position_group"])
    zxgs = _z_by_group(tmp, "_xgs", ["season_id", "position_group"])
    zgs = _z_by_group(tmp, "_gs", ["season_id", "position_group"])
    w = FIXED_BENCHMARK_TARGET_WEIGHTS
    return (
        float(w["xGF_per60"]) * zxgf
        + float(w["xGA_per60_flipped"]) * zxga
        + float(w["onice_xg_share"]) * zxgs
        + float(w["onice_goal_share"]) * zgs
    )


def _evaluate_fixed_benchmark(
    df: pd.DataFrame,
    *,
    score_col: str,
    eval_seasons: list[int] | None = None,
) -> dict[str, float]:
    out = {
        "r_all_season_mean": float("nan"),
        "r2_all_season_mean": float("nan"),
        "r_all_pooled": float("nan"),
        "r2_all_pooled": float("nan"),
        "r_F_season_mean": float("nan"),
        "r2_F_season_mean": float("nan"),
        "r_D_season_mean": float("nan"),
        "r2_D_season_mean": float("nan"),
    }
    if score_col not in df.columns:
        return out

    d = df.copy()
    d["season_id"] = pd.to_numeric(d.get("season_id"), errors="coerce").astype("Int64")
    d["position_group"] = d.get("position_group", pd.Series("", index=d.index)).astype(str).str.upper().str.strip()
    score = _to_num(d.get(score_col, pd.Series(np.nan, index=d.index)))
    target = _build_fixed_benchmark_target(d)

    if eval_seasons:
        eval_set = set(int(x) for x in eval_seasons)
        d = d[d["season_id"].isin(eval_set)].copy()
        score = score.reindex(d.index)
        target = target.reindex(d.index)

    r_pool = _corr(score.to_numpy(float), target.to_numpy(float))
    if np.isfinite(r_pool):
        out["r_all_pooled"] = float(r_pool)
        out["r2_all_pooled"] = float(r_pool * r_pool)

    rs_all: list[float] = []
    rs_f: list[float] = []
    rs_d: list[float] = []
    for (sid, pos), idx in d.groupby(["season_id", "position_group"], dropna=False).groups.items():
        _ = sid  # keep grouped key explicit for clarity
        r = _corr(score.loc[idx].to_numpy(float), target.loc[idx].to_numpy(float))
        if not np.isfinite(r):
            continue
        rs_all.append(float(r))
        p = str(pos).upper().strip()
        if p == "F":
            rs_f.append(float(r))
        elif p == "D":
            rs_d.append(float(r))

    if rs_all:
        ra = float(np.mean(rs_all))
        out["r_all_season_mean"] = ra
        out["r2_all_season_mean"] = float(ra * ra)
    if rs_f:
        rf = float(np.mean(rs_f))
        out["r_F_season_mean"] = rf
        out["r2_F_season_mean"] = float(rf * rf)
    if rs_d:
        rd = float(np.mean(rs_d))
        out["r_D_season_mean"] = rd
        out["r2_D_season_mean"] = float(rd * rd)
    return out


def _tune_k_by_category(
    train: pd.DataFrame,
    role: str,
    *,
    k_candidates: list[float],
    cap_noisy: float,
    default_k_general: float,
    default_k_noisy: float,
    objective_score: str = "corr",
    objective_mode: str = "anchor",
    residual_baseline_col: str = "overall_8cat_z",
    residual_alpha: float = 2.0,
    xgf_w: float = 0.35,
    xga_w: float = 0.35,
    xgs_w: float = 0.20,
    gs_w: float = 0.10,
    fallback_next_overall: bool = True,
    anchor_next_overall_blend: float = 0.75,
    fallback_source_col: str = "overall_8cat_z",
    anchor_type: str = "yoy",
    eval_seasons: list[int] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Pick per-category TOI shrinkage k for one role using train seasons.
    Criterion:
    - corr: maximize mean season-level correlation(category_shrunk, objective target)
    - logloss: minimize mean season-level binary log-loss vs next-year binary target
    """
    d = train.copy()
    pos = d["position_group"].astype(str).str.upper().str.strip()
    d = d[pos.eq(role)].copy()
    if d.empty:
        k_map = {
            c: (float(default_k_noisy) if c in NOISY_COLS else float(default_k_general))
            for c in CAT_COLS
        }
        return k_map, {c: float("nan") for c in CAT_COLS}

    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0).clip(lower=0.0)
    if str(objective_score).lower() == "logloss":
        y = _build_next_year_binary_target(
            d,
            source_col=str(fallback_source_col),
        )
    else:
        y = _build_objective_target(
            d,
            mode=objective_mode,
            anchor_type=str(anchor_type),
            residual_baseline_col=residual_baseline_col,
            residual_alpha=float(residual_alpha),
            xgf_w=float(xgf_w),
            xga_w=float(xga_w),
            xgs_w=float(xgs_w),
            gs_w=float(gs_w),
            fallback_next_overall=bool(fallback_next_overall),
            anchor_next_overall_blend=float(anchor_next_overall_blend),
            fallback_source_col=str(fallback_source_col),
        )

    def _season_mean_score(series: pd.Series) -> float:
        vals: list[float] = []
        dd = d.assign(_x=series, _y=y)
        if eval_seasons:
            season_iter = [g for sid, g in dd.groupby("season_id", dropna=False) if sid in set(int(x) for x in eval_seasons)]
        else:
            season_iter = [g for _, g in dd.groupby("season_id", dropna=False)]
        for g in season_iter:
            x = _to_num(g["_x"]).to_numpy(float)
            yy = _to_num(g["_y"]).to_numpy(float)
            if str(objective_score).lower() == "logloss":
                m = np.isfinite(x) & np.isfinite(yy)
                if int(m.sum()) < 20:
                    continue
                ys = yy[m]
                cls = np.unique(ys)
                if not set(cls.tolist()).issubset({0.0, 1.0}) or cls.size < 2:
                    continue
                xs = x[m]
                mu = float(np.nanmean(xs))
                sd = float(np.nanstd(xs))
                if not np.isfinite(sd) or sd < 1e-9:
                    continue
                z = (xs - mu) / sd
                p = 1.0 / (1.0 + np.exp(-z))
                p = np.clip(p, 1e-6, 1.0 - 1e-6)
                ll = -np.mean(ys * np.log(p) + (1.0 - ys) * np.log(1.0 - p))
                if np.isfinite(ll):
                    vals.append(float(-ll))
            else:
                r = _corr(x, yy)
                if np.isfinite(r):
                    vals.append(float(r))
        return float(np.mean(vals)) if vals else float("nan")

    k_sel: dict[str, float] = {}
    obj_sel: dict[str, float] = {}
    for c in CAT_COLS:
        raw = _to_num(d.get(c, pd.Series(np.nan, index=d.index)))
        if c in NOISY_COLS:
            raw = raw.clip(lower=-float(cap_noisy), upper=float(cap_noisy))
        best_k = float(default_k_noisy if c in NOISY_COLS else default_k_general)
        best_obj = -np.inf
        for k in k_candidates:
            rel = toi / (toi + float(k))
            obj = _season_mean_score(raw * rel)
            if np.isfinite(obj) and obj > best_obj:
                best_obj = float(obj)
                best_k = float(k)
        if not np.isfinite(best_obj):
            rel = toi / (toi + float(best_k))
            best_obj = _season_mean_score(raw * rel)
        k_sel[c] = best_k
        obj_sel[c] = float(best_obj) if np.isfinite(best_obj) else float("nan")
    return k_sel, obj_sel


def _apply_k_floors(
    k_map: dict[str, float],
    *,
    role: str,
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    out = {str(k): float(v) for k, v in k_map.items()}
    changes: dict[str, tuple[float, float]] = {}
    floors = MIN_K_FLOOR_BY_ROLE_CATEGORY.get(str(role).upper().strip(), {})
    for cat, floor in floors.items():
        if cat not in out:
            continue
        before = float(out[cat])
        after = float(max(before, float(floor)))
        if after != before:
            out[cat] = after
            changes[cat] = (before, after)
    return out, changes


def _winsor_by_group(
    df: pd.DataFrame,
    col: str,
    group_cols: list[str],
    *,
    lo: float = 0.025,
    hi: float = 0.975,
) -> pd.Series:
    x = _to_num(df[col]).copy()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        s = x.loc[idx]
        m = s.notna()
        if int(m.sum()) < 20:
            out.loc[idx] = s
            continue
        ql = float(np.nanquantile(s[m], lo))
        qh = float(np.nanquantile(s[m], hi))
        out.loc[idx] = s.clip(lower=ql, upper=qh)
    return out


def _event_exposure_proxy(df: pd.DataFrame, feat: str, toi: pd.Series) -> pd.Series:
    f = feat.lower()
    if "per_exit_touch" in f:
        zex = _to_num(df.get("a3z_zone_exits", pd.Series(np.nan, index=df.index)))
        cex = _to_num(df.get("a3z_controlled_exits", pd.Series(np.nan, index=df.index)))
        car = _to_num(df.get("a3z_carried_exits", pd.Series(np.nan, index=df.index)))
        pex = _to_num(df.get("a3z_passed_exits", pd.Series(np.nan, index=df.index)))
        n = zex.where(zex.notna(), cex.fillna(0.0) + car.fillna(0.0) + pex.fillna(0.0))
        return n.clip(lower=1.0)
    if "per_entry" in f:
        ent = _to_num(df.get("a3z_entries", pd.Series(np.nan, index=df.index)))
        return ent.clip(lower=1.0)
    return (toi / 60.0).clip(lower=1.0)


def _eb_shrink_by_group(
    df: pd.DataFrame,
    col: str,
    n_proxy: pd.Series,
    *,
    k: float,
    group_cols: list[str],
) -> pd.Series:
    x = _to_num(df[col])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        s = x.loc[idx]
        n = _to_num(n_proxy.loc[idx]).fillna(0.0).clip(lower=0.0)
        sm = s.to_numpy(float)
        gm = float(np.nanmean(sm)) if np.isfinite(sm).any() else float("nan")
        w = n / (n + float(max(k, 1e-9)))
        out.loc[idx] = w * s + (1.0 - w) * gm
    return out


def _ridge_fit_predict(X: np.ndarray, y: np.ndarray, Xp: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xp = np.asarray(Xp, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or Xp.ndim != 2 or X.shape[1] != Xp.shape[1]:
        return np.full(Xp.shape[0], np.nan, dtype=float)
    m = np.isfinite(y)
    if int(m.sum()) < max(40, 8 * X.shape[1]):
        return np.full(Xp.shape[0], np.nan, dtype=float)
    Xt = X[m]
    # These targets/features are standardized hockey-rate signals. Values far
    # outside this range are source sentinels or denominator artifacts, not
    # meaningful player seasons.
    yt = np.clip(y[m], -12.0, 12.0)
    mu = np.nanmean(Xt, axis=0)
    Xt = np.where(np.isfinite(Xt), Xt, mu[None, :])
    sd = np.nanstd(Xt, axis=0)
    keep = np.isfinite(sd) & (sd > 1e-6)
    if not keep.any():
        return np.full(Xp.shape[0], np.nan, dtype=float)
    Xt = (Xt[:, keep] - mu[keep]) / sd[keep]
    Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    Xt = np.clip(Xt, -12.0, 12.0)
    yp = yt - float(np.nanmean(yt))
    lam = float(max(alpha, 1e-9))
    xtx = Xt.T @ Xt
    xty = Xt.T @ yp
    try:
        beta = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0], dtype=float), xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx + lam * np.eye(xtx.shape[0], dtype=float), xty, rcond=None)[0]
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
    beta = np.clip(beta, -25.0, 25.0)
    y_mu = float(np.nanmean(yt))
    Xall = Xp[:, keep]
    out = np.full(Xp.shape[0], np.nan, dtype=float)
    if Xall.size == 0:
        return out
    # Impute missing eval-time feature values with training means so sparse rows
    # still receive predictions during walk-forward folds.
    Xfill = np.where(np.isfinite(Xall), Xall, mu[keep][None, :])
    Xallz = (Xfill - mu[keep]) / sd[keep]
    Xallz = np.nan_to_num(Xallz, nan=0.0, posinf=0.0, neginf=0.0)
    # A handful of malformed/sentinel source values can remain finite but many
    # standard deviations from the training distribution.  Bound their leverage
    # so one bad row cannot overflow an entire walk-forward objective.
    Xallz = np.clip(Xallz, -12.0, 12.0)
    # Explicit reduction is more stable than the platform BLAS matmul here
    # (Accelerate emitted spurious overflow/divide warnings on bounded arrays).
    out[:] = y_mu + np.sum(Xallz * beta[None, :], axis=1)
    return out


def _walkforward_ridge_predict(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    role: str,
    alpha_grid: list[float],
) -> tuple[pd.Series, dict[str, float]]:
    d = df[df["position_group"].astype(str).str.upper().eq(role)].copy()
    d["season_id"] = pd.to_numeric(d["season_id"], errors="coerce").astype("Int64")
    seasons = sorted([int(x) for x in d["season_id"].dropna().unique()])
    if len(seasons) < 4:
        return pd.Series(np.nan, index=df.index, dtype=float), {"alpha": float("nan"), "cv_r": float("nan")}

    eval_seasons = seasons[-2:]
    best_alpha = float(alpha_grid[0])
    best_score = float("nan")
    for a in alpha_grid:
        rs: list[float] = []
        for sid in eval_seasons:
            tr = d[d["season_id"] < sid]
            te = d[d["season_id"] == sid]
            if tr.empty or te.empty:
                continue
            Xtr = tr[feature_cols].to_numpy(float)
            ytr = _to_num(tr[target_col]).to_numpy(float)
            Xte = te[feature_cols].to_numpy(float)
            p = _ridge_fit_predict(Xtr, ytr, Xte, alpha=float(a))
            r = _corr(p, _to_num(te[target_col]).to_numpy(float))
            if np.isfinite(r):
                rs.append(float(r))
        mean_r = float(np.mean(rs)) if rs else float("nan")
        if np.isfinite(mean_r) and (not np.isfinite(best_score) or mean_r > best_score):
            best_score = mean_r
            best_alpha = float(a)

    last_sid = seasons[-1]
    tr_all = d[d["season_id"] < last_sid]
    Xtr = tr_all[feature_cols].to_numpy(float)
    ytr = _to_num(tr_all[target_col]).to_numpy(float)
    Xall = d[feature_cols].to_numpy(float)
    pred = _ridge_fit_predict(Xtr, ytr, Xall, alpha=best_alpha)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[d.index] = pred
    return out, {"alpha": float(best_alpha), "cv_r": float(best_score) if np.isfinite(best_score) else float("nan")}


def _rapm_defensive_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    xga = _to_num(df.get("rapm_hybrid_xGA/60", pd.Series(np.nan, index=df.index)))
    tmp = df[group_cols].copy()
    tmp["_rapm_def_skill"] = -xga
    return _z_by_group(tmp, "_rapm_def_skill", group_cols)


def _anchor_d_defense_z_to_rapm(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    sd_blend: float,
    td_blend: float,
) -> pd.DataFrame:
    """Pull D shot/transition defense z toward RAPM xGA/CA after latent v2 upgrades."""
    d = df.copy()
    rapm_z = _rapm_defensive_skill_z(d, group_cols)
    is_d = d["position_group"].astype(str).str.upper().str.strip().eq("D")
    for col, blend in [("cat_shot_defense_z", sd_blend), ("cat_transition_defense_z", td_blend)]:
        b = float(np.clip(float(blend), 0.0, 1.0))
        if b <= 0.0 or col not in d.columns:
            continue
        z = _to_num(d[col])
        out = z.copy()
        m = is_d & z.notna() & rapm_z.notna()
        out.loc[m] = (1.0 - b) * z.loc[m] + b * rapm_z.loc[m]
        d[col] = out
        adj_col = f"{col}__adj"
        if adj_col in d.columns:
            old_adj = _to_num(d[adj_col])
            ratio = old_adj / z.where(z.abs() > 1e-9, np.nan)
            d[adj_col] = out * ratio.fillna(1.0)
    return d


def _apply_defense_signal_upgrade(
    df: pd.DataFrame,
    *,
    feature_source: Path,
    enabled: bool,
    transition_v2_blend: float,
    transition_v2_blend_f: float | None,
    transition_v2_blend_d: float | None,
    shot_v2_blend: float,
    shot_v2_latent_weight: float,
    entry_weight: float,
    exit_weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {"enabled": bool(enabled), "applied": False}
    if not enabled:
        return df, info

    d = df.copy()
    need = [
        "player_id",
        "season_id",
        "position_group",
        "eh_rapm_TOI",
        "a3z_toi_5v5_min",
        "yoy_xGA_per60",
        "yoy_onice_xg_share",
        "eh_std_iHF/60",
        "eh_std_TAKE/60",
        "eh_std_iBLK/60",
        "eh_std_sh_iBLK/60",
        "a3z_failed_exits_per_exit_touch",
        "a3z_failed_entries_per_entry",
        "a3z_retrievals_leading_to_exits_per_60",
        "a3z_module_entry_defense",
        "a3z_module_breakout_exit_value",
        "a3z_zone_exits",
        "a3z_controlled_exits",
        "a3z_carried_exits",
        "a3z_passed_exits",
        "a3z_entries",
        "ptv_roll3_rapm_possession_defense",
        "ptv_roll3_rapm_threat_suppression",
        "ptv_pooled_rapm_possession_defense",
        "ptv_pooled_rapm_threat_suppression",
        "ptv_season_rapm_possession_defense",
        "ptv_season_rapm_threat_suppression",
        "rapm_hybrid_xGA/60",
        "eh_rapm_CA/60",
    ]
    need_missing = [c for c in need if c not in d.columns]
    if need_missing:
        fs = _read_table(feature_source)
        fs = _dedupe_player_season(fs, label="tune_all_in_one_8cat_model defense_upgrade feature_source")
        keep = ["player_id", "season_id"] + [c for c in need_missing if c in fs.columns]
        fs = fs[keep].copy()
        d = d.merge(fs, on=["player_id", "season_id"], how="left")

    required = [
        "cat_transition_defense_z",
        "cat_shot_defense_z",
        "cat_transition_defense_z__adj",
        "cat_shot_defense_z__adj",
        "yoy_xGA_per60",
        "yoy_onice_xg_share",
    ]
    if any(c not in d.columns for c in required):
        info["enabled"] = False
        info["reason"] = "missing_required_columns"
        return d, info

    d["position_group"] = d["position_group"].astype(str).str.upper().str.strip()
    group = ["season_id", "position_group"]
    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    d["toi_v2"] = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0)

    noisy_feats = [
        "eh_std_iHF/60",
        "eh_std_TAKE/60",
        "eh_std_iBLK/60",
        "eh_std_sh_iBLK/60",
        "a3z_failed_exits_per_exit_touch",
        "a3z_failed_entries_per_entry",
        "a3z_retrievals_leading_to_exits_per_60",
        "a3z_module_entry_defense",
        "a3z_module_breakout_exit_value",
    ]
    noisy_feats = [c for c in noisy_feats if c in d.columns]
    for c in noisy_feats:
        w = _winsor_by_group(d, c, group)
        n_proxy = _event_exposure_proxy(d, c, d["toi_v2"])
        k = 80.0 if ("per_entry" in c.lower() or "per_exit_touch" in c.lower()) else 30.0
        d[f"v2_qc_{c}"] = _eb_shrink_by_group(d.assign(**{c: w}), c, n_proxy, k=k, group_cols=group)

    def _coalesce(a: str, b: str, c: str) -> pd.Series:
        sa = _to_num(d.get(a, pd.Series(np.nan, index=d.index)))
        sb = _to_num(d.get(b, pd.Series(np.nan, index=d.index)))
        sc = _to_num(d.get(c, pd.Series(np.nan, index=d.index)))
        return sa.where(sa.notna(), sb).where(lambda x: x.notna(), sc)

    d["v2_ptv_poss_def"] = _coalesce(
        "ptv_season_rapm_possession_defense",
        "ptv_roll3_rapm_possession_defense",
        "ptv_pooled_rapm_possession_defense",
    )
    d["v2_ptv_threat_sup"] = _coalesce(
        "ptv_season_rapm_threat_suppression",
        "ptv_roll3_rapm_threat_suppression",
        "ptv_pooled_rapm_threat_suppression",
    )

    d["v2_target_xga_inv"] = _z_by_group(d.assign(_x=-_to_num(d["yoy_xGA_per60"])), "_x", group)
    d["v2_target_xgs"] = _z_by_group(d.assign(_x=_to_num(d["yoy_onice_xg_share"])), "_x", group)
    d["v2_target_shot_supp"] = 0.7 * d["v2_target_xga_inv"] + 0.3 * d["v2_target_xgs"]

    for c in noisy_feats:
        d[f"v2_z_{c}"] = _z_by_group(d.assign(_x=_to_num(d[f"v2_qc_{c}"])), "_x", group)
    d["v2_z_ptv_poss_def"] = _z_by_group(d.assign(_x=_to_num(d["v2_ptv_poss_def"])), "_x", group)
    d["v2_z_ptv_threat_sup"] = _z_by_group(d.assign(_x=_to_num(d["v2_ptv_threat_sup"])), "_x", group)
    if "rapm_hybrid_xGA/60" in d.columns:
        d["v2_z_rapm_xga_inv"] = _z_by_group(d.assign(_x=-_to_num(d["rapm_hybrid_xGA/60"])), "_x", group)
    if "eh_rapm_CA/60" in d.columns:
        d["v2_z_ca_inv"] = _z_by_group(d.assign(_x=-_to_num(d["eh_rapm_CA/60"])), "_x", group)

    entry_feats = [
        f
        for f in [
            "v2_z_a3z_module_entry_defense",
            "v2_z_ptv_poss_def",
            "v2_z_ptv_threat_sup",
            "v2_z_a3z_failed_entries_per_entry",
        ]
        if f in d.columns
    ]
    exit_feats = [
        f
        for f in [
            "v2_z_a3z_module_breakout_exit_value",
            "v2_z_a3z_failed_exits_per_exit_touch",
            "v2_z_rapm_xga_inv",
            "v2_z_ca_inv",
            "v2_z_ptv_poss_def",
        ]
        if f in d.columns
    ]
    shot_feats = [
        f
        for f in [
            "v2_z_rapm_xga_inv",
            "v2_z_ca_inv",
            "v2_z_ptv_threat_sup",
            "v2_z_ptv_poss_def",
            "v2_z_eh_std_iBLK/60",
            "v2_z_eh_std_sh_iBLK/60",
        ]
        if f in d.columns
    ]
    if not entry_feats or not exit_feats or not shot_feats:
        info["enabled"] = False
        info["reason"] = "insufficient_features"
        return d, info

    alpha_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    latent_meta: dict[str, dict[str, float]] = {}
    for role in ["F", "D"]:
        p1, m1 = _walkforward_ridge_predict(d, feature_cols=entry_feats, target_col="v2_target_xga_inv", role=role, alpha_grid=alpha_grid)
        p2, m2 = _walkforward_ridge_predict(d, feature_cols=exit_feats, target_col="v2_target_xgs", role=role, alpha_grid=alpha_grid)
        p3, m3 = _walkforward_ridge_predict(d, feature_cols=shot_feats, target_col="v2_target_shot_supp", role=role, alpha_grid=alpha_grid)
        ridx = d.index[d["position_group"].eq(role)]
        d.loc[ridx, "v2_latent_entry_denial_raw"] = p1.loc[ridx]
        d.loc[ridx, "v2_latent_exit_resilience_raw"] = p2.loc[ridx]
        d.loc[ridx, "v2_latent_shot_supp_raw"] = p3.loc[ridx]
        latent_meta[f"{role}_entry"] = m1
        latent_meta[f"{role}_exit"] = m2
        latent_meta[f"{role}_shot"] = m3

    d["v2_latent_entry_denial_z"] = _z_by_group(d.assign(_x=_to_num(d["v2_latent_entry_denial_raw"])), "_x", group)
    d["v2_latent_exit_resilience_z"] = _z_by_group(d.assign(_x=_to_num(d["v2_latent_exit_resilience_raw"])), "_x", group)
    d["v2_latent_shot_supp_z"] = _z_by_group(d.assign(_x=_to_num(d["v2_latent_shot_supp_raw"])), "_x", group)

    w_entry = float(entry_weight)
    w_exit = float(exit_weight)
    w_sum = w_entry + w_exit
    if w_sum <= 0.0:
        w_entry, w_exit = 0.45, 0.55
        w_sum = 1.0
    w_entry /= w_sum
    w_exit /= w_sum
    sh_lat_w = float(np.clip(shot_v2_latent_weight, 0.0, 1.0))
    tr_blend = float(np.clip(transition_v2_blend, 0.0, 1.0))
    tr_blend_f = tr_blend if transition_v2_blend_f is None else float(np.clip(transition_v2_blend_f, 0.0, 1.0))
    tr_blend_d = tr_blend if transition_v2_blend_d is None else float(np.clip(transition_v2_blend_d, 0.0, 1.0))
    tr_blend_row = pd.Series(
        np.where(d["position_group"].astype(str).str.upper().eq("D"), tr_blend_d, tr_blend_f),
        index=d.index,
        dtype=float,
    )
    sh_blend = float(np.clip(shot_v2_blend, 0.0, 1.0))

    tr_base = _to_num(d["cat_transition_defense_z"])
    sh_base = _to_num(d["cat_shot_defense_z"])
    tr_lat = w_entry * _to_num(d["v2_latent_entry_denial_z"]) + w_exit * _to_num(d["v2_latent_exit_resilience_z"])
    sh_lat = _to_num(d["v2_latent_shot_supp_z"])
    # Row-level fallback: if latent prediction is missing, keep the existing category
    # signal instead of propagating NaN into tuned outputs.
    tr_lat = tr_lat.where(tr_lat.notna(), tr_base)
    sh_lat = sh_lat.where(sh_lat.notna(), sh_base)
    tr_v2 = tr_lat
    sh_v2 = sh_lat_w * sh_lat + (1.0 - sh_lat_w) * sh_base
    sh_v2 = sh_v2.where(sh_v2.notna(), sh_base)
    tr_v2 = _z_by_group(d.assign(_x=tr_v2), "_x", group)
    sh_v2 = _z_by_group(d.assign(_x=sh_v2), "_x", group)
    # Preserve the full transition-process estimate for category display. The
    # overall model uses a conservative blend selected on future-value folds,
    # while the public bar can use the more stable pure entry/retrieval trait.
    tr_display_full = tr_v2.copy()
    tr_v2 = tr_blend_row * tr_v2 + (1.0 - tr_blend_row) * tr_base
    sh_v2 = sh_blend * sh_v2 + (1.0 - sh_blend) * sh_base
    tr_v2 = tr_v2.where(tr_v2.notna(), tr_base)
    sh_v2 = sh_v2.where(sh_v2.notna(), sh_base)
    tr_v2 = _z_by_group(d.assign(_x=tr_v2), "_x", group)
    sh_v2 = _z_by_group(d.assign(_x=sh_v2), "_x", group)

    old_tr = tr_base
    old_sh = sh_base
    old_tr_adj = _to_num(d["cat_transition_defense_z__adj"])
    old_sh_adj = _to_num(d["cat_shot_defense_z__adj"])

    tr_ratio = np.where(np.isfinite(old_tr) & (np.abs(old_tr) > 1e-9), old_tr_adj / old_tr, np.nan)
    sh_ratio = np.where(np.isfinite(old_sh) & (np.abs(old_sh) > 1e-9), old_sh_adj / old_sh, np.nan)
    tr_ratio = pd.Series(tr_ratio, index=d.index, dtype=float).fillna(0.0)
    sh_ratio = pd.Series(sh_ratio, index=d.index, dtype=float).fillna(0.0)

    d["cat_transition_defense_z"] = tr_v2
    d["cat_shot_defense_z"] = sh_v2
    d["cat_transition_defense_z__adj"] = _to_num(d["cat_transition_defense_z"]) * tr_ratio
    d["cat_transition_defense_display_z"] = tr_display_full
    d["cat_transition_defense_display_z__adj"] = tr_display_full * tr_ratio
    d["cat_shot_defense_z__adj"] = _to_num(d["cat_shot_defense_z"]) * sh_ratio

    d = _anchor_d_defense_z_to_rapm(
        d,
        group_cols=group,
        sd_blend=0.62,
        td_blend=0.40,
    )

    info.update(
        {
            "applied": True,
            "latent_meta": latent_meta,
            "blend": {
                "transition_v2_blend": tr_blend,
                "transition_v2_blend_F": tr_blend_f,
                "transition_v2_blend_D": tr_blend_d,
                "shot_v2_blend": sh_blend,
                "shot_v2_latent_weight": sh_lat_w,
                "entry_weight": w_entry,
                "exit_weight": w_exit,
            },
        }
    )
    return d, info


def _apply_a3z_microstats_upgrade(
    df: pd.DataFrame,
    *,
    feature_source: Path,
    enabled: bool,
    blend: float,
    blend_f: float | None,
    blend_d: float | None,
    max_features: int,
    min_non_null_frac: float,
    min_abs_corr: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    info: dict[str, object] = {"enabled": bool(enabled), "applied": False}
    if not enabled:
        return df, info

    d = df.copy()

    # Pull in complete A3Z microstat pool from feature table when not already present.
    fs = _read_table(feature_source)
    fs = _dedupe_player_season(fs, label="tune_all_in_one_8cat_model a3z_microstats feature_source")
    a3z_cols = [
        c
        for c in fs.columns
        if str(c).startswith("a3z_")
        and c not in {"a3z_team"}
    ]
    missing = [c for c in a3z_cols if c not in d.columns]
    if missing:
        keep = ["player_id", "season_id"] + missing
        d = d.merge(fs[keep], on=["player_id", "season_id"], how="left")
    else:
        # keep variable for downstream candidate filtering
        a3z_cols = [c for c in d.columns if str(c).startswith("a3z_") and c != "a3z_team"]

    group = ["season_id", "position_group"]
    d["v3_t_xgf"] = _z_by_group(d.assign(_x=_to_num(d.get("yoy_xGF_per60", pd.Series(np.nan, index=d.index)))), "_x", group)
    d["v3_t_xga_inv"] = _z_by_group(d.assign(_x=-_to_num(d.get("yoy_xGA_per60", pd.Series(np.nan, index=d.index)))), "_x", group)
    d["v3_t_xgs"] = _z_by_group(d.assign(_x=_to_num(d.get("yoy_onice_xg_share", pd.Series(np.nan, index=d.index)))), "_x", group)
    d["v3_t_goal"] = _z_by_group(d.assign(_x=_to_num(d.get("yoy_onice_goal_share", pd.Series(np.nan, index=d.index)))), "_x", group)

    target_map: dict[str, pd.Series] = {
        "cat_transition_offense_z": 0.45 * d["v3_t_xgf"] + 0.55 * d["v3_t_xgs"],
        "cat_shot_creation_z": 0.60 * d["v3_t_xgf"] + 0.40 * d["v3_t_xgs"],
        "cat_playmaking_z": 0.35 * d["v3_t_xgf"] + 0.65 * d["v3_t_xgs"],
        "cat_forecheck_z": 0.25 * d["v3_t_xgf"] + 0.75 * d["v3_t_xgs"],
        "cat_physicality_z": 0.35 * d["v3_t_xga_inv"] + 0.35 * d["v3_t_xgs"] + 0.30 * d["v3_t_xgf"],
        "cat_skating_z": 0.40 * d["v3_t_xgf"] + 0.30 * d["v3_t_xgs"] + 0.30 * d["v3_t_xga_inv"],
        "cat_shot_defense_z": 0.75 * d["v3_t_xga_inv"] + 0.25 * d["v3_t_xgs"],
        "cat_transition_defense_z": 0.55 * d["v3_t_xga_inv"] + 0.45 * d["v3_t_xgs"],
        # Finishing is noisy by nature; include goal-share signal with lower leverage.
        "cat_finishing_z": 0.25 * d["v3_t_xgf"] + 0.25 * d["v3_t_xgs"] + 0.50 * d["v3_t_goal"],
    }

    # Candidate pool: numeric A3Z microstats with some variation and coverage.
    cand_pool: list[str] = []
    for c in a3z_cols:
        x = _to_num(d.get(c, pd.Series(np.nan, index=d.index)))
        if int(x.notna().sum()) < 150:
            continue
        if int(x.nunique(dropna=True)) < 8:
            continue
        cand_pool.append(c)
    if not cand_pool:
        info["reason"] = "no_candidate_a3z_features"
        return d, info

    alpha_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    blend_w = float(np.clip(blend, 0.0, 1.0))
    blend_by_role = {"F": blend_w, "D": blend_w}
    if blend_f is not None and np.isfinite(float(blend_f)):
        blend_by_role["F"] = float(np.clip(float(blend_f), 0.0, 1.0))
    if blend_d is not None and np.isfinite(float(blend_d)):
        blend_by_role["D"] = float(np.clip(float(blend_d), 0.0, 1.0))
    comp_meta: dict[str, dict[str, object]] = {}

    for comp_col, tgt in target_map.items():
        if comp_col not in d.columns:
            continue
        d[f"v3_target__{comp_col}"] = tgt
        latent_raw = pd.Series(np.nan, index=d.index, dtype=float)
        role_meta: dict[str, object] = {}

        for role in ["F", "D"]:
            ridx = d.index[d["position_group"].astype(str).str.upper().eq(role)]
            if len(ridx) < 200:
                continue
            sub = d.loc[ridx]
            y = _to_num(sub[f"v3_target__{comp_col}"])
            if int(y.notna().sum()) < 120:
                continue
            sub_sid = pd.to_numeric(sub["season_id"], errors="coerce").astype("Int64")
            sub_seasons = sorted([int(x) for x in sub_sid.dropna().unique()])
            if len(sub_seasons) < 4:
                continue
            sub_eval_seasons = sub_seasons[-2:]
            train_ns: list[int] = []
            for sid in sub_eval_seasons:
                tr_m = sub_sid < int(sid)
                n_tr = int(y[tr_m].notna().sum())
                if n_tr > 0:
                    train_ns.append(n_tr)
            if not train_ns:
                continue

            scored: list[tuple[str, float]] = []
            for c in cand_pool:
                x = _to_num(sub.get(c, pd.Series(np.nan, index=sub.index)))
                # Coverage gate within role.
                if float(x.notna().mean()) < float(min_non_null_frac):
                    continue
                r = _corr(x.to_numpy(float), y.to_numpy(float))
                if np.isfinite(r) and abs(float(r)) >= float(min_abs_corr):
                    scored.append((c, abs(float(r))))

            if not scored:
                continue
            scored = sorted(scored, key=lambda t: t[1], reverse=True)
            # Keep feature count within walk-forward sample limits (ridge fitter
            # requires at least ~8 rows per feature on the train fold).
            max_by_sample = max(4, int(min(train_ns) // 8))
            max_role_feats = max(4, min(int(max_features), int(max_by_sample)))
            feat_cols = [c for c, _ in scored[:max_role_feats]]
            p_role, meta = _walkforward_ridge_predict(
                d,
                feature_cols=feat_cols,
                target_col=f"v3_target__{comp_col}",
                role=role,
                alpha_grid=alpha_grid,
            )
            latent_raw.loc[ridx] = p_role.loc[ridx]
            role_meta[role] = {
                "n_features": int(len(feat_cols)),
                "features_top": feat_cols[:12],
                "alpha": float(meta.get("alpha", np.nan)),
                "cv_r": float(meta.get("cv_r", np.nan)),
            }

        if latent_raw.notna().sum() < 200:
            continue

        lat_z = _z_by_group(d.assign(_x=_to_num(latent_raw)), "_x", group)
        base = _to_num(d[comp_col])
        lat_z = lat_z.where(lat_z.notna(), base)
        pos = d["position_group"].astype(str).str.upper().str.strip()
        blend_s = pd.Series(blend_w, index=d.index, dtype=float)
        blend_s.loc[pos.eq("F")] = blend_by_role["F"]
        blend_s.loc[pos.eq("D")] = blend_by_role["D"]
        new_c = blend_s * lat_z + (1.0 - blend_s) * base
        new_c = _z_by_group(d.assign(_x=_to_num(new_c)), "_x", group)
        new_c = new_c.where(new_c.notna(), base)

        adj_col = f"{comp_col}__adj"
        if adj_col in d.columns:
            old_base = _to_num(d[comp_col])
            old_adj = _to_num(d[adj_col])
            ratio = np.where(np.isfinite(old_base) & (np.abs(old_base) > 1e-9), old_adj / old_base, np.nan)
            ratio_s = pd.Series(ratio, index=d.index, dtype=float).fillna(0.0)
            d[adj_col] = _to_num(new_c) * ratio_s

        d[comp_col] = new_c
        comp_meta[comp_col] = role_meta

    if not comp_meta:
        info["reason"] = "no_components_upgraded"
        return d, info

    info.update(
        {
            "applied": True,
            "blend": float(blend_w),
            "blend_by_role": {k: float(v) for k, v in blend_by_role.items()},
            "max_features": int(max_features),
            "min_non_null_frac": float(min_non_null_frac),
            "min_abs_corr": float(min_abs_corr),
            "candidate_pool_size": int(len(cand_pool)),
            "component_meta": comp_meta,
        }
    )
    return d, info


def _apply_transition_defense_consistency_guard(
    df: pd.DataFrame,
    *,
    feature_source: Path,
    enabled: bool,
    shot_def_threshold: float,
    suppression_pct_threshold: float,
    cap_value: float,
    excess_shrink: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Guardrail: avoid elite forward transition-defense scores when both
    shot-defense and external suppression anchors are weak.
    """
    info: dict[str, object] = {"enabled": bool(enabled), "applied": False}
    if not enabled:
        return df, info

    d = df.copy()
    tr_col = "cat_transition_defense_z__adj"
    sh_col = "cat_shot_defense_z__adj"
    if tr_col not in d.columns or sh_col not in d.columns:
        info["reason"] = "missing_transition_or_shot_defense_cols"
        return d, info

    need = [
        "ptv_roll3_rapm_threat_suppression",
        "ptv_pooled_rapm_threat_suppression",
        "ptv_roll3_rapm_possession_defense",
        "ptv_pooled_rapm_possession_defense",
        "eh_rapm_xGA/60",
    ]
    miss = [c for c in need if c not in d.columns]
    if miss:
        fs = _read_table(feature_source)
        fs = _dedupe_player_season(fs, label="transition_defense_guard feature_source")
        keep = ["player_id", "season_id"] + [c for c in miss if c in fs.columns]
        fs = fs[keep].copy()
        d = d.merge(fs, on=["player_id", "season_id"], how="left")

    tr = _to_num(d[tr_col])
    sh = _to_num(d[sh_col])
    sup = _to_num(d.get("ptv_roll3_rapm_threat_suppression", pd.Series(np.nan, index=d.index)))
    sup = sup.where(sup.notna(), _to_num(d.get("ptv_pooled_rapm_threat_suppression", pd.Series(np.nan, index=d.index))))
    sup = sup.where(sup.notna(), _to_num(d.get("ptv_roll3_rapm_possession_defense", pd.Series(np.nan, index=d.index))))
    sup = sup.where(sup.notna(), _to_num(d.get("ptv_pooled_rapm_possession_defense", pd.Series(np.nan, index=d.index))))
    xga = _to_num(d.get("eh_rapm_xGA/60", pd.Series(np.nan, index=d.index)))
    sup = sup.where(sup.notna(), -xga)

    tmp = d.copy()
    tmp["_sup_proxy"] = sup
    sup_pct = _pct_by_group(tmp, "_sup_proxy", ["season_id", "position_group"])

    pos = d["position_group"].astype(str).str.upper().str.strip()
    suppress_weak = sup_pct < float(suppression_pct_threshold)
    shot_weak = sh < float(shot_def_threshold)
    above_cap = tr > float(cap_value)
    mask = pos.eq("F") & above_cap & shot_weak & suppress_weak
    n_flag = int(mask.sum())

    if n_flag <= 0:
        info.update(
            {
                "applied": True,
                "n_rows_flagged": 0,
                "shot_def_threshold": float(shot_def_threshold),
                "suppression_pct_threshold": float(suppression_pct_threshold),
                "cap_value": float(cap_value),
                "excess_shrink": float(excess_shrink),
            }
        )
        return d, info

    shr = float(np.clip(excess_shrink, 0.0, 1.0))
    tr_new = tr.copy()
    excess = (tr - float(cap_value)).clip(lower=0.0)
    tr_new.loc[mask] = float(cap_value) + (1.0 - shr) * excess.loc[mask]
    d[tr_col] = tr_new

    info.update(
        {
            "applied": True,
            "n_rows_flagged": int(n_flag),
            "n_rows_total": int(len(d)),
            "shot_def_threshold": float(shot_def_threshold),
            "suppression_pct_threshold": float(suppression_pct_threshold),
            "cap_value": float(cap_value),
            "excess_shrink": float(excess_shrink),
            "mean_transition_def_before_flagged": float(np.nanmean(tr.loc[mask].to_numpy(float))),
            "mean_transition_def_after_flagged": float(np.nanmean(tr_new.loc[mask].to_numpy(float))),
        }
    )
    return d, info


def _draw_weights(
    rng: np.random.Generator,
    role: str,
    cfg: TuneConfig,
    *,
    constraints: str = "philosophy",
) -> np.ndarray:
    # Order matches CAT_COLS
    idx = {c: i for i, c in enumerate(CAT_COLS)}
    pp_i = idx["cat_power_play_z"]
    pk_i = idx["cat_penalty_kill_z"]
    pen_i = idx.get("cat_penalty_value_z")
    fore_i = idx["cat_forecheck_z"]
    phys_i = idx["cat_physicality_z"]
    shot_create_i = idx["cat_shot_creation_z"]
    trans_off_i = idx["cat_transition_offense_z"]
    shot_def_i = idx["cat_shot_defense_z"]
    trans_def_i = idx["cat_transition_defense_z"]
    skate_i = idx.get("cat_skating_z")
    core_i = [
        shot_create_i,
        trans_off_i,
        shot_def_i,
        trans_def_i,
        idx["cat_playmaking_z"],
    ]
    if skate_i is not None:
        core_i.append(skate_i)
    mode = str(constraints).lower().strip()
    if mode == "relaxed":
        # Flatter Dirichlet so philosophy priors (F shot-creation 5.2, D shot-def 5.0)
        # stop pinning the search to the old mix.
        alpha = np.full(len(CAT_COLS), 1.8, dtype=float)
    elif mode == "bounded":
        if role == "F":
            alpha_by_col = {
                "cat_transition_offense_z": 2.6,
                "cat_shot_defense_z": 2.4,
                "cat_transition_defense_z": 2.0,
                "cat_shot_creation_z": 3.2,
                "cat_finishing_z": 1.8,
                "cat_playmaking_z": 2.4,
                "cat_power_play_z": 1.0,
                "cat_penalty_kill_z": 1.0,
                "cat_forecheck_z": 1.6,
                "cat_physicality_z": 1.0,
                "cat_skating_z": 1.8,
            }
        else:
            alpha_by_col = {
                "cat_transition_offense_z": 2.6,
                "cat_shot_defense_z": 3.2,
                "cat_transition_defense_z": 2.4,
                "cat_shot_creation_z": 2.8,
                "cat_finishing_z": 1.4,
                "cat_playmaking_z": 2.0,
                "cat_power_play_z": 1.0,
                "cat_penalty_kill_z": 1.0,
                "cat_forecheck_z": 1.4,
                "cat_physicality_z": 1.0,
                "cat_skating_z": 1.8,
            }
        alpha = np.array([float(alpha_by_col.get(c, 1.6)) for c in CAT_COLS], dtype=float)
    elif role == "F":
        alpha_by_col = {
            "cat_transition_offense_z": 2.6,
            "cat_shot_defense_z": 2.6,
            "cat_transition_defense_z": 1.9,
            "cat_shot_creation_z": 5.2,
            "cat_finishing_z": 2.1,
            "cat_playmaking_z": 3.6,
            "cat_power_play_z": 1.1,
            "cat_penalty_kill_z": 1.0,
            "cat_forecheck_z": 2.0,
            "cat_physicality_z": 1.2,
            "cat_skating_z": 2.1,
        }
        alpha = np.array([float(alpha_by_col.get(c, 1.5)) for c in CAT_COLS], dtype=float)
    else:
        alpha_by_col = {
            "cat_transition_offense_z": 2.8,
            "cat_shot_defense_z": 5.0,
            "cat_transition_defense_z": 3.6,
            "cat_shot_creation_z": 2.8,
            "cat_finishing_z": 1.2,
            "cat_playmaking_z": 2.2,
            "cat_power_play_z": 1.0,
            "cat_penalty_kill_z": 1.0,
            "cat_forecheck_z": 1.6,
            "cat_physicality_z": 1.8,
            "cat_skating_z": 2.2,
        }
        alpha = np.array([float(alpha_by_col.get(c, 1.5)) for c in CAT_COLS], dtype=float)
    w = rng.dirichlet(alpha)

    # Physicality is a descriptive/card trait, not a direct additive value
    # category.  Sampling it from a continuous Dirichlet and then requiring
    # ``w[phys_i] <= .001`` made the feasible region vanishingly small and
    # could leave the optimizer with zero valid candidates (and a NaN
    # objective).  Encode the model philosophy directly instead: give the
    # trait a structural zero and renormalize the value-bearing categories.
    w[phys_i] = 0.0
    w[pp_i] = 0.0
    w[pk_i] = 0.0
    if pen_i is not None:
        w[pen_i] = 0.0
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return None
    w = w / w_sum

    if mode == "relaxed":
        # Keep skating/finishing from monopolizing overlapping 3yr R²; everything
        # else is free on the 5v5 simplex.
        if skate_i is not None and w[skate_i] > 0.22:
            return None
        if w[idx["cat_finishing_z"]] > 0.18:
            return None
        if np.sum(w[core_i]) < 0.55:
            return None
        return w

    if mode == "bounded":
        # Drop SC-must-lead and tight SD caps, but keep a hockey rating:
        # defense floor, no single-category monopoly, modest style caps.
        sc = w[shot_create_i]
        to = w[trans_off_i]
        sd = w[shot_def_i]
        td = w[trans_def_i]
        if float(np.max(w)) > 0.38:
            return None
        if skate_i is not None and w[skate_i] > 0.16:
            return None
        if w[idx["cat_finishing_z"]] > 0.12:
            return None
        if w[fore_i] > 0.12:
            return None
        if np.sum(w[core_i]) < 0.62:
            return None
        if role == "F":
            if (sd + td) < 0.22:
                return None
            if (sc + to) < 0.28:
                return None
            if w[idx["cat_playmaking_z"]] < 0.03:
                return None
        else:
            if (sd + td) < 0.26:
                return None
            if (sc + sd + to) < 0.55:
                return None
            if w[idx["cat_playmaking_z"]] < 0.03:
                return None
        return w

    # Constraints:
    # 1) offense+defense core must be substantial.
    if (w[shot_create_i] + w[shot_def_i]) < float(cfg.min_5v5_share):
        return None
    # 1a) possession + defense must remain a core share (philosophy constraint).
    if role == "F":
        if (w[shot_def_i] + w[trans_def_i]) < 0.28:
            return None
        if (w[shot_create_i] + w[trans_off_i]) < 0.28:
            return None
        # Keep forward blend from becoming defense-tilted.
        if w[shot_def_i] > 0.22:
            return None
    else:
        # Relaxed: allow more weight to transition offense (r=0.55 vs shot_def r=0.29)
        if (w[shot_def_i] + w[trans_def_i]) < 0.32:
            return None
        # Keep defense blend from collapsing into one noisy axis.
        if w[shot_def_i] > 0.32:
            return None
    # 1b) noisy categories capped in final blend
    noisy_sum = w[pp_i] + w[pk_i] + (w[pen_i] if pen_i is not None else 0.0)
    if noisy_sum > 0.16:
        return None
    # 1c) style/trait sub-metrics should support, not dominate, overall rating.
    # Physicality is display-only, not included in overall (JFresh/HockeyViz don't use it).
    if w[phys_i] > 0.001:
        return None
    if (w[fore_i] + w[phys_i]) > 0.13:
        return None
    # Skating should be meaningful but not a primary overall driver.
    if role == "F":
        if w[skate_i] > 0.13:
            return None
    else:
        if w[skate_i] > 0.12:
            return None
    # Prevent core creator categories from being crowded out by defense-heavy blends.
    if role == "F":
        if w[idx["cat_playmaking_z"]] < 0.05:
            return None
        if w[idx["cat_finishing_z"]] > 0.10:
            return None
    else:
        if w[idx["cat_playmaking_z"]] < 0.04:
            return None
        if w[idx["cat_finishing_z"]] > 0.08:
            return None
    if np.sum(w[core_i]) < 0.70:
        return None
    # 2) role-primary categories must lead.
    if role == "F":
        if w[shot_create_i] < w[shot_def_i]:
            return None
        if w[shot_create_i] < np.max(np.array([w[i] for i in range(len(w)) if i != shot_create_i], dtype=float)):
            return None
    else:
        # Relaxed: allow shot_creation or transition_offense to lead for D
        # since they have higher outcome correlations (r=0.55) than shot_def (r=0.29)
        core_offense_def = w[shot_create_i] + w[shot_def_i] + w[trans_off_i]
        if core_offense_def < 0.55:
            return None
        # At least one of shot_def, shot_creation, or trans_off must lead
        lead_candidates = [w[shot_def_i], w[shot_create_i], w[trans_off_i]]
        others = np.array([w[i] for i in range(len(w)) if i not in [shot_def_i, shot_create_i, trans_off_i]], dtype=float)
        if max(lead_candidates) < np.max(others):
            return None
    return w


def _tune_role(
    d_role: pd.DataFrame,
    role: str,
    cfg: TuneConfig,
    *,
    objective_target: pd.Series | None = None,
    objective_score: str = "corr",
    eval_seasons: list[int] | None = None,
    prior_weights: np.ndarray | None = None,
    stability_penalty: float = 0.0,
    card_yoy: bool = False,
    weight_constraints: str = "philosophy",
    card_yoy_z_source: str = "adj",
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(cfg.seed) + (17 if role == "D" else 0))
    # Cards blend reliability-shrunk category z (__adj) with slot weights.
    # The first card_yoy search used raw z and understated what published HERO sees.
    use_raw = bool(card_yoy) and str(card_yoy_z_source).lower().strip() == "raw"
    adj_cols = list(CAT_COLS) if use_raw else [f"{c}__adj" for c in CAT_COLS]
    card_windows = None
    if card_yoy:
        pid = pd.to_numeric(d_role.get("player_id"), errors="coerce").to_numpy()
        sid = pd.to_numeric(d_role.get("season_id"), errors="coerce").to_numpy()
        toi, gp = _role_toi_gp(d_role)
        card_windows = _build_card_yoy_windows(pid, sid, toi, gp)

    best_w = None
    best_obj = -np.inf

    for _ in range(int(cfg.n_samples)):
        w = _draw_weights(rng, role, cfg, constraints=weight_constraints)
        if w is None:
            continue
        obj = _objective_for_role(
            d_role,
            w,
            adj_cols,
            objective_target=objective_target,
            objective_score=objective_score,
            eval_seasons=eval_seasons,
            card_windows=card_windows,
        )
        penalized_obj = float(obj)
        if prior_weights is not None and float(stability_penalty) > 0.0:
            # A quadratic stability cost prevents a noisy fold from causing a
            # wholesale philosophical rewrite of the rating. The unpenalized
            # walk-forward objective still has to earn every move away from the
            # accepted production model.
            penalized_obj -= float(stability_penalty) * float(np.sum((w - prior_weights) ** 2))
        if np.isfinite(penalized_obj) and penalized_obj > best_obj:
            best_obj = float(penalized_obj)
            best_w = w.copy()

    if best_w is None:
        # A zero-search/failure rebuild must honor explicitly supplied weights.
        # Defaults apply only when the caller supplied no incumbent prior.
        if prior_weights is not None:
            frozen = np.asarray(prior_weights, dtype=float)
            if (frozen.shape != (len(CAT_COLS),) or not np.isfinite(frozen).all()
                    or (frozen < 0.0).any() or frozen.sum() <= 0.0):
                raise ValueError("Invalid supplied category weights for fixed-weight fallback")
            return frozen / frozen.sum(), float("nan")
        # fallback deterministic defaults
        if role == "F":
            default = {
                "cat_transition_offense_z": 0.08,
                "cat_shot_defense_z": 0.12,
                "cat_transition_defense_z": 0.06,
                "cat_shot_creation_z": 0.30,
                "cat_finishing_z": 0.07,
                "cat_playmaking_z": 0.09,
                "cat_power_play_z": 0.03,
                "cat_penalty_kill_z": 0.02,
                "cat_forecheck_z": 0.06,
                "cat_physicality_z": 0.0,  # Display only, not in overall
                "cat_skating_z": 0.12,
            }
        else:
            default = {
                "cat_transition_offense_z": 0.09,
                "cat_shot_defense_z": 0.22,
                "cat_transition_defense_z": 0.13,
                "cat_shot_creation_z": 0.16,
                "cat_finishing_z": 0.03,
                "cat_playmaking_z": 0.06,
                "cat_power_play_z": 0.025,
                "cat_penalty_kill_z": 0.015,
                "cat_forecheck_z": 0.06,
                "cat_physicality_z": 0.0,  # Display only, not in overall
                "cat_skating_z": 0.10,
            }
        best_w = np.array([float(default.get(c, 0.0)) for c in CAT_COLS], dtype=float)
        s = float(best_w.sum())
        if s > 0:
            best_w = best_w / s
        best_obj = float("nan")

    return best_w, best_obj


def _apply_role_weights(
    df: pd.DataFrame,
    w_f: np.ndarray,
    w_d: np.ndarray,
    *,
    use_adjusted: bool = True,
) -> pd.Series:
    """Apply position-specific weights to category z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with category z-score columns.
    w_f : np.ndarray
        Forward weights (length = len(CAT_COLS)).
    w_d : np.ndarray
        Defense weights (length = len(CAT_COLS)).
    use_adjusted : bool, default True
        If True, use ``*__adj`` (reliability-shrunk) columns.
        If False, use raw z-score columns (no TOI shrinkage).
        Set to False for season-level percentiles to reduce disagreement
        with raw/process targets.

    Returns
    -------
    pd.Series
        Weighted composite score per player-season.
    """
    if use_adjusted:
        cols = [f"{c}__adj" for c in CAT_COLS]
    else:
        # Use raw z-score columns (no reliability shrinkage)
        cols = list(CAT_COLS)
    mat = df[cols].to_numpy(dtype=float)
    pos = df["position_group"].astype(str).str.upper().str.strip().to_numpy()

    out = np.full(len(df), np.nan, dtype=float)
    m_f = pos == "F"
    m_d = pos == "D"

    if m_f.any():
        out[m_f] = _weighted_row_score(mat[m_f], w_f)
    if m_d.any():
        out[m_d] = _weighted_row_score(mat[m_d], w_d)

    # fallback for unknown positions: average of F/D weights
    m_u = ~(m_f | m_d)
    if m_u.any():
        out[m_u] = _weighted_row_score(mat[m_u], (w_f + w_d) / 2.0)

    return pd.Series(out, index=df.index, dtype=float)


def _drop_category_weight_and_renorm(w: np.ndarray, cat_name: str) -> np.ndarray:
    out = np.asarray(w, dtype=float).copy()
    if cat_name not in CAT_COLS:
        return out
    i = CAT_COLS.index(cat_name)
    out[i] = 0.0
    s = float(np.sum(np.clip(out, 0.0, None)))
    if s > 0:
        out = np.clip(out, 0.0, None) / s
    return out


def _json_float_or_none(v: float) -> float | None:
    x = float(v)
    return x if np.isfinite(x) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores.parquet")
    ap.add_argument(
        "--feature-source",
        type=Path,
        default=ROOT / "data/pcv/player_season_feature_table_all_in_one.parquet",
        help="Feature table containing yoy_* outcome anchors for objective tuning.",
    )
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores_tuned.csv")
    ap.add_argument("--out-parquet", type=Path, default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores_tuned.parquet")
    ap.add_argument("--out-weights-json", type=Path, default=ROOT / "data/pcv/all_in_one_8cat_tuned_weights.json")
    ap.add_argument("--min-5v5-share", type=float, default=0.45)
    # cap_noisy 1.75 -> 2.0: random-search 2026-05 showed top F-obj configs
    # consistently sit at cap_noisy >= 2.0; 2.5 is the max we tested.
    ap.add_argument("--cap-noisy", type=float, default=2.0)
    ap.add_argument("--k-general", type=float, default=400.0, help="TOI shrinkage k for core categories (minutes; ignored when --auto-k-by-category is set).")
    ap.add_argument("--k-noisy", type=float, default=650.0, help="TOI shrinkage k for noisy categories (minutes; ignored when --auto-k-by-category is set).")
    # auto-k-by-category 2026-05: enabling this lifted D objective from 0.665
    # to 0.694 (+4.2%) and D YoY stability from 0.663 to 0.690. Single biggest
    # tuner improvement we found. Use --no-auto-k-by-category to revert.
    ap.add_argument(
        "--auto-k-by-category",
        dest="auto_k_by_category",
        action="store_true",
        default=True,
        help="Learn category-specific TOI shrinkage k per role from train seasons. Default: enabled.",
    )
    ap.add_argument(
        "--no-auto-k-by-category",
        dest="auto_k_by_category",
        action="store_false",
        help="Disable auto-learned category shrinkage; fall back to global k_general/k_noisy.",
    )
    ap.add_argument(
        "--k-candidates",
        type=str,
        default="150,250,400,550,700,900,1200",
        help="Comma-separated k values (minutes) used for category-specific shrinkage search.",
    )
    ap.add_argument("--n-samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--weight-prior-json",
        type=Path,
        default=None,
        help="Optional accepted weights JSON used as a stability prior.",
    )
    ap.add_argument(
        "--weight-stability-penalty",
        type=float,
        default=0.0,
        help="Quadratic penalty on movement away from --weight-prior-json.",
    )
    ap.add_argument("--train-seasons", type=str, default="20232024,20242025")
    # Forward defense penalty was disabled 2026-05 after the component
    # reweight: a parameter sweep across penalty_strength in {0.0..0.55} and
    # floor in {12..30} showed the F walk-forward objective monotonically
    # improves as the penalty is softened (0.6916 -> 0.6956 at strength=0).
    # The nonlinear penalty was masking shot-defense team-context issues by
    # over-punishing forwards on weak defensive teams rather than capturing
    # predictive signal. Leaving the args in place so callers can re-enable
    # an explicit penalty if desired.
    ap.add_argument("--fwd-defense-floor-pct", type=float, default=0.0)
    ap.add_argument("--fwd-defense-penalty-strength", type=float, default=0.0)
    ap.add_argument("--fwd-defense-penalty-power", type=float, default=1.4)
    # Spike regression also softened: large strength was over-regressing
    # bona-fide rookie/sophomore offensive seasons (e.g. Bedard, Bedard).
    # 0.15 keeps a mild guard for true outliers without flattening real talent.
    ap.add_argument("--fwd-spike-reg-strength", type=float, default=0.15)
    ap.add_argument("--fwd-spike-reg-toi-k", type=float, default=900.0)
    ap.add_argument(
        "--deoverlap-possession",
        dest="deoverlap_possession",
        action="store_true",
        default=False,
        help="Residualize possession-process category vs overlapping categories (legacy; category removed).",
    )
    ap.add_argument(
        "--no-deoverlap-possession",
        dest="deoverlap_possession",
        action="store_false",
        help="Disable possession de-overlap.",
    )
    ap.add_argument(
        "--deoverlap-alpha",
        type=float,
        default=2.0,
        help="Ridge alpha used for possession de-overlap residualization.",
    )
    ap.add_argument("--walkforward-start-season", type=int, default=20162017)
    ap.add_argument("--walkforward-end-season", type=int, default=20242025)
    ap.add_argument("--walkforward-min-train-seasons", type=int, default=3)
    ap.add_argument(
        "--orthogonalize-defense",
        dest="orthogonalize_defense",
        action="store_true",
        default=True,
        help="Reduce collinearity by residualizing forward Transition Defense vs Shot Defense.",
    )
    ap.add_argument(
        "--no-orthogonalize-defense",
        dest="orthogonalize_defense",
        action="store_false",
        help="Disable defense collinearity control.",
    )
    ap.add_argument(
        "--orthogonalize-defense-alpha",
        type=float,
        default=2.0,
        help="Ridge alpha for defense collinearity residualization.",
    )
    ap.add_argument(
        "--orthogonalize-defense-residual-weight",
        type=float,
        default=0.45,
        help="Blend weight for residualized transition-defense signal (0=off, 1=full residualization).",
    )
    ap.add_argument(
        "--orthogonalize-defense-min-abs-corr",
        type=float,
        default=0.30,
        help=(
            "Only residualize season-groups where |corr(transition_def, shot_def)| "
            "meets this threshold. Lowered from 0.45 -> 0.30 (random search 2026-05): "
            "the lower threshold consistently appeared in the top configs and "
            "added +0.003 to F objective without hurting D."
        ),
    )
    ap.add_argument(
        "--objective-score",
        type=str,
        default="corr",
        choices=["corr", "logloss"],
        help="Objective metric for tuning: corr (maximize) or logloss (minimize via maximizing -logloss).",
    )
    ap.add_argument(
        "--objective-mode",
        type=str,
        default="anchor",
        choices=["anchor", "residual"],
        help="anchor: original target blend; residual: optimize incremental signal over baseline score.",
    )
    ap.add_argument(
        "--objective-anchor-type",
        type=str,
        default="card_yoy",
        choices=["yoy", "same_season_impact", "card_yoy"],
        help="card_yoy: recency-weighted 3yr Overall persistence (production); yoy: next-season on-ice anchors; same_season_impact: EH RAPM + PTV current-season impact.",
    )
    ap.add_argument(
        "--weight-constraints",
        type=str,
        default=None,
        choices=["philosophy", "relaxed", "bounded"],
        help="philosophy: SC-must-lead / SD caps (default). bounded: drop those but keep defense floor and 38% cap. relaxed: no hockey floors (overfits card stickiness).",
    )
    ap.add_argument(
        "--card-yoy-z-source",
        type=str,
        default="adj",
        choices=["adj", "raw"],
        help="card_yoy score source. adj matches published HERO (__adj); raw is the first card_yoy search.",
    )
    ap.add_argument(
        "--residual-baseline-col",
        type=str,
        default="overall_8cat_z",
        help="Baseline score column used when --objective-mode residual.",
    )
    ap.add_argument(
        "--residual-alpha",
        type=float,
        default=2.0,
        help="Ridge alpha used to residualize target vs baseline in residual objective mode.",
    )
    ap.add_argument("--f-obj-xgf-w", type=float, default=0.35, help="Forward objective weight on next xGF/60.")
    ap.add_argument("--f-obj-xga-w", type=float, default=0.35, help="Forward objective weight on -next xGA/60.")
    ap.add_argument("--f-obj-xgs-w", type=float, default=0.20, help="Forward objective weight on next xG share.")
    ap.add_argument("--f-obj-gs-w", type=float, default=0.10, help="Forward objective weight on next goal share.")
    ap.add_argument("--d-obj-xgf-w", type=float, default=0.35, help="Defense objective weight on next xGF/60.")
    ap.add_argument("--d-obj-xga-w", type=float, default=0.35, help="Defense objective weight on -next xGA/60.")
    ap.add_argument("--d-obj-xgs-w", type=float, default=0.20, help="Defense objective weight on next xG share.")
    ap.add_argument("--d-obj-gs-w", type=float, default=0.10, help="Defense objective weight on next goal share.")
    ap.add_argument(
        "--objective-fallback-next-overall",
        dest="objective_fallback_next_overall",
        action="store_true",
        default=True,
        help="Fallback objective target to next-year overall_8cat_z where yoy anchors are missing.",
    )
    ap.add_argument(
        "--no-objective-fallback-next-overall",
        dest="objective_fallback_next_overall",
        action="store_false",
        help="Disable next-year overall fallback objective target.",
    )
    ap.add_argument(
        "--objective-anchor-next-overall-blend",
        type=float,
        default=0.75,
        help="Blend when both anchor and fallback targets exist (1.0=anchor only, 0.0=fallback only).",
    )
    ap.add_argument(
        "--objective-fallback-source-col",
        type=str,
        default="overall_8cat_z",
        help="Source column used to derive next-year fallback target.",
    )
    ap.add_argument(
        "--card-yoy-holdout-season",
        type=int,
        default=CARD_YOY_HOLDOUT_LEFT_SEASON_DEFAULT,
        help="Left season_id held out of card_yoy weight search (default 20232024, the last pair in the published 0.653 benchmark).",
    )
    ap.add_argument(
        "--drop-special-teams",
        dest="drop_special_teams",
        action="store_true",
        default=True,
        help="Exclude Power Play and Penalty Kill from overall rating (default; renormalize 5v5 categories).",
    )
    ap.add_argument(
        "--keep-special-teams",
        dest="drop_special_teams",
        action="store_false",
        help="Include Power Play and Penalty Kill in overall rating (omit per player when unit z is missing).",
    )
    ap.add_argument(
        "--defense-signal-upgrade",
        dest="defense_signal_upgrade",
        action="store_true",
        default=True,
        help="Apply v2 latent defense-signal upgrade before weight tuning.",
    )
    ap.add_argument(
        "--no-defense-signal-upgrade",
        dest="defense_signal_upgrade",
        action="store_false",
        help="Disable v2 latent defense-signal upgrade.",
    )
    ap.add_argument("--defense-upgrade-transition-blend", type=float, default=0.35)
    ap.add_argument("--defense-upgrade-transition-blend-f", type=float, default=None)
    ap.add_argument("--defense-upgrade-transition-blend-d", type=float, default=None)
    ap.add_argument("--defense-upgrade-shot-blend", type=float, default=0.30)
    ap.add_argument("--defense-upgrade-shot-latent-weight", type=float, default=0.70)
    ap.add_argument("--defense-upgrade-entry-weight", type=float, default=0.45)
    ap.add_argument("--defense-upgrade-exit-weight", type=float, default=0.55)
    ap.add_argument(
        "--a3z-microstats-upgrade",
        dest="a3z_microstats_upgrade",
        action="store_true",
        default=False,
        help="Augment category signals using walk-forward ridge latents from full A3Z microstat pool.",
    )
    ap.add_argument(
        "--no-a3z-microstats-upgrade",
        dest="a3z_microstats_upgrade",
        action="store_false",
        help="Disable A3Z full-pool microstats augmentation.",
    )
    ap.add_argument("--a3z-microstats-blend", type=float, default=0.20)
    ap.add_argument(
        "--a3z-microstats-blend-f",
        type=float,
        default=None,
        help="Optional F-only blend override (defaults to --a3z-microstats-blend).",
    )
    ap.add_argument(
        "--a3z-microstats-blend-d",
        type=float,
        default=None,
        help="Optional D-only blend override (defaults to --a3z-microstats-blend).",
    )
    ap.add_argument("--a3z-microstats-max-features", type=int, default=40)
    ap.add_argument("--a3z-microstats-min-nonnull-frac", type=float, default=0.15)
    ap.add_argument("--a3z-microstats-min-abs-corr", type=float, default=0.04)
    ap.add_argument(
        "--component-reliability-refinement",
        dest="component_reliability_refinement",
        action="store_true",
        default=True,
        help="Apply event-exposure EB stabilization to weak/noisy components before final weight tuning.",
    )
    ap.add_argument(
        "--no-component-reliability-refinement",
        dest="component_reliability_refinement",
        action="store_false",
        help="Disable event-exposure component reliability refinement.",
    )
    ap.add_argument(
        "--trdef-consistency-guard",
        dest="trdef_consistency_guard",
        action="store_true",
        default=True,
        help="Cap elite forward transition-defense scores when corroborating suppression anchors are weak.",
    )
    ap.add_argument(
        "--no-trdef-consistency-guard",
        dest="trdef_consistency_guard",
        action="store_false",
        help="Disable transition-defense consistency guard.",
    )
    ap.add_argument("--trdef-guard-shotdef-threshold", type=float, default=0.0)
    ap.add_argument("--trdef-guard-suppression-pct-threshold", type=float, default=50.0)
    ap.add_argument("--trdef-guard-cap", type=float, default=1.0)
    ap.add_argument("--trdef-guard-excess-shrink", type=float, default=0.65)
    args = ap.parse_args()
    if str(args.objective_anchor_type).lower() == "same_season_impact":
        args.objective_fallback_next_overall = False
        args.objective_anchor_next_overall_blend = 1.0
        if float(args.f_obj_xgs_w) == 0.20 and float(args.d_obj_xgs_w) == 0.20:
            args.f_obj_xgs_w = 0.45
            args.d_obj_xgs_w = 0.45
            args.f_obj_xgf_w = 0.20
            args.d_obj_xgf_w = 0.20
            args.f_obj_xga_w = 0.25
            args.d_obj_xga_w = 0.25

    cfg = TuneConfig(
        min_5v5_share=float(args.min_5v5_share),
        cap_noisy=float(args.cap_noisy),
        k_general=float(args.k_general),
        k_noisy=float(args.k_noisy),
        n_samples=int(args.n_samples),
        seed=int(args.seed),
    )

    df = _read_table(args.input)
    df = _dedupe_player_season(df, label="tune_all_in_one_8cat_model input")
    for c in ["player_id", "player_name", "season_id", "position_group", "a3z_toi_5v5_min"] + CAT_COLS:
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")

    # Bring in objective anchors/reliability fields if missing from score table.
    need_outcome_cols = ["yoy_xGF_per60", "yoy_xGA_per60", "yoy_onice_xg_share", "yoy_onice_goal_share"]
    if str(args.objective_anchor_type).lower() == "same_season_impact":
        need_outcome_cols = []
    optional_cols = list(dict.fromkeys(["eh_rapm_TOI", "a3z_toi_5v5_min", "eh_std_GP", "eh_rapm_GP",
                                      *SAME_SEASON_IMPACT_FEATURE_COLS, *EXPOSURE_SOURCE_COLUMNS]))
    d = df.copy()
    missing_cols = [c for c in (need_outcome_cols + optional_cols) if c not in d.columns]
    if missing_cols:
        fs = _read_table(args.feature_source)
        fs = _dedupe_player_season(fs, label="tune_all_in_one_8cat_model feature_source")
        preserve_tracking(fs)
        # Merge only truly missing columns; avoids creating *_x/*_y duplicates that
        # can silently drop the plain eh_rapm_TOI field used for reliability.
        keep = ["player_id", "season_id"] + [c for c in missing_cols if c in fs.columns]
        fs = fs[keep].copy()
        fs["player_id"] = pd.to_numeric(fs["player_id"], errors="coerce").astype("Int64")
        fs["season_id"] = pd.to_numeric(fs["season_id"], errors="coerce").astype("Int64")
        fs = fs[fs["player_id"].notna() & fs["season_id"].notna()].copy()
        fs["player_id"] = fs["player_id"].astype(int)
        fs["season_id"] = fs["season_id"].astype(int)
        fs = fs.drop_duplicates(["player_id", "season_id"], keep="last")
        d = d.merge(fs, on=["player_id", "season_id"], how="left")
    # Coalesce reliability TOI if a prior run created suffix columns.
    if "eh_rapm_TOI" not in d.columns:
        toi_x = _to_num(d.get("eh_rapm_TOI_x", pd.Series(np.nan, index=d.index)))
        toi_y = _to_num(d.get("eh_rapm_TOI_y", pd.Series(np.nan, index=d.index)))
        d["eh_rapm_TOI"] = toi_x.where(toi_x.notna(), toi_y)
    preserve_tracking(d)

    train_seasons = {int(x.strip()) for x in str(args.train_seasons).split(",") if x.strip()}
    season_num = pd.to_numeric(d.get("season_id", pd.Series(np.nan, index=d.index)), errors="coerce")
    wf_start = int(args.walkforward_start_season)
    wf_end = int(args.walkforward_end_season)
    use_walkforward = wf_start > 0 and wf_end > 0 and wf_end >= wf_start
    if use_walkforward:
        train = d[(season_num >= wf_start) & (season_num <= wf_end)].copy()
    else:
        train = d[d["season_id"].isin(train_seasons)].copy()
    train_toi_eh = _to_num(train.get("eh_rapm_TOI", pd.Series(np.nan, index=train.index)))
    train_toi_a3z = _to_num(train.get("a3z_toi_5v5_min", pd.Series(np.nan, index=train.index)))
    train_toi = train_toi_eh.where(train_toi_eh.notna(), train_toi_a3z).fillna(0.0)
    train = train[train_toi >= 120.0].copy()
    eval_seasons = (
        _resolve_walkforward_eval_seasons(
            train,
            start_season=wf_start,
            end_season=wf_end,
            min_train_seasons=int(args.walkforward_min_train_seasons),
        )
        if use_walkforward
        else None
    )
    card_yoy = str(args.objective_anchor_type).lower() == "card_yoy"
    weight_constraints = str(args.weight_constraints or "philosophy").lower()
    card_yoy_z_source = str(args.card_yoy_z_source).lower().strip()
    if card_yoy and eval_seasons:
        holdout_left = int(args.card_yoy_holdout_season)
        eval_seasons = [int(s) for s in eval_seasons if int(s) < holdout_left]
        print(
            f"card_yoy selection left-seasons={eval_seasons} "
            f"holdout_left>={holdout_left} "
            f"constraints={weight_constraints} z_source={card_yoy_z_source}",
            flush=True,
        )

    k_map_by_role = None
    k_fit_report = {}
    k_floor_applied: dict[str, dict[str, dict[str, float]]] = {"F": {}, "D": {}}
    if card_yoy:
        prior_path = args.weight_prior_json or (ROOT / "data/pcv/all_in_one_8cat_tuned_weights.json")
        prior_payload = json.loads(Path(prior_path).read_text(encoding="utf-8"))
        k_prior = prior_payload.get("k_by_role_category")
        if isinstance(k_prior, dict):
            k_map_by_role = {
                role: {c: float(k_prior.get(role, {}).get(c, cfg.k_general)) for c in CAT_COLS}
                for role in ("F", "D")
            }
            k_fit_report = {
                "frozen_from": str(prior_path),
                "reason": "card_yoy freezes production category k; only overall weights are searched",
            }
            print(f"card_yoy: froze category k from {prior_path}", flush=True)
    if (not card_yoy) and args.auto_k_by_category:
        k_grid = _parse_k_candidates(args.k_candidates)
        k_f, obj_f_cat = _tune_k_by_category(
            train,
            "F",
            k_candidates=k_grid,
            cap_noisy=float(cfg.cap_noisy),
            default_k_general=float(cfg.k_general),
            default_k_noisy=float(cfg.k_noisy),
            objective_score=str(args.objective_score),
            objective_mode=str(args.objective_mode),
            residual_baseline_col=str(args.residual_baseline_col),
            residual_alpha=float(args.residual_alpha),
            xgf_w=float(args.f_obj_xgf_w),
            xga_w=float(args.f_obj_xga_w),
            xgs_w=float(args.f_obj_xgs_w),
            gs_w=float(args.f_obj_gs_w),
            fallback_next_overall=bool(args.objective_fallback_next_overall),
            anchor_next_overall_blend=float(args.objective_anchor_next_overall_blend),
            fallback_source_col=str(args.objective_fallback_source_col),
            anchor_type=str(args.objective_anchor_type),
            eval_seasons=eval_seasons,
        )
        k_d, obj_d_cat = _tune_k_by_category(
            train,
            "D",
            k_candidates=k_grid,
            cap_noisy=float(cfg.cap_noisy),
            default_k_general=float(cfg.k_general),
            default_k_noisy=float(cfg.k_noisy),
            objective_score=str(args.objective_score),
            objective_mode=str(args.objective_mode),
            residual_baseline_col=str(args.residual_baseline_col),
            residual_alpha=float(args.residual_alpha),
            xgf_w=float(args.d_obj_xgf_w),
            xga_w=float(args.d_obj_xga_w),
            xgs_w=float(args.d_obj_xgs_w),
            gs_w=float(args.d_obj_gs_w),
            fallback_next_overall=bool(args.objective_fallback_next_overall),
            anchor_next_overall_blend=float(args.objective_anchor_next_overall_blend),
            fallback_source_col=str(args.objective_fallback_source_col),
            anchor_type=str(args.objective_anchor_type),
            eval_seasons=eval_seasons,
        )
        k_f, kf_changes = _apply_k_floors(k_f, role="F")
        k_d, kd_changes = _apply_k_floors(k_d, role="D")
        k_floor_applied = {
            "F": {k: {"before": float(v0), "after": float(v1)} for k, (v0, v1) in kf_changes.items()},
            "D": {k: {"before": float(v0), "after": float(v1)} for k, (v0, v1) in kd_changes.items()},
        }
        k_map_by_role = {"F": k_f, "D": k_d}
        k_fit_report = {
            "k_candidates": k_grid,
            "role_objective_corr_by_category": {
                "F": {k: (None if not np.isfinite(v) else float(v)) for k, v in obj_f_cat.items()},
                "D": {k: (None if not np.isfinite(v) else float(v)) for k, v in obj_d_cat.items()},
            },
            "min_k_floor_by_role_category": MIN_K_FLOOR_BY_ROLE_CATEGORY,
            "k_floor_applied": k_floor_applied,
        }

    d = _apply_shrink_and_caps(d, cfg, k_map_by_role=k_map_by_role)
    d, defense_upgrade_info = _apply_defense_signal_upgrade(
        d,
        feature_source=args.feature_source,
        enabled=bool(args.defense_signal_upgrade),
        transition_v2_blend=float(args.defense_upgrade_transition_blend),
        transition_v2_blend_f=args.defense_upgrade_transition_blend_f,
        transition_v2_blend_d=args.defense_upgrade_transition_blend_d,
        shot_v2_blend=float(args.defense_upgrade_shot_blend),
        shot_v2_latent_weight=float(args.defense_upgrade_shot_latent_weight),
        entry_weight=float(args.defense_upgrade_entry_weight),
        exit_weight=float(args.defense_upgrade_exit_weight),
    )
    d, a3z_micro_info = _apply_a3z_microstats_upgrade(
        d,
        feature_source=args.feature_source,
        enabled=bool(args.a3z_microstats_upgrade),
        blend=float(args.a3z_microstats_blend),
        blend_f=args.a3z_microstats_blend_f,
        blend_d=args.a3z_microstats_blend_d,
        max_features=int(args.a3z_microstats_max_features),
        min_non_null_frac=float(args.a3z_microstats_min_nonnull_frac),
        min_abs_corr=float(args.a3z_microstats_min_abs_corr),
    )
    d = _apply_forward_calibration_policies(
        d,
        defense_floor_pct=float(args.fwd_defense_floor_pct),
        defense_penalty_strength=float(args.fwd_defense_penalty_strength),
        defense_penalty_power=float(args.fwd_defense_penalty_power),
        spike_reg_strength=float(args.fwd_spike_reg_strength),
        spike_toi_k=float(args.fwd_spike_reg_toi_k),
    )
    d, deov_info = _apply_possession_deoverlap(
        d,
        enabled=bool(args.deoverlap_possession),
        alpha=float(args.deoverlap_alpha),
    )
    d, orth_info = _apply_defense_collinearity_control(
        d,
        enabled=bool(args.orthogonalize_defense),
        alpha=float(args.orthogonalize_defense_alpha),
        residual_weight=float(args.orthogonalize_defense_residual_weight),
        min_abs_corr=float(args.orthogonalize_defense_min_abs_corr),
    )
    d, trdef_guard_info = _apply_transition_defense_consistency_guard(
        d,
        feature_source=args.feature_source,
        enabled=bool(args.trdef_consistency_guard),
        shot_def_threshold=float(args.trdef_guard_shotdef_threshold),
        suppression_pct_threshold=float(args.trdef_guard_suppression_pct_threshold),
        cap_value=float(args.trdef_guard_cap),
        excess_shrink=float(args.trdef_guard_excess_shrink),
    )
    d, component_refine_info = _apply_component_reliability_refinement(
        d,
        enabled=bool(args.component_reliability_refinement),
    )
    if use_walkforward:
        train = d[(season_num >= wf_start) & (season_num <= wf_end)].copy()
    else:
        train = d[d["season_id"].isin(train_seasons)].copy()
    train_toi_eh = _to_num(train.get("eh_rapm_TOI", pd.Series(np.nan, index=train.index)))
    train_toi_a3z = _to_num(train.get("a3z_toi_5v5_min", pd.Series(np.nan, index=train.index)))
    train_toi = train_toi_eh.where(train_toi_eh.notna(), train_toi_a3z).fillna(0.0)
    train = train[train_toi >= 120.0].copy()

    f_train = train[train["position_group"].astype(str).str.upper().eq("F")].copy()
    d_train = train[train["position_group"].astype(str).str.upper().eq("D")].copy()
    if card_yoy:
        f_obj_target = None
        d_obj_target = None
        f_cov = d_cov = float("nan")
        print(
            "Objective: card_yoy pooled R² of recency-weighted 3yr Overall "
            f"(40+ GP, holdout left-season {int(args.card_yoy_holdout_season)}, "
            f"constraints={weight_constraints}, z={card_yoy_z_source})",
            flush=True,
        )
    elif str(args.objective_score).lower() == "logloss":
        f_obj_target = _build_next_year_binary_target(
            f_train,
            source_col=str(args.objective_fallback_source_col),
        )
        d_obj_target = _build_next_year_binary_target(
            d_train,
            source_col=str(args.objective_fallback_source_col),
        )
    else:
        f_obj_target = _build_objective_target(
            f_train,
            mode=str(args.objective_mode),
            anchor_type=str(args.objective_anchor_type),
            residual_baseline_col=str(args.residual_baseline_col),
            residual_alpha=float(args.residual_alpha),
            xgf_w=float(args.f_obj_xgf_w),
            xga_w=float(args.f_obj_xga_w),
            xgs_w=float(args.f_obj_xgs_w),
            gs_w=float(args.f_obj_gs_w),
            fallback_next_overall=bool(args.objective_fallback_next_overall),
            anchor_next_overall_blend=float(args.objective_anchor_next_overall_blend),
            fallback_source_col=str(args.objective_fallback_source_col),
        )
        d_obj_target = _build_objective_target(
            d_train,
            mode=str(args.objective_mode),
            anchor_type=str(args.objective_anchor_type),
            residual_baseline_col=str(args.residual_baseline_col),
            residual_alpha=float(args.residual_alpha),
            xgf_w=float(args.d_obj_xgf_w),
            xga_w=float(args.d_obj_xga_w),
            xgs_w=float(args.d_obj_xgs_w),
            gs_w=float(args.d_obj_gs_w),
            fallback_next_overall=bool(args.objective_fallback_next_overall),
            anchor_next_overall_blend=float(args.objective_anchor_next_overall_blend),
            fallback_source_col=str(args.objective_fallback_source_col),
        )
    if f_obj_target is None:
        f_cov = d_cov = float("nan")
    else:
        f_cov = float(pd.to_numeric(f_obj_target, errors="coerce").notna().mean()) if len(f_obj_target) else float("nan")
        d_cov = float(pd.to_numeric(d_obj_target, errors="coerce").notna().mean()) if len(d_obj_target) else float("nan")
        print(
            f"Objective target coverage: F={f_cov:.3f} D={d_cov:.3f} "
            f"score={str(args.objective_score)} "
            f"(fallback_next_overall={bool(args.objective_fallback_next_overall)} "
            f"blend={float(args.objective_anchor_next_overall_blend):.2f})",
            flush=True,
        )

    prior_f = prior_d = None
    if args.weight_prior_json is not None:
        prior_payload = json.loads(args.weight_prior_json.read_text(encoding="utf-8"))
        prior_f = np.asarray([float(prior_payload.get("weights_F", {}).get(c, 0.0)) for c in CAT_COLS], dtype=float)
        prior_d = np.asarray([float(prior_payload.get("weights_D", {}).get(c, 0.0)) for c in CAT_COLS], dtype=float)
        if prior_f.sum() > 0.0:
            prior_f /= prior_f.sum()
        if prior_d.sum() > 0.0:
            prior_d /= prior_d.sum()

    w_f, obj_f = _tune_role(
        f_train,
        "F",
        cfg,
        objective_target=f_obj_target,
        objective_score=str(args.objective_score),
        eval_seasons=eval_seasons,
        prior_weights=prior_f,
        stability_penalty=float(args.weight_stability_penalty),
        card_yoy=card_yoy,
        weight_constraints=weight_constraints,
        card_yoy_z_source=card_yoy_z_source,
    )
    w_d, obj_d = _tune_role(
        d_train,
        "D",
        cfg,
        objective_target=d_obj_target,
        objective_score=str(args.objective_score),
        eval_seasons=eval_seasons,
        prior_weights=prior_d,
        stability_penalty=float(args.weight_stability_penalty),
        card_yoy=card_yoy,
        weight_constraints=weight_constraints,
        card_yoy_z_source=card_yoy_z_source,
    )

    if bool(args.drop_special_teams):
        for st_col in ("cat_power_play_z", "cat_penalty_kill_z"):
            w_f = _drop_category_weight_and_renorm(w_f, st_col)
            w_d = _drop_category_weight_and_renorm(w_d, st_col)

    # Production rating eligibility. Historical rows remain available for the
    # projection training panel, but 2023-24 onward ratings require meaningful
    # category coverage and at least 100 effective 5v5 minutes.
    evidence_cols = [
        f"{c.replace('_z', '')}_n_features"
        for c in CAT_COLS
        if f"{c.replace('_z', '')}_n_features" in d.columns
    ]
    if evidence_cols:
        evidence = d[evidence_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0.0).sum(axis=1)
    else:
        evidence = d[CAT_COLS].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    toi_eh = _to_num(d.get("eh_rapm_TOI", pd.Series(np.nan, index=d.index)))
    toi_a3z = _to_num(d.get("a3z_toi_5v5_min", pd.Series(np.nan, index=d.index)))
    effective_toi = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0)
    recent_window = pd.to_numeric(d["season_id"], errors="coerce") >= 20232024
    rating_eligible = (~recent_window) | ((evidence >= 6) & (effective_toi >= 100.0))
    d["model_rating_evidence_categories"] = evidence.astype(int)
    d["model_rating_effective_toi"] = effective_toi
    d["model_rating_eligible"] = rating_eligible.astype(bool)

    # Compute tuned overall from RAW z-scores (no TOI shrinkage) for season-level percentiles.
    # This reduces disagreement between tuned_pct and raw_pct/process_pct from 50-70 points
    # down to just the weighting difference. TOI shrinkage is still applied downstream
    # (3yr rolling, card display) via the __adj columns.
    d["overall_8cat_tuned_raw"] = _apply_role_weights(d, w_f, w_d, use_adjusted=False).where(rating_eligible)
    d["overall_8cat_tuned_z"] = _z_by_group(d, "overall_8cat_tuned_raw", ["season_id", "position_group"])
    d["overall_8cat_tuned_pct"] = _pct_by_group(d, "overall_8cat_tuned_z", ["season_id", "position_group"])

    # Also compute adjusted version for downstream shrinkage (backward compatibility)
    d["overall_8cat_tuned_raw_adj"] = _apply_role_weights(d, w_f, w_d, use_adjusted=True).where(rating_eligible)
    d["overall_8cat_tuned_z_adj"] = _z_by_group(d, "overall_8cat_tuned_raw_adj", ["season_id", "position_group"])

    fixed_benchmark = _evaluate_fixed_benchmark(
        d,
        score_col="overall_8cat_tuned_raw",
        eval_seasons=eval_seasons,
    )

    # keep plus tuned fields
    keep = [
        "player_id", "player_name", "season_id", "position_group", "a3z_toi_5v5_min", "eh_rapm_TOI", "eh_std_TOI",
        "a3z_tracked_toi_5v5_min", "a3z_tracking_observed", "a3z_toi_is_fallback", "has_a3z_season",
        "eh_rapm_Team", "eh_std_Team", "a3z_team", "forecheck_event_exposure",
        "overall_8cat_z", "overall_8cat_pct",
        # Primary tuned fields (from RAW z-scores, no TOI shrinkage - for season-level percentiles)
        "overall_8cat_tuned_raw", "overall_8cat_tuned_z", "overall_8cat_tuned_pct",
        # Adjusted tuned fields (with TOI shrinkage - for downstream 3yr rolling/card shrinkage)
        "overall_8cat_tuned_raw_adj", "overall_8cat_tuned_z_adj",
        "model_rating_evidence_categories", "model_rating_effective_toi", "model_rating_eligible",
    ] + CAT_COLS + [f"{c}__adj" for c in CAT_COLS] + [
        "v2_latent_entry_denial_z", "v2_latent_exit_resilience_z",
        "cat_transition_defense_display_z", "cat_transition_defense_display_z__adj",
        "yoy_xGF_per60", "yoy_xGA_per60", "yoy_onice_xg_share", "yoy_onice_goal_share"
    ]
    # Preserve legacy display aliases used by downstream cards/trade/team views.
    for stem in ("cat_shot_offense", "cat_5v5_offense", "cat_5v5_defense", "cat_transition"):
        for suff in ("raw", "z", "pct", "n_features"):
            keep.append(f"{stem}_{suff}")
    for c in CAT_COLS:
        keep.append(f"{c.replace('_z', '')}_n_features")
    keep = [c for c in keep if c in d.columns]
    out = d[keep].copy().sort_values(["season_id", "position_group", "overall_8cat_tuned_z"], ascending=[True, True, False])
    out = _dedupe_player_season(out, label="tune_all_in_one_8cat_model output")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    weights = {
        "meta": {
            "min_5v5_share": cfg.min_5v5_share,
            "cap_noisy": cfg.cap_noisy,
            "k_general": cfg.k_general,
            "k_noisy": cfg.k_noisy,
            "auto_k_by_category": bool(args.auto_k_by_category),
            "n_samples": cfg.n_samples,
            "seed": cfg.seed,
            "weight_constraints": weight_constraints,
            "card_yoy_z_source": card_yoy_z_source if card_yoy else None,
            "weight_prior_json": str(args.weight_prior_json) if args.weight_prior_json is not None else None,
            "weight_stability_penalty": float(args.weight_stability_penalty),
            "train_seasons": sorted(list(train_seasons)),
            "walkforward": {
                "enabled": bool(use_walkforward),
                "start_season": int(wf_start),
                "end_season": int(wf_end),
                "min_train_seasons": int(args.walkforward_min_train_seasons),
                "eval_seasons": [int(x) for x in (eval_seasons or [])],
            },
            "objective": {
                "F": {
                    "xGF_per60": float(args.f_obj_xgf_w),
                    "xGA_per60_flipped": float(args.f_obj_xga_w),
                    "onice_xg_share": float(args.f_obj_xgs_w),
                    "onice_goal_share": float(args.f_obj_gs_w),
                },
                "D": {
                    "xGF_per60": float(args.d_obj_xgf_w),
                    "xGA_per60_flipped": float(args.d_obj_xga_w),
                    "onice_xg_share": float(args.d_obj_xgs_w),
                    "onice_goal_share": float(args.d_obj_gs_w),
                },
                "anchor_type": str(args.objective_anchor_type),
                "mode": str(args.objective_mode),
                "score": str(args.objective_score),
                "residual_baseline_col": str(args.residual_baseline_col),
                "residual_alpha": float(args.residual_alpha),
                "fallback_next_overall": bool(args.objective_fallback_next_overall),
                "anchor_next_overall_blend": float(args.objective_anchor_next_overall_blend),
                "fallback_source_col": str(args.objective_fallback_source_col),
                "drop_special_teams": bool(args.drop_special_teams),
            },
            "fixed_benchmark": {
                "enabled": True,
                "target_weights": FIXED_BENCHMARK_TARGET_WEIGHTS,
                "eval_seasons": [int(x) for x in (eval_seasons or [])],
                "metrics": fixed_benchmark,
            },
            "forward_calibration": {
                "defense_floor_pct": float(args.fwd_defense_floor_pct),
                "defense_penalty_strength": float(args.fwd_defense_penalty_strength),
                "defense_penalty_power": float(args.fwd_defense_penalty_power),
                "spike_reg_strength": float(args.fwd_spike_reg_strength),
                "spike_reg_toi_k": float(args.fwd_spike_reg_toi_k),
            },
            "deoverlap": {
                "possession_enabled": bool(args.deoverlap_possession),
                "alpha": float(args.deoverlap_alpha),
                "n_groups": int(deov_info.get("n_groups", 0.0)),
                "n_rows_residualized": int(deov_info.get("n_rows_residualized", 0.0)),
            },
            "defense_signal_upgrade": defense_upgrade_info,
            "a3z_microstats_upgrade": a3z_micro_info,
            "defense_collinearity": {
                "enabled": bool(args.orthogonalize_defense),
                "alpha": float(args.orthogonalize_defense_alpha),
                "residual_weight": float(args.orthogonalize_defense_residual_weight),
                "min_abs_corr": float(args.orthogonalize_defense_min_abs_corr),
                "n_groups": int(orth_info.get("n_groups", 0.0)),
                "n_rows_residualized": int(orth_info.get("n_rows_residualized", 0.0)),
                "n_groups_skipped_low_corr": int(orth_info.get("n_groups_skipped_low_corr", 0.0)),
            },
            "transition_defense_consistency_guard": trdef_guard_info,
            "component_reliability_refinement": component_refine_info,
        },
        "objective_value": {"F": _json_float_or_none(obj_f), "D": _json_float_or_none(obj_d)},
        "weights_F": {k: float(v) for k, v in zip(CAT_COLS, w_f)},
        "weights_C": {k: float(v) for k, v in zip(CAT_COLS, w_f)},
        "weights_W": {k: float(v) for k, v in zip(CAT_COLS, w_f)},
        "weights_D": {k: float(v) for k, v in zip(CAT_COLS, w_d)},
    }
    if k_map_by_role is not None:
        weights["k_by_role_category"] = {
            role: {k: float(v) for k, v in km.items()}
            for role, km in k_map_by_role.items()
        }
        weights["k_fit_report"] = k_fit_report
    args.out_weights_json.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    if str(args.objective_score).lower() == "logloss":
        print(f"Tuned objective (-logloss) F={obj_f:.4f} D={obj_d:.4f}")
        print(f"Implied mean logloss F={-obj_f:.4f} D={-obj_d:.4f}")
    else:
        print(f"Tuned objective F={obj_f:.4f} D={obj_d:.4f}")
    print(
        "Fixed benchmark (locked target):",
        f"R_season_mean={float(fixed_benchmark.get('r_all_season_mean', float('nan'))):.4f}",
        f"R2_season_mean={float(fixed_benchmark.get('r2_all_season_mean', float('nan'))):.4f}",
        f"R_pooled={float(fixed_benchmark.get('r_all_pooled', float('nan'))):.4f}",
    )
    print("Weights F:")
    for k, v in zip(CAT_COLS, w_f):
        print(f"  {k}: {v:.4f}")
    print("Weights D:")
    for k, v in zip(CAT_COLS, w_d):
        print(f"  {k}: {v:.4f}")
    print(
        "De-overlap possession:",
        f"enabled={bool(args.deoverlap_possession)}",
        f"alpha={float(args.deoverlap_alpha):.2f}",
        f"groups={int(deov_info.get('n_groups', 0.0))}",
        f"rows={int(deov_info.get('n_rows_residualized', 0.0))}",
    )
    print(
        "Defense signal upgrade:",
        f"enabled={bool(defense_upgrade_info.get('enabled', False))}",
        f"applied={bool(defense_upgrade_info.get('applied', False))}",
        (
            f"transition_blend={float(defense_upgrade_info.get('blend', {}).get('transition_v2_blend', 0.0)):.2f}"
            if isinstance(defense_upgrade_info.get("blend"), dict)
            else ""
        ),
        (
            f"shot_blend={float(defense_upgrade_info.get('blend', {}).get('shot_v2_blend', 0.0)):.2f}"
            if isinstance(defense_upgrade_info.get("blend"), dict)
            else ""
        ),
        (
            f"shot_latent={float(defense_upgrade_info.get('blend', {}).get('shot_v2_latent_weight', 0.0)):.2f}"
            if isinstance(defense_upgrade_info.get("blend"), dict)
            else ""
        ),
    )
    print(
        "A3Z microstats upgrade:",
        f"enabled={bool(a3z_micro_info.get('enabled', False))}",
        f"applied={bool(a3z_micro_info.get('applied', False))}",
        f"blend={float(a3z_micro_info.get('blend', 0.0)):.2f}",
        (
            f"blend_F={float(a3z_micro_info.get('blend_by_role', {}).get('F', a3z_micro_info.get('blend', 0.0))):.2f}"
            if isinstance(a3z_micro_info.get("blend_by_role"), dict)
            else ""
        ),
        (
            f"blend_D={float(a3z_micro_info.get('blend_by_role', {}).get('D', a3z_micro_info.get('blend', 0.0))):.2f}"
            if isinstance(a3z_micro_info.get("blend_by_role"), dict)
            else ""
        ),
        f"candidate_pool={int(a3z_micro_info.get('candidate_pool_size', 0))}",
    )
    print(
        "Defense collinearity control:",
        f"enabled={bool(args.orthogonalize_defense)}",
        f"alpha={float(args.orthogonalize_defense_alpha):.2f}",
        f"residual_weight={float(args.orthogonalize_defense_residual_weight):.2f}",
        f"min_abs_corr={float(args.orthogonalize_defense_min_abs_corr):.2f}",
        f"groups={int(orth_info.get('n_groups', 0.0))}",
        f"rows={int(orth_info.get('n_rows_residualized', 0.0))}",
        f"groups_skipped_low_corr={int(orth_info.get('n_groups_skipped_low_corr', 0.0))}",
    )
    print(
        "Component reliability refinement:",
        f"enabled={bool(component_refine_info.get('enabled', False))}",
        f"applied={bool(component_refine_info.get('applied', False))}",
        f"n_components={len(component_refine_info.get('components', {})) if isinstance(component_refine_info.get('components', {}), dict) else 0}",
    )
    if k_map_by_role is not None:
        print("Category-specific TOI k (minutes) by role:")
        for role in ["F", "D"]:
            print(f"  {role}:")
            for c in CAT_COLS:
                print(f"    {c}: {k_map_by_role[role][c]:.1f}")
        if any(bool(k_floor_applied.get(r)) for r in ["F", "D"]):
            print("Applied minimum k floors (before -> after):")
            for role in ["F", "D"]:
                ch = k_floor_applied.get(role, {})
                if not ch:
                    continue
                print(f"  {role}:")
                for c, vv in ch.items():
                    print(f"    {c}: {float(vv['before']):.1f} -> {float(vv['after']):.1f}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_parquet}")
    print(f"Wrote {args.out_weights_json}")


if __name__ == "__main__":
    main()
