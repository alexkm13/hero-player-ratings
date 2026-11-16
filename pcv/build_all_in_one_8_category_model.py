#!/usr/bin/env python3
"""
Build 10-category all-in-one player-season scores from merged feature table.

Eleven categories: 5v5 O/D, possession, shot creation, **finishing** (G vs ixG, SH% vs xSH%),
playmaking, transition, ST, penalties, forecheck, physicality.

Method:
- Per-feature z-score within (season_id, position_group)
- Category = weighted mean of available feature z-scores
- Category z-score re-standardized within (season_id, position_group)
- Category percentile within (season_id, position_group)

**Possession process (hybrid):** PTV captures *pre-shot* process (dangerous states before
a shot, including possessions that never become shots). EH RAPM xGA / xG± capture
*shot-based* on-ice results. Offense leans PTV (pre-shot, possession offense, shot
threat); defense blends PTV possession defense + threat suppression with EH xGA/xG± so
pre-shot denial and shot outcomes both count.

Note: current EH files are EV regular only, so Special Teams remains NaN placeholder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hero_evidence_exposure import exposure_proxies, numeric as evidence_numeric, preserve_tracking

from all_in_one_category_specs import (
    FINISHING_DERIVED_COLS,
    FORECHECK_PHYSICALITY_ROLE_SPECS,
    assert_component_feature_contract,
    assert_disjoint_a3z_microstats,
    assert_isolated_feature_ownership,
    assert_no_box_score_points_in_components,
)
from ptv_coef import assert_category_specs_use_rapm_pcv

ROOT = Path(__file__).resolve().parent
DEFAULT_EDGE_TABLE = ROOT / "data/pcv/nhl_edge_skater_stats_detail.csv"
DEFAULT_SKATING_WEIGHTS_JSON = ROOT / "data/pcv/skating_component_weights_best.json"
DEFAULT_SHOT_CREATION_WEIGHTS_JSON = ROOT / "data/pcv/shot_creation_component_weights_best.json"
DEFAULT_PCV_WEIGHTS_JSON = ROOT / "data/pcv/pcv_component_weights_best.json"

SHOT_CREATION_COMPONENTS = [
    "rapm_hybrid_xGF/60",
    "eh_std_ixG/60",
    "eh_std_iCF/60",
    "a3z_chances_per_60",
    "a3z_carries_with_chances_per_60",
    "a3z_shots_per_60",
    "ptv_coalesced_rapm_shot_threat",
    # Live feature-table column (alias ptv_coalesced_rapm_possession_offense is TO-owned).
    "ptv_coalesced_possession_offense",
]

# Production SC mix (volume + ixG included). Finishing overall anchor handles spam.
DEFAULT_SHOT_CREATION_WEIGHTS = {
    "rapm_hybrid_xGF/60": 0.24224169603458898,
    "eh_std_ixG/60": 0.24591860793556866,
    "eh_std_iCF/60": 0.0,
    "a3z_chances_per_60": 0.0,
    "a3z_carries_with_chances_per_60": 0.09817122715430951,
    "a3z_shots_per_60": 0.0,
    "ptv_coalesced_rapm_shot_threat": 0.15514444787379103,
    "ptv_coalesced_possession_offense": 0.2585240210017417,
}

from skating_pillar_features import (
    add_skating_pillar_features,
    load_skating_pillar_weights,
    skating_model_hybrid_spec,
)

EDGE_MERGE_COLUMNS = (
    "edge_speed_max_pct",
    "edge_speed_max_mph",
    "edge_speed_max_kmh",
    "edge_bursts_over20_pct",
    "edge_bursts_over22_pct",
    "edge_bursts_20_to_22_pct",
    "edge_bursts_18_to_20_pct",
    "edge_distance_total_pct",
    "edge_distance_max_game_pct",
    "edge_es_distance_per60_pct",
    "edge_es_distance_max_period_pct",
    "edge_es_distance_max_game_pct",
    "edge_all_distance_per60_pct",
    "edge_oz_time_ev_pctg_pct",
    "edge_nz_time_pctg_pct",
    "edge_dz_time_pctg_pct",
)


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


def _add_finishing_derived_features(df: pd.DataFrame) -> None:
    """Goals/shooting vs expected with volume-aware shrinkage for stability."""
    g = pd.to_numeric(df.get("eh_std_G/60"), errors="coerce")
    ixg = pd.to_numeric(df.get("eh_std_ixG/60"), errors="coerce")
    sh = pd.to_numeric(df.get("eh_std_Sh%"), errors="coerce")
    xsh = pd.to_numeric(df.get("eh_std_xFSh%"), errors="coerce")
    isf = pd.to_numeric(df.get("eh_std_iSF/60"), errors="coerce")
    toi_std = pd.to_numeric(df.get("eh_std_TOI"), errors="coerce")
    toi_rapm = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
    toi = toi_std.where(toi_std.notna(), toi_rapm).fillna(0.0)
    shots_total = (isf.fillna(0.0) * toi / 60.0).clip(lower=0.0)

    raw_g_above = (g - ixg).clip(lower=-1.0, upper=1.0)
    raw_sh_above = (sh - xsh).clip(lower=-10.0, upper=10.0)
    g_per_shot = np.where(isf > 0.0, g / isf, np.nan)
    xg_per_shot = np.where(isf > 0.0, ixg / isf, np.nan)
    raw_gps_above = pd.Series(g_per_shot - xg_per_shot, index=df.index, dtype=float).clip(lower=-0.22, upper=0.22)

    # Heavier shrinkage improves repeatability of conversion-skill estimates.
    rel_toi = toi / (toi + 2600.0)
    rel_shots = shots_total / (shots_total + 900.0)

    pid = pd.to_numeric(df.get("player_id"), errors="coerce")
    sid = pd.to_numeric(df.get("season_id"), errors="coerce")
    order = pd.DataFrame({"idx": np.arange(len(df)), "pid": pid, "sid": sid}).sort_values(
        ["pid", "sid", "idx"], kind="mergesort"
    )
    g_hist = pd.Series(np.nan, index=df.index, dtype=float)
    sh_hist = pd.Series(np.nan, index=df.index, dtype=float)
    gps_hist = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in order.groupby("pid", dropna=False).groups.items():
        g_seq = raw_g_above.loc[idx]
        sh_seq = raw_sh_above.loc[idx]
        gps_seq = raw_gps_above.loc[idx]
        # Rolling 3-season prior keeps finishing responsive while avoiding one-year spikes.
        g_hist.loc[idx] = g_seq.shift(1).rolling(window=3, min_periods=1).mean()
        sh_hist.loc[idx] = sh_seq.shift(1).rolling(window=3, min_periods=1).mean()
        gps_hist.loc[idx] = gps_seq.shift(1).rolling(window=3, min_periods=1).mean()

    g_blended = raw_g_above * rel_toi + g_hist.fillna(0.0) * (1.0 - rel_toi)
    sh_blended = raw_sh_above * rel_shots + sh_hist.fillna(0.0) * (1.0 - rel_shots)
    gps_blended = raw_gps_above * rel_shots + gps_hist.fillna(0.0) * (1.0 - rel_shots)

    df["eh_finishing_goals_above_ixg_per_60"] = raw_g_above
    df["eh_finishing_sh_pct_above_xsh"] = raw_sh_above
    df["eh_finishing_goals_above_ixg_per_60_blended"] = g_blended
    df["eh_finishing_sh_pct_above_xsh_blended"] = sh_blended
    df["eh_finishing_g_per_shot_above_xg_per_shot_blended"] = gps_blended
    df["eh_finishing_goals_above_ixg_per_60_shrunk"] = g_blended * rel_toi
    df["eh_finishing_sh_pct_above_xsh_shrunk"] = sh_blended * rel_shots
    df["eh_finishing_g_per_shot_above_xg_per_shot_shrunk"] = gps_blended * rel_shots


TEAMMATE_FINISHER_ASSIST_FEATURES: tuple[str, ...] = (
    "a3z_chance_assists_per_60",
    "a3z_primary_shot_assists_per_60",
    "a3z_home_plate_passes_per_60",
    "a3z_center_lane_assists_per_60",
)


def _pick_team_col(df: pd.DataFrame) -> str | None:
    for c in ("eh_rapm_Team", "eh_std_Team", "a3z_team"):
        if c in df.columns:
            return c
    return None


def _player_toi_series(df: pd.DataFrame) -> pd.Series:
    toi = _to_num(df.get("eh_rapm_TOI", pd.Series(np.nan, index=df.index)))
    toi = toi.where(toi.notna(), _to_num(df.get("eh_std_TOI", pd.Series(np.nan, index=df.index))))
    return toi.where(toi.notna(), _to_num(df.get("a3z_toi_5v5_min", pd.Series(np.nan, index=df.index))))


def _player_finishing_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """Forward finishing skill z from EH shrunk conversion features."""
    parts: list[pd.Series] = []
    for col in (
        "eh_finishing_goals_above_ixg_per_60_shrunk",
        "eh_finishing_sh_pct_above_xsh_shrunk",
        "eh_finishing_g_per_shot_above_xg_per_shot_shrunk",
    ):
        if col not in df.columns:
            continue
        tmp = df[group_cols].copy()
        tmp["v"] = _to_num(df[col])
        parts.append(_z_by_group(tmp, "v", group_cols))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _add_teammate_finishing_proxy_z(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    top_n_forwards: int = 6,
    team_col: str | None = None,
) -> None:
    """
    TOI-weighted finishing skill of each team's top forwards.

    Defensemen receive the full forward pool; forwards get leave-one-out so a
    player's own finishing does not define their teammate context.
    """
    team_key = team_col or _pick_team_col(df)
    if team_key is None:
        df["teammate_finishing_proxy_z"] = np.nan
        return

    pos = df["position_group"].astype(str).str.upper().str.strip()
    fin_z = _player_finishing_skill_z(df, group_cols)
    toi = _player_toi_series(df)
    team = df[team_key].astype(str).str.strip()
    season = pd.to_numeric(df["season_id"], errors="coerce")
    is_multi = team.str.upper().isin({"MULTI", "TOT", ""})

    out = pd.Series(np.nan, index=df.index, dtype=float)
    keys = pd.DataFrame({"season_id": season, "_team": team})
    for (_, _team), idx in keys.groupby(["season_id", "_team"], dropna=False).groups.items():
        if is_multi.loc[idx].any():
            continue
        g = df.loc[idx]
        g_pos = pos.loc[idx]
        fwd_idx = g.index[g_pos.eq("F")]
        if len(fwd_idx) == 0:
            continue
        fwd = pd.DataFrame(
            {
                "fin_z": fin_z.loc[fwd_idx],
                "toi": toi.loc[fwd_idx],
            },
            index=fwd_idx,
        )
        fwd = fwd[fwd["fin_z"].notna() & fwd["toi"].notna() & (fwd["toi"] > 0.0)]
        if fwd.empty:
            continue
        pool = fwd.nlargest(int(max(top_n_forwards, 1)), "toi")

        def _weighted_mean(sub: pd.DataFrame) -> float:
            w = sub["toi"].to_numpy(float)
            z = sub["fin_z"].to_numpy(float)
            if w.sum() <= 0.0:
                return float("nan")
            return float(np.average(z, weights=w))

        d_idx = g.index[g_pos.eq("D")]
        if len(d_idx):
            out.loc[d_idx] = _weighted_mean(pool)

        f_idx = g.index[g_pos.eq("F")]
        for i in f_idx:
            sub = pool.drop(index=i, errors="ignore")
            if sub.empty:
                sub = pool
            out.loc[i] = _weighted_mean(sub)

    df["teammate_finishing_proxy_z"] = out


def _residualize_assist_features_vs_teammate_finishing(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    features: tuple[str, ...] = TEAMMATE_FINISHER_ASSIST_FEATURES,
    roles: tuple[str, ...] = ("D",),
    ridge_alpha: float = 2.0,
    min_group_rows: int = 20,
    residual_weight: float = 1.0,
) -> dict[str, float]:
    """
    Remove the teammate-finisher-explained component from assist micro features.

    Fits y ~ teammate_finishing_proxy_z within each (season, position_group)
    cell and stores teammate-adjusted assist rates before playmaking z-scoring.
    """
    info: dict[str, float] = {
        "enabled": 1.0,
        "groups_used": 0.0,
        "rows_updated": 0.0,
    }
    if "teammate_finishing_proxy_z" not in df.columns:
        info["enabled"] = 0.0
        return info

    role_set = {str(r).upper().strip() for r in roles}
    pos = df["position_group"].astype(str).str.upper().str.strip()
    role_mask = pos.isin(role_set)
    x_all = _to_num(df["teammate_finishing_proxy_z"])
    w = float(np.clip(float(residual_weight), 0.0, 1.0))
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))

    for feat in features:
        if feat not in df.columns:
            continue
        backup = f"{feat}__pre_teammate_resid"
        if backup not in df.columns:
            df[backup] = _to_num(df[feat])
        y0 = _to_num(df[feat])
        out = y0.copy()
        for _, idx in df.groupby(group_cols, dropna=False).groups.items():
            sub_role = role_mask.loc[idx]
            if not sub_role.any():
                continue
            x = x_all.loc[idx]
            y = y0.loc[idx]
            m = sub_role & x.notna() & y.notna()
            n = int(m.sum())
            if n < mmin:
                continue
            xs = x[m].to_numpy(float)
            ys = y[m].to_numpy(float)
            mx = float(np.mean(xs))
            my = float(np.mean(ys))
            xv = xs - mx
            yv = ys - my
            denom = float(np.sum(xv * xv) + lam)
            if not np.isfinite(denom) or denom <= 1e-9:
                continue
            beta = float(np.sum(xv * yv) / denom)
            pred = my + beta * (xs - mx)
            resid = ys - pred
            y_new = (1.0 - w) * ys + w * resid
            out.loc[x[m].index] = y_new
            info["groups_used"] += 1.0
            info["rows_updated"] += float(n)
        df[feat] = out
    return info


def _apply_teammate_finishing_assist_residualization(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    enabled: bool,
    top_n_forwards: int,
    ridge_alpha: float,
    min_group_rows: int,
    residual_weight: float,
    roles: tuple[str, ...],
) -> dict[str, float]:
    info: dict[str, float] = {"enabled": float(enabled)}
    if not enabled:
        return info
    _add_teammate_finishing_proxy_z(df, group_cols=group_cols, top_n_forwards=top_n_forwards)
    resid_info = _residualize_assist_features_vs_teammate_finishing(
        df,
        group_cols=group_cols,
        features=TEAMMATE_FINISHER_ASSIST_FEATURES,
        roles=roles,
        ridge_alpha=ridge_alpha,
        min_group_rows=min_group_rows,
        residual_weight=residual_weight,
    )
    info.update(resid_info)
    return info


def _add_playmaking_derived_features(df: pd.DataFrame) -> None:
    """
    Expected-assist proxy from A3Z shot-assist volume rates.

    We do not currently have a direct EH xA field in the source tables, so this
    proxy uses chance and primary shot assists per-60 as the closest expected
    playmaking signal.
    """
    c = evidence_numeric(df, "a3z_chance_assists_per_60")
    p = evidence_numeric(df, "a3z_primary_shot_assists_per_60")
    # Slightly favor chance assists as broader xA-like creation signal.
    xa = 0.60 * c + 0.40 * p
    xa_tot = exposure_proxies(df)["playmaking_weighted_assists"]
    # Unknown tracking is not an observed zero-assist season.
    has_measurement = xa_tot.notna() & xa.notna()
    xa_observed = xa.where(has_measurement)
    rel_xa = (xa_tot / (xa_tot + 180.0)).where(has_measurement, 0.0)

    pid = pd.to_numeric(df.get("player_id"), errors="coerce")
    sid = pd.to_numeric(df.get("season_id"), errors="coerce")
    order = pd.DataFrame({"idx": np.arange(len(df)), "pid": pid, "sid": sid}).sort_values(
        ["pid", "sid", "idx"], kind="mergesort"
    )
    xa_hist = pd.Series(np.nan, index=df.index, dtype=float)
    for _, idx in order.groupby("pid", dropna=False).groups.items():
        xa_seq = xa_observed.loc[idx]
        xa_hist.loc[idx] = xa_seq.shift(1).rolling(window=3, min_periods=1).mean()
    xa_blended = xa_observed.fillna(0.0) * rel_xa + xa_hist.fillna(0.0) * (1.0 - rel_xa)
    pass_qual = (
        0.55 * evidence_numeric(df, "a3z_home_plate_passes_per_60")
        + 0.45 * evidence_numeric(df, "a3z_center_lane_assists_per_60")
    )

    df["a3z_expected_assists_proxy_per_60"] = xa
    df["a3z_expected_assists_proxy_per_60_blended"] = xa_blended.where(has_measurement | xa_hist.notna())
    df["a3z_expected_assists_proxy_per_60_shrunk"] = (xa_blended * rel_xa).where(has_measurement)
    df["a3z_playmaking_evidence_count_proxy"] = xa_tot
    df["a3z_playmaking_evidence_reliability"] = rel_xa.where(has_measurement)
    df["a3z_playmaking_quality_proxy_per_60"] = pass_qual


PTV_RAPM_STEMS = [
    "pre_shot_pressure",
    "possession_offense",
    "shot_threat",
    "possession_defense",
    "threat_suppression",
]


def _add_ptv_coalesced_columns(df: pd.DataFrame) -> None:
    """
    Build season-appropriate PTV columns using per-row fallback:
    season -> roll3 -> pooled.

    Intentionally excludes ptv_best_* to avoid leaking best-known signal
    into season-level ratings.
    """
    def _series(name: str) -> pd.Series:
        return pd.to_numeric(df.get(name, pd.Series(np.nan, index=df.index)), errors="coerce")

    for stem in PTV_RAPM_STEMS:
        coalesced_col = f"ptv_coalesced_{stem}"
        if coalesced_col in df.columns:
            df[f"ptv_coalesced_rapm_{stem}"] = _series(coalesced_col)
            continue
        s = _series(f"ptv_season_{stem}")
        r = _series(f"ptv_roll3_{stem}")
        p = _series(f"ptv_pooled_{stem}")
        df[f"ptv_coalesced_rapm_{stem}"] = s.where(s.notna(), r).where(lambda x: x.notna(), p)


def _add_rapm_hybrid_columns(df: pd.DataFrame) -> None:
    """Prefer prior-adjusted RAPM when present; fallback to EH RAPM."""
    for stem in ["xGF/60", "xGA/60", "xG±/60"]:
        hs = pd.to_numeric(df.get(f"hs_prior_adj_rapm_{stem}"), errors="coerce")
        eh = pd.to_numeric(df.get(f"eh_rapm_{stem}"), errors="coerce")
        df[f"rapm_hybrid_{stem}"] = hs.where(hs.notna(), eh)


DEFAULT_TEAM_CONTEXT_COLUMNS = (
    "eh_rapm_CA/60",
    "eh_rapm_xGA/60",
    "rapm_hybrid_xGA/60",
)


def _apply_team_context_correction(
    df: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    team_col: str = "eh_rapm_Team",
    lambda_: float = 0.5,
    min_team_n: int = 4,
) -> dict:
    """Subtract ``lambda_ * (team_mean - position_mean)`` from defensive RAPM columns.

    Grouping is by (season, team, position_group) -- F vs D are corrected
    against separate baselines so a forward on a bad defensive team isn't
    moved by the team's blueline performance.

    Rationale: while RAPM nominally controls for teammates and competition,
    on teams with low line/pair turnover RAPM has limited ability to fully
    separate individuals from team systems. Pulling a fraction of the team
    excess out of each player's score reduces the team-context drag.
    ``lambda_=0`` disables this step; ``lambda_=1`` removes the full team
    excess (which would over-correct elite defenders on bad teams).

    Rows whose ``team_col`` value is ``MULTI`` / ``TOT`` (i.e. EH's pooled
    multi-team aggregate for traded players) are not adjusted, since their
    "team" is not a single defensive system.
    """
    info: dict = {
        "enabled": False,
        "lambda": float(lambda_),
        "team_col": team_col,
        "min_team_n": int(min_team_n),
        "columns": list(columns),
        "rows_adjusted_by_col": {},
    }
    if lambda_ <= 0 or not columns:
        return info
    if team_col not in df.columns:
        info["warning"] = f"team_col {team_col!r} missing"
        return info

    team = df[team_col].astype(str).str.upper().str.strip()
    is_multi = team.isin({"MULTI", "TOT", ""})
    valid_team_row = ~is_multi

    season = df["season_id"]
    pos = df["position_group"].astype(str).str.upper().str.strip()

    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        pos_mean = s.groupby([season, pos]).transform("mean")
        s_eff = s.where(valid_team_row, np.nan)
        team_mean = s_eff.groupby([season, team, pos]).transform("mean")
        team_n = s_eff.groupby([season, team, pos]).transform("count")
        excess = (team_mean - pos_mean).fillna(0.0)
        apply_row = valid_team_row & (team_n >= int(min_team_n))
        adj_amount = (float(lambda_) * excess).where(apply_row, 0.0)
        df[col] = (s - adj_amount).astype(float)
        info["rows_adjusted_by_col"][col] = int(apply_row.sum())
    info["enabled"] = True
    return info


def _add_edge_derived_features(df: pd.DataFrame) -> None:
    """
    Build EDGE skating composites with robust fallback:
    - burst: blend over20 and over22 percentiles
    - engine: prefer ES distance/60, else all distance/60, else max-game distance (not season total miles)
    """
    def _edge_series(col: str) -> pd.Series:
        raw = df.get(col)
        if isinstance(raw, pd.Series):
            return pd.to_numeric(raw, errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    over20 = _edge_series("edge_bursts_over20_pct")
    over22 = _edge_series("edge_bursts_over22_pct")
    # Convert blended percentile to z-like centered signal; keep scale stable via group z later.
    burst = 0.6 * over20 + 0.4 * over22
    df["edge_burst_composite_pct"] = burst.where(burst.notna(), over20)

    es60 = _edge_series("edge_es_distance_per60_pct")
    all60 = _edge_series("edge_all_distance_per60_pct")
    max_game = _edge_series("edge_distance_max_game_pct")
    # Season total distance is mostly workload/TOI; avoid using it for skating skill.
    df["edge_engine_composite_pct"] = es60.where(es60.notna(), all60).where(max_game.notna(), max_game)


def _merge_edge_features(df: pd.DataFrame, edge_path: Path | None) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Left-merge NHL EDGE percentiles from the sidecar feature table.

    The default all-in-one feature table omits EDGE columns; skating depends on them.
    """
    path = edge_path if edge_path is not None else DEFAULT_EDGE_TABLE
    info = {"path": str(path), "merged": False, "rows_with_any_edge": 0.0}
    if not path.is_file():
        print(f"EDGE merge skipped: missing sidecar table {path}", flush=True)
        return df, info

    usecols = {"player_id", "season_id", *EDGE_MERGE_COLUMNS}
    if path.suffix.lower() == ".parquet":
        edge = pd.read_parquet(path)
        edge = edge[[c for c in edge.columns if c in usecols]].copy()
    else:
        edge = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    edge["player_id"] = pd.to_numeric(edge["player_id"], errors="coerce").astype("Int64")
    edge["season_id"] = pd.to_numeric(edge["season_id"], errors="coerce").astype("Int64")
    edge = edge[edge["player_id"].notna() & edge["season_id"].notna()].copy()
    edge["player_id"] = edge["player_id"].astype(int)
    edge["season_id"] = edge["season_id"].astype(int)
    edge = edge.drop_duplicates(["player_id", "season_id"], keep="last")

    present = [c for c in EDGE_MERGE_COLUMNS if c in edge.columns]
    if not present:
        print(f"EDGE merge skipped: no edge columns in {path}", flush=True)
        return df, info

    merge_cols = ["player_id", "season_id", *present]
    edge = edge[merge_cols]
    before = len(df)
    out = df.drop(columns=[c for c in present if c in df.columns], errors="ignore").merge(
        edge,
        on=["player_id", "season_id"],
        how="left",
    )
    if len(out) != before:
        raise RuntimeError("EDGE merge changed row count")

    any_edge = out[present].notna().any(axis=1)
    info["merged"] = True
    info["rows_with_any_edge"] = float(any_edge.sum())
    info["edge_column_coverage"] = {c: float(out[c].notna().mean()) for c in present}
    print(
        f"EDGE merge: path={path.name} cols={len(present)} "
        f"rows_with_any_edge={int(info['rows_with_any_edge']):,}/{len(out):,}",
        flush=True,
    )
    return out, info


def _load_skating_component_weights(path: Path | None) -> dict[str, dict[str, float]]:
    return load_skating_pillar_weights(path)


def _normalize_shot_creation_weight_dict(pick: dict[str, float]) -> dict[str, float]:
    out = {
        c: float(pick.get(c, DEFAULT_SHOT_CREATION_WEIGHTS[c]))
        for c in SHOT_CREATION_COMPONENTS
    }
    s = sum(out.values())
    if s <= 0:
        return dict(DEFAULT_SHOT_CREATION_WEIGHTS)
    return {c: float(out[c]) / s for c in SHOT_CREATION_COMPONENTS}


def _load_shot_creation_component_weights(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.is_file():
        base = _normalize_shot_creation_weight_dict(DEFAULT_SHOT_CREATION_WEIGHTS)
        return {"F": base, "D": dict(base)}
    raw = json.loads(path.read_text(encoding="utf-8"))
    f_pick = raw.get("F") or raw.get("default") or raw
    d_pick = raw.get("D") or f_pick
    f_weights = _normalize_shot_creation_weight_dict(f_pick)
    d_weights = _normalize_shot_creation_weight_dict(d_pick)
    return {"F": f_weights, "D": d_weights}


def _shot_creation_role_specs(weights_by_role: dict[str, dict[str, float]]) -> dict[str, list[tuple[str, float]]]:
    return {
        role: [
            (c, float(weights_by_role[role][c]))
            for c in SHOT_CREATION_COMPONENTS
            if c in weights_by_role[role] and float(weights_by_role[role][c]) > 0
        ]
        for role in weights_by_role
    }


def _skating_offense_hybrid_spec(role: str, comp_w: dict[str, float]) -> list[tuple[str, float]]:
    return skating_model_hybrid_spec(role, comp_w)


def _skating_defense_hybrid_spec(role: str, comp_w: dict[str, float]) -> list[tuple[str, float]]:
    return skating_model_hybrid_spec(role, comp_w)


def _transition_offense_micro_spec() -> list[tuple[str, float]]:
    """Prefer quality/rate micros over raw touch volume (entries/carries)."""
    return [
        ("a3z_entries_per_60", 0.14),
        ("a3z_carries_per_60", 0.14),
        ("a3z_entries_with_passing_play_per_60", 0.16),
        ("a3z_controlled_exits_per_60", 0.20),
        ("a3z_carried_exits_per_60", 0.12),
        ("a3z_passed_exits_per_60", 0.10),
        ("a3z_failed_entries_per_entry", -0.14),
    ]


def _playmaking_role_specs() -> dict[str, list[tuple[str, float]]]:
    """Playmaking: same expected-assist micro stack for F and D (D post-build micro anchor)."""
    specs = [
        ("a3z_expected_assists_proxy_per_60_shrunk", 0.60),
        ("a3z_playmaking_quality_proxy_per_60", 0.40),
    ]
    return {"F": specs, "D": list(specs)}


def _transition_defense_role_specs() -> dict[str, list[tuple[str, float]]]:
    """
    Transition defense: deny entries, recover pucks, and turn DZ touches into clean exits.

    This is intentionally not an xGA proxy. Shot/chance suppression belongs to
    Shot Defense; transition defense is a micro-process trait with only indirect
    outcome support through entry denial and exit failure rates.
    """
    return {
        "F": [
            ("a3z_denials_per_target", 0.20),
            ("a3z_dz_controlled_breakout_per_60", 0.04),
            ("a3z_retrievals_leading_to_exits_per_60", 0.38),
            ("a3z_dz_retrievals_per_60", 0.10),
            ("a3z_passes_allowed_per_target", -0.04),
            ("a3z_failed_exits_per_exit_touch", -0.10),
        ],
        "D": [
            ("a3z_denials_per_target", 0.24),
            ("a3z_dz_controlled_breakout_per_60", 0.03),
            ("a3z_retrievals_leading_to_exits_per_60", 0.42),
            ("a3z_dz_retrievals_per_60", 0.08),
            ("a3z_passes_allowed_per_target", -0.05),
            ("a3z_failed_exits_per_exit_touch", -0.10),
        ],
    }


DEFAULT_SHOT_DEFENSE_ROLE_SPECS: dict[str, list[tuple[str, float]]] = {
    "F": [
        ("ptv_coalesced_rapm_threat_suppression", 0.34),
        ("ptv_coalesced_rapm_possession_defense", 0.22),
        ("eh_rapm_CA/60", -0.18),
        ("rapm_hybrid_xGA/60", -0.14),
        ("a3z_carries_with_chance_against_per_target", -0.08),
        ("a3z_carries_with_chance_against_per_60", -0.04),
    ],
    "D": [
        ("ptv_coalesced_rapm_threat_suppression", 0.36),
        ("ptv_coalesced_rapm_possession_defense", 0.24),
        ("eh_rapm_CA/60", -0.16),
        ("rapm_hybrid_xGA/60", -0.14),
        ("a3z_carries_with_chance_against_per_target", -0.06),
        ("a3z_carries_with_chance_against_per_60", -0.04),
    ],
}

DEFAULT_TO_PTV_SHARE = {"F": 0.45, "D": 0.45}


def _shot_defense_role_specs(
    override: dict[str, list[tuple[str, float]]] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    Shot defense: suppress attempts and dangerous looks after the opponent is attacking.

    Direct xGA/CA is an outcome anchor, not the category definition. The bulk of
    this component comes from PTV threat/possession suppression and A3Z defensive
    micro so the bar measures repeatable defensive process instead of merely
    re-ranking players by on-ice xGA.
    """
    src = override or DEFAULT_SHOT_DEFENSE_ROLE_SPECS
    return {role: list(src.get(role, DEFAULT_SHOT_DEFENSE_ROLE_SPECS[role])) for role in ("F", "D")}


def _finishing_role_specs() -> dict[str, list[tuple[str, float]]]:
    """Role-specific finishing skill blend with stability-aware conversion signals."""
    return {
        "F": [
            ("eh_finishing_sh_pct_above_xsh_shrunk", 0.45),
            ("eh_finishing_sh_pct_above_xsh_blended", 0.35),
            ("eh_finishing_goals_above_ixg_per_60_shrunk", 0.10),
            ("eh_finishing_g_per_shot_above_xg_per_shot_shrunk", 0.10),
        ],
        "D": [
            ("eh_finishing_sh_pct_above_xsh_shrunk", 0.42),
            ("eh_finishing_sh_pct_above_xsh_blended", 0.38),
            ("eh_finishing_goals_above_ixg_per_60_shrunk", 0.10),
            ("eh_finishing_g_per_shot_above_xg_per_shot_shrunk", 0.10),
        ],
    }


def _transition_offense_role_specs(
    ptv_share_by_role: dict[str, float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    A3Z transition micro + PTV possession offense.

    D uses the same feature stack as F; D-only post-build micro anchors/caps
    prevent shutdown defensemen from grading as elite drivers.
    """
    shares = ptv_share_by_role or DEFAULT_TO_PTV_SHARE
    micro = _transition_offense_micro_spec()
    out: dict[str, list[tuple[str, float]]] = {}
    for role in ("F", "D"):
        ptv = float(np.clip(float(shares.get(role, DEFAULT_TO_PTV_SHARE[role])), 0.0, 0.85))
        out[role] = [(f, w * (1.0 - ptv)) for f, w in micro] + [
            ("ptv_coalesced_rapm_possession_offense", ptv),
        ]
    return out


def _load_pcv_category_mix(
    path: Path | None,
) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    """Load tuned PCV mix for transition offense and shot defense."""
    to_share = dict(DEFAULT_TO_PTV_SHARE)
    sd_specs = {role: list(specs) for role, specs in DEFAULT_SHOT_DEFENSE_ROLE_SPECS.items()}
    if path is None or not path.is_file():
        return to_share, sd_specs
    raw = json.loads(path.read_text(encoding="utf-8"))
    to_raw = raw.get("transition_offense") or {}
    for role in ("F", "D"):
        payload = to_raw.get(role) or {}
        if "ptv_share" in payload:
            to_share[role] = float(payload["ptv_share"])
    sd_raw = raw.get("shot_defense") or {}
    for role in ("F", "D"):
        feats = (sd_raw.get(role) or {}).get("features")
        if not feats:
            continue
        parsed: list[tuple[str, float]] = []
        for item in feats:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                parsed.append((str(item[0]), float(item[1])))
        if parsed:
            sd_specs[role] = parsed
    return to_share, sd_specs


def _collect_category_feature_ownership(
    category_spec: dict[str, list[tuple[str, float]]],
    *,
    shot_creation_role_specs: dict[str, list[tuple[str, float]]],
    transition_offense_role_specs: dict[str, list[tuple[str, float]]],
    transition_defense_role_specs: dict[str, list[tuple[str, float]]],
    playmaking_role_specs: dict[str, list[tuple[str, float]]],
    shot_defense_role_specs: dict[str, list[tuple[str, float]]],
    finishing_role_specs: dict[str, list[tuple[str, float]]],
    skating_off_role_specs: dict[str, list[tuple[str, float]]],
    skating_def_role_specs: dict[str, list[tuple[str, float]]],
) -> dict[str, list[str]]:
    """Map each category to the union of features used across roles."""
    out: dict[str, set[str]] = {}

    def _add(cat: str, specs: list[tuple[str, float]]) -> None:
        out.setdefault(cat, set()).update(f for f, _ in specs)

    for cat, specs in category_spec.items():
        _add(cat, specs)
    for specs in shot_creation_role_specs.values():
        _add("cat_shot_creation", specs)
    for specs in transition_offense_role_specs.values():
        _add("cat_transition_offense", specs)
    for specs in transition_defense_role_specs.values():
        _add("cat_transition_defense", specs)
    for specs in playmaking_role_specs.values():
        _add("cat_playmaking", specs)
    for specs in shot_defense_role_specs.values():
        _add("cat_shot_defense", specs)
    for specs in finishing_role_specs.values():
        _add("cat_finishing", specs)
    for role_specs in FORECHECK_PHYSICALITY_ROLE_SPECS.values():
        for cat, specs in role_specs.items():
            _add(cat, specs)
    for specs in skating_off_role_specs.values():
        _add("cat_skating", specs)
    for specs in skating_def_role_specs.values():
        out.setdefault("cat_skating", set()).update(f for f, _ in specs)
    return {cat: sorted(feats) for cat, feats in out.items()}


def _rapm_anchor_shot_defense_raw(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    blend: float,
) -> None:
    """For F only: blend shot-defense raw toward RAPM xGA/CA skill."""
    b = float(np.clip(float(blend), 0.0, 1.0))
    if b <= 0.0:
        return
    raw = pd.to_numeric(df.get("cat_shot_defense_raw"), errors="coerce")
    rapm_z = _rapm_defensive_skill_z(df, group_cols)
    is_f = df["position_group"].astype(str).str.upper().str.strip().eq("F")
    out = raw.copy()
    m = is_f & raw.notna() & rapm_z.notna()
    out.loc[m] = (1.0 - b) * raw.loc[m] + b * rapm_z.loc[m]
    df["cat_shot_defense_raw"] = out
    df["cat_shot_defense_z"] = _z_by_group(df, "cat_shot_defense_raw", group_cols)
    df["cat_shot_defense_pct"] = _pct_by_group(df, "cat_shot_defense_z", group_cols)


def _rapm_defensive_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """On-ice xGA-only skill (CA can diverge from xGA and inflate micro-heavy D)."""
    xga = pd.to_numeric(df.get("rapm_hybrid_xGA/60"), errors="coerce")
    tmp = df[group_cols].copy()
    tmp["rapm_def_skill"] = -xga
    return _z_by_group(tmp, "rapm_def_skill", group_cols)


def _rapm_offensive_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    xgf = pd.to_numeric(df.get("rapm_hybrid_xGF/60"), errors="coerce")
    xgp = pd.to_numeric(df.get("rapm_hybrid_xG±/60"), errors="coerce")
    rapm_skill = pd.concat([xgf, xgp], axis=1).mean(axis=1, skipna=True)
    tmp = df[group_cols].copy()
    tmp["rapm_off_skill"] = rapm_skill
    return _z_by_group(tmp, "rapm_off_skill", group_cols)


def _rapm_offensive_skill_z_for_d(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """D offensive skill: individual A3Z micro only (on-ice xGF is too noisy)."""
    return _d_individual_offense_micro_z(df, group_cols)


def _d_transition_offense_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    del group_cols
    return _d_weighted_micro_pool_z(
        df,
        (
            ("a3z_entries_per_60", 0.14),
            ("a3z_carries_per_60", 0.14),
            ("a3z_entries_with_passing_play_per_60", 0.16),
            ("a3z_controlled_exits_per_60", 0.20),
            ("a3z_carried_exits_per_60", 0.12),
            ("a3z_passed_exits_per_60", 0.10),
            ("a3z_failed_entries_per_entry", -0.14),
        ),
    )


def _d_playmaking_offense_skill_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    del group_cols
    return _d_weighted_micro_pool_z(
        df,
        (
            ("a3z_expected_assists_proxy_per_60_shrunk", 0.60),
            ("a3z_playmaking_quality_proxy_per_60", 0.40),
        ),
    )


def _micro_anchor_category_raw_for_d(
    df: pd.DataFrame,
    *,
    cat: str,
    skill_z: pd.Series,
    group_cols: list[str],
    blend: float,
) -> None:
    """Blend D category raw toward an individual micro skill z.

    Low-TOI seasons get a modest extra pull toward the skill anchor so
    incomplete/stale samples do not dominate the raw bar.
    """
    b = float(np.clip(float(blend), 0.0, 1.0))
    if b <= 0.0:
        return
    raw_col = f"{cat}_raw"
    raw = pd.to_numeric(df.get(raw_col), errors="coerce")
    is_d = df["position_group"].astype(str).str.upper().str.strip().eq("D")
    out = raw.copy()
    m = is_d & raw.notna() & skill_z.notna()
    if not m.any():
        return
    toi = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
    if toi.isna().all():
        toi = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce")
    # Reliability: full blend at high TOI; up to +0.12 toward skill when TOI is thin.
    rel = (toi.fillna(0.0) / (toi.fillna(0.0) + 450.0)).clip(0.0, 1.0)
    b_eff = (b + (1.0 - b) * (1.0 - rel) * 0.12).clip(upper=0.92)
    out.loc[m] = (1.0 - b_eff.loc[m]) * raw.loc[m] + b_eff.loc[m] * skill_z.loc[m]
    df[raw_col] = out
    df[f"{cat}_z"] = _z_by_group(df, raw_col, group_cols)
    df[f"{cat}_pct"] = _pct_by_group(df, f"{cat}_z", group_cols)


def _rapm_anchor_category_raw_for_d(
    df: pd.DataFrame,
    *,
    cat: str,
    group_cols: list[str],
    blend: float,
    offensive: bool,
) -> None:
    """Blend category raw toward micro offense or RAPM defense skill for D only."""
    if offensive:
        skill_z = _rapm_offensive_skill_z_for_d(df, group_cols)
        _micro_anchor_category_raw_for_d(
            df, cat=cat, skill_z=skill_z, group_cols=group_cols, blend=blend
        )
        return
    b = float(np.clip(float(blend), 0.0, 1.0))
    if b <= 0.0:
        return
    raw_col = f"{cat}_raw"
    raw = pd.to_numeric(df.get(raw_col), errors="coerce")
    rapm_z = _rapm_defensive_skill_z(df, group_cols)
    is_d = df["position_group"].astype(str).str.upper().str.strip().eq("D")
    out = raw.copy()
    m = is_d & raw.notna() & rapm_z.notna()
    if not m.any():
        return
    toi = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
    if toi.isna().all():
        toi = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce")
    rel = (toi.fillna(0.0) / (toi.fillna(0.0) + 450.0)).clip(0.0, 1.0)
    # Thin samples shrink harder toward RAPM (results) — stale/injury awareness.
    b_eff = (b + (1.0 - b) * (1.0 - rel) * 0.18).clip(upper=0.85)
    out.loc[m] = (1.0 - b_eff.loc[m]) * raw.loc[m] + b_eff.loc[m] * rapm_z.loc[m]
    df[raw_col] = out
    df[f"{cat}_z"] = _z_by_group(df, raw_col, group_cols)
    df[f"{cat}_pct"] = _pct_by_group(df, f"{cat}_z", group_cols)


def _d_micro_feature_z(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    tmp = df[group_cols].copy()
    tmp["micro_part"] = pd.to_numeric(df[col], errors="coerce")
    return _z_by_group(tmp, "micro_part", group_cols)


def _d_offense_micro_pool_z(df: pd.DataFrame, col: str) -> pd.Series:
    """Pooled D offense micro z across all seasons (not within-season re-ranking)."""
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_d = pos.eq("D")
    x = pd.to_numeric(df.get(col), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    sub = x[is_d]
    if int(sub.notna().sum()) < 2:
        return out
    mu = float(sub.mean())
    sd = float(sub.std())
    if sd <= 0.0:
        out.loc[is_d] = 0.0
    else:
        out.loc[is_d] = (x[is_d] - mu) / sd
    return out


def _d_weighted_micro_pool_z(df: pd.DataFrame, specs: tuple[tuple[str, float], ...]) -> pd.Series:
    parts: list[pd.Series] = []
    weights: list[float] = []
    for col, w in specs:
        z = _d_offense_micro_pool_z(df, col)
        if z.notna().any():
            parts.append(z)
            weights.append(float(w))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    mat = pd.concat(parts, axis=1)
    w_arr = np.asarray(weights, dtype=float)
    vals = mat.to_numpy(float)
    mask = np.isfinite(vals)
    num = np.nansum(np.where(mask, vals * w_arr, np.nan), axis=1)
    den = np.sum(np.where(mask, w_arr, 0.0), axis=1)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    m = den > 0.0
    out.loc[m] = num[m] / den[m]
    return out


def _transition_offense_group_micro_z(df: pd.DataFrame) -> pd.Series:
    cols = (
        "a3z_entries_per_60",
        "a3z_carries_per_60",
        "a3z_entries_with_passing_play_per_60",
        "a3z_controlled_exits_per_60",
    )
    parts = [_to_num(df.get(f"z__{c}")) for c in cols if f"z__{c}" in df.columns]
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _d_individual_offense_micro_z(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """D individual offensive micro skill (entry creation + pre-shot playmaking)."""
    del group_cols  # pooled signal intentionally ignores season grouping
    return _d_weighted_micro_pool_z(
        df,
        (
            ("a3z_entries_per_60", 0.12),
            ("a3z_carries_per_60", 0.12),
            ("a3z_entries_with_passing_play_per_60", 0.14),
            ("a3z_controlled_exits_per_60", 0.16),
            ("a3z_expected_assists_proxy_per_60_shrunk", 0.28),
            ("a3z_playmaking_quality_proxy_per_60", 0.18),
        ),
    )


def _cap_d_offense_for_elite_suppressors(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    shot_def_z_floor: float = 0.85,
    elite_off_micro_z: float = 0.85,
    max_above_entry: float = 0.06,
    max_above_micro: float = 0.10,
) -> None:
    """
    Shutdown D should not grade as elite transition/playmaking drivers.

    Cap toward individual micro skill; elite offensive D bypass via micro only.
    """
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_d = pos.eq("D")
    sd = pd.to_numeric(df.get("cat_shot_defense_z"), errors="coerce")
    micro_off = _d_individual_offense_micro_z(df, group_cols)
    entry_off = _d_transition_offense_skill_z(df, group_cols)
    pm_off = _d_playmaking_offense_skill_z(df, group_cols)
    elite_sd = is_d & sd.notna() & (sd >= float(shot_def_z_floor))
    elite_off = micro_off.notna() & (micro_off >= float(elite_off_micro_z))
    suppressor = elite_sd & ~elite_off
    for cat, skill, micro_pad in (
        ("cat_transition_offense", entry_off, max_above_entry),
        ("cat_playmaking", pm_off, max_above_micro),
    ):
        raw_col = f"{cat}_raw"
        if raw_col not in df.columns:
            continue
        raw = pd.to_numeric(df[raw_col], errors="coerce")
        pad = pd.Series(float(micro_pad), index=df.index, dtype=float)
        if cat == "cat_transition_offense":
            pad = pad.where(~(entry_off.notna() & (entry_off < -0.75)), 0.02)
            pad = pad.where(~(entry_off.notna() & (entry_off < -0.50) & (entry_off >= -0.75)), 0.04)
        cap = skill + pad
        m = suppressor & raw.notna() & cap.notna() & (raw > cap)
        if m.any():
            raw = raw.copy()
            raw.loc[m] = cap.loc[m]
            df[raw_col] = raw
            df[f"{cat}_z"] = _z_by_group(df, raw_col, group_cols)
            df[f"{cat}_pct"] = _pct_by_group(df, f"{cat}_z", group_cols)


def _cap_d_offense_for_negative_rapm(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    xgf_floor: float = -0.08,
    cap_ceiling_z: float = 0.65,
    pull_weight: float = 0.72,
) -> None:
    """
    Cap offensive categories for D with negative on-ice xGF/60 RAPM.

    Players like Sean Walker show high process metrics (entries, carries, shots)
    but negative actual on-ice impact. This prevents inflated offensive grades
    when RAPM disagrees with process metrics.
    """
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_d = pos.eq("D")
    xgf = pd.to_numeric(df.get("eh_rapm_xGF/60"), errors="coerce")
    negative_rapm = is_d & xgf.notna() & (xgf < float(xgf_floor))
    if not negative_rapm.any():
        return
    # Compute severity for all rows (with NaN where not applicable)
    severity = pd.Series(np.nan, index=df.index, dtype=float)
    severity.loc[negative_rapm] = ((-xgf.loc[negative_rapm] - float(xgf_floor)).clip(lower=0.0, upper=0.35) / 0.35)
    cap = float(cap_ceiling_z) - 0.35 * severity
    for cat in ("cat_transition_offense", "cat_shot_creation", "cat_playmaking"):
        raw_col = f"{cat}_raw"
        if raw_col not in df.columns:
            continue
        raw = pd.to_numeric(df[raw_col], errors="coerce")
        m = negative_rapm & raw.notna() & cap.notna() & (raw > cap)
        if m.any():
            raw = raw.copy()
            w = float(pull_weight)
            raw.loc[m] = (1.0 - w) * raw.loc[m] + w * cap.loc[m]
            df[raw_col] = raw
            df[f"{cat}_z"] = _z_by_group(df, raw_col, group_cols)
            df[f"{cat}_pct"] = _pct_by_group(df, f"{cat}_z", group_cols)


def _d_weak_entry_creation_mask(df: pd.DataFrame, *, cutoff: float = 0.0) -> pd.Series:
    """Below-average rush activators from direct transition-offense rate micros."""
    pos = df["position_group"].astype(str).str.upper().str.strip()
    parts = [
        pd.to_numeric(df.get(c), errors="coerce")
        for c in ("a3z_entries_per_60", "a3z_carries_per_60", "a3z_entries_with_passing_play_per_60")
        if c in df.columns
    ]
    entry_proxy = pd.concat(parts, axis=1).mean(axis=1, skipna=True) if parts else pd.Series(np.nan, index=df.index)
    return pos.eq("D") & entry_proxy.notna() & (entry_proxy < float(cutoff))


def _d_transition_offense_volume_discipline(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    entry_mod_cutoff: float = -0.15,
    residual_weight: float = 0.88,
    ridge_alpha: float = 2.0,
    min_group_rows: int = 12,
) -> None:
    """Strip touch-volume inflation from D TO when entry-creation micro is poor."""
    raw_col = "cat_transition_offense_raw"
    if raw_col not in df.columns:
        return
    pos = df["position_group"].astype(str).str.upper().str.strip()
    entry_micro = _d_transition_offense_skill_z(df, group_cols)
    weak = pos.eq("D") & entry_micro.notna() & (entry_micro < float(entry_mod_cutoff))
    if not weak.any():
        return
    for vol_col in ("a3z_entries_per_60", "a3z_carries_per_60"):
        zc = f"z__{vol_col}"
        if vol_col not in df.columns:
            df[vol_col] = np.nan
        if zc not in df.columns:
            df[zc] = _z_by_group(df, vol_col, group_cols)
    vol_z = pd.concat(
        [_to_num(df.get(f"z__{c}")) for c in ("a3z_entries_per_60", "a3z_carries_per_60")],
        axis=1,
    ).mean(axis=1, skipna=True)
    out = pd.to_numeric(df[raw_col], errors="coerce").copy()
    w = float(np.clip(float(residual_weight), 0.0, 1.0))
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))
    d_idx = df.index[weak]
    for _, idx in df.loc[d_idx].groupby(group_cols, dropna=False).groups.items():
        x = vol_z.loc[idx]
        y = pd.to_numeric(df.loc[idx, raw_col], errors="coerce")
        m = x.notna() & y.notna()
        if int(m.sum()) < mmin:
            continue
        xa = x[m].to_numpy(float)
        ya = y[m].to_numpy(float)
        if float(np.std(xa)) <= 0.0:
            resid = ya - float(np.mean(ya))
        else:
            beta = float(np.cov(xa, ya, ddof=0)[0, 1] / np.var(xa))
            if lam > 0.0:
                beta = beta / (1.0 + lam)
            fitted = float(np.mean(ya)) + beta * (xa - float(np.mean(xa)))
            resid = ya - fitted
        blend = (1.0 - w) * ya + w * (float(np.mean(ya)) + resid)
        out.loc[idx[m]] = blend
    df[raw_col] = out
    df["cat_transition_offense_z"] = _z_by_group(df, raw_col, group_cols)
    df["cat_transition_offense_pct"] = _pct_by_group(df, "cat_transition_offense_z", group_cols)


def _rapm_anchor_transition_offense_raw(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    blend: float,
) -> None:
    entry = _d_transition_offense_skill_z(df, group_cols)
    _micro_anchor_category_raw_for_d(
        df, cat="cat_transition_offense", skill_z=entry, group_cols=group_cols, blend=blend
    )
    raw_col = "cat_transition_offense_raw"
    raw = pd.to_numeric(df.get(raw_col), errors="coerce")
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_d = pos.eq("D")
    entry_micro = _d_transition_offense_skill_z(df, group_cols)
    m = is_d & entry.notna() & raw.notna() & entry_micro.notna() & (entry_micro < 0.0)
    if m.any():
        out = raw.copy()
        severity = (-entry_micro.loc[m]).clip(lower=0.0, upper=1.0)
        pull = 0.55 + 0.30 * severity
        target = entry.loc[m] + 0.02
        out.loc[m] = (1.0 - pull) * raw.loc[m] + pull * target
        df[raw_col] = out
    df["cat_transition_offense_z"] = _z_by_group(df, raw_col, group_cols)
    df["cat_transition_offense_pct"] = _pct_by_group(df, "cat_transition_offense_z", group_cols)


def _d_transition_offense_activation_inflation_guard(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    entry_pool_ceiling: float = 0.65,
    volume_z_floor: float = 0.25,
    pull_weight: float = 0.82,
) -> None:
    """Pull TO down for D who touch the puck a lot without pooled entry-creation skill."""
    raw_col = "cat_transition_offense_raw"
    if raw_col not in df.columns:
        return
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_d = pos.eq("D")
    entry_pool = _d_transition_offense_skill_z(df, group_cols)
    for vol_col in ("a3z_entries_per_60", "a3z_carries_per_60"):
        zc = f"z__{vol_col}"
        if vol_col not in df.columns:
            df[vol_col] = np.nan
        if zc not in df.columns:
            df[zc] = _z_by_group(df, vol_col, group_cols)
    vol_z = pd.concat(
        [_to_num(df.get(f"z__{c}")) for c in ("a3z_entries_per_60", "a3z_carries_per_60")],
        axis=1,
    ).mean(axis=1, skipna=True)
    raw = pd.to_numeric(df[raw_col], errors="coerce").copy()
    w = float(np.clip(float(pull_weight), 0.0, 1.0))
    m_extreme = is_d & entry_pool.notna() & vol_z.notna() & (vol_z > 2.0) & (entry_pool < 1.5)
    if m_extreme.any():
        raw.loc[m_extreme] = 0.08 * raw.loc[m_extreme] + 0.92 * entry_pool.loc[m_extreme]
    m = (
        is_d
        & entry_pool.notna()
        & vol_z.notna()
        & (entry_pool < float(entry_pool_ceiling))
        & (vol_z > float(volume_z_floor))
        & ~m_extreme
    )
    if m.any():
        target = entry_pool.loc[m] + 0.08
        raw.loc[m] = (1.0 - w) * raw.loc[m] + w * target.loc[m]
    if not (m_extreme.any() or m.any()):
        return
    df[raw_col] = raw
    df["cat_transition_offense_z"] = _z_by_group(df, raw_col, group_cols)
    df["cat_transition_offense_pct"] = _pct_by_group(df, "cat_transition_offense_z", group_cols)


def _f_transition_offense_activation_inflation_guard(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    volume_z_floor: float = 0.35,
    finishing_z_ceiling: float = -0.20,
    outcome_z_ceiling: float = 0.05,
    pull_weight: float = 0.78,
) -> None:
    """
    Pull F TO down when entry/carry volume is high but finishing and on-ice offense are not.

    Mirrors the D activation guard: touch volume without scoring impact should not read elite.
    """
    raw_col = "cat_transition_offense_raw"
    if raw_col not in df.columns:
        return
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_f = pos.eq("F")
    for vol_col in ("a3z_entries_per_60", "a3z_carries_per_60"):
        zc = f"z__{vol_col}"
        if vol_col not in df.columns:
            df[vol_col] = np.nan
        if zc not in df.columns:
            df[zc] = _z_by_group(df, vol_col, group_cols)
    vol_z = pd.concat(
        [_to_num(df.get(f"z__{c}")) for c in ("a3z_entries_per_60", "a3z_carries_per_60")],
        axis=1,
    ).mean(axis=1, skipna=True)
    fin_parts = [
        _to_num(df.get(c))
        for c in (
            "z__eh_finishing_goals_above_ixg_per_60_shrunk",
            "z__eh_finishing_sh_pct_above_xsh_shrunk",
        )
        if c in df.columns
    ]
    fin_proxy = (
        pd.concat(fin_parts, axis=1).mean(axis=1, skipna=True)
        if fin_parts
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    outcome_parts = [
        _to_num(df.get(c))
        for c in (
            "z__ptv_coalesced_rapm_possession_offense",
            "z__eh_rapm_xG±/60",
            "z__rapm_hybrid_xGF/60",
        )
        if c in df.columns
    ]
    outcome_proxy = (
        pd.concat(outcome_parts, axis=1).mean(axis=1, skipna=True)
        if outcome_parts
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    entry_micro = _transition_offense_group_micro_z(df)
    entry_pool = entry_micro.where(entry_micro.notna(), outcome_proxy)
    raw = pd.to_numeric(df[raw_col], errors="coerce").copy()
    w = float(np.clip(float(pull_weight), 0.0, 1.0))
    m_weak_outcome = (
        is_f
        & vol_z.notna()
        & fin_proxy.notna()
        & outcome_proxy.notna()
        & entry_pool.notna()
        & (vol_z > float(volume_z_floor))
        & (fin_proxy < float(finishing_z_ceiling))
        & (outcome_proxy < float(outcome_z_ceiling))
    )
    m_poor_finishing = (
        is_f
        & vol_z.notna()
        & fin_proxy.notna()
        & entry_pool.notna()
        & (vol_z > float(volume_z_floor))
        & (fin_proxy < -0.30)
    )
    m = m_weak_outcome | m_poor_finishing
    if not m.any():
        return
    outcome_for_sev = outcome_proxy.where(outcome_proxy.notna(), 0.0)
    severity = (
        (vol_z.loc[m] - float(volume_z_floor)).clip(lower=0.0, upper=1.5) / 1.5
        + (-fin_proxy.loc[m]).clip(lower=0.0, upper=1.2)
        + (-outcome_for_sev.loc[m]).clip(lower=0.0, upper=1.0)
    ) / 3.0
    target_weak = entry_pool.loc[m] + 0.02 - 0.55 * severity
    skill_anchor = (
        0.55 * outcome_for_sev.loc[m]
        + 0.45 * fin_proxy.loc[m]
        + 0.10
    )
    target_fin = skill_anchor - 0.45 * severity
    target = target_weak.copy()
    if m_poor_finishing.any():
        target.loc[m & m_poor_finishing] = target_fin.loc[m & m_poor_finishing]
    raw.loc[m] = (1.0 - w) * raw.loc[m] + w * target
    df[raw_col] = raw
    df["cat_transition_offense_z"] = _z_by_group(df, raw_col, group_cols)
    df["cat_transition_offense_pct"] = _pct_by_group(df, "cat_transition_offense_z", group_cols)


def _f_shot_creation_finishing_guard(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    volume_z_floor: float = 0.30,
    finishing_z_ceiling: float = -0.20,
    outcome_z_ceiling: float = 0.10,
    pull_weight: float = 0.65,
) -> None:
    """Pull F shot creation down when chance/shot volume is high but finishing is not."""
    raw_col = "cat_shot_creation_raw"
    if raw_col not in df.columns:
        return
    pos = df["position_group"].astype(str).str.upper().str.strip()
    is_f = pos.eq("F")
    for vol_col in ("eh_std_iCF/60", "a3z_chances_per_60", "eh_std_ixG/60"):
        zc = f"z__{vol_col}"
        if vol_col not in df.columns:
            df[vol_col] = np.nan
        if zc not in df.columns:
            df[zc] = _z_by_group(df, vol_col, group_cols)
    vol_z = pd.concat(
        [_to_num(df.get(f"z__{c}")) for c in ("eh_std_iCF/60", "a3z_chances_per_60", "eh_std_ixG/60")],
        axis=1,
    ).mean(axis=1, skipna=True)
    fin_parts = [
        _to_num(df.get(c))
        for c in (
            "z__eh_finishing_goals_above_ixg_per_60_shrunk",
            "z__eh_finishing_sh_pct_above_xsh_shrunk",
            "z__eh_finishing_g_per_shot_above_xg_per_shot_shrunk",
        )
        if c in df.columns
    ]
    fin_proxy = (
        pd.concat(fin_parts, axis=1).mean(axis=1, skipna=True)
        if fin_parts
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    outcome_parts = [
        _to_num(df.get(c))
        for c in (
            "z__rapm_hybrid_xGF/60",
            "z__ptv_coalesced_rapm_shot_threat",
            "z__eh_rapm_xG±/60",
        )
        if c in df.columns
    ]
    outcome_proxy = (
        pd.concat(outcome_parts, axis=1).mean(axis=1, skipna=True)
        if outcome_parts
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    raw = pd.to_numeric(df[raw_col], errors="coerce").copy()
    w = float(np.clip(float(pull_weight), 0.0, 1.0))
    m_weak_outcome = (
        is_f
        & vol_z.notna()
        & fin_proxy.notna()
        & outcome_proxy.notna()
        & (vol_z > float(volume_z_floor))
        & (fin_proxy < float(finishing_z_ceiling))
        & (outcome_proxy < float(outcome_z_ceiling))
    )
    m_poor_finishing = (
        is_f
        & vol_z.notna()
        & fin_proxy.notna()
        & (vol_z > float(volume_z_floor))
        & (fin_proxy < -0.32)
    )
    m = m_weak_outcome | m_poor_finishing
    if not m.any():
        return
    outcome_for_sev = outcome_proxy.where(outcome_proxy.notna(), 0.0)
    severity = (
        (vol_z.loc[m] - float(volume_z_floor)).clip(lower=0.0, upper=1.5) / 1.5
        + (-fin_proxy.loc[m]).clip(lower=0.0, upper=1.0)
        + (-outcome_for_sev.loc[m]).clip(lower=0.0, upper=0.8)
    ) / 3.0
    skill_anchor = 0.60 * fin_proxy.loc[m] + 0.40 * outcome_for_sev.loc[m] + 0.12
    target = skill_anchor - 0.38 * severity
    raw.loc[m] = (1.0 - w) * raw.loc[m] + w * target
    df[raw_col] = raw
    df["cat_shot_creation_z"] = _z_by_group(df, raw_col, group_cols)
    df["cat_shot_creation_pct"] = _pct_by_group(df, "cat_shot_creation_z", group_cols)


def _rapm_anchor_playmaking_raw(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    blend: float,
) -> None:
    pm_skill = _d_playmaking_offense_skill_z(df, group_cols)
    _micro_anchor_category_raw_for_d(
        df, cat="cat_playmaking", skill_z=pm_skill, group_cols=group_cols, blend=blend
    )


def _rapm_anchor_transition_defense_raw(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    blend: float,
    overshoot_gap_z: float = 0.28,
) -> None:
    """For D only: cap inflated exit/breakout defense when RAPM suppression strongly disagrees.

    Transition defense should stay an exit/breakout skill component.  RAPM xGA is
    already owned heavily by shot defense, so use it here only as an inflation
    guard instead of a general anchor.
    """
    b = float(np.clip(float(blend), 0.0, 1.0))
    if b <= 0.0:
        return
    raw = pd.to_numeric(df.get("cat_transition_defense_raw"), errors="coerce")
    rapm_z = _rapm_defensive_skill_z(df, group_cols)
    is_d = df["position_group"].astype(str).str.upper().str.strip().eq("D")
    out = raw.copy()
    m = is_d & raw.notna() & rapm_z.notna() & (raw > rapm_z + float(overshoot_gap_z))
    if not m.any():
        return
    toi = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
    if toi.isna().all():
        toi = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce")
    rel = (toi.fillna(0.0) / (toi.fillna(0.0) + 450.0)).clip(0.0, 1.0)
    b_eff = (b + (1.0 - b) * (1.0 - rel) * 0.15).clip(upper=0.80)
    out.loc[m] = (1.0 - b_eff.loc[m]) * raw.loc[m] + b_eff.loc[m] * rapm_z.loc[m]
    df["cat_transition_defense_raw"] = out
    df["cat_transition_defense_z"] = _z_by_group(df, "cat_transition_defense_raw", group_cols)
    df["cat_transition_defense_pct"] = _pct_by_group(df, "cat_transition_defense_z", group_cols)


def _stale_season_shrink_toward_anchor(
    df: pd.DataFrame,
    *,
    cat: str,
    anchor_z: pd.Series,
    group_cols: list[str],
    roles: tuple[str, ...] = ("F",),
    toi_ratio_floor: float = 0.62,
    max_pull: float = 0.20,
) -> None:
    """
    Modest stale/injury awareness: when current TOI falls vs prior season,
    pull category raw toward a results/possession anchor (no cliffs).
    """
    raw_col = f"{cat}_raw"
    if raw_col not in df.columns:
        return
    pos = df["position_group"].astype(str).str.upper().str.strip()
    role_m = pos.isin([r.upper() for r in roles])
    toi = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
    if toi.isna().all():
        toi = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce")
    pid = pd.to_numeric(df.get("player_id"), errors="coerce")
    season = pd.to_numeric(df.get("season_id"), errors="coerce")
    order = np.lexsort((season.to_numpy(), pid.to_numpy()))
    prior_toi = pd.Series(np.nan, index=df.index, dtype=float)
    toi_arr = toi.to_numpy(dtype=float)
    pid_arr = pid.to_numpy()
    prev_pid = None
    prev_toi_v = np.nan
    for idx in order:
        cur_pid = pid_arr[idx]
        if prev_pid is not None and cur_pid == prev_pid and np.isfinite(prev_toi_v):
            prior_toi.iloc[idx] = prev_toi_v
        prev_pid = cur_pid
        prev_toi_v = toi_arr[idx]
    ratio = toi / prior_toi.replace(0.0, np.nan)
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    m = (
        role_m
        & raw.notna()
        & anchor_z.notna()
        & ratio.notna()
        & prior_toi.notna()
        & (prior_toi >= 250.0)
        & (ratio < float(toi_ratio_floor))
    )
    if not m.any():
        return
    severity = ((float(toi_ratio_floor) - ratio.loc[m]) / float(toi_ratio_floor)).clip(0.0, 1.0)
    pull = float(max_pull) * severity
    out = raw.copy()
    out.loc[m] = (1.0 - pull) * raw.loc[m] + pull * anchor_z.loc[m]
    df[raw_col] = out
    df[f"{cat}_z"] = _z_by_group(df, raw_col, group_cols)
    df[f"{cat}_pct"] = _pct_by_group(df, f"{cat}_z", group_cols)


def _build_role_hybrid_category(
    df: pd.DataFrame,
    *,
    cat: str,
    role_specs: dict[str, list[tuple[str, float]]],
    group_cols: list[str],
) -> None:
    """Fill ``{cat}_raw/z/pct/n`` using different feature weights per position_group."""
    raw_col = f"{cat}_raw"
    n_col = f"{cat}_n_features"
    z_col = f"{cat}_z"
    pct_col = f"{cat}_pct"
    df[raw_col] = np.nan
    df[n_col] = 0.0
    pos = df["position_group"].astype(str).str.upper().str.strip()
    fallback_role = "F" if "F" in role_specs else next(iter(role_specs))

    def _apply_role(role: str, idx: pd.Index) -> None:
        if len(idx) == 0:
            return
        specs = role_specs[role]
        sub = df.loc[idx].copy()
        for feat, _ in specs:
            if feat not in sub.columns:
                sub[feat] = np.nan
            zc = f"z__{feat}"
            if zc not in df.columns:
                df[zc] = _z_by_group(df, feat, group_cols)
            elif zc not in sub.columns:
                sub[zc] = df.loc[idx, zc]
        z_specs = [(f"z__{f}", w) for f, w in specs]
        raw, n_used = _weighted_mean_from_z(sub, z_specs)
        df.loc[idx, raw_col] = raw
        df.loc[idx, n_col] = n_used

    for role in role_specs:
        _apply_role(role, df.index[pos == role])
    idx_other = df.index[~pos.isin(list(role_specs.keys()))]
    _apply_role(fallback_role, idx_other)
    df[z_col] = _z_by_group(df, raw_col, group_cols)
    df[pct_col] = _pct_by_group(df, z_col, group_cols)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _z_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    x = _to_num(df[col])
    g = df[group_cols]
    mu = x.groupby([g[c] for c in group_cols], dropna=False).transform("mean")
    sd = x.groupby([g[c] for c in group_cols], dropna=False).transform("std")
    z = (x - mu) / sd.replace(0.0, np.nan)
    return z


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


def _weighted_mean_from_z(df: pd.DataFrame, z_cols: list[tuple[str, float]]) -> tuple[pd.Series, pd.Series]:
    vals = []
    ws = []
    for col, w in z_cols:
        v = _to_num(df[col])
        vals.append(v.to_numpy(dtype=float))
        ws.append(float(w))
    vmat = np.vstack(vals).T if vals else np.zeros((len(df), 0), dtype=float)
    wvec = np.asarray(ws, dtype=float)

    finite = np.isfinite(vmat)
    weighted_sum = np.nansum(np.where(finite, vmat * wvec[None, :], 0.0), axis=1)
    weight_sum = np.sum(np.where(finite, np.abs(wvec[None, :]), 0.0), axis=1)
    score = np.full(len(df), np.nan, dtype=float)
    m = weight_sum > 0.0
    score[m] = weighted_sum[m] / weight_sum[m]
    n_used = np.sum(finite, axis=1).astype(float)
    return pd.Series(score, index=df.index, dtype=float), pd.Series(n_used, index=df.index, dtype=float)


def _corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 8:
        return float("nan")
    xa = x[m].to_numpy(float)
    ya = y[m].to_numpy(float)
    if float(np.std(xa)) <= 0.0 or float(np.std(ya)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def _deoverlap_shot_creation_vs_transition_offense(
    df: pd.DataFrame,
    *,
    enabled: bool,
    residual_weight: float | dict[str, float],
    ridge_alpha: float,
    min_group_rows: int,
    group_cols: list[str],
    roles: tuple[str, ...] = ("F", "D"),
) -> dict[str, float]:
    """
    Strip transition-activation overlap out of shot creation.

    Rush/exit offense belongs in transition offense; shot creation should reflect
    individual shot threat, not activation again.
    """
    role_weights = (
        residual_weight
        if isinstance(residual_weight, dict)
        else {r: float(residual_weight) for r in roles}
    )
    info: dict[str, float] = {
        "enabled": float(enabled),
        "groups_used": 0.0,
        "rows_updated": 0.0,
    }
    if not enabled:
        return info
    if "cat_shot_creation_raw" not in df.columns or "cat_transition_offense_raw" not in df.columns:
        return info

    pos = df["position_group"].astype(str).str.upper().str.strip()
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))
    out = pd.to_numeric(df["cat_shot_creation_raw"], errors="coerce").copy()
    df["cat_shot_creation_raw__pre_transition_residual"] = out

    for role in roles:
        w = float(np.clip(float(role_weights.get(role, 0.75)), 0.0, 1.0))
        role_idx = df.index[pos.eq(role)]
        if len(role_idx) == 0:
            continue
        y0 = pd.to_numeric(df.loc[role_idx, "cat_shot_creation_raw"], errors="coerce")
        x0 = pd.to_numeric(df.loc[role_idx, "cat_transition_offense_raw"], errors="coerce")
        info[f"corr_before_{role}"] = _corr(y0, x0)

        for _, idx in df.loc[role_idx].groupby(group_cols, dropna=False).groups.items():
            x = pd.to_numeric(df.loc[idx, "cat_transition_offense_raw"], errors="coerce")
            y = pd.to_numeric(df.loc[idx, "cat_shot_creation_raw"], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < mmin:
                continue
            xa = x[m].to_numpy(float)
            ya = y[m].to_numpy(float)
            if float(np.std(xa)) <= 0.0:
                resid = ya - float(np.mean(ya))
            else:
                xmean = float(np.mean(xa))
                ymean = float(np.mean(ya))
                beta = float(np.cov(xa, ya, ddof=0)[0, 1] / np.var(xa))
                alpha = ymean - beta * xmean
                if lam > 0.0:
                    beta = beta / (1.0 + lam)
                fitted = alpha + beta * xa
                resid = ya - fitted
            blend = (1.0 - w) * ya + w * (float(np.mean(ya)) + resid)
            out.loc[idx[m]] = blend
            info["groups_used"] = float(info["groups_used"]) + 1.0
            info["rows_updated"] = float(info["rows_updated"]) + float(m.sum())

        y1 = pd.to_numeric(out.loc[role_idx], errors="coerce")
        info[f"corr_after_{role}"] = _corr(y1, x0)

    df["cat_shot_creation_raw"] = out
    return info


def _deoverlap_shot_defense_vs_transition_defense(
    df: pd.DataFrame,
    *,
    enabled: bool,
    residual_weight: float | dict[str, float],
    ridge_alpha: float,
    min_group_rows: int,
    group_cols: list[str],
    roles: tuple[str, ...] = ("F", "D"),
) -> dict[str, float]:
    """Make shot defense the in-zone layer left after transition defense.

    Entry denial, retrieval and exit work belongs to transition defense. Shot
    defense retains the independent suppression signal after that upstream
    process is accounted for.
    """
    info: dict[str, float] = {"enabled": float(enabled), "groups_used": 0.0, "rows_updated": 0.0}
    if not enabled:
        return info
    if "cat_shot_defense_raw" not in df.columns or "cat_transition_defense_raw" not in df.columns:
        return info
    role_weights = residual_weight if isinstance(residual_weight, dict) else {
        r: float(residual_weight) for r in roles
    }
    pos = df["position_group"].astype(str).str.upper().str.strip()
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))
    out = pd.to_numeric(df["cat_shot_defense_raw"], errors="coerce").copy()
    df["cat_shot_defense_raw__pre_transition_residual"] = out
    for role in roles:
        w = float(np.clip(float(role_weights.get(role, 0.0)), 0.0, 1.0))
        role_idx = df.index[pos.eq(role)]
        x0 = pd.to_numeric(df.loc[role_idx, "cat_transition_defense_raw"], errors="coerce")
        y0 = pd.to_numeric(df.loc[role_idx, "cat_shot_defense_raw"], errors="coerce")
        info[f"corr_before_{role}"] = _corr(y0, x0)
        for _, idx in df.loc[role_idx].groupby(group_cols, dropna=False).groups.items():
            x = pd.to_numeric(df.loc[idx, "cat_transition_defense_raw"], errors="coerce")
            y = pd.to_numeric(df.loc[idx, "cat_shot_defense_raw"], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < mmin:
                continue
            xs, ys = x[m].to_numpy(float), y[m].to_numpy(float)
            mx, my = float(np.mean(xs)), float(np.mean(ys))
            xv, yv = xs - mx, ys - my
            denom = float(np.sum(xv * xv) + lam)
            if denom <= 1e-9 or not np.isfinite(denom):
                continue
            beta = float(np.sum(xv * yv) / denom)
            independent = my + (ys - (my + beta * (xs - mx)))
            out.loc[x[m].index] = (1.0 - w) * ys + w * independent
            info["groups_used"] += 1.0
            info["rows_updated"] += float(m.sum())
        info[f"corr_after_{role}"] = _corr(out.loc[role_idx], x0)
    df["cat_shot_defense_raw"] = out
    return info


def _deoverlap_physicality_vs_forecheck(
    df: pd.DataFrame,
    *,
    enabled: bool,
    residual_weight: float,
    ridge_alpha: float,
    min_group_rows: int,
    group_cols: list[str],
) -> dict[str, float]:
    """
    Residualize physicality raw signal against forecheck raw signal within each group.

    This preserves the independent part of physicality and reduces conceptual overlap.
    """
    info = {
        "enabled": bool(enabled),
        "groups_used": 0.0,
        "rows_updated": 0.0,
        "corr_before": float("nan"),
        "corr_after": float("nan"),
    }
    if not enabled:
        return info
    if "cat_physicality_raw" not in df.columns or "cat_forecheck_raw" not in df.columns:
        return info

    y0 = pd.to_numeric(df["cat_physicality_raw"], errors="coerce")
    x0 = pd.to_numeric(df["cat_forecheck_raw"], errors="coerce")
    info["corr_before"] = _corr(y0, x0)

    w = float(np.clip(float(residual_weight), 0.0, 1.0))
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))
    out = y0.copy()
    df["cat_physicality_raw__pre_forecheck_residual"] = out

    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        x = pd.to_numeric(df.loc[idx, "cat_forecheck_raw"], errors="coerce")
        y = pd.to_numeric(df.loc[idx, "cat_physicality_raw"], errors="coerce")
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < mmin:
            continue
        xs = x[m].to_numpy(float)
        ys = y[m].to_numpy(float)
        mx = float(np.mean(xs))
        my = float(np.mean(ys))
        xv = xs - mx
        yv = ys - my
        denom = float(np.sum(xv * xv) + lam)
        if not np.isfinite(denom) or denom <= 1e-9:
            continue
        beta = float(np.sum(xv * yv) / denom)
        pred = my + beta * (xs - mx)
        resid = ys - pred
        y_new = (1.0 - w) * ys + w * resid
        out.loc[x[m].index] = y_new
        info["groups_used"] += 1.0
        info["rows_updated"] += float(n)

    df["cat_physicality_raw"] = out
    info["corr_after"] = _corr(df["cat_physicality_raw"], df["cat_forecheck_raw"])
    return info


def _deoverlap_skating_vs_transition_offense(
    df: pd.DataFrame,
    *,
    enabled: bool,
    residual_weight: float | dict[str, float],
    ridge_alpha: float,
    min_group_rows: int,
    group_cols: list[str],
    roles: tuple[str, ...] = ("D",),
) -> dict[str, float]:
    """
    Remove puck-transition overlap from skating raw signal.

    A3Z mobility pillars are useful skating proxies, but for defensemen they can
    partly become "who carries/exits the puck" again.  This preserves the group
    mean and the independent skating variation while reducing transition double
    counting.
    """
    info: dict[str, float] = {
        "enabled": float(enabled),
        "groups_used": 0.0,
        "rows_updated": 0.0,
    }
    if not enabled:
        return info
    if "cat_skating_raw" not in df.columns or "cat_transition_offense_raw" not in df.columns:
        return info

    role_weights = residual_weight if isinstance(residual_weight, dict) else {
        r: float(residual_weight) for r in roles
    }
    lam = float(max(ridge_alpha, 0.0))
    mmin = int(max(min_group_rows, 8))
    pos = df["position_group"].astype(str).str.upper().str.strip()
    out = pd.to_numeric(df["cat_skating_raw"], errors="coerce").copy()
    df["cat_skating_raw__pre_transition_residual"] = out

    for role in roles:
        w = float(np.clip(float(role_weights.get(role, 0.0)), 0.0, 1.0))
        if w <= 0.0:
            continue
        role_idx = df.index[pos.eq(role)]
        if len(role_idx) == 0:
            continue
        y0 = pd.to_numeric(df.loc[role_idx, "cat_skating_raw"], errors="coerce")
        x0 = pd.to_numeric(df.loc[role_idx, "cat_transition_offense_raw"], errors="coerce")
        info[f"corr_before_{role}"] = _corr(y0, x0)

        for _, idx in df.loc[role_idx].groupby(group_cols, dropna=False).groups.items():
            x = pd.to_numeric(df.loc[idx, "cat_transition_offense_raw"], errors="coerce")
            y = pd.to_numeric(df.loc[idx, "cat_skating_raw"], errors="coerce")
            m = x.notna() & y.notna()
            if int(m.sum()) < mmin:
                continue
            xs = x[m].to_numpy(float)
            ys = y[m].to_numpy(float)
            mx = float(np.mean(xs))
            my = float(np.mean(ys))
            xv = xs - mx
            yv = ys - my
            denom = float(np.sum(xv * xv) + lam)
            if not np.isfinite(denom) or denom <= 1e-9:
                continue
            beta = float(np.sum(xv * yv) / denom)
            pred = my + beta * (xs - mx)
            resid_centered = ys - pred
            y_new = (1.0 - w) * ys + w * (my + resid_centered)
            out.loc[x[m].index] = y_new
            info["groups_used"] = float(info["groups_used"]) + 1.0
            info["rows_updated"] = float(info["rows_updated"]) + float(m.sum())

        y1 = pd.to_numeric(out.loc[role_idx], errors="coerce")
        info[f"corr_after_{role}"] = _corr(y1, x0)

    df["cat_skating_raw"] = out
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/pcv/player_season_feature_table_all_in_one.csv",
    )
    ap.add_argument(
        "--edge-table",
        type=Path,
        default=DEFAULT_EDGE_TABLE,
        help="Sidecar CSV with NHL EDGE columns merged before skating category build.",
    )
    ap.add_argument(
        "--skating-weights-json",
        type=Path,
        default=DEFAULT_SKATING_WEIGHTS_JSON,
        help="JSON with role-specific skating component weights (speed/burst/engine/mobility).",
    )
    ap.add_argument(
        "--shot-creation-weights-json",
        type=Path,
        default=DEFAULT_SHOT_CREATION_WEIGHTS_JSON,
        help="JSON with shot-creation component weights (threat/process blend).",
    )
    ap.add_argument(
        "--pcv-weights-json",
        type=Path,
        default=DEFAULT_PCV_WEIGHTS_JSON,
        help="JSON with tuned PCV mix for transition offense and shot defense.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores.csv",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores.parquet",
    )
    ap.add_argument(
        "--deoverlap-physicality-forecheck",
        dest="deoverlap_physicality_forecheck",
        action="store_true",
        default=True,
        help="Residualize Physicality vs Forecheck within season/position groups (default: enabled).",
    )
    ap.add_argument(
        "--causal-shot-defense",
        action="store_true",
        default=False,
        help="Challenger: residualize Shot Defense against upstream Transition Defense.",
    )
    ap.add_argument(
        "--no-deoverlap-physicality-forecheck",
        dest="deoverlap_physicality_forecheck",
        action="store_false",
        help="Disable Physicality-vs-Forecheck de-overlap residualization.",
    )
    ap.add_argument(
        "--physicality-residual-weight",
        type=float,
        default=0.10,
        help="Blend weight for residualized physicality raw signal (0=no change, 1=full residual).",
    )
    ap.add_argument(
        "--physicality-ridge-alpha",
        type=float,
        default=2.0,
        help="Ridge alpha used in physicality-vs-forecheck residualization.",
    )
    ap.add_argument(
        "--physicality-min-group-rows",
        type=int,
        default=30,
        help="Minimum rows per season/position group for physicality de-overlap fit.",
    )
    ap.add_argument(
        "--team-context-lambda",
        type=float,
        default=0.0,
        help=(
            "Fraction of (team_mean - position_mean) subtracted from each row's "
            "defensive RAPM components (eh_rapm_CA/60, eh_rapm_xGA/60, "
            "rapm_hybrid_xGA/60). 0 disables team-context correction. "
            "Empirically (sweep 2026-05): values >0 monotonically reduce the F "
            "walk-forward objective (0.6955 at L=0 -> 0.6931 at L=1.0) because "
            "the underlying RAPM columns are already regularized adjusted plus-minus "
            "estimates -- pulling out team baseline on top double-corrects and "
            "removes valid signal. Kept as an opt-in for callers who want to "
            "stack a per-position team-context shift on top of RAPM."
        ),
    )
    ap.add_argument(
        "--team-context-min-team-n",
        type=int,
        default=4,
        help="Minimum players per (season, team, position_group) cell to apply team-context correction.",
    )
    ap.add_argument(
        "--team-context-team-col",
        type=str,
        default="eh_rapm_Team",
        help="Team identifier column used for team-context grouping.",
    )
    ap.add_argument(
        "--team-context-columns",
        type=str,
        default=",".join(DEFAULT_TEAM_CONTEXT_COLUMNS),
        help="Comma-separated list of defensive component columns to apply team-context correction to.",
    )
    ap.add_argument(
        "--teammate-finishing-resid",
        dest="teammate_finishing_resid",
        action="store_true",
        default=True,
        help="Residualize assist micro features vs top-forward teammate finishing (default: enabled).",
    )
    ap.add_argument(
        "--no-teammate-finishing-resid",
        dest="teammate_finishing_resid",
        action="store_false",
        help="Disable teammate-finisher assist residualization.",
    )
    ap.add_argument(
        "--teammate-finishing-top-n",
        type=int,
        default=6,
        help="Top-N team forwards by TOI used for teammate finishing context.",
    )
    ap.add_argument(
        "--teammate-finishing-ridge-alpha",
        type=float,
        default=2.0,
        help="Ridge alpha for assist-vs-teammate-finishing residualization.",
    )
    ap.add_argument(
        "--teammate-finishing-min-group-rows",
        type=int,
        default=20,
        help="Minimum defensemen per season/position group for teammate residual fit.",
    )
    ap.add_argument(
        "--teammate-finishing-residual-weight",
        type=float,
        default=1.0,
        help="Blend weight for teammate-adjusted assist residuals (1=full residual).",
    )
    ap.add_argument(
        "--teammate-finishing-roles",
        type=str,
        default="D",
        help="Comma-separated position groups to teammate-adjust (default: D).",
    )
    args = ap.parse_args()

    if args.input.is_file():
        df = pd.read_csv(args.input)
    elif args.input.with_suffix(".parquet").is_file():
        df = pd.read_parquet(args.input.with_suffix(".parquet"))
    else:
        raise SystemExit(f"Missing feature table: {args.input}")
    need = ["player_id", "player_name", "season_id", "position_group"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing required columns in input: {miss}")
    df = _dedupe_player_season(df, label="build_all_in_one_8_category_model input")

    group_cols = ["season_id", "position_group"]
    print("PTV source for 11-category model: season -> roll3 -> pooled (per-row fallback)", flush=True)
    df, _edge_info = _merge_edge_features(df, args.edge_table if args.edge_table else None)
    _add_finishing_derived_features(df)
    teammate_roles = tuple(
        r.strip().upper() for r in str(args.teammate_finishing_roles).split(",") if r.strip()
    ) or ("D",)
    teammate_resid_info = _apply_teammate_finishing_assist_residualization(
        df,
        group_cols=group_cols,
        enabled=bool(args.teammate_finishing_resid),
        top_n_forwards=int(args.teammate_finishing_top_n),
        ridge_alpha=float(args.teammate_finishing_ridge_alpha),
        min_group_rows=int(args.teammate_finishing_min_group_rows),
        residual_weight=float(args.teammate_finishing_residual_weight),
        roles=teammate_roles,
    )
    print(
        "Teammate-finisher assist residualization:",
        f"enabled={bool(args.teammate_finishing_resid)}",
        f"roles={teammate_roles}",
        f"groups={int(teammate_resid_info.get('groups_used', 0.0))}",
        f"rows={int(teammate_resid_info.get('rows_updated', 0.0))}",
        flush=True,
    )
    preserve_tracking(df)
    _add_playmaking_derived_features(df)
    _add_rapm_hybrid_columns(df)
    _add_ptv_coalesced_columns(df)
    _add_edge_derived_features(df)
    add_skating_pillar_features(df)

    team_ctx_cols = tuple(c.strip() for c in args.team_context_columns.split(",") if c.strip())
    team_ctx_info = _apply_team_context_correction(
        df,
        columns=team_ctx_cols,
        team_col=args.team_context_team_col,
        lambda_=float(args.team_context_lambda),
        min_team_n=int(args.team_context_min_team_n),
    )
    print(
        f"Team-context correction: enabled={team_ctx_info['enabled']} "
        f"lambda={team_ctx_info['lambda']:.3f} team_col={team_ctx_info['team_col']!r} "
        f"cols_adjusted={team_ctx_info['rows_adjusted_by_col']}",
        flush=True,
    )
    skating_comp = _load_skating_component_weights(args.skating_weights_json)
    print(
        "Skating component weights: "
        f"F={skating_comp['F']} D={skating_comp['D']} "
        f"(json={args.skating_weights_json})",
        flush=True,
    )
    shot_creation_w = _load_shot_creation_component_weights(args.shot_creation_weights_json)
    shot_creation_role_specs = _shot_creation_role_specs(shot_creation_w)
    print(
        "Shot creation component weights: "
        f"F={shot_creation_w['F']} D={shot_creation_w['D']} "
        f"(json={args.shot_creation_weights_json})",
        flush=True,
    )
    to_ptv_share, sd_mix = _load_pcv_category_mix(args.pcv_weights_json)
    print(
        f"PCV category mix: TO_ptv_share={to_ptv_share} (json={args.pcv_weights_json})",
        flush=True,
    )

    category_spec: dict[str, list[tuple[str, float]]] = {}

    special_teams_spec: dict[str, list[tuple[str, float]]] = {
        "cat_power_play": [
            ("eh_std_pp_ixG/60", 0.65),
            ("eh_std_pp_iCF/60", 0.35),
        ],
        "cat_penalty_kill": [
            ("eh_std_sh_TAKE/60", 0.55),
            ("eh_std_sh_iPEN±/60", 0.45),
        ],
    }
    build_category_spec = {**category_spec, **special_teams_spec}

    transition_offense_role_specs = _transition_offense_role_specs(to_ptv_share)
    transition_defense_role_specs = _transition_defense_role_specs()
    playmaking_role_specs = _playmaking_role_specs()
    shot_defense_role_specs = _shot_defense_role_specs(sd_mix)
    finishing_role_specs = _finishing_role_specs()
    skating_off_role_specs = {
        "F": _skating_offense_hybrid_spec("F", skating_comp["F"]),
        "D": _skating_offense_hybrid_spec("D", skating_comp["D"]),
    }
    skating_def_role_specs = {
        "F": _skating_defense_hybrid_spec("F", skating_comp["F"]),
        "D": _skating_defense_hybrid_spec("D", skating_comp["D"]),
    }

    feature_ownership = _collect_category_feature_ownership(
        build_category_spec,
        shot_creation_role_specs=shot_creation_role_specs,
        transition_offense_role_specs=transition_offense_role_specs,
        transition_defense_role_specs=transition_defense_role_specs,
        playmaking_role_specs=playmaking_role_specs,
        shot_defense_role_specs=shot_defense_role_specs,
        finishing_role_specs=finishing_role_specs,
        skating_off_role_specs=skating_off_role_specs,
        skating_def_role_specs=skating_def_role_specs,
    )
    assert_isolated_feature_ownership(feature_ownership, label="all-in-one category specs")
    assert_component_feature_contract(feature_ownership, label="all-in-one category specs")
    assert_disjoint_a3z_microstats(feature_ownership, label="all-in-one category specs")
    print(
        f"Feature isolation check passed ({sum(len(v) for v in feature_ownership.values())} category-feature slots, "
        f"{len({f for fs in feature_ownership.values() for f in fs})} unique features)",
        flush=True,
    )

    # Build feature z columns once (standard categories + hybrid possession inputs).
    all_features = sorted({f for specs in build_category_spec.values() for f, _ in specs} | set(FINISHING_DERIVED_COLS))
    for specs in shot_creation_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in transition_offense_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in transition_defense_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in playmaking_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in shot_defense_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in finishing_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for role_specs in FORECHECK_PHYSICALITY_ROLE_SPECS.values():
        for specs in role_specs.values():
            all_features.extend(f for f, _ in specs)
    for specs in skating_off_role_specs.values():
        all_features.extend(f for f, _ in specs)
    for specs in skating_def_role_specs.values():
        all_features.extend(f for f, _ in specs)
    all_features = sorted(set(all_features))
    assert_no_box_score_points_in_components(all_features, label="all-in-one category specs")
    assert_category_specs_use_rapm_pcv(all_features)
    for f in all_features:
        if f not in df.columns:
            df[f] = np.nan
        zcol = f"z__{f}"
        df[zcol] = _z_by_group(df, f, group_cols)

    _build_role_hybrid_category(
        df,
        cat="cat_transition_offense",
        role_specs=transition_offense_role_specs,
        group_cols=group_cols,
    )
    _d_transition_offense_volume_discipline(df, group_cols=group_cols)
    _rapm_anchor_transition_offense_raw(df, group_cols=group_cols, blend=0.70)
    _d_transition_offense_activation_inflation_guard(df, group_cols=group_cols)
    _f_transition_offense_activation_inflation_guard(df, group_cols=group_cols)
    poss_off_z = _to_num(df.get("z__ptv_coalesced_rapm_possession_offense"))
    if poss_off_z.isna().all():
        poss_off_z = _to_num(df.get("z__ptv_coalesced_possession_offense"))
    _stale_season_shrink_toward_anchor(
        df,
        cat="cat_transition_offense",
        anchor_z=poss_off_z,
        group_cols=group_cols,
        roles=("F", "D"),
        toi_ratio_floor=0.62,
        max_pull=0.18,
    )
    _build_role_hybrid_category(
        df,
        cat="cat_shot_creation",
        role_specs=shot_creation_role_specs,
        group_cols=group_cols,
    )
    sc_deov_info = _deoverlap_shot_creation_vs_transition_offense(
        df,
        enabled=True,
        residual_weight={"F": 0.70, "D": 0.78},
        ridge_alpha=2.0,
        min_group_rows=12,
        group_cols=group_cols,
    )
    if "cat_shot_creation_raw" in df.columns:
        # Disabled: was penalizing high-volume scorers like Brady Tkachuk who don't beat xG
        # _f_shot_creation_finishing_guard(df, group_cols=group_cols)
        df["cat_shot_creation_z"] = _z_by_group(df, "cat_shot_creation_raw", group_cols)
        df["cat_shot_creation_pct"] = _pct_by_group(df, "cat_shot_creation_z", group_cols)
    _stale_season_shrink_toward_anchor(
        df,
        cat="cat_shot_creation",
        anchor_z=poss_off_z.where(poss_off_z.notna(), _rapm_offensive_skill_z(df, group_cols)),
        group_cols=group_cols,
        roles=("F", "D"),
        toi_ratio_floor=0.62,
        max_pull=0.16,
    )
    print(
        "Shot creation: residualized vs transition offense (F=0.70 D=0.78); possession+quality reweight; stale-TOI shrink",
        f"groups={int(sc_deov_info.get('groups_used', 0.0))}",
        f"F corr {float(sc_deov_info.get('corr_before_F', np.nan)):.3f}->{float(sc_deov_info.get('corr_after_F', np.nan)):.3f}",
        f"D corr {float(sc_deov_info.get('corr_before_D', np.nan)):.3f}->{float(sc_deov_info.get('corr_after_D', np.nan)):.3f}",
        flush=True,
    )
    _build_role_hybrid_category(
        df,
        cat="cat_shot_defense",
        role_specs=shot_defense_role_specs,
        group_cols=group_cols,
    )
    _rapm_anchor_shot_defense_raw(df, group_cols=group_cols, blend=0.16)
    print("Shot defense: process-first PTV/A3Z blend; F RAPM xGA guard=0.16; no D xGA re-anchor", flush=True)
    _build_role_hybrid_category(
        df,
        cat="cat_transition_defense",
        role_specs=transition_defense_role_specs,
        group_cols=group_cols,
    )
    _rapm_anchor_transition_defense_raw(df, group_cols=group_cols, blend=0.36, overshoot_gap_z=0.28)
    sd_deov_info = _deoverlap_shot_defense_vs_transition_defense(
        df,
        enabled=bool(args.causal_shot_defense),
        residual_weight={"F": 0.45, "D": 0.55},
        ridge_alpha=2.0,
        min_group_rows=12,
        group_cols=group_cols,
    )
    df["cat_shot_defense_z"] = _z_by_group(df, "cat_shot_defense_raw", group_cols)
    df["cat_shot_defense_pct"] = _pct_by_group(df, "cat_shot_defense_z", group_cols)
    print(
        "Transition defense: entry denial + DZ exit/retrieval process; D RAPM xGA overshoot guard=0.36 when raw exceeds RAPM by 0.28z",
        flush=True,
    )
    print(
        f"Shot defense conditional on transition defense: enabled={bool(args.causal_shot_defense)} (F=0.45 D=0.55)",
        f"F corr {float(sd_deov_info.get('corr_before_F', np.nan)):.3f}->{float(sd_deov_info.get('corr_after_F', np.nan)):.3f}",
        f"D corr {float(sd_deov_info.get('corr_before_D', np.nan)):.3f}->{float(sd_deov_info.get('corr_after_D', np.nan)):.3f}",
        flush=True,
    )
    _build_role_hybrid_category(
        df,
        cat="cat_playmaking",
        role_specs=playmaking_role_specs,
        group_cols=group_cols,
    )
    _rapm_anchor_playmaking_raw(df, group_cols=group_cols, blend=0.55)
    _cap_d_offense_for_elite_suppressors(df, group_cols=group_cols)
    _cap_d_offense_for_negative_rapm(df, group_cols=group_cols)
    print(
        "Playmaking (D): micro anchor blend=0.55; elite SD caps TO/PM (no on-ice xGF)",
        flush=True,
    )
    print(
        "Negative RAPM guard: D with xGF/60 < -0.08 have offensive categories capped",
        flush=True,
    )
    _build_role_hybrid_category(
        df,
        cat="cat_finishing",
        role_specs=finishing_role_specs,
        group_cols=group_cols,
    )
    print(
        "Transition offense: F/D micro+possession (55/45); quality>volume; D entry anchor (0.70); stale-TOI shrink",
        flush=True,
    )
    print(
        "Transition defense: rate/quality reweight; D RAPM overshoot guard=0.36 when raw exceeds RAPM by 0.28z",
        flush=True,
    )
    print(
        "Shot defense: isolated on-ice suppression + PTV defense + entry denial",
        flush=True,
    )
    print(
        "Finishing: volume-shrunk conversion skill blend (role-specific)",
        flush=True,
    )

    for cat, role_specs in [
        ("cat_forecheck", {r: FORECHECK_PHYSICALITY_ROLE_SPECS[r]["cat_forecheck"] for r in FORECHECK_PHYSICALITY_ROLE_SPECS}),
        ("cat_physicality", {r: FORECHECK_PHYSICALITY_ROLE_SPECS[r]["cat_physicality"] for r in FORECHECK_PHYSICALITY_ROLE_SPECS}),
    ]:
        _build_role_hybrid_category(df, cat=cat, role_specs=role_specs, group_cols=group_cols)
        print(f"Built {cat} from A3Z/EH microstats (role-specific weights)", flush=True)
    deov_info = _deoverlap_physicality_vs_forecheck(
        df,
        enabled=bool(args.deoverlap_physicality_forecheck),
        residual_weight=float(args.physicality_residual_weight),
        ridge_alpha=float(args.physicality_ridge_alpha),
        min_group_rows=int(args.physicality_min_group_rows),
        group_cols=group_cols,
    )
    # Re-standardize display columns after de-overlap raw adjustment.
    if "cat_physicality_raw" in df.columns:
        df["cat_physicality_z"] = _z_by_group(df, "cat_physicality_raw", group_cols)
        df["cat_physicality_pct"] = _pct_by_group(df, "cat_physicality_z", group_cols)
    print(
        "Physicality de-overlap:",
        f"enabled={bool(args.deoverlap_physicality_forecheck)}",
        f"weight={float(args.physicality_residual_weight):.2f}",
        f"groups={int(deov_info.get('groups_used', 0.0))}",
        f"rows={int(deov_info.get('rows_updated', 0.0))}",
        f"corr_before={float(deov_info.get('corr_before', np.nan)):.3f}",
        f"corr_after={float(deov_info.get('corr_after', np.nan)):.3f}",
        flush=True,
    )
    _build_role_hybrid_category(df, cat="cat_skating_offense", role_specs=skating_off_role_specs, group_cols=group_cols)
    _build_role_hybrid_category(df, cat="cat_skating_defense", role_specs=skating_def_role_specs, group_cols=group_cols)
    pos = df["position_group"].astype(str).str.upper().str.strip()
    w_off = pd.Series(np.where(pos.eq("D"), 0.35, 0.65), index=df.index, dtype=float)
    w_def = 1.0 - w_off
    sk_off_raw = pd.to_numeric(df.get("cat_skating_offense_raw"), errors="coerce")
    sk_def_raw = pd.to_numeric(df.get("cat_skating_defense_raw"), errors="coerce")
    sk_off_n = pd.to_numeric(df.get("cat_skating_offense_n_features"), errors="coerce")
    sk_def_n = pd.to_numeric(df.get("cat_skating_defense_n_features"), errors="coerce")
    sk_edge = sk_off_raw * w_off + sk_def_raw * w_def
    sk_edge_n = sk_off_n * w_off + sk_def_n * w_def
    # EDGE is sparse for older seasons.  Missing skating evidence should shrink
    # toward average, not borrow transition offense/defense and create a hidden
    # duplicate transition component.
    sk_fb = pd.Series(0.0, index=df.index, dtype=float)
    edge_rel = (
        pd.to_numeric(sk_edge_n, errors="coerce").fillna(0.0).clip(lower=0.0, upper=3.0) / 3.0
    )
    df["cat_skating_raw"] = sk_edge.fillna(0.0) * edge_rel + sk_fb.fillna(0.0) * (1.0 - edge_rel)
    df["cat_skating_n_features"] = edge_rel * 3.0
    skate_deov_info = _deoverlap_skating_vs_transition_offense(
        df,
        enabled=True,
        residual_weight=0.55,
        ridge_alpha=2.0,
        min_group_rows=20,
        group_cols=group_cols,
        roles=("D",),
    )
    df["cat_skating_z"] = _z_by_group(df, "cat_skating_raw", group_cols)
    df["cat_skating_pct"] = _pct_by_group(df, "cat_skating_z", group_cols)
    skate_nonnull = float(pd.to_numeric(df["cat_skating_z"], errors="coerce").notna().mean())
    print(
        f"Built cat_skating from five pillars (Speed/Acceleration/Agility/Balance/Endurance; non-null z rate={skate_nonnull:.1%})",
        flush=True,
    )
    print(
        "Skating de-overlap vs transition offense:",
        "roles=('D',)",
        "weight=0.55",
        f"groups={int(skate_deov_info.get('groups_used', 0.0))}",
        f"D corr {float(skate_deov_info.get('corr_before_D', np.nan)):.3f}->{float(skate_deov_info.get('corr_after_D', np.nan)):.3f}",
        flush=True,
    )
    if skate_nonnull < 0.50:
        print(
            "WARNING: cat_skating_z is sparse; check --edge-table merge or EDGE sidecar coverage.",
            flush=True,
        )

    category_spec_all = {
        "cat_shot_creation": [],
        "cat_shot_defense": [],
        "cat_transition_defense": [],
        "cat_playmaking": [],
        "cat_finishing": [],
        "cat_forecheck": [],
        "cat_physicality": [],
        "cat_skating": [],
        **category_spec,
    }

    # Category raw -> z -> pct (includes PP/PK; excluded from overall mean below).
    category_cols = [
        (
            "cat_shot_creation",
            "cat_shot_creation_raw",
            "cat_shot_creation_z",
            "cat_shot_creation_pct",
            "cat_shot_creation_n_features",
        ),
        (
            "cat_transition_offense",
            "cat_transition_offense_raw",
            "cat_transition_offense_z",
            "cat_transition_offense_pct",
            "cat_transition_offense_n_features",
        ),
        (
            "cat_shot_defense",
            "cat_shot_defense_raw",
            "cat_shot_defense_z",
            "cat_shot_defense_pct",
            "cat_shot_defense_n_features",
        ),
        (
            "cat_transition_defense",
            "cat_transition_defense_raw",
            "cat_transition_defense_z",
            "cat_transition_defense_pct",
            "cat_transition_defense_n_features",
        ),
        (
            "cat_playmaking",
            "cat_playmaking_raw",
            "cat_playmaking_z",
            "cat_playmaking_pct",
            "cat_playmaking_n_features",
        ),
        (
            "cat_finishing",
            "cat_finishing_raw",
            "cat_finishing_z",
            "cat_finishing_pct",
            "cat_finishing_n_features",
        ),
        ("cat_forecheck", "cat_forecheck_raw", "cat_forecheck_z", "cat_forecheck_pct", "cat_forecheck_n_features"),
        ("cat_physicality", "cat_physicality_raw", "cat_physicality_z", "cat_physicality_pct", "cat_physicality_n_features"),
        ("cat_skating", "cat_skating_raw", "cat_skating_z", "cat_skating_pct", "cat_skating_n_features"),
    ]
    for cat, specs in build_category_spec.items():
        raw_col = f"{cat}_raw"
        n_col = f"{cat}_n_features"
        z_col = f"{cat}_z"
        pct_col = f"{cat}_pct"

        if not specs:
            df[raw_col] = np.nan
            df[n_col] = 0.0
            df[z_col] = np.nan
            df[pct_col] = np.nan
        else:
            z_specs = [(f"z__{f}", w) for f, w in specs]
            raw, n_used = _weighted_mean_from_z(df, z_specs)
            df[raw_col] = raw
            df[n_col] = n_used
            df[z_col] = _z_by_group(df, raw_col, group_cols)
            df[pct_col] = _pct_by_group(df, z_col, group_cols)
        category_cols.append((cat, raw_col, z_col, pct_col, n_col))

    # Legacy combined special teams (same PP/PK blend weights as pre-split spec).
    pp_z = pd.to_numeric(df.get("cat_power_play_z"), errors="coerce")
    pk_z = pd.to_numeric(df.get("cat_penalty_kill_z"), errors="coerce")
    df["cat_special_teams_z"] = np.where(
        pp_z.notna() | pk_z.notna(),
        pp_z.fillna(0.0) * 0.93 + pk_z.fillna(0.0) * 0.07,
        np.nan,
    )
    df["cat_special_teams_pct"] = _pct_by_group(df, "cat_special_teams_z", group_cols)

    # Legacy aliases for downstream compatibility.
    for suff in ["raw", "z", "pct", "n_features"]:
        sc = f"cat_shot_creation_{suff}"
        fin = f"cat_finishing_{suff}"
        to = f"cat_transition_offense_{suff}"
        sd = f"cat_shot_defense_{suff}"
        td = f"cat_transition_defense_{suff}"
        so = f"cat_shot_offense_{suff}"
        if sc in df.columns and fin in df.columns:
            if suff == "n_features":
                df[so] = pd.to_numeric(df[sc], errors="coerce").fillna(0.0) + pd.to_numeric(
                    df[fin], errors="coerce"
                ).fillna(0.0)
            else:
                sc_v = pd.to_numeric(df[sc], errors="coerce")
                fin_v = pd.to_numeric(df[fin], errors="coerce")
                df[so] = sc_v.fillna(0.0) * 0.65 + fin_v.fillna(0.0) * 0.35
                df.loc[sc_v.isna() & fin_v.isna(), so] = np.nan
        elif sc in df.columns:
            df[so] = df[sc]
        if so in df.columns and to in df.columns:
            so_v = pd.to_numeric(df[so], errors="coerce")
            to_v = pd.to_numeric(df[to], errors="coerce")
            df[f"cat_5v5_offense_{suff}"] = so_v.fillna(0.0) * 0.7 + to_v.fillna(0.0) * 0.3
            df.loc[so_v.isna() & to_v.isna(), f"cat_5v5_offense_{suff}"] = np.nan
        if sd in df.columns and td in df.columns:
            df[f"cat_5v5_defense_{suff}"] = pd.to_numeric(df[sd], errors="coerce").fillna(0.0) * 0.7 + pd.to_numeric(df[td], errors="coerce").fillna(0.0) * 0.3
    if "cat_transition_offense_raw" in df.columns:
        for suff in ["raw", "z", "pct", "n_features"]:
            src = f"cat_transition_offense_{suff}"
            if src in df.columns:
                df[f"cat_transition_{suff}"] = df[src]

    # Completeness guard for downstream 2023-2026 reporting/cards:
    # when a category cannot be estimated for a player-season, keep a neutral
    # value instead of NaN so public tables are fully populated.
    if "a3z_toi_5v5_min" in df.columns:
        toi_a3z = pd.to_numeric(df["a3z_toi_5v5_min"], errors="coerce")
        toi_eh = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce")
        df["a3z_toi_5v5_min"] = toi_a3z.where(toi_a3z.notna(), toi_eh).fillna(0.0)
    if "eh_rapm_TOI" in df.columns:
        toi_eh = pd.to_numeric(df["eh_rapm_TOI"], errors="coerce")
        toi_a3z = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce")
        df["eh_rapm_TOI"] = toi_eh.where(toi_eh.notna() & (toi_eh > 0.0), toi_a3z).fillna(0.0)
    df["a3z_toi_is_fallback"] = (~df["a3z_tracking_observed"] &
                                  pd.to_numeric(df["a3z_toi_5v5_min"], errors="coerce").gt(0))

    complete_blocks = list(category_cols)
    for stem in ["cat_shot_offense", "cat_5v5_offense", "cat_5v5_defense", "cat_transition"]:
        complete_blocks.append(
            (
                stem,
                f"{stem}_raw",
                f"{stem}_z",
                f"{stem}_pct",
                f"{stem}_n_features",
            )
        )
    # Deduplicate by stem while preserving order.
    seen_stems: set[str] = set()
    dedup_blocks = []
    for stem, raw_col, z_col, pct_col, n_col in complete_blocks:
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        dedup_blocks.append((stem, raw_col, z_col, pct_col, n_col))

    for _, raw_col, z_col, pct_col, n_col in dedup_blocks:
        if raw_col in df.columns:
            df[raw_col] = pd.to_numeric(df[raw_col], errors="coerce").fillna(0.0)
        if z_col in df.columns:
            df[z_col] = pd.to_numeric(df[z_col], errors="coerce").fillna(0.0)
        if pct_col in df.columns:
            df[pct_col] = (
                pd.to_numeric(df[pct_col], errors="coerce")
                .fillna(50.0)
                .clip(lower=0.0, upper=100.0)
            )
        if n_col in df.columns:
            df[n_col] = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0)

    # Overall score: mean of available category z-scores (split O/D + remaining categories).
    overall_inputs = ["cat_transition_offense_z"] + [f"{cat}_z" for cat in category_spec_all]
    ov = df[overall_inputs].apply(pd.to_numeric, errors="coerce")
    rating_evidence_stems = (
        "cat_transition_offense", "cat_shot_defense", "cat_transition_defense",
        "cat_shot_creation", "cat_finishing", "cat_playmaking", "cat_forecheck",
        "cat_physicality", "cat_skating",
    )
    evidence_cols = [f"{stem}_n_features" for stem in rating_evidence_stems if f"{stem}_n_features" in df.columns]
    if evidence_cols:
        evidence = df[evidence_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0.0).sum(axis=1)
    else:
        evidence = ov.notna().sum(axis=1)
    effective_toi = pd.to_numeric(df.get("eh_rapm_TOI"), errors="coerce").fillna(0.0)
    recent_window = pd.to_numeric(df["season_id"], errors="coerce") >= 20232024
    eligible = (~recent_window) | ((evidence >= 6) & (effective_toi >= 100.0))
    df["model_rating_evidence_categories"] = evidence.astype(int)
    df["model_rating_effective_toi"] = effective_toi
    df["model_rating_eligible"] = eligible.astype(bool)
    df["overall_10cat_z"] = ov.mean(axis=1, skipna=True).where(eligible)
    df["overall_10cat_n_categories"] = ov.notna().sum(axis=1).astype(float)
    df["overall_10cat_z"] = _z_by_group(df, "overall_10cat_z", group_cols)
    df["overall_10cat_pct"] = _pct_by_group(df, "overall_10cat_z", group_cols)
    # Legacy column names (downstream scripts still reference overall_8cat_*).
    df["overall_8cat_z"] = df["overall_10cat_z"]
    df["overall_8cat_pct"] = df["overall_10cat_pct"]
    df["overall_8cat_n_categories"] = df["overall_10cat_n_categories"]

    # Compact output block + passthrough identifiers/context.
    # Event-based exposure for empirical-Bayes stabilization downstream. This
    # measures actual forecheck opportunities more directly than TOI alone.
    fc_pressure_n = pd.to_numeric(df.get("a3z_forecheck_pressures"), errors="coerce").fillna(0.0)
    fc_cycle_rate = pd.to_numeric(df.get("a3z_cycle_forecheck_events_per_60"), errors="coerce").fillna(0.0)
    fc_recovery_rate = pd.to_numeric(df.get("a3z_recoveries_per_60"), errors="coerce").fillna(0.0)
    fc_toi = pd.to_numeric(df.get("a3z_toi_5v5_min"), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["forecheck_event_exposure"] = (
        fc_pressure_n + (0.50 * fc_cycle_rate + 0.20 * fc_recovery_rate) * fc_toi / 60.0
    ).clip(lower=0.0)

    keep_front = [
        "player_id",
        "player_name",
        "season_id",
        "position_group",
        "a3z_toi_5v5_min",
        "a3z_tracked_toi_5v5_min", "a3z_tracking_observed", "a3z_toi_is_fallback", "has_a3z_season",
        "a3z_playmaking_evidence_count_proxy", "a3z_playmaking_evidence_reliability",
        "eh_rapm_TOI",
        "eh_std_TOI",
        "eh_rapm_Team",
        "eh_std_Team",
        "a3z_team",
        "forecheck_event_exposure",
        "model_rating_evidence_categories",
        "model_rating_effective_toi",
        "model_rating_eligible",
        "ptv_best_rapm_pre_shot_pressure",
        "ptv_season_rapm_pre_shot_pressure",
        "ptv_roll3_confidence_tier",
        "ptv_roll3_archetype_public",
        "ptv_roll3_leaderboard_qualified",
        "ptv_roll3_n_seasons_in_window",
        "ptv_pooled_confidence_tier",
        "ptv_pooled_archetype_public",
        "has_ptv_best",
        "has_eh_rapm",
        "has_hs_prior_rapm",
        "has_eh_std",
        "has_yoy_outcomes",
    ]
    keep_front = [c for c in keep_front if c in df.columns]

    keep_cats: list[str] = []
    for cat, raw_col, z_col, pct_col, n_col in category_cols:
        keep_cats.extend([raw_col, z_col, pct_col, n_col])
    audit_cols = [
        c for c in [
            "cat_shot_creation_raw__pre_transition_residual",
            "cat_shot_defense_raw__pre_transition_residual",
            "cat_physicality_raw__pre_forecheck_residual",
            "cat_skating_raw__pre_transition_residual",
        ] if c in df.columns
    ]
    legacy_alias_cats: list[str] = []
    for stem in ["cat_shot_offense", "cat_5v5_offense", "cat_5v5_defense", "cat_transition"]:
        for suff in ["raw", "z", "pct", "n_features"]:
            c = f"{stem}_{suff}"
            if c in df.columns:
                legacy_alias_cats.append(c)

    keep_tail = [
        "overall_10cat_z",
        "overall_10cat_pct",
        "overall_10cat_n_categories",
        "overall_8cat_z",
        "overall_8cat_pct",
        "overall_8cat_n_categories",
    ]

    out = df[keep_front + keep_cats + audit_cols + legacy_alias_cats + keep_tail].copy()
    out = _dedupe_player_season(out, label="build_all_in_one_8_category_model output")
    out = out.sort_values(["season_id", "position_group", "overall_8cat_z"], ascending=[True, True, False], na_position="last")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    print(f"Wrote {args.out_csv} rows={len(out):,} cols={len(out.columns):,}")
    print(f"Wrote {args.out_parquet} rows={len(out):,} cols={len(out.columns):,}")
    print("Power play non-null rows:", int(out["cat_power_play_z"].notna().sum()))
    print("Penalty kill non-null rows:", int(out["cat_penalty_kill_z"].notna().sum()))


if __name__ == "__main__":
    main()
