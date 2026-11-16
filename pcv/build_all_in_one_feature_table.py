#!/usr/bin/env python3
"""
Build an all-in-one player-season feature table by merging:
- **per-season joint PTV** from ``data/pcv/seasonal/`` (season-varying RAPM-PCV coefficients)
- **trailing 3-year rolling PTV** computed per (*player*, *season*) from those seasonal fits
- optional pooled PTV leaderboard (player-level fallback)
- A3Z season rates/modules (season-level)
- A3Z component weights by player-season (spine)
- EH RAPM EV regular stats (season-level)
- EH standard all-strength regular stats (season-level), with separate PP/SH tables
- YoY outcomes caches (season-level, where available)

Default target seasons: 20162017 through 20252026 (override with --seasons).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from a3z_feature_integrity import assess_a3z_season_coverage, sanitize_a3z_rates

from ptv_coef import (
    DEFAULT_SEASONAL_COEF_DIR,
    PTV_RAPM_STEMS,
    build_player_season_ptv_table,
    build_trailing_roll3_ptv_table,
    list_season_coef_seasons,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_A3Z_EXTENDED = ROOT / "data/pcv/a3z_player_season_rates_modules_extended.csv"
DEFAULT_A3Z_BASE = ROOT / "data/pcv/a3z_player_season_rates_modules.csv"
DEFAULT_A3Z_PXP_EVENTS = ROOT / "data/pcv/a3z_pxp_events.parquet"
DEFAULT_A3Z_GAME_STATS_RATES = ROOT / "data/pcv/a3z_game_stats_player_season_rates.parquet"
HS_PRIOR_XGF_INTERCEPT = 0.446297
HS_PRIOR_XGF_SLOPE = 0.280373
HS_PRIOR_XGA_INTERCEPT = 0.453489
HS_PRIOR_XGA_SLOPE = 0.203671


def _normalize_name(s: str) -> str:
    t = str(s or "").strip().lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    # Normalize punctuation variants so "O'Reilly", "O Reilly", and "OReilly" align.
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _alias_first_name_variants(first: str) -> set[str]:
    f = str(first or "").strip().lower()
    out = {f}
    translit_pairs = [
        ("egor", "yegor"),
        ("evgeni", "yevgeni"),
        ("evgeny", "yevgeny"),
        ("ilya", "ilja"),
        ("aleksander", "alexander"),
        ("nikolaj", "nikolai"),
    ]
    for a, b in translit_pairs:
        if f == a:
            out.add(b)
        if f == b:
            out.add(a)
    return out


def _normalize_position_group(s: object) -> str | None:
    t = str(s or "").strip().upper()
    if t == "":
        return None
    # Defense labels.
    if t in {"D", "LD", "RD", "DEF", "DEFENSE", "DEFENCEMAN", "DEFENSEMAN"}:
        return "D"
    # Forward labels.
    if t in {"F", "C", "LW", "RW", "L", "R", "W"}:
        return "F"
    if "/" in t:
        parts = [p.strip() for p in t.split("/") if p.strip()]
        if any(p in {"D", "LD", "RD"} for p in parts):
            return "D"
        if any(p in {"F", "C", "LW", "RW", "L", "R", "W"} for p in parts):
            return "F"
    if "D" in t and not any(ch in t for ch in ["C", "L", "R", "W", "F"]):
        return "D"
    return "F"


def _name_alias_keys(name: str) -> set[str]:
    base = _normalize_name(name)
    parts = [p for p in base.split(" ") if p]
    if len(parts) < 2:
        return {base}
    first = parts[0]
    tail = " ".join(parts[1:])
    out = set()
    for f in _alias_first_name_variants(first):
        out.add(_normalize_name(f"{f} {tail}"))
    return out


def _season_str_to_id(s: str) -> int | None:
    t = str(s).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})", t)
    if not m:
        return None
    y1 = int(m.group(1))
    y2 = int(m.group(2))
    if (y1 + 1) % 100 != y2:
        return None
    return int(f"{y1}{y1+1}")


def _eh_season_to_id(s: str) -> int | None:
    t = str(s).strip()
    m = re.fullmatch(r"(\d{2})-(\d{2})", t)
    if not m:
        return None
    y1 = int(m.group(1))
    y2 = int(m.group(2))
    if (y1 + 1) % 100 != y2:
        return None
    full_y1 = 2000 + y1
    return int(f"{full_y1}{full_y1+1}")


def _normalize_team_abbr(s: object) -> str:
    t = str(s or "").strip().upper().replace(" ", "")
    if t == "" or t == "NAN":
        return ""
    t = t.replace(".", "")
    mapping = {
        "LA": "LAK",
        "LAK": "LAK",
        "LAKINGS": "LAK",
        "TB": "TBL",
        "TBL": "TBL",
        "SJ": "SJS",
        "SJS": "SJS",
        "NJ": "NJD",
        "NJD": "NJD",
        "MON": "MTL",
        "VGK": "VGK",
        "UTA": "UTA",
    }
    if t in mapping:
        return mapping[t]
    if len(t) >= 3:
        return t[:3]
    return t


def _build_a3z_pxp_team_context(path: Path, target_seasons: set[int]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["season_id", "team_abbr"])
    if path.suffix.lower() == ".parquet":
        d = pd.read_parquet(path)
    else:
        d = pd.read_csv(path)
    if d.empty:
        return pd.DataFrame(columns=["season_id", "team_abbr"])
    d = d.copy()
    if "season" not in d.columns or "team" not in d.columns:
        return pd.DataFrame(columns=["season_id", "team_abbr"])

    d["season_id"] = d["season"].map(_season_str_to_id)
    d = d[d["season_id"].isin(target_seasons)].copy()
    if d.empty:
        return pd.DataFrame(columns=["season_id", "team_abbr"])
    d["team_abbr"] = d["team"].map(_normalize_team_abbr)
    d = d[d["team_abbr"].astype(str).str.len() > 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["season_id", "team_abbr"])

    d["is_shot_event"] = (
        d.get("shot_type", pd.Series("", index=d.index)).astype(str).str.strip().ne("")
        | d.get("shooter", pd.Series("", index=d.index)).astype(str).str.strip().ne("")
    ).astype(float)
    d["is_sog_event"] = pd.to_numeric(d.get("is_sog"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    d["is_sc_event"] = pd.to_numeric(d.get("is_sc"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    d["is_goal_event"] = pd.to_numeric(d.get("is_goal"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    d["is_entry_event"] = d.get("entry_by", pd.Series("", index=d.index)).astype(str).str.strip().ne("").astype(float)
    d["is_entry_defense_event"] = (
        d.get("defended_by", pd.Series("", index=d.index)).astype(str).str.strip().ne("")
    ).astype(float)
    d["is_pressure_event"] = (
        d.get("pressure", pd.Series("", index=d.index)).astype(str).str.strip().ne("")
    ).astype(float)
    d["is_exit_event"] = d.get("exit", pd.Series("", index=d.index)).astype(str).str.strip().ne("").astype(float)
    gid = d.get("game_id", pd.Series("", index=d.index)).astype(str).str.strip()
    d["game_id_norm"] = gid.where(gid.ne(""), pd.NA)

    g = (
        d.groupby(["season_id", "team_abbr"], as_index=False)
        .agg(
            a3z_pxp_team_events=("team_abbr", "size"),
            a3z_pxp_team_shot_events=("is_shot_event", "sum"),
            a3z_pxp_team_sog_events=("is_sog_event", "sum"),
            a3z_pxp_team_sc_events=("is_sc_event", "sum"),
            a3z_pxp_team_goal_events=("is_goal_event", "sum"),
            a3z_pxp_team_entry_events=("is_entry_event", "sum"),
            a3z_pxp_team_entry_defense_events=("is_entry_defense_event", "sum"),
            a3z_pxp_team_pressure_events=("is_pressure_event", "sum"),
            a3z_pxp_team_exit_events=("is_exit_event", "sum"),
            a3z_pxp_team_games=("game_id_norm", pd.Series.nunique),
        )
        .copy()
    )
    for c in [
        "a3z_pxp_team_events",
        "a3z_pxp_team_shot_events",
        "a3z_pxp_team_sog_events",
        "a3z_pxp_team_sc_events",
        "a3z_pxp_team_goal_events",
        "a3z_pxp_team_entry_events",
        "a3z_pxp_team_entry_defense_events",
        "a3z_pxp_team_pressure_events",
        "a3z_pxp_team_exit_events",
    ]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    games = pd.to_numeric(g["a3z_pxp_team_games"], errors="coerce").replace(0.0, np.nan)
    for c in [
        "a3z_pxp_team_events",
        "a3z_pxp_team_shot_events",
        "a3z_pxp_team_sog_events",
        "a3z_pxp_team_sc_events",
        "a3z_pxp_team_goal_events",
        "a3z_pxp_team_entry_events",
        "a3z_pxp_team_entry_defense_events",
        "a3z_pxp_team_pressure_events",
        "a3z_pxp_team_exit_events",
    ]:
        g[f"{c}_per_game"] = (pd.to_numeric(g[c], errors="coerce") / games).replace([np.inf, -np.inf], np.nan)
    return g


def _build_name_id_maps(player_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = player_meta[["player_id", "player_name", "position_group"]].copy()
    pm["position_group"] = pm["position_group"].map(_normalize_position_group)
    pm["name_key"] = pm["player_name"].map(_normalize_name)
    rows = []
    for _, r in pm.iterrows():
        for k in _name_alias_keys(str(r["player_name"])):
            rows.append(
                {
                    "name_key": k,
                    "player_id": int(r["player_id"]),
                    "player_name": str(r["player_name"]),
                    "position_group": str(r["position_group"] or ""),
                }
            )
    nm = pd.DataFrame(rows).drop_duplicates()
    c = nm.groupby("name_key", as_index=False)["player_id"].nunique().rename(columns={"player_id": "n_ids"})
    uniq = nm.merge(c, on="name_key", how="left")
    uniq = uniq[uniq["n_ids"] == 1].drop(columns=["n_ids"]).drop_duplicates("name_key", keep="first")
    uniq = uniq[["name_key", "player_id", "player_name", "position_group"]]

    c2 = (
        nm.groupby(["name_key", "position_group"], as_index=False)["player_id"]
        .nunique()
        .rename(columns={"player_id": "n_ids"})
    )
    by_pos = nm.merge(c2, on=["name_key", "position_group"], how="left")
    by_pos = by_pos[by_pos["n_ids"] == 1].drop(columns=["n_ids"]).drop_duplicates(
        ["name_key", "position_group"], keep="first"
    )
    by_pos = by_pos[["name_key", "position_group", "player_id", "player_name"]]
    return uniq, by_pos


def _attach_player_ids(
    df: pd.DataFrame,
    name_col: str,
    pos_col: str | None,
    unique_map: pd.DataFrame,
    by_pos_map: pd.DataFrame,
) -> pd.DataFrame:
    d = df.copy()
    d["name_key"] = d[name_col].map(_normalize_name)
    d["_pos_key"] = d[pos_col].map(_normalize_position_group) if pos_col and pos_col in d.columns else None

    um = unique_map[["name_key", "player_id"]].rename(columns={"player_id": "_pid_u"})
    d = d.merge(um, on="name_key", how="left")

    pm = by_pos_map[["name_key", "position_group", "player_id"]].rename(
        columns={"position_group": "_pos_key", "player_id": "_pid_p"}
    )
    d = d.merge(pm, on=["name_key", "_pos_key"], how="left")

    d["player_id"] = pd.to_numeric(d.get("_pid_u"), errors="coerce").where(
        pd.to_numeric(d.get("_pid_u"), errors="coerce").notna(),
        pd.to_numeric(d.get("_pid_p"), errors="coerce"),
    )
    return d.drop(columns=[c for c in ["_pid_u", "_pid_p", "_pos_key"] if c in d.columns])


def _prefix_except(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    out = df.copy()
    ren = {c: f"{prefix}{c}" for c in out.columns if c not in keep}
    return out.rename(columns=ren)


def _dedupe_player_season_by_toi(df: pd.DataFrame, toi_col_candidates: list[str]) -> pd.DataFrame:
    d = df.copy()
    toi_col = next((c for c in toi_col_candidates if c in d.columns), None)
    if toi_col is None:
        return d.drop_duplicates(["player_id", "season_id"], keep="last")
    d["_toi_sort_key"] = pd.to_numeric(d[toi_col], errors="coerce").fillna(-1.0)
    d = d.sort_values(["player_id", "season_id", "_toi_sort_key"]).drop_duplicates(
        ["player_id", "season_id"], keep="last"
    )
    return d.drop(columns=["_toi_sort_key"])


def _aggregate_player_season_rows(
    df: pd.DataFrame,
    toi_col_candidates: list[str],
    team_col_candidates: list[str] | None = None,
    gp_col_candidates: list[str] | None = None,
) -> pd.DataFrame:
    d = df.copy()
    keys = ["player_id", "season_id"]
    toi_col = next((c for c in toi_col_candidates if c in d.columns), None)
    team_col = next((c for c in (team_col_candidates or []) if c in d.columns), None)
    gp_col = next((c for c in (gp_col_candidates or []) if c in d.columns), None)

    if toi_col is not None:
        d[toi_col] = pd.to_numeric(d[toi_col], errors="coerce").fillna(0.0)
    if gp_col is not None:
        d[gp_col] = pd.to_numeric(d[gp_col], errors="coerce")

    rows: list[dict[str, object]] = []
    for _, g in d.groupby(keys, dropna=False):
        row: dict[str, object] = {
            "player_id": int(g["player_id"].iloc[0]),
            "season_id": int(g["season_id"].iloc[0]),
        }
        toi = pd.to_numeric(g[toi_col], errors="coerce").fillna(0.0).to_numpy(float) if toi_col else np.ones(len(g))
        has_weight = np.isfinite(toi) & (toi > 0)

        for c in d.columns:
            if c in keys:
                continue
            s = g[c]
            if c == toi_col:
                row[c] = float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())
                continue
            if c == gp_col:
                row[c] = float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())
                continue
            if c == team_col:
                teams = [str(t).strip() for t in s.dropna().tolist() if str(t).strip()]
                u = sorted(set(teams))
                row[c] = (u[0] if len(u) == 1 else "MULTI") if u else np.nan
                continue

            sn = pd.to_numeric(s, errors="coerce")
            numeric_like = int(sn.notna().sum()) > 0
            if numeric_like:
                vals = sn.to_numpy(float)
                m = np.isfinite(vals)
                if has_weight.any():
                    mw = m & has_weight
                    if mw.any():
                        row[c] = float(np.average(vals[mw], weights=toi[mw]))
                    elif m.any():
                        row[c] = float(np.nanmean(vals[m]))
                    else:
                        row[c] = np.nan
                else:
                    row[c] = float(np.nanmean(vals[m])) if m.any() else np.nan
            else:
                nz = s.dropna()
                row[c] = nz.iloc[0] if len(nz) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _full_season_id_range(start_year: int, end_year: int) -> str:
    parts = []
    for y in range(int(start_year), int(end_year) + 1):
        parts.append(f"{y}{y+1}")
    return ",".join(parts)


def _add_hockeystats_prior_rapm_proxy(
    df: pd.DataFrame,
    *,
    toi_col: str = "eh_rapm_TOI",
    xgf_col: str = "eh_rapm_xGF/60",
    xga_col: str = "eh_rapm_xGA/60",
    shrink_k_minutes: float = 650.0,
) -> pd.DataFrame:
    """
    Add HockeyStats-style prior-adjusted RAPM proxies.

    HockeyStats documents a forward-year trend prior for RAPM xGF/60 and xGA/60.
    We emulate the same prior structure by daisy-chaining a prior prediction and
    blending toward current-season EH RAPM with TOI reliability weighting.
    """
    if "player_id" not in df.columns or "season_id" not in df.columns:
        return df
    if xgf_col not in df.columns or xga_col not in df.columns:
        return df

    d = df.copy()
    d[toi_col] = pd.to_numeric(d.get(toi_col), errors="coerce").fillna(0.0)
    d[xgf_col] = pd.to_numeric(d.get(xgf_col), errors="coerce")
    d[xga_col] = pd.to_numeric(d.get(xga_col), errors="coerce")
    d = d.sort_values(["player_id", "season_id"], na_position="last").reset_index(drop=True)

    d["hs_prior_pred_rapm_xGF/60"] = np.nan
    d["hs_prior_pred_rapm_xGA/60"] = np.nan
    d["hs_prior_adj_rapm_xGF/60"] = np.nan
    d["hs_prior_adj_rapm_xGA/60"] = np.nan

    k = float(max(shrink_k_minutes, 1.0))
    for _, idx in d.groupby("player_id", dropna=False).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        toi = pd.to_numeric(d.loc[ii, toi_col], errors="coerce").fillna(0.0).to_numpy(float)
        xgf = pd.to_numeric(d.loc[ii, xgf_col], errors="coerce").to_numpy(float)
        xga = pd.to_numeric(d.loc[ii, xga_col], errors="coerce").to_numpy(float)
        prior_xgf = np.full(len(ii), np.nan, dtype=float)
        prior_xga = np.full(len(ii), np.nan, dtype=float)
        adj_xgf = np.full(len(ii), np.nan, dtype=float)
        adj_xga = np.full(len(ii), np.nan, dtype=float)

        for i in range(len(ii)):
            if i > 0 and np.isfinite(adj_xgf[i - 1]):
                prior_xgf[i] = HS_PRIOR_XGF_INTERCEPT + HS_PRIOR_XGF_SLOPE * adj_xgf[i - 1]
            if i > 0 and np.isfinite(adj_xga[i - 1]):
                prior_xga[i] = HS_PRIOR_XGA_INTERCEPT + HS_PRIOR_XGA_SLOPE * adj_xga[i - 1]

            w = float(toi[i] / (toi[i] + k)) if np.isfinite(toi[i]) and toi[i] > 0 else 0.0
            if np.isfinite(xgf[i]) and np.isfinite(prior_xgf[i]):
                adj_xgf[i] = w * xgf[i] + (1.0 - w) * prior_xgf[i]
            elif np.isfinite(xgf[i]):
                adj_xgf[i] = xgf[i]
            else:
                adj_xgf[i] = prior_xgf[i]

            if np.isfinite(xga[i]) and np.isfinite(prior_xga[i]):
                adj_xga[i] = w * xga[i] + (1.0 - w) * prior_xga[i]
            elif np.isfinite(xga[i]):
                adj_xga[i] = xga[i]
            else:
                adj_xga[i] = prior_xga[i]

        d.loc[ii, "hs_prior_pred_rapm_xGF/60"] = prior_xgf
        d.loc[ii, "hs_prior_pred_rapm_xGA/60"] = prior_xga
        d.loc[ii, "hs_prior_adj_rapm_xGF/60"] = adj_xgf
        d.loc[ii, "hs_prior_adj_rapm_xGA/60"] = adj_xga

    d["hs_prior_adj_rapm_xG±/60"] = (
        pd.to_numeric(d["hs_prior_adj_rapm_xGF/60"], errors="coerce")
        - pd.to_numeric(d["hs_prior_adj_rapm_xGA/60"], errors="coerce")
    )
    d["rapm_hybrid_xGF/60"] = pd.to_numeric(d["hs_prior_adj_rapm_xGF/60"], errors="coerce").where(
        pd.to_numeric(d["hs_prior_adj_rapm_xGF/60"], errors="coerce").notna(),
        pd.to_numeric(d.get("eh_rapm_xGF/60"), errors="coerce"),
    )
    d["rapm_hybrid_xGA/60"] = pd.to_numeric(d["hs_prior_adj_rapm_xGA/60"], errors="coerce").where(
        pd.to_numeric(d["hs_prior_adj_rapm_xGA/60"], errors="coerce").notna(),
        pd.to_numeric(d.get("eh_rapm_xGA/60"), errors="coerce"),
    )
    d["rapm_hybrid_xG±/60"] = pd.to_numeric(d["hs_prior_adj_rapm_xG±/60"], errors="coerce").where(
        pd.to_numeric(d["hs_prior_adj_rapm_xG±/60"], errors="coerce").notna(),
        pd.to_numeric(d.get("eh_rapm_xG±/60"), errors="coerce"),
    )
    d["has_hs_prior_rapm"] = d[
        ["hs_prior_adj_rapm_xGF/60", "hs_prior_adj_rapm_xGA/60", "hs_prior_adj_rapm_xG±/60"]
    ].notna().any(axis=1)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ptv-rolling",
        type=Path,
        default=ROOT / "data/pcv/ptv_public_leaderboard_rolling_3yr_advanced.csv",
        help="3-year TOI-weighted per-season RAPM PTV (player-level).",
    )
    ap.add_argument(
        "--ptv",
        type=Path,
        default=ROOT / "data/pcv/ptv_public_leaderboard_pooled_advanced.csv",
        help="Legacy pooled joint PTV (optional second merge).",
    )
    ap.add_argument(
        "--ptv-season-dir",
        type=Path,
        default=DEFAULT_SEASONAL_COEF_DIR,
        help="Per-season joint RAPM-PTV coefficients (player-season rows).",
    )
    ap.add_argument(
        "--a3z-season",
        type=Path,
        default=(DEFAULT_A3Z_EXTENDED if DEFAULT_A3Z_EXTENDED.is_file() else DEFAULT_A3Z_BASE),
    )
    ap.add_argument("--a3z-weights", type=Path, default=ROOT / "data/pcv/a3z_component_weights_by_player_season.csv")
    ap.add_argument(
        "--a3z-game-stats-rates",
        type=Path,
        default=DEFAULT_A3Z_GAME_STATS_RATES,
        help="Definition-compatible rates rebuilt from raw A3Z Game Stats rows.",
    )
    ap.add_argument("--eh-rapm", type=Path, default=ROOT / "data/EH_rapm_sk_stats_ev_regular_2026-05-17.csv")
    ap.add_argument("--eh-std", type=Path, default=ROOT / "data/EH_std_sk_stats_5v5_regular_no_adj_2026-09-23.csv")
    ap.add_argument("--eh-std-pp", type=Path, default=ROOT / "data/EH_std_sk_stats_pp_regular_no_adj_2026-05-17.csv")
    ap.add_argument("--eh-std-sh", type=Path, default=ROOT / "data/EH_std_sk_stats_sh_regular_no_adj_2026-05-17.csv")
    ap.add_argument("--player-meta", type=Path, default=ROOT / "data/pcv/player_meta_priors.csv")
    ap.add_argument("--outcomes-a", type=Path, default=ROOT / "data/pcv/yoy_player_outcomes_20232024.parquet")
    ap.add_argument("--outcomes-b", type=Path, default=ROOT / "data/pcv/yoy_player_outcomes_20242025.parquet")
    ap.add_argument("--a3z-pxp-events", type=Path, default=DEFAULT_A3Z_PXP_EVENTS)
    ap.add_argument(
        "--hs-prior-shrink-k-minutes",
        type=float,
        default=650.0,
        help="Reliability minutes for HockeyStats-style prior-adjusted RAPM proxy blend.",
    )
    ap.add_argument(
        "--refresh-a3z-pxp-etl",
        action="store_true",
        help="Rebuild data/pcv/a3z_pxp_events.parquet via build_a3z_pxp_etl.py before merge.",
    )
    ap.add_argument("--seasons", type=str, default=_full_season_id_range(2016, 2025))
    ap.add_argument("--out-csv", type=Path, default=ROOT / "data/pcv/player_season_feature_table_all_in_one.csv")
    ap.add_argument("--out-parquet", type=Path, default=ROOT / "data/pcv/player_season_feature_table_all_in_one.parquet")
    args = ap.parse_args()

    target_seasons = {int(x.strip()) for x in str(args.seasons).split(",") if x.strip()}
    if args.refresh_a3z_pxp_etl:
        script = ROOT / "build_a3z_pxp_etl.py"
        if script.is_file():
            print("Refreshing A3Z PXP ETL before feature merge…", flush=True)
            subprocess.run([sys.executable, str(script)], check=True)
        else:
            print(f"WARN --refresh-a3z-pxp-etl requested but missing {script}", flush=True)

    pm = pd.read_csv(args.player_meta)
    pm["player_id"] = pd.to_numeric(pm["player_id"], errors="coerce").astype("Int64")
    pm = pm[pm["player_id"].notna()].copy()
    pm["player_id"] = pm["player_id"].astype(int)
    name_map_unique, name_map_by_pos = _build_name_id_maps(pm)

    # Spine: player-season rows from A3Z component weights, expanded with A3Z season rows.
    sp = pd.read_csv(args.a3z_weights)
    sp["player_id"] = pd.to_numeric(sp["player_id"], errors="coerce").astype("Int64")
    sp["season_id"] = pd.to_numeric(sp["season_id"], errors="coerce").astype("Int64")
    sp = sp[sp["player_id"].notna() & sp["season_id"].notna()].copy()
    sp["player_id"] = sp["player_id"].astype(int)
    sp["season_id"] = sp["season_id"].astype(int)
    sp = sp[sp["season_id"].isin(target_seasons)].copy()
    sp_keys = sp[["player_id", "season_id"]].drop_duplicates().copy()

    # Add player-season keys observed in A3Z season modules so traded/aliased players
    # without a weight row are not dropped from the full model universe.
    a3z_sp = pd.read_csv(args.a3z_season, usecols=["player", "position", "season"])
    a3z_sp["season_id"] = a3z_sp["season"].map(_season_str_to_id)
    a3z_sp = a3z_sp[a3z_sp["season_id"].isin(target_seasons)].copy()
    a3z_sp = _attach_player_ids(
        a3z_sp,
        name_col="player",
        pos_col="position",
        unique_map=name_map_unique,
        by_pos_map=name_map_by_pos,
    )
    a3z_sp = a3z_sp[a3z_sp["player_id"].notna()].copy()
    a3z_sp["player_id"] = a3z_sp["player_id"].astype(int)
    a3z_sp["season_id"] = a3z_sp["season_id"].astype(int)
    a3z_sp = a3z_sp[["player_id", "season_id"]].drop_duplicates()

    sp_keys = pd.concat([sp_keys, a3z_sp], ignore_index=True).drop_duplicates(["player_id", "season_id"])

    # Add EH-observed player-season keys so years with sparse/missing A3Z (e.g. 20-21/21-22)
    # still appear in the model universe and can be scored from EH features.
    eh_key_parts: list[pd.DataFrame] = []
    if args.eh_rapm.is_file():
        eh_k = pd.read_csv(args.eh_rapm, usecols=["Player", "Position", "Season"])
        eh_k["season_id"] = eh_k["Season"].map(_eh_season_to_id)
        eh_k = eh_k[eh_k["season_id"].isin(target_seasons)].copy()
        eh_k = _attach_player_ids(
            eh_k,
            name_col="Player",
            pos_col="Position",
            unique_map=name_map_unique,
            by_pos_map=name_map_by_pos,
        )
        eh_k = eh_k[eh_k["player_id"].notna()].copy()
        eh_k["player_id"] = eh_k["player_id"].astype(int)
        eh_k["season_id"] = eh_k["season_id"].astype(int)
        eh_key_parts.append(eh_k[["player_id", "season_id"]].drop_duplicates())
    if args.eh_std.is_file():
        eh_s = pd.read_csv(args.eh_std, usecols=["Player", "Position", "Season"])
        eh_s["season_id"] = eh_s["Season"].map(_eh_season_to_id)
        eh_s = eh_s[eh_s["season_id"].isin(target_seasons)].copy()
        eh_s = _attach_player_ids(
            eh_s,
            name_col="Player",
            pos_col="Position",
            unique_map=name_map_unique,
            by_pos_map=name_map_by_pos,
        )
        eh_s = eh_s[eh_s["player_id"].notna()].copy()
        eh_s["player_id"] = eh_s["player_id"].astype(int)
        eh_s["season_id"] = eh_s["season_id"].astype(int)
        eh_key_parts.append(eh_s[["player_id", "season_id"]].drop_duplicates())
    if eh_key_parts:
        eh_keys = pd.concat(eh_key_parts, ignore_index=True).drop_duplicates(["player_id", "season_id"])
        sp_keys = pd.concat([sp_keys, eh_keys], ignore_index=True).drop_duplicates(["player_id", "season_id"])
    sp = _prefix_except(sp, "a3z_w_", {"player_id", "season_id"})

    out = sp_keys.merge(sp, on=["player_id", "season_id"], how="left")
    out = out.merge(pm[["player_id", "player_name", "position_group"]].drop_duplicates("player_id"), on="player_id", how="left")

    def _merge_ptv_table(path: Path, prefix: str) -> None:
        nonlocal out
        if not path.is_file():
            print(f"WARN skip PTV merge — missing {path}", flush=True)
            return
        ptv = pd.read_csv(path)
        ptv["player_id"] = pd.to_numeric(ptv["player_id"], errors="coerce").astype("Int64")
        ptv = ptv[ptv["player_id"].notna()].copy()
        ptv["player_id"] = ptv["player_id"].astype(int)
        keep_ptv = [
            "player_id",
            "season_scope",
            "n_seasons_in_window",
            "toi_5v5_min",
            "toi_5v5_sec",
            "reliability_factor",
            "leaderboard_qualified",
            "confidence_tier",
            "archetype_public",
            "rapm_possession_offense",
            "rapm_possession_defense",
            "rapm_net_possession_impact",
            "rapm_shot_threat",
            "rapm_pre_shot_pressure",
            "rapm_threat_suppression",
            "pct_possession_offense_position",
            "pct_possession_defense_position",
            "pct_net_possession_impact_position",
            "pct_shot_threat_position",
            "pre_shot_pressure_pct_position",
            "pct_threat_suppression_position",
        ]
        keep_ptv = [c for c in keep_ptv if c in ptv.columns]
        ptv = ptv[keep_ptv].drop_duplicates("player_id", keep="last")
        ptv = _prefix_except(ptv, prefix, {"player_id"})
        out = out.merge(ptv, on="player_id", how="left")

    # PTV: per-season joint fits (when present) + trailing roll3 + pooled fallback.
    ptv_coef_seasons: list[int] = []
    if args.ptv_season_dir.is_dir():
        ptv_coef_seasons = list_season_coef_seasons(args.ptv_season_dir)
        if ptv_coef_seasons:
            ps = build_player_season_ptv_table(args.ptv_season_dir, ptv_coef_seasons)
            ps = _prefix_except(ps, "ptv_season_", {"player_id", "season_id"})
            out = out.merge(ps, on=["player_id", "season_id"], how="left")
            print(
                f"Merged per-season PTV from {args.ptv_season_dir} "
                f"coef_seasons={ptv_coef_seasons} rows={len(ps):,}",
                flush=True,
            )
            roll_end = sorted(set(target_seasons) & set(ptv_coef_seasons))
            try:
                roll3 = build_trailing_roll3_ptv_table(args.ptv_season_dir, roll_end)
                roll3 = _prefix_except(roll3, "ptv_roll3_", {"player_id", "season_id"})
                out = out.merge(roll3, on=["player_id", "season_id"], how="left")
                print(
                    f"Merged trailing 3yr PTV end_seasons={roll_end} rows={len(roll3):,}",
                    flush=True,
                )
            except ValueError as e:
                print(f"WARN trailing roll3 PTV unavailable ({e}); trying static roll3 CSV", flush=True)
                _merge_ptv_table(args.ptv_rolling, "ptv_roll3_")
        else:
            print(f"WARN no seasonal PTV coef files under {args.ptv_season_dir}", flush=True)
            _merge_ptv_table(args.ptv_rolling, "ptv_roll3_")
    else:
        print(f"WARN skip per-season PTV — missing {args.ptv_season_dir}", flush=True)
        _merge_ptv_table(args.ptv_rolling, "ptv_roll3_")

    _merge_ptv_table(args.ptv, "ptv_pooled_")

    # Coalesced PTV for 8-category model: season > rolling > pooled.
    for stem in PTV_RAPM_STEMS:
        candidates = [
            f"ptv_season_{stem}",
            f"ptv_roll3_{stem}",
            f"ptv_roll3_rapm_{stem}",
            f"ptv_pooled_rapm_{stem}",
        ]
        present = [c for c in candidates if c in out.columns]
        if not present:
            continue
        best = pd.to_numeric(out[present[0]], errors="coerce")
        for c in present[1:]:
            best = best.fillna(pd.to_numeric(out[c], errors="coerce"))
        out[f"ptv_best_{stem}"] = best
        out[f"ptv_coalesced_{stem}"] = best

    # A3Z season rates/modules (map player name -> id).
    a3z = pd.read_csv(args.a3z_season)
    a3z, a3z_integrity = sanitize_a3z_rates(a3z)
    flagged = {key: value for key, value in a3z_integrity.items() if value}
    if flagged:
        print(f"A3Z integrity quarantine: {flagged}", flush=True)
    a3z_coverage = assess_a3z_season_coverage(a3z)
    a3z = a3z.merge(a3z_coverage, on="season", how="left", validate="many_to_one")
    partial = a3z_coverage.loc[
        a3z_coverage["source_coverage_status"].eq("PARTIAL_UNCERTIFIED"),
        ["season", "source_coverage_ratio"],
    ]
    if len(partial):
        print(
            "WARN partial A3Z source season(s); do not use as a release gate:\n"
            + partial.to_string(index=False),
            flush=True,
        )
    a3z["season_id"] = a3z["season"].map(_season_str_to_id)
    a3z = a3z[a3z["season_id"].isin(target_seasons)].copy()
    a3z = _attach_player_ids(
        a3z,
        name_col="player",
        pos_col="position",
        unique_map=name_map_unique,
        by_pos_map=name_map_by_pos,
    )
    a3z = a3z[a3z["player_id"].notna()].copy()
    a3z["player_id"] = a3z["player_id"].astype(int)
    a3z = a3z.drop(columns=[c for c in ["name_key"] if c in a3z.columns])
    a3z = _prefix_except(a3z, "a3z_", {"player_id", "season_id"})
    a3z = _aggregate_player_season_rows(
        a3z,
        toi_col_candidates=["a3z_toi_5v5_min"],
        team_col_candidates=["a3z_team"],
    )
    out = out.merge(a3z, on=["player_id", "season_id"], how="left")

    # Definition-compatible raw Game Stats rates.  These live under an
    # ``a3z_raw_`` namespace and do not silently replace legacy aggregates.
    if args.a3z_game_stats_rates.is_file():
        if args.a3z_game_stats_rates.suffix.lower() == ".parquet":
            a3z_raw = pd.read_parquet(args.a3z_game_stats_rates)
        else:
            a3z_raw = pd.read_csv(args.a3z_game_stats_rates)
        a3z_raw["season_id"] = a3z_raw["season"].map(_season_str_to_id)
        a3z_raw = a3z_raw[a3z_raw["season_id"].isin(target_seasons)].copy()
        a3z_raw = _attach_player_ids(
            a3z_raw,
            name_col="player",
            pos_col="position",
            unique_map=name_map_unique,
            by_pos_map=name_map_by_pos,
        )
        a3z_raw = a3z_raw[a3z_raw["player_id"].notna()].copy()
        a3z_raw["player_id"] = a3z_raw["player_id"].astype(int)
        a3z_raw = a3z_raw.drop(columns=[c for c in ["name_key"] if c in a3z_raw.columns])
        a3z_raw = _prefix_except(a3z_raw, "a3z_raw_", {"player_id", "season_id"})
        a3z_raw = _aggregate_player_season_rows(
            a3z_raw,
            toi_col_candidates=["a3z_raw_toi_5v5_min"],
            team_col_candidates=["a3z_raw_team"],
        )
        out = out.merge(a3z_raw, on=["player_id", "season_id"], how="left")
        print(
            f"Merged definition-compatible A3Z Game Stats rates from "
            f"{args.a3z_game_stats_rates} rows={len(a3z_raw):,}",
            flush=True,
        )
    else:
        print(
            f"WARN skip definition-compatible A3Z Game Stats rates — missing "
            f"{args.a3z_game_stats_rates}",
            flush=True,
        )

    # EH RAPM season table.
    eh_r = pd.read_csv(args.eh_rapm)
    eh_r["season_id"] = eh_r["Season"].map(_eh_season_to_id)
    eh_r = eh_r[eh_r["season_id"].isin(target_seasons)].copy()
    eh_r = _attach_player_ids(
        eh_r,
        name_col="Player",
        pos_col="Position",
        unique_map=name_map_unique,
        by_pos_map=name_map_by_pos,
    )
    eh_r = eh_r[eh_r["player_id"].notna()].copy()
    eh_r["player_id"] = eh_r["player_id"].astype(int)
    eh_r = eh_r.drop(columns=[c for c in ["name_key"] if c in eh_r.columns])
    eh_r = _prefix_except(eh_r, "eh_rapm_", {"player_id", "season_id"})
    eh_r = _aggregate_player_season_rows(
        eh_r,
        toi_col_candidates=["eh_rapm_TOI"],
        team_col_candidates=["eh_rapm_Team"],
        gp_col_candidates=["eh_rapm_GP"],
    )
    out = out.merge(eh_r, on=["player_id", "season_id"], how="left")
    out = _add_hockeystats_prior_rapm_proxy(
        out,
        toi_col="eh_rapm_TOI",
        xgf_col="eh_rapm_xGF/60",
        xga_col="eh_rapm_xGA/60",
        shrink_k_minutes=float(args.hs_prior_shrink_k_minutes),
    )

    def _load_eh_std(path: Path, prefix: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        d["season_id"] = d["Season"].map(_eh_season_to_id)
        d = d[d["season_id"].isin(target_seasons)].copy()
        d = _attach_player_ids(
            d,
            name_col="Player",
            pos_col="Position",
            unique_map=name_map_unique,
            by_pos_map=name_map_by_pos,
        )
        d = d[d["player_id"].notna()].copy()
        d["player_id"] = d["player_id"].astype(int)
        d = d.drop(columns=[c for c in ["name_key"] if c in d.columns])
        d = _prefix_except(d, prefix, {"player_id", "season_id"})
        d = _aggregate_player_season_rows(
            d,
            toi_col_candidates=[f"{prefix}TOI"],
            team_col_candidates=[f"{prefix}Team"],
            gp_col_candidates=[f"{prefix}GP"],
        )
        return d

    # The unqualified eh_std_ source now defaults to strict 5v5 Standard rates.
    # Keep its strength scope explicit in the merged feature table so downstream
    # models and audits can distinguish it from legacy all-strength/EV builds.
    out = out.merge(_load_eh_std(args.eh_std, "eh_std_"), on=["player_id", "season_id"], how="left")
    eh_std_name = args.eh_std.name.lower()
    if "5v5" in eh_std_name:
        eh_std_scope = "5v5"
    elif "_ev_" in eh_std_name:
        eh_std_scope = "EV"
    elif "_all_" in eh_std_name:
        eh_std_scope = "all_strengths"
    else:
        eh_std_scope = "unknown"
    out["eh_std_strength_scope"] = eh_std_scope
    if args.eh_std_pp.is_file():
        out = out.merge(_load_eh_std(args.eh_std_pp, "eh_std_pp_"), on=["player_id", "season_id"], how="left")
    if args.eh_std_sh.is_file():
        out = out.merge(_load_eh_std(args.eh_std_sh, "eh_std_sh_"), on=["player_id", "season_id"], how="left")

    # A3Z PXP team-season context merged to player-season via team+season.
    if args.a3z_pxp_events.is_file():
        pxp_team = _build_a3z_pxp_team_context(args.a3z_pxp_events, target_seasons)
        if len(pxp_team):
            team_cols = [c for c in ["a3z_team", "eh_std_Team", "eh_rapm_Team"] if c in out.columns]
            out["team_abbr"] = ""
            for c in team_cols:
                cand = out[c].map(_normalize_team_abbr)
                out["team_abbr"] = out["team_abbr"].where(out["team_abbr"].astype(str).str.len() > 0, cand)
            out = out.merge(pxp_team, on=["season_id", "team_abbr"], how="left")
            print(f"Merged A3Z PXP team context from {args.a3z_pxp_events} rows={len(pxp_team):,}", flush=True)
        else:
            print(f"WARN A3Z PXP loaded but no eligible season rows from {args.a3z_pxp_events}", flush=True)
    else:
        print(f"WARN skip A3Z PXP merge — missing {args.a3z_pxp_events}", flush=True)

    # YoY outcomes (available for 20232024 and 20242025).
    o_parts = []
    for p in [args.outcomes_a, args.outcomes_b]:
        if p.is_file():
            o_parts.append(pd.read_parquet(p))
    if o_parts:
        oy = pd.concat(o_parts, ignore_index=True)
        oy["player_id"] = pd.to_numeric(oy["player_id"], errors="coerce").astype("Int64")
        oy["season_id"] = pd.to_numeric(oy["season_id"], errors="coerce").astype("Int64")
        oy = oy[oy["player_id"].notna() & oy["season_id"].notna()].copy()
        oy["player_id"] = oy["player_id"].astype(int)
        oy["season_id"] = oy["season_id"].astype(int)
        oy = oy[oy["season_id"].isin(target_seasons)]
        oy = _prefix_except(oy, "yoy_", {"player_id", "season_id"})
        oy = oy.drop_duplicates(["player_id", "season_id"], keep="last")
        out = out.merge(oy, on=["player_id", "season_id"], how="left")

    # Add quality flags.
    def _has_ptv(prefix: str) -> pd.Series:
        cols = [f"{prefix}{stem}" for stem in PTV_RAPM_STEMS if f"{prefix}{stem}" in out.columns]
        return out[cols].notna().any(axis=1) if cols else pd.Series(False, index=out.index)

    out["has_ptv_season"] = _has_ptv("ptv_season_")
    out["has_ptv_roll3"] = _has_ptv("ptv_roll3_") | _has_ptv("ptv_roll3_rapm_")
    out["has_ptv_pooled"] = _has_ptv("ptv_pooled_rapm_")
    out["has_ptv_best"] = _has_ptv("ptv_best_")
    out["has_ptv_coalesced"] = _has_ptv("ptv_coalesced_")
    out["has_a3z_season"] = out.filter(regex=r"^a3z_").notna().any(axis=1)
    out["has_a3z_raw_game_stats"] = out.filter(regex=r"^a3z_raw_").notna().any(axis=1)
    out["has_eh_rapm"] = out.filter(regex=r"^eh_rapm_xG±/60$|^eh_rapm_xGF/60$|^eh_rapm_xGA/60$").notna().any(axis=1)
    if "has_hs_prior_rapm" not in out.columns:
        out["has_hs_prior_rapm"] = False
    out["has_eh_std"] = out.filter(regex=r"^eh_std_ixG/60$|^eh_std_Points/60$").notna().any(axis=1)
    out["has_eh_std_pp"] = out.filter(regex=r"^eh_std_pp_ixG/60$|^eh_std_pp_Points/60$").notna().any(axis=1)
    out["has_eh_std_sh"] = out.filter(regex=r"^eh_std_sh_iBLK/60$|^eh_std_sh_iPEN±/60$").notna().any(axis=1)
    out["has_yoy_outcomes"] = out.filter(regex=r"^yoy_").notna().any(axis=1)
    out["has_a3z_pxp_team_context"] = out.filter(regex=r"^a3z_pxp_team_").notna().any(axis=1)

    out = out.sort_values(["season_id", "player_name", "player_id"], na_position="last").reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    print(f"Wrote {args.out_csv} rows={len(out):,} cols={len(out.columns):,}")
    print(f"Wrote {args.out_parquet} rows={len(out):,} cols={len(out.columns):,}")
    print("Rows by season:")
    print(out.groupby("season_id", as_index=False).size().to_string(index=False))
    print("Coverage flags:")
    cov_cols = [
        "has_ptv_season",
        "has_ptv_best",
        "has_ptv_roll3",
        "has_ptv_pooled",
        "has_a3z_season",
        "has_a3z_raw_game_stats",
        "has_eh_rapm",
        "has_hs_prior_rapm",
        "has_eh_std",
        "has_eh_std_pp",
        "has_eh_std_sh",
        "has_yoy_outcomes",
        "has_a3z_pxp_team_context",
    ]
    print((out[cov_cols].mean() * 100.0).round(2).to_string())


if __name__ == "__main__":
    main()
