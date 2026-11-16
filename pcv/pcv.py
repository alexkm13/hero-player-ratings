"""
PCV/PTV (possession control/threat value) from xT deltas and possession summaries.

V2 (preferred when ``data/models/pcv_v2.joblib`` exists): calibrated shot xG
(geometry + rebound/rush/shot type/manpower) combined as survival xG at the
possession grain, plus an 8-second EPV threat head. See
``docs/modeling/pcv_v2/modeling_charter.md``.

Legacy heuristic still in this module:

- Per-event: xT_before, xT_after, xT_delta, xG (shots), xT_at_loc.
- Per-possession: sequence-aware PTV built from peak threat, shot value,
  cumulative positive threat, and sustained pressure.
- Player credit: event/role-weighted possession shares or xT_delta to p1_ID.
- Defensive: opponent PTV against plus direct negative-threat suppression events.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from coords import align_period_one_orientation, to_attacking_positive_x
from pbp_order import sort_pbp_chronological
from possession import tag_possessions
from xt.hockeynxt_core import XTFit, fit_xT
from xt.nhl_actions import build_actions_from_pbp

_SHOT_FOR_XG = frozenset({"SHOT", "MISS", "GOAL"})

DEFAULT_SEQUENCE_WEIGHTS = {
    "peak": 0.68,
    "cumulative": 0.22,
    "sustained": 0.10,
}


def _opp_tricode(tricode: str | Any, home: str, away: str) -> str:
    t = str(tricode).strip()
    if t == str(home).strip():
        return str(away)
    if t == str(away).strip():
        return str(home)
    return ""


def _sort_pbp(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    return sort_pbp_chronological(d, drop_tie_col=True)


def prepare_attacking_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """Period-unify, Ev_Team attacking +x; xC/yC in feet (attacking frame for Ev_Team)."""
    d = _sort_pbp(pbp)
    a = align_period_one_orientation(d)
    return to_attacking_positive_x(a)


def _smooth_grid_value_nhl_series(
    grid: np.ndarray,
    x_atk: pd.Series,
    y_atk: pd.Series,
) -> np.ndarray:
    """
    Bilinear rink value from a fitted xT/xG grid.

    The fitted model remains the same Markov grid, but event values are read
    from a smooth surface between cell centers so nearly identical events do
    not jump only because they cross a cell boundary.
    """
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.full(len(x_atk), np.nan, dtype=float)
    w, l = arr.shape
    sx = (pd.to_numeric(x_atk, errors="coerce").astype(float) + 100.0).clip(0.0, 200.0)
    sy = (pd.to_numeric(y_atk, errors="coerce").astype(float) + 42.5).clip(0.0, 85.0)
    ok = sx.notna() & sy.notna()
    out = np.full(len(sx), np.nan, dtype=float)
    if not bool(ok.any()):
        return out

    xf = (sx[ok].to_numpy(float) / 200.0) * l - 0.5
    yf_bottom = (sy[ok].to_numpy(float) / 85.0) * w - 0.5
    yf = (w - 1) - yf_bottom

    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xf - x0
    wy = yf - y0

    x0 = np.clip(x0, 0, l - 1)
    x1 = np.clip(x1, 0, l - 1)
    y0 = np.clip(y0, 0, w - 1)
    y1 = np.clip(y1, 0, w - 1)

    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]
    vals = (
        (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
        + wy * ((1.0 - wx) * v10 + wx * v11)
    )
    out[ok.to_numpy()] = vals
    return out


def add_per_event_xt(
    pbp: pd.DataFrame,
    xt: XTFit,
    *,
    smooth_location_values: bool = True,
) -> pd.DataFrame:
    """
    Add columns: xT_at_loc, xT_before, xT_after, xT_delta, xG (shots only, else NaN).

    - *xT_at_loc* — Markov xT in the binned cell of this event (used for “peak” PCV on shots
      and non-shots).
    - *xT_before* — xT at the puck *before* the play: same-possession previous event’s location;
      if there is no prior in the segment, current location.
    - *xT_after* — non-shots: xT at (xC, yC). SHOT/MISS/GOAL: **xG** (P(goal|shot) in cell),
      i.e. threat resolves into the shot’s goal probability.
    """
    if pbp.empty:
        return pbp
    d = pbp if "possession_id" in pbp.columns else tag_possessions(pbp)
    d = prepare_attacking_frame(d)
    x, y = d["xC"], d["yC"]
    if smooth_location_values:
        xT_at_loc = _smooth_grid_value_nhl_series(xt.xT, x, y)
        xG = _smooth_grid_value_nhl_series(xt.p_goal_given_shot, x, y)
    else:
        xT_at_loc = xt.xT_nhl_series(x, y)
        xG = xt.xG_nhl_series(x, y)
    d = d.assign(xT_at_loc=xT_at_loc, _xG_cell=xG)
    gcol = (
        d["NHL_game_id"] if "NHL_game_id" in d.columns else d["Game_Id"] if "Game_Id" in d.columns else 0
    )
    d["_gid"] = gcol
    prev_poss = d.groupby("_gid", sort=False)["possession_id"].shift(1)
    same = (d["possession_id"] == prev_poss) & d["possession_id"].notna()
    prev_x = d.groupby("_gid", sort=False)["xC"].shift(1)
    prev_y = d.groupby("_gid", sort=False)["yC"].shift(1)
    xT_before = np.where(
        same & prev_x.notna() & prev_y.notna(),
        _smooth_grid_value_nhl_series(xt.xT, prev_x, prev_y)
        if smooth_location_values
        else xt.xT_nhl_series(prev_x, prev_y),
        xT_at_loc,
    )
    is_shot = d["Event"].isin(_SHOT_FOR_XG).to_numpy()
    xT_after = np.where(is_shot, d["_xG_cell"].to_numpy(), xT_at_loc)
    d["xT_before"] = xT_before
    d["xT_after"] = xT_after
    d["xT_delta"] = d["xT_after"] - d["xT_before"]
    d["xG"] = np.where(is_shot, d["_xG_cell"].to_numpy(), np.nan)
    d = d.drop(columns=[c for c in ("_xG_cell", "_gid") if c in d.columns])
    return d


def _first_non_null_str(s: pd.Series) -> Any:
    for v in s:
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return None


def possession_pcv_table(
    events_with_xt: pd.DataFrame,
    *,
    sequence_weights: dict[str, float] | None = None,
    cumulative_cap: float = 0.35,
) -> pd.DataFrame:
    """
    One row per (Game_Id, possession_id):
    *poss_team*, *max_shot_xG*, *peak_xT*, *pcv*, *n_events* and component decomposition.

    *peak_xT* = max over event rows of (xT_before, xT_at_loc). *peak_pcv* is the
    legacy peak-only value. The current *pcv* is sequence-aware and uses peak
    value, cumulative positive threat, and sustained pressure while keeping
    the legacy peak value as a floor.
    *poss_team* = first non-null *Ev_Team* in the segment.
    """
    d = events_with_xt.copy()
    gcols: list[str]
    if "NHL_game_id" in d.columns and "possession_id" in d.columns:
        gcols = ["NHL_game_id", "possession_id"]
    else:
        gcols = [c for c in ("Game_Id", "possession_id") if c in d.columns]
    if len(gcols) < 2 or d.empty:
        return pd.DataFrame(
            columns=[
                "NHL_game_id",
                "Game_Id",
                "possession_id",
                "poss_team",
                "max_shot_xG",
                "peak_xT",
                "peak_pcv",
                "cumulative_threat_added",
                "sustained_threat",
                "pcv",
                "pcv_sequence",
                "shot_component",
                "threat_component",
                "no_shot_threat_component",
                "pcv_source",
                "shot_exists",
                "n_events",
            ]
        )
    d["_is_shot"] = d["Event"].isin(_SHOT_FOR_XG)
    if "xG" not in d.columns:
        d["xG"] = np.where(d["_is_shot"], d.get("xG", np.nan), np.nan)
    if "xT_at_loc" not in d.columns:
        raise KeyError("events must include xT_at_loc from add_per_event_xt")
    d["_peak_x"] = np.fmax(
        np.asarray(d["xT_before"], dtype=np.float64),
        np.asarray(d["xT_at_loc"], dtype=np.float64),
    )
    d["_xG_row"] = np.where(d["_is_shot"], d["xG"], np.nan)
    xtd = pd.to_numeric(d.get("xT_delta", pd.Series(np.nan, index=d.index)), errors="coerce").fillna(0.0)
    d["_positive_xt_delta"] = xtd.clip(lower=0.0)
    d["_pressure_x"] = pd.to_numeric(d["_peak_x"], errors="coerce").clip(lower=0.0)
    g = d.groupby(gcols, sort=False)
    poss_team = g["Ev_Team"].agg(_first_non_null_str)
    n_events = g.size()
    max_shot_xG = g["_xG_row"].max()
    peak_xT = g["_peak_x"].max()
    cumulative_threat = g["_positive_xt_delta"].sum().clip(lower=0.0, upper=float(cumulative_cap))
    sustained_threat = g["_pressure_x"].mean().fillna(0.0)
    out = (
        pd.DataFrame(
            {
                "poss_team": poss_team,
                "n_events": n_events,
                "max_shot_xG": max_shot_xG,
                "peak_xT": peak_xT,
                "cumulative_threat_added": cumulative_threat,
                "sustained_threat": sustained_threat,
            }
        )
        .reset_index()
    )
    mshot = out["max_shot_xG"].to_numpy()
    pxt = out["peak_xT"].to_numpy()
    peak_pcv = np.fmax(
        np.where(np.isfinite(mshot), mshot, 0.0),
        np.where(np.isfinite(pxt), pxt, 0.0),
    )
    wts = dict(DEFAULT_SEQUENCE_WEIGHTS)
    if sequence_weights:
        wts.update({str(k): float(v) for k, v in sequence_weights.items()})
    seq = (
        float(wts.get("peak", 0.68)) * peak_pcv
        + float(wts.get("cumulative", 0.22)) * out["cumulative_threat_added"].to_numpy(float)
        + float(wts.get("sustained", 0.10)) * out["sustained_threat"].to_numpy(float)
    )
    out["peak_pcv"] = peak_pcv
    out["pcv_sequence"] = np.clip(np.fmax(peak_pcv, seq), 0.0, 1.0)
    out["pcv"] = out["pcv_sequence"]
    shot_comp = np.where(np.isfinite(mshot), mshot, 0.0)
    threat_comp = out["pcv_sequence"].to_numpy(float)
    shot_exists = shot_comp > 0.0
    out["shot_component"] = shot_comp
    out["threat_component"] = threat_comp
    out["no_shot_threat_component"] = np.where(shot_exists, 0.0, threat_comp)
    out["pcv_source"] = np.where(shot_comp >= threat_comp, "shot", "threat")
    out["shot_exists"] = shot_exists
    id_col = gcols[0]
    out_cols = [
        id_col,
        "possession_id",
        "poss_team",
        "max_shot_xG",
        "peak_xT",
        "peak_pcv",
        "cumulative_threat_added",
        "sustained_threat",
        "pcv",
        "pcv_sequence",
        "shot_component",
        "threat_component",
        "no_shot_threat_component",
        "pcv_source",
        "shot_exists",
        "n_events",
    ]
    if id_col == "NHL_game_id" and "Game_Id" in d.columns:
        # carry 5-digit id for display / joins
        g2 = d.groupby(gcols, sort=False)["Game_Id"].first()
        out = out.merge(
            g2.reset_index().rename(columns={"Game_Id": "Game_Id"}),
            on=gcols,
            how="left",
        )
        return out[[
            "NHL_game_id",
            "Game_Id",
            "possession_id",
            "poss_team",
            "max_shot_xG",
            "peak_xT",
            "peak_pcv",
            "cumulative_threat_added",
            "sustained_threat",
            "pcv",
            "pcv_sequence",
            "shot_component",
            "threat_component",
            "no_shot_threat_component",
            "pcv_source",
            "shot_exists",
            "n_events",
        ]]
    return out[out_cols]


def attribute_possession_pcv_event_shares(
    pbp: pd.DataFrame,
    poss_table: pd.DataFrame,
    *,
    id_col: str = "p1_ID",
) -> pd.Series:
    """
    **Offensive** possession value split evenly across events: for each PBP row in a
    possession, credit *pcv* / *n_events* to *p1_ID* when *Ev_Team* is the possession
    offense (*poss_team* from *poss_table*). Rows without *p1_ID* do not receive that
    event’s share (the mass is left unassigned).

    *pbp* must include *NHL_game_id* and *possession_id* (10-digit game id in ingest cache
    convention). *poss_table* is the output of :func:`possession_pcv_table` for the same games.

    Returns a *Series* indexed by *player_id* (int) with the summed PCV “credits” (0–1 scale
    in the xT/PCV sense). Multiply or compare with :func:`player_pcv_per_60` when you have TOI.
    """
    gcols = ["NHL_game_id", "possession_id"]
    if pbp.empty or poss_table.empty:
        return pd.Series(dtype="float64")
    if not all(c in pbp.columns for c in gcols) or "possession_id" not in pbp.columns:
        raise KeyError("pbp must include NHL_game_id and possession_id")
    for c in (*gcols, "poss_team", "pcv", "n_events"):
        if c not in poss_table.columns:
            raise KeyError(f"poss_table missing {c!r}")
    pt = poss_table[[*gcols, "poss_team", "pcv", "n_events"]].drop_duplicates(subset=gcols)
    m = pbp.merge(pt, on=gcols, how="inner")
    if m.empty:
        return pd.Series(dtype="float64")
    same = m["Ev_Team"].astype(str).str.strip() == m["poss_team"].astype(str).str.strip()
    m = m.loc[same & m[id_col].notna()]
    if m.empty:
        return pd.Series(dtype="float64")
    m = m.copy()
    m[id_col] = pd.to_numeric(m[id_col], errors="coerce")
    m = m[m[id_col].notna()]
    m["_w"] = m["pcv"].astype(float) / m["n_events"].replace(0, np.nan).clip(lower=1).astype(float)
    return m.groupby(m[id_col].astype(int), sort=False)["_w"].sum()


def attribute_pcv(
    possession_events: pd.DataFrame,
    id_col: str = "p1_ID",
) -> dict[Any, float]:
    """
    Credit the listed player (*p1_ID*) on each event with *xT_delta* (signed).
    Off-ice or missing ID → no credit. Negative deltas (e.g. giveaways) stay negative.
    """
    out: dict[Any, float] = defaultdict(float)
    if possession_events.empty or "xT_delta" not in possession_events.columns:
        return dict(out)
    p = possession_events[pd.notna(possession_events.get(id_col))]
    if p.empty:
        return dict(out)
    ids = pd.to_numeric(p[id_col], errors="coerce")
    w = p["xT_delta"].to_numpy()
    for pid, d in zip(ids, w):
        if pd.isna(pid):
            continue
        k = int(pid)
        v = 0.0 if pd.isna(d) else float(d)
        out[k] += v
    return dict(out)


def player_xt_delta_totals(
    events_with_xt: pd.DataFrame,
    id_col: str = "p1_ID",
) -> pd.Series:
    """Per-player sum of *xT_delta* (all games / rows in frame)."""
    t = attribute_pcv(events_with_xt, id_col=id_col)
    return pd.Series(t, name="xT_delta_credits", dtype="float64").sort_index()


def player_pcv_per_60(
    total_pcv_credits: float | pd.Series,
    toi_sec: float | pd.Series,
) -> float | pd.Series:
    """
    (total / TOI_s) * 3600. If *total_pcv_credits* is a Series indexed by player_id, align *toi_sec*
    by index. Missing or zero TOI → NaN for that player.
    """
    if isinstance(total_pcv_credits, pd.Series):
        t = toi_sec if isinstance(toi_sec, pd.Series) else pd.Series(toi_sec, index=total_pcv_credits.index)
        t = t.reindex(total_pcv_credits.index)
        return (total_pcv_credits.astype("float64") / t.replace(0, np.nan)) * 3600.0
    if toi_sec is None or (not isinstance(toi_sec, pd.Series) and (pd.isna(toi_sec) or float(toi_sec) <= 0)):
        return float("nan")
    if isinstance(toi_sec, pd.Series):
        raise TypeError("scalar total_pcv_credits with Series toi_sec is ambiguous; pass both as Series")
    return (float(total_pcv_credits) / float(toi_sec)) * 3600.0


def defensive_pcv_team_against(
    poss_table: pd.DataFrame,
    home: str,
    away: str,
) -> pd.DataFrame:
    """
    Team-level: defending side gets the **negative** of the offensive PCV in each possession
    (no skater split — use shifts + on-ice for player-level DPCV).
    *def_team* = opponent of *poss_team*.
    """
    if poss_table.empty:
        return pd.DataFrame(
            columns=["Game_Id", "possession_id", "poss_team", "def_team", "pcv_against", "opponent_pcv"]
        )
    o = poss_table.copy()
    o["def_team"] = o["poss_team"].apply(lambda t: _opp_tricode(t, str(home), str(away)))
    o["opponent_pcv"] = o["pcv"]
    o["pcv_against"] = -o["opponent_pcv"]
    return o[
        [
            "Game_Id",
            "possession_id",
            "poss_team",
            "def_team",
            "pcv_against",
            "opponent_pcv",
        ]
    ]


def add_defensive_pcv_to_players(
    on_ice_rows: pd.DataFrame,
    poss_table: pd.DataFrame,
    *,
    home: str,
    away: str,
    player_id_col: str = "player_id",
    def_team_col: str = "team",
    game_id_col: str = "Game_Id",
) -> pd.DataFrame:
    """
    *on_ice_rows* — after you align shifts to (Game_Id, possession_id) / defending skaters, each
    row has *def_team* (column *def_team_col*), *player_id*, and matching *game_id_col*.

    Keeps only rows where *def_team* is the opponent of *poss_team*; *def_pcv* = - *opponent* PCV
    (what you “allowed” as the defending side).
    """
    if on_ice_rows.empty or poss_table.empty:
        return pd.DataFrame(
            columns=[
                game_id_col,
                "possession_id",
                def_team_col,
                player_id_col,
                "def_pcv",
            ]
        )
    p = poss_table[[game_id_col, "possession_id", "poss_team", "pcv"]].copy()
    p["_opp_poss"] = p["poss_team"].apply(
        lambda t: _opp_tricode(t, home, away) if pd.notna(t) else ""
    )

    j = on_ice_rows.merge(
        p,
        on=[game_id_col, "possession_id"],
        how="inner",
    )
    j = j[j[def_team_col] == j["_opp_poss"]]
    if j.empty:
        return pd.DataFrame(
            columns=[
                game_id_col,
                "possession_id",
                def_team_col,
                player_id_col,
                "def_pcv",
            ]
        )
    j["def_pcv"] = -j["pcv"]
    return j[[
        game_id_col,
        "possession_id",
        def_team_col,
        player_id_col,
        "def_pcv",
    ]]


def fit_and_build_event_table(
    pbp: pd.DataFrame,
    grid_x: int = 10,
    grid_y: int = 8,
) -> tuple[pd.DataFrame, XTFit]:
    """
    *fit_xT* on actions from *pbp*, then *add_per_event_xt*. One-call pipeline for a game.
    """
    act = build_actions_from_pbp(pbp)
    xt = fit_xT(act, grid_x=grid_x, grid_y=grid_y)
    ev = add_per_event_xt(pbp, xt)
    return ev, xt


def aggregate_pcv_from_pbp_list(
    pbps: list[pd.DataFrame],
    grid_x: int = 10,
    grid_y: int = 8,
) -> tuple[pd.DataFrame, XTFit]:
    """
    Stack PBP, fit a single *xT* on pooled actions, run *add_per_event_xt* on the stacked
    (multi-game) frame.
    """
    if not pbps:
        raise ValueError("pbps is empty")
    d = pd.concat(
        [tag_possessions(p) if "possession_id" not in p.columns else p for p in pbps],
        ignore_index=True,
    )
    d = sort_pbp_chronological(d, drop_tie_col=True)
    act = build_actions_from_pbp(d, with_possession=True)
    xt = fit_xT(act, grid_x=grid_x, grid_y=grid_y)
    return add_per_event_xt(d, xt), xt
