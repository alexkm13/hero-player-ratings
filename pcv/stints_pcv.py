"""
5v5 **stints** from shift-chart intervals + per-stint **PCV/60** targets for RAPM.

*Stint*: maximal interval where the same 10 skaters (5+5) are on ice; goalies excluded
from the 10. Boundaries are shift start/end; goalies and empty-net (no goalie) segments
excluded. Segments &lt; 3s dropped.

PCV in-stint: sum per-event *attributed* weight *w* to *p1_ID* on possession-offense rows, then
aggregate by *Ev_Team* (``home_pcv_credited`` / ``away_pcv_credited``). *Attribution* is
configurable: equal split, event-weighted split, A3Z-weighted split, or *xT_delta* — see
:func:`pbp_pcv_event_shares`.

**Event time vs. stint (half-open):** a stint is ``[t0, t1)``. An event with ``game_sec`` belongs
to that stint iff ``t0 &lt;= game_sec &lt; t1`` (inclusive of start, **exclusive of end**).
The same PBP/clock convention must be used for shifts and events, including *OT length*
(5:00 vs 20:00) via *is_playoff* — see :func:`pbp_dataframe_add_game_sec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ingest_shifts import (
    nhl_id_from_pbp_path,
    nhl_season_id_from_game_id,
    period_clock_to_game_seconds,
)


def game_elapsed_for_event(period: int, sec_in_period: float, *, is_playoff: bool) -> float:
    """
    Cumulative *playing* time for a PBP row, matching shift-chart
    :func:`ingest_shifts.period_clock_to_game_seconds` and *Period* + *Seconds_Elapsed*.

    *is_playoff* = True when NHL *gameType* is 3 (so OT blocks are 20:00, not 5:00 each).
    """
    return float(period_clock_to_game_seconds(int(period), float(sec_in_period), is_playoff=is_playoff))


def goalie_ids_from_pbp(pbp: dict) -> set[int]:
    """
    Every *playerId* in *rosterSpots* with *positionCode* G (dressed and listed on the PBP
    object). This includes the starter and, when listed, a backup who will enter; mid-game
    entrant usually appears in *rosterSpots* (verify edge cases in data quality).
    """
    g: set[int] = set()
    for spot in pbp.get("rosterSpots") or []:
        if (str(spot.get("positionCode") or "").upper() != "G"):
            continue
        pid = spot.get("playerId")
        if pid is not None:
            g.add(int(pid))
    return g


def home_away_from_pbp(pbp: dict) -> tuple[str, str]:
    """
    *home* / *away* tricodes from Game Center. In rare *neutral-site* (Stadium / Heritage) games
    the “home” and “away” may not match arena geography; the feed still uses this pair for
    *Ev_Team* and roster sides — be consistent, and spot-check neutral games if needed.
    """
    h, a = pbp.get("homeTeam") or {}, pbp.get("awayTeam") or {}
    return str((h or {}).get("abbrev") or "").strip(), str((a or {}).get("abbrev") or "").strip()


def pbp_dataframe_add_game_sec(
    pbp: pd.DataFrame,
    is_playoff: bool,
) -> pd.DataFrame:
    o = pbp.copy()
    o["game_sec"] = [
        game_elapsed_for_event(int(p), float(s) if s == s else 0.0, is_playoff=is_playoff)
        for p, s in zip(o["Period"], o["Seconds_Elapsed"], strict=False)
    ]
    return o


def score_state_at_t0(
    pbp: pd.DataFrame,
    t0: float,
) -> tuple[int, int]:
    """
    Cumulative *running* home/away goal counts (from PBP) at *start* of a stint, using the last
    event with game_sec &lt;= *t0*. Missing scores are forward-filled. If *t0* is before any event, 0–0.
    """
    if pbp.empty or "game_sec" not in pbp.columns:
        return 0, 0
    s = pbp.sort_values("game_sec", kind="mergesort").reset_index(drop=True)
    if "Home_Score" in s.columns:
        h = s["Home_Score"].ffill().bfill().fillna(0.0)
    else:
        h = pd.Series(0.0, index=s.index)
    if "Away_Score" in s.columns:
        a = s["Away_Score"].ffill().bfill().fillna(0.0)
    else:
        a = pd.Series(0.0, index=s.index)
    gs = s["game_sec"].to_numpy(dtype=np.float64)
    t = float(t0)
    j = int(np.searchsorted(gs, t, side="right")) - 1
    if j < 0:
        return 0, 0
    return int(h.iloc[j]), int(a.iloc[j])


PcvAttribution = Literal[
    "possession_split",
    "event_weighted_possession_split",
    "a3z_weighted_possession_split",
    "xt_delta",
]
PcvComponent = Literal["total_pcv", "shot_pcv", "threat_pcv", "no_shot_threat_pcv"]


def _opp_tricode(team: str, home: str, away: str) -> str:
    t = str(team).strip()
    if t == str(home).strip():
        return str(away).strip()
    if t == str(away).strip():
        return str(home).strip()
    return ""


def _event_role_weight(m: pd.DataFrame, component: PcvComponent) -> pd.Series:
    """
    Within-possession attribution prior from event context.

    This is deliberately simple and data-available: shots get more shot-component
    credit, positive threat-added events get more pre-shot/total credit, and late
    dangerous touches near the peak get extra credit. It is later renormalized
    within each possession, so it changes who gets credit, not total possession PTV.
    """
    idx = m.index
    event = m.get("Event", pd.Series("", index=idx)).astype(str).str.upper()
    typ = m.get("type_name", pd.Series("", index=idx)).astype(str).str.upper()
    xg = pd.to_numeric(m.get("xG", pd.Series(np.nan, index=idx)), errors="coerce").fillna(0.0)
    xtd = pd.to_numeric(m.get("xT_delta", pd.Series(np.nan, index=idx)), errors="coerce").fillna(0.0)
    xb = pd.to_numeric(m.get("xT_before", pd.Series(np.nan, index=idx)), errors="coerce")
    xa = pd.to_numeric(m.get("xT_at_loc", pd.Series(np.nan, index=idx)), errors="coerce")
    peak_touch = pd.Series(np.fmax(xb.fillna(0.0).to_numpy(float), xa.fillna(0.0).to_numpy(float)), index=idx)

    is_shot = event.isin({"SHOT", "MISS", "GOAL"}) | typ.eq("SHOT")
    is_move = typ.isin({"PASS", "MOVE", "ZONE ENTRY"}) | event.isin({"PASS", "ZONE ENTRY"})

    w = pd.Series(1.0, index=idx, dtype=float)
    if component == "shot_pcv":
        w += np.where(is_shot, 2.25, 0.0)
        w += (xg.clip(0.0, 0.35) * 8.0)
        w += xtd.clip(lower=0.0, upper=0.18) * 2.0
    elif component in {"threat_pcv", "no_shot_threat_pcv"}:
        w += np.where(is_move, 0.45, 0.0)
        w += xtd.clip(lower=0.0, upper=0.22) * 5.0
        w += peak_touch.clip(0.0, 0.35) * 2.0
    else:
        w += np.where(is_shot, 0.85, 0.0)
        w += np.where(is_move, 0.30, 0.0)
        w += xg.clip(0.0, 0.35) * 4.0
        w += xtd.clip(lower=0.0, upper=0.22) * 3.0
        w += peak_touch.clip(0.0, 0.35) * 1.5
    return w.clip(lower=0.05, upper=6.0)


def _context_multiplier(score_diff: float) -> float:
    """Neutralize score-state conservatism in the offensive stint target."""
    if not np.isfinite(score_diff):
        return 1.0
    # Leading teams generally attack less; trailing teams force offense. Keep
    # this as a small correction so it cannot overwhelm actual PTV.
    return float(np.clip(1.0 + 0.035 * np.clip(score_diff, -3.0, 3.0), 0.88, 1.12))


def pbp_pcv_event_shares(
    pbp: pd.DataFrame,
    poss: pd.DataFrame,
    *,
    attribution: PcvAttribution = "event_weighted_possession_split",
    component: PcvComponent = "total_pcv",
    near_peak_seconds: float = 0.0,
    near_peak_fraction: float = 0.0,
    near_peak_events_before: int = 0,
    # For a3z_weighted_possession_split:
    #   maps[component] -> {player_id -> weight}
    a3z_weight_maps: dict[PcvComponent, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """
    Per-row event weight *_pos_w* for aggregating team PCV in a stint.

    *possession_split* (v1a): ``pcv / n_events`` on rows where the event is possession-offense
    and *p1_ID* is set. This gives every event in the possession the same share of *possession
    PCV* (not the same as pure *xT_delta* per event; it is more role / participation-stable).

    *xt_delta* (v1b): use *xT_delta* on the same rows (possession-offense, *p1_ID* set). Compare
    YoY / RAPM stability between v1a and v1b to see if equal-split is driving “role” signal.

    *event_weighted_possession_split*: like *possession_split*, but weights
    shot, movement, positive xT-delta, and peak-touch events before normalizing
    within the possession.

    *a3z_weighted_possession_split*: like *event_weighted_possession_split*, but reweights the
    possession-offense event shares using non-negative A3Z player-season priors, then
    renormalizes within the possession so the sum of credited shares still equals the
    possession PCV component.
    """
    gcols = ["NHL_game_id", "possession_id"]
    for c in ("NHL_game_id", "possession_id", "Ev_Team", "p1_ID"):
        if c not in pbp.columns:
            raise KeyError(f"pbp missing {c!r} (enriched PBP required)")
    comp_col_map = {
        "total_pcv": "pcv",
        "shot_pcv": "shot_component",
        "threat_pcv": "threat_component",
        "no_shot_threat_pcv": "no_shot_threat_component",
    }
    val_col = comp_col_map[str(component)]
    poss_use = poss.copy()
    # Back-compat with older possession tables that have max_shot_xG / peak_xT but not component columns.
    if val_col not in poss_use.columns:
        if str(component) == "shot_pcv" and "max_shot_xG" in poss_use.columns:
            poss_use[val_col] = pd.to_numeric(poss_use["max_shot_xG"], errors="coerce").fillna(0.0)
        elif str(component) == "threat_pcv" and "peak_xT" in poss_use.columns:
            poss_use[val_col] = pd.to_numeric(poss_use["peak_xT"], errors="coerce").fillna(0.0)
        elif str(component) == "no_shot_threat_pcv" and "peak_xT" in poss_use.columns:
            mshot = (
                pd.to_numeric(poss_use["max_shot_xG"], errors="coerce").fillna(0.0)
                if "max_shot_xG" in poss_use.columns
                else 0.0
            )
            pxt = pd.to_numeric(poss_use["peak_xT"], errors="coerce").fillna(0.0)
            poss_use[val_col] = np.where(mshot > 0.0, 0.0, pxt)
    for c in (*gcols, "poss_team", val_col, "n_events"):
        if c not in poss_use.columns:
            raise KeyError(f"poss table missing {c!r}")
    if attribution == "xt_delta" and "xT_delta" not in pbp.columns:
        raise KeyError("attribution='xt_delta' requires PBP with xT_delta (add_per_event_xt).")
    pt = poss_use[[*gcols, "poss_team", val_col, "n_events"]].drop_duplicates(subset=gcols)
    pt = pt.rename(columns={val_col: "_pcv_component_value"})
    m = pbp.merge(pt, on=gcols, how="left")
    same = m["Ev_Team"].astype(str).str.strip() == m["poss_team"].astype(str).str.strip()
    w = m["possession_id"].notna() & m["poss_team"].notna() & m["n_events"].notna()
    p1 = pd.to_numeric(m["p1_ID"], errors="coerce")
    if attribution in {"possession_split", "event_weighted_possession_split", "a3z_weighted_possession_split"}:
        def _num_col(col: str) -> pd.Series:
            if col in m.columns:
                return pd.to_numeric(m[col], errors="coerce")
            return pd.Series(np.nan, index=m.index, dtype=float)

        near_sec = max(0.0, float(near_peak_seconds or 0.0))
        near_frac = float(near_peak_fraction or 0.0)
        near_frac = near_frac if 0.0 < near_frac <= 1.0 else 0.0
        near_ev = max(0, int(near_peak_events_before or 0))
        base_ok = same & w & p1.notna() & m["_pcv_component_value"].notna()
        credit_ok = base_ok.copy()
        if near_sec > 0.0 or near_frac > 0.0:
            if "game_sec" in m.columns:
                offense_t = m["game_sec"].where(base_ok)
                peak_t = offense_t.groupby([m["NHL_game_id"], m["possession_id"]], sort=False).transform("max")
                if near_sec > 0.0:
                    credit_ok = credit_ok & offense_t.notna() & peak_t.notna() & (offense_t >= (peak_t - near_sec))
            if near_frac > 0.0:
                if component == "shot_pcv":
                    sig = _num_col("xG")
                elif component in {"threat_pcv", "no_shot_threat_pcv"}:
                    xb = _num_col("xT_before")
                    xa = _num_col("xT_at_loc")
                    sig = np.fmax(
                        xb.fillna(-np.inf).to_numpy(dtype=float),
                        xa.fillna(-np.inf).to_numpy(dtype=float),
                    )
                    sig = pd.Series(sig, index=m.index, dtype=float)
                else:
                    xb = _num_col("xT_before")
                    xa = _num_col("xT_at_loc")
                    xt_peak = np.fmax(
                        xb.fillna(-np.inf).to_numpy(dtype=float),
                        xa.fillna(-np.inf).to_numpy(dtype=float),
                    )
                    xg = _num_col("xG").fillna(-np.inf).to_numpy(dtype=float)
                    sig = pd.Series(np.fmax(xt_peak, xg), index=m.index, dtype=float)
                sig = pd.to_numeric(sig, errors="coerce").where(base_ok)
                peak_sig = sig.groupby([m["NHL_game_id"], m["possession_id"]], sort=False).transform("max")
                credit_ok = credit_ok & sig.notna() & peak_sig.notna() & (sig >= (near_frac * peak_sig))
        if near_ev > 0:
            if component == "shot_pcv":
                sig = _num_col("xG")
            elif component in {"threat_pcv", "no_shot_threat_pcv"}:
                xb = _num_col("xT_before")
                xa = _num_col("xT_at_loc")
                sig = np.fmax(
                    xb.fillna(-np.inf).to_numpy(dtype=float),
                    xa.fillna(-np.inf).to_numpy(dtype=float),
                )
                sig = pd.Series(sig, index=m.index, dtype=float)
            else:
                xb = _num_col("xT_before")
                xa = _num_col("xT_at_loc")
                xt_peak = np.fmax(
                    xb.fillna(-np.inf).to_numpy(dtype=float),
                    xa.fillna(-np.inf).to_numpy(dtype=float),
                )
                xg = _num_col("xG").fillna(-np.inf).to_numpy(dtype=float)
                sig = pd.Series(np.fmax(xt_peak, xg), index=m.index, dtype=float)
            sig = pd.to_numeric(sig, errors="coerce").where(base_ok, -np.inf)
            grp = [m["NHL_game_id"], m["possession_id"]]
            ord_in_pos = m.groupby(grp, sort=False).cumcount()
            peak_ord = ord_in_pos.where(sig == sig.groupby(grp, sort=False).transform("max"))
            peak_ord = peak_ord.groupby(grp, sort=False).transform("max")
            credit_ok = credit_ok & peak_ord.notna() & (ord_in_pos <= peak_ord) & (ord_in_pos >= (peak_ord - near_ev))
        credit_n = credit_ok.groupby([m["NHL_game_id"], m["possession_id"]], sort=False).transform("sum")
        base_n = base_ok.groupby([m["NHL_game_id"], m["possession_id"]], sort=False).transform("sum")
        if attribution in {"event_weighted_possession_split", "a3z_weighted_possession_split"}:
            w_event = _event_role_weight(m, component)
            if attribution == "a3z_weighted_possession_split":
                if a3z_weight_maps is None:
                    raise KeyError("a3z_weight_maps required for a3z_weighted_possession_split")
                wm = a3z_weight_maps.get(component)
                if wm is None:
                    raise KeyError(f"a3z_weight_maps missing component {component!r}")
                pid_i = p1.astype("Int64")
                w_pid = pid_i.map(wm).astype(float).fillna(1.0).clip(lower=0.0)
                w_event = (w_event * w_pid).clip(lower=0.0, upper=12.0)

            credit_w_sum = w_event.where(credit_ok).groupby([m["NHL_game_id"], m["possession_id"]], sort=False).transform("sum")
            weighted_groups = credit_w_sum > 0.0
            pcv = m["_pcv_component_value"].astype(float)

            pos_w_weighted = np.where(
                base_ok & credit_ok & weighted_groups,
                pcv * w_event / credit_w_sum,
                0.0,
            )
            pos_w_fallback = np.where(
                base_ok & (~weighted_groups) & (base_n > 0.0),
                pcv / pd.Series(base_n, index=m.index).astype(float),
                0.0,
            )
            m["_pos_w"] = pos_w_weighted + pos_w_fallback
        else:
            denom = np.where(credit_n > 0, credit_n, base_n)
            m["_pos_w"] = np.where(
                base_ok & (denom > 0) & np.where(credit_n > 0, credit_ok, base_ok),
                m["_pcv_component_value"].astype(float) / pd.Series(denom, index=m.index).astype(float),
                0.0,
            )
    else:
        xtd = pd.to_numeric(m["xT_delta"], errors="coerce")
        m["_pos_w"] = np.where(
            same & w & p1.notna() & xtd.notna(),
            xtd.astype("float64").to_numpy(),
            0.0,
        )
    return m


@dataclass(frozen=True)
class Stint5v5:
    game_id: int
    t0: float
    t1: float
    home_skaters: frozenset[int]  # 5 player ids
    away_skaters: frozenset[int]
    has_home_goalie: bool
    has_away_goalie: bool
    # Optional context for v2 controls; leave 0/empty in v1 if unused
    season_id: int = 0


def _on_ice_during(
    t0: float,
    t1: float,
    sk: np.ndarray,  # player_id
    ss: np.ndarray,  # start
    se: np.ndarray,  # end
) -> set[int]:
    """Open-closed: shift overlaps [t0, t1) if start < t1 and end > t0."""
    out: set[int] = set()
    n = sk.shape[0]
    for i in range(n):
        if se[i] > t0 and ss[i] < t1:
            out.add(int(sk[i]))
    return out


def stints_5v5_for_game(
    game_id: int,
    shifts: pd.DataFrame,
    home_abbr: str,
    away_abbr: str,
    goalies: set[int],
    min_sec: float = 3.0,
) -> list[Stint5v5]:
    """
    Build 5v5, non-EN, duration ≥ *min_sec* s stints for one game. *shifts* must include
    *start_game_sec* / *end_game_sec* / *team_abbrev* / *player_id*.
    """
    if shifts.empty or not home_abbr or not away_abbr:
        return []
    s = shifts.copy()
    s["player_id"] = s["player_id"].astype(int)
    if "start_game_sec" not in s.columns or "end_game_sec" not in s.columns:
        return []

    sk = s[~s["player_id"].isin(goalies)]
    gdf = s[s["player_id"].isin(goalies)]

    bounds: set[float] = set()
    for a, b in zip(sk["start_game_sec"], sk["end_game_sec"]):
        bounds.add(float(a))
        bounds.add(float(b))
    b_sorted = sorted(bounds)
    stints: list[Stint5v5] = []
    th, ta = str(home_abbr).strip(), str(away_abbr).strip()

    sk_p = sk["player_id"].to_numpy()
    sk_t0 = sk["start_game_sec"].astype(float).to_numpy()
    sk_t1 = sk["end_game_sec"].astype(float).to_numpy()
    g_p = gdf["player_id"].to_numpy() if not gdf.empty else np.array([], dtype=np.int64)
    g_t0 = gdf["start_game_sec"].astype(float).to_numpy() if not gdf.empty else np.array([], dtype=np.float64)
    g_t1 = gdf["end_game_sec"].astype(float).to_numpy() if not gdf.empty else np.array([], dtype=np.float64)
    g_tab = gdf["team_abbrev"].astype(str).to_numpy() if not gdf.empty else np.array([], dtype=str)

    season_id = nhl_season_id_from_game_id(game_id)

    for i in range(len(b_sorted) - 1):
        t0, t1 = float(b_sorted[i]), float(b_sorted[i + 1])
        if t1 <= t0:
            continue
        d = t1 - t0
        if d < min_sec - 1e-9:
            continue

        on = _on_ice_during(t0, t1, sk_p, sk_t0, sk_t1)
        home: set[int] = set()
        away: set[int] = set()
        for p in on:
            row = sk[sk["player_id"] == p]
            if row.empty:
                continue
            tms = {str(x).strip() for x in row["team_abbrev"].head(3).tolist()}
            if th in tms:
                home.add(int(p))
            if ta in tms:
                away.add(int(p))
        h_goal = 0
        a_goal = 0
        for j in range(len(g_p)):
            if g_t0[j] < t1 and g_t1[j] > t0:
                ab = g_tab[j].strip()
                if ab == th:
                    h_goal = 1
                elif ab == ta:
                    a_goal = 1
        if h_goal < 1 or a_goal < 1:  # empty net or no goalie
            continue
        if len(home) != 5 or len(away) != 5:
            continue

        stints.append(
            Stint5v5(
                game_id=game_id,
                t0=t0,
                t1=t1,
                home_skaters=frozenset(home),
                away_skaters=frozenset(away),
                has_home_goalie=h_goal > 0,
                has_away_goalie=a_goal > 0,
                season_id=season_id,
            )
        )
    return stints


def home_away_pcv_sums_in_stint(
    events: pd.DataFrame,
    home: str,
    away: str,
    t0: float,
    t1: float,
    *,
    weight_col: str = "_pos_w",
) -> tuple[float, float]:
    """
    *events* pre-tagged with *game_sec* and *_pos_w*. Sum *w* by *Ev_Team* for the credited
    offensive side.

    **Boundary:** the stint is half-open ``[t0, t1)`` — an event with ``game_sec == t0`` is
    included, ``game_sec == t1`` is **excluded** (it belongs to the next stint on the [t1, …) axis).
    """
    wcol = events[weight_col] if weight_col in events.columns else 0.0
    m = (events["game_sec"] >= t0) & (events["game_sec"] < t1) & (wcol != 0) & wcol.notna()
    sub = events.loc[m]
    th, ta = str(home).strip(), str(away).strip()
    if sub.empty or "Ev_Team" not in sub.columns or weight_col not in sub.columns:
        return 0.0, 0.0
    evs = sub["Ev_Team"].astype(str).str.strip()
    hsum = sub.loc[evs == th, weight_col].sum()
    asum = sub.loc[evs == ta, weight_col].sum()
    return float(hsum), float(asum)


def _add_direct_defense_event_values(events: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """
    Add direct defensive credit to events.

    RAPM defense already uses negative opponent PTV. This extra component captures
    discrete suppression events available in ordinary PBP: opponent possessions
    losing threat, takeaways, giveaways by the offense, blocked shots, and missed
    shot pressure. It is intentionally capped and small relative to total PTV.
    """
    d = events.copy()
    idx = d.index
    ev = d.get("Event", pd.Series("", index=idx)).astype(str).str.upper()
    xtd = pd.to_numeric(d.get("xT_delta", pd.Series(np.nan, index=idx)), errors="coerce").fillna(0.0)
    x_before = pd.to_numeric(d.get("xT_before", pd.Series(np.nan, index=idx)), errors="coerce").fillna(0.0)

    val = (-xtd).clip(lower=0.0, upper=0.18)
    val += np.where(ev.isin({"GIVE", "GIVEAWAY"}), 0.018 + x_before.clip(0.0, 0.18) * 0.20, 0.0)
    val += np.where(ev.isin({"TAKE", "TAKEAWAY"}), 0.012 + x_before.clip(0.0, 0.18) * 0.12, 0.0)
    val += np.where(ev.isin({"BLOCK", "BLOCKED_SHOT"}), 0.010, 0.0)
    val += np.where(ev.eq("MISS"), 0.004, 0.0)
    d["_direct_def_w"] = pd.Series(val, index=idx, dtype=float).clip(lower=0.0, upper=0.22)
    d["_def_team"] = d.get("Ev_Team", pd.Series("", index=idx)).apply(lambda t: _opp_tricode(t, home, away))
    return d


def home_away_direct_defense_sums_in_stint(
    events: pd.DataFrame,
    home: str,
    away: str,
    t0: float,
    t1: float,
) -> tuple[float, float]:
    if events.empty or "_direct_def_w" not in events.columns or "_def_team" not in events.columns:
        return 0.0, 0.0
    m = (
        (events["game_sec"] >= t0)
        & (events["game_sec"] < t1)
        & events["_direct_def_w"].notna()
        & (events["_direct_def_w"] > 0.0)
    )
    sub = events.loc[m]
    if sub.empty:
        return 0.0, 0.0
    h = str(home).strip()
    a = str(away).strip()
    tm = sub["_def_team"].astype(str).str.strip()
    return (
        float(sub.loc[tm.eq(h), "_direct_def_w"].sum()),
        float(sub.loc[tm.eq(a), "_direct_def_w"].sum()),
    )


def stint_targets_table(
    pbp: pd.DataFrame,
    poss: pd.DataFrame,
    stints: list[Stint5v5],
    *,
    is_playoff: bool = False,
    attribution: PcvAttribution = "event_weighted_possession_split",
    near_peak_seconds: float = 0.0,
    near_peak_fraction: float = 0.0,
    near_peak_events_before: int = 0,
    a3z_weight_maps: dict[PcvComponent, dict[int, float]] | None = None,
) -> pd.DataFrame:
    """
    One row per stint. PCV/60 = (sum of *attributed* *_pos_w* in ``[t0, t1)`` for that side) / *duration* × 3600.
    """
    if not stints or pbp.empty:
        return pd.DataFrame(
            columns=[
                "NHL_game_id",
                "t0",
                "t1",
                "duration_sec",
                "home_skater_ids",
                "away_skater_ids",
                "home_pcv_credited",
                "away_pcv_credited",
                "home_pcv_per_60",
                "away_pcv_per_60",
                "home_total_pcv_per60",
                "away_total_pcv_per60",
                "home_shot_pcv_per60",
                "away_shot_pcv_per60",
                "home_threat_pcv_per60",
                "away_threat_pcv_per60",
                "home_no_shot_threat_pcv_per60",
                "away_no_shot_threat_pcv_per60",
                "home_direct_defense_pcv_per60",
                "away_direct_defense_pcv_per60",
                "home_pcv_per_60_context_adj",
                "away_pcv_per_60_context_adj",
                "home_total_pcv_per60_context_adj",
                "away_total_pcv_per60_context_adj",
                "season_id",
                "pcv_attribution",
                "home_score_at_t0",
                "away_score_at_t0",
                "home_score_diff_at_t0",
                "away_score_diff_at_t0",
                "abs_score_diff",
                "home_context_multiplier",
                "away_context_multiplier",
            ]
        )
    pbp_sec = pbp_dataframe_add_game_sec(pbp, is_playoff)
    ev_total = pbp_pcv_event_shares(
        pbp_sec,
        poss,
        attribution=attribution,
        component="total_pcv",
        near_peak_seconds=near_peak_seconds,
        near_peak_fraction=near_peak_fraction,
        near_peak_events_before=near_peak_events_before,
        a3z_weight_maps=a3z_weight_maps,
    )
    ev_shot = pbp_pcv_event_shares(
        pbp_sec,
        poss,
        attribution=attribution,
        component="shot_pcv",
        near_peak_seconds=near_peak_seconds,
        near_peak_fraction=near_peak_fraction,
        near_peak_events_before=near_peak_events_before,
        a3z_weight_maps=a3z_weight_maps,
    )
    ev_thr = pbp_pcv_event_shares(
        pbp_sec,
        poss,
        attribution=attribution,
        component="threat_pcv",
        near_peak_seconds=near_peak_seconds,
        near_peak_fraction=near_peak_fraction,
        near_peak_events_before=near_peak_events_before,
        a3z_weight_maps=a3z_weight_maps,
    )
    ev_nst = pbp_pcv_event_shares(
        pbp_sec,
        poss,
        attribution=attribution,
        component="no_shot_threat_pcv",
        near_peak_seconds=near_peak_seconds,
        near_peak_fraction=near_peak_fraction,
        near_peak_events_before=near_peak_events_before,
        a3z_weight_maps=a3z_weight_maps,
    )
    th = str(pbp["Home_Team"].iloc[0]) if "Home_Team" in pbp.columns else ""
    ta = str(pbp["Away_Team"].iloc[0]) if "Away_Team" in pbp.columns else ""
    if (not th or not ta) and "Home_Team" in ev_total.columns and "Away_Team" in ev_total.columns:
        h2 = ev_total["Home_Team"].dropna()
        t2 = ev_total["Away_Team"].dropna()
        th = th or (str(h2.iloc[0]) if len(h2) else "")
        ta = ta or (str(t2.iloc[0]) if len(t2) else "")
    ev_def = _add_direct_defense_event_values(pbp_sec, th, ta)
    rows: list[dict[str, Any]] = []
    gid = int(pbp["NHL_game_id"].iloc[0]) if "NHL_game_id" in pbp.columns else 0
    for st in stints:
        dsec = st.t1 - st.t0
        if dsec <= 0:
            continue
        hcr, acr = home_away_pcv_sums_in_stint(ev_total, th, ta, st.t0, st.t1)
        h_sh, a_sh = home_away_pcv_sums_in_stint(ev_shot, th, ta, st.t0, st.t1)
        h_th, a_th = home_away_pcv_sums_in_stint(ev_thr, th, ta, st.t0, st.t1)
        h_ns, a_ns = home_away_pcv_sums_in_stint(ev_nst, th, ta, st.t0, st.t1)
        h_dd, a_dd = home_away_direct_defense_sums_in_stint(ev_def, th, ta, st.t0, st.t1)
        h_s, a_s = score_state_at_t0(pbp_sec, st.t0)
        h_diff = float(int(h_s) - int(a_s))
        a_diff = -h_diff
        h_ctx = _context_multiplier(h_diff)
        a_ctx = _context_multiplier(a_diff)
        h60 = (hcr / dsec) * 3600.0
        a60 = (acr / dsec) * 3600.0
        h_sh60 = (h_sh / dsec) * 3600.0
        a_sh60 = (a_sh / dsec) * 3600.0
        h_th60 = (h_th / dsec) * 3600.0
        a_th60 = (a_th / dsec) * 3600.0
        h_ns60 = (h_ns / dsec) * 3600.0
        a_ns60 = (a_ns / dsec) * 3600.0
        h_dd60 = (h_dd / dsec) * 3600.0
        a_dd60 = (a_dd / dsec) * 3600.0
        rows.append(
            {
                "NHL_game_id": st.game_id,
                "t0": st.t0,
                "t1": st.t1,
                "duration_sec": dsec,
                "home_skater_ids": sorted(st.home_skaters),
                "away_skater_ids": sorted(st.away_skaters),
                "home_pcv_credited": hcr,
                "away_pcv_credited": acr,
                "home_pcv_per_60": h60,
                "away_pcv_per_60": a60,
                "home_total_pcv_per60": h60,
                "away_total_pcv_per60": a60,
                "home_shot_pcv_per60": h_sh60,
                "away_shot_pcv_per60": a_sh60,
                "home_threat_pcv_per60": h_th60,
                "away_threat_pcv_per60": a_th60,
                "home_no_shot_threat_pcv_per60": h_ns60,
                "away_no_shot_threat_pcv_per60": a_ns60,
                "home_direct_defense_pcv_per60": h_dd60,
                "away_direct_defense_pcv_per60": a_dd60,
                "home_pcv_per_60_context_adj": h60 * h_ctx,
                "away_pcv_per_60_context_adj": a60 * a_ctx,
                "home_total_pcv_per60_context_adj": h60 * h_ctx,
                "away_total_pcv_per60_context_adj": a60 * a_ctx,
                "home_shot_pcv_per60_context_adj": h_sh60 * h_ctx,
                "away_shot_pcv_per60_context_adj": a_sh60 * a_ctx,
                "home_threat_pcv_per60_context_adj": h_th60 * h_ctx,
                "away_threat_pcv_per60_context_adj": a_th60 * a_ctx,
                "home_no_shot_threat_pcv_per60_context_adj": h_ns60 * h_ctx,
                "away_no_shot_threat_pcv_per60_context_adj": a_ns60 * a_ctx,
                "season_id": st.season_id,
                "pcv_attribution": attribution,
                "home_score_at_t0": h_s,
                "away_score_at_t0": a_s,
                "home_score_diff_at_t0": h_diff,
                "away_score_diff_at_t0": a_diff,
                "abs_score_diff": float(abs(int(h_s) - int(a_s))),
                "home_context_multiplier": h_ctx,
                "away_context_multiplier": a_ctx,
            }
        )
    return pd.DataFrame(rows)


def load_pbp_possession_for_rapm(
    pbp_path: Any,
    poss_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PBP for one file + poss rows for the same *NHL_game_id*; inject *NHL_game_id* if missing."""
    pbp = pd.read_parquet(pbp_path)
    gid = nhl_id_from_pbp_path(pbp_path)
    if "NHL_game_id" not in pbp.columns:
        pbp = pbp.copy()
        pbp["NHL_game_id"] = gid
    pfo = poss_all[poss_all["NHL_game_id"] == int(gid)]
    if pfo.empty and "NHL_game_id" in pbp.columns:
        g2 = int(pbp["NHL_game_id"].iloc[0])
        pfo = poss_all[poss_all["NHL_game_id"] == g2]
    return pbp, pfo


__all__ = [
    "PcvAttribution",
    "PcvComponent",
    "game_elapsed_for_event",
    "goalie_ids_from_pbp",
    "home_away_from_pbp",
    "pbp_pcv_event_shares",
    "pbp_dataframe_add_game_sec",
    "stints_5v5_for_game",
    "stint_targets_table",
    "Stint5v5",
    "load_pbp_possession_for_rapm",
    "score_state_at_t0",
]
