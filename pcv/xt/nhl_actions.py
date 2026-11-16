"""
Build HockeynxT-style `actions` tables from NHL PBP + possession tags.

Each row: start_x, start_y, end_x, end_y, type_name, success, time, is_goal (for shots).
Coordinates are in **200×85 attacking space** (see hockeynxt_core.nhl_to_rink_200_85).
"""

from __future__ import annotations

import pandas as pd

from coords import align_period_one_orientation, to_attacking_positive_x
from possession import tag_possessions

from pbp_order import sort_pbp_chronological

from .hockeynxt_core import nhl_to_rink_200_85

_SHOT_EVENTS = frozenset({"SHOT", "MISS", "GOAL"})


def _time_bin(period: int, sec: float) -> int:
    """1..4 = quarter of period (HockeynxT time slice)."""
    q = int(sec // 300) + 1
    return int(min(4, max(1, q)))


def build_actions_from_pbp(
    pbp: pd.DataFrame,
    *,
    with_possession: bool = True,
) -> pd.DataFrame:
    """
    `pbp` from `ingest` (NHL API). Produces an actions DataFrame for `fit_xT`.
    - Shots: type_name Shot, is_goal, success, start=end in 200×85.
    - Synthetic moves: consecutive on-puck events in same possession segment with both coords
      (Move, success=1.0 when same team keeps possession at next event, else 0.0).
    """
    t = tag_possessions(pbp) if with_possession else pbp.copy()
    t = sort_pbp_chronological(t, drop_tie_col=True)
    a = align_period_one_orientation(t)
    a = to_attacking_positive_x(a)
    rx, ry = nhl_to_rink_200_85(a["xC"], a["yC"])
    a = a.assign(_rx=rx, _ry=ry)

    # Pass 1: shots
    rows: list[dict] = []
    for i in range(len(a)):
        r = a.iloc[i]
        e = r["Event"]
        if e not in _SHOT_EVENTS or pd.isna(r["_rx"]) or pd.isna(r["_ry"]):
            continue
        sx, sy = float(r["_rx"]), float(r["_ry"])
        g = 1.0 if e == "GOAL" else 0.0
        time_b = _time_bin(int(r["Period"]), float(r["Seconds_Elapsed"]))
        rows.append(
            {
                "start_x": sx,
                "start_y": sy,
                "end_x": sx,
                "end_y": sy,
                "type_name": "Shot",
                "success": g,
                "is_goal": int(g == 1.0),
                "time": time_b,
            }
        )
    adf = pd.DataFrame(rows)
    # Pass 2: synthetic move attempts between consecutive on-puck events.
    # Keep failed attempts (team switch) so xT can leak value on turnovers.
    move_rows: list[dict] = []
    if "possession_id" in a.columns and len(a) > 1:
        for i in range(1, len(a)):
            prev, cur = a.iloc[i - 1], a.iloc[i]
            if int(prev["possession_id"]) != int(cur["possession_id"]):
                continue
            if (
                pd.isna(cur["_rx"])
                or pd.isna(cur["_ry"])
                or pd.isna(prev["_rx"])
                or pd.isna(prev["_ry"])
            ):
                continue
            if prev["Event"] in _SHOT_EVENTS and cur["Event"] in _SHOT_EVENTS:
                continue
            prev_team = str(prev.get("Ev_Team", ""))
            cur_team = str(cur.get("Ev_Team", ""))
            success = 1.0 if prev_team and prev_team == cur_team else 0.0
            time_b = _time_bin(int(cur["Period"]), float(cur["Seconds_Elapsed"]))
            move_rows.append(
                {
                    "start_x": float(prev["_rx"]),
                    "start_y": float(prev["_ry"]),
                    "end_x": float(cur["_rx"]),
                    "end_y": float(cur["_ry"]),
                    "type_name": "Move",
                    "success": success,
                    "is_goal": 0,
                    "time": time_b,
                }
            )
    mdf = pd.DataFrame(move_rows)
    if adf.empty and mdf.empty:
        return adf
    if adf.empty:
        return mdf
    if mdf.empty:
        return adf
    return pd.concat([adf, mdf], ignore_index=True)
