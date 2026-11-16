"""
Chronological ordering for PBP when *Period* + *Seconds_Elapsed* tie.

NHL *eventId* is not play order: e.g. post-goal center *FAC* can have a **lower** id than the
*GOAL* at the same second, which breaks possession and goal context. We apply a **semantic
tiebreak** (then *event_id* as last resort).

Tie order (earlier in real time = lower rank): period markers, then shot outcomes, then other
hockey, then *FAC* last (often the next-play / post-goal faceoff at the same clock).
"""

from __future__ import annotations

import pandas as pd

# Lower = sort earlier in a same-(period, second) group
_EVENT_TIE_RANK: dict[str, int] = {
    "PSTR": 0,
    "EISTR": 1,
    "EIEND": 2,
    "PEND": 5,
    "GEND": 6,
    "SOC": 7,
    "CHL": 8,
    "GIVE": 15,
    "TAKE": 16,
    "HIT": 20,
    "GOAL": 30,
    "SHOT": 31,
    "MISS": 32,
    "BLOCK": 33,
    "DELPEN": 40,
    "PENL": 41,
    "STOP": 50,
    "EGPID": 60,
    "FAC": 100,  # post-goal / next-play faceoffs: after GOAL, SHOT, etc.
}


def event_tie_rank(event: str | object) -> int:
    e = (str(event) or "").strip()
    return int(_EVENT_TIE_RANK.get(e, 45))


def sort_pbp_chronological(
    d: pd.DataFrame,
    *,
    event_col: str = "Event",
    drop_tie_col: bool = True,
) -> pd.DataFrame:
    """
    Sort by ``Game_Id``, ``Period``, ``Seconds_Elapsed``, semantic event rank, ``event_id``.
    Inserts a temporary column ``_evt_tie`` unless *drop_tie_col* and you remove it.
    """
    if d.empty:
        return d
    out = d.copy()
    s = out[event_col].map(event_tie_rank).astype("int32")
    out["_evt_tie"] = s
    keys: list = [c for c in ("Game_Id", "Period", "Seconds_Elapsed", "_evt_tie") if c in out.columns]
    if "event_id" in out.columns:
        keys.append("event_id")
    if len(keys) < 3 + int("_evt_tie" in keys or False):
        keys = list(out.columns[:4])  # fall back, rare
    out = out.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if drop_tie_col and "_evt_tie" in out.columns:
        out = out.drop(columns=["_evt_tie"])
    return out
