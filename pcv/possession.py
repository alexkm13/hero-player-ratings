"""
Infer puck possession from NHL PBP (events only). PBP does not log possession; this tags a
`possession_id` and `possess_after` (team in possession *after* the event) per the rules
used in this project.

**PBP order:** :func:`pbp_order.sort_pbp_chronological` so at the same clock, **GOAL/shot** rows
sort before **FAC**; raw *eventId* is not true play order.

**Whistle policy (default: continuous, see** ``WHISTLE_CONTINUOUS``**):** Administrative rows
(*PSTR*, *PEND*, *GEND*, *STOP*, *PENL*, *DELPEN*, …) do **not** create a new *possession_id* by
themselves—the last play’s *pid* is carried. A *FAC* after a *STOP* keeps the same *pid* if the
faceoff winner is the same team that had the puck (threat continuity). If the *FAC* gives the
puck to the other team, *pid* bumps. The **strict** alternative (every stoppage + *FAC* = new
segment regardless of who wins) is not the default; set :data:`WHISTLE_CONTINUOUS` to *False* and
pass *whistle_continuous=False* to :func:`tag_possessions` to treat every *FAC* after
admin/*STOP* as a new segment (stricter; not the default).
"""

from __future__ import annotations

import pandas as pd

from pbp_order import sort_pbp_chronological

_NEUTRAL_EVENTS: frozenset[str] = frozenset({"HIT"})

# Clock / admin: do not allocate a new segment by themselves; inherit *pid* from last play.
_NON_PLAY: frozenset[str] = frozenset(
    {
        "PSTR",
        "PEND",
        "GEND",
        "STOP",
        "PENL",
        "DELPEN",
        "EISTR",
        "EIEND",
        "CHL",
        "EGPID",
        "SOC",
    }
)
_SHOT_FAM = frozenset({"SHOT", "MISS", "GOAL"})

# Default: same *possession_id* across *STOP* → *FAC* when the same team retains.
WHISTLE_CONTINUOUS: bool = True

_MEANINGFUL_EVENTS = frozenset(
    {"SHOT", "GOAL", "MISS", "BLOCK", "GIVE", "TAKE", "HIT"}
)


def _opp(home: str, away: str, t: str) -> str:
    if t == home:
        return away
    if t == away:
        return str(home)
    return ""


def infer_possess_after(row: pd.Series, home: str, away: str) -> str | None:
    e = row.get("Event")
    ev = row.get("Ev_Team")
    if ev is None or (isinstance(ev, float) and pd.isna(ev)):
        ev = ""
    else:
        ev = str(ev).strip()

    if e in _NON_PLAY:
        return None
    if e in _NEUTRAL_EVENTS:
        return None
    if e == "GIVE" and ev:
        return _opp(home, away, ev) or None
    if e in ("FAC", "SHOT", "MISS", "GOAL", "TAKE", "BLOCK"):
        if ev:
            return ev
        return None
    if ev:
        return ev
    return None


def _same_period(r1: pd.Series, r2: pd.Series) -> bool:
    p1, p2 = r1.get("Period"), r2.get("Period")
    if pd.isna(p1) or pd.isna(p2):
        return False
    return int(p1) == int(p2)


def _dt_sec(prev: pd.Series, row: pd.Series) -> float:
    return float(row["Seconds_Elapsed"]) - float(prev["Seconds_Elapsed"])


def _start_new_id(pid: int) -> int:
    if pid == 0:
        return 1
    return pid + 1


def _prev_is_non_play(prev: pd.Series | None) -> bool:
    if prev is None:
        return False
    return str(prev.get("Event") or "") in _NON_PLAY


def tag_possessions(
    events_df: pd.DataFrame,
    *,
    home_col: str = "Home_Team",
    away_col: str = "Away_Team",
    rebound_sec: float = 2.0,
    sort: bool = True,
    whistle_continuous: bool = True,
) -> pd.DataFrame:
    """
    Tag ``possession_id`` and ``possess_after``.

    * *whistle_continuous* — if True (default), *FAC* after *STOP*/*PSTR*/*PEND* tail does not
      bump *pid* when the faceoff winner is the same as *current*; bump if the other team wins
      the draw. If False, *FAC* always bumps after any *NON_PLAY* (stricter "new segment at every
      draw after a whistle" — adjust logic in a follow-up if you need it).
    """
    if events_df.empty:
        return events_df
    d = events_df.copy()
    if sort:
        d = sort_pbp_chronological(d, drop_tie_col=True)
    else:
        sk = [c for c in ("Game_Id", "Period", "Seconds_Elapsed", "event_id") if c in d.columns]
        d = d.sort_values(sk or list(d.columns[:2]), kind="mergesort").reset_index(drop=True)

    pids: list[int] = []
    pa: list[str | None] = []

    pid = 0
    current: str | None = None
    prev: pd.Series | None = None
    wc = whistle_continuous

    for i in range(len(d)):
        row = d.iloc[i]
        h, a = str(row[home_col]), str(row[away_col])
        e = row["Event"]
        if "Game_Id" in row.index and i > 0 and row.get("Game_Id") != d.iloc[i - 1].get("Game_Id"):
            pid, current, prev = 0, None, None
        if prev is not None and str(prev.get("Event")) == "GOAL":
            if pid:
                pid += 1
            else:
                pid = 1
            current = None

        if e in _NON_PLAY:
            this_pid = max(1, pid) if pid else 1
            pids.append(this_pid)
            pa.append(current)
            if e in ("PEND", "GEND"):
                pid = _start_new_id(this_pid)
                current = None
            elif e == "PENL":
                current = None
                pid = this_pid
            else:
                # *PSTR*, *STOP*, …: carry the last play’s *pid* into the state machine
                pid = this_pid
            prev = row
            continue

        nt0 = infer_possess_after(row, h, a)
        if e in ("HIT",) and nt0 is None:
            nt0 = current

        nt: str | None = nt0
        if (
            prev is not None
            and _same_period(prev, row)
            and 0.0 <= _dt_sec(prev, row) < rebound_sec
        ):
            if (
                e in _SHOT_FAM
                and str(prev.get("Event") or "") in _SHOT_FAM
                and (prev.get("Ev_Team") or "") == (row.get("Ev_Team") or "")
                and nt
                and current
                and nt != current
            ):
                nt = current

        if e == "FAC" and nt:
            skip_bump = False
            if prev is not None:
                pev = str(prev.get("Event") or "")
                if pev == "GOAL":
                    skip_bump = True
                elif wc and _prev_is_non_play(prev):
                    if (not current and nt) or (current and nt == current):
                        # First *FAC* after a break, or same team keeps the draw after a whistle
                        skip_bump = True
            if not skip_bump:
                if pid:
                    pid += 1
                else:
                    pid = 1
            current = nt
            pids.append(pid)
            pa.append(current)
            prev = row
            continue

        if nt is None:
            pids.append(max(1, pid) if pid else 1)
            if not pid:
                pid = 1
            pa.append(current)
            prev = row
            continue

        if current is None:
            if pid == 0:
                pid = 1
            current = nt
        elif nt != current:
            if pid:
                pid += 1
            else:
                pid = 1
            current = nt
        pids.append(pid)
        pa.append(current)
        prev = row

    d["possession_id"] = pids
    d["possess_after"] = pa
    return d


def enrich_pbp_with_possession(
    pbp: pd.DataFrame,
    *,
    sort: bool = True,
) -> pd.DataFrame:
    """
    Run :func:`tag_possessions`, then add columns for use as a cached, self-contained PBP table:

    * **possessing_team_after** — same meaning as *possess_after*; stable name for downstream
      and parquet (do not recompute *infer_possess_after*).
    * **is_team_change** — *True* when, for this event, the team with the puck (after) differs
      from the most recent *non-blank* ``possessing_team_after``; *False* when
      *possessing_team_after* is null/blank, or for the first stretching possession team.
    """
    if pbp.empty:
        return pbp
    t = tag_possessions(pbp, sort=sort)
    t = t.copy()
    pa = t["possess_after"]
    t["possessing_team_after"] = pa
    n = len(t)
    out: list[bool] = [False] * n
    last: str | None = None
    for i in range(n):
        v = pa.iloc[i]
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out[i] = False
            continue
        s = str(v).strip()
        if not s:
            out[i] = False
            continue
        if last is None:
            out[i] = False
        else:
            out[i] = s != last
        last = s
    t["is_team_change"] = out
    return t


def validate_possession_rules(events_df: pd.DataFrame) -> list[str]:
    """
    Invariants: any segment that contains a *GOAL* should end on that *GOAL* (no later events
    in the same ``possession_id`` in chronological PBP order).
    """
    errs: list[str] = []
    if events_df.empty or "possession_id" not in events_df.columns:
        return errs
    d = sort_pbp_chronological(events_df.copy(), drop_tie_col=True) if "event_id" in events_df.columns else events_df
    gcols: list[str] = [c for c in ("Game_Id", "possession_id") if c in d.columns]
    if len(gcols) < 1:
        return errs
    sort_k = [c for c in ("Game_Id", "Period", "Seconds_Elapsed", "event_id") if c in d.columns]
    for key, g in d.groupby(gcols, sort=False):
        g2 = g.sort_values(sort_k or g.columns[:1].tolist(), kind="mergesort")
        g2 = g2[g2.get("Event").notna()] if "Event" in g2.columns else g2
        if "Event" not in g2.columns or g2.empty:
            continue
        goal_mask = g2["Event"] == "GOAL"
        if not goal_mask.any():
            continue
        last = str(g2["Event"].iloc[-1])
        if last != "GOAL":
            kstr = ",".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
            errs.append(
                f"possession ({kstr}): GOAL is not the last event (last={last})"
            )
    return errs


def add_possession_meaningful_flag(events_df: pd.DataFrame, *, out_col: str = "possession_meaningful") -> pd.DataFrame:
    """Set *out_col* to True for rows with :data:`_MEANINGFUL_EVENTS` (hockey actions)."""
    o = events_df.copy()
    if "Event" not in o.columns:
        o[out_col] = False
        return o
    o[out_col] = o["Event"].astype(str).isin(_MEANINGFUL_EVENTS)
    return o
