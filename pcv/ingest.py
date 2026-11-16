"""
Play-by-play from the official NHL Web API (Game Center), no HTML merge.

Endpoint: GET https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

One row per play; coordinates come from a single JSON source (all goals typically have x/y).
**Rows include** *p1_ID* (scorer/shooter/winner or ``details.playerId`` for e.g. GIVE/TAKE) and
*P1_Team* (tricode from game roster). **Cached** Parquet/CSV from :func:`load_or_scrape_pbp` also
keeps *possession_id*, *possess_after*, *possessing_team_after*, and *is_team_change* from
:func:`possession.enrich_pbp_with_possession` so you can analyze without re-tagging.

Row-level column names align with what `coords.py` and `validate_rink.py` expect (Event, Period,
xC, yC, Home_Team, Away_Team, Ev_Team, Home_Zone, ...).

SSL: uses `requests` + `certifi` (avoids common cert issues on macOS).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Union

import certifi
import pandas as pd
import requests
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout, Timeout

from pbp_order import sort_pbp_chronological

# Set by :func:`possession.enrich_pbp_with_possession` for cached, analysis-ready PBP.
PBP_DERIVED_COLUMNS: tuple[str, ...] = (
    "possession_id",
    "possess_after",
    "possessing_team_after",
    "is_team_change",
)

# Mirror hockey_scraper JSON event labels for compatibility
_EVENT_MAP = {
    "PERIOD START": "PSTR",
    "FACEOFF": "FAC",
    "BLOCKED SHOT": "BLOCK",
    "GAME END": "GEND",
    "GIVEAWAY": "GIVE",
    "GOAL": "GOAL",
    "HIT": "HIT",
    "MISSED SHOT": "MISS",
    "PERIOD END": "PEND",
    "SHOT ON GOAL": "SHOT",
    "STOPPAGE": "STOP",
    "TAKEAWAY": "TAKE",
    "PENALTY": "PENL",
    "EARLY INT START": "EISTR",
    "EARLY INT END": "EIEND",
    "SHOOTOUT COMPLETE": "SOC",
    "CHALLENGE": "CHL",
    "EMERGENCY GOALTENDER": "EGPID",
    "DELAYED PENALTY": "DELPEN",  # not in HS; keep stable short code
}

_IGNORE = {
    "PERIOD READY",
    "PERIOD OFFICIAL",
    "GAME READY",
    "GAME OFFICIAL",
    "GAME SCHEDULED",
}

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{}/play-by-play"
# (connect, read) — long runs hit slow reads; :func:`fetch_play_by_play` also retries.
TIMEOUT = (30, 180)


def _time_in_period_to_seconds(s: str) -> float:
    if not s or not str(s).strip():
        return 0.0
    parts = str(s).strip().split(":")
    if len(parts) == 2:
        m, sec = int(parts[0]), float(parts[1])
        return m * 60.0 + sec
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600.0 + m * 60.0 + sec
    return 0.0


def _normalize_type_key(type_desc_key: str) -> str:
    k = type_desc_key.replace("-", " ").strip().upper()
    return k


def _map_event_type(type_desc_key: str) -> str:
    k = _normalize_type_key(type_desc_key)
    if not k:
        return ""
    if k in _IGNORE:
        return ""
    return _EVENT_MAP.get(k, k)


def _roster_player_to_team_abbrev(pbp: dict, tid: dict[int, str]) -> dict[int, str]:
    """Map playerId -> tricode for everyone on the game roster (from PBP `rosterSpots`)."""
    m: dict[int, str] = {}
    for spot in pbp.get("rosterSpots") or []:
        pid, team = spot.get("playerId"), spot.get("teamId")
        if pid is not None and team is not None and int(team) in tid:
            m[int(pid)] = tid[int(team)]
    return m


def _team_abbrevs(roots: Dict[str, Any]) -> tuple[str, str, dict[int, str]]:
    h = roots["homeTeam"]
    a = roots["awayTeam"]
    th, ta = h["abbrev"], a["abbrev"]
    tid: dict[int, str] = {int(h["id"]): th, int(a["id"]): ta}
    return th, ta, tid


def _ev_team_from_event(details: Dict[str, Any], tid: dict[int, str]) -> str:
    eid = details.get("eventOwnerTeamId")
    if eid is None:
        return ""
    return tid.get(int(eid), "")


def _home_zone(ev_team: str, home: str, zone_code: str) -> str:
    """Map API zone (O/D/N, shooter's team) to Home_Zone: Off/Def/Neu from home perspective."""
    if not zone_code or not ev_team or not home:
        return ""
    z = str(zone_code).upper()
    if z in ("N", "NEU", "NEUTRAL"):
        return "Neu"
    is_h = ev_team == home
    if z in ("O", "OFF", "OFFENSIVE"):
        return "Off" if is_h else "Def"
    if z in ("D", "DEF", "DEFENSIVE"):
        return "Def" if is_h else "Off"
    return ""


def _parse_play(play: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any] | None:
    tkey = play.get("typeDescKey") or ""
    event = _map_event_type(tkey)
    if not event:
        return None

    home = meta["home"]
    away = meta["away"]
    tid = meta["team_ids"]
    player_to_team: dict[int, str] = meta.get("player_to_team") or {}
    period = int(play.get("periodDescriptor", {}).get("number") or 0)
    sec = _time_in_period_to_seconds(play.get("timeInPeriod") or "")

    det = play.get("details") or {}
    x = det.get("xCoord")
    y = det.get("yCoord")
    ev_team = _ev_team_from_event(det, tid)
    wpid = det.get("winningPlayerId")
    if event == "FAC" and wpid is not None and int(wpid) in player_to_team:
        ev_team = player_to_team[int(wpid)]
    zc = det.get("zoneCode") or ""
    hzone = _home_zone(ev_team, home, str(zc))

    p1 = (
        det.get("scoringPlayerId")
        or det.get("shootingPlayerId")
        or det.get("winningPlayerId")
        or det.get("playerId")
    )
    p1_team = _p1_tricode(p1, player_to_team)

    row: Dict[str, Any] = {
        "Event": event,
        "Period": period,
        "Seconds_Elapsed": sec,
        "p1_ID": p1,
        "P1_Team": p1_team,
        "Ev_Team": ev_team,
        "Home_Zone": hzone,
        "Ev_Zone": {"O": "Off", "D": "Def", "N": "Neu"}.get(
            (str(zc) or "").upper()[:1], ""
        ),
        "xC": float(x) if x is not None else None,
        "yC": float(y) if y is not None else None,
        "Home_Score": det.get("homeScore"),
        "Away_Score": det.get("awayScore"),
        "Shot_Type": det.get("shotType") or "",
        "Situation_Code": str(play.get("situationCode") or ""),
        "Goalie_In_Net_Id": det.get("goalieInNetId"),
        "Miss_Reason": det.get("reason") or "",
    }
    return row


def _p1_tricode(p1: object, player_to_team: dict[int, str]) -> str:
    if p1 is None or (isinstance(p1, float) and pd.isna(p1)):
        return ""
    try:
        return player_to_team.get(int(p1), "")  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""


def fetch_play_by_play(game_id: Union[str, int], *, session: requests.Session | None = None) -> dict:
    """Return raw JSON from the NHL gamecenter play-by-play endpoint.

    Retries on transient network errors and 5xx/429 so long batch jobs (e.g. RAPM) are not
    killed by a single slow or flaky response.
    """
    url = PBP_URL.format(int(game_id))
    sess = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; pcv/ingest/1.0; +https://github.com/)"}
    last: Exception | None = None
    for attempt in range(10):
        try:
            r = sess.get(url, timeout=TIMEOUT, verify=certifi.where(), headers=headers)
            r.raise_for_status()
            return r.json()
        except (ReadTimeout, ConnectionError, Timeout) as e:
            last = e
            time.sleep(min(90.0, 3.0 * (1.6**attempt)))
        except HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 500, 502, 503, 504):
                last = e
                time.sleep(min(120.0, 5.0 * (1.7**attempt)))
            else:
                raise
    assert last is not None
    raise last


def pbp_json_to_dataframe(pbp: dict, game_id: int | str) -> pd.DataFrame:
    """
    Build a PBP DataFrame with columns used elsewhere in this project. Includes *P1_Team*
    (giver/primary *playerId* via roster) but **not** possession tags—use
    :func:`fetch_pbp_dataframe` or :func:`possession.enrich_pbp_with_possession` for
    *possession_id* and related columns.
    `game_id` is the 10-digit id (e.g. 2024020500); Game_Id in the table is the 5-digit tail
    to match the previous hockey_scraper convention.
    """
    home, away, tid = _team_abbrevs(pbp)
    ptt = _roster_player_to_team_abbrev(pbp, tid)
    game_date = pbp.get("gameDate") or pbp.get("startTimeUTC", "")[:10]
    s = str(int(game_id))
    gid5 = s[-5:]

    meta: Dict[str, Any] = {"home": home, "away": away, "team_ids": tid, "player_to_team": ptt}
    rows: List[Dict[str, Any]] = []
    for play in pbp.get("plays") or []:
        r = _parse_play(play, meta)
        if r is None:
            continue
        r["Game_Id"] = gid5
        r["Date"] = str(game_date) if game_date else ""
        r["Home_Team"] = home
        r["Away_Team"] = away
        r["event_id"] = play.get("eventId")
        rows.append(r)

    if not rows:
        return pd.DataFrame()
    return sort_pbp_chronological(pd.DataFrame(rows), drop_tie_col=True)


def _backfill_pbp_baseline_schema(d: pd.DataFrame) -> pd.DataFrame:
    """Old cache without *P1_Team*: add empty column so downstream has a stable schema."""
    o = d.copy()
    if "P1_Team" not in o.columns:
        o["P1_Team"] = ""
    return o


def pbp_dataframe_needs_enrichment(d: pd.DataFrame) -> bool:
    if d is None or d.empty:
        return False
    return any(c not in d.columns for c in PBP_DERIVED_COLUMNS)


def fetch_pbp_dataframe(
    game_id: Union[str, int],
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """GET play-by-play, then :func:`possession.enrich_pbp_with_possession` (possession tags)."""
    gid = int(game_id)
    raw = fetch_play_by_play(gid, session=session)
    d = pbp_json_to_dataframe(raw, gid)
    from possession import enrich_pbp_with_possession

    return enrich_pbp_with_possession(d, sort=True)


def load_or_scrape_pbp(
    game_id: Union[str, int],
    *,
    cache_dir: Path | str | None = None,
    if_scrape_shifts: bool = False,
) -> pd.DataFrame:
    """
    Return PBP for one game, optionally reading Parquet/CSV from cache_dir.
    Cache is **upgraded in place** when it predates *PBP_DERIVED_COLUMNS* (possession tags).
    `if_scrape_shifts` is kept for call compatibility; shifts are not part of this API surface.
    """
    _ = if_scrape_shifts
    gid = int(game_id)
    from possession import enrich_pbp_with_possession

    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        pq = cdir / f"pbp_{gid}.parquet"
        csv = cdir / f"pbp_{gid}.csv"
        if pq.exists():
            df = pd.read_parquet(pq)
            df = _backfill_pbp_baseline_schema(df)
            if pbp_dataframe_needs_enrichment(df):
                df = enrich_pbp_with_possession(df, sort=True)
                try:
                    df.to_parquet(pq, index=False)
                except (ImportError, OSError, ValueError):
                    pass
            return df
        if csv.exists():
            df = pd.read_csv(csv)
            df = _backfill_pbp_baseline_schema(df)
            if pbp_dataframe_needs_enrichment(df):
                df = enrich_pbp_with_possession(df, sort=True)
                try:
                    df.to_csv(csv, index=False)
                except (ImportError, OSError, ValueError):
                    pass
            return df

    df = fetch_pbp_dataframe(gid)
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cdir / f"pbp_{gid}.parquet", index=False)
        except (ImportError, OSError, ValueError):
            df.to_csv(cdir / f"pbp_{gid}.csv", index=False)
    return df
