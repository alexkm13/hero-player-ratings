"""
NHL **shift chart** (api.nhle.com/stats shiftcharts) for in-sample time on ice.

Row cache includes *start_game_sec* / *end_game_sec* (cumulative game time) for stint construction;
*duration* strings are per-shift clock time (M:SS or H:MM:SS). We sum *duration* in seconds
per *playerId* and cache one Parquet per game under ``{cache_dir}/shifts_{gameId}.parquet``.

Use :func:`aggregate_in_sample_toi` to match :func:`build_player_pcv_per_60` PBP file lists.
Stints: see :mod:`stints_pcv` and :mod:`build_rapm_pcv`.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import certifi
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; pcv/ingest_shifts/1.0; +https://github.com/)"}
SHIFTCHARTS = "https://api.nhle.com/stats/rest/en/shiftcharts"
# Bumped when *start_game_sec* semantics change (e.g. OT length reg vs PO) so stale caches re-fetch.
SHIFTS_INGEST_VERSION = 2
REGOT_PERIOD_LEN = 300.0
PLAYOFF_OT_PERIOD_LEN = 1200.0


def nhl_season_id_from_game_id(gid: int) -> int:
    """NHL 10-digit *gameId* → *seasonId* (e.g. 2023… games → 20232024)."""
    y = int(str(int(gid))[:4])
    return int(f"{y}{y + 1}")


def nhl_id_from_pbp_path(f: Path) -> int:
    s = f.stem
    if not s.startswith("pbp_"):
        raise ValueError(f"expected pbp_*.parquet, got {f.name}")
    return int(s[4:])


def clock_to_seconds(s: str | None) -> float:
    """NHL *duration* / *startTime* / *endTime* (M:SS or H:MM:SS) → seconds."""
    if not s or not str(s).strip():
        return 0.0
    t = str(s).strip()
    if t in {"-16:0-",}:
        return 1200.0
    parts = t.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), float(parts[1])
            return m * 60.0 + sec
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600.0 + m * 60.0 + sec
    except (ValueError, TypeError):
        return 0.0
    return 0.0


def nhl_game_type_id(pbp: Mapping[str, Any] | None) -> int:
    """
    1=preseason, 2=regular, 3=playoffs (NHL *gameType* on Game Center root). Defaults to 2.
    """
    if pbp is None:
        return 2
    gt = pbp.get("gameType")
    try:
        return int(gt) if gt is not None else 2
    except (TypeError, ValueError):
        return 2


def is_playoff_pbp(pbp: Mapping[str, Any] | None) -> bool:
    return nhl_game_type_id(pbp) == 3


def period_clock_to_game_seconds(
    period: int,
    clock_sec_in_period: float,
    *,
    is_playoff: bool = False,
) -> float:
    """
    Cumulative *playing* time in seconds, aligned to PBP *Period* / in-period *Seconds_Elapsed*.

    * Regulation: periods 1–3, 20:00 per period.
    * OT: period 4+; **length differs** — regular (and pre): 5:00 per OT. Playoffs: 20:00 per OT.
    * Does **not** add shootout “periods” (use :func:`is_shootout_only_period` and drop such rows
      from shift data; SO is not 5v5 stints we care about).
    """
    p = int(period)
    s = float(clock_sec_in_period)
    if p <= 0:
        return 0.0
    if p <= 3:
        return (p - 1) * 1200.0 + s
    olen = PLAYOFF_OT_PERIOD_LEN if is_playoff else REGOT_PERIOD_LEN
    ot_index = p - 3
    return 3600.0 + (ot_index - 1) * olen + s


def split_clock_to_in_period(clock: str) -> float:
    """'MM:SS' in period (or H:MM:SS) -> seconds in that period."""
    return clock_to_seconds(clock)


def is_shootout_only_period(period: int, *, pbp: Mapping[str, Any] | None) -> bool:
    """
    True for shift rows that are **not** live 5v5/5v4 hockey. Playoffs: never shootout. For
    pre/regular, NHL has at most one 5:00 OT (period 4) before a shootout, so *period* ≥ 5 is
    the skills competition (or data noise). *Neutral-site* rules still use 5v5 periods before SO.
    """
    if pbp is not None and nhl_game_type_id(pbp) not in (1, 2):
        return False
    if pbp is not None and is_playoff_pbp(pbp):
        return False
    return int(period) >= 5


def row_start_end_game_seconds(
    period: int,
    start_t: str,
    end_t: str,
    *,
    is_playoff: bool = False,
) -> tuple[float, float]:
    a = period_clock_to_game_seconds(period, split_clock_to_in_period(start_t or "0:0"), is_playoff=is_playoff)
    b = period_clock_to_game_seconds(period, split_clock_to_in_period(end_t or "0:0"), is_playoff=is_playoff)
    if b < a:
        b = a
    return a, b


def fetch_game_shifts_dataframe(
    game_id: int,
    session: requests.Session,
    *,
    page_limit: int = 2000,
    is_playoff: bool = False,
    pbp: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    All shift rows for *game_id*; one or more paged requests.

    *is_playoff* must match the game (NHL *gameType* 3): OT periods are 20:00; else 5:00 per OT.
    Rows in *period* ≥ 5 for pre/regular are treated as **shootout** (not real 5v5) and dropped.
    """
    if "User-Agent" not in session.headers:
        session.headers.update(UA)
    rows: list[dict] = []
    start = 0
    while True:
        r = session.get(
            SHIFTCHARTS,
            params={
                "cayenneExp": f"gameId={int(game_id)}",
                "start": str(start),
                "limit": str(page_limit),
            },
            timeout=120,
            verify=certifi.where(),
        )
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or []
        for row in data:
            p = int(row.get("period", 0) or 0)
            if is_shootout_only_period(p, pbp=pbp):
                continue
            pid_raw = row.get("playerId")
            if pid_raw is None:
                continue
            d = row.get("duration") or "0:0"
            st = str(row.get("startTime") or "0:0")
            en = str(row.get("endTime") or st)
            sg, eg = row_start_end_game_seconds(p, st, en, is_playoff=is_playoff)
            rows.append(
                {
                    "game_id": int(row.get("gameId", game_id)),
                    "player_id": int(pid_raw),
                    "duration_sec": clock_to_seconds(d),
                    "start_game_sec": float(sg),
                    "end_game_sec": float(eg),
                    "start_time": st,
                    "end_time": en,
                    "period": p,
                    "team_abbrev": str(row.get("teamAbbrev") or ""),
                    "_ingest_v": SHIFTS_INGEST_VERSION,
                }
            )
        if not data or len(data) < page_limit:
            break
        start += page_limit
    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "player_id",
                "duration_sec",
                "start_game_sec",
                "end_game_sec",
                "start_time",
                "end_time",
                "period",
                "team_abbrev",
                "_ingest_v",
            ]
        )
    return pd.DataFrame(rows)


def _shifts_cache_is_valid(df: pd.DataFrame) -> bool:
    if df.empty or "start_game_sec" not in df.columns or "end_game_sec" not in df.columns:
        return False
    v = int(df["_ingest_v"].iloc[0]) if "_ingest_v" in df.columns and len(df) else 0
    if v < SHIFTS_INGEST_VERSION:
        return False
    return True


def load_or_fetch_shifts_parquet(
    game_id: int,
    cache_dir: Path,
    session: requests.Session,
    *,
    allow_fetch: bool = True,
    pbp: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    path = cache_dir / f"shifts_{int(game_id)}.parquet"
    alt = path.with_suffix(".csv")
    if path.is_file():
        try:
            df0 = pd.read_parquet(path)
        except (OSError, ValueError):
            df0 = pd.DataFrame()
        if _shifts_cache_is_valid(df0):
            return df0
    if alt.is_file():
        try:
            df0 = pd.read_csv(alt)
        except (OSError, ValueError):
            df0 = pd.DataFrame()
        if _shifts_cache_is_valid(df0):
            return df0
    if not allow_fetch:
        return pd.DataFrame(
            columns=[
                "game_id",
                "player_id",
                "duration_sec",
                "start_game_sec",
                "end_game_sec",
                "start_time",
                "end_time",
                "period",
                "team_abbrev",
                "_ingest_v",
            ]
        )
    pbp_m = pbp
    if pbp_m is None:
        from ingest import fetch_play_by_play

        pbp_m = fetch_play_by_play(game_id, session=session)
    is_po = is_playoff_pbp(pbp_m)
    df = fetch_game_shifts_dataframe(
        int(game_id),
        session,
        is_playoff=is_po,
        pbp=pbp_m,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        try:
            df.to_parquet(path, index=False)
        except (ImportError, OSError, ValueError):
            df.to_csv(alt, index=False)
    return df


def aggregate_in_sample_toi(
    cache_dir: Path,
    session: requests.Session,
    *,
    pbp_glob: str = "pbp_*.parquet",
    on_progress: Callable[[int, int, str], None] | None = None,
    allow_fetch: bool = True,
) -> dict[int, float]:
    """
    For every ``pbp_*.parquet`` in *cache_dir*, load or fetch shifts and sum *duration_sec*
    by *player_id* across the whole in-sample set.
    """
    files = sorted(cache_dir.glob(pbp_glob))
    toi: dict[int, float] = defaultdict(float)
    t0 = time.time()
    n_files = len(files)
    for i, f in enumerate(files):
        gid = nhl_id_from_pbp_path(f)
        if on_progress and i and i % 200 == 0:
            on_progress(i, n_files, f"shifts  {i}/{n_files}  {time.time() - t0:.0f}s")
        try:
            df = load_or_fetch_shifts_parquet(gid, cache_dir, session, allow_fetch=allow_fetch)
        except (OSError, ValueError) as e:
            if on_progress:
                on_progress(i, n_files, f"warn: game {gid} shifts failed: {e}")
            continue
        if df.empty:
            continue
        for pid, s in df.groupby("player_id")["duration_sec"].sum().items():
            toi[int(pid)] += float(s)
    return dict(toi)


def aggregate_in_sample_toi_by_season(
    cache_dir: Path,
    session: requests.Session,
    *,
    pbp_glob: str = "pbp_*.parquet",
    on_progress: Callable[[int, int, str], None] | None = None,
    allow_fetch: bool = True,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """
    Returns (total toi by player, (player_id, season_id) -> sec).
    """
    files = sorted(cache_dir.glob(pbp_glob))
    toi: dict[int, float] = defaultdict(float)
    toi_s: dict[tuple[int, int], float] = defaultdict(float)
    t0 = time.time()
    n_files = len(files)
    for i, f in enumerate(files):
        gid = nhl_id_from_pbp_path(f)
        season_id = nhl_season_id_from_game_id(gid)
        if on_progress and i and i % 200 == 0:
            on_progress(i, n_files, f"shifts+season  {i}/{n_files}  {time.time() - t0:.0f}s")
        try:
            df = load_or_fetch_shifts_parquet(gid, cache_dir, session, allow_fetch=allow_fetch)
        except (OSError, ValueError) as e:
            if on_progress:
                on_progress(i, n_files, f"warn: game {gid} shifts failed: {e}")
            continue
        if df.empty:
            continue
        for pid, s in df.groupby("player_id")["duration_sec"].sum().items():
            pid = int(pid)
            sf = float(s)
            toi[pid] += sf
            toi_s[pid, season_id] = toi_s.get((pid, season_id), 0.0) + sf
    return dict(toi), dict(toi_s)
