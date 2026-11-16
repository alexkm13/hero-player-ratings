#!/usr/bin/env python3
"""
Fetch and cache PBP (enriched) for many NHL games via the Game Center API.

Discovers game IDs with ``GET /v1/club-schedule-season/{team}/{season}`` (32 teams ×
season) and de-duplicates. **Resume-safe:** skips when ``pbp_{game_id}.parquet`` already exists.

Typical: last three seasons (``20232024``, ``20242025``, ``20252026`` in NHL season-id form).

  python3 batch_scrape_pbp.py --seasons 20232024,20242025,20252026
  python3 batch_scrape_pbp.py --seasons 20232024 --dry-run
  python3 batch_scrape_pbp.py --max-games 5   # test wiring

Respect the API: default 0.35s delay between fetches. Run overnight for full 3-year pulls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import certifi
import requests

from ingest import fetch_pbp_dataframe

NHL_BASE = "https://api-web.nhle.com/v1"
DEFAULT_UA = "Mozilla/5.0 (compatible; pcv/0.2; +https://github.com/)"
# Fallback if standings shape changes
FALLBACK_TEAMS: tuple[str, ...] = (
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM",
    "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT",
    "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH",
)


def _get_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = DEFAULT_UA
    s.headers["Accept"] = "application/json"
    return s


def standings_team_abbrevs(sess: requests.Session) -> list[str]:
    r = sess.get(
        f"{NHL_BASE}/standings/now", timeout=90, verify=certifi.where()
    )
    r.raise_for_status()
    j = r.json()
    ab: set[str] = set()
    for row in j.get("standings") or []:
        t = row.get("teamAbbrev") or {}
        if isinstance(t, dict) and t.get("default"):
            ab.add(str(t["default"]).strip().upper())
    if len(ab) >= 32:
        return sorted(ab)
    return list(FALLBACK_TEAMS)


def _team_slug(abbrev: str) -> str:
    """NHL *club-schedule-season* path segment is lowercase 3-char (e.g. tbl, njd, lak)."""
    return abbrev.strip().lower()


def season_game_ids(
    sess: requests.Session,
    season_id: int,
    teams: Iterable[str],
    *,
    game_types: set[int] | None,
    include_future: bool,
    schedule_delay: float = 0.05,
) -> set[int]:
    """All unique game IDs in *season* that appear in any of *teams*' season schedules."""
    out: set[int] = set()
    for ab in teams:
        url = f"{NHL_BASE}/club-schedule-season/{_team_slug(ab)}/{season_id}"
        r = sess.get(url, timeout=90, verify=certifi.where())
        if r.status_code == 404:
            print(f"  WARN 404 {url} — skip team", file=sys.stderr)
            continue
        r.raise_for_status()
        for g in r.json().get("games") or []:
            gid = g.get("id")
            if gid is None:
                continue
            ngt = g.get("gameType")
            if game_types is not None:
                if ngt is None or int(ngt) not in game_types:
                    continue
            if not include_future:
                # Skip unplayed games (typically FUT/PRE), which return empty PBP.
                gs = str(g.get("gameState") or "").strip().upper()
                if gs in {"FUT", "PRE"}:
                    continue
            out.add(int(gid))
        if schedule_delay > 0:
            time.sleep(schedule_delay)
    return out


def _write_fail(cache_dir: Path, gid: int, err: str) -> None:
    p = cache_dir / "scrape_failures.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"game_id": gid, "error": err[:2000]}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--seasons",
        type=str,
        default="20232024,20242025,20252026",
        help="Comma-separated NHL season ids, e.g. 20232024,20242025,20252026",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "cache",
        help="Directory for pbp_<gameid>.parquet",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between PBP fetches (polite to API).",
    )
    ap.add_argument(
        "--schedule-delay",
        type=float,
        default=0.05,
        help="Seconds between season-schedule (per-team) API calls.",
    )
    ap.add_argument(
        "--game-types",
        type=str,
        default="2,3",
        help="1=preseason, 2=regular, 3=playoff. Comma list; use empty for all. Default: 2,3",
    )
    ap.add_argument(
        "--include-future",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include FUT/PRE schedule entries (default: false).",
    )
    ap.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="Do not re-fetch if parquet already exists (default: true).",
    )
    ap.add_argument("--max-games", type=int, default=0, help="Stop after this many *downloads* (0=unlimited).")
    ap.add_argument("--dry-run", action="store_true", help="Only list season game counts and total unique IDs.")
    ap.add_argument("--progress-every", type=int, default=20, help="Log every N fetches.")
    args = ap.parse_args()

    seasons: list[int] = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    if not seasons:
        raise SystemExit("No --seasons parsed")
    gtypes: set[int] | None
    s = (args.game_types or "").strip()
    if s == "":
        gtypes = None
    else:
        gtypes = {int(x.strip()) for x in s.split(",") if x.strip()}

    sess = _get_session()
    print("Resolving team list from /standings/now …")
    teams = standings_team_abbrevs(sess)
    print(f"  {len(teams)} teams: {', '.join(teams[:8])} …")
    if len(teams) < 32:
        print("  (using fallback/short list — check standings API)", file=sys.stderr)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    all_ids: set[int] = set()
    per: dict[str, int] = {}
    for se in seasons:
        t0 = time.time()
        ids = season_game_ids(
            sess,
            se,
            teams,
            game_types=gtypes,
            include_future=bool(args.include_future),
            schedule_delay=args.schedule_delay,
        )
        all_ids |= ids
        per[str(se)] = len(ids)
        print(
            f"  season {se}: {len(ids)} unique games in union this pass "
            f"({time.time() - t0:.1f}s)"
        )
    if args.dry_run:
        print("DRY-RUN: total unique game IDs", len(all_ids), "per-season counts:", per)
        return

    ids_sorted = sorted(all_ids)
    n_done = 0
    n_skip = 0
    n_err = 0
    t_start = time.time()
    n_limit = int(args.max_games or 0)

    for gid in ids_sorted:
        out = args.cache_dir / f"pbp_{gid}.parquet"
        if args.skip_existing and out.is_file():
            n_skip += 1
            continue
        if n_limit and n_done >= n_limit:
            break
        try:
            df = fetch_pbp_dataframe(gid, session=sess)
            if df is None or df.empty:
                _write_fail(args.cache_dir, gid, "empty pbp")
                n_err += 1
            else:
                try:
                    df.to_parquet(out, index=False)
                except (ImportError, OSError, ValueError):
                    out_csv = args.cache_dir / f"pbp_{gid}.csv"
                    df.to_csv(out_csv, index=False)
                n_done += 1
        except Exception as e:
            n_err += 1
            _write_fail(args.cache_dir, gid, f"{type(e).__name__}: {e!s}")
        time.sleep(max(0.0, args.delay))
        if n_done and n_done % int(args.progress_every) == 0:
            el = time.time() - t_start
            print(f"  … {n_done} new files, {n_skip} skipped, {n_err} err, {el:.0f}s elapsed", flush=True)

    el = time.time() - t_start
    print(
        f"Done.  New/updated: {n_done}  Skipped (existing): {n_skip}  "
        f"Failed: {n_err}  Unique IDs in set: {len(all_ids)}  {el:.1f}s"
    )
    if n_err and (args.cache_dir / "scrape_failures.log").is_file():
        print(f"See {args.cache_dir / 'scrape_failures.log'}")


if __name__ == "__main__":
    main()
