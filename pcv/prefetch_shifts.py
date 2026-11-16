#!/usr/bin/env python3
"""
Pre-fetch shifts for every PBP file in cache for a given season, with retry+backoff.

Why this exists: load_or_fetch_shifts_parquet catches OSError/ValueError only, not
requests.exceptions.ConnectionError. A single NHL API disconnect crashes the 1400-game
stint+outcomes builds. Run this first to populate the shift cache safely; downstream
builds then hit cache and skip network entirely.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from ingest_shifts import (
    SHIFTS_INGEST_VERSION,
    _shifts_cache_is_valid,
    fetch_game_shifts_dataframe,
    is_playoff_pbp,
    nhl_id_from_pbp_path,
    nhl_season_id_from_game_id,
)
from ingest import fetch_play_by_play

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"


def fetch_one_with_retry(gid: int, cache_dir: Path, session: requests.Session, max_retries: int = 6) -> str:
    path = cache_dir / f"shifts_{gid}.parquet"
    if path.is_file():
        try:
            import pandas as pd

            df0 = pd.read_parquet(path)
            if _shifts_cache_is_valid(df0):
                return "cached"
        except (OSError, ValueError):
            pass

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            pbp = fetch_play_by_play(gid, session=session)
            is_po = is_playoff_pbp(pbp)
            df = fetch_game_shifts_dataframe(gid, session, is_playoff=is_po, pbp=pbp)
            if not df.empty:
                df.to_parquet(path, index=False)
            return "fetched" if not df.empty else "empty"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError, OSError) as e:
            last_exc = e
            wait = 2.0 ** attempt + 0.5
            time.sleep(wait)
            session.close()
    return f"FAILED: {last_exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--delay", type=float, default=0.15, help="Sec between fetches (politeness)")
    args = ap.parse_args()

    files = sorted(args.cache_dir.glob("pbp_*.parquet"))
    season_files = []
    for f in files:
        try:
            gid = nhl_id_from_pbp_path(f)
        except Exception:
            continue
        if nhl_season_id_from_game_id(gid) == int(args.season):
            season_files.append((gid, f))

    print(f"season {args.season}: {len(season_files)} PBP games to ensure shifts for", flush=True)
    sess = requests.Session()
    t0 = time.time()
    n_cached = n_fetched = n_failed = 0
    for i, (gid, _f) in enumerate(season_files, 1):
        status = fetch_one_with_retry(gid, args.cache_dir, sess, max_retries=args.max_retries)
        if status == "cached":
            n_cached += 1
        elif status == "fetched" or status == "empty":
            n_fetched += 1
            time.sleep(args.delay)
        else:
            n_failed += 1
            print(f"  game {gid}: {status}", file=sys.stderr, flush=True)
        if i % args.progress_every == 0:
            print(
                f"  {i}/{len(season_files)}  cached={n_cached}  fetched={n_fetched}  failed={n_failed}  {time.time() - t0:.0f}s",
                flush=True,
            )

    print(f"DONE  cached={n_cached}  fetched={n_fetched}  failed={n_failed}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
