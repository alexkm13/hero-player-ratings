#!/usr/bin/env python3
"""Fill missing NHL shift caches from official HTML TOI reports (TH/TV)."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import certifi
import pandas as pd
import requests
from bs4 import BeautifulSoup

from ingest import fetch_play_by_play
from ingest_shifts import (
    SHIFTS_INGEST_VERSION,
    clock_to_seconds,
    is_playoff_pbp,
    nhl_id_from_pbp_path,
    nhl_season_id_from_game_id,
    row_start_end_game_seconds,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"
REPORT_URL = "https://www.nhl.com/scores/htmlreports/{season}/{kind}{game_code}.HTM"
PLAYER_HEADING_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*$")


def _default_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("default") or "").strip()
    return str(value or "").strip()


def _side_roster(pbp: dict, side: str) -> tuple[str, dict[int, int]]:
    team = pbp.get(f"{side}Team") or {}
    team_id = int(team.get("id") or 0)
    abbrev = str(team.get("abbrev") or "").strip()
    by_number: dict[int, int] = {}
    for spot in pbp.get("rosterSpots") or []:
        if int(spot.get("teamId") or 0) != team_id:
            continue
        number = spot.get("sweaterNumber")
        player_id = spot.get("playerId")
        if number is not None and player_id is not None:
            by_number[int(number)] = int(player_id)
    return abbrev, by_number


def _elapsed_clock(text: str) -> str:
    return str(text).split("/", 1)[0].strip()


def parse_shift_report(
    html: str,
    *,
    game_id: int,
    team_abbrev: str,
    player_by_number: dict[int, int],
    is_playoff: bool,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    unmapped: set[int] = set()
    first_heading = soup.find("td", class_=lambda value: value and "playerHeading" in value)
    player_table = first_heading.find_parent("table") if first_heading is not None else None
    if player_table is None:
        return pd.DataFrame()
    player_id: int | None = None
    for tr in player_table.find_all("tr", recursive=False):
        heading = tr.find("td", class_=lambda value: value and "playerHeading" in value)
        if heading is not None:
            match = PLAYER_HEADING_RE.match(heading.get_text(" ", strip=True))
            player_id = None
            if match:
                sweater = int(match.group(1))
                player_id = player_by_number.get(sweater)
                if player_id is None:
                    unmapped.add(sweater)
            continue
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if (
            player_id is None
            or len(cells) < 5
            or not cells[0].strip().isdigit()
            or not cells[1].strip().isdigit()
        ):
            continue
        period = int(cells[1].strip())
        start_clock = _elapsed_clock(cells[2])
        end_clock = _elapsed_clock(cells[3])
        duration = clock_to_seconds(cells[4])
        start_sec, end_sec = row_start_end_game_seconds(
            period,
            start_clock,
            end_clock,
            is_playoff=is_playoff,
        )
        rows.append(
            {
                "game_id": int(game_id),
                "player_id": int(player_id),
                "duration_sec": float(duration),
                "start_game_sec": float(start_sec),
                "end_game_sec": float(end_sec),
                "start_time": start_clock,
                "end_time": end_clock,
                "period": period,
                "team_abbrev": team_abbrev,
                "_ingest_v": SHIFTS_INGEST_VERSION,
                "shift_source": "nhl_html_report",
            }
        )
    if unmapped:
        raise ValueError(f"unmapped sweater numbers for {team_abbrev}: {sorted(unmapped)}")
    return pd.DataFrame(rows)


def fetch_game_html_shifts(
    game_id: int,
    session: requests.Session,
    *,
    timeout: float = 60.0,
) -> pd.DataFrame:
    pbp = fetch_play_by_play(game_id, session=session)
    season = nhl_season_id_from_game_id(game_id)
    game_code = str(int(game_id))[-6:]
    playoff = is_playoff_pbp(pbp)
    parts: list[pd.DataFrame] = []
    for kind, side in (("TH", "home"), ("TV", "away")):
        abbrev, roster = _side_roster(pbp, side)
        response = session.get(
            REPORT_URL.format(season=season, kind=kind, game_code=game_code),
            timeout=timeout,
            verify=certifi.where(),
        )
        response.raise_for_status()
        parsed = parse_shift_report(
            response.text,
            game_id=game_id,
            team_abbrev=abbrev,
            player_by_number=roster,
            is_playoff=playoff,
        )
        if parsed.empty:
            raise ValueError(f"empty {kind} report")
        parts.append(parsed)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--seasons", default="20232024,20242025,20252026")
    ap.add_argument("--delay", type=float, default=0.10)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument(
        "--repair-html",
        action="store_true",
        help="Rebuild existing files only when they were created by this HTML fallback.",
    )
    args = ap.parse_args()

    seasons = {int(value.strip()) for value in args.seasons.split(",") if value.strip()}
    missing: list[tuple[int, Path]] = []
    for pbp_path in sorted(args.cache_dir.glob("pbp_*.parquet")):
        game_id = nhl_id_from_pbp_path(pbp_path)
        if nhl_season_id_from_game_id(game_id) not in seasons:
            continue
        shift_path = args.cache_dir / f"shifts_{game_id}.parquet"
        repair = False
        if args.repair_html and shift_path.is_file():
            try:
                source = pd.read_parquet(shift_path, columns=["shift_source"])["shift_source"]
                repair = source.astype(str).eq("nhl_html_report").all()
            except (OSError, ValueError, KeyError):
                repair = False
        if not shift_path.is_file() or repair:
            missing.append((game_id, pbp_path))
    if args.max_games > 0:
        missing = missing[: args.max_games]

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; pcv/html-shifts/1.0)"})
    fetched = failed = 0
    failures: list[dict[str, object]] = []
    started = time.time()
    for index, (game_id, _) in enumerate(missing, 1):
        error = ""
        for attempt in range(args.max_retries):
            try:
                shifts = fetch_game_html_shifts(game_id, session)
                shifts.to_parquet(args.cache_dir / f"shifts_{game_id}.parquet", index=False)
                fetched += 1
                error = ""
                break
            except (requests.RequestException, OSError, ValueError, KeyError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.5 * (attempt + 1))
        if error:
            failed += 1
            failures.append({"game_id": game_id, "error": error})
        if args.delay > 0:
            time.sleep(args.delay)
        if index % args.progress_every == 0 or index == len(missing):
            print(
                f"{index}/{len(missing)} fetched={fetched} failed={failed} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )

    metadata = {
        "requested_missing_games": len(missing),
        "fetched": fetched,
        "failed": failed,
        "failures": failures,
        "seconds": time.time() - started,
    }
    (args.cache_dir / "html_shift_fallback_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in metadata.items() if k != "failures"}, indent=2))


if __name__ == "__main__":
    main()
