"""Cross-check a dated bulk snapshot against NHL schedules and standings."""
from datetime import date
import json
import os
from pathlib import Path

import pandas as pd
import requests

from hero.common import ARTIFACTS, DATA

OUT = ARTIFACTS / "live"
RAW = DATA / "live" / "raw"
CUT = date.fromisoformat(os.environ.get("HERO_LIVE_CUTOFF", "2026-11-15")).isoformat()
SEASON = 20262027
WEB = "https://api-web.nhle.com/v1"


def refresh():
    response = requests.get(f"{WEB}/standings/{CUT}", timeout=30)
    response.raise_for_status()
    standings = response.json()["standings"]
    expected = {row["teamAbbrev"]["default"]: row["gamesPlayed"] for row in standings}
    assert len(expected) == 32

    schedule = pd.read_parquet(RAW / "nhl_schedule_2027.parquet")
    keep = (schedule.game_type.eq("R") & schedule.game_state.isin(["FINAL", "OFF"])
            & schedule.game_date.le(CUT))
    s = schedule[keep].sort_values(["game_date", "game_id"])
    assert s.game_id.is_unique
    played = pd.concat([s.home_team_abbr, s.away_team_abbr]).value_counts().to_dict()
    assert played == expected, "Bulk schedule and official standings disagree at this cutoff."
    rows = [{"id": int(g.game_id), "gameDate": g.game_date,
             "homeTeam": {"abbrev": g.home_team_abbr},
             "awayTeam": {"abbrev": g.away_team_abbr}}
            for g in s.itertuples(index=False)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schedule.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Verified {len(rows)} games through {CUT} against NHL standings.")


if __name__ == "__main__":
    refresh()
