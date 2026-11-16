#!/usr/bin/env python3
"""Build updated 5v5 PTV stint targets from cached PBP, shifts, and possession PTV."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from fit_pooled_xt import load_xtfit
from ingest_shifts import nhl_season_id_from_game_id
from pcv import add_per_event_xt
from stints_pcv import stint_targets_table, stints_5v5_for_game

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_POSS = ROOT / "data" / "pcv" / "possession_pcv_all.parquet"
DEFAULT_XT = ROOT / "data" / "models" / "xt_pooled_10x8.npz"
DEFAULT_OUT_DIR = ROOT / "data" / "pcv" / "seasonal" / "stint_slices"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _game_id(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _player_positions(cache_dir: Path) -> dict[int, str]:
    positions: dict[int, str] = {}
    for path in cache_dir.glob("player_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            player_id = int(payload.get("playerId") or payload.get("id") or path.stem[7:])
            position = str(payload.get("position") or payload.get("positionCode") or "").upper()
            if position:
                positions[player_id] = position
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return positions


def _infer_goalies_from_shifts(cache_dir: Path, positions: dict[int, str]) -> set[int]:
    paths = sorted(cache_dir.glob("shifts_*.parquet"))
    if not paths:
        return set()
    shifts = ds.dataset([str(path) for path in paths], format="parquet").to_table(
        columns=["game_id", "player_id", "duration_sec", "team_abbrev"]
    ).to_pandas()
    totals = shifts.groupby(
        ["game_id", "team_abbrev", "player_id"], as_index=False, sort=False
    )["duration_sec"].sum()
    leaders = totals.loc[
        totals.groupby(["game_id", "team_abbrev"], sort=False)["duration_sec"].idxmax()
    ].copy()
    leader_ids = set(pd.to_numeric(leaders["player_id"], errors="coerce").dropna().astype(int))
    known_skaters = {
        player_id for player_id, position in positions.items() if position in {"C", "L", "R", "D"}
    }
    return leader_ids - known_skaters


def _teams(pbp: pd.DataFrame) -> tuple[str, str]:
    home = pbp["Home_Team"].dropna() if "Home_Team" in pbp.columns else pd.Series(dtype=str)
    away = pbp["Away_Team"].dropna() if "Away_Team" in pbp.columns else pd.Series(dtype=str)
    return (
        str(home.iloc[0]).strip() if len(home) else "",
        str(away.iloc[0]).strip() if len(away) else "",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--poss", type=Path, default=DEFAULT_POSS)
    ap.add_argument("--xt", type=Path, default=DEFAULT_XT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--attribution", default="event_weighted_possession_split")
    ap.add_argument("--min-stint-sec", type=float, default=3.0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()

    out = args.out or DEFAULT_OUT_DIR / f"stint_pcv_targets_{args.season}.parquet"
    poss = pd.read_parquet(args.poss)
    poss = poss[pd.to_numeric(poss["NHL_game_id"], errors="coerce").astype("Int64").map(
        lambda x: nhl_season_id_from_game_id(int(x)) if pd.notna(x) else 0
    ) == int(args.season)].copy()
    poss_by_game = {int(gid): frame for gid, frame in poss.groupby("NHL_game_id", sort=False)}
    xt = load_xtfit(args.xt)
    positions = _player_positions(args.cache_dir)
    goalies = {player_id for player_id, position in positions.items() if position == "G"}
    goalie_source = "nhl_player_profiles"
    if not goalies:
        goalies = _infer_goalies_from_shifts(args.cache_dir, positions)
        goalie_source = "team_game_shift_toi_leader_excluding_known_skaters"
    if not goalies:
        raise SystemExit("Could not identify goalies from profiles or cached shifts.")

    pbp_files = [
        path
        for path in sorted(args.cache_dir.glob("pbp_*.parquet"))
        if nhl_season_id_from_game_id(_game_id(path)) == int(args.season)
    ]
    if int(args.max_games) > 0:
        pbp_files = pbp_files[: int(args.max_games)]
    parts: list[pd.DataFrame] = []
    games_with_shifts = games_built = 0
    t0 = time.time()
    for i, pbp_path in enumerate(pbp_files, 1):
        gid = _game_id(pbp_path)
        shift_path = args.cache_dir / f"shifts_{gid}.parquet"
        if not shift_path.is_file():
            continue
        games_with_shifts += 1
        pbp = pd.read_parquet(pbp_path)
        if pbp.empty:
            continue
        pbp = pbp.copy()
        pbp["NHL_game_id"] = gid
        pbp = add_per_event_xt(pbp, xt)
        shifts = pd.read_parquet(shift_path)
        home, away = _teams(pbp)
        stints = stints_5v5_for_game(
            gid,
            shifts,
            home,
            away,
            goalies,
            min_sec=float(args.min_stint_sec),
        )
        if not stints:
            continue
        game_poss = poss_by_game.get(gid, pd.DataFrame())
        if game_poss.empty:
            continue
        targets = stint_targets_table(
            pbp,
            game_poss,
            stints,
            is_playoff=str(gid)[4:6] == "03",
            attribution=args.attribution,
        )
        if not targets.empty:
            parts.append(targets)
            games_built += 1
        if i % int(args.progress_every) == 0:
            print(
                f"{i}/{len(pbp_files)} games shifts={games_with_shifts} "
                f"built={games_built} stints={sum(len(x) for x in parts):,} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    metadata = {
        "artifact": "pcv_stint_targets",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "season": int(args.season),
        "attribution": args.attribution,
        "pbp_games": len(pbp_files),
        "games_with_shifts": games_with_shifts,
        "games_built": games_built,
        "shift_coverage": games_with_shifts / len(pbp_files) if pbp_files else 0.0,
        "confirmed_goalies": len(goalies),
        "goalie_source": goalie_source,
        "stints": len(result),
        "output": str(out),
        "output_sha256": _sha256_file(out),
        "possession_corpus_path": str(args.poss.resolve()),
        "possession_corpus_sha256": _sha256_file(args.poss),
        "possession_corpus_lineage": None,
        "xt_path": str(args.xt.resolve()),
        "xt_sha256": _sha256_file(args.xt),
        "build_script_sha256": _sha256_file(Path(__file__)),
        "seconds": time.time() - t0,
    }
    poss_meta = args.poss.with_suffix(".meta.json")
    if poss_meta.is_file():
        metadata["possession_corpus_lineage"] = json.loads(poss_meta.read_text(encoding="utf-8"))
    out.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
