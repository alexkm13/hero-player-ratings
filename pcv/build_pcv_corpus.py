#!/usr/bin/env python3
"""
Build league-wide **possession PCV** (and optional per-player xT_delta rollups) from all
``data/cache/pbp_*.parquet`` using the pooled xT model (``data/models/xt_pooled_10x8.npz``).

* Injects *NHL_game_id* (10-digit) from the filename so :func:`pcv.add_per_event_xt` and
  :func:`pcv.possession_pcv_table` group games correctly (5-digit *Game_Id* can collide).
* Writes ``possession_pcv_all.parquet`` (and optionally ``player_xt_delta_totals.parquet``).

  python3 build_pcv_corpus.py
  python3 build_pcv_corpus.py --no-players   # possession table only (faster)
  python3 build_pcv_corpus.py --max-games 10  # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fit_pooled_xt import load_xtfit
from pcv import add_per_event_xt, player_xt_delta_totals, possession_pcv_table
from pcv_v2 import enrich_game, score_possessions_with_pack

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_XT = ROOT / "data" / "models" / "xt_pooled_10x8.npz"
DEFAULT_OUT = ROOT / "data" / "pcv" / "possession_pcv_all.parquet"
DEFAULT_PLAYERS = ROOT / "data" / "pcv" / "player_xt_delta_totals.parquet"
DEFAULT_PCV_V2 = ROOT / "data" / "models" / "pcv_v2.joblib"


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_tree_sha256(files: list[Path], root: Path) -> str:
    """Content-address the exact ordered PBP input set without storing thousands of hashes."""
    digest = hashlib.sha256()
    for path in files:
        rel = str(path.relative_to(root)).encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _nhl_game_id_from_path(f: Path) -> int:
    s = f.stem
    if not s.startswith("pbp_"):
        raise ValueError(f"expected pbp_<id>.parquet, got {f.name}")
    return int(s[4:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--xt", type=Path, default=DEFAULT_XT, help="npz from fit_pooled_xt")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output possession PCV table")
    ap.add_argument(
        "--out-players",
        type=Path,
        default=DEFAULT_PLAYERS,
        help="Output player xT_delta sum (set --no-players to skip)",
    )
    ap.add_argument("--no-players", action="store_true", help="Skip player delta aggregation")
    ap.add_argument("--max-games", type=int, default=0, help="Process at most this many files (0=all)")
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--pcv-v2", type=Path, default=DEFAULT_PCV_V2, help="joblib pack from fit_pcv_v2.py")
    ap.add_argument("--legacy-pcv", action="store_true", help="Use heuristic max(xG,xT) PCV even if V2 pack exists")
    ap.add_argument("--meta", type=Path, help="Lineage JSON; defaults to <out>.meta.json")
    args = ap.parse_args()

    if not args.xt.is_file():
        print(f"Missing xT model: {args.xt}\nRun: python3 fit_pooled_xt.py", file=sys.stderr)
        raise SystemExit(1)

    files = sorted(args.cache_dir.glob("pbp_*.parquet"))
    if not files:
        print(f"No pbp_*.parquet in {args.cache_dir}", file=sys.stderr)
        raise SystemExit(1)
    if int(args.max_games or 0) > 0:
        files = files[: int(args.max_games)]

    print(f"Load xT: {args.xt}")
    xt = load_xtfit(args.xt)
    pack = None
    if not args.legacy_pcv and args.pcv_v2.is_file():
        import joblib

        pack = joblib.load(args.pcv_v2)
        print(f"Load PCV V2: {args.pcv_v2}")
    elif not args.legacy_pcv:
        print(f"No PCV V2 pack at {args.pcv_v2}; using heuristic PCV")

    poss_parts: list[pd.DataFrame] = []
    player_sum: pd.Series | None = None
    t0 = time.time()
    n = len(files)
    for i, f in enumerate(files):
        p = pd.read_parquet(f)
        gid = _nhl_game_id_from_path(f)
        p["NHL_game_id"] = gid
        ev = add_per_event_xt(p, xt)
        if pack is not None:
            ev2 = enrich_game(p, gid)
            t = score_possessions_with_pack(ev2, pack)
        else:
            t = possession_pcv_table(ev)
        if not t.empty:
            ht = p["Home_Team"].dropna()
            at = p["Away_Team"].dropna()
            t = t.copy()
            t["home_team"] = str(ht.iloc[0]) if len(ht) else ""
            t["away_team"] = str(at.iloc[0]) if len(at) else ""
        poss_parts.append(t)
        if not args.no_players:
            s = player_xt_delta_totals(ev)
            player_sum = s if player_sum is None else player_sum.add(s, fill_value=0.0)
        if (i + 1) % int(args.progress_every) == 0 or (i + 1) == n:
            el = time.time() - t0
            print(f"  {i+1}/{n} games  {el:.0f}s", flush=True)

    poss = pd.concat(poss_parts, ignore_index=True) if poss_parts else pd.DataFrame()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  rows={len(poss):,}  cols={list(poss.columns)}")

    meta_path = args.meta or args.out.with_suffix(".meta.json")
    season_counts: dict[str, int] = {}
    for path in files:
        season = str(_nhl_game_id_from_path(path))[:4]
        season_counts[season] = season_counts.get(season, 0) + 1
    metadata = {
        "artifact": "possession_pcv_corpus",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "pcv_v2" if pack is not None else "legacy_xt",
        "output_path": str(args.out.resolve()),
        "output_sha256": _sha256_file(args.out),
        "output_rows": int(len(poss)),
        "output_columns": list(poss.columns),
        "pbp_cache_dir": str(args.cache_dir.resolve()),
        "pbp_game_count": len(files),
        "pbp_season_start_counts": season_counts,
        "pbp_tree_sha256": _cache_tree_sha256(files, args.cache_dir),
        "xt_path": str(args.xt.resolve()),
        "xt_sha256": _sha256_file(args.xt),
        "pcv_model_path": str(args.pcv_v2.resolve()) if pack is not None else None,
        "pcv_model_sha256": _sha256_file(args.pcv_v2) if pack is not None else None,
        "pcv_model_train_seasons": pack.get("train_seasons") if pack is not None else None,
        "pcv_model_production": bool(pack.get("production", False)) if pack is not None else None,
        "build_script_sha256": _sha256_file(Path(__file__)),
        "pcv_v2_code_sha256": _sha256_file(ROOT / "pcv_v2.py"),
        "possession_code_sha256": _sha256_file(ROOT / "possession.py"),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote lineage {meta_path}")

    if not args.no_players and player_sum is not None and not player_sum.empty:
        pl = player_sum.sort_values(ascending=False).rename("xT_delta_sum")
        pl.to_frame().to_parquet(args.out_players, index=True)
        print(f"Wrote {args.out_players}  players={len(pl):,}")
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
