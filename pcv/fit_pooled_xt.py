#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pcv import prepare_attacking_frame
from xt.hockeynxt_core import XTFit, fit_xT
from xt.nhl_actions import build_actions_from_pbp

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_OUT = ROOT / "data" / "models" / "xt_pooled_10x8.npz"


def save_xtfit(path: Path, xt: XTFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        l=np.int32(xt.l),
        w=np.int32(xt.w),
        xT=np.asarray(xt.xT, dtype=float),
        shot_prob=np.asarray(xt.shot_prob, dtype=float),
        move_prob=np.asarray(xt.move_prob, dtype=float),
        p_goal_given_shot=np.asarray(xt.p_goal_given_shot, dtype=float),
        transition=np.asarray(xt.transition, dtype=float),
    )


def load_xtfit(path: Path) -> XTFit:
    z = np.load(path)
    return XTFit(
        l=int(z["l"]),
        w=int(z["w"]),
        xT=np.asarray(z["xT"], dtype=float),
        shot_prob=np.asarray(z["shot_prob"], dtype=float),
        move_prob=np.asarray(z["move_prob"], dtype=float),
        p_goal_given_shot=np.asarray(z["p_goal_given_shot"], dtype=float),
        transition=np.asarray(z["transition"], dtype=float),
    )


def _fit_from_cache(cache_dir: Path, *, grid_x: int, grid_y: int, max_games: int = 0) -> XTFit:
    files = sorted(cache_dir.glob("pbp_*.parquet"))
    if not files:
        raise SystemExit(f"No pbp_*.parquet files in {cache_dir}")
    if int(max_games) > 0:
        files = files[: int(max_games)]

    parts: list[pd.DataFrame] = []
    for i, f in enumerate(files, start=1):
        pbp = pd.read_parquet(f)
        atk = prepare_attacking_frame(pbp)
        parts.append(build_actions_from_pbp(atk))
        if i % 200 == 0 or i == len(files):
            print(f"  loaded {i}/{len(files)} games", flush=True)
    actions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if actions.empty:
        raise SystemExit("No actions built from cache; cannot fit xT")
    return fit_xT(actions, grid_x=grid_x, grid_y=grid_y)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit pooled xT from cached PBP and save as npz.")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--grid-x", type=int, default=10)
    ap.add_argument("--grid-y", type=int, default=8)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    xt = _fit_from_cache(
        args.cache_dir,
        grid_x=int(args.grid_x),
        grid_y=int(args.grid_y),
        max_games=int(args.max_games),
    )
    save_xtfit(args.out, xt)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
