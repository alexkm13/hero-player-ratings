#!/usr/bin/env python3
"""Fit joint offensive/defensive RAPM coefficients from updated PTV stint targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from rapm_pcv_fit import player_stint_5v5_toi_sec, stint_frame_all_player_ids

ROOT = Path(__file__).resolve().parent
DEFAULT_POSS = ROOT / "data" / "pcv" / "possession_pcv_all.parquet"

TARGET_COLUMNS = {
    "total_pcv": ("home_total_pcv_per60_context_adj", "away_total_pcv_per60_context_adj"),
    "shot_pcv": ("home_shot_pcv_per60_context_adj", "away_shot_pcv_per60_context_adj"),
    "no_shot_threat_pcv": (
        "home_no_shot_threat_pcv_per60_context_adj",
        "away_no_shot_threat_pcv_per60_context_adj",
    ),
}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _as_ids(value: object) -> list[int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [int(x) for x in value]


def _game_teams(poss_path: Path) -> dict[int, tuple[str, str]]:
    poss = pd.read_parquet(poss_path, columns=["NHL_game_id", "home_team", "away_team"])
    poss = poss.drop_duplicates("NHL_game_id")
    return {
        int(row.NHL_game_id): (str(row.home_team), str(row.away_team))
        for row in poss.itertuples()
    }


def build_joint_matrix(
    stints: pd.DataFrame,
    target: str,
    game_teams: dict[int, tuple[str, str]],
    *,
    direct_defense_adjustment: bool = True,
) -> tuple[sparse.csc_matrix, np.ndarray, np.ndarray, list[int], list[str]]:
    home_col, away_col = TARGET_COLUMNS[target]
    player_ids = sorted(stint_frame_all_player_ids(stints))
    pidx = {player_id: i for i, player_id in enumerate(player_ids)}
    teams = sorted({team for pair in game_teams.values() for team in pair if team and team != "nan"})
    tidx = {team: i for i, team in enumerate(teams)}
    n_players = len(player_ids)
    off_base = 0
    def_base = n_players
    team_off_base = 2 * n_players
    team_def_base = team_off_base + len(teams)
    home_col_idx = team_def_base + len(teams)
    n_cols = home_col_idx + 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    y = np.zeros(2 * len(stints), dtype=float)
    weights = np.zeros(2 * len(stints), dtype=float)

    for i, row in enumerate(stints.itertuples()):
        gid = int(row.NHL_game_id)
        home_team, away_team = game_teams.get(gid, ("", ""))
        home_players = _as_ids(row.home_skater_ids)
        away_players = _as_ids(row.away_skater_ids)
        home_rate = float(getattr(row, home_col))
        away_rate = float(getattr(row, away_col))
        home_direct = float(getattr(row, "home_direct_defense_pcv_per60", 0.0) or 0.0)
        away_direct = float(getattr(row, "away_direct_defense_pcv_per60", 0.0) or 0.0)
        duration = float(row.duration_sec)

        for focal_home in (True, False):
            ridx = 2 * i + (0 if focal_home else 1)
            attackers = home_players if focal_home else away_players
            defenders = away_players if focal_home else home_players
            focal_team = home_team if focal_home else away_team
            opponent_team = away_team if focal_home else home_team
            if direct_defense_adjustment:
                y[ridx] = (home_rate - away_direct) if focal_home else (away_rate - home_direct)
            else:
                y[ridx] = home_rate if focal_home else away_rate
            weights[ridx] = duration
            for player_id in attackers:
                rows.append(ridx)
                cols.append(off_base + pidx[player_id])
                vals.append(1.0)
            for player_id in defenders:
                rows.append(ridx)
                cols.append(def_base + pidx[player_id])
                vals.append(-1.0)
            if focal_team in tidx:
                rows.append(ridx)
                cols.append(team_off_base + tidx[focal_team])
                vals.append(1.0)
            if opponent_team in tidx:
                rows.append(ridx)
                cols.append(team_def_base + tidx[opponent_team])
                vals.append(-1.0)
            if focal_home:
                rows.append(ridx)
                cols.append(home_col_idx)
                vals.append(1.0)

    matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(len(y), n_cols)).tocsc()
    return matrix, y, weights, player_ids, teams


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stints", type=Path, required=True)
    ap.add_argument("--target", choices=sorted(TARGET_COLUMNS), required=True)
    ap.add_argument("--poss", type=Path, default=DEFAULT_POSS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=50000.0)
    ap.add_argument(
        "--direct-defense-adjustment",
        choices=("subtract", "none"),
        default="subtract",
        help="Preserve legacy subtraction or model the offensive component without duplicating direct-defense credit.",
    )
    args = ap.parse_args()

    stints = pd.read_parquet(args.stints)
    game_teams = _game_teams(args.poss)
    use_direct_defense = args.direct_defense_adjustment == "subtract"
    matrix, y, weights, player_ids, teams = build_joint_matrix(
        stints,
        args.target,
        game_teams,
        direct_defense_adjustment=use_direct_defense,
    )
    target_weighted_mean = float(np.average(y, weights=weights))
    centered_y = y - target_weighted_mean
    model = Ridge(alpha=float(args.alpha), fit_intercept=False, max_iter=10000, solver="lsqr")
    model.fit(matrix, centered_y, sample_weight=weights)
    n_players = len(player_ids)
    coef = np.asarray(model.coef_).ravel()
    if not np.isfinite(coef).all() or not np.isfinite(target_weighted_mean):
        raise RuntimeError("joint RAPM fit produced nonfinite coefficients or baseline")
    out = pd.DataFrame(
        {
            "player_id": player_ids,
            f"offensive_rapm_{args.target}_joint": coef[:n_players],
            f"defensive_rapm_{args.target}_joint": coef[n_players : 2 * n_players],
            "joint_alpha": float(args.alpha),
            "joint_alpha_hits_grid_max": False,
            "n_rows_joint": len(y),
            "joint_target": args.target,
        }
    )
    toi = player_stint_5v5_toi_sec(stints)
    out["stint_5v5_toi_sec"] = out["player_id"].map(toi).fillna(0.0)
    numeric = out.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError("joint RAPM output contains nonfinite numeric values")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".parquet":
        out.to_parquet(args.out, index=False)
        out.to_csv(args.out.with_suffix(".csv"), index=False)
    else:
        out.to_csv(args.out, index=False)
        out.to_parquet(args.out.with_suffix(".parquet"), index=False)
    metadata = {
        "stints_path": str(args.stints),
        "stints_sha256": _sha256(args.stints),
        "stints_lineage_path": str(args.stints.with_suffix(".meta.json")),
        "stints_lineage_present": args.stints.with_suffix(".meta.json").is_file(),
        "possession_path": str(args.poss),
        "possession_sha256": _sha256(args.poss),
        "out_coef": str(args.out),
        "out_coef_sha256": _sha256(args.out),
        "build_script_sha256": _sha256(Path(__file__)),
        "target": args.target,
        "alpha": float(args.alpha),
        "fit_intercept": False,
        "target_centering": "duration_weighted_mean",
        "target_weighted_mean": target_weighted_mean,
        "n_stints": len(stints),
        "n_joint_rows": len(y),
        "n_games": int(stints["NHL_game_id"].nunique()),
        "n_players": n_players,
        "n_teams": len(teams),
        "include_team_effects": True,
        "direct_defense_target_adjustment": use_direct_defense,
    }
    args.out.with_name(f"joint_meta_{args.out.stem}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
