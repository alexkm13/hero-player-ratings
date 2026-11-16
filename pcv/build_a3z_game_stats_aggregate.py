#!/usr/bin/env python3
"""Build definition-compatible A3Z player-season rates from raw Game Stats.

This ETL deliberately keeps its output separate from the legacy pivot-derived
aggregate.  Numerators and denominators are taken from the same raw player-game
rows, and every probability is checked before export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = (ROOT / "data" / "a3z" / "2025-26 Season/Season Totals Sheet.xlsx")
DEFAULT_CSV = ROOT / "data/pcv/a3z_game_stats_player_season_rates.csv"
DEFAULT_PARQUET = ROOT / "data/pcv/a3z_game_stats_player_season_rates.parquet"
DEFAULT_META = ROOT / "data/pcv/a3z_game_stats_player_season_rates.meta.json"

SOURCE_TO_OUTPUT = {
    "5v5 TOI": "toi_5v5_min",
    "Shots": "shots",
    "Chances": "chances",
    "Passes": "passes",
    "Primary Shot Assists": "primary_shot_assists",
    "Chance Assists": "chance_assists",
    "Zone Entries": "entries",
    "Carries": "entry_carries",
    "Failed Entries": "failed_entries",
    "Entries w/ Passing Play": "entries_w_passing",
    "Recoveries": "recoveries",
    "Carries w/ Chances": "carries_w_chances",
    "Forecheck Pressures": "forecheck_pressures",
    "DZ Puck Touches": "dz_puck_touches",
    "DZ Retrievals": "dz_retrievals",
    "Zone Exits": "zone_exits",
    "Exits w/ Possession": "controlled_exits",
    "Carried Exits": "carried_exits",
    "Passed Exits": "passed_exits",
    "Clears": "clears",
    "Missed Passes": "missed_pass_exits",
    "Retrievals Leading to Exits": "retrievals_to_exits",
    "Botched Retrievals": "botched_retrievals",
    "Exchanges": "exchanges",
    "Failed Exit": "failed_exits",
    "Rushed Exits": "rushed_exits",
    "Second Touch Exits": "second_touch_exits",
    "Targets": "entry_defense_targets",
    "Denials": "entry_denials",
    "Passes Allowed": "entry_passes_allowed",
    "Carries w/ Chance Against": "entry_carries_with_chance_against",
}

RATE_COLUMNS = (
    "entry_failure_rate",
    "retrieval_to_exit_rate",
    "botched_retrieval_rate",
    "failed_exit_rate",
    "missed_pass_exit_rate",
    "controlled_exit_share",
    "shot_chance_rate",
)


def _strip_headers(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    normalized = [re.sub(r"\s+", " ", str(col).strip()) for col in out.columns]
    seen: dict[str, int] = {}
    unique: list[str] = []
    for col in normalized:
        occurrence = seen.get(col, 0)
        unique.append(col if occurrence == 0 else f"{col}.{occurrence}")
        seen[col] = occurrence + 1
    out.columns = unique
    return out


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(num, errors="coerce")
    denominator = pd.to_numeric(den, errors="coerce").replace(0.0, np.nan)
    return numerator / denominator


def aggregate_game_stats(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    d = _strip_headers(raw)
    required = {"Player", "Team", "Pos.", "Year", "Game", *SOURCE_TO_OUTPUT}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing Game Stats columns: {missing}")

    regular = d["Year"].astype(str).str.fullmatch(r"20\d{2}-\d{2}", na=False)
    d = d.loc[regular, ["Player", "Team", "Pos.", "Year", "Game", *SOURCE_TO_OUTPUT]].copy()
    d = d.rename(
        columns={
            "Player": "player",
            "Team": "team",
            "Pos.": "position",
            "Year": "season",
            "Game": "game",
            **SOURCE_TO_OUTPUT,
        }
    )
    d["player"] = d["player"].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    d["team"] = d["team"].astype(str).str.strip()
    d["position"] = d["position"].astype(str).str.strip().str.upper()
    d["game"] = d["game"].astype(str).str.strip()
    key = ["player", "team", "season", "game"]
    duplicate_rows = int(d.duplicated(key, keep=False).sum())
    if duplicate_rows:
        raise ValueError(f"Duplicate raw player-game keys: {duplicate_rows} rows")

    numeric = list(SOURCE_TO_OUTPUT.values())
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    group = ["player", "team", "position", "season"]
    totals = d.groupby(group, dropna=False, as_index=False)[numeric].sum()
    exposure = (
        d.groupby(group, dropna=False, as_index=False)
        .agg(source_rows=("game", "size"), tracked_games=("game", "nunique"))
    )
    out = totals.merge(exposure, on=group, how="left", validate="one_to_one")

    out["entry_attempts"] = out["entries"] + out["failed_entries"]
    out["retrieval_opportunities"] = out["dz_retrievals"] + out["botched_retrievals"]
    out["exit_resolution_attempts"] = (
        out["zone_exits"] + out["missed_pass_exits"] + out["failed_exits"]
    )
    out["entry_failure_rate"] = _safe_div(out["failed_entries"], out["entry_attempts"])
    out["retrieval_to_exit_rate"] = _safe_div(
        out["retrievals_to_exits"], out["retrieval_opportunities"]
    )
    out["botched_retrieval_rate"] = _safe_div(
        out["botched_retrievals"], out["retrieval_opportunities"]
    )
    out["failed_exit_rate"] = _safe_div(out["failed_exits"], out["exit_resolution_attempts"])
    out["missed_pass_exit_rate"] = _safe_div(
        out["missed_pass_exits"], out["exit_resolution_attempts"]
    )
    out["controlled_exit_share"] = _safe_div(
        out["controlled_exits"], out["exit_resolution_attempts"]
    )
    out["shot_chance_rate"] = _safe_div(out["chances"], out["shots"])

    for col in RATE_COLUMNS:
        bad = out[col].notna() & ~out[col].between(0.0, 1.0, inclusive="both")
        if bool(bad.any()):
            raise ValueError(f"Impossible {col}: {int(bad.sum())} player-season-team rows")

    toi = out["toi_5v5_min"].replace(0.0, np.nan)
    for col in (
        "entries",
        "entry_carries",
        "passes",
        "retrievals_to_exits",
        "dz_retrievals",
        "zone_exits",
        "controlled_exits",
        "forecheck_pressures",
        "shots",
        "chances",
    ):
        out[f"{col}_per_60"] = out[col] * 60.0 / toi

    out = out.sort_values(["season", "player", "team"]).reset_index(drop=True)
    season_summary = (
        d.groupby("season", as_index=False)
        .agg(
            raw_rows=("game", "size"),
            tracked_games=("game", "nunique"),
            tracked_teams=("team", "nunique"),
            tracked_players=("player", "nunique"),
            tracked_player_toi=("toi_5v5_min", "sum"),
        )
        .to_dict(orient="records")
    )
    diagnostics: dict[str, object] = {
        "grain": "unique player-team-season",
        "raw_grain": "unique player-team-season-game",
        "raw_rows": int(len(d)),
        "duplicate_raw_keys": duplicate_rows,
        "output_rows": int(len(out)),
        "duplicate_output_keys": int(out.duplicated(group, keep=False).sum()),
        "seasons": season_summary,
        "rate_bounds_pass": True,
    }
    return out, diagnostics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out-meta", type=Path, default=DEFAULT_META)
    args = parser.parse_args()

    raw = pd.read_excel(args.workbook, sheet_name="Game Stats", usecols="A:CQ")
    output, diagnostics = aggregate_game_stats(raw)
    diagnostics["source_workbook"] = str(args.workbook)
    diagnostics["source_sha256"] = _sha256(args.workbook)

    for path in (args.out_csv, args.out_parquet, args.out_meta):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out_csv, index=False)
    output.to_parquet(args.out_parquet, index=False)
    args.out_meta.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_parquet}")
    print(f"Wrote {args.out_meta}")


if __name__ == "__main__":
    main()
