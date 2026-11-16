#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent

DEFAULT_EXISTING = ROOT / "data" / "pcv" / "a3z_player_season_rates_modules.csv"
DEFAULT_OUT_HIST = ROOT / "data" / "pcv" / "a3z_player_season_rates_modules_2016_2020.csv"
DEFAULT_OUT_MERGED = ROOT / "data" / "pcv" / "a3z_player_season_rates_modules_extended.csv"

SEASON_DIRS_DEFAULT = {
    "2016-17": (ROOT / "data" / "a3z" / "2016-17 Season/Team Pages"),
    "2017-18": (ROOT / "data" / "a3z" / "2017-18 Season/Team Pages"),
    "2018-19": (ROOT / "data" / "a3z" / "2018-19 Season/Team Pages"),
    "2019-20": (ROOT / "data" / "a3z" / "2019-20 Season/Team Pages"),
}

TEAM_NAME_TO_ABBR = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Detroit Red WIngs": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Vancouver Canucks": "VAN",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
    "Winnipeg JEts": "WPG",
    "Vegas Golden Knights": "VGK",
}

FINAL_COLS = [
    "player", "team", "position", "season", "toi_5v5_min",
    "entries", "shots", "chances", "carries", "passes", "chance_assists", "primary_shot_assists",
    "controlled_exits", "carried_exits", "passed_exits", "retrievals_to_exits", "forecheck_pressures",
    "denials", "targets", "passes_allowed", "failed_entries", "failed_exits", "zone_exits",
    "entries_w_passing", "carries_w_chances", "recoveries", "shots_off_forecheck_cycle",
    "assists_off_forecheck", "shots_off_cycle", "assists_off_cycle", "home_plate_passes",
    "center_lane_assists", "carries_w_chance_against", "oz_touches", "nz_touches", "dz_touches_zone",
    "dz_retrieval", "dz_counter", "nz_turnover", "nz_regroup", "nz_reload", "dz_controlled_breakout",
    "microstat_game_score", "gs_offense", "gs_entry_defense", "gs_exits", "gs_forechecking",
    "gs_special_teams", "gs_5v5", "entries_per_60", "carries_per_60", "passes_per_60",
    "chance_assists_per_60", "primary_shot_assists_per_60", "entries_with_passing_play_per_60",
    "carries_with_chances_per_60", "recoveries_per_60", "controlled_exits_per_60", "carried_exits_per_60",
    "passed_exits_per_60", "retrievals_leading_to_exits_per_60", "forecheck_pressures_per_60", "shots_per_60",
    "chances_per_60", "denials_per_target", "passes_allowed_per_target", "carries_with_chance_against_per_target",
    "carries_with_chance_against_per_60", "failed_entries_per_entry", "failed_exits_per_exit_touch",
    "home_plate_passes_per_60", "center_lane_assists_per_60", "oz_touches_per_60", "dz_retrievals_per_60",
    "dz_controlled_breakout_per_60", "cycle_forecheck_events_per_60", "non_shot_engine_per_60",
    "defensive_exits_after_retrieval_rate", "module_transition_entry_creation", "module_pre_shot_playmaking",
    "module_recovery_possession_extension", "module_breakout_exit_value", "module_entry_defense",
]

NUMERIC_BASE_COLS = [c for c in FINAL_COLS if c not in {"player", "team", "position", "season"}]


def _norm_name(x: object) -> str:
    s = str(x or "").strip()
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _key_name(x: object) -> str:
    s = _norm_name(x).lower().replace(".", "")
    return s


def _norm_col(x: object) -> str:
    s = str(x or "").strip().lower()
    s = s.replace("\xa0", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _as_num(x: object) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _find_header_row(path: Path, sheet_name: str, *, must_have: str = "player", max_scan_rows: int = 20) -> int | None:
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            return None
        ws = wb[sheet_name]
        target = _norm_col(must_have)
        for r in range(1, max_scan_rows + 1):
            vals = [_norm_col(v) for v in next(ws.iter_rows(min_row=r, max_row=r, values_only=True))]
            if target in vals:
                return r
        return None
    finally:
        wb.close()


def _read_table(path: Path, sheet_name: str) -> pd.DataFrame:
    h = _find_header_row(path, sheet_name, must_have="player")
    if h is None:
        return pd.DataFrame()
    try:
        d = pd.read_excel(path, sheet_name=sheet_name, header=h - 1)
    except Exception:
        return pd.DataFrame()
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d.columns = [_norm_col(c) for c in d.columns]
    d = d.loc[:, ~pd.Index(d.columns).duplicated(keep="first")]
    if "player" not in d.columns:
        return pd.DataFrame()
    d["player"] = d["player"].map(_norm_name)
    d = d[d["player"].astype(str).str.len() > 0].copy()
    d["player_key"] = d["player"].map(_key_name)
    return d


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _extract_sheet_values(df: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["player_key", "player"] + list(mapping.keys()))
    out = df[["player_key", "player"]].copy()
    for tgt, cands in mapping.items():
        c = _pick_col(df, cands)
        out[tgt] = pd.to_numeric(df[c], errors="coerce") if c is not None else np.nan
    # Historical Shots & Passes sheets can repeat players across situations.
    # Collapsing duplicate rows via median numeric values generally recovers
    # the 5v5 line while preserving one row per player.
    if out.duplicated(["player_key"]).any():
        num_cols = [c for c in out.columns if c not in {"player_key", "player"}]
        agg = {c: "median" for c in num_cols}
        agg["player"] = "first"
        out = out.groupby("player_key", as_index=False).agg(agg)
    return out


def _read_team_workbook(path: Path, season: str) -> pd.DataFrame:
    team_name = re.sub(r"\s+\d{4}-\d{2}$", "", path.stem.strip())
    team_abbr = TEAM_NAME_TO_ABBR.get(team_name)
    if not team_abbr:
        return pd.DataFrame()

    entries = _read_table(path, "Entries")
    exits = _read_table(path, "Exits")
    entry_def = _read_table(path, "Entry Defense")
    fore = _read_table(path, "Forechecking")
    shots = _read_table(path, "Shots & Passes")
    ptypes = _read_table(path, "Pass Types")

    if all(x.empty for x in [entries, exits, entry_def, shots]):
        return pd.DataFrame()

    m_entries = {
        "toi_5v5_min": ["5v5_toi"],
        "entries": ["entries"],
        "carries": ["carries"],
        "passes": ["passes"],
        "failed_entries": ["failed_entries", "failed", "fail"],
        "recoveries": ["recovered_dump_ins"],
        "entries_w_passing": ["entries_w_passing_play"],
    }
    m_exits = {
        "toi_5v5_min_exits": ["5v5_toi"],
        "zone_exits": ["exits"],
        "controlled_exits": ["possession_exits"],
        "carried_exits": ["carry"],
        "passed_exits": ["pass"],
        "failed_exits": ["failed_exit", "fail"],
        "dz_controlled_breakout": ["transition"],
    }
    m_entry_def = {
        "targets": ["targets"],
        "denials": ["break_up", "break_ups"],
        "passes_allowed": ["passes_allowed"],
        "carries_w_chance_against": ["shots_off_entries"],
    }
    m_fore = {
        "forecheck_pressures": ["applied_pressure"],
        "shots_off_forecheck_cycle": ["pressure_forced_fail_turn"],
        "assists_off_forecheck": ["pressure_forced_clear"],
        "shots_off_cycle": ["dumps_forced"],
    }
    m_shots = {
        "shots": ["shots"],
        "chances": ["chances"],
        "passes_shots": ["passes"],
        "primary_shot_assists": ["a1"],
        "chance_assists": ["a1"],
        "gs_offense": ["chances"],
    }
    m_ptypes = {
        "home_plate_passes": ["home_plate"],
        "center_lane_assists": ["high_danger", "danger_zone", "hd"],
        "oz_touches": ["oz"],
        "nz_touches": ["nz"],
        "dz_touches_zone": ["dz"],
    }

    tables = [
        _extract_sheet_values(entries, m_entries),
        _extract_sheet_values(exits, m_exits),
        _extract_sheet_values(entry_def, m_entry_def),
        _extract_sheet_values(fore, m_fore),
        _extract_sheet_values(shots, m_shots),
        _extract_sheet_values(ptypes, m_ptypes),
    ]

    base = pd.DataFrame(columns=["player_key", "player"])
    for t in tables:
        if t.empty:
            continue
        base = t if base.empty else base.merge(t, on=["player_key", "player"], how="outer")

    if base.empty:
        return pd.DataFrame()

    base["toi_5v5_min"] = pd.to_numeric(base.get("toi_5v5_min"), errors="coerce")
    toi2 = pd.to_numeric(base.get("toi_5v5_min_exits"), errors="coerce")
    base.loc[base["toi_5v5_min"].isna(), "toi_5v5_min"] = toi2.loc[base["toi_5v5_min"].isna()]
    base = base.drop(columns=[c for c in ["toi_5v5_min_exits"] if c in base.columns], errors="ignore")

    if "passes_shots" in base.columns:
        base["passes"] = pd.to_numeric(base.get("passes"), errors="coerce")
        p2 = pd.to_numeric(base.get("passes_shots"), errors="coerce")
        base.loc[base["passes"].isna(), "passes"] = p2.loc[base["passes"].isna()]
        base = base.drop(columns=["passes_shots"], errors="ignore")

    base["team"] = team_abbr
    base["position"] = pd.Series(pd.NA, index=base.index, dtype="object")
    if not entries.empty:
        cpos = _pick_col(entries, ["pos", "pos_"])
        if cpos is not None:
            pos_map = entries.set_index("player_key")[cpos].astype(str)
            base["position"] = base["player_key"].map(pos_map)
    if not exits.empty:
        cpos2 = _pick_col(exits, ["pos", "pos_"])
        if cpos2 is not None:
            miss = base["position"].isna()
            pos_map2 = exits.set_index("player_key")[cpos2].astype(str)
            base.loc[miss, "position"] = base.loc[miss, "player_key"].map(pos_map2)

    base["position"] = base["position"].astype(str).str.upper().str.strip()
    base.loc[~base["position"].isin(["C", "LW", "RW", "L", "R", "F", "D", "LD", "RD"]), "position"] = np.nan
    base.loc[base["position"].isin(["C", "LW", "RW", "L", "R", "F"]), "position"] = "F"
    base.loc[base["position"].isin(["D", "LD", "RD"]), "position"] = "D"

    base["season"] = season
    return base


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    d = pd.to_numeric(den, errors="coerce").replace(0.0, np.nan)
    n = pd.to_numeric(num, errors="coerce")
    return (n / d).replace([np.inf, -np.inf], np.nan)


def _per60(num: pd.Series, toi_min: pd.Series) -> pd.Series:
    d = pd.to_numeric(toi_min, errors="coerce").replace(0.0, np.nan)
    n = pd.to_numeric(num, errors="coerce")
    return (n / d) * 60.0


def _score_module(df: pd.DataFrame, cols: list[str], neg_cols: list[str] | None = None) -> pd.Series:
    neg = set(neg_cols or [])
    parts: list[pd.Series] = []
    for c in cols:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        mu = float(v.mean(skipna=True))
        sd = float(v.std(skipna=True, ddof=0))
        if not np.isfinite(sd) or sd < 1e-9:
            z = v * 0.0
        else:
            z = (v - mu) / sd
        if c in neg:
            z = -z
        parts.append(z)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _aggregate_cross_team(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    num_cols = [c for c in work.columns if c not in {"player_key", "player", "team", "position", "season"}]
    for c in num_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    rows = []
    for (season, pkey), g in work.groupby(["season", "player_key"], dropna=False):
        row = {"season": season, "player_key": pkey}
        names = g["player"].dropna().astype(str)
        row["player"] = names.iloc[0] if len(names) else ""
        poss = g["position"].dropna().astype(str)
        row["position"] = poss.mode().iloc[0] if len(poss) else ""
        teams = [t for t in g["team"].dropna().astype(str).tolist() if t]
        u = sorted(set(teams))
        row["team"] = u[0] if len(u) == 1 else ("MULTI" if len(u) > 1 else "")

        for c in num_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            if c == "toi_5v5_min":
                # Keep one TOI value per team and then sum across teams.
                # This prevents cross-sheet duplication from inflating TOI.
                by_team = g.groupby("team", dropna=False)[c].max(min_count=1).dropna()
                row[c] = float(by_team.sum()) if len(by_team) else np.nan
            else:
                row[c] = float(s.sum(skipna=True)) if s.notna().any() else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def _compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Ensure expected base columns exist.
    for c in [
        "entries", "shots", "chances", "carries", "passes", "chance_assists", "primary_shot_assists",
        "controlled_exits", "carried_exits", "passed_exits", "retrievals_to_exits", "forecheck_pressures",
        "denials", "targets", "passes_allowed", "failed_entries", "failed_exits", "zone_exits",
        "entries_w_passing", "carries_w_chances", "recoveries", "shots_off_forecheck_cycle",
        "assists_off_forecheck", "shots_off_cycle", "assists_off_cycle", "home_plate_passes",
        "center_lane_assists", "carries_w_chance_against", "oz_touches", "nz_touches", "dz_touches_zone",
        "dz_retrieval", "dz_counter", "nz_turnover", "nz_regroup", "nz_reload", "dz_controlled_breakout",
        "microstat_game_score", "gs_offense", "gs_entry_defense", "gs_exits", "gs_forechecking", "gs_special_teams", "gs_5v5",
    ]:
        if c not in d.columns:
            d[c] = np.nan

    t = pd.to_numeric(d["toi_5v5_min"], errors="coerce")
    d["entries_per_60"] = _per60(d["entries"], t)
    d["carries_per_60"] = _per60(d["carries"], t)
    d["passes_per_60"] = _per60(d["passes"], t)
    d["chance_assists_per_60"] = _per60(d["chance_assists"], t)
    d["primary_shot_assists_per_60"] = _per60(d["primary_shot_assists"], t)
    d["entries_with_passing_play_per_60"] = _per60(d["entries_w_passing"], t)
    d["carries_with_chances_per_60"] = _per60(d["carries_w_chances"], t)
    d["recoveries_per_60"] = _per60(d["recoveries"], t)
    d["controlled_exits_per_60"] = _per60(d["controlled_exits"], t)
    d["carried_exits_per_60"] = _per60(d["carried_exits"], t)
    d["passed_exits_per_60"] = _per60(d["passed_exits"], t)
    d["retrievals_leading_to_exits_per_60"] = _per60(d["retrievals_to_exits"], t)
    d["forecheck_pressures_per_60"] = _per60(d["forecheck_pressures"], t)
    d["shots_per_60"] = _per60(d["shots"], t)
    d["chances_per_60"] = _per60(d["chances"], t)
    d["denials_per_target"] = _safe_div(d["denials"], d["targets"])
    d["passes_allowed_per_target"] = _safe_div(d["passes_allowed"], d["targets"])
    d["carries_with_chance_against_per_target"] = _safe_div(d["carries_w_chance_against"], d["targets"])
    d["carries_with_chance_against_per_60"] = _per60(d["carries_w_chance_against"], t)
    d["failed_entries_per_entry"] = _safe_div(d["failed_entries"], d["entries"])
    d["failed_exits_per_exit_touch"] = _safe_div(d["failed_exits"], d["zone_exits"])
    d["home_plate_passes_per_60"] = _per60(d["home_plate_passes"], t)
    d["center_lane_assists_per_60"] = _per60(d["center_lane_assists"], t)
    d["oz_touches_per_60"] = _per60(d["oz_touches"], t)
    d["dz_retrievals_per_60"] = _per60(d["dz_retrieval"], t)
    d["dz_controlled_breakout_per_60"] = _per60(d["dz_controlled_breakout"], t)
    d["cycle_forecheck_events_per_60"] = _per60(
        pd.to_numeric(d["shots_off_forecheck_cycle"], errors="coerce")
        + pd.to_numeric(d["assists_off_forecheck"], errors="coerce")
        + pd.to_numeric(d["shots_off_cycle"], errors="coerce")
        + pd.to_numeric(d["assists_off_cycle"], errors="coerce"),
        t,
    )
    d["non_shot_engine_per_60"] = (
        pd.to_numeric(d["chance_assists_per_60"], errors="coerce")
        + pd.to_numeric(d["recoveries_per_60"], errors="coerce")
        + pd.to_numeric(d["entries_with_passing_play_per_60"], errors="coerce")
    )
    # Deprecated: these source fields are not a compatible numerator and
    # denominator.  Preserve the schema but fail closed until the provider
    # supplies a certified retrieval-opportunity count.
    d["defensive_exits_after_retrieval_rate"] = np.nan

    # Season-wise module scoring.
    mods = []
    for sid, g in d.groupby("season", dropna=False):
        x = g.copy()
        x["module_transition_entry_creation"] = _score_module(
            x,
            ["entries_per_60", "carries_per_60", "entries_with_passing_play_per_60", "carries_with_chances_per_60"],
        )
        x["module_pre_shot_playmaking"] = _score_module(
            x,
            ["primary_shot_assists_per_60", "chance_assists_per_60", "home_plate_passes_per_60", "center_lane_assists_per_60"],
        )
        x["module_recovery_possession_extension"] = _score_module(
            x,
            ["recoveries_per_60", "forecheck_pressures_per_60", "non_shot_engine_per_60", "cycle_forecheck_events_per_60"],
        )
        x["module_breakout_exit_value"] = _score_module(
            x,
            ["controlled_exits_per_60", "carried_exits_per_60", "passed_exits_per_60", "retrievals_leading_to_exits_per_60", "failed_exits_per_exit_touch"],
            neg_cols=["failed_exits_per_exit_touch"],
        )
        x["module_entry_defense"] = _score_module(
            x,
            ["denials_per_target", "passes_allowed_per_target", "carries_with_chance_against_per_target", "failed_exits_per_exit_touch"],
            neg_cols=["passes_allowed_per_target", "carries_with_chance_against_per_target", "failed_exits_per_exit_touch"],
        )
        mods.append(x)
    d = pd.concat(mods, ignore_index=True) if mods else d

    # Simple fill for game-score fields if absent.
    if d["gs_5v5"].isna().all():
        d["gs_5v5"] = d[[
            "module_transition_entry_creation",
            "module_pre_shot_playmaking",
            "module_recovery_possession_extension",
            "module_breakout_exit_value",
            "module_entry_defense",
        ]].mean(axis=1, skipna=True)
    if d["microstat_game_score"].isna().all():
        d["microstat_game_score"] = d["gs_5v5"]
    if d["gs_offense"].isna().all():
        d["gs_offense"] = d["module_pre_shot_playmaking"]
    if d["gs_entry_defense"].isna().all():
        d["gs_entry_defense"] = d["module_entry_defense"]
    if d["gs_exits"].isna().all():
        d["gs_exits"] = d["module_breakout_exit_value"]
    if d["gs_forechecking"].isna().all():
        d["gs_forechecking"] = d["module_recovery_possession_extension"]
    if d["gs_special_teams"].isna().all():
        d["gs_special_teams"] = np.nan

    return d


def _season_team_pages_to_player_season(season: str, team_pages_dir: Path) -> pd.DataFrame:
    files = sorted(team_pages_dir.glob("*.xlsx"))
    rows = []
    for p in files:
        d = _read_team_workbook(p, season)
        if len(d):
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    raw = pd.concat(rows, ignore_index=True)
    agg = _aggregate_cross_team(raw)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-dir-2016", type=Path, default=SEASON_DIRS_DEFAULT["2016-17"])
    ap.add_argument("--season-dir-2017", type=Path, default=SEASON_DIRS_DEFAULT["2017-18"])
    ap.add_argument("--season-dir-2018", type=Path, default=SEASON_DIRS_DEFAULT["2018-19"])
    ap.add_argument("--season-dir-2019", type=Path, default=SEASON_DIRS_DEFAULT["2019-20"])
    ap.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    ap.add_argument("--out-historical", type=Path, default=DEFAULT_OUT_HIST)
    ap.add_argument("--out-merged", type=Path, default=DEFAULT_OUT_MERGED)
    args = ap.parse_args()

    parts = []
    specs = [
        ("2016-17", args.season_dir_2016),
        ("2017-18", args.season_dir_2017),
        ("2018-19", args.season_dir_2018),
        ("2019-20", args.season_dir_2019),
    ]
    for season, d in specs:
        if not d.is_dir():
            print(f"Skip {season}: missing dir {d}")
            continue
        x = _season_team_pages_to_player_season(season, d)
        if x.empty:
            print(f"No rows parsed for {season}")
            continue
        parts.append(x)
        print(f"Parsed {season}: rows={len(x):,} from {d}")

    if not parts:
        raise SystemExit("No historical rows parsed.")

    hist = pd.concat(parts, ignore_index=True)
    hist = _compute_derived(hist)

    # Ensure full schema.
    for c in FINAL_COLS:
        if c not in hist.columns:
            hist[c] = np.nan
    hist = hist[FINAL_COLS].copy()

    # Coerce numerics.
    for c in NUMERIC_BASE_COLS:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")

    args.out_historical.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(args.out_historical, index=False)

    merged = hist.copy()
    if args.existing.is_file():
        cur = pd.read_csv(args.existing)
        for c in FINAL_COLS:
            if c not in cur.columns:
                cur[c] = np.nan
        cur = cur[FINAL_COLS].copy()
        merged = pd.concat([hist, cur], ignore_index=True)
        # keep one row per player+season; prefer existing for overlapping modern seasons
        merged["_ord"] = np.arange(len(merged))
        merged["_is_existing"] = 0
        merged.loc[merged.index >= len(hist), "_is_existing"] = 1
        merged = merged.sort_values(["player", "season", "_is_existing", "_ord"], ascending=[True, True, False, True])
        merged = merged.drop_duplicates(["player", "season"], keep="first").drop(columns=["_ord", "_is_existing"])

    args.out_merged.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_merged, index=False)

    print(f"Wrote historical: {args.out_historical} rows={len(hist):,}")
    print(f"Wrote merged: {args.out_merged} rows={len(merged):,}")
    print("Historical seasons:", sorted(hist["season"].dropna().astype(str).unique().tolist()))


if __name__ == "__main__":
    main()
