#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DEFAULT_SEASON_DIRS = [
    (ROOT / "data" / "a3z" / "2019-20 Season"),
    (ROOT / "data" / "a3z" / "2021-22 Season"),
    (ROOT / "data" / "a3z" / "2022-23 Season"),
    (ROOT / "data" / "a3z" / "2022-23 Season(1)"),
    (ROOT / "data" / "a3z" / "2023-24 Season"),
    (ROOT / "data" / "a3z" / "2024-25 Season"),
    (ROOT / "data" / "a3z" / "2025-26 Season"),
]

DEFAULT_OUT_PARQUET = ROOT / "data" / "pcv" / "a3z_pxp_events.parquet"
DEFAULT_OUT_CSV = ROOT / "data" / "pcv" / "a3z_pxp_events.csv.gz"
DEFAULT_OUT_INVENTORY = ROOT / "data" / "pcv" / "a3z_pxp_etl_inventory.csv"


def _norm_col(x: object) -> str:
    s = str(x or "").strip().lower()
    s = s.replace("\xa0", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _as_num(v: pd.Series) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def _as_text(v: pd.Series) -> pd.Series:
    return v.astype(str).replace({"nan": "", "None": ""}).str.strip()


def _guess_season_from_path(path: Path) -> str:
    text = " / ".join([path.name, *[p.name for p in path.parents]])
    m = re.search(r"(20\d{2})-(\d{2})", text)
    if not m:
        return ""
    y1 = int(m.group(1))
    y2 = int(m.group(2))
    if (y1 + 1) % 100 != y2:
        return ""
    return f"{y1}-{y2:02d}"


def _guess_game_id_from_stem(path: Path) -> str:
    m = re.match(r"^(\d{5,})", path.stem.strip())
    return m.group(1) if m else ""


def _parse_clock_to_seconds(s: pd.Series) -> pd.Series:
    t = _as_text(s)
    out = pd.Series(np.nan, index=t.index, dtype=float)
    m = t.str.extract(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.\d+)?$")
    hh = _as_num(m[0]).fillna(0.0)
    mm = _as_num(m[1]).fillna(np.nan)
    ss = _as_num(m[2]).fillna(np.nan)
    valid = mm.notna() & ss.notna()
    out.loc[valid] = hh.loc[valid] * 3600.0 + mm.loc[valid] * 60.0 + ss.loc[valid]
    m2 = t.str.extract(r"^(\d{1,2}):(\d{2})(?:\.\d+)?$")
    mm2 = _as_num(m2[0])
    ss2 = _as_num(m2[1])
    valid2 = out.isna() & mm2.notna() & ss2.notna()
    out.loc[valid2] = mm2.loc[valid2] * 60.0 + ss2.loc[valid2]
    return out


def _is_candidate_workbook(path: Path) -> bool:
    p = str(path).lower()
    include_tokens = [
        "game log",
        "shot game log",
        "passing pxp",
        "playoff pxp",
        "tracking",
        "pxp",
    ]
    exclude_tokens = [
        "team pages",
        "team standings",
        "season totals sheet",
        "player master list",
        "checklist",
        "goalie sheet",
        "team stat",
    ]
    if any(t in p for t in exclude_tokens):
        return False
    return any(t in p for t in include_tokens)


def _find_workbooks(season_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in season_dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*.xlsx"):
            if _is_candidate_workbook(p):
                files.append(p)
    return sorted(set(files))


def _is_event_sheet(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    cols = set(df.columns)
    required_any = {"period", "time", "team"}
    if len(required_any.intersection(cols)) < 2:
        return False
    event_signals = {
        "shooter",
        "shot_type",
        "entry_type",
        "entry_by",
        "defended_by",
        "origin",
        "state",
        "retrieval",
        "pressure",
        "exit",
        "result",
    }
    return len(event_signals.intersection(cols)) >= 2


def _coerce_yes_no(v: pd.Series) -> pd.Series:
    t = _as_text(v).str.lower()
    return t.map(
        {
            "y": 1.0,
            "yes": 1.0,
            "true": 1.0,
            "1": 1.0,
            "n": 0.0,
            "no": 0.0,
            "false": 0.0,
            "0": 0.0,
        }
    )


def _sheet_to_events(path: Path, sheet_name: str, season_hint: str) -> pd.DataFrame:
    try:
        raw = pd.read_excel(path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()

    d = raw.copy()
    d.columns = [_norm_col(c) for c in d.columns]
    d = d.loc[:, ~pd.Index(d.columns).duplicated(keep="first")].copy()
    if not _is_event_sheet(d):
        return pd.DataFrame()

    out = pd.DataFrame(index=d.index)
    out["season"] = season_hint
    out["source_file"] = str(path)
    out["source_sheet"] = str(sheet_name)
    out["source_row"] = np.arange(2, len(d) + 2)

    col_map = {
        "period": ["period"],
        "clock": ["time"],
        "strength": ["strength"],
        "team": ["team"],
        "shooter": ["shooter"],
        "shot_type": ["shot_type", "shot_type_"],
        "a1": ["a1"],
        "a2": ["a2"],
        "a3": ["a3"],
        "a1_zone": ["a1_zone"],
        "a2_zone": ["a2_zone"],
        "a3_zone": ["a3_zone"],
        "is_sc": ["sc"],
        "is_sog": ["sog"],
        "screen": ["screen", "screen_"],
        "rush": ["rush", "rush_"],
        "origin": ["origin"],
        "oddman": ["oddman", "oddman_"],
        "is_goal": ["g"],
        "state": ["state"],
        "entry_type": ["entry_type"],
        "entry_by": ["entry_by"],
        "defended_by": ["defended_by"],
        "pass_flag": ["pass"],
        "lane": ["lane"],
        "dump_recovered": ["dump_recovered"],
        "chance_flag": ["chance"],
        "retrieval": ["retrieval"],
        "result": ["result"],
        "pressure": ["pressure"],
        "exit": ["exit"],
        "result_2": ["result_1"],
        "game_id": ["game_id"],
        "home_team": ["home", "home_"],
        "away_team": ["road", "away", "away_"],
        "event_date": ["date"],
    }
    cols = set(d.columns)
    for out_col, cands in col_map.items():
        src = next((c for c in cands if c in cols), None)
        out[out_col] = d[src] if src is not None else np.nan

    out["period"] = _as_num(out["period"])
    out["clock"] = _as_text(out["clock"])
    out["clock_seconds"] = _parse_clock_to_seconds(out["clock"])
    out["team"] = _as_text(out["team"])
    out["strength"] = _as_text(out["strength"])
    out["game_id"] = _as_text(out["game_id"])
    out.loc[out["game_id"] == "", "game_id"] = _guess_game_id_from_stem(path)

    for c in ["is_sc", "is_sog", "screen", "rush", "oddman", "is_goal", "pass_flag", "dump_recovered", "chance_flag"]:
        out[c] = _coerce_yes_no(out[c])

    text_cols = [
        "shooter",
        "shot_type",
        "a1",
        "a2",
        "a3",
        "a1_zone",
        "a2_zone",
        "a3_zone",
        "origin",
        "state",
        "entry_type",
        "entry_by",
        "defended_by",
        "lane",
        "retrieval",
        "result",
        "pressure",
        "exit",
        "result_2",
        "home_team",
        "away_team",
        "event_date",
    ]
    for c in text_cols:
        out[c] = _as_text(out[c])

    # Drop mostly empty lines, preserving rows with any key event fields.
    keep = (
        out["team"].ne("")
        | out["shooter"].ne("")
        | out["entry_by"].ne("")
        | out["defended_by"].ne("")
        | out["shot_type"].ne("")
        | out["origin"].ne("")
    )
    out = out[keep].copy()
    return out.reset_index(drop=True)


def _parse_workbook(path: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    season = _guess_season_from_path(path)
    inventory_rows: list[dict[str, object]] = []
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        inventory_rows.append(
            {
                "source_file": str(path),
                "season": season,
                "status": "open_failed",
                "notes": str(exc),
                "n_rows": 0,
                "n_sheets_parsed": 0,
            }
        )
        return pd.DataFrame(), inventory_rows

    parsed: list[tuple[str, pd.DataFrame]] = []
    for sn in xl.sheet_names:
        d = _sheet_to_events(path, sn, season)
        if d.empty:
            continue
        parsed.append((sn, d))

    if not parsed:
        inventory_rows.append(
            {
                "source_file": str(path),
                "season": season,
                "status": "no_event_sheets",
                "notes": ",".join(xl.sheet_names),
                "n_rows": 0,
                "n_sheets_parsed": 0,
            }
        )
        return pd.DataFrame(), inventory_rows

    # Prefer one canonical event sheet per workbook when present.
    # This avoids duplicating the same events from parallel/subset tabs
    # (e.g., Shots vs 5v5/5v4).
    by_name = {_norm_col(sn): (sn, d) for sn, d in parsed}
    chosen: list[tuple[str, pd.DataFrame]]
    if "tracking" in by_name:
        chosen = [by_name["tracking"]]
    elif "sheet1" in by_name:
        chosen = [by_name["sheet1"]]
    elif "shots" in by_name:
        chosen = [by_name["shots"]]
    else:
        chosen = parsed

    out = pd.concat([d for _, d in chosen], ignore_index=True)
    parsed_sheet_names = [sn for sn, _ in chosen]
    inventory_rows.append(
        {
            "source_file": str(path),
            "season": season,
            "status": "ok",
            "notes": ",".join(parsed_sheet_names),
            "n_rows": int(len(out)),
            "n_sheets_parsed": int(len(parsed_sheet_names)),
        }
    )
    return out, inventory_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--season-dirs",
        type=str,
        default=",".join(str(p) for p in DEFAULT_SEASON_DIRS),
        help="Comma-separated season root directories.",
    )
    ap.add_argument("--max-files", type=int, default=0, help="Optional cap for quick dry runs.")
    ap.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--out-inventory", type=Path, default=DEFAULT_OUT_INVENTORY)
    args = ap.parse_args()

    season_dirs = [Path(x.strip()) for x in str(args.season_dirs).split(",") if x.strip()]
    books = _find_workbooks(season_dirs)
    if args.max_files and args.max_files > 0:
        books = books[: int(args.max_files)]

    print(f"Candidate workbooks: {len(books):,}")
    all_events: list[pd.DataFrame] = []
    inv_rows: list[dict[str, object]] = []
    for i, p in enumerate(books, start=1):
        d, inv = _parse_workbook(p)
        inv_rows.extend(inv)
        if not d.empty:
            all_events.append(d)
        if i % 50 == 0:
            rows_so_far = int(sum(len(x) for x in all_events)) if all_events else 0
            print(f"  parsed {i:,}/{len(books):,} files; rows={rows_so_far:,}")

    inv_df = pd.DataFrame(inv_rows)
    args.out_inventory.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(args.out_inventory, index=False)
    print(f"Wrote inventory: {args.out_inventory} rows={len(inv_df):,}")

    if not all_events:
        raise SystemExit("No event rows parsed.")

    events = pd.concat(all_events, ignore_index=True)
    events["season"] = events["season"].replace("", np.nan)

    # Cross-file dedupe (most importantly for copied season folders such as
    # "2022-23 Season" and "2022-23 Season(1)").
    dedupe_cols = [
        "season",
        "game_id",
        "period",
        "clock",
        "team",
        "strength",
        "shooter",
        "shot_type",
        "a1",
        "a2",
        "a3",
        "origin",
        "state",
        "entry_type",
        "entry_by",
        "defended_by",
        "result",
        "pressure",
        "exit",
    ]
    available = [c for c in dedupe_cols if c in events.columns]
    if available:
        events = events.sort_values(["source_file", "source_sheet", "source_row"], na_position="last")
        events = events.drop_duplicates(available, keep="first")

    events = events.sort_values(["season", "game_id", "period", "clock_seconds", "source_file", "source_row"], na_position="last")
    events = events.reset_index(drop=True)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.out_parquet, index=False)
    events.to_csv(args.out_csv, index=False)
    print(f"Wrote events parquet: {args.out_parquet} rows={len(events):,}")
    print(f"Wrote events csv: {args.out_csv} rows={len(events):,}")

    season_counts = events["season"].fillna("unknown").value_counts().sort_index()
    print("Rows by season:")
    for sid, n in season_counts.items():
        print(f"  {sid}: {int(n):,}")


if __name__ == "__main__":
    main()
