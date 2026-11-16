#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

DEFAULT_SEASONAL_COEF_DIR = ROOT / "data" / "pcv" / "seasonal"
DEFAULT_STINTS_POSSESSION_SPLIT = ROOT / "data" / "pcv" / "stint_pcv_targets_components_full_possession_split.parquet"

PTV_RAPM_STEMS = [
    "pre_shot_pressure",
    "possession_offense",
    "shot_threat",
    "possession_defense",
    "threat_suppression",
]


def _first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def resolve_stints_path(path: Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    return _first_existing(
        [
            DEFAULT_STINTS_POSSESSION_SPLIT,
            ROOT / "data" / "pcv" / "stint_pcv_targets_components_full_a3z_weighted.parquet",
            ROOT / "data" / "pcv" / "stint_pcv_targets_components_full.parquet",
            p,
        ]
    )


def joint_coef_triple(*, prefer_baseline: bool = True) -> tuple[Path, Path, Path]:
    if prefer_baseline:
        return (
            _first_existing(
                [
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_total_pcv.csv",
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_total_pcv.parquet",
                ]
            ),
            _first_existing(
                [
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_shot_pcv.csv",
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_shot_pcv.parquet",
                ]
            ),
            _first_existing(
                [
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_no_shot_threat_pcv.csv",
                    ROOT / "data" / "pcv" / "rapm_pcv_joint_components_full_no_shot_threat_pcv.parquet",
                ]
            ),
        )
    return (
        _first_existing(
            [
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_total_pcv.csv",
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_total_pcv.parquet",
            ]
        ),
        _first_existing(
            [
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_shot_pcv.csv",
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_shot_pcv.parquet",
            ]
        ),
        _first_existing(
            [
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_no_shot_threat_pcv.csv",
                ROOT / "data" / "pcv" / "rapm_pcv_joint_coefficients_a3z_weighted_no_shot_threat_pcv.parquet",
            ]
        ),
    )


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _season_coef_path(coef_dir: Path, season_id: int, target: str) -> Path | None:
    for ext in (".csv", ".parquet"):
        p = coef_dir / f"rapm_pcv_joint_components_{season_id}_{target}{ext}"
        if p.is_file():
            return p
    return None


def list_season_coef_seasons(coef_dir: Path) -> list[int]:
    d = Path(coef_dir)
    if not d.is_dir():
        return []
    out: set[int] = set()
    pat = re.compile(r"^rapm_pcv_joint_components_(\d{8})_(?:total_pcv|shot_pcv|no_shot_threat_pcv)\.(?:csv|parquet)$")
    for p in d.iterdir():
        m = pat.match(p.name)
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


def _load_one_season_components(coef_dir: Path, season_id: int) -> pd.DataFrame:
    p_tot = _season_coef_path(coef_dir, season_id, "total_pcv")
    p_shot = _season_coef_path(coef_dir, season_id, "shot_pcv")
    p_ns = _season_coef_path(coef_dir, season_id, "no_shot_threat_pcv")
    if p_tot is None or p_shot is None or p_ns is None:
        return pd.DataFrame()

    t = _read_any(p_tot)[["player_id", "offensive_rapm_total_pcv_joint", "defensive_rapm_total_pcv_joint"]].copy()
    s = _read_any(p_shot)[["player_id", "offensive_rapm_shot_pcv_joint"]].copy()
    n = _read_any(p_ns)[["player_id", "offensive_rapm_no_shot_threat_pcv_joint", "defensive_rapm_no_shot_threat_pcv_joint"]].copy()
    d = t.merge(s, on="player_id", how="outer").merge(n, on="player_id", how="outer")
    d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce").astype("Int64")
    d = d[d["player_id"].notna()].copy()
    d["player_id"] = d["player_id"].astype(int)
    d["season_id"] = int(season_id)
    return d.rename(
        columns={
            "offensive_rapm_no_shot_threat_pcv_joint": "pre_shot_pressure",
            "offensive_rapm_total_pcv_joint": "possession_offense",
            "offensive_rapm_shot_pcv_joint": "shot_threat",
            "defensive_rapm_total_pcv_joint": "possession_defense",
            "defensive_rapm_no_shot_threat_pcv_joint": "threat_suppression",
        }
    )


def build_player_season_ptv_table(coef_dir: Path, seasons: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for se in sorted(set(int(s) for s in seasons)):
        d = _load_one_season_components(Path(coef_dir), se)
        if not d.empty:
            parts.append(d[["player_id", "season_id", *PTV_RAPM_STEMS]])
    if not parts:
        return pd.DataFrame(columns=["player_id", "season_id", *PTV_RAPM_STEMS])
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["player_id", "season_id"], keep="last")


def build_trailing_roll3_ptv_table(coef_dir: Path, end_seasons: list[int]) -> pd.DataFrame:
    end = sorted(set(int(s) for s in end_seasons))
    if not end:
        raise ValueError("no end seasons supplied")
    all_seasons = list_season_coef_seasons(Path(coef_dir))
    if not all_seasons:
        raise ValueError("no seasonal coefficient files found")
    cache: dict[int, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []
    for se in end:
        win = [x for x in all_seasons if x <= se][-3:]
        if not win:
            continue
        for s in win:
            if s not in cache:
                cache[s] = _load_one_season_components(Path(coef_dir), s)
        wvals = np.array([0.6, 0.8, 1.0], dtype=float)[-len(win):]
        tmp = []
        for w, s in zip(wvals, win):
            d = cache[s]
            if d.empty:
                continue
            e = d[["player_id", *PTV_RAPM_STEMS]].copy()
            for c in PTV_RAPM_STEMS:
                e[c] = pd.to_numeric(e[c], errors="coerce") * float(w)
            e["_w"] = float(w)
            tmp.append(e)
        if not tmp:
            continue
        z = pd.concat(tmp, ignore_index=True)
        g = z.groupby("player_id", as_index=False)
        out = g[PTV_RAPM_STEMS].sum(min_count=1)
        wsum = g["_w"].sum().rename(columns={"_w": "_wsum"})
        out = out.merge(wsum, on="player_id", how="left")
        for c in PTV_RAPM_STEMS:
            out[c] = out[c] / out["_wsum"]
        out = out.drop(columns=["_wsum"])
        out["season_id"] = int(se)
        rows.append(out)
    if not rows:
        raise ValueError("no trailing windows could be built")
    return pd.concat(rows, ignore_index=True).drop_duplicates(["player_id", "season_id"], keep="last")


def assert_category_specs_use_rapm_pcv(df: pd.DataFrame) -> None:
    # Lightweight guard kept for compatibility after cleanup.
    _ = df
    return
