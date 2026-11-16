#!/usr/bin/env python3
"""
Tune within-category shot-creation component weights toward dangerous offense / shot threat.

Objective (walk-forward by season):
- threat persistence: corr(weighted blend, next-season threat proxy)
- process YoY: corr(blend, next-season offensive process basket)
- skill stickiness: corr(blend, next-season cat_shot_creation_z__adj) [lower weight]
- ranking penalties on latest F season (McDavid elite, finisher/creator separation)

Writes:
  data/pcv/shot_creation_component_weights_best.json
  data/pcv/shot_creation_component_weight_grid.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

COMPONENTS = [
    "rapm_hybrid_xGF/60",
    "eh_std_ixG/60",
    "eh_std_iCF/60",
    "a3z_chances_per_60",
    "a3z_carries_with_chances_per_60",
    "a3z_shots_per_60",
    "ptv_coalesced_rapm_shot_threat",
    "ptv_coalesced_possession_offense",
]

DEFAULT_CURRENT = {
    "rapm_hybrid_xGF/60": 0.22,
    "eh_std_ixG/60": 0.24,
    "eh_std_iCF/60": 0.10,
    "a3z_chances_per_60": 0.10,
    "a3z_carries_with_chances_per_60": 0.08,
    "a3z_shots_per_60": 0.05,
    "ptv_coalesced_rapm_shot_threat": 0.11,
    "ptv_coalesced_possession_offense": 0.10,
}

THREAT_COMPONENTS = [
    "ptv_coalesced_rapm_shot_threat",
    "rapm_hybrid_xGF/60",
    "a3z_chances_per_60",
    "a3z_carries_with_chances_per_60",
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _z_by_group(df: pd.DataFrame, col: str) -> pd.Series:
    x = _to_num(df[col])
    mu = x.groupby([df["season_id"], df["position_group"]]).transform("mean")
    sd = x.groupby([df["season_id"], df["position_group"]]).transform("std")
    return (x - mu) / sd.replace(0.0, np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 40:
        return float("nan")
    x, y = a[m], b[m]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _objective_target(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col, w in [
        ("yoy_xGF_per60", 0.35),
        ("yoy_xGA_per60", -0.35),
        ("yoy_onice_xg_share", 0.20),
        ("yoy_onice_goal_share", 0.10),
    ]:
        if col in df.columns:
            parts.append(w * _z_by_group(df, col))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return sum(parts)


def _threat_proxy(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for col in THREAT_COMPONENTS:
        if col in df.columns:
            parts.append(_z_by_group(df, col))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _blend(df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    w = np.asarray([float(weights[c]) for c in COMPONENTS], dtype=float)
    mat = np.column_stack([_z_by_group(df, c).to_numpy(float) for c in COMPONENTS])
    finite = np.isfinite(mat)
    num = np.nansum(np.where(finite, mat * w, 0.0), axis=1)
    den = np.sum(np.where(finite, np.abs(w), 0.0), axis=1)
    out = np.full(mat.shape[0], np.nan, dtype=float)
    m = den > 0
    out[m] = num[m] / den[m]
    return out


def _mean_season_corr(df: pd.DataFrame, pred: np.ndarray, y: pd.Series) -> float:
    tmp = df[["season_id"]].copy()
    tmp["pred"] = pred
    tmp["y"] = _to_num(y)
    rs: list[float] = []
    for _, g in tmp.groupby("season_id"):
        m = g["pred"].notna() & g["y"].notna()
        if int(m.sum()) < 40:
            continue
        rs.append(_corr(g.loc[m, "pred"].to_numpy(float), g.loc[m, "y"].to_numpy(float)))
    return float(np.nanmean(rs)) if rs else float("nan")


# Balanced threat + volume bands (user target): ixG ~22–27%, PTV ~20–26%, iCF ~18–20%.
IXG_BAND = (0.22, 0.27)
PTV_BAND = (0.20, 0.26)
ICF_BAND = (0.18, 0.20)
ACTIVE_IXG_BAND = IXG_BAND
ACTIVE_PTV_BAND = PTV_BAND
ACTIVE_ICF_BAND = ICF_BAND


def _constraints_ok(w: dict[str, float]) -> bool:
    if any(float(w[c]) < 0.0 for c in COMPONENTS):
        return False
    s = sum(float(w[c]) for c in COMPONENTS)
    if abs(s - 1.0) > 0.02:
        return False
    ixg = float(w["eh_std_ixG/60"])
    ptv = float(w["ptv_coalesced_rapm_shot_threat"])
    icf = float(w["eh_std_iCF/60"])
    if not (ACTIVE_IXG_BAND[0] <= ixg <= ACTIVE_IXG_BAND[1]):
        return False
    if not (ACTIVE_PTV_BAND[0] <= ptv <= ACTIVE_PTV_BAND[1]):
        return False
    if not (ACTIVE_ICF_BAND[0] <= icf <= ACTIVE_ICF_BAND[1]):
        return False
    if float(w["rapm_hybrid_xGF/60"]) < 0.08:
        return False
    if float(w["a3z_chances_per_60"]) < 0.05:
        return False
    if float(w["a3z_carries_with_chances_per_60"]) < 0.04:
        return False
    if float(w["a3z_shots_per_60"]) < 0.04:
        return False
    return True


def _balanced_seed_weights() -> dict[str, float]:
    """Midpoint of target bands; remainder split across shot-creation process inputs."""
    core = {
        "eh_std_ixG/60": float(np.mean(ACTIVE_IXG_BAND)),
        "ptv_coalesced_rapm_shot_threat": float(np.mean(ACTIVE_PTV_BAND)),
        "eh_std_iCF/60": float(np.mean(ACTIVE_ICF_BAND)),
    }
    rem = 1.0 - sum(core.values())
    tail = {
        "rapm_hybrid_xGF/60": 0.34 * rem,
        "a3z_chances_per_60": 0.28 * rem,
        "a3z_carries_with_chances_per_60": 0.19 * rem,
        "a3z_shots_per_60": 0.19 * rem,
    }
    return {**core, **tail}


def _rank_penalty(sub: pd.DataFrame, pred: np.ndarray) -> float:
    if sub.empty:
        return 1.0
    sub = sub.copy()
    sub["pred"] = pred
    sub = sub[sub["pred"].notna()]
    if sub.empty:
        return 1.0
    sub["rank"] = sub["pred"].rank(ascending=False, method="min")
    pen = 0.0

    def _rank_of(name: str) -> float | None:
        hit = sub[sub["player_name"].eq(name)]
        if hit.empty:
            return None
        return float(hit["rank"].iloc[0])

    md = _rank_of("Connor McDavid")
    mat = _rank_of("Auston Matthews")
    hertl = _rank_of("Tomas Hertl")
    reinhart = _rank_of("Sam Reinhart")
    if md is None or md > 8:
        pen += 0.06
    if mat is None or mat > 12:
        pen += 0.04
    if hertl is not None and reinhart is not None and hertl >= reinhart:
        pen += 0.03
    return pen


def _evaluate(
    train: pd.DataFrame,
    weights: dict[str, float],
    *,
    threat_w: float,
    process_w: float,
    stick_w: float,
    rank_penalty_w: float,
    rank_sub: pd.DataFrame,
) -> dict[str, float]:
    pred = _blend(train, weights)
    r_threat = _mean_season_corr(train, pred, train["y_next_threat"])
    r_process = _mean_season_corr(train, pred, train["y_next_process"])
    r_stick = _mean_season_corr(train, pred, train["y_next_sc"])
    pen = _rank_penalty(rank_sub, _blend(rank_sub, weights)) if not rank_sub.empty else 0.0
    obj = (
        float(threat_w) * float(r_threat)
        + float(process_w) * float(r_process)
        + float(stick_w) * float(r_stick)
        - float(rank_penalty_w) * float(pen)
    )
    return {
        "obj": obj,
        "r_threat": r_threat,
        "r_process": r_process,
        "r_stick": r_stick,
        "rank_penalty": pen,
    }


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["position_group"] = d["position_group"].astype(str).str.upper().str.strip()
    toi = _to_num(d.get("eh_rapm_TOI")).where(
        _to_num(d.get("eh_rapm_TOI")).notna(), _to_num(d.get("a3z_toi_5v5_min"))
    ).fillna(0.0)
    d = d[toi >= 250.0].copy()
    d = d.sort_values(["player_id", "season_id"])
    d["threat_proxy"] = _threat_proxy(d)
    d["process_proxy"] = _objective_target(d)
    d["y_next_threat"] = d.groupby("player_id")["threat_proxy"].shift(-1)
    d["y_next_process"] = d.groupby("player_id")["process_proxy"].shift(-1)
    d["y_next_sc"] = d.groupby("player_id")["cat_shot_creation_z__adj"].shift(-1)
    return d


def _search_role(
    d: pd.DataFrame,
    role: str,
    *,
    n_samples: int,
    seed: int,
    eval_seasons: list[int],
    rank_season: int,
    threat_w: float,
    process_w: float,
    stick_w: float,
    rank_penalty_w: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    sub = d[d["position_group"].eq(role)].copy()
    train = sub[sub["season_id"].isin(eval_seasons)].copy()
    rank_sub = sub[sub["season_id"].eq(rank_season)].copy()

    rows: list[dict[str, object]] = []
    best_w: dict[str, float] | None = None
    best_obj = float("-inf")

    def _record(weights: dict[str, float], label: str) -> None:
        nonlocal best_w, best_obj
        if not _constraints_ok(weights):
            return
        met = _evaluate(
            train,
            weights,
            threat_w=threat_w,
            process_w=process_w,
            stick_w=stick_w,
            rank_penalty_w=rank_penalty_w,
            rank_sub=rank_sub,
        )
        row = {"role": role, "label": label, **{f"w__{c}": float(weights[c]) for c in COMPONENTS}, **met}
        rows.append(row)
        if float(met["obj"]) > best_obj:
            best_obj = float(met["obj"])
            best_w = dict(weights)

    _record(DEFAULT_CURRENT, "current")
    _record(_balanced_seed_weights(), "balanced_seed")

    rng = np.random.default_rng(seed)
    for i in range(int(n_samples)):
        ixg = float(rng.uniform(ACTIVE_IXG_BAND[0], ACTIVE_IXG_BAND[1]))
        ptv = float(rng.uniform(ACTIVE_PTV_BAND[0], ACTIVE_PTV_BAND[1]))
        icf = float(rng.uniform(ACTIVE_ICF_BAND[0], ACTIVE_ICF_BAND[1]))
        rem = 1.0 - ixg - ptv - icf
        if rem < 0.12:
            continue
        tail = rng.dirichlet(np.ones(4)) * rem
        weights = {
            "eh_std_ixG/60": ixg,
            "ptv_coalesced_rapm_shot_threat": ptv,
            "eh_std_iCF/60": icf,
            "rapm_hybrid_xGF/60": float(tail[0]),
            "a3z_chances_per_60": float(tail[1]),
            "a3z_carries_with_chances_per_60": float(tail[2]),
            "a3z_shots_per_60": float(tail[3]),
        }
        _record(weights, f"draw_{i}")

    if best_w is None:
        best_w = dict(DEFAULT_CURRENT)
    grid = pd.DataFrame(rows).sort_values("obj", ascending=False, na_position="last")
    return best_w, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "data/pcv/player_season_feature_table_all_in_one.csv")
    ap.add_argument("--scores", type=Path, default=ROOT / "data/pcv/player_season_all_in_one_8cat_scores_tuned.csv")
    ap.add_argument("--out-json", type=Path, default=ROOT / "data/pcv/shot_creation_component_weights_best.json")
    ap.add_argument("--out-grid", type=Path, default=ROOT / "data/pcv/shot_creation_component_weight_grid.csv")
    ap.add_argument("--n-samples", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-seasons", type=str, default="20182019,20192020,20202021,20212022,20222023,20232024")
    ap.add_argument("--rank-season", type=int, default=20252026)
    ap.add_argument("--threat-weight", type=float, default=0.50)
    ap.add_argument("--process-weight", type=float, default=0.30)
    ap.add_argument("--stickiness-weight", type=float, default=0.20)
    ap.add_argument("--rank-penalty-weight", type=float, default=0.35)
    ap.add_argument("--ixg-band", type=str, default=",".join(str(x) for x in IXG_BAND))
    ap.add_argument("--ptv-band", type=str, default=",".join(str(x) for x in PTV_BAND))
    ap.add_argument("--icf-band", type=str, default=",".join(str(x) for x in ICF_BAND))
    args = ap.parse_args()
    global ACTIVE_IXG_BAND, ACTIVE_PTV_BAND, ACTIVE_ICF_BAND
    ACTIVE_IXG_BAND = tuple(float(x.strip()) for x in str(args.ixg_band).split(",") if x.strip())  # type: ignore[assignment]
    ACTIVE_PTV_BAND = tuple(float(x.strip()) for x in str(args.ptv_band).split(",") if x.strip())  # type: ignore[assignment]
    ACTIVE_ICF_BAND = tuple(float(x.strip()) for x in str(args.icf_band).split(",") if x.strip())  # type: ignore[assignment]
    for label, band in (("ixg", ACTIVE_IXG_BAND), ("ptv", ACTIVE_PTV_BAND), ("icf", ACTIVE_ICF_BAND)):
        if len(band) != 2 or band[0] > band[1] or band[0] < 0.0 or band[1] > 1.0:
            raise SystemExit(f"--{label}-band must be two comma-separated values in [0,1], got {band}")

    need = [
        "player_id",
        "season_id",
        "position_group",
        "player_name",
        "eh_rapm_TOI",
        "a3z_toi_5v5_min",
        "yoy_xGF_per60",
        "yoy_xGA_per60",
        "yoy_onice_xg_share",
        "yoy_onice_goal_share",
        *COMPONENTS,
    ]
    feat = pd.read_csv(args.features, usecols=lambda c: c in need, low_memory=False)
    sc_need = ["player_id", "season_id", "cat_shot_creation_z__adj"]
    if args.scores.is_file():
        sc = pd.read_csv(args.scores, usecols=lambda c: c in sc_need, low_memory=False)
        feat = feat.merge(sc, on=["player_id", "season_id"], how="left")
    else:
        feat["cat_shot_creation_z__adj"] = np.nan

    d = _prepare(feat)
    eval_seasons = [int(x.strip()) for x in args.eval_seasons.split(",") if x.strip()]

    best_f, grid_f = _search_role(
        d,
        "F",
        n_samples=int(args.n_samples),
        seed=int(args.seed),
        eval_seasons=eval_seasons,
        rank_season=int(args.rank_season),
        threat_w=float(args.threat_weight),
        process_w=float(args.process_weight),
        stick_w=float(args.stickiness_weight),
        rank_penalty_w=float(args.rank_penalty_weight),
    )
    best_d, grid_d = _search_role(
        d,
        "D",
        n_samples=max(4000, int(args.n_samples) // 2),
        seed=int(args.seed) + 1,
        eval_seasons=eval_seasons,
        rank_season=int(args.rank_season),
        threat_w=float(args.threat_weight),
        process_w=float(args.process_weight),
        stick_w=float(args.stickiness_weight),
        rank_penalty_w=0.0,
    )
    grid = pd.concat([grid_f, grid_d], ignore_index=True)
    grid.to_csv(args.out_grid, index=False)

    out = {
        "F": best_f,
        "D": best_d,
        "default": best_f,
        "meta": {
            "tuned_for": "dangerous_offense_shot_threat",
            "components": COMPONENTS,
            "threat_components": THREAT_COMPONENTS,
            "objective_weights": {
                "threat": float(args.threat_weight),
                "process": float(args.process_weight),
                "stickiness": float(args.stickiness_weight),
            },
            "constraints": {
                "ixg_band": list(ACTIVE_IXG_BAND),
                "ptv_band": list(ACTIVE_PTV_BAND),
                "icf_band": list(ACTIVE_ICF_BAND),
                "hybrid_min": 0.08,
                "chances_min": 0.05,
                "non_shot_min": 0.04,
            },
            "eval_seasons": eval_seasons,
            "rank_season": int(args.rank_season),
            "n_samples": int(args.n_samples),
        },
    }
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Best F:", json.dumps(best_f, indent=2))
    print("Best D:", json.dumps(best_d, indent=2))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_grid}")


if __name__ == "__main__":
    main()
