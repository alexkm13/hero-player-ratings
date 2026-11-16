#!/usr/bin/env python3
"""
Walk-forward fit for PCV V2 (shot xG + 8s EPV + possession combiner).

  python3 fit_pcv_v2.py --max-games 80
  python3 fit_pcv_v2.py --out-dir data/pcv/validation/pcv_v2
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fit_pooled_xt import load_xtfit
from pcv_v2 import (
    DEFAULT_GBM_PARAMS,
    SHOT_EVENTS,
    build_epv_matrix,
    build_poss_matrix,
    build_shot_matrix,
    cap_shot_xg,
    enrich_game,
    geometry_matrix,
    is_regular_season,
    possession_feature_table,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_XT = ROOT / "data" / "models" / "xt_pooled_10x8.npz"
DEFAULT_OUT = ROOT / "data" / "pcv" / "validation" / "pcv_v2"


def _clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    y = np.asarray(y, dtype=float)
    p = _clip_p(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    tot = 0.0
    n = 0
    for b in range(n_bins):
        m = idx == b
        if int(m.sum()) < 1:
            continue
        tot += float(m.sum()) * abs(float(y[m].mean()) - float(p[m].mean()))
        n += int(m.sum())
    return float(tot / n) if n else float("nan")


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = _clip_p(p)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "mean_p": float(p.mean()) if len(p) else float("nan"),
        "log_loss": float("nan"),
        "brier": float("nan"),
        "auc": float("nan"),
        "ece": float("nan"),
    }
    if len(y) < 20 or y.min() == y.max():
        return out
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    out["ece"] = _ece(y, p)
    return out


def fit_logit(X: np.ndarray, y: np.ndarray) -> Pipeline:
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    C=1.0,
                    solver="lbfgs",
                    random_state=28,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def fit_gbm(X: np.ndarray, y: np.ndarray, cat_mask: np.ndarray) -> CalibratedClassifierCV:
    cat_idx = np.flatnonzero(np.asarray(cat_mask, dtype=bool))
    est = HistGradientBoostingClassifier(
        categorical_features=cat_idx.tolist() if len(cat_idx) else None,
        **DEFAULT_GBM_PARAMS,
    )
    cal = CalibratedClassifierCV(est, method="sigmoid", cv=3)
    cal.fit(X, y)
    return cal


def predict_p(model, X: np.ndarray) -> np.ndarray:
    return _clip_p(model.predict_proba(X)[:, 1])


def nhl_id_from_path(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def load_events(cache: Path, *, max_games: int = 0) -> pd.DataFrame:
    files = sorted(cache.glob("pbp_*.parquet"))
    by_season: dict[str, list[Path]] = {}
    for f in files:
        gid = nhl_id_from_path(f)
        if not is_regular_season(gid):
            continue
        by_season.setdefault(str(gid)[:4], []).append(f)
    chosen: list[Path] = []
    if max_games and by_season:
        per = max(1, int(np.ceil(max_games / len(by_season))))
        for _s, fl in sorted(by_season.items()):
            chosen.extend(fl[:per])
        chosen = chosen[:max_games]
    else:
        chosen = [f for fl in by_season.values() for f in fl]
    parts: list[pd.DataFrame] = []
    kept = 0
    t0 = time.time()
    for i, f in enumerate(chosen, start=1):
        gid = nhl_id_from_path(f)
        pbp = pd.read_parquet(f)
        ev = enrich_game(pbp, gid)
        parts.append(ev)
        kept += 1
        if kept % 200 == 0 or kept == len(chosen):
            print(f"  loaded {kept}/{len(chosen)} games {time.time()-t0:.0f}s", flush=True)
        if max_games and kept >= max_games:
            break
    if not parts:
        raise SystemExit("No regular-season PBP loaded")
    ev = pd.concat(parts, ignore_index=True)
    print(f"Events {len(ev):,} from {kept} games in {time.time()-t0:.1f}s")
    return ev


def add_grid_xg(events: pd.DataFrame, xt_path: Path) -> pd.DataFrame:
    d = events.copy()
    d["grid_xg"] = np.nan
    if not xt_path.is_file():
        return d
    xt = load_xtfit(xt_path)
    m = d["event"].isin(SHOT_EVENTS) & d["x_atk"].notna()
    d.loc[m, "grid_xg"] = xt.xG_nhl_series(d.loc[m, "x_atk"], d.loc[m, "y_atk"])
    return d


def fold_table(seasons: list[int], min_train: int = 1) -> list[tuple[list[int], int]]:
    seasons = sorted(seasons)
    folds = []
    for i, test in enumerate(seasons):
        train = seasons[:i]
        if len(train) < min_train:
            continue
        folds.append((train, test))
    return folds


def slice_metrics(shot_df: pd.DataFrame, pcol: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    y = shot_df["is_goal"].to_numpy(int)
    p = shot_df[pcol].to_numpy(float)
    out["all"] = classification_metrics(y, p)
    for name, mask in [
        ("rebound", shot_df["rebound"] == 1),
        ("rush", shot_df["rush"] == 1),
        ("close", shot_df["shot_distance"] <= 20),
        ("even", shot_df["skater_diff"].fillna(0) == 0),
        ("empty_net", shot_df["opp_empty_net"] == 1),
    ]:
        m = mask.to_numpy()
        if int(m.sum()) >= 40:
            out[name] = classification_metrics(y[m], p[m])
    return out


def team_game_score(poss: pd.DataFrame, pcol: str) -> dict[str, float]:
    g = poss.groupby(["NHL_game_id", "poss_team"], sort=False).agg(
        pcv=(pcol, "sum"),
        goals=("scored", "sum"),
    )
    if g.empty:
        return {"n": 0, "mae": float("nan"), "corr": float("nan"), "ratio": float("nan")}
    mae = float(np.mean(np.abs(g["pcv"] - g["goals"])))
    if float(g["pcv"].std()) <= 0 or float(g["goals"].std()) <= 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(g["pcv"], g["goals"])[0, 1])
    ratio = float(g["pcv"].sum() / g["goals"].sum()) if float(g["goals"].sum()) else float("nan")
    return {"n": int(len(g)), "mae": mae, "corr": corr, "ratio": ratio}


def _fit_production_pack(events: pd.DataFrame) -> dict:
    print("Fitting production pack on all loaded seasons", flush=True)
    shot = build_shot_matrix(events)
    epv = build_epv_matrix(events)
    shot_model = fit_gbm(shot.X, shot.y, shot.cat_mask)
    epv_model = fit_gbm(epv.X, epv.y, epv.cat_mask)
    scored = events.copy()
    scored["xg"] = np.nan
    scored["epv"] = np.nan
    scored.loc[shot.index, "xg"] = cap_shot_xg(
        predict_p(shot_model, shot.X),
        pd.to_numeric(events.loc[shot.index, "opp_empty_net"], errors="coerce").fillna(0.0).to_numpy(float),
    )
    scored.loc[epv.index, "epv"] = predict_p(epv_model, epv.X)
    poss = possession_feature_table(scored)
    pm = build_poss_matrix(poss)
    poss_model = fit_logit(pm.X, pm.y)
    return {
        "shot_model": shot_model,
        "shot_mappings": shot.mappings,
        "epv_model": epv_model,
        "epv_mappings": epv.mappings,
        "poss_model": poss_model,
        "constant_rate": float(shot.y.mean()),
        "train_seasons": sorted(int(s) for s in events["season_start"].dropna().unique()),
        "test_season": None,
        "production": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--xt", type=Path, default=DEFAULT_XT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--min-train-seasons", type=int, default=1)
    ap.add_argument("--production-only", action="store_true", help="Skip walk-forward; refit pack on all loaded games")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.cache_dir, max_games=int(args.max_games or 0))
    events = add_grid_xg(events, args.xt)
    if args.production_only:
        pack = _fit_production_pack(events)
        model_path = ROOT / "data" / "models" / "pcv_v2.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pack, model_path)
        print(f"Wrote production pack {model_path}")
        return
    seasons = sorted(int(s) for s in events["season_start"].dropna().unique())
    folds = fold_table(seasons, min_train=int(args.min_train_seasons))
    print("seasons", seasons, "folds", folds)

    fold_rows: list[dict] = []
    shot_pred_parts: list[pd.DataFrame] = []
    poss_pred_parts: list[pd.DataFrame] = []
    last_pack: dict = {}

    for train_seasons, test_season in folds:
        print(f"\n=== fold train={train_seasons} test={test_season} ===", flush=True)
        tr = events["season_start"].isin(train_seasons)
        te = events["season_start"] == test_season
        train_ev = events.loc[tr]
        test_ev = events.loc[te]

        shot_tr = build_shot_matrix(train_ev)
        shot_te = build_shot_matrix(test_ev, mappings=shot_tr.mappings)
        geo_tr = geometry_matrix(train_ev)
        geo_te = geometry_matrix(test_ev)

        constant_p = float(shot_tr.y.mean())
        p_const = np.full(len(shot_te.y), constant_p)
        logit_geo = fit_logit(geo_tr.X, geo_tr.y)
        p_geo = predict_p(logit_geo, geo_te.X)
        logit_full = fit_logit(shot_tr.X, shot_tr.y)
        p_logit = predict_p(logit_full, shot_te.X)
        gbm = fit_gbm(shot_tr.X, shot_tr.y, shot_tr.cat_mask)
        p_gbm = predict_p(gbm, shot_te.X)
        p_gbm = cap_shot_xg(
            p_gbm,
            pd.to_numeric(test_ev.loc[shot_te.index, "opp_empty_net"], errors="coerce").fillna(0.0).to_numpy(float),
        )
        grid = pd.to_numeric(test_ev.loc[shot_te.index, "grid_xg"], errors="coerce").to_numpy(float)
        grid = np.where(np.isfinite(grid), grid, constant_p)

        m_const = classification_metrics(shot_te.y, p_const)
        m_geo = classification_metrics(shot_te.y, p_geo)
        m_grid = classification_metrics(shot_te.y, grid)
        m_logit = classification_metrics(shot_te.y, p_logit)
        m_gbm = classification_metrics(shot_te.y, p_gbm)
        print(
            f"  shot logloss  const={m_const['log_loss']:.4f}  geo={m_geo['log_loss']:.4f}  "
            f"grid={m_grid['log_loss']:.4f}  logit={m_logit['log_loss']:.4f}  gbm={m_gbm['log_loss']:.4f}"
        )

        epv_tr = build_epv_matrix(train_ev)
        epv_te = build_epv_matrix(test_ev, mappings=epv_tr.mappings)
        epv_logit = fit_logit(epv_tr.X, epv_tr.y)
        epv_gbm = fit_gbm(epv_tr.X, epv_tr.y, epv_tr.cat_mask)
        p_epv_te = predict_p(epv_gbm, epv_te.X)
        p_epv_logit = predict_p(epv_logit, epv_te.X)
        m_epv = classification_metrics(epv_te.y, p_epv_te)
        m_epv_logit = classification_metrics(epv_te.y, p_epv_logit)
        print(f"  epv  logit={m_epv_logit['log_loss']:.4f}  gbm={m_epv['log_loss']:.4f}  auc={m_epv['auc']:.3f}")

        # Score train+test events with train-fit models for possession features.
        shot_tr_p = cap_shot_xg(
            predict_p(gbm, shot_tr.X),
            pd.to_numeric(train_ev.loc[shot_tr.index, "opp_empty_net"], errors="coerce").fillna(0.0).to_numpy(float),
        )
        epv_tr_p = predict_p(epv_gbm, epv_tr.X)
        train_scored = train_ev.copy()
        test_scored = test_ev.copy()
        train_scored["xg"] = np.nan
        test_scored["xg"] = np.nan
        train_scored.loc[shot_tr.index, "xg"] = shot_tr_p
        test_scored.loc[shot_te.index, "xg"] = p_gbm
        train_scored["epv"] = np.nan
        test_scored["epv"] = np.nan
        train_scored.loc[epv_tr.index, "epv"] = epv_tr_p
        test_scored.loc[epv_te.index, "epv"] = p_epv_te

        poss_tr = possession_feature_table(train_scored)
        poss_te = possession_feature_table(test_scored)
        pm_tr = build_poss_matrix(poss_tr)
        pm_te = build_poss_matrix(poss_te)
        poss_logit = fit_logit(pm_tr.X, pm_tr.y)
        p_poss = predict_p(poss_logit, pm_te.X)
        p_xg_surv = pd.to_numeric(poss_te["xg_surv"], errors="coerce").fillna(0.0).to_numpy(float)
        p_xg_surv = _clip_p(p_xg_surv)
        noshot = poss_te["n_shots"].to_numpy() <= 0
        p_combo = p_poss.copy()
        # Keep combiner leak-free; shotless mass comes only from EPV, not from goal-event peaks.
        p_combo[noshot] = _clip_p(
            0.25
            * pd.to_numeric(poss_te.loc[noshot, "epv_peak_preshot"], errors="coerce").fillna(0.0).to_numpy(float)
        )
        m_poss = classification_metrics(pm_te.y, p_poss)
        m_surv = classification_metrics(pm_te.y, p_xg_surv)
        m_combo = classification_metrics(pm_te.y, p_combo)
        print(f"  poss logloss  surv={m_surv['log_loss']:.4f}  logit={m_poss['log_loss']:.4f}  combo={m_combo['log_loss']:.4f}")

        poss_te = poss_te.copy()
        poss_te["pcv"] = p_poss
        poss_te["pcv_surv"] = p_xg_surv
        poss_te["pcv_combo"] = p_combo
        tg = team_game_score(poss_te, "pcv")
        tg_surv = team_game_score(poss_te, "pcv_surv")
        print(f"  team-game MAE pcv={tg['mae']:.3f} surv={tg_surv['mae']:.3f}  corr={tg['corr']:.3f}  sum(pcv)/goals={tg['ratio']:.3f}")

        shot_view = test_ev.loc[shot_te.index, ["NHL_game_id", "season_start", "is_goal", "rebound", "rush", "shot_distance", "skater_diff", "opp_empty_net"]].copy()
        shot_view["p_const"] = p_const
        shot_view["p_geo"] = p_geo
        shot_view["p_grid"] = grid
        shot_view["p_logit"] = p_logit
        shot_view["p_gbm"] = p_gbm
        shot_pred_parts.append(shot_view)
        slices = slice_metrics(shot_view, "p_gbm")

        row = {
            "train_seasons": train_seasons,
            "test_season": test_season,
            "shot_constant": m_const,
            "shot_geo": m_geo,
            "shot_grid": m_grid,
            "shot_logit": m_logit,
            "shot_gbm": m_gbm,
            "epv_logit": m_epv_logit,
            "epv_gbm": m_epv,
            "poss_surv": m_surv,
            "poss_logit": m_poss,
            "poss_combo": m_combo,
            "team_game_pcv": tg,
            "team_game_surv": tg_surv,
            "shot_slices_gbm": slices,
        }
        fold_rows.append(row)
        poss_pred_parts.append(poss_te)
        last_pack = {
            "shot_model": gbm,
            "shot_mappings": shot_tr.mappings,
            "epv_model": epv_gbm,
            "epv_mappings": epv_tr.mappings,
            "poss_model": poss_logit,
            "constant_rate": constant_p,
            "train_seasons": train_seasons,
            "test_season": test_season,
        }

    def _mean_metric(key: str, nested: str) -> float:
        vals = [r[key][nested] for r in fold_rows if np.isfinite(r[key][nested])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_folds": len(fold_rows),
        "mean_shot_logloss_constant": _mean_metric("shot_constant", "log_loss"),
        "mean_shot_logloss_geo": _mean_metric("shot_geo", "log_loss"),
        "mean_shot_logloss_grid": _mean_metric("shot_grid", "log_loss"),
        "mean_shot_logloss_logit": _mean_metric("shot_logit", "log_loss"),
        "mean_shot_logloss_gbm": _mean_metric("shot_gbm", "log_loss"),
        "mean_shot_brier_gbm": _mean_metric("shot_gbm", "brier"),
        "mean_shot_ece_gbm": _mean_metric("shot_gbm", "ece"),
        "mean_shot_auc_gbm": _mean_metric("shot_gbm", "auc"),
        "mean_epv_logloss_gbm": _mean_metric("epv_gbm", "log_loss"),
        "mean_poss_logloss_surv": _mean_metric("poss_surv", "log_loss"),
        "mean_poss_logloss_logit": _mean_metric("poss_logit", "log_loss"),
        "gbm_beats_grid_folds": int(
            sum(1 for r in fold_rows if r["shot_gbm"]["log_loss"] < r["shot_grid"]["log_loss"])
        ),
        "gbm_beats_geo_folds": int(
            sum(1 for r in fold_rows if r["shot_gbm"]["log_loss"] < r["shot_geo"]["log_loss"])
        ),
    }
    beats_grid = summary["mean_shot_logloss_gbm"] < summary["mean_shot_logloss_grid"]
    beats_geo = summary["mean_shot_logloss_gbm"] < summary["mean_shot_logloss_geo"]
    ece_ok = summary["mean_shot_ece_gbm"] < 0.02
    majority = summary["gbm_beats_grid_folds"] >= max(1, int(np.ceil(len(fold_rows) / 2)))
    decision = "keep" if (beats_grid and beats_geo and ece_ok and majority) else "follow-up"
    summary["decision"] = decision
    print("\nSUMMARY", json.dumps(summary, indent=2))

    model_path = ROOT / "data" / "models" / "pcv_v2.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if last_pack:
        joblib.dump(last_pack, model_path)
        print(f"Wrote walk-forward pack {model_path}")
    if decision == "keep":
        prod = _fit_production_pack(events)
        joblib.dump(prod, model_path)
        print(f"Wrote production pack {model_path}")

    shot_preds = pd.concat(shot_pred_parts, ignore_index=True) if shot_pred_parts else pd.DataFrame()
    poss_preds = pd.concat(poss_pred_parts, ignore_index=True) if poss_pred_parts else pd.DataFrame()
    shot_preds.to_parquet(out_dir / "shot_heldout_predictions.parquet", index=False)
    poss_preds.to_parquet(out_dir / "possession_heldout_predictions.parquet", index=False)
    (out_dir / "fold_metrics.json").write_text(json.dumps(fold_rows, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote metrics under {out_dir}")


if __name__ == "__main__":
    main()
