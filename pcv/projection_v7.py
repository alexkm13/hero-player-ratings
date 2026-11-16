#!/usr/bin/env python3
"""Apply the promoted V7 next-card policy without replacing the V6 archive."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from projection_v5 import DEFAULT_FEATURES, DEFAULT_SCORES, prepare_panel
from projection_v6 import predict_selected


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "data/pcv/predictive/projection_v7_bundle.joblib"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, low_memory=False)


def apply_saved_bundle(
    rows: pd.DataFrame,
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    if not Path(bundle_path).is_file():
        return rows
    bundle = joblib.load(bundle_path)
    source = rows.copy()
    if "eval_season_id" not in source.columns:
        source["eval_season_id"] = int(pd.to_numeric(_read(score_path)["season_id"], errors="coerce").max())
    prepared = prepare_panel(source, feature_path=feature_path, score_path=score_path, attach_targets=False)
    stable = predict_selected(bundle["stable"], prepared)
    season = (
        predict_selected(bundle["season"], prepared)
        if bundle.get("season") is not None else pd.Series(np.nan, index=rows.index)
    )
    anchor = pd.to_numeric(source.get("overall_player_rating_display_pct"), errors="coerce")
    stable_delta = stable - anchor

    snapshot = bundle.get("current_predictions") or {}
    snapshot_season = bundle.get("current_prediction_season")
    if snapshot and snapshot_season is not None and "player_id" in source.columns:
        sid = pd.to_numeric(source["eval_season_id"], errors="coerce")
        pid = pd.to_numeric(source["player_id"], errors="coerce")
        use = sid.eq(int(snapshot_season))
        delta_map = {
            int(k): float(v["stable_delta"])
            for k, v in snapshot.items() if v.get("stable_delta") is not None
        }
        season_map = {
            int(k): float(v["season"])
            for k, v in snapshot.items() if v.get("season") is not None
        }
        delta_override = pid.map(delta_map)
        translated = (anchor + delta_override).clip(0.0, 100.0)
        stable = translated.where(use & translated.notna(), stable)
        stable_delta = delta_override.where(use & delta_override.notna(), stable_delta)
        season_override = pid.map(season_map)
        season = season_override.where(use & season_override.notna(), season)

    out = rows.copy()
    out["projected_next_card_overall_pct_v7"] = stable.to_numpy(float)
    out["projected_next_card_overall_delta_v7"] = stable_delta.to_numpy(float)
    out["projected_next_season_performance_pct_v7"] = season.to_numpy(float)
    out["projected_next_year_pct_pre_v7"] = pd.to_numeric(out.get("projected_next_year_pct"), errors="coerce")
    promoted = str(bundle.get("promotion_decision", "")).lower() == "promote"
    applicable = promoted & stable.notna()
    out["projected_next_year_v7_applied"] = applicable.astype(int)
    out["projected_next_year_pct"] = stable.where(applicable, out["projected_next_year_pct_pre_v7"])
    if promoted:
        version = str(bundle.get("version", "V7 stable-overall"))
        out["projection_model_version"] = np.where(applicable, version, out.get("projection_model_version", "V6"))
    return out
