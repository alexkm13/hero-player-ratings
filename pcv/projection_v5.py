#!/usr/bin/env python3
"""Projection V5: regime-aware, interpretable next-season player forecasts.

V5 deliberately separates:

* a long-history macro backbone (RAPM, on-ice process, age, role, context);
* a tracking/category layer that is used only when comparable evidence exists;
* nine category heads whose features are category-isolated;
* a direct overall head and an independent next-season process head;
* talent and next-season availability.

The module is shared by the walk-forward tuner and card builder.  Targets are
fit in latent normal-score space and converted to role-relative percentiles at
prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "data/pcv/player_season_feature_table_all_in_one.parquet"
DEFAULT_SCORES = ROOT / "data/pcv/player_season_all_in_one_8cat_scores_tuned.parquet"
DEFAULT_BUNDLE = ROOT / "data/pcv/predictive/projection_v5_bundle.joblib"

OVERALL_TARGET = "target_next_overall_8cat_tuned_pct"
SMOOTH_TARGET = "target_blended_future_pct"
INDEPENDENT_TARGET = "target_independent_process_pct"
AVAILABILITY_TARGET = "target_available_next_season"

CATEGORY_LABELS = (
    "Shot Creation",
    "Transition Offense",
    "Shot Defense",
    "Transition Defense",
    "Finishing",
    "Playmaking",
    "Forecheck",
    "Physicality",
    "Skating",
)

CATEGORY_TARGETS = {
    label: f"target_next_{label.lower().replace(' ', '_')}_pct" for label in CATEGORY_LABELS
}

RAW_MACRO_COLS = (
    "eh_rapm_xG±/60",
    "eh_rapm_xGF/60",
    "eh_rapm_xGA/60",
    "eh_rapm_C±/60",
    "eh_rapm_G±/60",
    "rapm_hybrid_xGF/60",
    "rapm_hybrid_xGA/60",
    "rapm_hybrid_xG±/60",
    "ptv_roll3_pre_shot_pressure",
    "ptv_roll3_possession_offense",
    "ptv_roll3_shot_threat",
    "ptv_roll3_possession_defense",
    "ptv_roll3_threat_suppression",
    "yoy_raw_pcv_per60",
    "yoy_raw_shot_pcv_per60",
    "yoy_raw_preshot_pcv_per60",
    "yoy_onice_team_pcv_share",
    "yoy_onice_xg_share",
    "yoy_xGF_per60",
    "yoy_xGA_per60",
    "eh_std_ixG/60",
    "eh_std_iCF/60",
    "eh_std_iHF/60",
    "eh_std_iBLK/60",
)

MACRO_EXACT = {
    "total_toi_5v5_min_3yr",
    "total_toi_display_min_3yr",
    "current_season_toi_display_min",
    "n_seasons_present",
    "age_years",
    "age_centered",
    "age_centered_sq",
    "injury_recency_flag",
    "injury_latest_toi_ratio_vs_prior",
    "injury_latest_recency_scale",
    "team_override_applied",
    "overall_current_impact_pct",
    "impact_anchor_z_3yr",
    "rapm_alignment_penalty_pct",
    "rapm_anchor_used",
    "rapm_gap_pct",
    "rating_shrink_weight",
    "overall_latest_weight_share_3yr",
    "overall_regression_pattern_flag",
    "overall_regression_span_drop",
    "impact_regression_last_drop",
    "overall_regression_decline_steps",
    "is_defenseman",
    "is_forward",
    "age_young_upside",
    "age_decline_risk",
    "rookie_or_one_season",
    "second_year_or_less",
    "rookie_low_sample_risk",
    "young_low_toi_signal",
    "durability_signal",
    "injury_risk_signal",
}

MACRO_PREFIXES = ("ctx_", "srcraw_", "hist_overall_", "hist_toi_")

FULL_EXTRA_EXACT = {
    "source_overall_player_rating_pct",
    "overall_8cat_z_weighted_3yr_raw",
    "overall_8cat_tuned_pct_weighted_3yr",
    "overall_current_season_pct",
    "overall_prior_window_pct",
    "overall_model_3yr_pct",
    "overall_player_rating_display_pct",
    "overall_stable_talent_pct",
    "category_mean_pct",
    "category_peak_pct",
    "category_floor_pct",
    "category_spread_pct",
    "offense_process_stack_pct",
    "defense_process_stack_pct",
    "two_way_process_stack_pct",
    "low_trust_stack_pct",
}


@dataclass(frozen=True)
class V5Config:
    macro_k: int = 40
    full_k: int = 60
    half_life_seasons: float = 0.75
    tracking_min_train_coverage: float = 0.55

    @property
    def key(self) -> str:
        return f"m{self.macro_k}_f{self.full_k}_h{self.half_life_seasons:g}"


DEFAULT_CONFIG = V5Config()


def season_shift(season_id: int, delta: int) -> int:
    start = int(season_id) // 10000
    return (start + int(delta)) * 10000 + (start + int(delta) + 1)


def percentile_to_latent(values: pd.Series) -> pd.Series:
    pct = pd.to_numeric(values, errors="coerce").clip(lower=0.5, upper=99.5) / 100.0
    return pd.Series(norm.ppf(pct), index=values.index, dtype=float)


def rank_within_role(values: pd.Series | np.ndarray, roles: pd.Series) -> pd.Series:
    value = pd.Series(np.asarray(values, dtype=float), index=roles.index)
    role = roles.astype(str).str.upper().str.strip().replace({"": "UNK"})
    out = value.groupby(role).rank(method="average", pct=True) * 100.0
    return out.clip(lower=0.0, upper=100.0)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, low_memory=False)


def _source_rollups(panel: pd.DataFrame, feature_path: Path = DEFAULT_FEATURES) -> pd.DataFrame:
    """Attach current and 3Y weighted macro process fields to source-season rows."""
    feature = _read(feature_path)
    keep = ["player_id", "season_id", *[c for c in RAW_MACRO_COLS if c in feature.columns]]
    feature = feature[keep].drop_duplicates(["player_id", "season_id"], keep="last").copy()
    feature["player_id"] = pd.to_numeric(feature["player_id"], errors="coerce").astype("Int64")
    feature["season_id"] = pd.to_numeric(feature["season_id"], errors="coerce").astype("Int64")
    out = panel.copy()
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["eval_season_id"] = pd.to_numeric(out["eval_season_id"], errors="coerce").astype("Int64")
    raw_cols = [c for c in keep if c not in {"player_id", "season_id"}]
    lag_values: dict[str, list[pd.Series]] = {c: [] for c in raw_cols}
    for lag, weight in ((0, 0.60), (1, 0.30), (2, 0.10)):
        key = out[["player_id", "eval_season_id"]].copy()
        key["season_id"] = key["eval_season_id"].map(
            lambda x: season_shift(int(x), -lag) if pd.notna(x) else pd.NA
        ).astype("Int64")
        joined = key.merge(feature, on=["player_id", "season_id"], how="left", sort=False)
        for col in raw_cols:
            vals = pd.to_numeric(joined[col], errors="coerce")
            lag_values[col].append(vals.set_axis(out.index))
            if lag == 0:
                out[f"srcraw_{col}_latest"] = vals.to_numpy(float)
    weights = (0.60, 0.30, 0.10)
    for col, vals in lag_values.items():
        num = sum(w * s.fillna(0.0) for w, s in zip(weights, vals))
        den = sum(w * s.notna().astype(float) for w, s in zip(weights, vals))
        out[f"srcraw_{col}_w3"] = num / den.replace(0.0, np.nan)
    return out


def _coverage_features(
    panel: pd.DataFrame,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    """Attach explicit tracking-regime flags; never encode missing tracking as zero."""
    out = panel.copy()
    feature = _read(feature_path)
    scores = _read(score_path)
    fcols = [c for c in ["player_id", "season_id", "has_a3z_season", "has_eh_rapm", "has_yoy_outcomes"] if c in feature.columns]
    scols = [c for c in ["player_id", "season_id", "cat_skating_n_features"] if c in scores.columns]
    cov = feature[fcols].drop_duplicates(["player_id", "season_id"], keep="last")
    if len(scols) > 2:
        cov = cov.merge(scores[scols].drop_duplicates(["player_id", "season_id"], keep="last"), on=["player_id", "season_id"], how="outer")
    cov = cov.rename(columns={"season_id": "eval_season_id"})
    for frame in (out, cov):
        frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce").astype("Int64")
        frame["eval_season_id"] = pd.to_numeric(frame["eval_season_id"], errors="coerce").astype("Int64")
    out = out.drop(columns=[c for c in ["has_a3z_season", "has_eh_rapm", "has_yoy_outcomes", "cat_skating_n_features"] if c in out.columns])
    out = out.merge(cov, on=["player_id", "eval_season_id"], how="left", sort=False)
    a3z = pd.to_numeric(out.get("has_a3z_season"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    skating = pd.to_numeric(out.get("cat_skating_n_features"), errors="coerce").fillna(0.0).gt(0.0).astype(float)
    rapm = pd.to_numeric(out.get("has_eh_rapm"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["tracking_a3z_available"] = a3z
    out["tracking_skating_available"] = skating
    out["macro_rapm_available"] = rapm
    # A3Z owns most tracking-derived traits; skating is separately sourced.
    out["tracking_coverage_score"] = (0.82 * a3z + 0.18 * skating).clip(0.0, 1.0)
    return out


def _rating_history_features(
    panel: pd.DataFrame,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    """Attach leakage-safe season t..t-3 rating and category trajectories.

    The card aggregate contains a current value and a prior-window average.  Two
    players can therefore look identical despite very different paths into that
    average.  These fields retain the individual source seasons so the forecast
    can learn momentum, mean reversion, and volatility without ever seeing t+1.
    """
    out = panel.copy()
    scores = _read(score_path).copy()
    scores["player_id"] = pd.to_numeric(scores["player_id"], errors="coerce").astype("Int64")
    scores["season_id"] = pd.to_numeric(scores["season_id"], errors="coerce").astype("Int64")
    scores["position_group"] = scores.get("position_group", "").astype(str).str.upper().str.strip()

    overall_col = "overall_8cat_tuned_pct" if "overall_8cat_tuned_pct" in scores else "overall_8cat_pct"
    toi_col = "eh_rapm_TOI" if "eh_rapm_TOI" in scores else "a3z_toi_5v5_min"
    take = ["player_id", "season_id", overall_col, toi_col]
    category_cols: dict[str, str] = {}
    for label in CATEGORY_LABELS:
        stem = label.lower().replace(" ", "_")
        col = f"cat_{stem}_z__adj"
        if col not in scores:
            col = f"cat_{stem}_z"
        if col in scores:
            pct_col = f"_hist_{stem}_pct"
            raw = pd.to_numeric(scores[col], errors="coerce")
            scores[pct_col] = raw.groupby(
                [scores["season_id"], scores["position_group"]]
            ).rank(method="average", pct=True) * 100.0
            category_cols[label] = pct_col
            take.append(pct_col)
    hist = scores[take].drop_duplicates(["player_id", "season_id"], keep="last")

    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")
    out["eval_season_id"] = pd.to_numeric(out["eval_season_id"], errors="coerce").astype("Int64")
    for lag in range(4):
        joined = out[["player_id", "eval_season_id"]].copy()
        joined["season_id"] = joined["eval_season_id"].map(
            lambda x: season_shift(int(x), -lag) if pd.notna(x) else pd.NA
        ).astype("Int64")
        joined = joined.merge(hist, on=["player_id", "season_id"], how="left", sort=False)
        out[f"hist_overall_pct_lag{lag}"] = pd.to_numeric(joined[overall_col], errors="coerce").to_numpy(float)
        out[f"hist_toi_lag{lag}"] = pd.to_numeric(joined[toi_col], errors="coerce").to_numpy(float)
        for label, col in category_cols.items():
            stem = label.lower().replace(" ", "_")
            out[f"hist_{stem}_pct_lag{lag}"] = pd.to_numeric(joined[col], errors="coerce").to_numpy(float)

    def add_trajectory(prefix: str) -> None:
        cols = [f"{prefix}_lag{lag}" for lag in range(4)]
        values = out[cols].apply(pd.to_numeric, errors="coerce")
        out[f"{prefix}_mean4"] = values.mean(axis=1, skipna=True)
        out[f"{prefix}_std4"] = values.std(axis=1, skipna=True)
        out[f"{prefix}_delta_1y"] = values[cols[0]] - values[cols[1]]
        out[f"{prefix}_delta_2y"] = values[cols[0]] - values[cols[2]]
        # Positive means the recent season is better than the older history.
        older = values[cols[1:]].mean(axis=1, skipna=True)
        out[f"{prefix}_recent_vs_prior"] = values[cols[0]] - older

    add_trajectory("hist_overall_pct")
    add_trajectory("hist_toi")
    for label in category_cols:
        add_trajectory(f"hist_{label.lower().replace(' ', '_')}_pct")
    return out


def _attach_future_targets(
    panel: pd.DataFrame,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    """Attach independent process and availability labels from season t+1."""
    out = panel.copy()
    feature = _read(feature_path)
    scores = _read(score_path)
    target_cols = [
        "player_id",
        "season_id",
        "position_group",
        "eh_rapm_xG±/60",
        "yoy_raw_pcv_per60",
        "yoy_onice_team_pcv_share",
        "yoy_onice_xg_share",
        "yoy_xGF_per60",
        "yoy_xGA_per60",
    ]
    target_cols = [c for c in target_cols if c in feature.columns]
    nxt = feature[target_cols].drop_duplicates(["player_id", "season_id"], keep="last").copy()
    toi_col = "eh_rapm_TOI" if "eh_rapm_TOI" in scores.columns else "a3z_toi_5v5_min"
    toi = scores[["player_id", "season_id", toi_col]].drop_duplicates(["player_id", "season_id"], keep="last")
    nxt = nxt.merge(toi, on=["player_id", "season_id"], how="outer")
    nxt["position_group"] = nxt.get("position_group", pd.Series(index=nxt.index, dtype=object)).astype(str).str.upper().str.strip()
    if "position_group" not in target_cols:
        pos = scores[["player_id", "season_id", "position_group"]].drop_duplicates(["player_id", "season_id"], keep="last")
        nxt = nxt.drop(columns=["position_group"], errors="ignore").merge(pos, on=["player_id", "season_id"], how="left")
    parts: list[tuple[float, pd.Series]] = []
    specs = (
        ("eh_rapm_xG±/60", 0.35, 1.0),
        ("yoy_onice_xg_share", 0.25, 1.0),
        ("yoy_raw_pcv_per60", 0.15, 1.0),
        ("yoy_onice_team_pcv_share", 0.10, 1.0),
        ("yoy_xGF_per60", 0.075, 1.0),
        ("yoy_xGA_per60", 0.075, -1.0),
    )
    group = [pd.to_numeric(nxt["season_id"], errors="coerce"), nxt["position_group"].astype(str)]
    for col, weight, sign in specs:
        if col not in nxt.columns:
            continue
        raw = sign * pd.to_numeric(nxt[col], errors="coerce")
        pct = raw.groupby(group).rank(method="average", pct=True) * 100.0
        parts.append((weight, pct))
    num = sum(w * s.fillna(0.0) for w, s in parts)
    den = sum(w * s.notna().astype(float) for w, s in parts)
    nxt[INDEPENDENT_TARGET] = (num / den.replace(0.0, np.nan)).clip(0.0, 100.0)
    nxt[AVAILABILITY_TARGET] = pd.to_numeric(nxt[toi_col], errors="coerce").fillna(0.0).ge(200.0).astype(int)
    nxt["eval_season_id"] = pd.to_numeric(nxt["season_id"], errors="coerce").map(
        lambda x: season_shift(int(x), -1) if pd.notna(x) else pd.NA
    )
    keep = ["player_id", "eval_season_id", INDEPENDENT_TARGET, AVAILABILITY_TARGET]
    nxt = nxt[keep].drop_duplicates(["player_id", "eval_season_id"], keep="last")
    out = out.drop(columns=[INDEPENDENT_TARGET, AVAILABILITY_TARGET], errors="ignore")
    out = out.merge(nxt, on=["player_id", "eval_season_id"], how="left", sort=False)
    # A player absent from t+1 is a real non-availability observation when t+1 exists.
    latest_target_source = season_shift(int(scores["season_id"].max()), -1)
    known = pd.to_numeric(out["eval_season_id"], errors="coerce").le(latest_target_source)
    out.loc[known & out[AVAILABILITY_TARGET].isna(), AVAILABILITY_TARGET] = 0.0
    return out


def prepare_panel(
    panel: pd.DataFrame,
    *,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
    attach_targets: bool = True,
) -> pd.DataFrame:
    out = panel.copy().replace([np.inf, -np.inf], np.nan)
    if "eval_season_id" not in out.columns:
        out["eval_season_id"] = int(_read(score_path)["season_id"].max())
    out = _source_rollups(out, feature_path)
    out = _coverage_features(out, feature_path, score_path)
    if attach_targets:
        out = _attach_future_targets(out, feature_path, score_path)
    overall_source = SMOOTH_TARGET if SMOOTH_TARGET in out.columns else OVERALL_TARGET if OVERALL_TARGET in out.columns else None
    if overall_source is not None:
        out["target_overall_latent"] = percentile_to_latent(out[overall_source])
    if INDEPENDENT_TARGET in out.columns:
        out["target_independent_latent"] = percentile_to_latent(out[INDEPENDENT_TARGET])
    for label, col in CATEGORY_TARGETS.items():
        if col in out.columns:
            out[f"target_{label.lower().replace(' ', '_')}_latent"] = percentile_to_latent(out[col])
    return out


def macro_features(rows: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for col in rows.columns:
        if col in MACRO_EXACT or col.startswith(MACRO_PREFIXES):
            if pd.api.types.is_numeric_dtype(rows[col]) and pd.to_numeric(rows[col], errors="coerce").notna().sum() >= 80:
                out.append(col)
    for col in ("tracking_coverage_score", "tracking_a3z_available", "tracking_skating_available", "macro_rapm_available"):
        if col in rows.columns and col not in out:
            out.append(col)
    return out


def full_features(rows: pd.DataFrame, macro: Iterable[str]) -> list[str]:
    out = list(dict.fromkeys(macro))
    for col in rows.columns:
        category_col = any(
            col == f"{label}__pct"
            or col == f"{label}__rank_pct"
            or col.startswith(label.lower().replace(" ", "_") + "_reliab")
            or col.startswith("cat_" + label.lower().replace(" ", "_"))
            for label in CATEGORY_LABELS
        )
        reliability_col = col.endswith("_reliability") or col.endswith("_reliable_pct") or col.endswith("_unreliable_extreme")
        history_category_col = col.startswith("hist_") and not col.startswith(("hist_overall_", "hist_toi_"))
        if col in FULL_EXTRA_EXACT or category_col or reliability_col or history_category_col:
            if pd.api.types.is_numeric_dtype(rows[col]) and pd.to_numeric(rows[col], errors="coerce").notna().sum() >= 80:
                out.append(col)
    return list(dict.fromkeys(out))


def category_features(rows: pd.DataFrame, macro: Iterable[str], label: str) -> list[str]:
    stem = label.lower().replace(" ", "_")
    exact = {f"{label}__pct", f"{label}__rank_pct", f"cat_{stem}_z__adj__pct_w3"}
    prefixes = (f"{stem}_reliability", f"{stem}_reliable", f"{stem}_unreliable", f"cat_{stem}_", f"hist_{stem}_")
    own = [
        col for col in rows.columns
        if (col in exact or col.startswith(prefixes))
        and pd.api.types.is_numeric_dtype(rows[col])
        and pd.to_numeric(rows[col], errors="coerce").notna().sum() >= 80
    ]
    return list(dict.fromkeys([*macro, *own]))


def xframe(rows: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    cols = list(features)
    out = rows.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).clip(lower=-1e6, upper=1e6)


def recency_weights(rows: pd.DataFrame, prediction_season: int, half_life: float) -> np.ndarray:
    sid = pd.to_numeric(rows["eval_season_id"], errors="coerce").to_numpy(float)
    pred_start = int(prediction_season) // 10000
    starts = np.floor(sid / 10000.0)
    age = np.maximum(0.0, pred_start - starts - 1.0)
    return np.exp(-np.log(2.0) * age / max(float(half_life), 0.1))


def _regressor(seed: int = 42) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        depth=4,
        iterations=450,
        learning_rate=0.03,
        l2_leaf_reg=14.0,
        random_seed=seed,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )


def _select_features(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: np.ndarray,
    k: int,
    seed: int,
) -> list[str]:
    usable = [c for c in features if pd.to_numeric(train[c], errors="coerce").notna().sum() >= 40]
    if len(usable) <= int(k):
        return usable
    ranker = _regressor(seed)
    ranker.fit(xframe(train, usable), target, sample_weight=weights)
    ranked = sorted(zip(usable, ranker.get_feature_importance()), key=lambda item: (-item[1], item[0]))
    return [name for name, _ in ranked[: int(k)]]


def fit_layered_regressor(
    train: pd.DataFrame,
    *,
    target_col: str,
    prediction_season: int,
    config: V5Config,
    macro_candidates: list[str],
    full_candidates: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    target = pd.to_numeric(train[target_col], errors="coerce")
    fit = train.loc[target.notna()].copy()
    target = target.loc[fit.index]
    weights = recency_weights(fit, prediction_season, config.half_life_seasons)
    if "target_sample_weight" in fit.columns:
        weights *= pd.to_numeric(fit["target_sample_weight"], errors="coerce").fillna(1.0).clip(0.25, 2.5).to_numpy(float)
    macro_selected = _select_features(fit, macro_candidates, target, weights, config.macro_k, seed)
    macro_model = _regressor(seed)
    macro_model.fit(xframe(fit, macro_selected), target, sample_weight=weights)
    coverage = pd.to_numeric(fit.get("tracking_coverage_score"), errors="coerce").fillna(0.0)
    tracking_fit = fit.loc[coverage.ge(config.tracking_min_train_coverage)].copy()
    tracking_target = pd.to_numeric(tracking_fit[target_col], errors="coerce")
    full_model = None
    full_selected: list[str] = []
    if len(tracking_fit) >= 140:
        tracking_weights = recency_weights(tracking_fit, prediction_season, config.half_life_seasons)
        if "target_sample_weight" in tracking_fit.columns:
            tracking_weights *= pd.to_numeric(tracking_fit["target_sample_weight"], errors="coerce").fillna(1.0).clip(0.25, 2.5).to_numpy(float)
        full_selected = _select_features(
            tracking_fit, full_candidates, tracking_target, tracking_weights, config.full_k, seed + 1
        )
        full_model = _regressor(seed + 1)
        full_model.fit(xframe(tracking_fit, full_selected), tracking_target, sample_weight=tracking_weights)
    return {
        "macro_model": macro_model,
        "macro_features": macro_selected,
        "full_model": full_model,
        "full_features": full_selected,
        "target_col": target_col,
    }


def predict_layered(model: dict[str, Any], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    macro = np.asarray(model["macro_model"].predict(xframe(rows, model["macro_features"])), dtype=float)
    full_model = model.get("full_model")
    if full_model is None:
        return macro, macro.copy(), np.zeros(len(rows), dtype=float)
    full = np.asarray(full_model.predict(xframe(rows, model["full_features"])), dtype=float)
    gate = pd.to_numeric(rows.get("tracking_coverage_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(float)
    layered = macro + gate * (full - macro)
    return layered, macro, gate


def fit_availability(train: pd.DataFrame, prediction_season: int, features: list[str], config: V5Config) -> dict[str, Any] | None:
    y = pd.to_numeric(train.get(AVAILABILITY_TARGET), errors="coerce")
    fit = train.loc[y.notna()].copy()
    y = y.loc[fit.index].astype(int)
    if len(fit) < 200 or y.nunique() < 2:
        return None
    selected = features[: min(len(features), config.macro_k)]
    weights = recency_weights(fit, prediction_season, config.half_life_seasons)
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=4,
        iterations=350,
        learning_rate=0.035,
        l2_leaf_reg=14.0,
        random_seed=73,
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )
    model.fit(xframe(fit, selected), y, sample_weight=weights)
    return {"model": model, "features": selected}


def predict_availability(model: dict[str, Any] | None, rows: pd.DataFrame) -> np.ndarray:
    if model is None:
        return np.full(len(rows), np.nan)
    return np.asarray(model["model"].predict_proba(xframe(rows, model["features"]))[:, 1], dtype=float)


def load_category_weights(path: Path | None = None) -> dict[str, dict[str, float]]:
    import json

    path = path or ROOT / "data/pcv/all_in_one_8cat_tuned_weights.json"
    data = json.loads(path.read_text())
    mapping = {
        "Transition Offense": "cat_transition_offense_z",
        "Shot Defense": "cat_shot_defense_z",
        "Transition Defense": "cat_transition_defense_z",
        "Shot Creation": "cat_shot_creation_z",
        "Finishing": "cat_finishing_z",
        "Playmaking": "cat_playmaking_z",
        "Forecheck": "cat_forecheck_z",
        "Physicality": "cat_physicality_z",
        "Skating": "cat_skating_z",
    }
    out: dict[str, dict[str, float]] = {}
    for role in ("F", "D"):
        raw = data[f"weights_{role}"]
        vals = {label: max(0.0, float(raw.get(col, 0.0))) for label, col in mapping.items()}
        total = sum(vals.values()) or 1.0
        out[role] = {label: value / total for label, value in vals.items()}
    return out


def category_composite(latent_predictions: dict[str, np.ndarray], roles: pd.Series) -> pd.Series:
    weights = load_category_weights()
    raw = pd.Series(0.0, index=roles.index, dtype=float)
    den = pd.Series(0.0, index=roles.index, dtype=float)
    role_clean = roles.astype(str).str.upper().str.strip()
    for label, pred in latent_predictions.items():
        values = pd.Series(pred, index=roles.index, dtype=float)
        for role in ("F", "D"):
            w = float(weights[role].get(label, 0.0))
            mask = role_clean.eq(role) & values.notna()
            raw.loc[mask] += w * values.loc[mask]
            den.loc[mask] += w
    raw = raw / den.replace(0.0, np.nan)
    return rank_within_role(raw, role_clean)


def fit_bundle(train: pd.DataFrame, prediction_season: int, config: V5Config) -> dict[str, Any]:
    macro = macro_features(train)
    full = full_features(train, macro)
    bundle: dict[str, Any] = {
        "config": config,
        "direct": fit_layered_regressor(
            train,
            target_col="target_overall_latent",
            prediction_season=prediction_season,
            config=config,
            macro_candidates=macro,
            full_candidates=full,
            seed=42,
        ),
        "categories": {},
        "macro_candidates": macro,
        "full_candidates": full,
    }
    if "target_independent_latent" in train.columns and pd.to_numeric(train["target_independent_latent"], errors="coerce").notna().sum() >= 150:
        bundle["independent"] = fit_layered_regressor(
            train,
            target_col="target_independent_latent",
            prediction_season=prediction_season,
            config=config,
            macro_candidates=macro,
            full_candidates=macro,
            seed=84,
        )
    for i, label in enumerate(CATEGORY_LABELS):
        target_col = f"target_{label.lower().replace(' ', '_')}_latent"
        if target_col not in train.columns or pd.to_numeric(train[target_col], errors="coerce").notna().sum() < 140:
            continue
        own = category_features(train, macro, label)
        bundle["categories"][label] = fit_layered_regressor(
            train,
            target_col=target_col,
            prediction_season=prediction_season,
            config=config,
            macro_candidates=macro,
            full_candidates=own,
            seed=120 + i * 7,
        )
    bundle["availability"] = fit_availability(train, prediction_season, macro, config)
    return bundle


def predict_bundle(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    roles = rows["position_group"].astype(str).str.upper().str.strip()
    direct_latent, macro_latent, gate = predict_layered(bundle["direct"], rows)
    out["v5_direct_pct"] = rank_within_role(direct_latent, roles)
    out["v5_macro_pct"] = rank_within_role(macro_latent, roles)
    out["v5_tracking_gate"] = gate
    category_latent: dict[str, np.ndarray] = {}
    category_model_weights = bundle.get("category_model_weights", {})
    for label, model in bundle.get("categories", {}).items():
        latent, _, _ = predict_layered(model, rows)
        model_pct = rank_within_role(latent, roles)
        source_pct = pd.to_numeric(rows.get(f"{label}__pct"), errors="coerce")
        model_weight = float(np.clip(category_model_weights.get(label, 1.0), 0.0, 1.0))
        blended_pct = model_pct.copy()
        both = model_pct.notna() & source_pct.notna()
        blended_pct.loc[both] = model_weight * model_pct.loc[both] + (1.0 - model_weight) * source_pct.loc[both]
        blended_pct = rank_within_role(blended_pct, roles)
        category_latent[label] = percentile_to_latent(blended_pct).to_numpy(float)
        out[f"v5_projected_{label.lower().replace(' ', '_')}_pct"] = blended_pct
    out["v5_category_composite_pct"] = category_composite(category_latent, roles)
    if bundle.get("independent") is not None:
        independent_latent, _, _ = predict_layered(bundle["independent"], rows)
        out["v5_independent_pct"] = rank_within_role(independent_latent, roles)
    else:
        out["v5_independent_pct"] = np.nan
    out["v5_availability_probability"] = predict_availability(bundle.get("availability"), rows)
    return out


def apply_saved_bundle(
    rows: pd.DataFrame,
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    """Attach V5 components and, when promoted, blend its direct residual into V4."""
    import joblib

    if not Path(bundle_path).is_file():
        return rows
    bundle = joblib.load(bundle_path)
    source = rows.copy()
    if "eval_season_id" not in source.columns:
        source["eval_season_id"] = int(_read(score_path)["season_id"].max())
    prepared = prepare_panel(
        source,
        feature_path=feature_path,
        score_path=score_path,
        attach_targets=False,
    )
    pred = predict_bundle(bundle, prepared)
    out = rows.copy()
    rename = {
        "v5_direct_pct": "projected_next_year_pct_v5_direct",
        "v5_macro_pct": "projected_next_year_pct_v5_macro",
        "v5_tracking_gate": "projected_next_year_v5_tracking_gate",
        "v5_category_composite_pct": "projected_next_year_pct_v5_category_composite",
        "v5_independent_pct": "projected_next_year_pct_v5_independent_process",
        "v5_availability_probability": "projected_next_year_availability_probability",
    }
    for old, new in rename.items():
        out[new] = pd.to_numeric(pred.get(old), errors="coerce").to_numpy(float)
    for label in CATEGORY_LABELS:
        stem = label.lower().replace(" ", "_")
        source_col = f"v5_projected_{stem}_pct"
        if source_col in pred.columns:
            out[f"projected_{stem}_pct_v5"] = pd.to_numeric(pred[source_col], errors="coerce").to_numpy(float)
    v4 = pd.to_numeric(out.get("projected_next_year_pct"), errors="coerce")
    direct = pd.to_numeric(out.get("projected_next_year_pct_v5_direct"), errors="coerce")
    direct_weight = float(np.clip(bundle.get("v5_direct_weight", 0.0), 0.0, 1.0))
    v4_weight = float(np.clip(bundle.get("v4_weight", 1.0 - direct_weight), 0.0, 1.0))
    total = v4_weight + direct_weight
    if total <= 0.0:
        v4_weight, direct_weight, total = 1.0, 0.0, 1.0
    v4_weight /= total
    direct_weight /= total
    promoted = str(bundle.get("promotion_decision", "")).lower() == "promote"
    applicable = promoted & v4.notna() & direct.notna()
    hybrid = (v4_weight * v4 + direct_weight * direct).clip(0.0, 100.0)
    out["projected_next_year_pct_v4_pre_v5"] = v4
    out["projected_next_year_v5_weight_v4"] = v4_weight
    out["projected_next_year_v5_weight_direct"] = direct_weight
    out["projected_next_year_v5_applied"] = applicable.astype(int)
    out["projected_next_year_pct"] = hybrid.where(applicable, v4)
    availability = pd.to_numeric(out.get("projected_next_year_availability_probability"), errors="coerce")
    out["projected_next_year_expected_contribution_pct"] = (
        pd.to_numeric(out["projected_next_year_pct"], errors="coerce") * availability
    ).clip(0.0, 100.0)
    out["projection_model_version"] = np.where(applicable, "V5 hybrid", "V4")
    return out
