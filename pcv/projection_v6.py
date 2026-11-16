#!/usr/bin/env python3
"""Projection V6 challenger: aligned next-card Overall and honest season forecasts.

V6 keeps three estimands separate:

* stable next-card Overall (the headline card projection);
* next-season single-season performance (validation/research);
* availability (retained from V5 and never folded into talent).

The regression heads are role-specific.  Players with at most two seasons of
history also receive a sparse-history residual head that can use prospect and
development evidence which V5 did not admit to its feature whitelist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.stats import norm

from projection_v5 import (
    DEFAULT_FEATURES,
    DEFAULT_SCORES,
    full_features,
    macro_features,
    percentile_to_latent,
    prepare_panel,
    recency_weights,
    xframe,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = ROOT / "data/pcv/predictive/projection_v6_bundle.joblib"
STABLE_TARGET = "target_next_card_overall_pct"
SEASON_TARGET = "target_next_overall_8cat_tuned_pct"

# These columns already exist in the card-validation panel but were omitted by
# V5's candidate whitelist.  They are especially important for young players.
EXTRA_EXACT = {
    "rookie_prospect_prior_pct",
    "rookie_projection_gap_pct",
    "development_current_vs_prior_pct",
    "development_current_vs_model_pct",
    "development_projection_vs_current_pct",
    "development_young_improvement_signal",
    "development_decline_signal",
    "display_future_gap_pct",
    "display_current_model_gap_pct",
    "d_transition_offense_pct",
    "d_playmaking_pct",
    "f_finishing_pct",
    "f_physicality_pct",
    "weak_category_defense_noise_pct",
    "weak_category_defense_stack_pct",
    "weak_category_offense_minus_defense_pct",
    "target_reliability_weight",  # dropped by leakage guard below
}

LEAKAGE_PREFIXES = ("target_", "v5_projected_", "v6_")
LEAKAGE_EXACT = {
    "projected_next_year_pct",
    "projected_next_year_pct_raw",
    "learned_calibrated_display_pct",
    "learned_unweighted_display_pct",
    "learned_blended_target_display_pct",
    "learned_robust_target_display_pct",
    "learned_role_age_robust_target_display_pct",
    "learned_role_age_blended_target_display_pct",
    "learned_calibration_segment_robust_target_display_pct",
    "learned_grid_unweighted_display_pct",
    "learned_grid_robust_target_display_pct",
    "learned_residual_robust_display_pct",
    "blend_unweighted_blended_current_display_pct",
    "blend_grid_robust_residual_current_display_pct",
    "blend_v1_ensemble_display_pct",
    "blend_v1_production_display_pct",
    "segment_selected_robust_display_pct",
    "segment_selected_production_display_pct",
}


def candidate_features(rows: pd.DataFrame) -> list[str]:
    """Return numeric, source-season-only candidates."""
    macro = macro_features(rows)
    candidates = full_features(rows, macro)
    candidates.extend(c for c in sorted(EXTRA_EXACT) if c in rows.columns)
    out: list[str] = []
    for col in dict.fromkeys(candidates):
        if col in LEAKAGE_EXACT or col.startswith(LEAKAGE_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(rows[col]):
            continue
        if pd.to_numeric(rows[col], errors="coerce").notna().sum() < 60:
            continue
        out.append(col)
    return out


def _regressor(seed: int, *, loss: str = "RMSE") -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function=loss,
        depth=4,
        iterations=500,
        learning_rate=0.03,
        l2_leaf_reg=16.0,
        random_seed=int(seed),
        verbose=False,
        thread_count=4,
        allow_writing_files=False,
    )


def _select_features(
    train: pd.DataFrame,
    candidates: list[str],
    target: pd.Series,
    weights: np.ndarray,
    *,
    k: int,
    seed: int,
) -> list[str]:
    usable = [
        c for c in candidates
        if c in train.columns and pd.to_numeric(train[c], errors="coerce").notna().sum() >= 50
    ]
    if len(usable) <= int(k):
        return usable
    ranker = _regressor(seed)
    ranker.fit(xframe(train, usable), target, sample_weight=weights)
    ranked = sorted(
        zip(usable, ranker.get_feature_importance()),
        key=lambda item: (-float(item[1]), item[0]),
    )
    selected = [name for name, _ in ranked[: int(k)]]
    # Preserve the principal anchor and sparse-history evidence even if a noisy
    # early fold ranks one of them just below the cutoff.
    mandatory = {
        "overall_player_rating_display_pct",
        "overall_current_season_pct",
        "n_seasons_present",
        "age_years",
        "current_season_toi_display_min",
        "rookie_prospect_prior_pct",
        "development_current_vs_prior_pct",
        "development_current_vs_model_pct",
    }
    selected.extend(c for c in usable if c in mandatory and c not in selected)
    return selected


def _fit_one(
    train: pd.DataFrame,
    features: list[str],
    target: pd.Series,
    weights: np.ndarray,
    seed: int,
) -> CatBoostRegressor:
    model = _regressor(seed)
    model.fit(xframe(train, features), target, sample_weight=weights)
    return model


def _pct_to_latent_array(values: pd.Series) -> np.ndarray:
    return percentile_to_latent(values).to_numpy(float)


def _latent_to_pct(values: np.ndarray) -> np.ndarray:
    return np.clip(norm.cdf(np.asarray(values, dtype=float)) * 100.0, 0.0, 100.0)


def fit_specialist_bundle(
    train: pd.DataFrame,
    *,
    target_col: str,
    anchor_col: str,
    prediction_season: int,
    half_life: float = 1.0,
    feature_k: int = 85,
    seed: int = 42,
) -> dict[str, Any]:
    target_pct = pd.to_numeric(train[target_col], errors="coerce")
    anchor_pct = pd.to_numeric(train[anchor_col], errors="coerce")
    fit = train.loc[target_pct.notna() & anchor_pct.notna()].copy()
    target_latent = percentile_to_latent(target_pct.loc[fit.index])
    anchor_latent = percentile_to_latent(anchor_pct.loc[fit.index])
    residual = target_latent - anchor_latent
    weights = recency_weights(fit, prediction_season, half_life)
    if target_col == SEASON_TARGET and "target_sample_weight" in fit.columns:
        weights *= (
            pd.to_numeric(fit["target_sample_weight"], errors="coerce")
            .fillna(1.0).clip(0.20, 2.5).to_numpy(float)
        )
    candidates = candidate_features(fit)
    features = _select_features(
        fit, candidates, target_latent, weights, k=feature_k, seed=seed
    )
    bundle: dict[str, Any] = {
        "target_col": target_col,
        "anchor_col": anchor_col,
        "features": features,
        "half_life": float(half_life),
        "prediction_season": int(prediction_season),
        "role_models": {},
        "sparse_model": None,
    }
    roles = fit["position_group"].astype(str).str.upper().str.strip()
    for i, role in enumerate(("F", "D")):
        idx = fit.index[roles.eq(role)]
        if len(idx) < 100:
            continue
        loc = fit.index.get_indexer(idx)
        direct = _fit_one(fit.loc[idx], features, target_latent.loc[idx], weights[loc], seed + 10 + i)
        resid = _fit_one(fit.loc[idx], features, residual.loc[idx], weights[loc], seed + 20 + i)
        bundle["role_models"][role] = {"direct": direct, "residual": resid}

    sparse_mask = pd.to_numeric(fit.get("n_seasons_present"), errors="coerce").fillna(3.0).le(2.0)
    sparse = fit.loc[sparse_mask].copy()
    if len(sparse) >= 140:
        idx = sparse.index
        loc = fit.index.get_indexer(idx)
        bundle["sparse_model"] = _fit_one(
            sparse, features, residual.loc[idx], weights[loc], seed + 40
        )
        bundle["sparse_n"] = int(len(sparse))
    return bundle


def predict_candidates(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.DataFrame:
    features = list(bundle["features"])
    anchor = pd.to_numeric(rows[bundle["anchor_col"]], errors="coerce")
    anchor_latent = _pct_to_latent_array(anchor)
    roles = rows["position_group"].astype(str).str.upper().str.strip()
    direct_latent = np.full(len(rows), np.nan, dtype=float)
    residual_latent = np.full(len(rows), np.nan, dtype=float)
    for role, models in bundle.get("role_models", {}).items():
        mask = roles.eq(role).to_numpy()
        if not mask.any():
            continue
        direct_latent[mask] = models["direct"].predict(xframe(rows.loc[mask], features))
        residual_latent[mask] = anchor_latent[mask] + models["residual"].predict(
            xframe(rows.loc[mask], features)
        )
    direct_pct = _latent_to_pct(direct_latent)
    residual_pct = _latent_to_pct(residual_latent)
    out = pd.DataFrame(index=rows.index)
    out["anchor"] = anchor.to_numpy(float)
    out["role_direct"] = direct_pct
    out["role_residual"] = residual_pct
    for weight in (0.25, 0.50, 0.75):
        key = int(round(weight * 100))
        out[f"residual_w{key}"] = (
            weight * out["role_residual"] + (1.0 - weight) * out["anchor"]
        ).clip(0.0, 100.0)
        out[f"direct_w{key}"] = (
            weight * out["role_direct"] + (1.0 - weight) * out["anchor"]
        ).clip(0.0, 100.0)
    sparse_model = bundle.get("sparse_model")
    sparse_mask = pd.to_numeric(rows.get("n_seasons_present"), errors="coerce").fillna(3.0).le(2.0)
    if sparse_model is not None and sparse_mask.any():
        sparse_resid = anchor_latent[sparse_mask.to_numpy()] + sparse_model.predict(
            xframe(rows.loc[sparse_mask], features)
        )
        sparse_pct = pd.Series(_latent_to_pct(sparse_resid), index=rows.index[sparse_mask])
        for base in ("role_residual", "residual_w50", "residual_w75"):
            for weight in (0.25, 0.50, 0.75):
                col = f"{base}_sparse{int(weight * 100)}"
                out[col] = out[base]
                out.loc[sparse_mask, col] = (
                    weight * sparse_pct + (1.0 - weight) * out.loc[sparse_mask, base]
                ).clip(0.0, 100.0)
    return out


def predict_selected(bundle: dict[str, Any], rows: pd.DataFrame) -> pd.Series:
    candidates = predict_candidates(bundle, rows)
    selected = str(bundle.get("selected_candidate", "residual_w50"))
    if selected not in candidates:
        selected = "residual_w50"
    pred = pd.to_numeric(candidates[selected], errors="coerce").clip(0.0, 100.0)
    segment_policy = bundle.get("selected_candidate_by_segment") or {}
    if segment_policy:
        roles = rows["position_group"].astype(str).str.upper().str.strip()
        histories = pd.to_numeric(rows.get("n_seasons_present"), errors="coerce")
        segmented = pred.copy()
        for role in ("F", "D"):
            role_mask = roles.eq(role)
            for sparse, suffix in ((True, "sparse"), (False, "established")):
                mask = role_mask & (histories.le(2.0) if sparse else histories.ge(3.0))
                candidate = str(segment_policy.get(f"{role}_{suffix}", segment_policy.get(role, selected)))
                if candidate in candidates:
                    segmented.loc[mask] = pd.to_numeric(candidates.loc[mask, candidate], errors="coerce")
            unknown = role_mask & histories.isna()
            candidate = str(segment_policy.get(role, selected))
            if candidate in candidates:
                segmented.loc[unknown] = pd.to_numeric(candidates.loc[unknown, candidate], errors="coerce")
        pred = segmented.clip(0.0, 100.0)
    calibration = bundle.get("calibration", {})
    roles = rows["position_group"].astype(str).str.upper().str.strip()
    if calibration:
        out = pred.copy()
        fallback = calibration.get("all", {"slope": 1.0, "intercept": 0.0})
        for role in roles.unique():
            params = calibration.get(str(role), fallback)
            idx = roles.eq(role)
            out.loc[idx] = (
                float(params.get("slope", 1.0)) * pred.loc[idx]
                + float(params.get("intercept", 0.0))
            ).clip(0.0, 100.0)
        pred = out
    return pred


def apply_saved_bundle(
    rows: pd.DataFrame,
    *,
    bundle_path: Path = DEFAULT_BUNDLE,
    feature_path: Path = DEFAULT_FEATURES,
    score_path: Path = DEFAULT_SCORES,
) -> pd.DataFrame:
    """Attach V6 outputs, replacing the headline only after explicit promotion."""
    if not Path(bundle_path).is_file():
        return rows
    bundle = joblib.load(bundle_path)
    source = rows.copy()
    if "eval_season_id" not in source.columns:
        score = pd.read_parquet(score_path) if Path(score_path).suffix == ".parquet" else pd.read_csv(score_path, low_memory=False)
        source["eval_season_id"] = int(pd.to_numeric(score["season_id"], errors="coerce").max())
    prepared = prepare_panel(source, feature_path=feature_path, score_path=score_path, attach_targets=False)
    out = rows.copy()
    stable = predict_selected(bundle["stable"], prepared)
    season = predict_selected(bundle["season"], prepared) if bundle.get("season") else pd.Series(np.nan, index=rows.index)
    stable_delta = stable - pd.to_numeric(
        source.get("overall_player_rating_display_pct"), errors="coerce"
    )
    # Current forecasts are generated from a validation-style aggregate rebuilt
    # by the same code used for every historical fold.  Prefer that stored
    # snapshot by player id so production-only posthoc card calibration cannot
    # create a train/deploy feature shift.  Unseen players keep the model
    # fallback above.
    snapshot = bundle.get("current_predictions")
    snapshot_season = bundle.get("current_prediction_season")
    if snapshot and snapshot_season is not None and "player_id" in source.columns:
        sid = pd.to_numeric(source["eval_season_id"], errors="coerce")
        pid = pd.to_numeric(source["player_id"], errors="coerce")
        use = sid.eq(int(snapshot_season))
        stable_map = {
            int(k): float(v["stable"])
            for k, v in snapshot.items() if v.get("stable") is not None
        }
        stable_delta_map = {
            int(k): float(v["stable_delta"])
            for k, v in snapshot.items() if v.get("stable_delta") is not None
        }
        season_map = {
            int(k): float(v["season"])
            for k, v in snapshot.items() if v.get("season") is not None
        }
        stable_override = pid.map(stable_map)
        if stable_delta_map:
            production_anchor = pd.to_numeric(
                source.get("overall_player_rating_display_pct"), errors="coerce"
            )
            delta_override = pid.map(stable_delta_map)
            translated = (production_anchor + delta_override).clip(0.0, 100.0)
            stable_override = translated.where(translated.notna(), stable_override)
            stable_delta = delta_override.where(use & delta_override.notna(), stable_delta)
        season_override = pid.map(season_map)
        stable = stable_override.where(use & stable_override.notna(), stable)
        season = season_override.where(use & season_override.notna(), season)
    out["projected_next_card_overall_pct_v6"] = stable.to_numpy(float)
    out["projected_next_card_overall_delta_v6"] = stable_delta.to_numpy(float)
    out["projected_next_season_performance_pct_v6"] = season.to_numpy(float)
    out["projected_next_year_pct_pre_v6"] = pd.to_numeric(out.get("projected_next_year_pct"), errors="coerce")
    promoted = str(bundle.get("promotion_decision", "")).lower() == "promote"
    applicable = promoted & stable.notna()
    out["projected_next_year_v6_applied"] = applicable.astype(int)
    out["projected_next_year_pct"] = stable.where(applicable, out["projected_next_year_pct_pre_v6"])
    if promoted:
        out["projection_model_version"] = np.where(applicable, "V6 stable-overall", out.get("projection_model_version", "V5"))
    return out
