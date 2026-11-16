#!/usr/bin/env python3
"""Matched HERO/category ablation for the frozen V3 team-shot cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from research import team_supply as v3

from research.paths import ARCHIVE as ROOT, output

OUT = output("hero_conditioned")
HERO_INPUT = ROOT / "outputs/ability_patch_v1/inputs/tuned_categories.parquet"
HERO_WEIGHTS = ROOT / "outputs/ability_patch_v1/inputs/weights_v7.json"
HERO_REPLAY = ROOT / "outputs/hero_continuous_shrink_v1/historical_ratings.parquet"
SEASONS = (20222023, 20232024, 20242025, 20252026)
CATS = (
    "cat_shot_creation_z", "cat_transition_offense_z", "cat_shot_defense_z",
    "cat_transition_defense_z", "cat_finishing_z", "cat_playmaking_z",
    "cat_forecheck_z", "cat_physicality_z", "cat_skating_z",
)
ARMS = ("v3_base", "v3_shares", "hero_scalar", "hero_scalar_shares",
        "hero_categories", "hero_categories_shares")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hero_profiles() -> tuple[pd.DataFrame, dict]:
    p = pd.read_parquet(HERO_INPUT, columns=["player_id", "season_id",
        "position_group", "model_rating_eligible", "eh_rapm_TOI",
        "a3z_toi_5v5_min", *CATS])
    assert not p.duplicated(["player_id", "season_id"]).any()
    p = p[p.model_rating_eligible.fillna(False).astype(bool) &
          p.season_id.between(20192020, 20252026)].copy()
    p["toi"] = pd.to_numeric(p.eh_rapm_TOI, errors="coerce").where(
        pd.to_numeric(p.eh_rapm_TOI, errors="coerce").gt(0),
        pd.to_numeric(p.a3z_toi_5v5_min, errors="coerce")
    ).fillna(0.).clip(lower=0.)
    p = p[p.toi.gt(0)].copy()
    frames = []
    for cutoff in (20212022, 20222023, 20232024, 20242025, 20252026):
        latest = p.loc[p.season_id.eq(cutoff),
            ["player_id", "position_group"]].copy()
        hist = p[p.season_id.between(cutoff-20002, cutoff) &
                 p.player_id.isin(latest.player_id)].copy()
        age = cutoff//10000 - hist.season_id//10000
        hist["weight"] = hist.toi * np.power(0.4, age)
        hist["normalized"] = hist.weight / hist.groupby("player_id").weight.transform("sum")
        hist["missing_fraction"] = hist[list(CATS)].isna().sum(axis=1) / len(CATS)
        pool = hist[["player_id"]].copy()
        for cat in CATS:
            pool[cat] = hist[cat].fillna(0.).to_numpy(float) * hist.normalized.to_numpy(float)
        pool["missing_fraction"] = hist.missing_fraction * hist.normalized
        pooled = pool.groupby("player_id", as_index=False).sum()
        minutes = hist.groupby("player_id").toi.sum().rename("window_toi")
        q = latest.merge(pooled, on="player_id", validate="one_to_one")
        q = q.merge(minutes, on="player_id", validate="one_to_one")
        shrink = q.window_toi / (q.window_toi + 400.)
        for cat in CATS:
            q[cat] *= shrink
        q["source_season"] = cutoff
        q["hero_raw"] = 0.
        weights = json.loads(HERO_WEIGHTS.read_text())
        for cat in CATS:
            q["hero_raw"] += q[cat] * q.position_group.map(
                {role: weights[f"weights_{role}"][cat] for role in ("F", "D")})
        frames.append(q)
    profile = pd.concat(frames, ignore_index=True)
    assert not profile.duplicated(["player_id", "source_season"]).any()
    assert np.isfinite(profile[[*CATS, "hero_raw", "missing_fraction"]]).all().all()
    prior = pd.read_parquet(HERO_REPLAY, columns=["player_id", "source_season", "k_400_raw"])
    check = profile.merge(prior, on=["player_id", "source_season"],
                          validate="one_to_one", indicator=True)
    assert check._merge.eq("both").all() and len(check)==len(profile)==len(prior)
    maxerr = float((check.hero_raw-check.k_400_raw).abs().max())
    assert maxerr < 1e-10, f"HERO V8 replay mismatch {maxerr}"
    return profile, {"rows": len(profile), "v8_max_abs_replay_error": maxerr,
        "source_seasons": sorted(profile.source_season.unique().tolist()),
        "missing_fraction_by_cutoff": {
            str(k): float(v) for k,v in profile.groupby("source_season").missing_fraction.mean().items()}}


def add_source_profiles(pg: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    z = pg.copy()
    z["source_season"] = z.season - 10001
    use = q[["player_id", "source_season", "position_group", "hero_raw",
             "missing_fraction", *CATS]].rename(columns={
                 "position_group": "hero_role"})
    z = z.merge(use, on=["player_id", "source_season"], how="left",
                validate="many_to_one")
    assert z.loc[z.hero_raw.notna(), "source_season"].lt(z.loc[z.hero_raw.notna(), "season"]).all()
    z["hero_available"] = z.hero_raw.notna().astype(float)
    z[["hero_raw", "missing_fraction", *CATS]] = z[["hero_raw",
        "missing_fraction", *CATS]].fillna(0.)
    return z


def context_features(selected: pd.DataFrame, pg: pd.DataFrame,
                     source_years: list[int], kind: str) -> sparse.csr_matrix:
    rows = pg[pg.season.isin(source_years + [int(selected.season.max())])].copy()
    keys = pd.MultiIndex.from_frame(selected[["game", "side"]])
    own = keys.get_indexer(pd.MultiIndex.from_frame(rows[["game", "side"]]))
    opposite = rows[["game", "side"]].copy()
    opposite["side"] = 1-opposite.side
    opp = keys.get_indexer(pd.MultiIndex.from_frame(opposite))
    assert (own>=0).all() and (opp>=0).all()
    share = rows.seconds.to_numpy(float) / (5*selected.seconds.to_numpy(float)[own])
    assert np.all((share > 0) & (share <= 1))
    columns = ["hero_raw"] if kind=="scalar" else list(CATS)
    x = []
    roles = rows.hero_role.to_numpy()
    for index in (own, opp):
        for role in ("F", "D"):
            r = (roles==role).astype(float)
            for col in (*columns, "hero_available", "missing_fraction"):
                val = rows[col].to_numpy(float) * share * r
                x.append(np.bincount(index, weights=val, minlength=len(selected)))
    arr = np.column_stack(x)
    train = selected.season.isin(source_years).to_numpy()
    mean = arr[train].mean(axis=0)
    scale = arr[train].std(axis=0)
    scale = np.where(scale>1e-8, scale, 1.)
    out = (arr-mean)/scale
    assert np.isfinite(out).all()
    return sparse.csr_matrix(out)


def make_design(tg: pd.DataFrame, pg: pd.DataFrame, source_years: list[int],
                target: int, arm: str):
    shares = arm.endswith("shares")
    selected, base_x, train, test, audit = v3.design(tg, pg, source_years,
                                                    target, shares)
    if arm.startswith("hero_"):
        kind = "scalar" if arm.startswith("hero_scalar") else "categories"
        extra = context_features(selected, pg, source_years, kind)
        x = sparse.hstack([base_x, extra], format="csr")
        audit["hero_columns"] = extra.shape[1]
    else:
        x = base_x
        audit["hero_columns"] = 0
    return selected, x, train, test, audit


def paired_interval(pred: pd.DataFrame, seed: int=20261004, draws: int=2000) -> list[float]:
    rng = np.random.default_rng(seed)
    fold_groups = []
    for season in v3.TARGETS:
        d = pred[pred.season.eq(season)].copy()
        delta = (d.hero_categories_shares-d.actual)**2 - (d.hero_categories-d.actual)**2
        groups = pd.DataFrame({"game": d.game, "delta": delta}).groupby("game").delta.agg(["sum","count"]).to_numpy(float)
        fold_groups.append(groups)
    sim = np.empty(draws)
    for i in range(draws):
        means = []
        for groups in fold_groups:
            draw = groups[rng.integers(len(groups), size=len(groups))]
            means.append(draw[:,0].sum()/draw[:,1].sum())
        sim[i] = np.mean(means)
    return np.quantile(sim, [.025,.975]).tolist()


def main() -> None:
    q, q_audit = hero_profiles()  # Gate lineage before reading any outer target.
    tg_path = v3.BASE/"team_games.parquet"
    pg_path = v3.BASE/"player_games.parquet"
    event_path = v3.BASE/"isolated_event_panel.parquet"
    tg = pd.read_parquet(tg_path)
    pg = pd.read_parquet(pg_path)
    measure = pd.read_parquet(v3.MEASURE)
    assert tg.season.max()==20252026 and pg.season.max()==20252026
    tg = v3.attach_pregame_form(tg)
    tg = tg[tg.season.isin(SEASONS)].copy()
    pg = v3.source_prior_features(pg[pg.season.isin(SEASONS)], measure)
    pg = add_source_profiles(pg, q)
    event = pd.read_parquet(event_path)[["season", "event_game", "team"]].drop_duplicates()
    event.rename(columns={"event_game":"game"}, inplace=True)
    tg = tg.merge(event.assign(is_isolated_event=True), on=["season","game","team"],
                  how="left", validate="one_to_one")
    tg["is_isolated_event"] = tg.is_isolated_event.eq(True)
    report = {"status":"inspected_development", "question":"HERO-conditioned share increment",
              "hero_replay":q_audit, "folds":[]}
    predictions = []
    for target in v3.TARGETS:
        source_years = [s for s in SEASONS if s<target]
        previous = target-10001
        inner_years = [s for s in source_years if s<previous]
        fold = {"target":target, "source_years":source_years, "arms":{}}
        outer_preds = {}
        for arm in ARMS:
            inner_z, inner_x, inner_train, inner_valid, inner_audit = make_design(
                tg, pg, inner_years, previous, arm)
            y_inner = inner_z.loc[inner_valid, "shots"].to_numpy(float)
            grid = {}
            for alpha in v3.ALPHAS:
                pred = v3.fit_predict(inner_z, inner_x, inner_train, inner_valid, alpha)
                grid[str(alpha)] = float(np.mean((pred-y_inner)**2))
            alpha = min(v3.ALPHAS, key=lambda a:(grid[str(a)],a))
            outer_z, outer_x, outer_train, outer_test, outer_audit = make_design(
                tg, pg, source_years, target, arm)
            outer_preds[arm] = v3.fit_predict(outer_z, outer_x, outer_train, outer_test, alpha)
            fold["arms"][arm] = {"alpha":alpha,"inner_mse":grid,
                                  "inner_design":inner_audit,"outer_design":outer_audit}
        d = outer_z.loc[outer_test, ["game","season","team","side","seconds",
            "shots","xg","goals","is_isolated_event"]].copy()
        d.rename(columns={"shots":"actual"}, inplace=True)
        for arm in ARMS:
            d[arm] = outer_preds[arm]
        d.reset_index(drop=True, inplace=True)
        for arm in ARMS:
            fold["arms"][arm]["outer_mse"] = float(np.mean((d[arm]-d.actual)**2))
            fold["arms"][arm]["observed_predicted_ratio"] = float(d.actual.sum()/d[arm].sum())
        fold["primary_delta"] = fold["arms"]["hero_categories_shares"]["outer_mse"] - fold["arms"]["hero_categories"]["outer_mse"]
        sub = d[d.is_isolated_event]
        fold["isolated_event_n"] = len(sub)
        fold["isolated_event_delta"] = float(np.mean((sub.hero_categories_shares-sub.actual)**2 -
                                                      (sub.hero_categories-sub.actual)**2))
        fold["hero_profile_exposure_share"] = float(pg.loc[pg.season.eq(target)].assign(
            covered=lambda z:z.seconds*z.hero_available).covered.sum()/pg.loc[pg.season.eq(target),"seconds"].sum())
        report["folds"].append(fold)
        predictions.append(d)
    pred = pd.concat(predictions, ignore_index=True)
    old = pd.read_parquet(ROOT/"outputs/team_supply_prior_v3/heldout_team_games.parquet")
    check = pred.merge(old[["game","side","base","candidate"]],
                       on=["game","side"],validate="one_to_one")
    assert len(check)==len(pred)==len(old)
    replay = {"base":float((check.v3_base-check.base).abs().max()),
              "shares":float((check.v3_shares-check.candidate).abs().max())}
    assert max(replay.values())<1e-6, replay
    report["v3_prediction_replay_max_abs"] = replay
    for arm in ARMS:
        report.setdefault("pooled_mse",{})[arm] = float(np.mean((pred[arm]-pred.actual)**2))
    report["primary_equal_fold_delta"] = float(np.mean([f["primary_delta"] for f in report["folds"]]))
    report["primary_game_bootstrap95"] = paired_interval(pred)
    report["primary_gate_pass"] = bool(all(f["primary_delta"]<0 for f in report["folds"]) and
                                       report["primary_game_bootstrap95"][1]<0)
    source_paths = [tg_path,pg_path,event_path,v3.MEASURE,v3.SHOT_SHARE,v3.A1_SHARE,
                    HERO_INPUT,HERO_WEIGHTS,HERO_REPLAY]
    report["source_sha256"] = {str(p):sha(p) for p in source_paths}
    report["code_sha256"] = sha(Path(__file__))
    pred.to_parquet(OUT/"heldout_team_games.parquet", index=False)
    (OUT/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("hero_replay","v3_prediction_replay_max_abs",
        "pooled_mse","primary_equal_fold_delta","primary_game_bootstrap95",
        "primary_gate_pass")},indent=2))


if __name__=="__main__":
    main()
