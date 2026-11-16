#!/usr/bin/env python3
"""Matched team-game ridge with pregame form and conditional share priors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

from research.paths import ARCHIVE as ROOT, output

OUT = output("team_supply")
BASE = ROOT / "outputs/changed_lineup_v1"
MEASURE = ROOT / "outputs/staged_v2/full_source_measurements.parquet"
SHOT_SHARE = ROOT / "outputs/opportunity_share_v1/official_shot_share_seasons.parquet"
A1_SHARE = ROOT / "outputs/opportunity_share_v1/tracked_a1_share_seasons.parquet"
ALPHAS = (0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.)
TARGETS = (20242025, 20252026)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_prior_features(pg: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    z = pg.copy()
    z["source_season"] = z.season - 10001
    mm = m[["player_id", "season_id", "position_group", "strict_minutes",
        "tracked_minutes"]].rename(
        columns={"season_id": "source_season"})
    assert not mm.duplicated(["player_id", "source_season"]).any()
    shot = pd.read_parquet(SHOT_SHARE).rename(columns={
        "season_id": "source_season", "opportunities": "shot_opp", "successes": "shot_success"})
    a1 = pd.read_parquet(A1_SHARE).rename(columns={
        "season_id": "source_season", "opportunities": "a1_opp", "successes": "a1_success"})
    assert not shot.duplicated(["player_id", "source_season"]).any()
    assert not a1.duplicated(["player_id", "source_season"]).any()
    mm = mm.merge(shot, on=["player_id", "source_season"], how="left",
                  validate="one_to_one")
    mm = mm.merge(a1, on=["player_id", "source_season"], how="left",
                  validate="one_to_one")
    z = z.merge(mm, on=["player_id", "source_season"], how="left",
                validate="many_to_one")
    source_valid = (mm.position_group.isin(["F", "D"]) &
        mm.strict_minutes.ge(300) & mm.tracked_minutes.ge(100) &
        mm.shot_opp.ge(50) & mm.a1_opp.ge(30))
    p = mm[source_valid].groupby(["source_season", "position_group"]).agg(
        shot_num=("shot_success", "sum"), shot_den=("shot_opp", "sum"),
        a1_num=("a1_success", "sum"), a1_den=("a1_opp", "sum")).reset_index()
    p["shot_prior"] = p.shot_num / p.shot_den
    p["a1_prior"] = p.a1_num / p.a1_den
    z = z.merge(p[["source_season", "position_group", "shot_prior", "a1_prior"]],
                on=["source_season", "position_group"], how="left",
                validate="many_to_one")
    eligible = (z.position_group.isin(["F", "D"]) &
        z.strict_minutes.ge(300) & z.tracked_minutes.ge(100) &
        z.shot_opp.ge(50) & z.a1_opp.ge(30))
    z["has_source_behavior"] = eligible
    shot_dev = ((z.shot_success + 150 * z.shot_prior) /
                (z.shot_opp + 150)) - z.shot_prior
    a1_dev = ((z.a1_success + 150 * z.a1_prior) /
              (z.a1_opp + 150)) - z.a1_prior
    z["shot_deviation"] = shot_dev.where(eligible, 0.).fillna(0.)
    z["a1_deviation"] = a1_dev.where(eligible, 0.).fillna(0.)
    assert np.isfinite(z[["shot_deviation", "a1_deviation"]]).all().all()
    return z


def attach_pregame_form(tg: pd.DataFrame) -> pd.DataFrame:
    z = tg.copy()
    z["team_canonical"] = z.team.replace({"UTA": "ARI"})
    prior = z.groupby(["season", "team_canonical"]).agg(
        shots=("shots", "sum"), seconds=("seconds", "sum")).reset_index()
    prior["prior_season_rate"] = prior.shots * 3600 / prior.seconds
    prior["season"] += 10001
    z = z.merge(prior[["season", "team_canonical", "prior_season_rate"]],
                on=["season", "team_canonical"], how="left", validate="many_to_one")
    z = z.sort_values(["season", "team_canonical", "date", "game"])
    g = z.groupby(["season", "team_canonical"], sort=False)
    prev_shots = g.shots.cumsum() - z.shots
    prev_seconds = g.seconds.cumsum() - z.seconds
    ten_game_seconds = 30000.
    z["pregame_own_form"] = ((z.prior_season_rate * ten_game_seconds / 3600 +
        prev_shots) / (ten_game_seconds + prev_seconds)) * 3600
    opponent = z[["game", "side", "pregame_own_form"]].copy()
    opponent.side = 1 - opponent.side
    opponent.rename(columns={"pregame_own_form": "pregame_opponent_form"}, inplace=True)
    z = z.merge(opponent, on=["game", "side"], validate="one_to_one")
    assert np.isfinite(z.loc[z.season.ge(20222023),
        ["pregame_own_form", "pregame_opponent_form"]]).all().all()
    return z


def design(tg: pd.DataFrame, pg: pd.DataFrame, source_years: list[int],
           target: int, candidate: bool):
    selected = tg[tg.season.isin(source_years + [target])].copy().reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(selected[["game", "side"]])
    rows = pg[pg.season.isin(source_years + [target])].copy()
    own = keys.get_indexer(pd.MultiIndex.from_frame(rows[["game", "side"]]))
    opposite_keys = rows[["game", "side"]].copy()
    opposite_keys["side"] = 1 - opposite_keys.side
    opp = keys.get_indexer(pd.MultiIndex.from_frame(opposite_keys))
    assert (own >= 0).all() and (opp >= 0).all()
    assert np.array_equal(selected.game.to_numpy()[own], rows.game.to_numpy())
    player_vocab = np.sort(rows[rows.season.isin(source_years)].player_id.unique())
    pid = pd.Index(player_vocab).get_indexer(rows.player_id)
    valid = pid >= 0
    n = len(selected)
    k = len(player_vocab)
    denom = selected.seconds.to_numpy(float)[own] * 5.
    share = rows.seconds.to_numpy(float) / denom
    assert np.all(share > 0)
    off = sparse.coo_matrix((share[valid], (own[valid], pid[valid])),
                             shape=(n, 2 * k)).tocsr()
    deff = sparse.coo_matrix((share[valid], (opp[valid], k + pid[valid])),
                              shape=(n, 2 * k)).tocsr()
    player_x = off + deff
    teams = sorted(selected[selected.season.isin(source_years)].team_canonical.unique())
    opponent = selected[["game", "side", "team_canonical"]].copy()
    opponent.side = 1 - opponent.side
    opponent.rename(columns={"team_canonical": "opponent_team"}, inplace=True)
    selected = selected.merge(opponent, on=["game", "side"], validate="one_to_one")
    team_cols = np.column_stack(
        [(selected.team_canonical == team).to_numpy(float) for team in teams] +
        [(selected.opponent_team == team).to_numpy(float) for team in teams] +
        [(selected.side == 0).to_numpy(float)])
    form = selected[["pregame_own_form", "pregame_opponent_form"]].to_numpy(float)
    train = selected.season.isin(source_years).to_numpy()
    mean = form[train].mean(axis=0)
    std = np.maximum(form[train].std(axis=0), 1e-8)
    team_cols = np.column_stack([team_cols, (form - mean) / std])
    x = sparse.hstack([player_x, sparse.csr_matrix(team_cols)], format="csr")
    if candidate:
        profile = np.column_stack([
            np.bincount(own, weights=share * rows.shot_deviation.to_numpy(float), minlength=n),
            np.bincount(own, weights=share * rows.a1_deviation.to_numpy(float), minlength=n),
        ])
        train = selected.season.isin(source_years).to_numpy()
        std = profile[train].std(axis=0)
        std = np.where(std > 1e-8, std, 1.)
        profile /= std
        x = sparse.hstack([x, sparse.csr_matrix(profile)], format="csr")
    source_mask = selected.season.isin(source_years).to_numpy()
    target_mask = selected.season.eq(target).to_numpy()
    target_exposure = rows.loc[rows.season.eq(target), "seconds"].sum()
    source_share = (float(rows.loc[rows.season.eq(target) & rows.has_source_behavior,
                                  "seconds"].sum() / target_exposure)
                    if target_exposure else None)
    unknown_share = (float(share[(rows.season.to_numpy() == target) & ~valid].sum() /
                           target_mask.sum()) if target_mask.sum() else None)
    return selected, x, source_mask, target_mask, {
        "source_players": k, "source_behavior_exposure_share": source_share,
        "unknown_target_offensive_player_share": unknown_share,
        "team_vocab": teams, "columns": x.shape[1]}


def fit_predict(selected: pd.DataFrame, x: sparse.csr_matrix,
                train: np.ndarray, test: np.ndarray, alpha: float) -> np.ndarray:
    h = selected.seconds.to_numpy(float) / 3600.
    count = selected.shots.to_numpy(float)
    y = count / h
    # Exactly unpenalized intercept in an augmented weighted ridge system.
    # This also avoids the host's faulty accelerated dense matmul path.
    xt = x[train].tocsr()
    sw = np.sqrt(h[train])
    upper = sparse.hstack([xt.multiply(sw[:, None]),
        sparse.csr_matrix(sw[:, None])], format="csr")
    lower = sparse.hstack([np.sqrt(alpha) * sparse.eye(x.shape[1], format="csr"),
        sparse.csr_matrix((x.shape[1], 1))], format="csr")
    system = sparse.vstack([upper, lower], format="csr")
    target = np.r_[sw * y[train], np.zeros(x.shape[1])]
    coef = lsqr(system, target, atol=1e-8, btol=1e-8, iter_lim=3000)[0]
    rate = np.asarray(x[test].dot(coef[:-1])).ravel() + coef[-1]
    assert np.isfinite(rate).all()
    return h[test] * np.maximum(rate, 0.)


def scores(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> dict:
    return {"n": len(y), "base_mse": float(np.mean((base-y)**2)),
        "candidate_mse": float(np.mean((candidate-y)**2)),
        "candidate_minus_base": float(np.mean((candidate-y)**2 - (base-y)**2)),
        "base_observed_ratio": float(y.sum()/base.sum()),
        "candidate_observed_ratio": float(y.sum()/candidate.sum())}


def ci(d: pd.DataFrame) -> list[float]:
    delta = (d.candidate - d.actual)**2 - (d.base - d.actual)**2
    groups = pd.DataFrame({"game": d.game, "delta": delta}).groupby("game").delta.agg(["sum", "count"]).to_numpy(float)
    rng = np.random.default_rng(20261001)
    draws = np.empty(1000)
    for j in range(len(draws)):
        s = groups[rng.integers(len(groups), size=len(groups))]
        draws[j] = s[:,0].sum()/s[:,1].sum()
    return np.quantile(draws,[.025,.975]).tolist()


def main() -> None:
    tg_path = BASE / "team_games.parquet"
    pg_path = BASE / "player_games.parquet"
    panel_path = BASE / "isolated_event_panel.parquet"
    tg = pd.read_parquet(tg_path)
    pg = pd.read_parquet(pg_path)
    m = pd.read_parquet(MEASURE)
    tg = attach_pregame_form(tg)
    tg = tg[tg.season.between(20222023, 20252026)].copy()
    pg = source_prior_features(pg[pg.season.between(20222023, 20252026)], m)
    assert np.allclose(pg.groupby(["game", "side"]).seconds.sum().to_numpy(),
        5 * tg.set_index(["game", "side"]).loc[pg.groupby(["game", "side"]).seconds.sum().index].seconds.to_numpy())
    event = pd.read_parquet(panel_path)[["season", "event_game", "team"]].drop_duplicates()
    event.rename(columns={"event_game": "game"}, inplace=True)
    tg = tg.merge(event.assign(is_isolated_event=True), on=["season", "game", "team"],
        how="left", validate="one_to_one")
    tg["is_isolated_event"] = tg.is_isolated_event.eq(True)
    report = {"status": "inspected_development", "folds": []}
    predictions = []
    for target in TARGETS:
        source_years = sorted(tg.loc[tg.season.lt(target), "season"].unique().tolist())
        previous = target - 10001
        inner_years = [year for year in source_years if year < previous]
        fold = {"target": target, "source_years": source_years, "models": {}}
        outer_pred = {}
        for label, use_prior in (("base", False), ("candidate", True)):
            inner_z, inner_x, inner_train, inner_valid, inner_audit = design(
                tg, pg, inner_years, previous, use_prior)
            assert inner_valid.sum() > 0 and inner_train.sum() > 0
            truth = inner_z.loc[inner_valid, "shots"].to_numpy(float)
            grid = {}
            for alpha in ALPHAS:
                estimate = fit_predict(inner_z, inner_x, inner_train, inner_valid, alpha)
                grid[str(alpha)] = float(np.mean((estimate-truth)**2))
            selected_alpha = min(ALPHAS, key=lambda a: (grid[str(a)], a))
            outer_z, outer_x, outer_train, outer_test, outer_audit = design(
                tg, pg, source_years, target, use_prior)
            outer_pred[label] = fit_predict(outer_z, outer_x, outer_train, outer_test,
                                            selected_alpha)
            fold["models"][label] = {"inner_mse": grid, "selected_alpha": selected_alpha,
                "inner_design": inner_audit, "outer_design": outer_audit}
        assert len(outer_pred["base"]) == len(outer_pred["candidate"])
        d = outer_z.loc[outer_test, ["game", "season", "team", "side", "seconds",
            "shots", "xg", "goals", "is_isolated_event"]].copy()
        d.rename(columns={"shots": "actual"}, inplace=True)
        d["base"] = outer_pred["base"]
        d["candidate"] = outer_pred["candidate"]
        d["form_only"] = (outer_z.loc[outer_test, "pregame_own_form"].to_numpy(float)
                          * d.seconds.to_numpy(float) / 3600)
        d.reset_index(drop=True, inplace=True)
        fold["all"] = scores(d.actual.to_numpy(), d.base.to_numpy(), d.candidate.to_numpy())
        fold["all"]["form_only_mse"] = float(np.mean((d.form_only-d.actual)**2))
        fold["all"]["game_bootstrap95"] = ci(d)
        subset = d[d.is_isolated_event]
        fold["changed_lineup"] = scores(subset.actual.to_numpy(), subset.base.to_numpy(), subset.candidate.to_numpy())
        fold["changed_lineup"]["form_only_mse"] = float(np.mean((subset.form_only-subset.actual)**2))
        fold["changed_lineup"]["game_bootstrap95"] = ci(subset)
        report["folds"].append(fold)
        predictions.append(d)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_parquet(OUT / "heldout_team_games.parquet", index=False)
    report["pooled_all"] = scores(pred.actual.to_numpy(), pred.base.to_numpy(), pred.candidate.to_numpy())
    report["pooled_all"]["form_only_mse"] = float(np.mean((pred.form_only-pred.actual)**2))
    report["pooled_all"]["game_bootstrap95"] = ci(pred)
    sub = pred[pred.is_isolated_event]
    report["pooled_changed_lineup"] = scores(sub.actual.to_numpy(), sub.base.to_numpy(), sub.candidate.to_numpy())
    report["pooled_changed_lineup"]["form_only_mse"] = float(np.mean((sub.form_only-sub.actual)**2))
    report["pooled_changed_lineup"]["game_bootstrap95"] = ci(sub)
    report["source_sha256"] = {str(p): sha(p) for p in (
        tg_path, pg_path, panel_path, MEASURE, SHOT_SHARE, A1_SHARE)}
    report["code_sha256"] = sha(Path(__file__))
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"folds": [{"target": f["target"], "all": f["all"],
        "changed_lineup": f["changed_lineup"]} for f in report["folds"]],
        "pooled_all": report["pooled_all"],
        "pooled_changed_lineup": report["pooled_changed_lineup"]}, indent=2))


if __name__ == "__main__":
    main()
