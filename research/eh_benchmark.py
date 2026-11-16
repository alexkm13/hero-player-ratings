"""Compare local Evolving-Hockey exports on a common historical player cohort."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hero.common import sha
from research.paths import ARCHIVE, output
from research.public_benchmark import rho, team

FULL_YEARS = [20212022, 20222023, 20232024, 20242025]
OUTCOMES = ["shot_net60", "xg_net60", "goal_net60"]
MODELS = ["hero_v8", "lb_rate", "ha_rate", "eh_war60", "eh_xwar60", "eh_war", "eh_xwar"]
KEY = ["player_id", "source_season"]
STINT = [*KEY, "team", "role"]


def normalize_role(s):
    if s in {"D", "LD", "RD"}:
        return "D"
    if s in {"C", "F", "L", "R", "LW", "RW", "W"} or "/" in str(s):
        return "F"
    return None


def provider_table(path, value):
    d = pd.read_csv(path)
    assert d.Season.str.fullmatch(r"\d{2}-\d{2}").all()
    first = 2000 + d.Season.str[:2].astype(int)
    assert ((first + 1) % 100 == d.Season.str[-2:].astype(int)).all()
    d = d.assign(source_season=first * 10001 + 1).rename(columns={"API ID": "player_id"})
    d = d[d.source_season.isin(FULL_YEARS)].copy()
    d["role"] = d.Position.map(normalize_role)
    d["team"] = d.Team.map(team)
    d = d[d.player_id.notna() & d.role.notna() & d.TOI_All.gt(0) & np.isfinite(d[value])]
    d.player_id = d.player_id.astype(int)
    d = d[STINT + ["TOI_All", value]].drop_duplicates()
    assert not d.duplicated(STINT).any(), "Conflicting duplicate EH team stint."
    return d


def build_sources(archive, war, xwar):
    p = archive / "outputs/hero_competitor_benchmark_v1"
    source = pd.read_parquet(p / "source_scores.parquet")
    pg = pd.read_parquet(archive / "outputs/changed_lineup_v1/player_games.parquet")
    club_sets = (pg.assign(team=pg.team.map(team)).groupby(["player_id", "season"])
                 .team.agg(frozenset).rename("canonical_stints").rename_axis(KEY))
    eh = provider_table(war, "WAR").merge(
        provider_table(xwar, "xWAR"), on=STINT, how="outer",
        validate="one_to_one", suffixes=("", "_x"), indicator=True)
    # Every GAR stint needs the corresponding xGAR stint, TOI and stable role.
    eh["stint_valid"] = eh._merge.eq("both") & np.isclose(eh.TOI_All, eh.TOI_All_x)
    e = eh.groupby(KEY).agg(
        eh_role=("role", "first"), role_count=("role", "nunique"),
        eh_stints=("team", frozenset), complete=("stint_valid", "all"),
        minutes=("TOI_All", "sum"), eh_war=("WAR", "sum"), eh_xwar=("xWAR", "sum"))
    e = e.join(club_sets)
    e = e[e.complete & e.role_count.eq(1) & e.canonical_stints.eq(e.eh_stints)].reset_index()
    e["eh_war60"] = e.eh_war * 60 / e.minutes
    e["eh_xwar60"] = e.eh_xwar * 60 / e.minutes
    z = source.merge(e[KEY + ["eh_role", *MODELS[3:]]], on=KEY, validate="one_to_one")
    z = z[z.position_group.eq(z.eh_role)].copy()
    assert not z.duplicated(KEY).any()
    return z, pg, p


def coverage(z, pg):
    pg = pg[pg.season.sub(10001).isin(FULL_YEARS)].copy()
    pg["source_season"] = pg.season - 10001
    p = pg.merge(z[KEY].assign(has_score=True), on=KEY, how="left", validate="many_to_one")
    p["covered_seconds"] = p.seconds * p.has_score.eq(True)
    a = p.groupby(["season", "game", "side"])[["seconds", "covered_seconds"]].sum()
    a["fraction"] = a.covered_seconds / a.seconds
    g = a.groupby(["season", "game"]).fraction.min()
    return [{"season": int(y), "games": len(s), "at_80_percent": int(s.ge(.80).sum()),
             "median_minimum_side_coverage": float(s.median())}
            for y, s in g.groupby(level="season")]


def persistence(z, path):
    ice = pd.read_csv(path, usecols=["nhl_api_id", "season", "position", "games_played", "avg_toi"])
    ice = ice.rename(columns={"nhl_api_id": "player_id", "season": "source_season"})
    assert not ice.duplicated(KEY).any()
    ice["position_group"] = ice.position.map(normalize_role)
    ice = ice[ice.source_season.isin(FULL_YEARS) & ice.position_group.notna()].copy()
    ice["all_seconds"] = ice.games_played * ice.avg_toi
    ice = ice.sort_values(["all_seconds", "player_id"], ascending=[False, True])
    ix = ice.groupby(["source_season", "position_group"]).cumcount()
    top = ice.loc[ix.lt(ice.position_group.map({"F": 384, "D": 192})),
                  KEY + ["position_group"]]
    m = z.merge(top, on=KEY + ["position_group"], validate="one_to_one")
    # Standardization uses that season's common eligible snapshot, before joining pairs.
    cells = m.groupby(["source_season", "position_group"])[MODELS]
    m[MODELS] = (m[MODELS] - cells.transform("mean")) / cells.transform("std")
    later = m[KEY + ["position_group", *MODELS]].copy()
    later["source_season"] -= 10001
    pair = m.merge(later, on=KEY + ["position_group"], suffixes=("", "_next"), validate="one_to_one")
    ret = []
    for r in ("F", "D"):
        p = pair[pair.position_group.eq(r)]
        seasons = p.groupby("source_season").size().to_dict()
        for name, g in [("pooled", p), *p.groupby("source_season")]:
            for arm in MODELS:
                correlation = float(g[arm].corr(g[arm + "_next"]))
                ret.append(dict(role=r, fold=str(name), model=arm, n=len(g),
                                pearson=correlation, r_squared=correlation ** 2,
                                pair_counts_by_source=seasons))
    return ret


def process(z, targets):
    p = z.merge(targets, on=KEY, validate="one_to_one")
    p = p[p.target_seconds.ge(18_000) & p.target_season.isin([20242025, 20252026])].copy()
    p["mover"] = (p.source_team.ne("MULTI") & p.target_team.ne("MULTI")
                  & p.source_team.ne(p.target_team))
    assert (p.target_season == p.source_season + 10001).all()
    output = []
    for subset, sub in [("all", p), ("clean_movers", p[p.mover])]:
        group = sub.groupby(["target_season", "position_group"])
        for target in OUTCOMES:
            for arm in MODELS:
                cells = [dict(season=int(y), role=r, n=len(g), rho=rho(g[arm], g[target]))
                         for (y, r), g in group]
                v = [c["rho"] for c in cells if c["rho"] is not None]
                # Small cells are suppressed rather than silently averaged over fewer folds.
                score = float(np.mean(v)) if len(v) == 4 else None
                output.append(dict(subset=subset, model=arm, endpoint=target,
                                   n=len(sub), unique_players=int(sub.player_id.nunique()),
                                   equal_role_season_rho=score, cells=cells))
    return p, output


def primary_intervals(p):
    ret = []
    for subset, s in [("all", p), ("clean_movers", p[p.mover])]:
        s = s.reset_index(drop=True)
        sims = {k: [] for k in ["eh_war60", "eh_xwar60", "lb_rate", "ha_rate"]}
        sizes = s.groupby(["target_season", "position_group"]).size()
        if len(sizes) != 4 or sizes.min() < 10:
            ret.extend(dict(subset=subset, competitor=k, valid_draws=0, interval=None,
                            note="Observed role/season cell below the minimum size") for k in sims)
            continue
        ids = list(s.groupby("player_id").indices.values())
        rng = np.random.default_rng(20261116)
        for _ in range(2000):
            r = s.iloc[np.concatenate([ids[i] for i in rng.integers(0, len(ids), size=len(ids))])]
            ds = {k: [] for k in sims}
            for _, g in r.groupby(["target_season", "position_group"]):
                a = rho(g.hero_v8, g.shot_net60)
                for k in sims:
                    b = rho(g[k], g.shot_net60)
                    if a is not None and b is not None:
                        ds[k].append(a - b)
            for k, values in ds.items():
                if len(values) == 4:
                    sims[k].append(float(np.mean(values)))
        for k, x in sims.items():
            ret.append(dict(subset=subset, competitor=k, valid_draws=len(x),
                            interval=np.quantile(x, [.025, .975]).tolist() if x else None))
    return ret


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--war", type=Path, required=True, help="Local EH GAR export with actual WAR.")
    ap.add_argument("--xwar", type=Path, required=True, help="Local EH xGAR export with actual xWAR.")
    args = ap.parse_args()
    dst = output("eh_benchmark")
    if (dst / "evaluation.json").is_file():
        raise FileExistsError("Use a fresh HERO_ARTIFACTS_ROOT for a new benchmark.")
    p = args.archive / "outputs/hero_competitor_benchmark_v1"
    t = p / "player_targets.parquet"
    h = p / "inputs/hockey_alchemy_player_seasons_2013_2026.csv"
    files = [args.war, args.xwar, t, h, p / "source_scores.parquet",
             args.archive / "outputs/changed_lineup_v1/player_games.parquet", Path(__file__)]
    hashes = {str(f): sha(f) for f in files}
    z, pg, _ = build_sources(args.archive, args.war, args.xwar)
    rows, metrics = process(z, pd.read_parquet(t))
    result = dict(
        scope="historical retrospective, no 2026–27 outcomes; no HockeyStats player inputs",
        source_hashes=hashes,
        common_sources=z.groupby("source_season").size().to_dict(),
        team_coverage=coverage(z, pg), player_process=metrics,
        player_intervals95=primary_intervals(rows), persistence=persistence(z, h))
    assert all(sha(f) == v for f, v in hashes.items())
    assert set(rows.target_season) == {20242025, 20252026}
    (dst / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote matched comparison: {len(rows)} player-seasons to {dst}")


if __name__ == "__main__":
    main()
