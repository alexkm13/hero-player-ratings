#!/usr/bin/env python3
"""Source-only A3Z actor production as an increment to HERO-conditioned team supply."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from research import hero_conditioned as hero
from research import team_supply as v3

from research.paths import ARCHIVE as ROOT, output

OUT = output("action_opportunity")
INCIDENCE = ROOT / 'outputs/a3z_actor_value_v1/action_incidence.parquet'
MICRO = ROOT / 'work/canonical_v1/linked_micro.parquet'
RISK = ROOT / 'work/canonical_v1/risk.parquet'
META = ROOT / 'work/canonical_v1/player_meta.parquet'
PREVIOUS = ROOT / 'outputs/challenger_hero_conditioned_v1/heldout_team_games.parquet'
ALPHA = 1000.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def profiles(incidence, seasons):
    """Use preceding two years only, centered in each source window/role."""
    assert not incidence.duplicated(['season_id','game','side','player_id','action_type']).any()
    pieces = []
    for target in seasons:
        source = [target - 10001, target - 20002]
        d = incidence[incidence.season_id.isin(source)].copy()
        assert d.season_id.max() < target
        roles = d.sort_values('season_id').drop_duplicates('player_id',keep='last')[
            ['player_id','actor_role']]
        g = d.groupby(['player_id','action_type'], as_index=False).agg(
            count=('count','sum'), hours=('hours','sum'))
        g = g.merge(roles,on='player_id',validate='many_to_one')
        ref = d[d.actor_role.isin(['F','D'])].groupby(['actor_role','action_type']).agg(
            count=('count','sum'), hours=('hours','sum'))
        ref['rate'] = ref['count'] / ref['hours']
        g = g.join(ref['rate'], on=['actor_role','action_type'])
        g['deviation'] = (g['count']/g['hours'] - g['rate']) * g['hours']/(g['hours']+5.)
        g['covered'] = (g.hours >= 1.).astype(float)
        g.loc[~g.actor_role.isin(['F','D']), ['deviation','covered']] = 0.
        assert np.isfinite(g[['deviation','covered']]).all().all()
        wide = g.pivot(index='player_id', columns='action_type', values=['deviation','covered'])
        wide.columns = [f'{a}_{b}' for a,b in wide.columns]
        wide = wide.reset_index().fillna(0)
        wide['season'] = target
        pieces.append(wide)
    z = pd.concat(pieces, ignore_index=True)
    assert not z.duplicated(['season','player_id']).any()
    return z


def early_incidence():
    """Apply the prior incidence panel's fixed completeness rule to 2021-22."""
    m = pd.read_parquet(MICRO,columns=['season_id','game','source_file','source_row',
        'action_type','player_id','side'])
    m = m[m.season_id.eq(20212022)&m.action_type.isin(['entry','retrieval'])]
    assert not m.duplicated(['game','source_file','source_row','action_type','player_id']).any()
    counts = m.groupby(['game','action_type']).size().unstack(fill_value=0)
    games = counts.index[(counts.entry>=20)&(counts.retrieval>=20)]
    m = m[m.game.isin(games)]
    r = pd.read_parquet(RISK,columns=['game','season','seconds','home_players',
        'away_players','home_team','away_team'])
    r = r[r.game.isin(games)]
    assert r.game.nunique()==len(games)
    parts = []
    for side,roster,team,opponent in ((0,'home_players','home_team','away_team'),
                                      (1,'away_players','away_team','home_team')):
        ids = np.stack(r[roster].to_numpy()).astype(np.int64)
        assert ids.shape[1]==5
        parts.append(pd.DataFrame({'season_id':np.repeat(r.season.to_numpy(int),5),
            'game':np.repeat(r.game.to_numpy(int),5),'side':side,'player_id':ids.ravel(),
            'team':np.repeat(r[team].to_numpy(str),5),
            'opponent':np.repeat(r[opponent].to_numpy(str),5),
            'seconds':np.repeat(r.seconds.to_numpy(float),5)}))
    e = pd.concat(parts,ignore_index=True).groupby(
        ['season_id','game','side','player_id','team','opponent'],as_index=False).seconds.sum()
    c = m.groupby(['season_id','game','side','player_id','action_type']).size().rename('count')
    panel = pd.concat([e.assign(action_type=a) for a in ('entry','retrieval')],ignore_index=True)
    panel = panel.merge(c.reset_index(),on=['season_id','game','side','player_id','action_type'],
                        how='left',validate='one_to_one')
    panel['count'] = panel['count'].fillna(0).astype(np.int16)
    assert int(panel['count'].sum())==len(m)
    panel['hours'] = panel.seconds/3600
    panel['home_attack'] = panel.side.eq(0).astype(np.int8)
    meta = pd.read_parquet(META,columns=['season_id','player_id','position_group'])
    panel = panel.merge(meta.rename(columns={'position_group':'actor_role'}),
                        on=['season_id','player_id'],how='left',validate='many_to_one')
    panel['actor_role'] = panel.actor_role.fillna('U')
    panel.to_parquet(OUT/'early_action_incidence.parquet',index=False)
    return panel


def action_design(selected, pg, train, profile_columns):
    keys = pd.MultiIndex.from_frame(selected[['game','side']])
    rows = pg[pg.season.isin(selected.season.unique())]
    own = keys.get_indexer(pd.MultiIndex.from_frame(rows[['game','side']]))
    opposite = rows[['game','side']].copy()
    opposite['side'] = 1 - opposite.side
    opp = keys.get_indexer(pd.MultiIndex.from_frame(opposite))
    assert (own >= 0).all() and (opp >= 0).all()
    share = rows.seconds.to_numpy(float) / (5*selected.seconds.to_numpy(float)[own])
    assert np.isfinite(share).all() and (share > 0).all()
    cols = []
    for index in (own, opp):
        for name in profile_columns:
            cols.append(np.bincount(index, weights=share*rows[name].to_numpy(float),
                                    minlength=len(selected)))
    z = np.column_stack(cols)
    scale = z[train].std(axis=0)
    z /= np.where(scale > 1e-8, scale, 1.)
    assert np.isfinite(z).all()
    return sparse.csr_matrix(z)


def ci(pred, seed=20261008, draws=2000):
    rng = np.random.default_rng(seed)
    groups = pred.groupby('game').delta.agg(['sum','count']).to_numpy(float)
    sampled = np.empty(draws)
    for j in range(draws):
        take = groups[rng.integers(len(groups),size=len(groups))]
        sampled[j] = take[:,0].sum()/take[:,1].sum()
    return np.quantile(sampled,[.025,.975]).tolist()


def main():
    q, replay_audit = hero.hero_profiles()
    team_path = v3.BASE/'team_games.parquet'
    player_path = v3.BASE/'player_games.parquet'
    tg = pd.read_parquet(team_path)
    tg = v3.attach_pregame_form(tg)
    tg = tg[tg.season.isin(hero.SEASONS)].copy()
    pg = pd.read_parquet(player_path)
    measure = pd.read_parquet(v3.MEASURE)
    pg = v3.source_prior_features(pg[pg.season.isin(hero.SEASONS)],measure)
    pg = hero.add_source_profiles(pg,q)
    incidence = pd.concat([early_incidence(),pd.read_parquet(INCIDENCE)],ignore_index=True)
    prof = profiles(incidence, hero.SEASONS)
    cols = ['deviation_entry','deviation_retrieval','covered_entry']
    assert set(cols).issubset(prof.columns)
    pg = pg.merge(prof[['season','player_id',*cols]],on=['season','player_id'],
                  how='left',validate='many_to_one')
    pg[cols] = pg[cols].fillna(0.)
    assert (pg.season > pg.source_season).all()
    frames, fold_reports = [], []
    for target in v3.TARGETS:
        source = [s for s in hero.SEASONS if s < target]
        selected,x0,train,test,audit = hero.make_design(tg,pg,source,target,
                                                        'hero_categories_shares')
        x1 = sparse.hstack([x0, action_design(selected,pg,train,cols)],format='csr')
        base = v3.fit_predict(selected,x0,train,test,ALPHA)
        candidate = v3.fit_predict(selected,x1,train,test,ALPHA)
        d = selected.loc[test,['game','season','team','side','seconds','shots','xg','goals']].copy()
        d['base'] = base
        d['candidate'] = candidate
        d['delta'] = (candidate-d.shots.to_numpy())**2 - (base-d.shots.to_numpy())**2
        d.reset_index(drop=True,inplace=True)
        frames.append(d)
        fold_reports.append({'target':target,'source':source,'design':audit,
            'n':len(d),'base_mse':float(np.mean((base-d.shots)**2)),
            'candidate_mse':float(np.mean((candidate-d.shots)**2)),
            'delta':float(d.delta.mean()),'game_bootstrap95':ci(d,seed=target),
            'base_observed_ratio':float(d.shots.sum()/base.sum()),
            'candidate_observed_ratio':float(d.shots.sum()/candidate.sum()),
            'profile_exposure_share':float((pg.loc[pg.season.eq(target),'seconds']*
                pg.loc[pg.season.eq(target),'covered_entry']).sum()/
                pg.loc[pg.season.eq(target),'seconds'].sum())})
        print('finished',target,fold_reports[-1]['delta'],flush=True)
    pred = pd.concat(frames,ignore_index=True)
    prior = pd.read_parquet(PREVIOUS)[['game','side','hero_categories_shares']]
    check = pred.merge(prior,on=['game','side'],validate='one_to_one')
    assert len(check)==len(pred)==len(prior)
    replay = check.groupby('season').apply(
        lambda d:float((d.base-d.hero_categories_shares).abs().max()),
        include_groups=False).to_dict()
    assert max(replay.values()) < 1e-6, replay
    gate = all(f['delta']<0 for f in fold_reports) and ci(pred)[1]<0
    report = {'status':'inspected_development','folds':fold_reports,
        'pooled':{'n':len(pred),'base_mse':float(np.mean((pred.base-pred.shots)**2)),
                  'candidate_mse':float(np.mean((pred.candidate-pred.shots)**2)),
                  'delta':float(pred.delta.mean()),'game_bootstrap95':ci(pred),
                  'primary_gate_pass':bool(gate)},
        'prior_baseline_difference_by_season':replay,
        'hero_replay':replay_audit,
        'sources':{str(p):sha(p) for p in (team_path,player_path,v3.MEASURE,
            INCIDENCE,MICRO,RISK,META,hero.HERO_INPUT,hero.HERO_WEIGHTS,
            hero.HERO_REPLAY,PREVIOUS)},
        'code_sha256':sha(Path(__file__))}
    pred.drop(columns='delta').to_parquet(OUT/'heldout_team_games.parquet',index=False)
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['pooled'],indent=2))


if __name__=='__main__':
    main()
