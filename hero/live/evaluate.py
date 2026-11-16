"""Fit 5v5 process proxies and select a chronological exposure-based HERO overlay."""
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
from hero.overall import pool_as_of, category_panel
from scipy import linalg,sparse
from scipy.stats import spearmanr

from hero.live.prepare import WORK,OUT,RAW
from hero.common import DATA, sha

V8 = DATA / 'hero_v8'
TUNED = V8 / 'tuned_categories.parquet'
WEIGHTS = V8 / 'weights_v7.json'
OLD = V8 / 'current_ratings.csv'
HIST=[y*10000+y+1 for y in range(2021,2026)]
NOW=20262027
K={'zero':None, 'k7200':7200, 'k3600':3600, 'k1800':1800, 'k900':900, 'k450':450}
ROLES=['F','D']


def dump(p,x):
    (OUT/p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def estimate(d):
    n=len(d)
    h,a=[np.stack(d[col].to_numpy()).astype('int64') for col in ['home_players','away_players']]
    ids=pd.Index(np.unique(np.r_[h.ravel(),a.ravel()]))
    teams=pd.Index(np.unique(np.r_[d.home_team.to_numpy(),d.away_team.to_numpy()]))
    p,t=len(ids),len(teams)
    index=np.column_stack([ids.get_indexer(h.ravel()).reshape(n,5),
          ids.get_indexer(a.ravel()).reshape(n,5),p+teams.get_indexer(d.home_team),p+teams.get_indexer(d.away_team)])
    x=sparse.coo_matrix((np.tile(np.r_[np.ones(5),-np.ones(5),1.,-1.],n),
               (np.repeat(np.arange(n),12),index.ravel())),shape=(n,p+t)).tocsr()
    score,period=d.score.to_numpy(),d.period.to_numpy()
    context=np.column_stack([np.ones(n),*[score==j for j in [-2,-1,1,2]],period==2,period==3])
    X=sparse.hstack([x,sparse.csr_matrix(context)],format='csr')
    htime=d.seconds.to_numpy()/3600.
    count=d.home_shots.to_numpy(dtype=float)-d.away_shots.to_numpy(dtype=float)
    normal=(X.T@X.multiply(htime[:,None])).toarray()
    penalty=np.r_[np.full(p,3.),np.full(t,18.),0.,np.full(6,.1)]
    normal.flat[::len(penalty)+1]+=penalty
    rhs=np.asarray(X.T@count).ravel()
    beta=linalg.solve(normal,rhs,assume_a='pos')
    res=np.linalg.norm(np.einsum('ij,j->i',normal,beta)-rhs)/max(1.,np.linalg.norm(rhs))
    assert res<1e-10,res
    exposed=np.r_[h.ravel(),a.ravel()]
    weights=np.tile(np.repeat(htime*60,5),2)
    minutes=np.bincount(ids.get_indexer(exposed),weights=weights,minlength=p)
    assert np.isclose(minutes.sum(),10*60*htime.sum(),rtol=1e-12)
    return pd.DataFrame({'player_id':ids,'minutes':minutes,'shot_rapm':beta[:p]}),float(res)


def priors():
    weights = json.loads(WEIGHTS.read_text())
    seasons = pd.read_parquet(TUNED)
    prior = pd.concat([
        pool_as_of(category_panel(seasons, weights, target - 10001), target - 10001)
        .rename(columns={"current_ability_raw": "hero_raw", "window_toi": "toi"})
        .assign(target=target, prior_end=target - 10001)
        for target in HIST + [NOW]
    ], ignore_index=True)
    ref=pd.read_csv(OLD)[['player_id','current_ability_raw','current_ability_pct']]
    check=ref.merge(prior[prior.target.eq(NOW)],on='player_id',validate='one_to_one')
    assert len(check)==len(ref)
    err=float(np.max(np.abs(check.current_ability_raw-check.hero_raw)))
    assert err<1e-10,err
    return prior,err


def calibrate(train, target):
    assert (train.target<target).all() and not train.empty
    d={}
    for role in ROLES:
        sub=train[train.position_group.eq(role)]
        assert len(sub)>75
        b={}
        for x in ['hero_raw','early_shot_rapm']:
            a=np.column_stack([np.ones(len(sub)),sub[x].to_numpy(float)])
            coef,*_=np.linalg.lstsq(a,sub.later_shot_rapm.to_numpy(float),rcond=None)
            b[x]=coef.tolist()
        d[role]=b
    return d


def infer(rows, coefficients):
    d=rows.copy()
    for name,x in [('prior_proxy','hero_raw'),('early_proxy','early_shot_rapm')]:
        d[name]=np.nan
        for role in ROLES:
            keep=d.position_group.eq(role)
            b=coefficients[role][x]
            d.loc[keep,name]=b[0]+b[1]*d.loc[keep,x]
    return d


def blend(d, arm):
    eligible=d.early_minutes.ge(75)&d.early_proxy.notna()&d.prior_proxy.notna()
    k=K[arm]
    m=d.early_minutes.to_numpy(float)
    w=np.where(eligible,np.minimum(.30,m/(m+k)),0.) if k is not None else np.zeros(len(d))
    early=d.early_proxy.fillna(d.prior_proxy).to_numpy(float)
    before=d.prior_proxy.to_numpy(float)
    return before+np.where(w>0,w*(early-before),0.),w


def block_mse(d,col):
    return float(((d[col]-d.later_shot_rapm)**2).groupby([d.target,d.position_group]).mean().mean())


def select(d):
    scores={k:block_mse(d,k) for k in K}
    return min(scores,key=scores.get),scores


def cluster(d,col,draws=4000):
    diff=(d[col]-d.later_shot_rapm)**2-(d.zero-d.later_shot_rapm)**2
    groups=pd.DataFrame({'player':d.player_id,'target':d.target,'role':d.position_group,
                         'diff':diff}).groupby(['player','target','role'])['diff'].agg(['sum','count'])
    sums=groups['sum'].unstack(['target','role'],fill_value=0.)
    counts=groups['count'].unstack(['target','role'],fill_value=0.)
    assert sums.index.equals(counts.index) and sums.columns.equals(counts.columns)
    a,b=sums.to_numpy(),counts.to_numpy()
    rg=np.random.default_rng(20261116)
    bs=[]
    for _ in range(draws//200):
        idx=rg.integers(len(a),size=(200,len(a)))
        bs.extend((a[idx].sum(axis=1)/b[idx].sum(axis=1)).mean(axis=1).tolist())
    return {'point':float(block_mse(d,col)-block_mse(d,'zero')),
            'ci95':np.quantile(bs,[.025,.975]).tolist(),'clusters':len(a)}


def metrics(d,col):
    return {role:{'rows':len(g),'mse':float(np.mean((g[col]-g.later_shot_rapm)**2)),
                  'spearman':float(spearmanr(g[col],g.later_shot_rapm).statistic)}
            for role,g in d.groupby('position_group')}


def main():
    if (OUT/'evaluation.json').exists():
        raise RuntimeError('Completed experiment exists; preserve it.')
    data_files=[WORK/f'{kind}_{s}.parquet' for s in HIST+[NOW] for kind in ['intervals','games']]
    manifest={str(p):sha(p) for p in [*data_files,TUNED,WEIGHTS,OLD,Path(__file__),
                                      Path(__file__).with_name('prepare.py'),OUT/'coverage.json',OUT/'bulk_manifest.json',OUT/'schedule.json']}
    dump('run_manifest.json',{'started_utc':datetime.now(timezone.utc).isoformat(),
         'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
         'pandas':pd.__version__,'scipy':scipy.__version__,'files':manifest})
    prior,err=priors()
    panels=[]
    fits=[]
    now=None
    for season in HIST+[NOW]:
        r=pd.read_parquet(WORK/f'intervals_{season}.parquet')
        a=pd.read_parquet(WORK/f'games_{season}.parquet')
        assert r.season.eq(season).all()
        early=a[a.phase.eq('early')]
        late=a[a.phase.eq('later')]
        assert len(late)==0 if season==NOW else early.date.max()<late.date.min()
        assert set(early.game).isdisjoint(late.game) and len(early)+len(late)==r.game.nunique()
        e,eerr=estimate(r[r.game.isin(early.game)])
        e=e.rename(columns={'minutes':'early_minutes','shot_rapm':'early_shot_rapm'})
        season_prior=prior[prior.target.eq(season)]
        if season==NOW:
            now=season_prior.merge(e,on='player_id',how='outer',validate='one_to_one')
            now['target']=season
            now['early_minutes']=now.early_minutes.fillna(0)
            fits.append({'season':season,'early_games':len(early),'cutoff':early.date.max(),'early_normal_residual':eerr})
        else:
            l,lerr=estimate(r[r.game.isin(late.game)])
            l=l.rename(columns={'minutes':'later_minutes','shot_rapm':'later_shot_rapm'})
            panel=season_prior.merge(e,on='player_id',validate='one_to_one').merge(l,on='player_id',validate='one_to_one')
            panel=panel[panel.early_minutes.ge(75)&panel.later_minutes.ge(400)]
            panels.append(panel)
            fits.append({'season':season,'early_games':len(early),'later_games':len(late),'cutoff':early.date.max(),
                         'early_normal_residual':eerr,'later_normal_residual':lerr,'eligible_players':len(panel)})
        print('Shot process measurements complete',fits[-1],flush=True)
    assert now is not None
    panel=pd.concat(panels,ignore_index=True)
    forward=[]
    report=[]
    calibrators={}
    for s in HIST[1:]:
        train=panel[panel.target.lt(s)]
        c=calibrate(train,s)
        calibrators[str(s)]=c
        d=infer(panel[panel.target.eq(s)],c)
        for k in K:
            d[k],d[k+'_weight']=blend(d,k)
        d['early_only']=d.early_proxy
        assert np.array_equal(d.zero.to_numpy(),d.prior_proxy.to_numpy())
        fold={'target':s,'calibration_seasons':sorted(train.target.unique().tolist()),'early_rows':len(d),
              'median_early_minutes':float(d.early_minutes.median()),
              'fixed_fit':{k:{'mse':block_mse(d,k),'by_role':metrics(d,k),
                             'median_weight':float(d[k+'_weight'].median())} for k in K},
              'early_only_mse':block_mse(d,'early_only')}
        if forward:
            arm,selection=select(pd.concat(forward,ignore_index=True))
            d['nested']=d[arm]
            d['nested_weight']=d[arm+'_weight']
            fold.update({'nested_arm':arm,'selection_mses':selection,'nested_mse':block_mse(d,'nested'),
                'nested_delta':block_mse(d,'nested')-block_mse(d,'zero'),'nested_by_role':metrics(d,'nested'),
                'nested_median_weight':float(d.nested_weight.median())})
        forward.append(d)
        report.append(fold)
    pred=pd.concat(forward,ignore_index=True)
    nested=pred[pred.nested.notna()]
    cl=cluster(nested,'nested')
    arm,selection=select(pred)
    by_role={r:float(block_mse(nested[nested.position_group.eq(r)],'nested')-
                     block_mse(nested[nested.position_group.eq(r)],'zero')) for r in ROLES}
    gate=(sum(f.get('nested_delta',0)<0 for f in report)>=2 and cl['ci95'][1]<0
          and all(d<=0 for d in by_role.values()) and arm!='zero')
    rec=arm if gate else 'zero'
    c=calibrate(panel,NOW)
    calibrators[str(NOW)]=c
    short=infer(now,c)
    short['research_shot_proxy'],short['recommended_weight']=blend(short,rec)
    short['selection_only_proxy'],short['selection_only_weight']=blend(short,arm)
    # The live file is anchored to the exact published rating, never re-percentiled.
    exact=pd.read_csv(OLD)[['player_id','player_name','current_ability_raw','current_ability_pct','league_rank']].rename(columns={'player_name':'card_name'})
    live=exact.merge(short,on='player_id',how='left',validate='one_to_one')
    assert (live.current_ability_raw-live.hero_raw).abs().max()<1e-10
    eligible=live[live.early_minutes.ge(75)]
    assert eligible.prior_proxy.notna().all()
    assert live.loc[live.early_minutes.lt(75),'recommended_weight'].eq(0).all()
    mstats={'cards':len(live),'at_least_75_minutes':len(eligible),'any_5v5_minutes':int(live.early_minutes.gt(0).sum()),
         'players_with_process_data':int(now.early_minutes.gt(0).sum()),
         'process_players_without_HERO_prior':int((now.early_minutes.gt(0)&now.hero_raw.isna()).sum()),
         'median_minutes_eligible_cards':float(eligible.early_minutes.median()),
         'median_recommended_weight':float(eligible.recommended_weight.median()),
         'median_selection_only_weight':float(eligible.selection_only_weight.median()),
         'selected_weight_at_minutes':{str(m): float(min(.30,m/(m+K[arm]))) if K[arm] is not None else 0. for m in [75,150,250,300,400]},
         'recommended_weight_quantiles_eligible':{str(k):float(v) for k,v in eligible.recommended_weight.quantile([.1,.5,.9]).items()}}
    assert all(sha(p)==h for p,h in manifest.items())
    assert not pred.target.eq(NOW).any()
    panel.to_parquet(OUT/'historical_cohorts.parquet',index=False)
    pred.to_parquet(OUT/'historical_predictions.parquet',index=False)
    short.to_parquet(OUT/'current_player_signal.parquet',index=False)
    live.sort_values('league_rank').to_csv(OUT/'current_card_shadow.csv',index=False)
    dump('calibrators.json',calibrators)
    dump('verification.json',{'status':'PASS','max_hero_card_replay_error':err,
         'no_current_season_labels_in_training_or_validation':True,'chronology_and_exposure_accounting':True,
         'source_hashes_unchanged':True,'zero_arm_identical_to_prior':True,
         'all_phase_normal_equations_relative_residual_below_1e_10':True})
    outcome={'primary':'equal-role, equal-season MSE against disjoint later-season adjusted net unblocked shots/60',
        'process_fits':fits,'folds':report,'nested':{**cl,'prior_mse':block_mse(nested,'zero'),
        'candidate_mse':block_mse(nested,'nested'),'role_delta':by_role},'legal_oos_fixed_arm_mse':selection,
        'selection_only_arm':arm,'gate_pass':bool(gate),'recommended_research_arm':rec,'current':mstats}
    dump('evaluation.json',outcome)
    print(json.dumps({k:outcome[k] for k in ['nested','legal_oos_fixed_arm_mse','selection_only_arm',
               'gate_pass','recommended_research_arm','current']},indent=2),flush=True)


if __name__=='__main__':
    main()
