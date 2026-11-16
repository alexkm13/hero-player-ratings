"""Matched forward-only process adapters for fixed HERO and actual public ratings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from hero.common import team as normalize_team
from research.paths import ARCHIVE as ROOT, output

OUT = output("public_benchmark")
INPUTS = ROOT / "outputs/hero_competitor_benchmark_v1"
MODELS = ["hero_v8", "ha_rate", "lb_rate", "ha_total", "lb_total"]


def team(value):
    code = normalize_team(str(value).strip())
    return "MTL" if code == "MON" else code


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
from research.team_supply import attach_pregame_form

ALPHAS=(.01,.1,1.,10.,100.,1000.)
TARGETS=(20242025,20252026)
SEASONS=(20222023,20232024,20242025,20252026)
ARMS=['home','form',*MODELS]
BASE=['is_home','pregame_own_form','pregame_opponent_form','own_F_coverage','own_D_coverage','opp_F_coverage','opp_D_coverage']

def rho(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);good=np.isfinite(a)&np.isfinite(b)
    a=a[good];b=b[good]
    if len(a)<10 or np.std(a)<1e-12 or np.std(b)<1e-12:return None
    x=rankdata(a);y=rankdata(b)
    return float(np.corrcoef(x,y)[0,1])

def process_panel(tg,pg,scores):
    """No outcome values select players, coverage, movers or lineup score features."""
    tg=tg.copy();pg=pg.copy()
    assert not tg.duplicated(['game','side']).any()
    assert not pg.duplicated(['game','side','player_id']).any()
    assert tg.groupby('game').side.agg(set).map(lambda s:s=={0,1}).all()
    tg=attach_pregame_form(tg)
    tg=tg[tg.season.isin(SEASONS+(20262027,))].copy()
    pg=pg[pg.season.isin(tg.season.unique())].copy()
    pg['source_season']=pg.season-10001
    use=['player_id','source_season','position_group','source_team',*[f'z_{m}' for m in MODELS]]
    p=pg.merge(scores[use],on=['player_id','source_season'],how='left',validate='many_to_one')
    assert p.source_season.lt(p.season).all()
    denom=tg[['game','side','seconds']].rename(columns={'seconds':'game_seconds'})
    p=p.merge(denom,on=['game','side'],validate='many_to_one')
    p['share']=p.seconds/(5*p.game_seconds)
    assert p.share.gt(0).all()
    assert np.max(np.abs(p.groupby(['game','side']).share.sum()-1))<1e-7
    p['available']=p.z_hero_v8.notna()
    assert np.array_equal(p.available.to_numpy(),p[[f'z_{m}' for m in MODELS]].notna().all(axis=1).to_numpy())
    agg=[]
    for r in ('F','D'):
        coverage=p.share*p.position_group.eq(r)*p.available
        agg.append(coverage.rename(f'own_{r}_coverage'))
        for m in MODELS:
            agg.append((coverage*p[f'z_{m}'].fillna(0.)).rename(f'own_{r}_{m}'))
    a=pd.concat([p[['game','side']],*agg],axis=1).groupby(['game','side'],as_index=False).sum()
    z=tg.merge(a,on=['game','side'],validate='one_to_one')
    z['own_coverage']=z.own_F_coverage+z.own_D_coverage
    cols=[c for c in a if c not in ('game','side')]
    opp=a.copy();opp.side=1-opp.side
    opp.rename(columns={c:c.replace('own_','opp_',1) for c in cols},inplace=True)
    z=z.merge(opp,on=['game','side'],validate='one_to_one')
    z['is_home']=(z.side==0).astype(float)
    z['min_game_coverage']=z.groupby('game').own_coverage.transform('min')
    z['included']=z.min_game_coverage.ge(.80)
    # Full-season exposure defines clean movers independently of who scores or receives credit.
    clubs=pg.assign(code=pg.team.map(team)).groupby(['player_id','season']).code.agg(lambda x:x.iloc[0] if x.nunique()==1 else 'MULTI').rename('target_team').reset_index()
    p=p.merge(clubs,on=['player_id','season'],validate='many_to_one')
    p['clean_mover']=p.available & p.source_team.notna() & p.source_team.ne('MULTI') & p.target_team.ne('MULTI') & p.source_team.ne(p.target_team)
    mover=p[p.clean_mover & p.seconds.ge(480)].groupby(['game','side']).agg(mover_count=('player_id','size'),mover_id=('player_id','first'),mover_role=('position_group','first')).reset_index()
    z=z.merge(mover,on=['game','side'],how='left',validate='one_to_one')
    z['one_mover']=z.mover_count.eq(1)
    return z,p

def columns(arm):
    if arm=='home':return ['is_home']
    if arm=='form':return BASE
    return BASE+[f'{s}_{r}_{arm}' for s in ('own','opp') for r in ('F','D')]

def fit_model(d,train,arm,endpoint,alpha):
    cols=columns(arm)
    x=d.loc[train,cols].to_numpy(float)
    h=d.loc[train,'seconds'].to_numpy(float)/3600
    y=d.loc[train,endpoint].to_numpy(float)/h
    mu=np.average(x,axis=0,weights=h)
    std=np.sqrt(np.average((x-mu)**2,axis=0,weights=h));std=np.where(std>1e-8,std,1.)
    x=np.column_stack([np.ones(len(x)),(x-mu)/std])
    lhs=np.einsum('ni,nj,n->ij',x,x,h)
    pen=np.eye(x.shape[1])*alpha;pen[0,0]=0
    beta=np.linalg.solve(lhs+pen,np.einsum('ni,n,n->i',x,y,h))
    assert np.isfinite(beta).all()
    return {'columns':cols,'mean':mu.tolist(),'scale':std.tolist(),'beta':beta.tolist(),'alpha':alpha,'endpoint':endpoint,'max_training_target_season':int(d.loc[train,'season'].max())}

def predict(model,d):
    x=(d[model['columns']].to_numpy(float)-np.array(model['mean']))/np.array(model['scale'])
    rate=np.array(model['beta'])[0]+np.einsum('ni,i->n',x,np.array(model['beta'])[1:])
    return np.maximum(rate,0)*d.seconds.to_numpy(float)/3600

def select_model(d,target,arm,endpoint):
    previous=target-10001
    inner_train=d.season.lt(previous);valid=d.season.eq(previous)
    assert inner_train.any() and valid.any()
    grid={}
    for alpha in ALPHAS:
        m=fit_model(d,inner_train,arm,endpoint,alpha)
        grid[str(alpha)]=float(np.mean((predict(m,d[valid])-d.loc[valid,endpoint].to_numpy(float))**2))
    chosen=min(ALPHAS,key=lambda a:(grid[str(a)],-a))
    m=fit_model(d,d.season.lt(target),arm,endpoint,chosen)
    m.update(inner_validation_season=previous,inner_grid=grid)
    assert m['max_training_target_season']<target
    return m

def metrics(d,arm):
    y=d.actual.to_numpy(float);p=d[arm].to_numpy(float)
    if not len(y):return {'n':0,'games':0,'mse':None,'mae':None,'observed_predicted':None}
    return {'n':len(y),'games':int(d.game.nunique()),'mse':float(np.mean((p-y)**2)),
        'mae':float(np.mean(np.abs(p-y))),'observed_predicted':float(y.sum()/p.sum()) if p.sum()>0 else None}

def intervals(d,competitor,cluster='game',level=.975,seed=20261029):
    groups=[]
    for _,g in d.groupby('season'):
        g=g.copy();g['delta']=(g.hero_v8-g.actual)**2-(g[competitor]-g.actual)**2
        if cluster=='mover_id':g=g.dropna(subset=['mover_id'])
        if not len(g):continue
        a=g.groupby(cluster).delta.agg(['sum','count']).to_numpy(float);groups.append(a)
    if not groups:return None
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(2000):
        vals=[]
        for a in groups:
            b=a[rng.integers(len(a),size=len(a))];vals.append(b[:,0].sum()/b[:,1].sum())
        draws.append(np.mean(vals))
    tail=(1-level)/2
    return {'level':level,'cluster':cluster,'delta_mse':float(np.mean([a[:,0].sum()/a[:,1].sum() for a in groups])),
            'ci':np.quantile(draws,[tail,1-tail]).tolist(),'draws':2000}

def player_summary(scores,targets,target_seasons=TARGETS,output_dir=OUT):
    p=scores.merge(targets,on=['player_id','source_season'],validate='one_to_one')
    p=p[p.target_seconds.ge(18000)&p.target_season.isin(target_seasons)].copy()
    p['clean_mover']=p.source_team.ne('MULTI')&p.target_team.ne('MULTI')&p.source_team.ne(p.target_team)
    if p.empty:
        p.to_parquet(output_dir/'matched_player_rows.parquet',index=False)
        return {'cells':[],'equal_role_season_spearman':[],'player_cluster_intervals':[],'n':0,'movers':0}
    cells=[]
    for (season,role),g in p.groupby(['target_season','position_group']):
        for slice_name,s in [('all',g),('clean_movers',g[g.clean_mover])]:
            for endpoint in ['shot_net60','xg_net60','goal_net60']:
                for m in MODELS:
                    cells.append({'target_season':int(season),'role':role,'slice':slice_name,'endpoint':endpoint,'model':m,'n':len(s),'spearman':rho(s[m],s[endpoint])})
    c=pd.DataFrame(cells)
    c.to_csv(output_dir/'player_cells.csv',index=False);p.to_parquet(output_dir/'matched_player_rows.parquet',index=False)
    summary_frame=c.groupby(['slice','endpoint','model']).spearman.mean().reset_index()
    summary=summary_frame.astype(object).where(summary_frame.notna(),None).to_dict('records')
    cis=[]
    for slice_name,s in [('all',p),('clean_movers',p[p.clean_mover])]:
        groups={pid:g.index.to_numpy() for pid,g in s.groupby('player_id')};players=np.array(list(groups))
        if not len(players):
            for m in ('ha_rate','lb_rate'):
                cis.append({'slice':slice_name,'competitor':m,'endpoint':'shot_net60','n':0,'players':0,'delta_ci95':None,'valid_draws':0})
            continue
        rng=np.random.default_rng(20261029)
        sims={m:[] for m in ('ha_rate','lb_rate')}
        for _ in range(2000):
            idx=np.concatenate([groups[i] for i in rng.choice(players,len(players),replace=True)])
            g=s.loc[idx];vals={m:[] for m in sims}
            for _,h in g.groupby(['target_season','position_group']):
                a=rho(h.hero_v8,h.shot_net60)
                for m in vals:
                    b=rho(h[m],h.shot_net60)
                    if a is not None and b is not None:vals[m].append(a-b)
            for m in sims:
                if len(vals[m])==2*len(target_seasons):sims[m].append(np.mean(vals[m]))
        for m,a in sims.items():
            cis.append({'slice':slice_name,'competitor':m,'endpoint':'shot_net60','n':len(s),'players':len(players),'delta_ci95':np.quantile(a,[.025,.975]).tolist() if len(a) else None,'valid_draws':len(a)})
    return {'cells':cells,'equal_role_season_spearman':summary,'player_cluster_intervals':cis,'n':len(p),'movers':int(p.clean_mover.sum())}

def aggregate_teams(pred,output_dir=OUT):
    rows=[]
    for endpoint,g in pred.groupby('endpoint'):
        opp=g[['game','side','actual',*ARMS]].copy();opp.side=1-opp.side
        opp.rename(columns={c:'opp_'+c for c in ['actual',*ARMS]},inplace=True)
        s=g.merge(opp,on=['game','side'],validate='one_to_one')
        for (season,club),a in s.groupby(['season','team']):
            row={'endpoint':endpoint,'season':int(season),'team':club,'games':len(a),'hours':float(a.seconds.sum()/3600)}
            for c in ['actual',*ARMS]:row[c]=float((a[c]-a['opp_'+c]).sum()/row['hours'])
            rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(output_dir/'team_season_aggregation.csv',index=False)
    cells=[]
    for (endpoint,season),g in d.groupby(['endpoint','season']):
        for m in ARMS:
            y=g.actual.to_numpy();pr=g[m].to_numpy()
            cells.append({'endpoint':endpoint,'season':int(season),'model':m,'teams':len(g),'net60_spearman':rho(pr,y),
                'net60_mae':float(np.mean(np.abs(pr-y))),'net60_r2_score':float(1-np.sum((pr-y)**2)/np.sum((y-y.mean())**2))})
    return cells

def main():
    assert not (OUT/'evaluation.json').exists(), 'Do not overwrite completed run.'
    scores=pd.read_parquet(INPUTS/'source_scores.parquet')
    tg=pd.read_parquet(ROOT/'outputs/changed_lineup_v1/team_games.parquet')
    pg=pd.read_parquet(ROOT/'outputs/changed_lineup_v1/player_games.parquet')
    assert tg.season.max()==pg.season.max()==20252026
    z,_=process_panel(tg,pg,scores)
    z[['game','season','team','side','own_coverage','min_game_coverage','included']].to_csv(OUT/'game_coverage.csv',index=False)
    excluded=z[~z.included];excluded.to_parquet(OUT/'excluded_team_games.parquet',index=False)
    d=z[z.included].copy()
    assert d.groupby('game').size().eq(2).all()
    assert all(d.season.eq(s).sum()>=200 for s in SEASONS)
    d.to_parquet(OUT/'process_features.parquet',index=False)
    assert set(ARMS).isdisjoint({c for arm in ARMS for c in columns(arm)}), 'Prediction columns must not overwrite any input.'
    predictions=[];fits=[];future={}
    for endpoint in ('shots','xg','goals'):
        for target in TARGETS:
            te=d[d.season.eq(target)].copy()
            for arm in ARMS:
                model=select_model(d,target,arm,endpoint)
                te[arm]=predict(model,te);fits.append({'target_season':target,'model':arm,**model})
            te['endpoint']=endpoint;te['actual']=te[endpoint]
            predictions.append(te[['game','season','date','side','team','seconds','endpoint','actual','min_game_coverage','one_mover','mover_id','mover_role',*ARMS]])
            print(endpoint,target,{m:round(metrics(te,m)['mse'],6) for m in ARMS},flush=True)
        for arm in ARMS:
            future[f'{endpoint}:{arm}']=select_model(d,20262027,arm,endpoint)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_parquet(OUT/'heldout_process_predictions.parquet',index=False)
    dump('fitted_models.json',fits);dump('frozen_future_calibrations.json',future)
    report={'status':'inspected retrospective development; historical vintage REVIEW REQUIRED','coverage':z.groupby('season').agg(rows=('game','size'),included_rows=('included','sum')).reset_index().to_dict('records'),'endpoints':{}}
    for endpoint,g in pred.groupby('endpoint'):
        fold={str(s):{m:metrics(h,m) for m in ARMS} for s,h in g.groupby('season')}
        equal={m:float(np.mean([v[m]['mse'] for v in fold.values()])) for m in ARMS}
        report['endpoints'][endpoint]={'folds':fold,'equal_season_mse':equal,'pooled':{m:metrics(g,m) for m in ARMS},
            'hero_minus_competitor':{m:intervals(g,m) for m in ('ha_rate','lb_rate')},
            'team_season_cluster_sensitivity':{m:intervals(g,m,cluster='team') for m in ('ha_rate','lb_rate')},
            'one_mover':{m:metrics(g[g.one_mover],m) for m in ARMS},
            'one_mover_intervals':{m:{'game':intervals(g[g.one_mover],m,level=.95),'player':intervals(g[g.one_mover],m,cluster='mover_id',level=.95)} for m in ('ha_rate','lb_rate')},
            'one_mover_by_role':{r:{m:metrics(g[g.one_mover & g.mover_role.eq(r)],m) for m in ARMS} for r in ('F','D')}}
        strict=g[g.min_game_coverage.ge(.90)]
        report['endpoints'][endpoint]['coverage90_fixed_adapter_sensitivity']={m:metrics(strict,m) for m in ARMS}
    report['player_diagnostics']=player_summary(scores,pd.read_parquet(INPUTS/'player_targets.parquet'))
    report['team_aggregation']=aggregate_teams(pred)
    checks={}
    primary=report['endpoints']['shots']
    for competitor in ('ha_rate','lb_rate'):
        improvement=1-primary['equal_season_mse']['hero_v8']/primary['equal_season_mse'][competitor]
        both=all(f['hero_v8']['mse']<f[competitor]['mse'] for f in primary['folds'].values())
        secondary=all(v['equal_season_mse']['hero_v8']<=v['equal_season_mse'][competitor]*1.01 for k,v in report['endpoints'].items() if k!='shots')
        mover=primary['one_mover_intervals'][competitor]['game']['delta_mse']<=0
        checks[competitor]={'relative_shot_mse_improvement':improvement,'improves_both_folds':both,'upper97_5_game_interval_below0':primary['hero_minus_competitor'][competitor]['ci'][1]<0,'secondary_noninferiority':secondary,'nonnegative_mover_signal':mover}
        checks[competitor]['development_gate_pass']=bool(improvement>=.01 and both and checks[competitor]['upper97_5_game_interval_below0'] and secondary and mover)
    report['gates']=checks
    report['claim']='No prospective or general talent-superiority claim; report endpoint-specific development results only.'
    dump('evaluation.json',report)
    print(json.dumps({'gates':checks,'player_n':report['player_diagnostics']['n']},indent=2),flush=True)

if __name__=='__main__':main()
