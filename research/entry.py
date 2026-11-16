"""Chronological entry count, actor choice and crossing, with team-shot transfer."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, xlogy
from sklearn.preprocessing import OneHotEncoder

from research import action_opportunity as older
from research.count_models import PoissonRidge
from research.changed_linemates import cluster_ci, process_scores
from research.paths import ARCHIVE as ROOT, output
from hero.common import team, sha

OUT = output("entry")
OLD = ROOT / "outputs/action_opportunity_v1"
CHANGE = ROOT / 'outputs/changed_linemate_transport_v1'
TEAM = older.v3.BASE / 'team_games.parquet'
PG = older.v3.BASE / 'player_games.parquet'
SEASONS = (20222023, 20232024, 20242025, 20252026)
TARGETS = SEASONS[-2:]
KEY = ['game', 'side', 'player_id']


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def prepare(tg):
    fields = ['game', 'season_id', 'side', 'interval_index', 'player_id', 'source_file',
              'source_row', 'action_type', 'entry_type', 'period', 'game_sec']
    micro = pd.read_parquet(older.MICRO, columns=fields, filters=[('season_id', '<=', 20252026)])
    assert not micro.duplicated(['source_file','source_row','action_type','player_id']).any()
    counts = micro[micro.action_type.isin(['entry','retrieval'])].groupby(['game','action_type']).size().unstack(fill_value=0)
    games = counts.index[counts.entry.ge(20) & counts.retrieval.ge(20)]
    orig = micro[micro.game.isin(games) & micro.action_type.eq('entry')].copy()
    key = ['game','source_file','source_row']
    code = orig.entry_type.str.strip().str.upper()
    case_counts = orig.assign(code=code).groupby(['season_id','code']).size()
    assert not orig.duplicated(key).any(), 'Ambiguous recorded entry actor'
    row = orig[code.isin(['C','D','F'])].copy()
    row['code'] = code.loc[row.index]
    row['cross'] = row.code.ne('F').astype(int)
    risk = pd.read_parquet(older.RISK, columns=['game','season','interval_index','t0','t1',
        'home_players','away_players','home_team','away_team'], filters=[('season', '<=', 20252026)])
    d = row.merge(risk, on=['game','interval_index'], validate='many_to_one')
    assert len(d) == len(row) and d.season.eq(d.season_id).all()
    assert d.game_sec.between(d.t0,d.t1).all() and d.period.isin([1,2,3]).all()
    h,a = [np.stack(d[col]) for col in ['home_players','away_players']]
    sides = d.side.to_numpy()[:,None]
    own = np.where(sides==0,h,a)
    opp = np.where(sides==0,a,h)
    assert np.all(np.diff(np.sort(own,axis=1),axis=1)>0) and np.all(np.diff(np.sort(opp,axis=1),axis=1)>0)
    assert not (own[:,:,None]==opp[:,None,:]).any()
    credited = own==d.player_id.to_numpy()[:,None]
    assert np.all(credited.sum(axis=1)==1)
    d['slot'] = credited.argmax(axis=1)
    d['team'] = np.where(d.side.eq(0),d.home_team,d.away_team)
    d['opponent'] = np.where(d.side.eq(0),d.away_team,d.home_team)
    for c in ['team','opponent']:
        d[c] = d[c].map(team)
    d['home'] = d.side.eq(0).astype(int)
    e = d.groupby(['game','side']).size().rename('attempts').reset_index()
    sides = tg[tg.game.isin(games)].merge(e,on=['game','side'],how='left',validate='one_to_one')
    assert len(sides)==2*len(games)
    sides['attempts'] = sides.attempts.fillna(0).astype(int)
    assert sides.attempts.sum()==len(d)
    sides['home'] = sides.side.eq(0).astype(int)
    other = sides[['game','side','team']].copy()
    other['side'] = 1-other.side
    sides = sides.merge(other.rename(columns={'team':'opponent'}),on=['game','side'],validate='one_to_one')
    for c in ['team','opponent']:
        sides[c] = sides[c].map(team)
    check = d[['game','side','team','opponent']].merge(sides[['game','side','team','opponent']],
           on=['game','side'],validate='many_to_one',suffixes=('','_check'))
    assert check.team.eq(check.team_check).all() and check.opponent.eq(check.opponent_check).all()
    d.drop(columns=['home_players','away_players'],inplace=True)
    audit = {}
    for s,g in d.groupby('season'):
        n = len(g)
        audit[str(s)] = {'games':int(g.game.nunique()), 'eligible_game_sides':int(sides.season.eq(s).sum()),
            'attempts':n, 'carries':int(g.code.eq('C').sum()), 'dumps':int(g.code.eq('D').sum()),
            'failures':int(g.code.eq('F').sum()), 'cross_rate':float(g.cross.mean()),
            'code_counts_before_filter':{str(k):int(v) for k,v in case_counts.loc[s].items()},
            'canonical_games':int(tg.loc[tg.season.eq(s),'game'].nunique()),
            'unique_actors':int(g.player_id.nunique())}
    dump('data_audit.json',{'decision':'GO for recorded attempt allocation and crossing; no possession opportunity denominator',
         'game_selection':'at least 20 linked entries and 20 linked retrievals', 'seasons':audit})
    print('Canonical entry audit passes:',len(d),'attempts in',len(games),'games',flush=True)
    return d,own,opp,sides


def code_matrix(indices, size):
    ix = np.asarray(indices)
    if ix.ndim==1:
        ix=ix[:,None]
    n,k = ix.shape
    flat = ix.ravel()
    keep=flat>=0
    return sparse.coo_matrix((np.ones(keep.sum()),(np.repeat(np.arange(n),k)[keep],flat[keep])),
                             shape=(n,size)).tocsr()


def role_codes(meta, player, season):
    ref = meta.set_index(['season_id','player_id']).position_group
    role = ref.reindex(pd.MultiIndex.from_arrays([season,player.ravel()])).map({'F':0,'D':1})
    return role.fillna(2).to_numpy(int).reshape(player.shape)


def model_arrays(d,own,opp,roles,players,enc):
    slot = d.slot.to_numpy()
    r = np.arange(len(d))
    # Make the encoding independent of eligible action result and credited slot.
    # The credited slot is used only for the execution question and training labels.
    p = players.get_indexer(own.ravel()).reshape(own.shape)
    a0 = code_matrix(roles.ravel(),3)
    a1 = sparse.hstack([a0,code_matrix(p.ravel(),len(players))],format='csr')
    other = p[np.arange(5)[None,:]!=slot[:,None]].reshape(-1,4)
    op = players.get_indexer(opp.ravel()).reshape(opp.shape)
    base = sparse.hstack([np.ones((len(d),1)), enc.transform(d[['team','opponent','period','home']]),
             code_matrix(roles[r,slot],3), .2*code_matrix(other,len(players)),
             .2*code_matrix(op,len(players))],format='csr')
    actor = sparse.hstack([base,code_matrix(p[r,slot],len(players))],format='csr')
    return a0,a1,base,actor


def fit(x,y,w,penalty,multinomial):
    n = len(y)
    row = np.arange(n)
    init = np.zeros(x.shape[1])
    def f(b):
        eta = x @ b
        if multinomial:
            z = eta.reshape(-1,5)
            norm = logsumexp(z,axis=1)
            resid = np.exp(z-norm[:,None])
            resid[row,y] -= 1
            resid *= w[:,None]
            loss = np.sum(w*(norm-z[row,y]))
        else:
            resid = w*(expit(eta)-y)
            loss = np.sum(w*(np.logaddexp(0,eta)-eta*y))
        return (float(loss+.5*np.sum(penalty*b*b)), np.asarray(x.T@resid.ravel()).ravel()+penalty*b)
    r = minimize(f,init,method='L-BFGS-B',jac=True,options={'maxiter':1000,'ftol':1e-12,'gtol':1e-5})
    assert r.success and np.max(np.abs(r.jac))/n < 2e-6, (r.message,np.max(np.abs(r.jac)),n)
    return r.x, {'iterations':int(r.nit),'max_abs_gradient_per_trial':float(np.max(np.abs(r.jac))/n)}


def predict(x,b,multi):
    z=x@b
    if multi:
        z=z.reshape(-1,5)
        p=np.exp(z-logsumexp(z,axis=1)[:,None])
        assert np.allclose(p.sum(axis=1),1.,atol=1e-12)
    else:
        p=expit(z)
    assert ((p>0)&(p<1)).all()
    return p


def stats(d, h, seed):
    if d.empty:
        return {'rows':0,'players':0}
    b=d[h+'_baseline_loss']
    m=d[h+'_recent_loss']
    equal=d[h+'_equal_loss']
    delta=m-b
    return {'rows':len(d),'games':int(d.game.nunique()),'players':int(d.player_id.nunique()),
         'baseline_loss':float(b.mean()),'recent_loss':float(m.mean()),'equal_loss':float(equal.mean()),
         'delta':float(delta.mean()),'relative_improvement_pct':float(-100*delta.mean()/b.mean()),
         'recent_minus_equal':float((m-equal).mean()),
         'game':cluster_ci(delta,d.game,seed),'player':cluster_ci(delta,d.player_id,seed+1),
         **({'cross_rate':float(d.cross.mean()), 'recent_predicted_rate':float(d.cross_recent.mean()),
             'baseline_predicted_rate':float(d.cross_baseline.mean())} if h=='cross' else {})}


def fit_heads(d,own,opp,sides,meta,change):
    reports, profiles, preds, tested_volume = [], [], [], []
    for target in SEASONS:
        source = list(range(max(20212022,target-30003),target,10001))
        train = d.season.isin(source).to_numpy()
        test = d.season.eq(target).to_numpy()
        outcome = d.loc[train]
        assert len(outcome) and outcome.season.max()<target
        assert not set(d.loc[test,'game']) & set(outcome.game)
        y = [outcome.slot.to_numpy(),outcome.cross.to_numpy()]
        train_role = role_codes(meta,own[train],np.repeat(outcome.season.to_numpy(),5))
        vocabulary=pd.Index(np.unique(np.r_[own[train].ravel(),opp[train].ravel()]))
        enc=OneHotEncoder(handle_unknown='ignore').fit(outcome[['team','opponent','period','home']])
        x=model_arrays(outcome,own[train],opp[train],train_role,vocabulary,enc)
        latest = meta[meta.season_id.lt(target)].sort_values('season_id').drop_duplicates('player_id',keep='last')
        assert latest.season_id.max()<target
        observed_roles = latest.set_index('player_id').position_group
        profile_role = observed_roles.reindex(vocabulary).fillna('U')
        w=(.5**((target-10001-outcome.season.to_numpy())//10001)).astype(float)
        w/=w.mean()
        pars, diagnostics = {}, {}
        for tag,j,prior,weights in [('baseline',0,False,w),('recent',1,True,w),('equal',1,True,np.ones(len(w)))]:
            ps, ds = [], []
            for head,ix in enumerate([j,2+j]):
                mat=x[ix]
                penalties=np.full(mat.shape[1],20.)
                if head==1:
                    penalties[0]=0.
                if prior:
                    penalties[-len(vocabulary):]=100.
                coef,info=fit(mat,y[head],weights,penalties,head==0)
                ps.append(coef)
                ds.append(info)
            pars[tag]=ps
            diagnostics[tag]=ds
        actor_attempts=outcome.player_id.value_counts().reindex(vocabulary,fill_value=0)
        onice=pd.Series(own[train].ravel()).value_counts().reindex(vocabulary,fill_value=0)
        assert int(actor_attempts.sum())==len(outcome) and int(onice.sum())==5*len(outcome)
        profiles.append(pd.DataFrame({'season':target,'source_end':target-10001,'player_id':vocabulary,
            'role':profile_role.to_numpy(),'attempts':actor_attempts.to_numpy(),
            'on_ice_attempts':onice.to_numpy(), 'covered':onice.gt(0).to_numpy(float),
            'allocation':pars['recent'][0][-len(vocabulary):],
            'crossing':pars['recent'][1][-len(vocabulary):],
            'allocation_equal':pars['equal'][0][-len(vocabulary):],
            'crossing_equal':pars['equal'][1][-len(vocabulary):]}))
        if target not in TARGETS:
            continue
        hold = d.loc[test].copy().reset_index(drop=True)
        oi, op = own[test], opp[test]
        hold_roles=observed_roles.reindex(oi.ravel()).map({'F':0,'D':1}).fillna(2).to_numpy(int).reshape(-1,5)
        valid=model_arrays(hold,oi,op,hold_roles,vocabulary,enc)
        alter=model_arrays(hold.assign(code='F',cross=1-hold.cross),oi,op,hold_roles,vocabulary,enc)
        assert all((a!=b).nnz==0 for a,b in zip(valid,alter))
        perm=[4,3,2,1,0]
        shuffled=model_arrays(hold.assign(slot=4-hold.slot),oi[:,perm],op[:,perm],hold_roles[:,perm],vocabulary,enc)
        row,slot=np.arange(len(hold)),hold.slot.to_numpy()
        for tag,j in [('baseline',0),('recent',1),('equal',1)]:
            ap=predict(valid[j],pars[tag][0],True)
            cp=predict(valid[2+j],pars[tag][1],False)
            assert np.allclose(ap,predict(shuffled[j],pars[tag][0],True)[:,perm],atol=1e-12)
            assert np.allclose(cp,predict(shuffled[2+j],pars[tag][1],False),atol=1e-12)
            hold['allocation_'+tag]=ap[row,slot]
            hold['allocation_'+tag+'_loss']=-np.log(ap[row,slot])
            hold['cross_'+tag]=cp
            hold['cross_'+tag+'_loss']=-np.log(np.where(hold.cross.eq(1),cp,1-cp))
            for k in range(5):
                hold[f'p_{tag}_{k}']=ap[:,k]
        cohort=change[change.season_id.eq(target)].set_index(KEY).cohort
        hold['cohort']=cohort.reindex(pd.MultiIndex.from_frame(hold[KEY])).to_numpy()
        assert hold.cohort.notna().all()
        on_ice=pd.MultiIndex.from_arrays([np.repeat(hold.game.to_numpy(),5),np.repeat(hold.side.to_numpy(),5),oi.ravel()])
        cm=cohort.reindex(on_ice)
        assert cm.notna().all()
        hold['changed_in_lineup']=cm.eq('changed').to_numpy().reshape(-1,5).any(axis=1)
        for k in range(5):
            hold[f'own_{k}']=oi[:,k]
            hold[f'opp_{k}']=op[:,k]
        slices={'all':hold,'lineup_changed':hold[hold.changed_in_lineup]}
        slices.update({c:hold[hold.cohort.eq(c)] for c in ['changed','stable','clean_mover']})
        metrics={h:{k:stats(sub,h,target+7*i) for i,(k,sub) in enumerate(slices.items())} for h in ['allocation','cross']}
        volume=fit_volume(sides,source,target)
        tested_volume.append(volume)
        metrics['volume']=volume_stats(volume)
        reports.append({'source':source,'target':target,'train_attempts':len(outcome),
            'test_attempts':len(hold),'fit':diagnostics,'metrics':metrics})
        preds.append(hold)
        print('Entry heads complete:',target,'allocation changed delta',metrics['allocation']['changed']['delta'],
              'cross delta',metrics['cross']['changed']['delta'],flush=True)
    return reports,pd.concat(profiles,ignore_index=True),pd.concat(preds,ignore_index=True),pd.concat(tested_volume,ignore_index=True)


def fit_volume(sides,source,target):
    a=sides[sides.season.isin(source)]
    b=sides[sides.season.eq(target)].copy()
    cols=['team','opponent','home']
    enc=OneHotEncoder(handle_unknown='ignore').fit(a[cols])
    train_hours=a.seconds.to_numpy()/3600
    h=b.seconds.to_numpy()/3600
    y=a.attempts.to_numpy()
    x=enc.transform(a[cols]).tocsr()
    w=.5**((target-10001-a.season.to_numpy())//10001)
    w/=w.mean()
    model=PoissonRidge(np.full(x.shape[1],20.)).fit(x,w*y,w*train_hours)
    b['baseline']=h*np.sum(y*w)/np.sum(train_hours*w)
    b['context']=model.predict(enc.transform(b[cols]).tocsr(),h)
    return b


def volume_stats(d):
    y=d.attempts.to_numpy()
    ret={'team_sides':len(d),'games':int(d.game.nunique())}
    for k in ['baseline','context']:
        mu=d[k].to_numpy()
        ret[k+'_deviance']=float(np.mean(2*(xlogy(y,y/mu)-y+mu)))
        ret[k+'_predicted_over_observed']=float(mu.sum()/y.sum())
    return ret


def bridge(tg, profiles):
    hero,v3=older.hero,older.v3
    q,lineage=hero.hero_profiles()
    tg=v3.attach_pregame_form(tg)
    tg=tg[tg.season.isin(SEASONS)].copy()
    pg=pd.read_parquet(PG)
    pg=v3.source_prior_features(pg[pg.season.isin(SEASONS)],pd.read_parquet(v3.MEASURE))
    pg=hero.add_source_profiles(pg,q)
    incidence=pd.concat([pd.read_parquet(OLD/'early_action_incidence.parquet'),pd.read_parquet(older.INCIDENCE)],
                        ignore_index=True)
    raw=older.profiles(incidence,SEASONS)
    raw_cols=['deviation_entry','deviation_retrieval','covered_entry']
    new_cols=['allocation','covered','crossing']
    assert profiles.source_end.lt(profiles.season).all()
    pg=pg.merge(raw[['season','player_id',*raw_cols]],on=['season','player_id'],how='left',validate='many_to_one')
    pg=pg.merge(profiles[['season','player_id',*new_cols]],on=['season','player_id'],how='left',validate='many_to_one')
    pg[raw_cols+new_cols]=pg[raw_cols+new_cols].fillna(0.)
    old=pd.read_parquet(OLD/'heldout_team_games.parquet')[['game','side','candidate']].rename(columns={'candidate':'old_volume'})
    cut=pd.read_parquet(CHANGE/'process_predictions.parquet')[['game','side','process_cohort']]
    preds,reports=[],[]
    for target in TARGETS:
        source=[s for s in SEASONS if s<target]
        selected,x0,train,test,_=hero.make_design(tg,pg,source,target,'hero_categories_shares')
        base=sparse.hstack([x0,older.action_design(selected,pg,train,raw_cols)],format='csr')
        result=selected.loc[test,['game','season','team','side','seconds','shots']].rename(columns={'shots':'actual'}).copy()
        result['baseline']=v3.fit_predict(selected,base,train,test,older.ALPHA)
        for name,cols in [('allocation',new_cols[:2]),('full',new_cols)]:
            aug=sparse.hstack([base,older.action_design(selected,pg,train,cols)],format='csr')
            result[name]=v3.fit_predict(selected,aug,train,test,older.ALPHA)
        n=len(result)
        result=result.merge(old,on=['game','side'],validate='one_to_one').merge(cut,on=['game','side'],validate='one_to_one')
        assert len(result)==n
        difference=float(np.max(np.abs(result.baseline-result.old_volume)))
        assert difference < 1e-6, difference
        high=result[result.process_cohort.eq('high_change')]
        sub={'all':result,'high_change':high,'low_change':result[result.process_cohort.eq('low_change')],
             'low_support':result[result.process_cohort.eq('low_support')]}
        r={name:{kind:process_scores(sub[kind],'baseline',name,target+11*i)
                 for i,kind in enumerate(sub)} for name in ['allocation','full']}
        r['crossing_increment']={kind:process_scores(sub[kind],'allocation','full',target+19+i)
                                  for i,kind in enumerate(sub)}
        reports.append({'target':target,'source':source,'baseline_max_replay_difference':difference,'scores':r})
        preds.append(result)
        print('Shot bridge complete:',target,'all delta',r['full']['all']['delta'],
              'high-change delta',r['full']['high_change']['delta'],flush=True)
    assert sum(map(len,preds))==len(old)==len(cut)
    return reports,pd.concat(preds,ignore_index=True),lineage


def main():
    if (OUT/'evaluation.json').exists():
        raise RuntimeError('Completed experiment exists; preserve it.')
    source_files = {TEAM, PG, older.v3.MEASURE, older.v3.SHOT_SHARE, older.v3.A1_SHARE,
                    older.INCIDENCE, older.MICRO, older.RISK, older.META,
                    older.hero.HERO_INPUT, older.hero.HERO_WEIGHTS, older.hero.HERO_REPLAY}
    source_files.update([Path(__file__),OLD/'evaluation.json',
        OLD/'early_action_incidence.parquet', OLD/'heldout_team_games.parquet',
        CHANGE/'target_exposure.parquet',CHANGE/'process_predictions.parquet',
        Path(older.__file__),Path(older.hero.__file__),Path(older.v3.__file__),Path(sys.modules[PoissonRidge.__module__].__file__)])
    manifest={'started_utc':datetime.now(timezone.utc).isoformat(),
              'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__},
              'files':{str(p):sha(p) for p in sorted(source_files)},'max_completed_season':20252026}
    dump('run_manifest.json',manifest)
    tg=pd.read_parquet(TEAM,filters=[('season','<=',20252026)])
    meta=pd.read_parquet(older.META,columns=['season_id','player_id','position_group'],filters=[('season_id','<=',20252026)])
    assert not meta.duplicated(['season_id','player_id']).any()
    change=pd.read_parquet(CHANGE/'target_exposure.parquet',columns=['season_id','cohort',*KEY])
    d,own,opp,side=prepare(tg)
    behavior,prof,event,volume=fit_heads(d,own,opp,side,meta,change)
    process,shot,lineage=bridge(tg,prof)
    decisions={}
    for h in ['allocation','cross']:
        fold=[r['metrics'][h]['changed'] for r in behavior]
        decisions[h+'_pass']=all(s['delta']<0 and s['game']['ci95'][1]<0 and
                                 s['player']['ci95'][1]<0 for s in fold)
        if h=='allocation':
            decisions[h+'_pass'] &= all(r['metrics'][h]['lineup_changed']['delta']<0 for r in behavior)
        decisions[h+'_recency_improves_both']=all(s['recent_minus_equal']<0 for s in fold)
    decisions['team_shot_pass']=all(r['scores']['full']['all']['delta']<0 and
        r['scores']['full']['high_change']['delta']<0 and
        r['scores']['full']['high_change']['game_cluster']['ci95'][1]<0 for r in process)
    decisions['advance_to_value_research']=all(decisions[k] for k in ['allocation_pass','cross_pass','team_shot_pass'])
    assert all(sha(p)==h for p,h in manifest['files'].items())
    for name,x in [('source_profiles',prof),('test_event_predictions',event),
                   ('test_volume_predictions',volume),('test_shot_predictions',shot)]:
        x.to_parquet(OUT/f'{name}.parquet',index=False)
    dump('verification.json',{'status':'PASS','source_inputs_unchanged':True,
        'event_key_roster_and_phase_membership':True,'permutation_label_and_probability_checks':True,
        'old_bridge_replayed_to_tolerance':True,'canonical_attempts':len(d),
        'max_observed_season':int(d.season.max()),'production_changed':False})
    dump('evaluation.json',{'status':'historical_development','behavior':behavior,'process':process,
        'lineage':lineage,'decisions':decisions})
    print(decisions,flush=True)


if __name__=='__main__':
    main()
