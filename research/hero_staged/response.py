"""Exposure-complete lagged-skill additive team response, with cut feedback."""
import json,zipfile,itertools
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from .common import DATA,CACHE,OUT,PRIOR,PCV,SEASONS,TESTS,dump,sha,log,now,finite
from .measure import prepare as measurements,ACTIVITY,EXECUTION,transform_fit,transform

FAMILIES=['context','identities','activity','execution','both']
GRID=list(itertools.product([.025,.075,.15],[30.,300.],[1.,3.]))
CTX=['intercept','home','period2','period3','score_m3','score_m2','score_m1','score_p1','score_p2','score_p3','early']

def contexts(home,period,diff,early):
    diff=np.clip(diff,-3,3)
    return np.column_stack([np.ones(len(home)),home,period==2,period==3,
        *[diff==d for d in [-3,-2,-1,1,2,3]],early]).astype(float)

def goals():
    path=CACHE/'all_goals.parquet'
    if path.exists():return pd.read_parquet(path)
    parts=[];sources={};disagreement=0;official_sources={}
    required_games=set(pd.read_parquet(DATA/'risk.parquet',columns=['game']).game.unique())
    for start in range(2021,2026):
        p=PRIOR/f'hero_portable_talent_program/raw/moneypuck_shots/shots_{start}.zip'
        with zipfile.ZipFile(p) as z:
            d=pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith('.csv'))),
                usecols=['game_id','shotID','goal','time','isHomeTeam','homeTeamGoals','awayTeamGoals'],low_memory=False)
        d=d[d.goal.eq(1)].sort_values(['game_id','time','shotID']).copy()
        h=d.groupby('game_id').isHomeTeam.cumsum()-d.isHomeTeam
        a=d.groupby('game_id').cumcount()-h
        source_bad=int(((h!=d.homeTeamGoals)|(a!=d.awayTeamGoals)).sum())
        disagreement+=source_bad
        d['game']=start*1_000_000+d.game_id;d['step']=np.where(d.isHomeTeam.eq(1),1,-1)
        parts.append(d[['game','shotID','time','step']]);sources[p.name]=sha(p)
    source_goals=pd.concat(parts,ignore_index=True);source_goals.to_parquet(CACHE/'moneypuck_goal_ledger.parquet',index=False)
    # Shot feeds omit some goals. Cached official play-by-play provides actual
    # goal timing at all strengths; do not infer missing goals retrospectively.
    records=[];post_score_bad=0
    for i,game in enumerate(sorted(required_games)):
        p=PCV/f'data/cache/pbp_{game}.parquet'
        if not p.exists():raise FileNotFoundError(f'Official goal context cache missing: {p}')
        d=pd.read_parquet(p,columns=['Event','Period','Seconds_Elapsed','Ev_Team','Home_Team','Away_Team','Home_Score','Away_Score','event_id'])
        d=d[d.Event.eq('GOAL')&d.Period.le(3)].sort_values(['Period','Seconds_Elapsed','event_id']).copy()
        d['step']=np.where(d.Ev_Team.eq(d.Home_Team),1,np.where(d.Ev_Team.eq(d.Away_Team),-1,0))
        if d.step.eq(0).any():raise ValueError(f'Unknown official goal side: {game}')
        home=d.step.eq(1).cumsum();away=d.step.eq(-1).cumsum()
        post_score_bad+=int(((home!=d.Home_Score)|(away!=d.Away_Score)).sum())
        for ev in d.itertuples():records.append({'game':int(game),'shotID':int(ev.event_id),'time':float((ev.Period-1)*1200+ev.Seconds_Elapsed),'step':int(ev.step)})
        official_sources[str(game)]=sha(p)
        if i%1500==0:log(f'official goal context cache {i+1}/{len(required_games)}')
    if post_score_bad:raise ValueError(f'Official cumulative score mismatch: {post_score_bad}')
    result=pd.DataFrame(records);result.to_parquet(path,index=False)
    dump(OUT/'goal_context_audit.json',{'goal_rows':len(result),'moneypuck_cumulative_score_disagreements':disagreement,
         'official_post_goal_score_disagreements':post_score_bad,'sources':sources,'official_pbp_hashes':official_sources,
         'preflight_repair':'use cached official goals; retain incomplete shot-feed goal ledger',
         'clock':'source time is absolute game seconds','goal_context':'strict left limit; goal cannot enter its own score predictor'})
    return result

def prepare():
    path=CACHE/'response_rows.parquet'
    if path.exists():return pd.read_parquet(path)
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    r=pd.read_parquet(DATA/'risk.parquet');s=pd.read_parquet(DATA/'shots.parquet');g=goals()
    gl={int(game):v.sort_values(['time','shotID']) for game,v in g.groupby('game')}
    sl={int(game):v for game,v in s.groupby('game')}
    parts=[];zero_event_contexts=0;goal_event_checks=0;goal_clock_repairs=[]
    # Within each original interval, event and integrated-risk contexts are
    # compiled separately. Zero-exposure event keys retain boundary evidence.
    for gi,(game,risk) in enumerate(r.groupby('game',sort=False)):
        risk=risk.reset_index(drop=True);gg=gl.get(int(game))
        gt=np.array([]) if gg is None else gg.time.to_numpy(float)
        cumulative=np.r_[0,np.cumsum(np.array([]) if gg is None else gg.step.to_numpy(int))]
        def score(t):return cumulative[np.searchsorted(gt,t,side='left')]
        starts=risk.t0.to_numpy(float);ends=risk.t1.to_numpy(float)
        key_to_row={};indices=[];periods=[];scores=[];early=[];expos=[]
        def keyrow(i,t,exposure=0.,score_override=None):
            period=min(3,int(max(0,t-1e-7)//1200)+1)
            diff=int(np.clip(score(t) if score_override is None else score_override,-3,3));age=int(t-starts[i]<3)
            key=(i,period,diff,age)
            j=key_to_row.get(key)
            if j is None:
                j=len(indices);key_to_row[key]=j;indices.append(i);periods.append(period);scores.append(diff);early.append(age);expos.append(0.)
            expos[j]+=exposure;return j
        for i,(begin,end) in enumerate(zip(starts,ends)):
            inside=gt[(gt>begin)&(gt<end)]
            cuts=np.unique(np.r_[begin,end,inside, [v for v in [begin+3,1200,2400] if begin<v<end]])
            for left,right in zip(cuts[:-1],cuts[1:]):keyrow(i,(left+right)/2,right-left)
        shot=sl.get(int(game));event_targets={}
        local=pd.Index(risk.interval_index)
        if shot is not None:
            for ev in shot.itertuples():
                event_score=None
                if ev.goal:
                    # Independent raw source pre-event scoreboard validated above.
                    own_goal=gg[gg.time.sub(ev.event_seconds).abs().le(2)&gg.step.eq(1 if ev.side==0 else -1)] if gg is not None else gg
                    if own_goal is None or len(own_goal)!=1:raise ValueError('canonical goal missing unique score event')
                    official_time=float(own_goal.iloc[0].time);event_score=score(official_time)
                    if official_time!=ev.event_seconds:goal_clock_repairs.append({'game':int(game),'shotID':int(ev.shotID),'shot_clock':float(ev.event_seconds),'official_goal_clock':official_time})
                    goal_event_checks+=1
                i=local.get_loc(ev.interval_index);j=keyrow(i,float(ev.event_seconds),score_override=event_score)
                v=event_targets.setdefault((j,int(ev.side)),np.zeros(3));v+=np.array([1,ev.xg,ev.goal])
        ix=np.asarray(indices);frame=risk.iloc[ix][['game','season','date','home_players','away_players','interval_index']].reset_index(drop=True)
        frame['period']=periods;frame['home_score_diff']=scores;frame['early']=early;frame['seconds']=expos
        target=np.zeros((len(ix),2,3))
        for (j,side),v in event_targets.items():target[j,side]=v
        for side,label in [(0,'home'),(1,'away')]:
            for k,head in enumerate(['shots','xg','goals']):frame[label+'_'+head]=target[:,side,k]
        zero_event_contexts+=int((frame.seconds.eq(0)&(target.sum(axis=(1,2))>0)).sum())
        np.testing.assert_allclose(frame.seconds.sum(),risk.seconds.sum(),rtol=1e-12)
        parts.append(frame)
        if gi%1000==0:log(f'compile score/period/exposure: game {gi+1}')
    frame=pd.concat(parts,ignore_index=True)
    # Same lineup/context rows inside a game are exchangeable for this model.
    frame['home_players']=frame.home_players.map(tuple);frame['away_players']=frame.away_players.map(tuple)
    keys=['game','season','date','home_players','away_players','period','home_score_diff','early']
    frame=frame.groupby(keys,sort=False,as_index=False).sum(numeric_only=True).drop(columns='interval_index')
    assert np.isclose(frame.seconds.sum(),r.seconds.sum())
    assert frame[['home_shots','away_shots']].sum().sum()==len(s)
    assert np.isclose(frame[['home_xg','away_xg']].sum().sum(),s.xg.sum())
    frame.to_parquet(path,index=False)
    dump(OUT/'response_data_audit.json',{'rows':len(frame),'games':int(frame.game.nunique()),'seconds':float(frame.seconds.sum()),
        'shots':len(s),'zero_exposure_event_contexts_retained':zero_event_contexts,'canonical_goal_context_checks':goal_event_checks,
        'source':sha(DATA/'risk.parquet'),'score_context_split':'all goals, period changes, raw interval first-three-second boundary',
        'all_original_exposure_retained':True,'event_left_limit_context':True,'source_goal_clock_differences':goal_clock_repairs,
        'clock_difference_rule':'unique same-game, same-side official goal within two seconds; event keeps original clock/lineup, score excludes its own official goal'})
    return frame

def design(frame,measure,family,priors=None,scale=None,players=None):
    if players is None:players=np.unique(np.r_[np.stack(frame.home_players).ravel(),np.stack(frame.away_players).ravel()])
    lookup=pd.Index(players);h=lookup.get_indexer(np.stack(frame.home_players).ravel()).reshape(-1,5);a=lookup.get_indexer(np.stack(frame.away_players).ravel()).reshape(-1,5)
    own=np.stack([h,a],axis=1).reshape(-1,5);opp=np.stack([a,h],axis=1).reshape(-1,5)
    row=np.repeat(np.arange(len(own)),5);N=len(players)
    def matrix(ids):
        col=ids.ravel();ok=col>=0
        return sparse.csr_matrix((np.ones(ok.sum()),(row[ok],col[ok])),shape=(len(own),N))
    O=matrix(own);P=matrix(opp)
    channels=ACTIVITY if family=='activity' else (EXECUTION if family=='execution' else (ACTIVITY+EXECUTION if family=='both' else []))
    priorframe=measure[measure.season_id.isin(frame.season.unique()-10001)]
    if priors is None:priors=transform_fit(priorframe,channels) if channels else {}
    names=CTX.copy();blocks=[sparse.csr_matrix(contexts(np.tile([1,0],len(frame)),np.repeat(frame.period,2),
        np.stack([frame.home_score_diff,-frame.home_score_diff],axis=1).ravel(),np.repeat(frame.early,2)))]
    if family!='context':blocks.extend([O,-P]);names.extend([f'own_{p}' for p in players]+[f'supp_{p}' for p in players])
    if channels:
        pvals=np.zeros((len(frame)*2,len(channels)))
        qvals=np.zeros_like(pvals)
        for season in frame.season.unique():
            prior=measure[measure.season_id.eq(season-10001)].set_index('player_id').reindex(players).reset_index()
            prior['position_group']=prior.position_group.fillna('U')
            values=transform(prior,channels,priors)[:,::3]
            for j in range(len(channels)):
                # Centering is fixed by source training profiles, never test rows.
                center=np.mean([v[channels[j]]['prior'] for v in priors.values()])
                values[:,j]-=center
            use=np.repeat(frame.season.to_numpy()==season,2)
            pvals[use]=O[use]@values;qvals[use]=P[use]@values
        features=np.column_stack([pvals,qvals])
        if scale is None:
            scale=features.std(axis=0);scale[scale<1e-8]=1.
        features/=scale
        blocks.append(sparse.csr_matrix(features));names.extend(['own_'+c for c in channels]+['opp_'+c for c in channels])
    else:scale=[]
    X=sparse.hstack(blocks,format='csr')
    target=np.stack([frame[['home_shots','home_xg','home_goals']].to_numpy(),frame[['away_shots','away_xg','away_goals']].to_numpy()],axis=1).reshape(-1,3)
    return {'X':X,'y':target,'exposure':np.repeat(frame.seconds.to_numpy(float),2),'games':np.repeat(frame.game.to_numpy(),2),
        'season':np.repeat(frame.season.to_numpy(),2),'players':players,'names':names,'priors':priors,'scale':np.asarray(scale),'family':family}

def fit(d,head,sd,ridge,half,warm=None):
    X=d['X'];y=d['y'][:,head];exposure=d['exposure'];seasons=d['season']//10000
    weight=2.**(-(seasons.max()-seasons)/half)
    penalty=np.full(X.shape[1],ridge);penalty[:len(CTX)]=.01;penalty[0]=0
    if d['family']!='context':penalty[len(CTX):len(CTX)+2*len(d['players'])]=1/sd**2
    rate=(np.sum(y*weight)+.01)/(np.sum(exposure*weight)+1)
    initial=np.zeros(X.shape[1]);initial[0]=np.log(rate)
    if warm is not None and len(warm)==len(initial):initial=warm.copy()
    curvature=np.asarray(X.power(2).T@(exposure*weight*rate)).ravel()+penalty
    coordinate=1/np.sqrt(np.maximum(curvature,1e-8))
    def objective(z):
        beta=z*coordinate;eta=X@beta
        mu=exposure*np.exp(eta)
        loss=np.sum(weight*(mu-y*eta))+.5*np.sum(penalty*beta**2)
        grad=np.asarray(X.T@(weight*(mu-y))).ravel()+penalty*beta
        return float(loss),finite(grad*coordinate)
    z=initial/coordinate;attempts=[]
    for attempt in range(3):
        result=minimize(objective,z,jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-11,'gtol':1e-5,'maxls':30,'maxcor':15})
        z=result.x;attempts.append({'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
          'objective':float(result.fun),'scaled_gradient_inf':float(np.max(np.abs(result.jac)))})
        if result.success:break
    return finite(z*coordinate),{'success':bool(result.success),'attempts':attempts,'sd':sd,'ridge':ridge,'half':half,
        'head':['shots','xg','goals'][head],'training_outcome_max':int(d['season'].max())}

def scores(d,beta,head):
    eta=d['X']@beta;mu=d['exposure']*np.exp(eta);y=d['y'][:,head]
    loss=mu-y*eta
    table=pd.DataFrame({'game':d['games'],'loss':loss,'expected':mu,'observed':y,'seconds':d['exposure']})
    table=table.groupby('game',as_index=False).sum()
    return table

def saved_design(train,test,measure,family):
    # Include players absent from training with zero residual prior; identities
    # are known request keys, not outcome talent information.
    players=np.unique(np.r_[np.stack(train.home_players).ravel(),np.stack(train.away_players).ravel(),
                            np.stack(test.home_players).ravel(),np.stack(test.away_players).ravel()])
    td=design(train,measure,family,players=players)
    vd=design(test,measure,family,priors=td['priors'],scale=td['scale'],players=players)
    return td,vd

def run_fold(frame,measure,test,export=False):
    label='current' if export else str(test);root=OUT/'response'/label;root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():log(f'skip response {label}');return
    # Training outcome 2021 has no preceding raw measurement season; use 2022+.
    train=frame[frame.season.lt(test)&frame.season.ge(SEASONS[1])]
    future=frame[frame.season.eq(test)] if not export else frame[frame.season.eq(SEASONS[-1])].iloc[:10]
    last=int(train.season.max());innertrain=train[train.season.lt(last)];innervalid=train[train.season.eq(last)]
    if innertrain.empty:
        dates=sorted(innervalid.date.unique());cut=dates[int(.8*len(dates))]
        innertrain=innervalid[innervalid.date.lt(cut)];innervalid=innervalid[innervalid.date.ge(cut)]
    assert innertrain.date.max()<innervalid.date.min()
    selections={};trials=[];outer=[]
    for family in FAMILIES:
        log(f'response {label}/{family}: chronological tuning')
        td,vd=saved_design(innertrain,innervalid,measure,family)
        configurations=[(.075,300,3)] if family=='context' else ([(sd,300,half) for sd in [.025,.075,.15] for half in [1,3]] if family=='identities' else GRID)
        best=None;warm=None
        for i,(sd,ridge,half) in enumerate(configurations):
            beta,audit=fit(td,0,sd,ridge,half,warm);warm=beta
            score=float(scores(vd,beta,0).loss.sum())
            rec={'family':family,'trial':i,'sd':sd,'ridge':ridge,'half':half,'inner_loss':score,**audit}
            trials.append(rec)
            dump(root/'inner_trials.json',trials)
            if audit['success'] and (best is None or score<best[0]):best=(score,sd,ridge,half)
            log(f'  trial {i+1}/{len(configurations)} loss={score:.4f} converged={audit["success"]}')
        if best is None:raise RuntimeError(f'no converged response configuration: {family}')
        selections[family]={'inner_loss':best[0],'sd':best[1],'ridge':best[2],'half':best[3],
            'inner_training_outcome_max':int(innertrain.season.max()),'inner_validation_outcome':last}
        # Save selection BEFORE any corresponding outer score.
        dump(root/'selections.json',selections)
        del td,vd
        td,vd=saved_design(train,future,measure,family)
        arrays={};head_audits=[]
        for head in range(3):
            beta,audit=fit(td,head,*best[1:]);arrays[['shots','xg','goals'][head]]=beta;head_audits.append(audit)
            if not audit['success']:raise RuntimeError(f'failed final response {label}/{family}/{head}')
            if not export:
                sc=scores(vd,beta,head);sc['family']=family;sc['head']=['shots','xg','goals'][head];sc['test_season']=test;outer.append(sc)
        np.savez_compressed(root/f'{family}.npz',players=td['players'],scale=td['scale'],**arrays)
        dump(root/f'{family}.json',{'family':family,'priors':td['priors'],'names':td['names'],'training_outcome_max':int(train.season.max()),
            'selection':selections[family],'solver':head_audits,'status':'conditional research response; identities are associations',
            'parameter_sha256':sha(root/f'{family}.npz')})
        # Strictly source-available fitted identity associations for later
        # player forecast inputs. Older source seasons are never rewritten.
        if family=='identities':
            values={'player_id':td['players'],'season_id':int(train.season.max())}
            for head in ['shots','xg','goals']:
                b=arrays[head];N=len(td['players']);values['shift_'+head+'_off']=b[len(CTX):len(CTX)+N]
                values['shift_'+head+'_def']=b[len(CTX)+N:len(CTX)+2*N]
            pd.DataFrame(values).to_parquet(root/'source_identity_features.parquet',index=False)
        del td,vd
    if outer:pd.concat(outer,ignore_index=True).to_parquet(root/'game_scores.parquet',index=False)
    selected=min(FAMILIES,key=lambda f:selections[f]['inner_loss'])
    dump(root/'manifest.json',{'created':now(),'outcome_season':test,'selected_family':selected,
        'selection_uses_outer_outcomes':False,'training_outcome_max':int(train.season.max()),'families':FAMILIES,'export':export})

def run():
    m=measurements();f=prepare()
    # Source 2021 has no v2 response fit. Source 2022 is the first source-
    # available fit; all later fits remain fixed when advancing time.
    for test in TESTS:run_fold(f,m,test)
    run_fold(f,m,20262027,export=True)

if __name__=='__main__':run()
