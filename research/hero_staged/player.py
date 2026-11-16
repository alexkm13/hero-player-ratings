"""Frozen macro forecast plus chronological out-of-sample skill corrections."""
import itertools,json
import numpy as np
import pandas as pd
from scipy.linalg import solve
from .common import PCV,CACHE,OUT,ENDS,TESTS,SEASONS,SEED,finite,dump,log,now
from .measure import prepare as measurements,ACTIVITY,EXECUTION,CHANNELS,transform_fit,transform
from .rich import prepare as full_measurements,RICH

ENDNAMES=list(ENDS)
GRID=list(itertools.product([10.,100.,1000.],[3.,100.]))
FAMILIES=['m1','activity','execution','both','both_shift','full_measurements','nonlinear_measurements','persistent_measurements']
BASE=[f'{prefix}_{e}' for prefix in ['current','lag1','lag2'] for e in ENDNAMES]+['log_toi','lag1_log_toi','lag2_log_toi','has_lag1','has_lag2','age_centered','age_squared']
CURRENT=[f'current_{e}' for e in ENDNAMES]+['log_toi','age_centered','age_squared']
SHIFT=[f'shift_{h}_{s}' for h in ['shots','xg','goals'] for s in ['off','def']]

def panel():
    path=CACHE/'player_panel.parquet'
    if path.exists():return pd.read_parquet(path)
    raw=pd.read_parquet(PCV/'data/pcv/player_season_feature_table_all_in_one.parquet',columns=['player_id','player_name','season_id','position_group','eh_rapm_TOI','eh_rapm_Team',*[x[0] for x in ENDS.values()]])
    assert not raw.duplicated(['player_id','season_id']).any()
    raw=raw[raw.position_group.isin(['F','D'])].copy()
    ages=pd.read_csv(PCV/'data/pcv/player_season_ages_2016_2026.csv',usecols=['player_id','season_id','age_years'])
    raw=raw.merge(ages,on=['player_id','season_id'],how='left',validate='one_to_one')
    raw['log_toi']=np.log1p(raw.eh_rapm_TOI.clip(lower=0));raw['age_centered']=raw.age_years-27;raw['age_squared']=raw.age_centered**2
    for e,(c,sign) in ENDS.items():raw['current_'+e]=sign*pd.to_numeric(raw[c],errors='coerce')
    for lag in [1,2]:
        g=raw[['player_id','season_id','log_toi',*[f'current_{e}' for e in ENDNAMES]]].copy()
        g.season_id+=lag*10001;g=g.rename(columns={'log_toi':f'lag{lag}_log_toi',**{f'current_{e}':f'lag{lag}_{e}' for e in ENDNAMES}})
        raw=raw.merge(g,on=['player_id','season_id'],how='left',validate='one_to_one');raw[f'has_lag{lag}']=raw[f'lag{lag}_xgf'].notna().astype(float)
    future=raw[['player_id','season_id','position_group','eh_rapm_TOI','eh_rapm_Team',*[f'current_{e}' for e in ENDNAMES]]].copy()
    future['target_season']=future.season_id;future.season_id-=10001
    future=future.rename(columns={'position_group':'target_role','eh_rapm_TOI':'target_toi','eh_rapm_Team':'target_team',**{f'current_{e}':'target_'+e for e in ENDNAMES}})
    raw=raw.merge(future,on=['player_id','season_id'],how='left',validate='one_to_one')
    m=full_measurements();cols=['player_id','season_id','strict_minutes','tracked_minutes']+[c+s for c in CHANNELS+RICH for s in ['', '_num','_den']]
    raw=raw.merge(m[cols],on=['player_id','season_id'],how='left',validate='one_to_one')
    for lag in [1,2]:
        hist=m[['player_id','season_id',*[c+s for c in CHANNELS+RICH for s in ['', '_den']]]].copy();hist.season_id+=lag*10001
        hist=hist.rename(columns={c+s:f'hist{lag}_{c}'+s for c in CHANNELS+RICH for s in ['', '_den']})
        raw=raw.merge(hist,on=['player_id','season_id'],how='left',validate='one_to_one')
    features=[]
    for p in sorted((OUT/'response').glob('*/source_identity_features.parquet')):
        features.append(pd.read_parquet(p))
    if features:
        sh=pd.concat(features,ignore_index=True);assert not sh.duplicated(['player_id','season_id']).any()
        raw=raw.merge(sh,on=['player_id','season_id'],how='left',validate='one_to_one')
    else:
        for c in SHIFT:raw[c]=np.nan
    hero=pd.read_parquet(PCV/'data/pcv/player_season_all_in_one_8cat_scores_tuned.parquet',columns=['player_id','season_id','overall_8cat_tuned_z','overall_8cat_tuned_pct'])
    raw=raw.merge(hero.rename(columns={'overall_8cat_tuned_z':'benchmark_hero_single','overall_8cat_tuned_pct':'hero_pct'}),on=['player_id','season_id'],how='left',validate='one_to_one')
    raw['benchmark_hero_3yr']=np.nan
    for _,g in raw.groupby('player_id'):
        for i,r in g.iterrows():
            gap=(r.season_id-g.season_id)//10001;keep=gap.between(0,2)&g.hero_pct.notna()
            toi=g.loc[keep,'eh_rapm_TOI'].clip(lower=0).to_numpy();weights=toi*.4**gap[keep].to_numpy()
            if weights.sum()==0 or not np.isfinite(r.hero_pct):continue
            avg=np.average(g.loc[keep,'hero_pct'],weights=weights);total=toi.sum();shrink=1 if total>=2500 else total/(total+1200)*(1 if keep.sum()>1 else .7)
            raw.loc[i,'benchmark_hero_3yr']=.7*(50+shrink*(avg-50))+.3*r.hero_pct
    targetcols=['target_'+e for e in ENDNAMES]
    raw['eligible']=raw.eh_rapm_TOI.ge(300)&raw.target_toi.ge(300)&raw.target_role.eq(raw.position_group)&raw[targetcols].notna().all(axis=1)
    raw.to_parquet(path,index=False)
    dump(OUT/'player_data_audit.json',{'source_rows':len(raw),'matched_players':raw[raw.eligible].groupby('season_id').size().to_dict(),
        'scope':'provided provider RAPM/EV fields; exact strict-5v5 conformity and historical publication vintages not independently certified',
        'hero_inputs':'benchmarks only; three-year displayed basis reconstructed, not exact archived production snapshots',
        'target_units':'native provider contribution per60, defense sign reversed; no future-population target normalization',
        'shift_features':'source-completed fits only; missing before first fit remains explicit'})
    return raw

def fit_ridge(x,y,source,alpha,half):
    x=np.asarray(x,float);y=finite(y)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    x=np.where(np.isfinite(x),x,med);w=2.**(-((np.max(source)-source)/10001)/half)
    mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-8]=1
    z=(x-mean)/scale;center=np.average(y,axis=0,weights=w)
    gram=np.einsum('ij,ik,i->jk',z,z,w,optimize=False)+alpha*np.eye(z.shape[1])
    rhs=np.einsum('ij,ik,i->jk',z,y-center,w,optimize=False)
    beta=solve(gram,rhs,assume_a='pos');err=np.max(np.abs(np.einsum('ij,jk->ik',gram,beta,optimize=False)-rhs))
    if err>1e-6:raise FloatingPointError('ridge normal-equation failure')
    return {'median':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'beta':finite(beta).tolist(),'intercept':center.tolist(),
        'alpha':alpha,'half':half,'source_max':int(np.max(source)),'n':len(y),'gradient_inf':float(err)}

def predict_ridge(model,x):
    if model.get('kind')=='per_head_ridge':
        return np.column_stack([predict_ridge(head,x)[:,j] for j,head in enumerate(model['heads'])])
    if model.get('kind')=='extra_trees':return predict_forest(model,x)
    x=np.asarray(x,float);x=np.where(np.isfinite(x),x,np.asarray(model['median']))
    z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return finite(np.asarray(model['intercept'])+np.einsum('ij,jk->ik',z,np.asarray(model['beta']),optimize=False))

def predict_forest(model,x):
    x=np.asarray(x,float);x=np.where(np.isfinite(x),x,np.asarray(model['median'])).astype(np.float32)
    pred=np.zeros((len(x),6))
    for tree in model['trees']:
        left=np.asarray(tree['left']);right=np.asarray(tree['right']);feature=np.asarray(tree['feature']);threshold=np.asarray(tree['threshold'])
        node=np.zeros(len(x),dtype=int)
        while True:
            active=np.flatnonzero(left[node]>=0)
            if len(active)==0:break
            current=node[active];choose=x[active,feature[current]]<=threshold[current]
            node[active]=np.where(choose,left[current],right[current])
        pred+=np.asarray(tree['value'])[node]
    return finite(pred/len(model['trees'])*np.asarray(model['target_sd']))

def fit_forest(x,y,source,depth,minleaf,sd):
    from sklearn.ensemble import ExtraTreesRegressor
    import sklearn
    x=np.asarray(x,float);y=finite(y);sd=np.maximum(np.asarray(sd),1e-8)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T]);x=np.where(np.isfinite(x),x,med)
    weight=2.**(-((np.max(source)-source)/10001)/3)
    estimator=ExtraTreesRegressor(n_estimators=100,max_depth=int(depth),min_samples_leaf=int(minleaf),max_features=.7,
        bootstrap=False,random_state=SEED,n_jobs=1)
    estimator.fit(x,y/sd,sample_weight=weight)
    trees=[]
    for e in estimator.estimators_:
        t=e.tree_;trees.append({'left':t.children_left.tolist(),'right':t.children_right.tolist(),'feature':t.feature.tolist(),
            'threshold':t.threshold.tolist(),'value':t.value.reshape(t.node_count,6).tolist()})
    model={'kind':'extra_trees','median':med.tolist(),'target_sd':sd.tolist(),'trees':trees,'depth':int(depth),'minleaf':int(minleaf),
        'n_estimators':100,'half':3,'source_max':int(np.max(source)),'n':len(y),'sklearn_training_version':sklearn.__version__}
    expected=estimator.predict(x[:50])*sd;actual=predict_forest(model,x[:50]);error=float(np.max(np.abs(expected-actual)))
    assert error<1e-10
    model['manual_runtime_replay_error']=error
    return model

def target(frame):return frame[['target_'+e for e in ENDNAMES]].to_numpy(float)
def normalized_mse(y,p,train):
    sd=target(train).std(axis=0);sd[sd<1e-8]=1
    return float(np.mean(((y-p)/sd)**2))

def base_select(frame,source,columns):
    train=frame[frame.season_id.lt(source)&frame.eligible]
    validation_sources=sorted(train.season_id.unique())[-3:]
    trials=[];scores=[]
    for alpha,half in GRID:
        losses=[]
        for valid in validation_sources:
            tr=train[train.season_id.lt(valid)];va=train[train.season_id.eq(valid)]
            if tr.season_id.nunique()<2 or len(tr)<80:continue
            fit=fit_ridge(tr[columns],target(tr),tr.season_id.to_numpy(),alpha,half)
            pred=predict_ridge(fit,va[columns]);sd=target(tr).std(axis=0);sd[sd<1e-8]=1
            losses.append(np.mean(((target(va)-pred)/sd)**2,axis=0))
        score=np.mean(losses,axis=0) if losses else np.full(6,1e20 if alpha==100 and half==100 else 1e21)
        scores.append(score)
        trials.append({'alpha':alpha,'half':half,'inner_loss':float(np.mean(score)),'head_losses':dict(zip(ENDNAMES,map(float,score))),'inner_folds':len(losses)})
    heads=[];configs=[]
    for j in range(6):
        k=min(range(len(GRID)),key=lambda k:(scores[k][j],-GRID[k][0],-GRID[k][1]));alpha,half=GRID[k]
        heads.append(fit_ridge(train[columns],target(train),train.season_id.to_numpy(),alpha,half));configs.append({'endpoint':ENDNAMES[j],'alpha':alpha,'half':half})
    model={'kind':'per_head_ridge','heads':heads,'features':columns,'head_settings':configs,'source_max':int(train.season_id.max()),
        'alpha':[c['alpha'] for c in configs],'half':[c['half'] for c in configs]}
    return model,trials

def correction_features(train,test,family,params=None):
    channels=ACTIVITY if family=='activity' else (EXECUTION if family=='execution' else (CHANNELS+RICH if family in ['full_measurements','nonlinear_measurements','persistent_measurements'] else CHANNELS))
    if family=='persistent_measurements':train=pool_measurements(train,channels);test=pool_measurements(test,channels)
    if params is None:params=transform_fit(train,channels)
    x=transform(train,channels,params);v=transform(test,channels,params)
    if family=='both_shift':x=np.column_stack([x,train[SHIFT]]);v=np.column_stack([v,test[SHIFT]])
    if family=='nonlinear_measurements':x=np.column_stack([x,train[BASE]]);v=np.column_stack([v,test[BASE]])
    return x,v,params

def pool_measurements(frame,channels):
    pooled=frame.copy()
    for c in channels:
        numerator=np.zeros(len(frame));denominator=np.zeros(len(frame))
        for lag in [0,1,2]:
            prefix='' if lag==0 else f'hist{lag}_';name=prefix+c
            value=pd.to_numeric(frame[name],errors='coerce').to_numpy(float);den=pd.to_numeric(frame[name+'_den'],errors='coerce').fillna(0).to_numpy(float)
            valid=np.isfinite(value)&(den>0);weight=.5**lag
            numerator+=np.where(valid,value*den,0)*weight;denominator+=np.where(valid,den,0)*weight
        pooled[c]=np.divide(numerator,denominator,out=np.full(len(frame),np.nan),where=denominator>0)
        pooled[c+'_den']=denominator;pooled[c+'_num']=numerator
    return pooled

def correction_fit(train,test,family,alpha,half,basecols):
    # No historical measured skill evidence means no learned correction.
    known=train.tracked_minutes.fillna(0).gt(0)|train.strict_minutes.fillna(0).gt(0)
    if known.sum()<40:return np.zeros((len(test),6)),{'available':False,'reason':'insufficient earlier measured sources'}
    # Sources with absent measurements do not create artificial zero-skill
    # targets. Fit the correction only where canonical source evidence exists.
    train=train[known]
    x,v,params=correction_features(train,test,family)
    residual=target(train)-train[basecols].to_numpy(float)
    model=fit_forest(x,residual,train.season_id.to_numpy(),alpha,half,target(train).std(axis=0)) if family=='nonlinear_measurements' else fit_ridge(x,residual,train.season_id.to_numpy(),alpha,half)
    model.update({'available':True,'family':family,'measurement_priors':params,'base_is_frozen':True,
        'residual_training':'earlier chronological out-of-sample M1 errors; no feedback to measurements or M1'})
    return predict_ridge(model,v),model

def run():
    frame=panel();allpred=[];alltrials=[];current=[];models={}
    basecols=['oof_m1_'+e for e in ENDNAMES]
    for c in basecols:frame[c]=np.nan
    # Building chronological M1 predictions once provides genuinely OOF
    # residual targets for every skill-value fit, including inner tuning.
    for role in ['F','D']:
        rf=frame[frame.position_group.eq(role)]
        for source in sorted(rf.season_id.unique()):
            if source<20182019:continue
            tr=rf[rf.season_id.lt(source)&rf.eligible]
            if len(tr)<80:continue
            test=rf[rf.season_id.eq(source)]
            model,trials=base_select(rf,source,BASE);pred=predict_ridge(model,test[BASE])
            frame.loc[test.index,basecols]=pred
            models[(role,int(source))]=model
            for trial in trials:alltrials.append({'stage':'macro','role':role,'source':int(source),**trial})
            log(f'player macro OOF {role}/{source}: alpha={model["alpha"]} half={model["half"]}')
    for role in ['F','D']:
        rf=frame[frame.position_group.eq(role)]
        for source in [t-10001 for t in TESTS]+[SEASONS[-1]]:
            tr=rf[rf.season_id.lt(source)&rf.eligible&rf[basecols].notna().all(axis=1)]
            test=rf[rf.season_id.eq(source)]
            final=source==SEASONS[-1]
            if not final:test=test[test.eligible]
            if test.empty:raise RuntimeError('empty player outer fold')
            basepred=test[basecols].to_numpy(float);preds={'m1':basepred};audits={};innerloss={}
            valids=sorted(tr[tr.strict_minutes.notna()].season_id.unique())[-3:]
            # All candidate corrections tune on identical inner sources and
            # fixed macro predictions; absent earlier evidence falls back zero.
            for family in FAMILIES:
                best=None
                configs=[(100.,100.)] if family=='m1' else (list(itertools.product([4,6],[30,60])) if family=='nonlinear_measurements' else GRID)
                for alpha,half in configs:
                    losses=[]
                    for valid in valids:
                        it=tr[tr.season_id.lt(valid)];iv=tr[tr.season_id.eq(valid)]
                        if it.empty:continue
                        correction=np.zeros((len(iv),6)) if family=='m1' else correction_fit(it,iv,family,alpha,half,basecols)[0]
                        pred=iv[basecols].to_numpy()+correction
                        losses.append(normalized_mse(target(iv),pred,it))
                    score=float(np.mean(losses)) if losses else 1e20
                    alltrials.append({'stage':'skill_value','role':role,'source':source,'family':family,'alpha':alpha,'half':half,
                        'configuration':{'max_depth':alpha,'min_samples_leaf':half,'training_half_life':3} if family=='nonlinear_measurements' else {'ridge_alpha':alpha,'training_half_life':half},
                        'inner_loss':score,'inner_sources':[int(v) for v in valids],'inner_folds':len(losses)})
                    key=(score,alpha,-half) if family=='nonlinear_measurements' else (score,-alpha,-half)
                    if best is None or key<best[0]:best=(key,alpha,half)
                innerloss[family]=best[0][0]
                if family!='m1':
                    correction,model=correction_fit(tr,test,family,best[1],best[2],basecols)
                    preds[family]=basepred+correction;audits[family]=model
            selected=min(FAMILIES,key=lambda f:(innerloss[f],FAMILIES.index(f)))
            # Freeze selection before reading outer targets for reporting.
            sel={'role':role,'source_season':source,'outcome_season':source+10001,'family':selected,'inner_losses':innerloss,
                'outer_score_used':False,'max_training_target':int(tr.target_season.max())}
            dump(OUT/'player'/f'selection_{role}_{source}.json',sel)
            preds['selected']=preds[selected]
            alltrain=rf[rf.season_id.lt(source)&rf.eligible]
            cm,cmtrials=base_select(rf,source,CURRENT);preds['current_macro']=predict_ridge(cm,test[CURRENT]);preds['role_mean']=np.repeat(target(alltrain).mean(axis=0)[None],len(test),axis=0)
            for trial in cmtrials:alltrials.append({'stage':'current_macro_baseline','role':role,'source':source,**trial})
            benchmarks={}
            for benchmark in ['benchmark_hero_single','benchmark_hero_3yr']:
                bt=alltrain[alltrain[benchmark].notna()]
                bm=fit_ridge(bt[[benchmark]],target(bt),bt.season_id.to_numpy(),10,100)
                preds[benchmark]=predict_ridge(bm,test[[benchmark]])
                benchmarks[benchmark]=bm
            keep=['player_id','player_name','season_id','position_group','age_years','eh_rapm_TOI','target_toi','eh_rapm_Team','target_team','tracked_minutes']
            for family,pred in preds.items():
                out=test[keep].copy();out['family']=family;out['selected_family']=selected;out['outcome_season']=source+10001
                for j,e in enumerate(ENDNAMES):out['pred_'+e]=pred[:,j];out['actual_'+e]=test['target_'+e].to_numpy()
                out['pred_net_goals']=pred[:,2]+pred[:,5];out['actual_net_goals']=test.target_gf.to_numpy()+test.target_ga_suppression.to_numpy()
                (current if final else allpred).append(out)
            dump(OUT/'player'/f'model_{role}_{source}.json',{'macro':models[(role,source)],'corrections':audits,'selected':sel,
                'current_macro':cm,'benchmark_calibrators':benchmarks,'information_cutoff':'2026-09-17' if final else f'completed source {source}',
                'scope':'future provider endpoint forecast; no certified causal/neutral/replacement talent claim'})
            log(f'player {role}/{source}: selected={selected}')
    pd.concat(allpred,ignore_index=True).to_parquet(OUT/'player_predictions.parquet',index=False)
    pd.concat(current,ignore_index=True).to_parquet(OUT/'current_player_predictions.parquet',index=False)
    dump(OUT/'player_inner_trials.json',alltrials)
    # Persist the OOF states for replay and time-boundary auditing.
    frame[['player_id','season_id','position_group',*basecols]].to_parquet(OUT/'macro_oof_predictions.parquet',index=False)
    dump(OUT/'player_run_manifest.json',{'created':now(),'families':FAMILIES,'outer_outcomes':TESTS,
        'macro_oof_score_before_future_target':True,'measurement_feedback':False,'maximum_current_training_target':SEASONS[-1],
        'no_2026_27_outcomes':True,'all_target_normalization':'training SD only in inner selection; native targets in predictions'})

if __name__=='__main__':run()
