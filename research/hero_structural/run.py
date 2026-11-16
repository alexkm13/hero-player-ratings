"""Restartable research runner. No production tables are written."""
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .common import OUT,DATA,SEASONS,SEED,dump,sha,now
from .factors import Canonical,StateIndex,build_factors
from .model import JointModel,empirical_skill_cov,filter_update,K,R,D,SKILL_SD,ROLE_SD,AR

CONFIG={'experiment':'20260917-structural-talent-01','seed':SEED,
        'test_seasons':SEASONS[2:],'initial_maxiter':450,'correlation_refit_maxiter':450,
        'continuation_maxiter':450,'daily_maxiter':120,'skill_daily_variance_fraction':1/365,
        'responsibility_daily_variance_fraction':1/90,'annual_skill_ar':.85,
        'annual_responsibility_ar':.50,'parameter_uncertainty':'not included',
        'curvature':'within-player conditional blocks; choice curvature diagonal approximation',
        'primary':'macro marked point-process NLL / 60 team clock minutes; seconds intensity units',
        'declaration':'solver continuation only on identical training objective; no outer-score tuning'}

def announce(message):print(message,flush=True)

def fit_context(factors):
    betas=[];audits=[]
    for f in factors:
        state=np.zeros((f.matrices[0].shape[1],D))
        def fun(b):
            l,_,g=f.objective(state,b)
            return l+.5*np.sum(f.penalty*b*b),g+f.penalty*b
        fit=minimize(fun,f.initial(),jac=True,method='L-BFGS-B',bounds=f.bounds(),
                     options={'maxiter':300,'ftol':1e-11,'gtol':1e-5})
        betas.append(fit.x);audits.append({'factor':f.name,'success':bool(fit.success),'message':str(fit.message)})
    return betas,audits

def save_fit(path,index,state,cov,betas,factors,skill_cov,transform,audit,variant,train):
    path.mkdir(parents=True,exist_ok=True)
    index.inventory.to_parquet(path/'inventory.parquet',index=False)
    np.savez_compressed(path/'parameters.npz',state=state,cov=cov,skill_cov=skill_cov,
                        **{'beta_'+f.name:b for f,b in zip(factors,betas)})
    dump(path/'manifest.json',{'created':now(),'variant':variant,'training_seasons':train,
         'skating_transform':transform,'factors':{f.name:{'kind':f.kind,'size':f.size,'rows':len(f.y)} for f in factors},
         'solver':audit,'config':CONFIG,'parameters_sha256':sha(path/'parameters.npz')})

def fit_one(data,train,variant,label):
    path=OUT/'fits'/label/variant
    if (path/'manifest.json').exists():announce(f'skip completed {label}/{variant}');return
    subset=data.subset(seasons=train);index=StateIndex(subset.state_inventory(),subset.meta)
    factors,transform=build_factors(subset,index,variant='macro' if variant=='context' else variant)
    announce(f'fit {label}/{variant}: {index.n} states; {len(factors)} factors')
    if variant=='context':
        betas,audit=fit_context(factors);state=np.zeros((index.n,D));skill_cov=np.diag(SKILL_SD**2)
        cov=np.repeat(np.diag(np.r_[SKILL_SD**2,np.full(R,ROLE_SD**2)])[None],index.n,axis=0)
        save_fit(path,index,state,cov,betas,factors,skill_cov,transform,audit,variant,train);return
    def progress(stage):
        return lambda it,loss,grad:announce(f'{label}/{variant}/{stage} iter={it} objective={loss:.6f} scaled_grad={grad:.4f}')
    model=JointModel(factors,index.n,index.previous)
    state,betas,first=model.fit(CONFIG['initial_maxiter'],callback=progress('diagonal'))
    # Declared training-only correlation sensitivity; no held-out inputs.
    skill_cov=empirical_skill_cov(state,index.previous)
    correlated=JointModel(factors,index.n,index.previous,skill_cov)
    state,betas,second=correlated.fit(CONFIG['correlation_refit_maxiter'],state,betas,progress('correlated'))
    audits={'initial':first,'correlated':second}
    if not second['success']:
        state,betas,third=correlated.fit(CONFIG['continuation_maxiter'],state,betas,progress('continuation'))
        audits['continuation']=third
    covariance=correlated.curvature(state,betas)
    audits['final_success']=bool(list(audits.values())[-1]['success'])
    audits['finite_state']=bool(np.isfinite(state).all())
    audits['minimum_covariance_eigenvalue']=float(np.linalg.eigvalsh(covariance).min())
    audits['latent_bound_hits']=int(np.sum(np.abs(state/np.r_[SKILL_SD,np.full(R,ROLE_SD)])>=7.999))
    save_fit(path,index,state,covariance,betas,factors,skill_cov,transform,audits,variant,train)
    announce(f'saved {label}/{variant}: {audits}')

def load_fit(path):
    manifest=json.loads((path/'manifest.json').read_text())
    params=dict(np.load(path/'parameters.npz'))
    return manifest,pd.read_parquet(path/'inventory.parquet'),params

def active_fit(test,variant):
    whitened=OUT/'fits_whitened'/str(test)/variant
    if (whitened/'manifest.json').exists():return whitened
    polished=OUT/'fits_polished'/str(test)/variant
    if (polished/'manifest.json').exists():return polished
    original=OUT/'fits'/str(test)/variant
    if variant!='context':
        m=json.loads((original/'manifest.json').read_text())
        if not m['solver']['final_success']:raise RuntimeError(f'Numerical continuation pending: {original}; run hero_structural.polish')
    return original

def initialise_test(index,oldinv,params,test_season):
    mean=np.zeros((index.n,D));skill_cov=params['skill_cov']
    q=np.zeros((D,D));q[:K,:K]=skill_cov;q[K:,K:]=np.eye(R)*ROLE_SD**2
    covariance=np.repeat(q[None],index.n,axis=0)
    latest=oldinv.sort_values('season').groupby('player_id').tail(1)
    oldlookup={int(r.player_id):int(i) for i,r in latest.iterrows()}
    for i,r in index.inventory.iterrows():
        j=oldlookup.get(int(r.player_id))
        if j is None:continue
        gap=max(1,(int(test_season)-int(oldinv.iloc[j].season))//10001)
        a=AR**gap
        mean[i]=params['state'][j]*a
        covariance[i]=params['cov'][j]*a[:,None]*a[None,:]
        for g in range(gap):covariance[i]+=q*(AR**g)[:,None]*(AR**g)[None,:]
    return mean,covariance,q

def factor_scores(factors,state,betas,variant,date,index):
    rows=[];cal=[]
    for f,b in zip(factors,betas):
        if f.name=='skating_measurement':continue
        eta,_=f.eta(state,b);loss,_,weight=f.loss_residual(eta)
        if f.kind=='choice':
            from scipy.special import logsumexp
            z=eta.reshape(-1,5);prediction=np.exp(z-logsumexp(z,axis=1,keepdims=True))
            target=f.y.reshape(-1,5);brier=((prediction-target)**2).sum(axis=1)
            pred_observed=(prediction*target).sum(axis=1)
            totals=pd.DataFrame({'game':f.game,'side':f.side,'loss':loss,'n':np.ones(len(loss)),
                                 'brier':brier,'expected':pred_observed,'observed':np.ones(len(loss))})
            actor=np.asarray(f.matrices[0].argmax(axis=1)).ravel()[f.y.astype(bool)]
            totals['position']=index.role[actor]
        else:
            if f.kind=='poisson':prediction=weight; brier=np.zeros(len(loss))
            else:
                from scipy.special import expit
                prediction=expit(eta);brier=(prediction-f.y)**2
                for lo in np.arange(0,1,.1):
                    keep=(prediction>=lo)&(prediction<lo+.1)
                    if keep.any():cal.append({'variant':variant,'date':date,'factor':f.name,'bin':round(float(lo),1),
                        'n':int(keep.sum()),'predicted_sum':float(prediction[keep].sum()),'observed_sum':float(f.y[keep].sum())})
            totals=pd.DataFrame({'game':f.game,'side':f.side,'loss':loss,'n':np.ones(len(loss)),
                                 'brier':brier,'expected':prediction,'observed':f.y})
            if f.kind=='poisson':totals['exposure']=f.exposure
            if f.kind=='poisson':totals['position']='ALL'
            else:
                actor_matrix=f.matrices[2]
                actor=np.asarray(actor_matrix.argmax(axis=1)).ravel()
                totals['position']=np.where(np.asarray(actor_matrix.sum(axis=1)).ravel()>0,index.role[actor],'U')
        aggregate=totals.groupby(['game','side','position']).sum().reset_index()
        aggregate['factor']=f.name;aggregate['variant']=variant;aggregate['date']=date
        rows.append(aggregate)
    return rows,cal

def evaluate_one(data,test,variant):
    label=str(test);path=OUT/'evaluation'/label/variant
    if (path/'manifest.json').exists():announce(f'skip completed evaluation {label}/{variant}');return
    path.mkdir(parents=True,exist_ok=True)
    manifest,oldinv,params=load_fit(active_fit(test,variant))
    subset=data.subset(seasons=[test]);index=StateIndex(subset.state_inventory(),subset.meta)
    state,cov,q=initialise_test(index,oldinv,params,test)
    if variant=='context':state[:]=0
    beta_map={k[5:]:v for k,v in params.items() if k.startswith('beta_')}
    dates=sorted(subset.risk.date.unique());rows=[];cal=[];updates=[];snapshots=[]
    # Assimilate only prior-completed-season physical evidence before game 1.
    if variant=='joint':
        prior_factors,_=build_factors(subset.subset(date=dates[0]),index,variant='joint',skate_transform=manifest['skating_transform'],include_skating=False)
        # Physical rows have no game date; compile once with the season's
        # risk table solely to construct the separate normal measurement.
        full_factors,_=build_factors(subset,index,variant='joint',skate_transform=manifest['skating_transform'])
        physical=[f for f in full_factors if f.name=='skating_measurement'];del full_factors,prior_factors
        if physical:
            state,cov,audit=filter_update(physical,state,cov,[beta_map[f.name] for f in physical],CONFIG['daily_maxiter'])
            updates.append({'date':'before_first_game_prior_season_EDGE',**audit})
    last=pd.Timestamp(dates[0])
    for di,date in enumerate(dates):
        current=pd.Timestamp(date);gap=max(0,(current-last).days);last=current
        if di and variant!='context':
            decay=np.r_[np.full(K,.85**(gap/365)),np.full(R,.50**(gap/90))]
            state*=decay
            cov=cov*decay[None,:,None]*decay[None,None,:]
            drift=q.copy();drift[:K,:K]*=gap/365;drift[K:,K:]*=gap/90;cov+=drift
        day=subset.subset(date=date)
        factors,_=build_factors(day,index,variant='macro' if variant=='context' else variant,
                                skate_transform=manifest['skating_transform'],include_skating=False)
        betas=[beta_map[f.name] for f in factors]
        # This is the prediction state. All same-date outcomes are unseen.
        if variant!='context':snapshots.append(state.copy().astype(np.float32))
        dayrows,daycal=factor_scores(factors,state,betas,variant,str(date),index);rows.extend(dayrows);cal.extend(daycal)
        if variant!='context':
            state,cov,audit=filter_update(factors,state,cov,betas,CONFIG['daily_maxiter'])
            updates.append({'date':str(date),**audit})
        if di%25==0:announce(f'evaluate {test}/{variant}: date {di+1}/{len(dates)} {date}')
    pd.concat(rows,ignore_index=True).to_parquet(path/'game_component_scores.parquet',index=False)
    pd.DataFrame(cal).to_parquet(path/'calibration_bins.parquet',index=False)
    index.inventory.to_parquet(path/'inventory.parquet',index=False)
    if snapshots:np.savez_compressed(path/'pregame_states.npz',dates=np.asarray(dates,dtype=str),state=np.stack(snapshots))
    np.savez_compressed(path/'end_states.npz',state=state,cov=cov,skill_cov=params['skill_cov'])
    dump(path/'filter_audit.json',updates)
    dump(path/'manifest.json',{'created':now(),'test_season':test,'variant':variant,'training_manifest':manifest,
         'dates':len(dates),'same_date_score_before_update':True,'failed_updates':sum(not x['success'] for x in updates),
         'filter_coordinates':'full prior covariance Cholesky',
         'minimum_end_covariance_eigenvalue':float(np.linalg.eigvalsh(cov).min()),'config':CONFIG})

def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    path=OUT/'executable_config.json'
    if path.exists():
        if json.loads(path.read_text())!=CONFIG:raise RuntimeError('Configuration drift; start a new experiment')
    else:dump(path,CONFIG)

def main():
    warnings.simplefilter('error',RuntimeWarning)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['fit','evaluate','report','export'])
    parser.add_argument('--test-season',type=int);parser.add_argument('--variant',choices=['context','macro','joint'])
    parser.add_argument('--final',action='store_true');args=parser.parse_args();prepare()
    if args.stage=='report':
        from .report import main as report;report();return
    if args.stage=='export':
        from .export import main as export;export();return
    data=Canonical.load()
    tests=[args.test_season] if args.test_season else SEASONS[2:]
    variants=[args.variant] if args.variant else ['context','macro','joint']
    if args.stage=='fit':
        for test in tests:
            train=[s for s in SEASONS if s<test]
            for variant in variants:fit_one(data,train,variant,str(test))
        if args.final:
            for variant in ['macro','joint']:fit_one(data,SEASONS,variant,'completed_2025_26')
    else:
        for test in tests:
            for variant in variants:evaluate_one(data,test,variant)

if __name__=='__main__':main()
