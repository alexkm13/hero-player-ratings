"""Identical-objective numerical continuation with curvature preconditioning.

Keeps original completed fits; writes separate fits_polished manifests. No
held-out scores, outcomes, model coefficients or priors are selected here.
"""
import argparse,json
import numpy as np
from .common import OUT,SEASONS,dump
from .factors import Canonical,StateIndex,build_factors
from .model import JointModel,D,SKILL_SD,ROLE_SD
from .run import load_fit,save_fit

def precondition(model,state,betas):
    covariance=model.curvature(state,betas)
    local=np.sqrt(np.diagonal(covariance,axis1=1,axis2=2)).ravel()
    global_scales=[]
    for f,b in zip(model.factors,betas):
        eta,_=f.eta(state,b);_,_,w=f.loss_residual(eta)
        columns=[]
        for m,mask in zip(f.matrices,f.learn):
            if mask.any():columns.append(m@state[:,mask])
        columns.append(f.ctx);x=np.column_stack(columns)
        info=np.einsum('ij,i,ij->j',x,w,x,optimize=False)+f.penalty
        global_scales.append(1/np.sqrt(info))
    model.scale=np.r_[local,*global_scales]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test-season',type=int,required=True);parser.add_argument('--variant',choices=['macro','joint'],required=True)
    args=parser.parse_args();test=args.test_season;variant=args.variant
    original=OUT/'fits'/str(test)/variant;target=OUT/'fits_polished'/str(test)/variant
    if (target/'manifest.json').exists():print('skip completed polish',flush=True);return
    manifest,inventory,p=load_fit(original)
    if manifest['solver']['final_success']:print('original converged; no continuation needed',flush=True);return
    data=Canonical.load().subset(seasons=manifest['training_seasons']);index=StateIndex(inventory,data.meta)
    factors,transform=build_factors(data,index,variant,manifest['skating_transform'])
    state=p['state'];betas=[p['beta_'+f.name] for f in factors]
    model=JointModel(factors,index.n,index.previous,p['skill_cov']);audits={'original_solver':manifest['solver']}
    before=model.objective(model.initial(state,betas))[0]
    for attempt in range(2):
        precondition(model,state,betas)
        callback=lambda it,loss,grad:print(f'polish {test}/{variant}/{attempt+1} iter={it} objective={loss:.6f} scaled_grad={grad:.5f}',flush=True)
        state,betas,audit=model.fit(900,state,betas,callback)
        audits['curvature_attempt_'+str(attempt+1)]=audit
        if audit['success']:break
    covariance=model.curvature(state,betas);after=model.objective(model.initial(state,betas))[0]
    if after>before+1e-4:raise AssertionError('Continuation worsened identical objective')
    audits.update(final_success=bool(audit['success']),final_attempt_completed=True,
        finite_state=bool(np.isfinite(state).all()),minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),
        objective_before=before,objective_after=after,latent_bound_hits=int(np.sum(np.abs(state/np.r_[SKILL_SD,np.full(D-8,ROLE_SD)])>=7.999)))
    save_fit(target,index,state,covariance,betas,factors,p['skill_cov'],transform,audits,variant,manifest['training_seasons'])
    print(f'Polished saved: {audits}',flush=True)

if __name__=='__main__':main()
