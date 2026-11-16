"""Same-objective optimizer using full within-player curvature coordinates.

Bounded tanh reparameterization preserves the physical skill/role envelope.
Shared coefficients retain exact L-BFGS bounds. No test data or model selection.
"""
import argparse
import numpy as np
from scipy.optimize import minimize
from .common import OUT
from .factors import Canonical,StateIndex,build_factors
from .model import JointModel,D,K,R,SKILL_SD,ROLE_SD
from .run import load_fit,save_fit
from .polish import precondition

class Coordinates:
    def __init__(self,model,state,betas):
        self.model=model;self.n=model.nstates;self.off=self.n*D
        self.bound=np.r_[SKILL_SD,np.full(R,ROLE_SD)]*8
        self.base=np.arctanh(np.clip(state/self.bound,-1+1e-12,1-1e-12))
        self.L=np.linalg.cholesky(model.curvature(state,betas))
        self.b=np.concatenate(betas);self.bs=model.scale[self.off:]
        self.bounds=[(None,None)]*self.off
        offset=0
        for f in model.factors:
            self.bounds.extend([( (lo-self.b[offset+i])/self.bs[offset+i],(hi-self.b[offset+i])/self.bs[offset+i]) for i,(lo,hi) in enumerate(f.bounds())])
            offset+=f.size

    def physical(self,z):
        x=z[:self.off].reshape(self.n,D)
        u=self.base+np.einsum('nij,nj->ni',self.L,x,optimize=False)/self.bound
        t=np.tanh(u);state=t*self.bound
        b=self.b+z[self.off:]*self.bs
        return np.r_[state.ravel(),b],1-t*t

    def objective(self,z):
        physical,derivative=self.physical(z)
        value,g=self.model.objective(physical/self.model.scale)
        g=g/self.model.scale
        state_grad=g[:self.off].reshape(self.n,D)*derivative
        gz=np.einsum('nji,nj->ni',self.L,state_grad,optimize=False).ravel()
        return value,np.r_[gz,g[self.off:]*self.bs]

    def fit(self,maxiter=900):
        callback_count=[0]
        def callback(z):
            callback_count[0]+=1
            if callback_count[0]%20==0:
                l,g=self.objective(z);print(f'full-whiten iter={callback_count[0]} objective={l:.6f} grad={abs(g).max():.6f}',flush=True)
        result=minimize(self.objective,np.zeros(self.model.size),jac=True,method='L-BFGS-B',bounds=self.bounds,
            callback=callback,options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-4,'maxcor':20,'maxls':40})
        physical,_=self.physical(result.x)
        state,betas=self.model.unpack(physical/self.model.scale)
        projected=result.jac.copy()
        for i,(lo,hi) in enumerate(self.bounds):
            if lo is not None and result.x[i]<=lo+1e-6 and projected[i]>0:projected[i]=0
            if hi is not None and result.x[i]>=hi-1e-6 and projected[i]<0:projected[i]=0
        return state,betas,{'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),
            'evaluations':int(result.nfev),'objective':float(result.fun),'gradient_transformed_inf':float(abs(result.jac).max()),
            'projected_gradient_transformed_inf':float(abs(projected).max())}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--test-season',type=int,required=True);parser.add_argument('--variant',choices=['macro','joint'],required=True)
    args=parser.parse_args();test=args.test_season;variant=args.variant
    original=OUT/'fits'/str(test)/variant;polished=OUT/'fits_polished'/str(test)/variant;target=OUT/'fits_whitened'/str(test)/variant
    if (target/'manifest.json').exists():print('skip completed full whitening',flush=True);return
    source=polished if (polished/'manifest.json').exists() else original
    manifest,inventory,p=load_fit(source)
    if manifest['solver']['final_success']:print('source converged; no continuation needed',flush=True);return
    data=Canonical.load().subset(seasons=manifest['training_seasons']);index=StateIndex(inventory,data.meta)
    factors,transform=build_factors(data,index,variant,manifest['skating_transform'])
    state=p['state'];betas=[p['beta_'+f.name] for f in factors];model=JointModel(factors,index.n,index.previous,p['skill_cov'])
    audits={'source':str(source),'previous_solver':manifest['solver']};before=model.objective(model.initial(state,betas))[0]
    for attempt in range(2):
        precondition(model,state,betas);coordinates=Coordinates(model,state,betas)
        state,betas,audit=coordinates.fit(900);audits['full_curvature_attempt_'+str(attempt+1)]=audit
        if audit['success']:break
    after=model.objective(model.initial(state,betas))[0]
    if after>before+1e-4:raise AssertionError('Whitened continuation worsened objective')
    covariance=model.curvature(state,betas)
    audits.update(final_success=bool(audit['success']),final_attempt_completed=True,finite_state=bool(np.isfinite(state).all()),
        minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(covariance).min()),objective_before=before,objective_after=after,
        latent_bound_hits=int(np.sum(np.abs(state/np.r_[SKILL_SD,np.full(R,ROLE_SD)])>=7.999)))
    save_fit(target,index,state,covariance,betas,factors,p['skill_cov'],transform,audits,variant,manifest['training_seasons'])
    print(f'Full curvature saved: {audits}',flush=True)

if __name__=='__main__':main()
