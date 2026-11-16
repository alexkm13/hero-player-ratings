"""Integrity and numerical checks exercising observed risk and actual factors."""
import warnings
import numpy as np
from .common import OUT,dump,SEED
from .factors import Canonical,StateIndex,build_factors
from .model import JointModel,filter_update,D

def main():
    warnings.simplefilter('error',RuntimeWarning)
    all_data=Canonical.load();first=sorted(all_data.risk.date.unique())[0]
    data=all_data.subset(date=first);index=StateIndex(data.state_inventory(),data.meta)
    factors,_=build_factors(data,index,include_skating=False)
    model=JointModel(factors,index.n,index.previous)
    rng=np.random.default_rng(SEED);z=model.initial()+rng.normal(0,.01,model.size)
    value,gradient=model.objective(z)
    candidates=np.r_[rng.choice(index.n*D,min(30,index.n*D),replace=False),np.arange(index.n*D,model.size)]
    errors=[]
    for i in candidates:
        eps=1e-4;lo=z.copy();hi=z.copy();lo[i]-=eps;hi[i]+=eps
        numerical=(model.objective(hi)[0]-model.objective(lo)[0])/(2*eps)
        errors.append(abs(numerical-gradient[i])/max(1,abs(numerical),abs(gradient[i])))
    assert max(errors)<2e-5, max(errors)
    from .polish import precondition
    from .whiten import Coordinates
    ss,bb=model.unpack(z);precondition(model,ss,bb)
    coordinates=Coordinates(model,ss,bb);zz=np.zeros(model.size)
    vv,gg=coordinates.objective(zz);whiten_errors=[]
    for i in candidates:
        eps=1e-4;lo=zz.copy();hi=zz.copy();lo[i]-=eps;hi[i]+=eps
        numerical=(coordinates.objective(hi)[0]-coordinates.objective(lo)[0])/(2*eps)
        whiten_errors.append(abs(numerical-gg[i])/max(1,abs(numerical),abs(gg[i])))
    assert max(whiten_errors)<2e-5
    # Restore original test coordinates before checking the remaining factors.
    model=JointModel(factors,index.n,index.previous)
    state,betas=model.unpack(z)
    covariance=model.curvature(state,betas)
    assert np.linalg.eigvalsh(covariance).min()>0
    shot=factors[0]
    assert np.any(shot.y==0) and np.all(shot.exposure>0)
    assert len(shot.y)==2*len(data.risk)
    noevent=data.subset(date=first)
    noevent.shots=noevent.shots.iloc[:0];noevent.micro=noevent.micro.iloc[:0];noevent.roots=noevent.roots.iloc[:0]
    nof,_=build_factors(noevent,index,include_skating=False)
    no_model=JointModel(nof,index.n,index.previous)
    no_loss,no_gradient=no_model.objective(no_model.initial())
    assert np.isfinite(no_loss) and np.isfinite(no_gradient).all()
    assert nof[0].exposure.sum()==2*data.risk.seconds.sum() and nof[0].y.sum()==0
    assert not all_data.roots.duplicated(['source_file','source_row']).any(), 'multiple root opportunities per source row'
    assert not all_data.shots.duplicated(['game','shot_id']).any(), 'duplicate macro shots'
    sums=[]
    for f,b in zip(factors,betas):
        if f.kind!='choice':continue
        from scipy.special import logsumexp
        eta,_=f.eta(state,b);eta=eta.reshape(-1,5)
        p=np.exp(eta-logsumexp(eta,axis=1,keepdims=True))
        sums.append(float(np.max(np.abs(p.sum(axis=1)-1))))
        assert np.all(f.y.reshape(-1,5).sum(axis=1)==1)
    assert max(sums,default=0)<1e-12
    # Future-date mutation cannot enter the day's compiler or initial state.
    reference=model.objective(z)[0]
    changed=all_data.shots.copy();future=changed.date.ne(first)
    changed.loc[future,'goal']=1-changed.loc[future,'goal']
    all_data.shots=changed
    altered=all_data.subset(date=first)
    ff,_=build_factors(altered,index,include_skating=False)
    alternate=JointModel(ff,index.n,index.previous)
    assert alternate.objective(z)[0]==reference
    posterior,posterior_cov,audit=filter_update(factors,state,covariance,betas,120)
    assert np.isfinite(posterior).all() and np.linalg.eigvalsh(posterior_cov).min()>0
    assert np.array_equal(state,model.unpack(z)[0]), 'filter mutated prediction input'
    from .lineup import Response
    response=Response.joint({f.name:b for f,b in zip(factors,betas)},[[.03]*20,[.20]*20])
    risk=data.risk.iloc[:20];h=index.map(risk.season.to_numpy(),np.stack(risk.h));a=index.map(risk.season.to_numpy(),np.stack(risk.a))
    own=state[h].copy();opp=state[a].copy();roles=index.role[h]=='D';opproles=index.role[a]=='D'
    gf,shots60,_=response.goals(own,opp,roles,home=1.)
    eta,_=shot.eta(state,betas[0])
    assert np.allclose(shots60,np.exp(eta[:40:2])*3600), 'lineup response differs from fitted shot likelihood'
    better=own.copy();better[...,3]+=.1
    gf_better,shots_better,_=response.goals(better,opp,roles,home=1.)
    assert np.all(gf_better>gf) and np.array_equal(shots_better,shots60)
    suppression=own.copy();suppression[...,5]+=.05
    ga=response.goals(opp,own,opproles,home=0.)[0]
    ga_suppressed=response.goals(opp,suppression,opproles,home=0.)[0]
    assert np.all(ga_suppressed<ga)
    role_shift=own.copy();role_shift[:,0,8]+=.2
    assert np.array_equal(response.goals(role_shift,opp,roles,home=1.)[1],shots60), 'allocation changed opportunity intensity'
    try:response.adaptive()
    except NotImplementedError:pass
    else:raise AssertionError('unvalidated adaptation silently enabled')
    result={'gradient_parameters':len(candidates),'maximum_scaled_finite_difference_error':max(errors),
        'zero_shot_risk_fraction':float(np.mean(shot.y==0)), 'risk_rows':len(data.risk),
        'softmax_maximum_sum_error':max(sums,default=0),
        'minimum_covariance_eigenvalue':float(np.linalg.eigvalsh(covariance).min()),
        'future_outcome_mutation_invariance':True,'prediction_input_immutable':True,
        'filter_smoke_solver':audit,'test_date':str(first),'RuntimeWarnings':'errors'}
    result['lineup_likelihood_agreement']=True
    result['finishing_goal_conversion_monotonicity']=True
    result['opposing_suppression_monotonicity']=True
    result['responsibility_does_not_create_opportunities']=True
    result['unvalidated_adaptation_disabled']=True
    result['maximum_whitened_gradient_error']=max(whiten_errors)
    result['zero_event_day_preserves_exposure']=True
    result['one_root_per_source_row']=True
    result['unique_macro_shots']=True
    dump(OUT/'verification.json',result);print(result)

if __name__=='__main__':main()
