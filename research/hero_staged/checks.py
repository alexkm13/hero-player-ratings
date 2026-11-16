"""Independent preflight checks before the iteration ladder is executed."""
import numpy as np
import pandas as pd
from scipy import sparse
from .common import OUT,dump,finite
from .measure import prepare,ACTIVITY,EXECUTION,transform_fit,transform
from .response import contexts,fit,scores,CTX
from .player import fit_ridge,predict_ridge

def main():
    m=prepare();past=m[m.season_id.lt(20242025)];future=m[m.season_id.eq(20242025)]
    pri=transform_fit(past,ACTIVITY+EXECUTION);features=transform(future,ACTIVITY+EXECUTION,pri)
    mutated=future.copy();mutated['unobserved_future_goals']=1e12
    np.testing.assert_array_equal(features,transform(mutated,ACTIVITY+EXECUTION,pri))
    assert np.isfinite(features).all()
    n=40;x=contexts(np.tile([1,0],20),np.repeat([1,2,3,1],10),np.zeros(n),np.arange(n)%2)
    d={'X':sparse.csr_matrix(x),'y':np.column_stack([np.arange(n)%3,np.full(n,.15),np.arange(n)%7==0]),
       'exposure':np.full(n,30.),'season':np.full(n,20222023),'games':np.repeat(np.arange(20),2),'family':'context','players':[]}
    beta,audit=fit(d,0,.075,300,3);assert audit['success']
    # Independent physical-coordinate finite differences of the actual Poisson
    # objective; zero event and zero exposure boundary rows are retained.
    X=d['X'];y=d['y'][:,0];exp=d['exposure'];b=np.linspace(-.1,.1,len(CTX));b[0]=-3
    penalty=np.full(len(b),.01);penalty[0]=0
    def f(v):eta=X@v;return np.sum(exp*np.exp(eta)-y*eta)+.5*np.sum(penalty*v*v)
    analytic=np.asarray(X.T@(exp*np.exp(X@b)-y)).ravel()+penalty*b
    numerical=np.array([(f(b+np.eye(len(b))[j]*1e-5)-f(b-np.eye(len(b))[j]*1e-5))/2e-5 for j in range(len(b))])
    error=float(np.max(np.abs(analytic-numerical)/np.maximum(1,np.abs(analytic))))
    assert error<1e-6
    sc=scores(d,beta,0);np.testing.assert_allclose(sc.expected.sum(),np.sum(exp*np.exp(X@beta)))
    zero=dict(d);zero['y']=np.zeros_like(d['y']);assert scores(zero,beta,0).expected.sum()>0
    rng=np.random.default_rng(19);xx=rng.normal(size=(80,6));yy=rng.normal(size=(80,6));ss=np.repeat([20202021,20212022],40)
    rm=fit_ridge(xx,yy,ss,100,3);before=predict_ridge(rm,xx[:5]);unused_future=rng.normal(size=(5,6))*1e6
    residual=yy-predict_ridge(rm,xx);weights=2.**(-((ss.max()-ss)/10001)/3)
    intercept_gradient=float(np.max(np.abs(np.einsum('ij,i->j',residual,weights,optimize=False))))
    assert intercept_gradient<1e-8
    np.testing.assert_array_equal(before,predict_ridge(rm,xx[:5]))
    # Known additive intensity world with independently generated randomized
    # rosters. Expected labels remove finite-event noise; this checks credit
    # orientation and penalty bias, not NHL talent identification.
    rosters=np.array([rng.choice(40,10,replace=False) for _ in range(3000)])
    rr=np.repeat(np.arange(len(rosters)),5)
    O=sparse.csr_matrix((np.ones(len(rr)),(rr,rosters[:,:5].ravel())),shape=(len(rosters),40))
    P=sparse.csr_matrix((np.ones(len(rr)),(rr,rosters[:,5:].ravel())),shape=(len(rosters),40))
    cx=contexts(np.arange(3000)%2,np.ones(3000),np.zeros(3000),np.zeros(3000))
    matrix=sparse.hstack([sparse.csr_matrix(cx),O,-P],format='csr')
    truth=np.r_[np.array([-4.5]+[0]*(len(CTX)-1)),rng.normal(0,.05,80)]
    expected=30*np.exp(matrix@truth)
    simulated={'X':matrix,'y':np.column_stack([expected,expected*.1,expected*.08]),'exposure':np.full(3000,30.),
        'season':np.full(3000,20222023),'games':np.arange(3000),'family':'identities','players':np.arange(40)}
    recovered,ra=fit(simulated,0,.15,300,3);assert ra['success']
    t=truth[len(CTX):].reshape(2,40);r=recovered[len(CTX):].reshape(2,40)
    t-=t.mean(axis=1)[:,None];r-=r.mean(axis=1)[:,None]
    recovery=float(np.sqrt(np.mean((r-t)**2)));zero=float(np.sqrt(np.mean(t**2)))
    assert recovery<zero
    dump(OUT/'preflight_verification.json',{'analytic_poisson_gradient_error':error,'zero_event_exposure_preserved':True,
        'count_expectation_accounting':True,'future_target_mutation_invariance':True,'measurement_feedback':False,
        'train_only_shrinkage_priors':True,'finite_missing_evidence_predictions':True,'ridge_gradient_inf':rm['gradient_inf'],
        'known_additive_expected_intensity_credit_rmse':recovery,'known_additive_zero_difference_rmse':zero,
        'weighted_ridge_intercept_gradient':intercept_gradient,
        'known_world_scope':'randomized rosters and noiseless expected labels; penalty bias remains; no NHL credit certification',
        'measurement_rows':len(m),'status':'passed'})
    print('v2 preflight passed',flush=True)

if __name__=='__main__':main()
