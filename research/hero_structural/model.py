"""Joint anchored generalized-likelihood state model with analytic gradients.

Training MAP fits historical states only inside the completed training window.
Held-out inference fixes shared coefficients and uses a forward local MAP filter.
Within-player curvature is retained; between-player posterior covariance is not.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

K=8
R=6
D=K+R
# Scale/orientation priors in each category's declared anchor units.
SKILL_SD=np.array([.35,.55,.45,.35,.50,.10,.45,1.0])
ROLE_SD=.50
AR=np.r_[np.full(K,.85),np.full(R,.50)]

def unit(dim,value=1.):
    out=np.zeros(D);out[dim]=value;return out

def design(ids,states,values=None):
    """Rows are observations; columns are state IDs. Missing IDs contribute zero."""
    ids=np.asarray(ids,dtype=int)
    if ids.ndim==1:ids=ids[:,None]
    n,width=ids.shape
    row=np.repeat(np.arange(n),width);col=ids.ravel()
    valid=col>=0
    data=np.ones(len(col)) if values is None else np.asarray(values).ravel()
    return sparse.csr_matrix((data[valid],(row[valid],col[valid])),shape=(n,states))

@dataclass
class Factor:
    name:str
    kind:str
    y:np.ndarray
    ctx:np.ndarray
    matrices:list
    fixed:list
    learn:list
    exposure:np.ndarray=None
    variance:float=1.
    choices:int=5
    group:np.ndarray=None
    game:np.ndarray=None
    side:np.ndarray=None

    def __post_init__(self):
        self.y=np.asarray(self.y,dtype=float)
        self.ctx=np.asarray(self.ctx,dtype=float)
        self.fixed=[np.asarray(v,dtype=float) for v in self.fixed]
        self.learn=[np.asarray(v,dtype=bool) for v in self.learn]
        self.slices=[];offset=0
        for mask in self.learn:
            size=int(mask.sum());self.slices.append(slice(offset,offset+size));offset+=size
        self.context_slice=slice(offset,offset+self.ctx.shape[1])
        self.size=offset+self.ctx.shape[1]
        assert all(m.shape[0]==len(self.y) for m in self.matrices)
        assert all(len(v)==D for v in self.fixed)
        self.penalty=np.full(self.size,1/.30**2)
        self.penalty[self.context_slice]=.01

    def coefficients(self,beta):
        result=[]
        for base,mask,sl in zip(self.fixed,self.learn,self.slices):
            v=base.copy();v[mask]+=beta[sl];result.append(v)
        return result

    def eta(self,state,beta):
        vectors=self.coefficients(beta)
        eta=np.einsum('ij,j->i',self.ctx,beta[self.context_slice],optimize=False)
        for m,v in zip(self.matrices,vectors):
            if np.any(v):eta=eta+m@np.einsum('ij,j->i',state,v,optimize=False)
        return eta,vectors

    def loss_residual(self,eta):
        if len(eta)==0:return np.zeros(0),np.zeros(0),np.zeros(0)
        if self.kind=='poisson':
            # A line-search trial can have an implausible but still finite
            # rate. Penalize it through its actual likelihood, allowing the
            # solver to backtrack; do not abort a valid search at exp(30).
            if np.max(eta)>600:raise FloatingPointError('intensity exceeded finite numerical envelope')
            mu=np.exp(eta)*self.exposure
            return mu-self.y*eta,mu-self.y,mu
        if self.kind=='binomial':
            p=expit(eta)
            return np.logaddexp(0,eta)-self.y*eta,p-self.y,p*(1-p)
        if self.kind=='normal':
            residual=(eta-self.y)/self.variance
            return .5*(eta-self.y)**2/self.variance,residual,np.full(len(eta),1/self.variance)
        if self.kind=='choice':
            z=eta.reshape(-1,self.choices)
            logp=z-logsumexp(z,axis=1,keepdims=True)
            target=self.y.reshape(-1,self.choices)
            p=np.exp(logp)
            # One observed actor per group. Other choices remain alternatives.
            return -(target*logp).sum(axis=1),(p-target).ravel(),(p*(1-p)).ravel()
        raise ValueError(self.kind)

    def objective(self,state,beta,with_beta=True):
        eta,vectors=self.eta(state,beta)
        loss,residual,_=self.loss_residual(eta)
        grad=np.zeros_like(state);bg=np.zeros(self.size)
        for m,v,mask,sl in zip(self.matrices,vectors,self.learn,self.slices):
            if np.any(v):grad+=(m.T@residual)[:,None]*v
            if with_beta and mask.any():bg[sl]=np.einsum('ij,i->j',m@state[:,mask],residual,optimize=False)
        if with_beta:bg[self.context_slice]=np.einsum('ij,i->j',self.ctx,residual,optimize=False)
        return float(loss.sum()),grad,bg

    def blocks(self,state,beta):
        eta,vectors=self.eta(state,beta)
        _,_,weight=self.loss_residual(eta)
        result=np.zeros((len(state),D,D))
        for i,(m,v) in enumerate(zip(self.matrices,vectors)):
            if not np.any(v):continue
            for j in range(i,len(vectors)):
                n,w=self.matrices[j],vectors[j]
                if not np.any(w):continue
                common=m.multiply(n)
                if not common.nnz:continue
                mass=np.asarray(common.T@weight).ravel()
                outer=np.outer(v,w)
                if i!=j:outer+=outer.T
                result+=mass[:,None,None]*outer
        return result

    def initial(self):
        b=np.zeros(self.size)
        # Intercept is first context field; the last can be raw logit-xG.
        if self.kind=='poisson':b[self.context_slice.start]=np.log((self.y.sum()+1)/(self.exposure.sum()+1))
        elif self.kind=='binomial':
            p=(self.y.sum()+.5)/(len(self.y)+1)
            b[self.context_slice.start]=np.log(p/(1-p))
            if self.name=='goal_conversion':
                b[self.context_slice.start]=0.;b[self.context_slice.stop-1]=1.
        elif self.kind=='choice':b[self.context_slice.start]=0.
        return b

    def bounds(self):
        out=[(-1.5,1.5)]*self.context_slice.start
        out += [(-15,8)] + [(-3,3)]*(self.ctx.shape[1]-1)
        if self.name=='goal_conversion':out[-1]=(.5,2.)
        return out

class JointModel:
    def __init__(self,factors,nstates,previous=None,skill_cov=None):
        self.factors=factors;self.nstates=nstates
        self.previous=np.full(nstates,-1,dtype=int) if previous is None else previous
        self.skill_cov=np.diag(SKILL_SD**2) if skill_cov is None else skill_cov
        cov=np.zeros((D,D));cov[:K,:K]=self.skill_cov;cov[K:,K:]=np.eye(R)*ROLE_SD**2
        self.precision=np.linalg.inv(cov)
        self.factor_slices=[];offset=nstates*D
        for factor in factors:
            self.factor_slices.append(slice(offset,offset+factor.size));offset+=factor.size
        self.size=offset
        # Global coefficients have many more observations than one state;
        # rescale them to avoid ill-conditioned joint optimization.
        self.scale=np.r_[np.tile(np.r_[SKILL_SD,np.full(R,ROLE_SD)],nstates),np.full(offset-nstates*D,.025)]

    def initial(self,state=None,betas=None):
        if state is None:state=np.zeros((self.nstates,D))
        if betas is None:betas=[f.initial() for f in self.factors]
        return np.r_[state.ravel(),*betas]/self.scale

    def unpack(self,z):
        v=z*self.scale
        return v[:self.nstates*D].reshape(self.nstates,D),[v[s] for s in self.factor_slices]

    def prior(self,state):
        pred=np.zeros_like(state);has=self.previous>=0
        pred[has]=state[self.previous[has]]*AR
        delta=state-pred
        g=np.einsum('ij,jk->ik',delta,self.precision,optimize=False)
        loss=.5*np.sum(g*delta)
        back=np.zeros_like(state)
        np.add.at(back,self.previous[has],-g[has]*AR)
        return loss,g+back

    def objective(self,z):
        state,betas=self.unpack(z);loss,grad=self.prior(state);bg=[]
        for f,b in zip(self.factors,betas):
            l,g,h=f.objective(state,b)
            loss+=l+.5*np.sum(f.penalty*b*b)
            grad+=g;bg.append(h+f.penalty*b)
        g=np.r_[grad.ravel(),*bg]*self.scale
        if not np.isfinite(loss) or not np.isfinite(g).all():raise FloatingPointError('nonfinite joint objective')
        return loss,g

    def fit(self,maxiter=300,state=None,betas=None,callback=None):
        z=self.initial(state,betas)
        physical=np.tile(np.r_[SKILL_SD,np.full(R,ROLE_SD)],self.nstates)*8
        bounds=list(zip(-physical/self.scale[:self.nstates*D],physical/self.scale[:self.nstates*D]))
        for f,sl in zip(self.factors,self.factor_slices):
            bounds.extend([(a/self.scale[sl.start+i],b/self.scale[sl.start+i]) for i,(a,b) in enumerate(f.bounds())])
        iteration=[0]
        def cb(x):
            iteration[0]+=1
            if callback and iteration[0]%20==0:
                loss,grad=self.objective(x);callback(iteration[0],loss,float(np.max(np.abs(grad))))
        result=minimize(self.objective,z,jac=True,method='L-BFGS-B',bounds=bounds,callback=cb,
                        options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-4,'maxcor':12,'maxls':30})
        state,betas=self.unpack(result.x)
        audit={'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),'objective':float(result.fun),'gradient_scaled_inf':float(np.max(np.abs(result.jac)))}
        projected=result.jac.copy();limits=np.asarray(bounds)
        projected[((result.x<=limits[:,0]+1e-6)&(projected>0))|((result.x>=limits[:,1]-1e-6)&(projected<0))]=0
        audit['projected_gradient_scaled_inf']=float(np.max(np.abs(projected)))
        return state,betas,audit

    def curvature(self,state,betas):
        h=np.repeat(self.precision[None,:,:],self.nstates,axis=0)
        # A training state is also constrained by its child's annual transition.
        has=self.previous>=0
        np.add.at(h,self.previous[has],AR[:,None]*self.precision*AR[None,:])
        for f,b in zip(self.factors,betas):h+=f.blocks(state,b)
        return np.linalg.inv(h)

def empirical_skill_cov(state,previous):
    """Training-only regularized covariance, deliberately conservative.

    Posterior means attenuate covariance; this is a prior sensitivity estimate,
    not a recovered distribution of true player talent.
    """
    has=previous>=0
    delta=state[has,:K]-.85*state[previous[has],:K]
    if len(delta)<30:return np.diag(SKILL_SD**2)
    c=np.cov(delta,rowvar=False)
    scale=np.sqrt(np.maximum(np.diag(c),1e-8))
    corr=c/np.outer(scale,scale)
    # Half learned correlation, half identity gives a PD regularized prior.
    corr=.5*corr+.5*np.eye(K)
    return corr*np.outer(SKILL_SD,SKILL_SD)

def filter_update(factors,state,cov,betas,maxiter=80):
    """One date's outcomes update states only after that date is scored."""
    prior=state.copy();precision=np.linalg.inv(cov)
    # Whiten the full carried prior covariance. With coefficients fixed, this
    # is an exact linear reparameterization of the same convex MAP objective.
    # Fixed marginal skill scales are poorly conditioned after months of data.
    root=np.linalg.cholesky(cov)
    def objective(z):
        current=prior+np.einsum('nij,nj->ni',root,z.reshape(state.shape),optimize=False)
        delta=current-prior;g=np.einsum('nij,nj->ni',precision,delta,optimize=False)
        loss=.5*np.sum(g*delta)
        for f,b in zip(factors,betas):
            l,v,_=f.objective(current,b,False);loss+=l;g+=v
        gz=np.einsum('nji,nj->ni',root,g,optimize=False)
        return float(loss),gz.ravel()
    initial=np.zeros(state.size)
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',
                 options={'maxiter':maxiter,'gtol':2e-4,'ftol':1e-9,'maxcor':6,'maxls':25})
    first={'success':bool(fit.success),'iterations':int(fit.nit),'message':str(fit.message)}
    if not fit.success:
        fit=minimize(objective,fit.x,jac=True,method='L-BFGS-B',
                 options={'maxiter':maxiter,'gtol':2e-4,'ftol':1e-9,'maxcor':10,'maxls':30})
    current=prior+np.einsum('nij,nj->ni',root,fit.x.reshape(state.shape),optimize=False)
    h=precision.copy()
    for f,b in zip(factors,betas):h+=f.blocks(current,b)
    covariance=np.linalg.inv(h)
    return current,covariance,{'success':bool(fit.success),'iterations':int(fit.nit),'objective':float(fit.fun),'gradient_inf':float(np.max(np.abs(fit.jac))),'message':str(fit.message),'initial_attempt':first,'coordinates':'full prior covariance Cholesky'}
