"""Supported fixed-role research contrasts, integrating marks to goal rates.

This is a two-bin shot-quality approximation. It is neither a puck-state EPV
engine nor a causal player-swap identification claim.
"""
import argparse,json
import numpy as np
from scipy.special import expit,logsumexp
from .common import OUT
from .model import D

class Response:
    def __init__(self,betas,quality_grid,goalie_effect=0.):
        self.betas=betas;self.grid=np.asarray(quality_grid);self.goalie_effect=goalie_effect
    @staticmethod
    def dot(x,w):return np.einsum('...j,j->...',x,w,optimize=False)
    def goals(self,own,opp,own_defender,home=1.):
        """Return GF/60, shots/60, dangerous proportion for batch 5-player sides."""
        own=np.asarray(own);opp=np.asarray(opp);own_defender=np.asarray(own_defender)
        if own.ndim==2:own=own[None];opp=opp[None];own_defender=own_defender[None]
        b=self.betas['shot_incidence'];own_dims=self.own_dims
        eta=self.dot(own.sum(axis=1)[...,own_dims],b[:len(own_dims)])-opp[...,5].sum(axis=1)+b[-2]+b[-1]*home
        shots=np.exp(eta)*3600
        b=self.betas['shooter_allocation']
        logits=own[...,0]+own[...,8]+b[-1]*own_defender
        share=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        b=self.betas['shot_quality'];nd=len(self.quality_own_dims);npd=len(self.quality_opp_dims)
        quality_eta=self.dot(own.sum(axis=1)[...,self.quality_own_dims],b[:nd])+self.dot(opp.sum(axis=1)[...,self.quality_opp_dims],b[nd:nd+npd])
        quality_eta=quality_eta[:,None]+own[...,0]+b[-3]+b[-2]*home+b[-1]*own_defender
        quality=expit(quality_eta)
        b=self.betas['goal_conversion']
        goal_eta=own[...,3]-self.goalie_effect+b[-4]+b[-3]*home+b[-2]*own_defender
        grid_logit=np.log(self.grid/(1-self.grid))
        low=expit(goal_eta[...,None]+b[-1]*grid_logit[0]).mean(axis=-1)
        high=expit(goal_eta[...,None]+b[-1]*grid_logit[1]).mean(axis=-1)
        probability=(1-quality)*low+quality*high
        return shots*(share*probability).sum(axis=1),shots,(share*quality).sum(axis=1)
    @classmethod
    def joint(cls,betas,quality_grid,goalie_effect=0.):
        result=cls(betas,quality_grid,goalie_effect)
        result.own_dims=[0,1,2,4,7];result.quality_own_dims=[1,2,7];result.quality_opp_dims=[5,6]
        return result
    @classmethod
    def macro(cls,betas,quality_grid,goalie_effect=0.):
        result=cls(betas,quality_grid,goalie_effect)
        result.own_dims=[0];result.quality_own_dims=[];result.quality_opp_dims=[5]
        return result
    def paired(self,own,opp,roles_own,roles_opp,home=.5):
        gf,shotsf,qf=self.goals(own,opp,roles_own,home)
        ga,shotsa,qa=self.goals(opp,own,roles_opp,1-home)
        return {'gf60':gf,'ga60':ga,'net60':gf-ga,'shots_for60':shotsf,'shots_against60':shotsa,
                'dangerous_for':qf,'dangerous_against':qa}
    def adaptive(self,*args,**kwargs):
        raise NotImplementedError('Adaptive roles and complementarity are not admitted in v1; require separate held-out mechanism tests')

def main():
    parser=argparse.ArgumentParser(description='Conditional fixed-role lineup research response')
    parser.add_argument('--home',type=int,nargs=5,required=True);parser.add_argument('--away',type=int,nargs=5,required=True)
    parser.add_argument('--adaptive',action='store_true');args=parser.parse_args()
    if len(set(args.home+args.away))!=10:raise ValueError('Ten distinct eligible players required')
    spec=json.loads((OUT/'lineup_response.json').read_text());p=dict(np.load(OUT/'current_response.npz'))
    ids=p['player_id'];lookup={int(pid):i for i,pid in enumerate(ids)}
    for pid in args.home+args.away:
        if pid not in lookup:raise ValueError(f'Unmeasured or ineligible skater {pid}; no automatic average imputation')
    h=np.array([lookup[x] for x in args.home]);a=np.array([lookup[x] for x in args.away])
    response=Response.joint({k[5:]:v for k,v in p.items() if k.startswith('beta_')},spec['quality_grid'],spec['goalie_effect'])
    if args.adaptive:response.adaptive()
    value=response.paired(p['state'][h],p['state'][a],p['position'][h]=='D',p['position'][a]=='D',home=1.)
    result={k:float(v[0]) for k,v in value.items()}
    result.update({'scope':'observed-support conditional research response; fixed responsibility; average historical goalie; two-bin xG integration',
        'support_warning':'single requested lineup has no certified joint support; use exported observed environment panels for valuation',
        'as_of':spec['as_of'],'production_admission':False})
    result['development_event_gate']=spec.get('development_event_gate','unavailable; read evaluation_summary.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
