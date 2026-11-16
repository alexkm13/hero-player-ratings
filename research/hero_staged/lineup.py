"""Portable saved conditional response and same-position substitution."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import OUT,finite
from .measure import ACTIVITY,EXECUTION,CHANNELS,transform
from .response import CTX,contexts

_CACHE={}

def load(root=OUT):
    root=Path(root);spec=json.loads((root/'conditional_response.json').read_text())
    params=dict(np.load(root/'conditional_response.npz',allow_pickle=False))
    return spec,params

def _predict_matrix(spec,params,own,opp,home=1,period=1,score=0,elapsed=0):
    own=list(map(int,own));opp=list(map(int,opp))
    if len(own)!=5 or len(opp)!=5 or len(set(own+opp))!=10:raise ValueError('ten distinct skaters required')
    if home not in [0,1] or period not in [1,2,3] or not np.isfinite(elapsed) or elapsed<0:raise ValueError('invalid context')
    family=spec['family'];players=params['players'];lookup=pd.Index(players);N=len(players)
    rows=[contexts(np.array([home]),np.array([period]),np.array([score]),np.array([elapsed<3]))[0],
          contexts(np.array([1-home]),np.array([period]),np.array([-score]),np.array([elapsed<3]))[0]]
    channels=ACTIVITY if family=='activity' else (EXECUTION if family=='execution' else (CHANNELS if family=='both' else []))
    for i,(ids,other) in enumerate([(own,opp),(opp,own)]):
        if family!='context':
            vec=np.zeros(2*N);a=lookup.get_indexer(ids);b=lookup.get_indexer(other);vec[a[a>=0]]=1;vec[N+b[b>=0]]=-1
            rows[i]=np.r_[rows[i],vec]
        if channels:
            values=[]
            for group in [ids,other]:
                frame=pd.DataFrame([spec['profiles'].get(str(pid),{'player_id':pid,'position_group':'U'}) for pid in group])
                for c in channels:
                    if c not in frame:frame[c]=np.nan
                    if c+'_den' not in frame:frame[c+'_den']=0
                frame['position_group']=frame.position_group.fillna('U')
                shrunk=transform(frame,channels,spec['priors'])[:,::3]
                center=np.array([np.mean([p[c]['prior'] for p in spec['priors'].values()]) for c in channels])
                values.extend((shrunk-center).sum(axis=0))
            rows[i]=np.r_[rows[i],np.asarray(values)/params['scale']]
    x=finite(rows);result={}
    for head in ['shots','xg','goals']:
        rate=np.exp(np.einsum('ij,j->i',x,params[head],optimize=False))*3600
        result[head+'_for60']=float(rate[0]);result[head+'_against60']=float(rate[1]);result['net_'+head+'60']=float(rate[0]-rate[1])
    result.update({'family':family,'as_of':'2026-09-17','production_admission':False,'development_gate':spec['development_gate'],
        'scope':'conditional additive team response on observed lineup support; not the future-provider Overall estimand',
        'support':'arbitrary requested lineup has no certified joint/causal/replacement support',
        'unknown_players':[p for p in own+opp if str(p) not in spec['profiles']]})
    return result

def _compile(spec,params):
    key=(id(spec),id(params))
    if key in _CACHE:return _CACHE[key][2]
    family=spec['family'];players=params['players'];N=len(players);lookup=pd.Index(players)
    channels=ACTIVITY if family=='activity' else (EXECUTION if family=='execution' else (CHANNELS if family=='both' else []))
    ids=list(set(map(int,players))|set(map(int,spec['profiles'])))+[0]
    features=np.zeros((len(ids),0))
    if channels:
        frame=pd.DataFrame([spec['profiles'].get(str(pid),{'player_id':pid,'position_group':'U'}) for pid in ids])
        for c in channels:
            if c not in frame:frame[c]=np.nan
            if c+'_den' not in frame:frame[c+'_den']=0
        frame['position_group']=frame.position_group.fillna('U');features=transform(frame,channels,spec['priors'])[:,::3]
        center=np.array([np.mean([p[c]['prior'] for p in spec['priors'].values()]) for c in channels]);features-=center
    where=lookup.get_indexer(ids);result={}
    for head in ['shots','xg','goals']:
        b=params[head];off=np.zeros(len(ids));opp=np.zeros(len(ids));start=len(CTX)
        if family!='context':
            valid=where>=0;off[valid]=b[start+where[valid]];opp[valid]=-b[start+N+where[valid]];start+=2*N
        if channels:
            n=len(channels);off+=np.einsum('ij,j->i',features,b[start:start+n]/params['scale'][:n],optimize=False)
            opp+=np.einsum('ij,j->i',features,b[start+n:start+2*n]/params['scale'][n:],optimize=False)
        result[head]={'own':dict(zip(ids,off)),'opp':dict(zip(ids,opp)),'context':b[:len(CTX)]}
    _CACHE[key]=(spec,params,result);return result

def predict(spec,params,own,opp,home=1,period=1,score=0,elapsed=0):
    own=list(map(int,own));opp=list(map(int,opp))
    if len(own)!=5 or len(opp)!=5 or len(set(own+opp))!=10:raise ValueError('ten distinct skaters required')
    if home not in [0,1] or period not in [1,2,3] or not np.isfinite(elapsed) or elapsed<0:raise ValueError('invalid context')
    compiled=_compile(spec,params);ctx=contexts(np.array([home,1-home]),np.array([period]*2),np.array([score,-score]),np.array([elapsed<3]*2))
    result={}
    for head,m in compiled.items():
        eta=np.einsum('ij,j->i',ctx,m['context'],optimize=False)
        eta[0]+=sum(m['own'].get(p,m['own'][0]) for p in own)+sum(m['opp'].get(p,m['opp'][0]) for p in opp)
        eta[1]+=sum(m['own'].get(p,m['own'][0]) for p in opp)+sum(m['opp'].get(p,m['opp'][0]) for p in own)
        rate=np.exp(eta)*3600
        result[head+'_for60']=float(rate[0]);result[head+'_against60']=float(rate[1]);result['net_'+head+'60']=float(rate[0]-rate[1])
    result.update({'family':spec['family'],'as_of':'2026-09-17','production_admission':False,'development_gate':spec['development_gate'],
        'scope':'conditional additive team response on observed lineup support; not the future-provider Overall estimand',
        'support':'arbitrary requested lineup has no certified joint/causal/replacement support',
        'unknown_players':[p for p in own+opp if str(p) not in spec['profiles']]})
    return result

def swap(spec,params,own,opp,remove,add,**context):
    own=list(map(int,own));remove=int(remove);add=int(add)
    if remove not in own:raise ValueError('removed player not in own lineup')
    old=spec['profiles'].get(str(remove));new=spec['profiles'].get(str(add))
    if old is None or new is None or old['position_group'] not in ['F','D'] or old['position_group']!=new['position_group']:
        raise ValueError('substitution requires two observed compatible F/D positions')
    before=predict(spec,params,own,opp,**context);after=predict(spec,params,[add if p==remove else p for p in own],opp,**context)
    result={'before':before,'after':after,'interpretation':'conditional same-role association; not identified causal credit'}
    for head in ['shots','xg','goals']:
        off=after[head+'_for60']-before[head+'_for60'];defense=before[head+'_against60']-after[head+'_against60']
        result[head+'_offense_added60']=off;result[head+'_defense_prevented60']=defense;result[head+'_net_added60']=off+defense
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=OUT);p.add_argument('--own',nargs=5,type=int,required=True)
    p.add_argument('--opp',nargs=5,type=int,required=True);p.add_argument('--home',type=int,default=1);p.add_argument('--period',type=int,default=1)
    p.add_argument('--score',type=int,default=0);p.add_argument('--elapsed',type=float,default=0);p.add_argument('--remove',type=int);p.add_argument('--add',type=int)
    a=p.parse_args();spec,params=load(a.bundle);context={'home':a.home,'period':a.period,'score':a.score,'elapsed':a.elapsed}
    if (a.remove is None)!=(a.add is None):p.error('provide both --remove and --add')
    result=predict(spec,params,a.own,a.opp,**context) if a.remove is None else swap(spec,params,a.own,a.opp,a.remove,a.add,**context)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
