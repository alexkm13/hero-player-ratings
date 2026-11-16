"""Observation contract and canonical-data adapters for the joint model."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .common import DATA
from .model import D,K,Factor,design,unit

RESP={'shot':8,'entry':9,'retrieval':10,'exit':11,'pass_a1':12,'recoverer':13}

def mask(*dims):
    out=np.zeros(D,dtype=bool);out[list(dims)]=True;return out

def zero():return np.zeros(D)

@dataclass
class Canonical:
    risk:pd.DataFrame
    shots:pd.DataFrame
    micro:pd.DataFrame
    roots:pd.DataFrame
    meta:pd.DataFrame
    skating:pd.DataFrame

    @classmethod
    def load(cls):
        r=pd.read_parquet(DATA/'risk.parquet')
        r['h']=r.home_players.map(lambda x:np.asarray(x,dtype=np.int64))
        r['a']=r.away_players.map(lambda x:np.asarray(x,dtype=np.int64))
        return cls(r,pd.read_parquet(DATA/'shots.parquet'),pd.read_parquet(DATA/'linked_micro.parquet'),
                   pd.read_parquet(DATA/'roots.parquet'),pd.read_parquet(DATA/'player_meta.parquet'),pd.read_parquet(DATA/'skating.parquet'))

    def subset(self,seasons=None,date=None):
        select=lambda f: (f[f.season_id.isin(seasons)] if 'season_id' in f else f[f.season.isin(seasons)]) if seasons is not None else f
        if date is not None:
            select=lambda f:f[f.date.eq(date)] if 'date' in f else f.iloc[:0]
        return Canonical(select(self.risk),select(self.shots),select(self.micro),select(self.roots),self.meta,select(self.skating))

    def state_inventory(self):
        parts=[]
        for seas,g in self.risk.groupby('season'):
            ids=np.unique(np.r_[np.stack(g.h).ravel(),np.stack(g.a).ravel(),self.shots.loc[self.shots.season.eq(seas),'goalie_id'].to_numpy()])
            for pid in ids[ids>0]:parts.append((int(seas),int(pid)))
        return pd.DataFrame(parts,columns=['season','player_id']).sort_values(['season','player_id']).reset_index(drop=True)

class StateIndex:
    def __init__(self,inventory,meta):
        self.inventory=inventory
        self.n=len(inventory)
        self.seasons=sorted(set(inventory.season))
        self.players=np.unique(inventory.player_id)
        self.pindex=pd.Index(self.players)
        self.lookup=np.full((len(self.seasons),len(self.players)),-1,dtype=int)
        self.sindex={s:i for i,s in enumerate(self.seasons)}
        for i,r in inventory.iterrows():self.lookup[self.sindex[r.season],self.pindex.get_loc(r.player_id)]=i
        self.previous=np.full(self.n,-1,dtype=int)
        self.role=np.full(self.n,'U',dtype=object)
        # Role is a structural prior from a completed earlier season, not a
        # future-season final aggregate. Unknown role remains explicit.
        hist={pid:g.sort_values('season_id') for pid,g in meta.groupby('player_id')}
        last={}
        for i,r in inventory.iterrows():
            if r.player_id in last:self.previous[i]=last[r.player_id]
            last[r.player_id]=i
            g=hist.get(r.player_id)
            if g is not None:
                past=g[g.season_id.lt(r.season)]
                if len(past):self.role[i]=str(past.iloc[-1].position_group)

    def map(self,seasons,pids):
        pids=np.asarray(pids,dtype=np.int64)
        shape=pids.shape
        seas=np.asarray(seasons,dtype=np.int64)
        if seas.ndim==0:seas=np.full(shape,seas)
        elif pids.ndim==2 and seas.ndim==1:seas=np.repeat(seas[:,None],shape[1],axis=1)
        pos=self.pindex.get_indexer(pids.ravel())
        spos=np.array([self.sindex.get(s,-1) for s in seas.ravel()],dtype=int)
        keep=(pos>=0)&(spos>=0)
        out=np.full(pids.size,-1,dtype=int)
        out[keep]=self.lookup[spos[keep],pos[keep]]
        return out.reshape(shape)

def ctx(home,role=None,extra=None):
    values=[np.ones(len(home)),np.asarray(home)]
    if role is not None:values.append(np.asarray(role,dtype=float))
    if extra is not None:values.append(np.asarray(extra,dtype=float))
    return np.column_stack(values)

def build_factors(data,index,variant='joint',skate_transform=None,include_skating=True):
    """Compile all factors on identical macro rows, with no Overall/RAPM input."""
    n=index.n;factors=[];r=data.risk.reset_index(drop=True)
    if len(r)==0:return [],skate_transform
    h=np.stack(r.h);a=np.stack(r.a);seas=r.season.to_numpy()
    hi=index.map(seas,h);ai=index.map(seas,a)
    if (hi<0).any() or (ai<0).any():raise ValueError('risk player absent from state inventory')
    own=np.stack([hi,ai],axis=1).reshape(-1,5);opp=np.stack([ai,hi],axis=1).reshape(-1,5)
    O=design(own,n);P=design(opp,n)
    rowmap=np.full(int(data.risk.interval_index.max())+1,-1,dtype=int)
    rowmap[r.interval_index.to_numpy(int)]=np.arange(len(r))
    s=data.shots.copy();s['local']=rowmap[s.interval_index.to_numpy(int)]
    if s.local.lt(0).any():raise ValueError('shot risk mismatch')
    count=np.bincount(2*s.local.to_numpy(int)+s.side.to_numpy(int),minlength=2*len(r))
    home=np.tile([1.,0.],len(r));games=np.repeat(r.game.to_numpy(),2);sides=np.tile([0,1],len(r))
    own_mask=mask(0) if variant=='macro' else mask(0,1,2,4,7)
    factors.append(Factor('shot_incidence','poisson',count,ctx(home),[O,P],
                          [zero(),unit(5,-1)], [own_mask,mask()],
                          exposure=np.repeat(r.seconds.to_numpy(float),2),game=games,side=sides))

    def event_lineups(e):
        loc=rowmap[e.interval_index.to_numpy(int)]
        if (loc<0).any():raise ValueError('micro risk mismatch')
        side=e.side.to_numpy(int)
        ownids=np.where(side[:,None]==0,hi[loc],ai[loc])
        oppids=np.where(side[:,None]==0,ai[loc],hi[loc])
        return loc,ownids,oppids

    def binary(name,e,target,actor_vec,opp_vec=None,own_learn=None,opp_learn=None,actor_column='player_id',defender=False,extra=None):
        if len(e)==0:return
        _,oi,pi=event_lineups(e)
        pid=e[actor_column].to_numpy(int)
        actor=index.map(e.season.to_numpy() if 'season' in e and np.issubdtype(e.season.dtype,np.number) else e.season_id.to_numpy(),pid)
        matrices=[design(oi,n),design(pi,n),design(actor,n)]
        fixed=[zero(),zero() if opp_vec is None else opp_vec,actor_vec]
        learn=[mask() if own_learn is None else own_learn,mask() if opp_learn is None else opp_learn,mask()]
        role=np.where(actor>=0,index.role[np.maximum(actor,0)]=='D',False)
        if defender:
            defi=index.map(e.season_id.to_numpy(),e.defender_id.to_numpy(int))
            good=(pi==defi[:,None]).any(axis=1)&(defi>=0)
            defi=np.where(good,defi,-1)
            matrices.append(design(defi,n));fixed.append(unit(6,-1));learn.append(mask())
            extra=np.where(good,0.,1.)
        factor=Factor(name,'binomial',np.asarray(target,dtype=float),ctx(1-e.side.to_numpy(),role,extra),matrices,fixed,learn,game=e.game.to_numpy(),side=e.side.to_numpy())
        factors.append(factor)

    # Actor allocation is a single softmax choice for one observed action.
    def allocation(name,e,cat,responsibility,actor_column='player_id'):
        if len(e)==0:return
        _,oi,_=event_lineups(e)
        season=e.season.to_numpy() if 'season' in e and np.issubdtype(e.season.dtype,np.number) else e.season_id.to_numpy()
        actual=index.map(season,e[actor_column].to_numpy(int))
        valid=(oi==actual[:,None]).any(axis=1)&(actual>=0)
        e=e.loc[valid];oi=oi[valid];actual=actual[valid]
        if len(e)==0:return
        y=(oi==actual[:,None]).astype(float).ravel()
        role=index.role[oi.ravel()]=='D'
        # Constant and venue cancel inside the choice set but are retained for
        # a stable coefficient schema; regularization identifies them at zero.
        vector=unit(cat)+unit(responsibility)
        factor=Factor(name,'choice',y,ctx(np.repeat(1-e.side.to_numpy(),5),role),[design(oi.ravel(),n)],[vector],[mask()],choices=5,game=e.game.to_numpy(),side=e.side.to_numpy())
        factors.append(factor)

    s['player_id']=s.shooter_id
    allocation('shooter_allocation',s,0,RESP['shot'])
    binary('shot_quality',s,s.dangerous,unit(0),own_learn=mask() if variant=='macro' else mask(1,2,7),opp_learn=mask(5) if variant=='macro' else mask(5,6))
    # Goal conversion is conditioned on the exact observed xG. It is not
    # appended as an independent macro aggregate or RAPM coefficient.
    binary('goal_conversion',s,s.goal,unit(3),actor_column='shooter_id',extra=np.log(np.clip(s.xg.to_numpy(),1e-5,1-1e-5)/(1-np.clip(s.xg.to_numpy(),1e-5,1-1e-5))))
    if len(s):
        conversion=factors[-1]
        goalie=index.map(s.season.to_numpy(),s.goalie_id.to_numpy(int))
        conversion.matrices.append(design(goalie,n));conversion.fixed.append(unit(3,-1));conversion.learn.append(mask())
        # The appended matrix has no new learned coefficient.
        conversion.slices.append(slice(0,0))
    if variant=='macro':return factors,skate_transform

    roots=data.roots.copy()
    for family,name in [(1,'entry'),(2,'retrieval'),(3,'exit')]:
        e=roots[roots.family.eq(family)]
        # Exposure is all verified tracked games, not only intervals containing
        # a positive event of this family.
        tracked=np.isin(games,np.unique(data.micro.game))
        positions=np.flatnonzero(tracked)
        counts=np.zeros(2*len(r))
        if len(e):
            loc=rowmap[e.interval_index.to_numpy(int)]
            np.add.at(counts,2*loc+e.side.to_numpy(int),1)
        fixed_op=zero()
        own_dims=(1,2,4,7) if family==1 else ((4,6,7) if family==2 else (1,6,7))
        factors.append(Factor(name+'_incidence','poisson',counts[positions],ctx(home[positions]),[O[positions],P[positions]],
                              [zero(),fixed_op],[mask(*own_dims),mask(6)],exposure=np.repeat(r.seconds.to_numpy(),2)[positions],game=games[positions],side=sides[positions]))
        known=e[e.player_id.gt(0)]
        allocation(name+'_allocation',known,1 if family in [1,3] else 6,RESP[name])
    entry=roots[roots.family.eq(1)&roots.entry_type.str.upper().isin(['C','D','F'])&roots.player_id.gt(0)].copy()
    binary('entry_execution',entry,entry.entry_type.str.upper().ne('F'),unit(1),defender=True)
    success=entry[entry.entry_type.str.upper().isin(['C','D'])]
    binary('entry_choice',success,success.entry_type.str.upper().eq('C'),unit(RESP['entry']),own_learn=mask(1,7))
    retrieval=roots[roots.family.eq(2)&roots.entry_result.str.upper().isin(['EXC','BOT','CEX','PEX','CLE','MEX','FEX'])&roots.player_id.gt(0)]
    binary('retrieval_execution',retrieval,retrieval.entry_result.str.upper().ne('BOT'),unit(6))
    exits=data.micro[data.micro.action_type.eq('exit')&data.micro.exit_result.str.upper().isin(['CEX','PEX','CLE','MEX','FEX'])].copy()
    binary('exit_execution',exits,exits.exit_result.str.upper().isin(['CEX','PEX']),unit(1))
    passer=data.micro[data.micro.action_type.eq('pass_a1')]
    allocation('primary_passer_allocation',passer,2,RESP['pass_a1'])
    dumps=entry[entry.entry_type.str.upper().eq('D')].copy()
    raw=dumps.dump_recovered.str.upper()
    observed=raw.isin(['N','NN'])|dumps.recoverer_id.gt(0)
    dumps=dumps[observed]
    binary('dump_recovery',dumps,dumps.recoverer_id.gt(0),zero(),own_learn=mask(4,7),opp_learn=mask(6))
    allocation('recoverer_allocation',dumps[dumps.recoverer_id.gt(0)],4,RESP['recoverer'],'recoverer_id')
    if include_skating:
        e=data.skating.copy()
        ids=index.map(e.season_id.to_numpy(),e.player_id.to_numpy())
        valid=(ids>=0)&e.edge_speed_max_mph.notna().to_numpy()
        e=e[valid];ids=ids[valid]
        if len(e):
            if skate_transform is None:
                skate_transform={'mean':float(e.edge_speed_max_mph.mean()),'sd':float(e.edge_speed_max_mph.std(ddof=0)), 'source':'prior-season all-state EDGE maximum speed','variance':.8}
            z=(e.edge_speed_max_mph.to_numpy()-skate_transform['mean'])/skate_transform['sd']
            factors.append(Factor('skating_measurement','normal',z,np.ones((len(e),1)),[design(ids,n)],[unit(7)],[mask()],variance=skate_transform['variance'],game=np.full(len(e),-1),side=np.zeros(len(e),dtype=int)))
    return factors,skate_transform
