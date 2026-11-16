"""Shared identification diagnostics; these are not production rating code."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import sparse

from research.paths import ARCHIVE as ROOT, PCV, output

OUT=output('identification')
YEARS=(20212022,20222023,20232024,20242025,20252026)
MAIN=YEARS[-3:]
PENALTIES=(5.,20.,80.)
RISK=ROOT/'work/canonical_v1/risk.parquet'
SHOTS=ROOT/'work/canonical_v1/shots.parquet'
LEDGER=ROOT/'outputs/action_interval_v3/canonical_intervals.parquet'
CONTEXT=ROOT/'outputs/interval_opportunity_v2/calibrated_intervals_with_shots.parquet'
PG=ROOT/'outputs/changed_lineup_v1/player_games.parquet'
TG=ROOT/'outputs/changed_lineup_v1/team_games.parquet'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def dump(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def team(x):
    return {'ARI':'UTA','UTAH':'UTA','LA':'LAK','NJ':'NJD','SJ':'SJS','TB':'TBL'}.get(str(x).replace('.','').upper(),str(x).replace('.','').upper())

def load_risk(year):
    return pd.read_parquet(RISK,filters=[('season','=',year)])

def incidence(ids,players):
    ix=pd.Index(players).get_indexer(ids.ravel())
    if np.any(ix<0):raise ValueError('Unmapped NHL player ID.')
    return sparse.coo_matrix((np.ones(ids.size),(np.repeat(np.arange(len(ids)),5),ix)),shape=(len(ids),len(players))).tocsr()

def grams(q,players):
    h=np.stack(q.home_players.to_numpy()).astype(np.int64)
    a=np.stack(q.away_players.to_numpy()).astype(np.int64)
    for ids in [h,a]:
        assert ids.shape==(len(q),5)
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
    assert not np.any(np.sort(h,axis=1)[:,:,None]==np.sort(a,axis=1)[:,None,:])
    h=incidence(h,players);a=incidence(a,players)
    w=q.seconds.to_numpy(float)/3600
    hw=h.multiply(w[:,None]);aw=a.multiply(w[:,None])
    same=(h.T@hw+a.T@aw).toarray()
    cross=(h.T@aw+a.T@hw).toarray()
    assert np.allclose(np.diag(cross),0,rtol=0,atol=1e-12)
    assert np.allclose(same.sum(axis=1),5*np.diag(same),rtol=0,atol=1e-7)
    assert np.allclose(cross.sum(axis=1),5*np.diag(same),rtol=0,atol=1e-7)
    return same,cross

def system_names(teams,team_seasons):
    context=['intercept','is_home','score_-2','score_-1','score_1','score_2','period_2','period_3','context_missing']
    return context+[f'{side}_team:{t}' for side in ['own','opp'] for t in teams]+[f'{side}_teamseason:{t}:{s}' for side in ['own','opp'] for t,s in team_seasons]

def design(q,players,names):
    """Two team perspectives. Positive suppression reduces opposing shot rates."""
    n=len(q);k=len(players)
    h=np.stack(q.home_players.to_numpy()).astype(np.int64)
    a=np.stack(q.away_players.to_numpy()).astype(np.int64)
    own=np.stack([h,a],axis=1).reshape(-1,5)
    opp=np.stack([a,h],axis=1).reshape(-1,5)
    ix=pd.Index(players)
    oi=ix.get_indexer(own.ravel()).reshape(-1,5)
    di=ix.get_indexer(opp.ravel()).reshape(-1,5)
    assert (oi>=0).all() and (di>=0).all()
    rr=np.repeat(np.arange(2*n),10)
    cc=np.column_stack([oi,di+k]).ravel()
    vv=np.tile(np.r_[np.ones(5),-np.ones(5)],2*n)
    xp=sparse.coo_matrix((vv,(rr,cc)),shape=(2*n,2*k)).tocsr()
    lookup={v:i for i,v in enumerate(names)}
    rows=[];cols=[];values=[]
    def add(row,col,value):
        if col in lookup and value!=0:
            rows.append(row);cols.append(lookup[col]);values.append(value)
    # Context construction is small per chunk; vectorized player incidence dominates.
    records=q[['home_team','away_team','season','score_state','period','context_missing']].itertuples(index=False,name=None)
    for j,(home,away,season,score,period,missing) in enumerate(records):
        for side,own_team,opp_team in [(0,home,away),(1,away,home)]:
            row=2*j+side
            add(row,'intercept',1.)
            add(row,'is_home',.5 if side==0 else -.5)
            if missing:add(row,'context_missing',1.)
            else:
                sc=int(score)*(1 if side==0 else -1)
                add(row,f'score_{sc}',1.)
            add(row,f'period_{int(period)}',1.)
            for direction,t in [('own',own_team),('opp',opp_team)]:
                add(row,f'{direction}_team:{t}',1.)
                add(row,f'{direction}_teamseason:{t}:{int(season)}',1.)
    xn=sparse.coo_matrix((values,(rows,cols)),shape=(2*n,len(names))).tocsr()
    counts=q[['home_shots','away_shots']].to_numpy(float).ravel()
    hours=np.repeat(q.seconds.to_numpy(float)/3600,2)
    return xp,xn,counts,hours

def prepare_ledger(year):
    q=pd.read_parquet(LEDGER,filters=[('season','=',year)]).copy()
    ctx=pd.read_parquet(CONTEXT,filters=[('season','=',year)],columns=['game','season','t0','t1'])
    assert not ctx.duplicated(['game','season','t0','t1']).any()
    q=q.merge(ctx.assign(context_present=True),on=['game','season','t0','t1'],how='left',validate='one_to_one')
    q['context_missing']=q.context_present.ne(True)
    q['home_team']=q.home_team.map(team);q['away_team']=q.away_team.map(team)
    return q

def normal_stats(year,players,names):
    q=prepare_ledger(year);k=len(players);m=len(names)
    assert q.season.eq(year).all()
    g=np.load(OUT/f'geometry_{year}.npz')
    s=g['same'];c=g['cross']
    normal=np.zeros((2*k+m,2*k+m))
    normal[:k,:k]=s;normal[k:2*k,k:2*k]=s
    normal[:k,k:2*k]=-c;normal[k:2*k,:k]=-c
    rhs=np.zeros(2*k+m)
    for start in range(0,len(q),30000):
        xp,xn,y,h=design(q.iloc[start:start+30000],players,names)
        pn=(xp.T@xn.multiply(h[:,None])).toarray()
        normal[:2*k,2*k:]+=pn;normal[2*k:,:2*k]+=pn.T
        normal[2*k:,2*k:]+=(xn.T@xn.multiply(h[:,None])).toarray()
        rhs[:2*k]+=np.asarray(xp.T@y).ravel();rhs[2*k:]+=np.asarray(xn.T@y).ravel()
    assert np.max(np.abs(normal-normal.T))<1e-8
    return normal,rhs,{'year':year,'intervals':len(q),'games':int(q.game.nunique()),'shots':int(q.home_shots.sum()+q.away_shots.sum()),'missing_context_intervals':int(q.context_missing.sum())}
