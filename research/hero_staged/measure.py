"""Source-owned numerators and denominators; no future skill/value feedback."""
import numpy as np
import pandas as pd
from .common import DATA,OUT,CACHE,PCV,SEASONS,dump,sha,now,log

ACTIVITY=['act_shots','act_entries','act_retrievals','act_primary_passes','act_recoveries']
EXECUTION=['exec_danger','exec_xg_per_shot','exec_entry','exec_retrieval','exec_denial','exec_recovery','exec_finishing','exec_speed']
CHANNELS=ACTIVITY+EXECUTION
ANCHORS=['act_shots','exec_entry','act_primary_passes','exec_finishing','exec_recovery','source_ca_suppression','exec_denial','exec_speed']

def prepare():
    path=CACHE/'measurements_verified.parquet'
    if path.exists():return pd.read_parquet(path)
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    r=pd.read_parquet(DATA/'risk.parquet');s=pd.read_parquet(DATA/'shots.parquet')
    roots=pd.read_parquet(DATA/'roots.parquet');m=pd.read_parquet(DATA/'linked_micro.parquet')
    rr=r.set_index('interval_index').loc[roots.interval_index]
    home=np.stack(rr.home_players);away=np.stack(rr.away_players)
    own=np.where(roots.side.to_numpy()[:,None]==0,home,away)
    opp=np.where(roots.side.to_numpy()[:,None]==0,away,home)
    invalid={}
    for col,lineup in [('player_id',own),('defender_id',opp),('recoverer_id',own)]:
        bad=roots[col].gt(0).to_numpy()&~(lineup==roots[col].to_numpy()[:,None]).any(axis=1)
        invalid[col]=int(bad.sum());roots.loc[bad,col]=-1
    del rr,home,away,own,opp
    meta=pd.read_parquet(DATA/'player_meta.parquet')
    rows=[];audits=[]
    for season in SEASONS:
        log(f'measure source {season}')
        risk=r[r.season.eq(season)];shots=s[s.season.eq(season)];root=roots[roots.season_id.eq(season)];micro=m[m.season_id.eq(season)]
        ids=np.column_stack([np.stack(risk.home_players),np.stack(risk.away_players)])
        players,inverse=np.unique(ids,return_inverse=True);inverse=inverse.reshape(ids.shape);n=len(players)
        loc=pd.Index(players);clock=risk.seconds.to_numpy(float)
        minutes=np.bincount(inverse.ravel(),weights=np.repeat(clock/60,10),minlength=n)
        tracked=risk.game.isin(micro.game.unique()).to_numpy()
        tracked_min=np.bincount(inverse[tracked].ravel(),weights=np.repeat(clock[tracked]/60,10),minlength=n)
        rec=pd.DataFrame({'player_id':players,'season_id':season,'strict_minutes':minutes,'tracked_minutes':tracked_min})
        def counts(e,column='player_id',weight=None):
            ix=loc.get_indexer(e[column]);valid=ix>=0
            return np.bincount(ix[valid],weights=None if weight is None else np.asarray(weight)[valid],minlength=n)
        def channel(name,num,den):
            rec[name+'_num']=num;rec[name+'_den']=den
            rec[name]=np.divide(num,den,out=np.full(n,np.nan),where=np.asarray(den)>0)
        shot_n=counts(shots,'shooter_id');xg=counts(shots,'shooter_id',shots.xg);goal=counts(shots,'shooter_id',shots.goal)
        channel('act_shots',shot_n,minutes/60)
        channel('exec_danger',counts(shots,'shooter_id',shots.dangerous),shot_n)
        channel('exec_xg_per_shot',xg,shot_n)
        rec['source_shot_xg']=xg;rec['source_shot_goals']=goal
        # Approximate log conversion residual with a half-event stabilizer.
        channel('exec_finishing',np.log((goal+.5)/(xg+.5))*shot_n,shot_n)
        for family,name in [(1,'entries'),(2,'retrievals')]:
            e=root[root.family.eq(family)]
            channel('act_'+name,counts(e),tracked_min/60)
        entry=root[root.family.eq(1)&root.entry_type.str.upper().isin(['C','D','F'])]
        channel('exec_entry',counts(entry,weight=entry.entry_type.str.upper().ne('F')),counts(entry))
        retr=root[root.family.eq(2)&root.entry_result.str.upper().isin(['EXC','BOT','CEX','PEX','CLE','MEX','FEX'])]
        channel('exec_retrieval',counts(retr,weight=retr.entry_result.str.upper().ne('BOT')),counts(retr))
        denied=entry[entry.defender_id.gt(0)]
        channel('exec_denial',counts(denied,'defender_id',denied.entry_type.str.upper().eq('F')),counts(denied,'defender_id'))
        # Passing is observed involvement, not observed pass-completion skill.
        passes=micro[micro.action_type.eq('pass_a1')]
        channel('act_primary_passes',counts(passes),tracked_min/60)
        dumps=entry[entry.entry_type.str.upper().eq('D')]
        known=dumps[dumps.recoverer_id.gt(0)|dumps.dump_recovered.str.upper().isin(['N','NN'])]
        channel('act_recoveries',counts(known[known.recoverer_id.gt(0)],'recoverer_id'),tracked_min/60)
        # On-ice named recovery / recorded known dump trials: involvement
        # conditional on team opportunity, not an identified personal success.
        risk_index=risk.set_index('interval_index')
        if len(known):
            kr=risk_index.loc[known.interval_index]
            own=np.where(known.side.to_numpy()[:,None]==0,np.stack(kr.home_players),np.stack(kr.away_players))
            pos=loc.get_indexer(own.ravel());den=np.bincount(pos[pos>=0],minlength=n)
        else:den=np.zeros(n)
        channel('exec_recovery',counts(known[known.recoverer_id.gt(0)],'recoverer_id'),den)
        edge=pd.read_parquet(DATA/'skating.parquet');edge=edge[edge.measurement_source_season.eq(season)]
        speed=edge.set_index('player_id').edge_speed_max_mph.reindex(players).to_numpy(float)
        channel('exec_speed',np.where(np.isfinite(speed),speed,0),np.isfinite(speed).astype(float))
        roles=meta[meta.season_id.eq(season)].set_index('player_id')
        rec['position_group']=roles.position_group.reindex(players).fillna('U').to_numpy()
        rec['player_name']=roles.player_name.reindex(players).fillna('').to_numpy()
        rows.append(rec)
        assert np.isclose(minutes.sum()*60,10*clock.sum())
        assert int(shot_n.sum())==len(shots)
        audits.append({'source_season':season,'players':n,'shots':len(shots),'tracked_games':int(micro.game.nunique()),
                       'risk_games':int(risk.game.nunique()),'coverage_minutes':float(tracked_min.sum()/minutes.sum()),
                       'known_dump_trials':len(known),'named_defender_trials':len(denied)})
    panel=pd.concat(rows,ignore_index=True)
    base=pd.read_parquet(PCV/'data/pcv/player_season_feature_table_all_in_one.parquet',columns=['player_id','season_id','eh_rapm_CA/60'])
    panel=panel.merge(base,on=['player_id','season_id'],how='left',validate='one_to_one')
    panel['source_ca_suppression']=-pd.to_numeric(panel.pop('eh_rapm_CA/60'),errors='coerce')
    panel.to_parquet(path,index=False);panel.to_parquet(OUT/'source_measurements.parquet',index=False)
    dump(OUT/'measurement_audit.json',{'created':now(),'seasons':audits,'channels':CHANNELS,'category_anchors':ANCHORS,
         'nonmember_actor_annotations_excluded':invalid,'earlier_preflight_cache':'measurements.parquet retained; verified data used for every fit',
         'recovery_interpretation':'named on-ice recovery involvement given observed team opportunity; incomplete naming',
         'pass_interpretation':'recorded named primary shot-assister activity, not passing execution',
         'sources':{p.name:sha(p) for p in [DATA/'risk.parquet',DATA/'shots.parquet',DATA/'roots.parquet',DATA/'linked_micro.parquet',DATA/'skating.parquet']}})
    return panel

def transform_fit(frame,channels):
    """Shrink to role priors estimated only from the model's training sources."""
    params={}
    for role in ['F','D','U']:
        g=frame[frame.position_group.eq(role)]
        if g.empty:g=frame
        params[role]={}
        for c in channels:
            value=pd.to_numeric(g[c],errors='coerce').to_numpy(float)
            den=pd.to_numeric(g[c+'_den'],errors='coerce').fillna(0).to_numpy(float)
            valid=np.isfinite(value)&(den>0)
            prior=float(np.average(value[valid],weights=np.minimum(den[valid],np.percentile(den[valid],90)))) if valid.any() else 0.
            strength=200/60 if c.startswith('act_') else (100 if c=='exec_finishing' else (0 if c=='exec_speed' else 50))
            params[role][c]={'prior':prior,'strength':strength}
    return params

def transform(frame,channels,params):
    cols=[]
    for c in channels:
        v=pd.to_numeric(frame[c],errors='coerce').to_numpy(float)
        den=pd.to_numeric(frame[c+'_den'],errors='coerce').fillna(0).to_numpy(float)
        pri=np.array([params.get(str(role),params['U'])[c]['prior'] for role in frame.position_group])
        strength=np.array([params.get(str(role),params['U'])[c]['strength'] for role in frame.position_group])
        valid=np.isfinite(v)&(den>0)
        reliability=np.divide(den,den+strength,out=np.zeros(len(v)),where=(den+strength)>0)
        reliability=np.where(valid,reliability,0)
        shrunk=pri+reliability*(np.where(valid,v,pri)-pri)
        cols.extend([shrunk,1-valid.astype(float),reliability])
    return np.column_stack(cols) if cols else np.zeros((len(frame),0))

if __name__=='__main__':prepare()
