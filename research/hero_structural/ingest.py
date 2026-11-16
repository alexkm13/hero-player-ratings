"""Restartable, read-only source ingestion; past experiments are not rewritten."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import json
import re
import zipfile

import numpy as np
import pandas as pd
import openpyxl

from .common import DATA, OUT, PCV, PRIOR, SEASONS, dump, sha, now, norm_name, team, season_id

def supplemental_file(path):
    """Raw defender/recovery annotations omitted by the old actor ledger."""
    try:
        w=openpyxl.load_workbook(path,read_only=True,data_only=True)
        names={str(n).strip().lower().replace(' ','_'):n for n in w.sheetnames}
        trname=next((names[n] for n in ['tracking','raw','sheet1','shots'] if n in names),None)
        if trname is None:
            w.close();return path,[],{'status':'missing_tracking'}
        tr=w[trname]; rows=tr.iter_rows(values_only=True); head=next(rows)
        def norm(v):return re.sub(r'[^a-z0-9]+','_',str(v).strip().lower()).strip('_')
        headers=[norm(v) for v in head]
        idx={name:headers.index(name) for name in ['defended_by','pressure','dump_recovered'] if name in headers}
        out=[]
        for rownum,row in enumerate(rows,2):
            values={k:row[i] if i<len(row) else None for k,i in idx.items()}
            if any(v is not None and str(v).strip() for v in values.values()):
                out.append({'source_file':path,'source_row':rownum,**{k:'' if v is None else str(v).strip() for k,v in values.items()}})
        w.close();return path,out,{'status':'ok','rows':len(out),'fields':list(idx)}
    except Exception as e:return path,[],{'status':'failed','error':repr(e)}

def identity_map():
    d=pd.read_parquet(PRIOR/'rtv_full_trait_repair_v2/traits/identity_join_audit.parquet')
    d=d[d.player_id.notna()].copy();d['name_key']=d.player.map(norm_name)
    d['team_key']=d.team.map(team)
    d.player_id=d.player_id.astype(int)
    keys=['season_id','name_key','team_key']
    uniq=d.groupby(keys).player_id.nunique()
    bad=set(uniq[uniq>1].index)
    records={}
    for r in d.itertuples():
        key=(int(r.season_id),r.name_key,r.team_key)
        if key not in bad:records[key]=int(r.player_id)
    # Unique source/display spelling fallback without fuzzy matching.
    full=pd.read_parquet(PCV/'data/pcv/player_season_feature_table_all_in_one.parquet')
    f=full[['player_id','player_name','season_id','position_group']].copy()
    fallback={}
    for r in f.itertuples():fallback.setdefault((int(r.season_id),norm_name(r.player_name)),set()).add(int(r.player_id))
    for r in d.itertuples():fallback.setdefault((int(r.season_id),r.name_key),set()).add(int(r.player_id))
    for col in ['a3z_player','eh_rapm_Player','eh_std_Player']:
        if col not in full:continue
        for pid,seas,name in zip(full.player_id,full.season_id,full[col]):
            if pd.notna(name):fallback.setdefault((int(seas),norm_name(name)),set()).add(int(pid))
    fallback={k:next(iter(v)) for k,v in fallback.items() if len(v)==1}
    return records,fallback,f

def micro_stage(supplement=True,workers=6):
    destination=DATA/'micro_actions.parquet'
    if destination.exists():print('reuse canonical micro actions',flush=True);return
    d=pd.read_parquet(PRIOR/'poev_touch_metric/a3z_touch_ledger.parquet')
    d=d[d.strength.str.upper().eq('5V5')].copy()
    d['season_id']=d.season.map(season_id)
    d['game']=d.season.str[:4].astype(int)*1_000_000+pd.to_numeric(d.game_id,errors='coerce')
    d=d[d.game.notna()].copy();d.game=d.game.astype(np.int64)
    # Game codes are already 20001, hence start_year*1e6 + code.
    srcmap,fallback,meta=identity_map()
    def identify(season,name,tm):
        key=norm_name(name)
        return srcmap.get((season,key,team(tm)),fallback.get((season,key),-1))
    d['player_id']=[identify(int(s),n,t) for s,n,t in zip(d.season_id,d.player_name,d.actor_team)]
    d['actor_team']=d.actor_team.map(team)
    # Supplemental source extraction is cached before lineage/joins.
    sp=DATA/'supplemental.parquet'
    if supplement and not sp.exists():
        files=sorted(set(d.source_file));parts=[];audits=[]
        print('extract supplemental raw Tracking annotations',len(files),flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,(p,rows,audit) in enumerate(pool.map(supplemental_file,files,chunksize=8),1):
                parts.extend(rows);audits.append({'path':p,'sha256':sha(p),**audit})
                if i%100==0:print('supplement files',i,'/',len(files),flush=True)
        pd.DataFrame(parts).to_parquet(sp,index=False)
        dump(DATA/'supplement_manifest.json',{'time':now(),'files':audits})
    if sp.exists():
        s=pd.read_parquet(sp)
        if s.duplicated(['source_file','source_row']).any():raise ValueError('supplement duplicate row')
        d=d.merge(s,on=['source_file','source_row'],how='left',validate='many_to_one')
    for col in ['defended_by','pressure','dump_recovered']:
        if col not in d:d[col]=''
        d[col]=d[col].fillna('')
    # Jersey/source-roster mapping derived only from observed source actors;
    # no invented player IDs for missing raw roster annotations.
    roster=d[d.player_id>0].groupby(['source_file','actor_team','jersey']).player_id.agg(lambda v: sorted(set(v)))
    roster={k:v[0] for k,v in roster.items() if len(v)==1}
    gm=pd.read_csv(PRIOR/'rtv_observed_transfer_v1_run03/game_metadata.csv')
    gm.home_team=gm.home_team.map(team);gm.away_team=gm.away_team.map(team)
    d=d.merge(gm[['game','home_team','away_team','date']],on='game',how='left',validate='many_to_one')
    d['side']=np.where(d.actor_team.eq(d.home_team),0,np.where(d.actor_team.eq(d.away_team),1,-1))
    def token(value,default):
        text=str(value).upper().replace(' ','')
        m=re.fullmatch(r'(\d+)(?:\.0)?([A-Z.]{2,4})?',text)
        if not m:return -1
        return roster.get((current_file,team(m[2] or default),m[1]),-1)
    defender=[];recoverer=[]
    for r in d.itertuples():
        current_file=r.source_file
        opposition=r.away_team if r.side==0 else r.home_team
        defender.append(token(r.defended_by,opposition))
        recoverer.append(token(r.dump_recovered,r.actor_team))
    d['defender_id']=defender;d['recoverer_id']=recoverer
    d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
    if d.duplicated(['source_file','source_row','action_type','jersey','actor_team']).any():raise ValueError('actor key collision')
    d.to_parquet(destination,index=False)
    dump(DATA/'micro_audit.json',{'actor_rows':len(d),'unmapped_player_rows':int((d.player_id<0).sum()),'bad_side':int((d.side<0).sum()),'defender_named':int((d.defender_id>0).sum()),'recoverer_named':int((d.recoverer_id>0).sum()),'source_files':int(d.source_file.nunique()),'source_row_groups':len(d.drop_duplicates(['source_file','source_row'])),'seasons':d.groupby('season_id').size().to_dict()})

def base_stage():
    if (DATA/'risk.parquet').exists():print('reuse canonical base',flush=True);return
    raw_path=PRIOR/'rtv_short_interval_fullpanel_v3/raw_intervals.parquet'
    shot_path=PRIOR/'rtv_usable_response_v5/assignments.parquet'
    events_path=PRIOR/'rtv_event_information_v1/events.parquet'
    r=pd.read_parquet(raw_path)
    r=r[(r.seconds>0)&(r.t1<=3600)&r.season.isin(SEASONS)].copy()
    if r.duplicated(['game','raw_interval_id']).any():raise ValueError('interval key collision')
    if not np.allclose(r.seconds,r.t1-r.t0):raise ValueError('interval duration discrepancy')
    gm=pd.read_csv(PRIOR/'rtv_observed_transfer_v1_run03/game_metadata.csv');gm['date']=pd.to_datetime(gm.date).dt.strftime('%Y-%m-%d')
    gm.home_team=gm.home_team.map(team);gm.away_team=gm.away_team.map(team)
    r=r.merge(gm,on='game',how='left',validate='many_to_one')
    if r.date.isna().any():raise ValueError('missing game date')
    r=r.sort_values(['date','game','t0']).reset_index(drop=True);r['interval_index']=np.arange(len(r))
    shots=pd.read_parquet(shot_path)
    ev=pd.read_parquet(events_path,columns=['game','shot_id','shotID'])
    shots=shots.merge(ev,on=['game','shot_id'],validate='one_to_one')
    goalie_parts=[]
    for p in sorted((PRIOR/'hero_portable_talent_program/raw/moneypuck_shots').glob('shots_*.zip')):
        start=int(p.stem[-4:])
        if start not in range(2021,2026):continue
        with zipfile.ZipFile(p) as z:
            member=next(n for n in z.namelist() if n.endswith('.csv'))
            with z.open(member) as h:
                d=pd.read_csv(h,usecols=['game_id','shotID','goalieIdForShot'],low_memory=False)
        d['game']=start*1_000_000+d.game_id.astype(np.int64);goalie_parts.append(d[['game','shotID','goalieIdForShot']])
    goalies=pd.concat(goalie_parts,ignore_index=True)
    if goalies.duplicated(['game','shotID']).any():raise ValueError('goalie source collision')
    shots=shots.merge(goalies,on=['game','shotID'],how='left',validate='one_to_one')
    shots['goalie_id']=pd.to_numeric(shots.goalieIdForShot,errors='coerce').fillna(-1).astype(int)
    i=pd.MultiIndex.from_frame(r[['game','raw_interval_id']])
    shots['interval_index']=i.get_indexer(pd.MultiIndex.from_arrays([shots.game,shots.primary_interval_id]))
    rejected=shots[shots.interval_index<0].copy();shots=shots[shots.interval_index>=0].copy()
    ix=shots.interval_index.to_numpy(int);own=np.stack(r.home_players)
    away=np.stack(r.away_players);own=np.where(shots.side.to_numpy()[:,None]==0,own[ix],away[ix])
    contained=(own==shots.shooter_id.to_numpy()[:,None]).any(axis=1)
    rejected=pd.concat([rejected,shots[~contained]],ignore_index=True);shots=shots[contained].copy()
    shots['date']=r.date.to_numpy()[shots.interval_index];shots['dangerous']=shots.xg.ge(.10).astype(int)
    if shots.duplicated(['game','shot_id']).any():raise ValueError('shot duplication')
    r.to_parquet(DATA/'risk.parquet',index=False)
    shots.to_parquet(DATA/'shots.parquet',index=False);rejected.to_parquet(DATA/'shot_exclusions.parquet',index=False)
    # Static identity/position inventory, not an input talent prior.
    _,_,f=identity_map()
    f.to_parquet(DATA/'player_meta.parquet',index=False)
    edge=pd.read_parquet(PCV/'data/pcv/nhl_edge_skater_stats_detail.parquet')
    edge=edge[edge.game_type.eq(2)].copy()
    if edge.duplicated(['player_id','season_id']).any():raise ValueError('EDGE ambiguous player season')
    edge['measurement_source_season']=edge.season_id
    edge['season_id']=edge.season_id+10001
    edge['bursts_per_game']=edge.edge_bursts_over20_count/edge.games_played.replace(0,np.nan)
    edge[['player_id','season_id','measurement_source_season','edge_speed_max_mph','bursts_per_game']].to_parquet(DATA/'skating.parquet',index=False)
    dump(DATA/'base_audit.json',{'risk_intervals':len(r),'risk_games':int(r.game.nunique()),'risk_seconds':float(r.seconds.sum()),'retained_shots':len(shots),'excluded_shots':len(rejected),'missing_goalie':int((shots.goalie_id<0).sum()),'goals':int(shots.goal.sum()),'xg':float(shots.xg.sum()),'seasons':r.groupby('season').agg(intervals=('game','size'),games=('game','nunique'),seconds=('seconds','sum')).reset_index().to_dict('records'),'source_hashes':{str(p):sha(p) for p in [raw_path,shot_path,events_path]}})
    print('base canonical complete',len(r),'risk intervals',len(shots),'shots',flush=True)

def link_stage():
    if (DATA/'linked_micro.parquet').exists():print('reuse linked micro',flush=True);return
    d=pd.read_parquet(DATA/'micro_actions.parquet');r=pd.read_parquet(DATA/'risk.parquet')
    parts=[];ex=[]
    for game,actions in d.groupby('game',sort=True):
        g=r[r.game.eq(game)]
        if g.empty:
            actions=actions.copy();actions['exclusion']='missing_risk_game';ex.append(actions);continue
        starts=g.t0.to_numpy();ends=g.t1.to_numpy();ix=g.interval_index.to_numpy()
        hp=np.stack(g.home_players);ap=np.stack(g.away_players)
        pos=np.searchsorted(starts,actions.game_sec.to_numpy(),side='right')-1
        linked=np.full(len(actions),-1,dtype=int)
        # Primary is latest-start interval; exact boundary fallback is permitted
        # only when actor membership uniquely selects the earlier interval.
        for delta in [0,-1,1]:
            candidate=pos+delta;safe=np.clip(candidate,0,len(g)-1)
            t=actions.game_sec.to_numpy();side=actions.side.to_numpy();pid=actions.player_id.to_numpy()
            players=np.where(side[:,None]==0,hp[safe],ap[safe])
            ok=(linked<0)&(candidate>=0)&(candidate<len(g))&(t>=starts[safe])&(t<=ends[safe])&(side>=0)&(pid>0)&(players==pid[:,None]).any(axis=1)
            linked[ok]=ix[safe[ok]]
        actions=actions.copy();actions['interval_index']=linked
        bad=actions[linked<0].copy();bad['exclusion']='unmapped_or_outside_supported_lineup';ex.append(bad)
        parts.append(actions[linked>=0])
    q=pd.concat(parts,ignore_index=True)
    q.to_parquet(DATA/'linked_micro.parquet',index=False)
    rejected=pd.concat(ex,ignore_index=True);rejected.to_parquet(DATA/'micro_exclusions.parquet',index=False)
    # Root taxonomy creates one incidence event for each non-shot source row.
    priority={'shot':0,'entry':1,'retrieval':2,'exit':3,'pass_a1':4,'pass_a2':5,'pass_a3':6}
    d['root_priority']=d.action_type.map(priority)
    root=d.sort_values('root_priority').drop_duplicates(['source_file','source_row'])
    root=root[root.action_type.isin(['entry','retrieval','exit']) & root.side.ge(0)].copy()
    root=root.merge(q[['source_file','source_row','action_type','interval_index']],on=['source_file','source_row','action_type'],how='left',validate='one_to_one')
    # An unmapped actor still supplies a timed team event when its team and risk
    # interval are observed. It contributes incidence, but no actor likelihood.
    fallback=[]
    for game,events in root.groupby('game',sort=True):
        risk=r[r.game.eq(game)];events=events.copy()
        if risk.empty:continue
        t=events.game_sec.to_numpy();starts=risk.t0.to_numpy();ends=risk.t1.to_numpy()
        pos=np.searchsorted(starts,t,side='right')-1;safe=np.clip(pos,0,len(risk)-1)
        valid=(pos>=0)&(t>=starts[safe])&(t<=ends[safe])
        ix=np.where(valid,risk.interval_index.to_numpy()[safe],-1)
        events.interval_index=events.interval_index.fillna(pd.Series(ix,index=events.index)).astype(int)
        fallback.append(events[events.interval_index.ge(0)])
    root=pd.concat(fallback,ignore_index=True)
    root['family']=root.action_type.map({'entry':1,'retrieval':2,'exit':3})
    root.to_parquet(DATA/'roots.parquet',index=False)
    dump(DATA/'link_audit.json',{'linked_actor_rows':len(q),'excluded_actor_rows':len(rejected),'linked_share':len(q)/(len(q)+len(rejected)),'roots':root.groupby('action_type').size().to_dict(),'tracked_games':int(d.game.nunique()),'linked_games':int(q.game.nunique()),'boundary_rule':'latest start first; actor-contained adjacent exact endpoint fallback','root_rule':'shot > entry > retrieval > exit; one root incidence per non-shot source row; shot source rows have no extra incidence likelihood'})
    print('linked micro',len(q),'excluded',len(rejected),'roots',len(root),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['base','micro','link','all']);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    DATA.mkdir(parents=True,exist_ok=True)
    if a.stage in ['base','all']:base_stage()
    if a.stage in ['micro','all']:micro_stage(workers=a.workers)
    if a.stage in ['link','all']:link_stage()

if __name__=='__main__':main()
