"""Research profiles and paired observed-environment contrasts; no card writes."""
import json,shutil,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .common import OUT,DATA,TASK,PCV,SEASONS,SHORT,CATEGORIES,SEED,dump,sha,now
from .factors import Canonical,StateIndex
from .run import load_fit,active_fit
from .lineup import Response
from .model import K,D,R,SKILL_SD,ROLE_SD,Factor,design,unit,filter_update

def main():
    data=Canonical.load();test=SEASONS[-1]
    evaluation=OUT/'evaluation'/str(test)/'joint'
    ev=json.loads((evaluation/'manifest.json').read_text())
    fit,oldinv,params=load_fit(active_fit(test,'joint'))
    index=StateIndex(pd.read_parquet(evaluation/'inventory.parquet'),data.meta)
    # At the completed-season export cutoff, that season's observed position
    # is available. This never rewrites held-out pregame role contexts.
    completed_roles=data.meta[data.meta.season_id.le(test)].sort_values('season_id').groupby('player_id').tail(1).set_index('player_id').position_group.to_dict()
    for i,row in index.inventory.iterrows():
        role=completed_roles.get(int(row.player_id))
        if role in ['F','D']:index.role[i]=role
    final=dict(np.load(evaluation/'end_states.npz'));state=final['state'];cov=final['cov']
    risk=data.risk[data.risk.season.eq(test)].copy();shots=data.shots[data.shots.season.lt(test)]
    grid=np.array([np.quantile(shots.loc[shots.dangerous.eq(b),'xg'],np.linspace(.05,.95,20)) for b in [0,1]])
    goalie_ids=np.unique(data.shots.loc[data.shots.season.eq(test),'goalie_id'])
    skaterids=np.unique(np.r_[np.stack(risk.h).ravel(),np.stack(risk.a).ravel()])
    skaterindex=index.map(test,skaterids);skaterindex=skaterindex[skaterindex>=0]
    goalieindex=index.map(test,goalie_ids);goalieindex=goalieindex[goalieindex>=0]
    # Average goalie is standardized at zero, the model's nuisance anchor.
    goalie_effect=0.
    betas={k[5:]:v for k,v in params.items() if k.startswith('beta_')}
    response=Response.joint(betas,grid,goalie_effect)
    observed_asof=str(sorted(risk.date.unique())[-1]);asof='2026-09-17'
    days=max(0,(pd.Timestamp(asof)-pd.Timestamp(observed_asof)).days)
    decay=np.r_[np.full(K,.85**(days/365)),np.full(R,.50**(days/90))]
    state=state*decay;cov=cov*decay[None,:,None]*decay[None,None,:]
    q=np.zeros((D,D));q[:K,:K]=final['skill_cov']*days/365;q[K:,K:]=np.eye(R)*ROLE_SD**2*days/90;cov+=q
    # The canonical future-season label here means PRIOR 2025-26 physical
    # measurement, not any 2026-27 game outcome. Assimilate it once now.
    speed=data.skating[data.skating.measurement_source_season.eq(test)].copy()
    ids=index.map(test,speed.player_id.to_numpy());valid=(ids>=0)&speed.edge_speed_max_mph.notna().to_numpy()
    speed=speed[valid];ids=ids[valid];physical_audit=None
    if len(speed):
        transform=fit['skating_transform'];z=(speed.edge_speed_max_mph.to_numpy()-transform['mean'])/transform['sd']
        physical=Factor('skating_measurement','normal',z,np.ones((len(z),1)),[design(ids,index.n)],[unit(7)],[np.zeros(D,dtype=bool)],variance=transform['variance'])
        state,cov,physical_audit=filter_update([physical],state,cov,[betas['skating_measurement']],120)
    dump(OUT/'lineup_response.json',{'as_of':asof,'last_observed_game_date':observed_asof,'training_seasons':fit['training_seasons'],
        'offseason_forecast_days':days,'latest_physical_measurement_source_season':test,'preseason_physical_solver':physical_audit,
        'quality_grid':grid.tolist(),'goalie_effect':goalie_effect,'average_goalie_definition':'zero nuisance anchor',
        'fixed_role':True,'adaptive_admitted':False,'complementarity_admitted':False,
        'quality_integration':'training-only 20 quantiles within each dangerous-shot bin; no full shot-density model',
        'uncertainty':'within-player conditional curvature; excludes reference and shared coefficients',
        'reference_environments':'observed 2025-26 lineups grouped by fitted net rate; retrospective descriptive support',
        'references':'same-position all-skater mean skills and sampled completed-season low-usage cohort; neither certified replacement level'})
    np.savez_compressed(OUT/'current_response.npz',player_id=index.inventory.iloc[skaterindex].player_id.to_numpy(),
        state=state[skaterindex],cov=cov[skaterindex],position=index.role[skaterindex].astype(str),
        **{'beta_'+name:b for name,b in betas.items()})
    names=data.meta.sort_values('season_id').groupby('player_id').tail(1).set_index('player_id').player_name.to_dict()
    records=[]
    micro=data.micro[data.micro.season_id.eq(test)]
    measured_skate=set(speed.player_id)
    usage={};shot_count=data.shots[data.shots.season.eq(test)].shooter_id.value_counts().to_dict()
    for ids,seconds in zip(np.r_[np.stack(risk.h),np.stack(risk.a)],np.r_[risk.seconds,risk.seconds]):
        for pid in ids:usage[int(pid)]=usage.get(int(pid),0)+float(seconds)
    actor_count=micro.player_id.value_counts().to_dict()
    roots=data.roots[data.roots.season_id.eq(test)]
    entry_count=roots.loc[roots.family.eq(1)&roots.entry_type.str.upper().isin(['C','D','F']),'player_id'].value_counts().to_dict()
    exit_count=micro.loc[micro.action_type.eq('exit')&micro.exit_result.str.upper().isin(['CEX','PEX','CLE','MEX','FEX']),'player_id'].value_counts().to_dict()
    pass_count=micro.loc[micro.action_type.eq('pass_a1'),'player_id'].value_counts().to_dict()
    recovery_count=roots.loc[roots.family.eq(1)&roots.recoverer_id.gt(0),'recoverer_id'].value_counts().to_dict()
    retrieval_count=roots.loc[roots.family.eq(2)&roots.entry_result.str.upper().isin(['EXC','BOT','CEX','PEX','CLE','MEX','FEX']),'player_id'].value_counts().to_dict()
    defender_count=roots.loc[roots.family.eq(1)&roots.defender_id.gt(0),'defender_id'].value_counts().to_dict()
    for i in skaterindex:
        pid=int(index.inventory.iloc[i].player_id)
        rec={'player_id':pid,'player_name':names.get(pid,str(pid)),'position':index.role[i],'as_of':asof,
             'minutes_5v5':usage.get(pid,0)/60,'shots':shot_count.get(pid,0),'linked_micro_actors':actor_count.get(pid,0),
             'prior_season_speed_measured':pid in measured_skate,'admission':'research only'}
        rec.update(creation_anchor_shots=shot_count.get(pid,0),finishing_conversion_shots=shot_count.get(pid,0),
            transition_recorded_entry_execution=entry_count.get(pid,0),transition_recorded_exit_execution=exit_count.get(pid,0),
            playmaking_named_primary_shot_assists=pass_count.get(pid,0),forechecking_named_recoveries=recovery_count.get(pid,0),
            zone_defense_named_targets_raw=defender_count.get(pid,0),zone_defense_recorded_retrieval_marks=retrieval_count.get(pid,0),
            suppression_evidence_scope='shared opponent shot clock/quality; no independent player execution mark')
        for j,name in enumerate(SHORT):
            rec[name+'_skill']=float(state[i,j]);rec[name+'_conditional_sd']=float(np.sqrt(cov[i,j,j]))
        records.append(rec)
    profiles=pd.DataFrame(records)
    for name in SHORT:profiles[name+'_display_percentile']=profiles.groupby('position')[name+'_skill'].rank(pct=True)*100
    profiles['skating_display_scope']=np.where(profiles.prior_season_speed_measured,'prior-season all-state max speed + hockey likelihood','latent only; no direct physical measure')
    profiles.to_csv(OUT/'player_skill_profiles.csv',index=False)
    # Preserve existing additional display categories as labeled companions;
    # they never enter inference or this 5v5 research scalar.
    source=PCV/'data/pcv/player_season_all_in_one_8cat_scores.parquet'
    if source.exists():
        import pyarrow.parquet as pq
        schema=pq.read_schema(source).names
        keys=[x for x in ['player_id','season_id','player_name','position_group'] if x in schema]
        companions=[x for x in schema if any(x.startswith('cat_'+c+'_') for c in ['physicality','power_play','penalty_kill']) and '__' not in x]
        display=pd.read_parquet(source,columns=keys+companions)
        if 'season_id' in display:display=display[display.season_id.eq(test)]
        display['scope']='legacy display companions; not reestimated; excluded from structural inference and 5v5 scalar'
        display.to_csv(OUT/'display_companions.csv',index=False)
    # Observed five-player sides are compatible by construction. Keep intact
    # teammate/opponent vectors, replacing only one same-position slot.
    rng=np.random.default_rng(SEED)
    sampled=risk.iloc[rng.choice(len(risk),min(4000,len(risk)),replace=False)]
    hi=index.map(test,np.stack(sampled.h));ai=index.map(test,np.stack(sampled.a))
    allown=np.r_[hi,ai];allopp=np.r_[ai,hi]
    base=response.paired(state[allown],state[allopp],index.role[allown]=='D',index.role[allopp]=='D')
    cuts=np.quantile(base['net60'],[1/3,2/3]);panel=np.digitize(base['net60'],cuts)
    panels={};references=[];value_rows=[]
    for role in ['F','D']:
        eligible=skaterindex[index.role[skaterindex]==role]
        if not len(eligible):continue
        mean=state[eligible,:K].mean(axis=0)
        qualified=[i for i in eligible if usage.get(int(index.inventory.iloc[i].player_id),0)>=200*60]
        if not qualified:qualified=list(eligible)
        cutoff=np.quantile([usage.get(int(index.inventory.iloc[i].player_id),0) for i in qualified],.25)
        low=np.array([i for i in qualified if usage.get(int(index.inventory.iloc[i].player_id),0)<=cutoff])
        for i in low:references.append({'position':role,'player_id':int(index.inventory.iloc[i].player_id),
                                      'minutes_5v5':usage.get(int(index.inventory.iloc[i].player_id),0)/60})
        for number,label in enumerate(['weak','middle','strong']):
            possible=np.flatnonzero((panel==number)&(index.role[allown]==role).any(axis=1))
            selected=rng.choice(possible,min(32,len(possible)),replace=False)
            own=allown[selected];opp=allopp[selected]
            slots=np.array([rng.choice(np.flatnonzero(index.role[x]==role)) for x in own])
            lowrefs=rng.choice(low,len(selected),replace=True)
            panels[(role,label)]=(own,opp,slots,lowrefs,mean)
    support_rows=[]
    for (role,label),(oi,pi,slots,low,mean) in panels.items():
        for k in range(len(oi)):support_rows.append({'position':role,'panel':label,'own_players':index.inventory.iloc[oi[k]].player_id.to_list(),
            'opponent_players':index.inventory.iloc[pi[k]].player_id.to_list(),'slot':int(slots[k]),'low_usage_reference':int(index.inventory.iloc[low[k]].player_id)})
    dump(OUT/'reference_environments.json',support_rows)
    pd.DataFrame(references).to_csv(OUT/'low_usage_reference_pool.csv',index=False)
    for i in skaterindex:
        role=index.role[i]
        if role not in ['F','D']:continue
        pid=int(index.inventory.iloc[i].player_id)
        for label in ['weak','middle','strong']:
            oi,pi,slots,low,mean=panels[(role,label)]
            # Avoid duplicate players in either counterfactual side.
            keep=~((oi==i).any(axis=1)|(pi==i).any(axis=1));oi=oi[keep];pi=pi[keep];slots=slots[keep];low=low[keep]
            if not len(oi):continue
            own=state[oi].copy();opp=state[pi].copy();row=np.arange(len(oi));ro=index.role[oi]=='D';rp=index.role[pi]=='D'
            avg=own.copy();avg[row,slots,:K]=mean
            lowstate=own.copy();lowstate[row,slots,:K]=state[low,:K]
            new=own.copy();new[row,slots,:K]=state[i,:K]
            average=response.paired(avg,opp,ro,rp);lowvalue=response.paired(lowstate,opp,ro,rp);actual=response.paired(new,opp,ro,rp)
            contrast=float(np.mean(actual['net60']-average['net60']))
            gradient=[]
            for j in range(K):
                minus=new.copy();plus=new.copy();minus[row,slots,j]-=1e-4;plus[row,slots,j]+=1e-4
                gradient.append(float(np.mean(response.paired(plus,opp,ro,rp)['net60']-response.paired(minus,opp,ro,rp)['net60'])/2e-4))
            gradient=np.asarray(gradient);conditional_sd=float(np.sqrt(np.einsum('i,ij,j->',gradient,cov[i,:K,:K],gradient,optimize=False)))
            skill_distance=float(np.linalg.norm((state[i,:K]-mean)/SKILL_SD))
            value_rows.append({'player_id':pid,'player_name':names.get(pid,str(pid)),'position':role,'environment':label,'contexts':len(oi),
                'above_average_net_goals60':contrast,'above_average_offense_goals60':float(np.mean(actual['gf60']-average['gf60'])),
                'above_average_defense_goals60':float(np.mean(average['ga60']-actual['ga60'])),
                'above_low_usage_net_goals60':float(np.mean(actual['net60']-lowvalue['net60'])),
                'conditional_player_only_sd_goals60':conditional_sd,'skill_distance_prior_units':skill_distance,
                'support':'observed environment; individual substitution extrapolation' if skill_distance>4 else 'observed environment; fixed-role substitution',
                'replacement_level_certified':False,'admission':'research only'})
    values=pd.DataFrame(value_rows);values.to_csv(OUT/'lineup_contrasts.csv',index=False)
    overall=values.groupby(['player_id','player_name','position']).agg(research_net_goals60=('above_average_net_goals60','mean'),
        research_offense_goals60=('above_average_offense_goals60','mean'),research_defense_goals60=('above_average_defense_goals60','mean'),
        above_low_usage_goals60=('above_low_usage_net_goals60','mean'),environment_range_goals60=('above_average_net_goals60',lambda x:float(x.max()-x.min()))).reset_index()
    overall['research_display_percentile']=overall.groupby('position').research_net_goals60.rank(pct=True)*100
    overall['admission']='research only; neither replacement-level nor production Overall'
    overall.to_csv(OUT/'research_overall.csv',index=False)
    dump(OUT/'export_manifest.json',{'created':now(),'as_of':asof,'last_observed_game_date':observed_asof,'skater_profiles':len(profiles),'contrast_rows':len(values),
       'offseason_forecast_days':days,'prior_completed_season_EDGE_only':True,'preseason_physical_solver':physical_audit,
       'eligible_overall_players':len(overall),'not_promoted':True,'quality_grid_training_seasons':fit['training_seasons'],
       'reference_panel_seed':SEED,'reference_pool_choice':'completed-2025-26 descriptive low-usage, >=200min then bottom quartile per position',
       'lineup_value_uncertainty':'conditional focal-player delta method; excludes global/reference/posterior cross-player uncertainty'})
    # Copy compact source audit evidence and portable implementation.
    audit={n:json.loads((DATA/n).read_text()) for n in ['base_audit.json','micro_audit.json','link_audit.json']}
    dump(OUT/'ingestion_audit.json',audit)
    code=OUT/'code'/'hero_structural';code.mkdir(parents=True,exist_ok=True)
    for source in (TASK/'work'/'hero_structural').glob('*.py'):shutil.copy2(source,code/source.name)
    requirements='numpy>=1.24\npandas>=2.0\nscipy>=1.10\npyarrow>=12\nopenpyxl>=3.1\n'
    (OUT/'code'/'requirements.txt').write_text(requirements)
    dump(OUT/'code_manifest.json',{p.name:sha(p) for p in code.glob('*.py')})
    for name in ['source_audit.json','implementation_contract_v1.md','IMPLEMENTATION_MEMORY.md']:
        source=TASK/'outputs'/name
        if source.exists():shutil.copy2(source,OUT/name)
    with zipfile.ZipFile(OUT/'implementation_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in (OUT/'code').rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(OUT)))
        for name in ['RESULTS.md','README.md','MODEL_SPEC.md','IMPLEMENTATION_MEMORY.md','implementation_contract_v1.md','experiment_record.md','executable_config.json','numerical_protocol.json','verification.json','ingestion_audit.json','source_audit.json','source_manifest.json','code_manifest.json','lineup_response.json','current_response.npz','export_manifest.json','evaluation_summary.json','installation.json']:
            p=OUT/name
            if p.exists():z.write(p,name)
    print(f'Exported {len(profiles)} profiles, {len(values)} contrasts; research only',flush=True)

if __name__=='__main__':main()
