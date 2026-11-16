"""Export every category and the selected future-impact research challenger."""
import json,shutil
import numpy as np
import pandas as pd
from .common import OUT,PCV,CACHE,DATA,SEASONS,CATS,ENDS,dump,sha,now,log
from .measure import prepare as measurements,CHANNELS,ANCHORS,transform_fit,transform
from .rich import prepare as full_measurements,RICH
from .player import panel,BASE,SHIFT,ENDNAMES,predict_ridge,correction_features
from .lineup import predict,swap

def main():
    m=full_measurements();current=m[m.season_id.eq(SEASONS[-1])].copy().reset_index(drop=True)
    source=panel();latest=source[source.season_id.eq(SEASONS[-1])].set_index('player_id')
    # Begin with all canonical players, including small-sample and unknown-role
    # rows. Provider absence creates a documented prior, never a fabricated score.
    frame=latest.reindex(current.player_id).reset_index()
    frame['player_name']=current.player_name.to_numpy();frame['position_group']=current.position_group.to_numpy();frame['season_id']=SEASONS[-1]
    frame['provider_source_available']=frame.eh_rapm_TOI.notna()
    for c in CHANNELS+RICH:
        for suffix in ['','_num','_den']:frame[c+suffix]=current[c+suffix].to_numpy()
    for lag in [1,2]:
        hist=m[m.season_id.eq(SEASONS[-1]-lag*10001)].set_index('player_id').reindex(frame.player_id)
        for c in CHANNELS+RICH:
            for suffix in ['', '_den']:frame[f'hist{lag}_{c}'+suffix]=hist[c+suffix].to_numpy()
    frame['strict_minutes']=current.strict_minutes.to_numpy();frame['tracked_minutes']=current.tracked_minutes.to_numpy()
    # Current complete source clock can supply reliability for a newly mapped
    # player; lagged provider measurements remain missing if unavailable.
    frame['log_toi']=frame.log_toi.fillna(np.log1p(frame.strict_minutes))
    for lag in [1,2]:
        previous=source[source.season_id.eq(SEASONS[-1]-lag*10001)].set_index('player_id').reindex(frame.player_id)
        for e in ENDNAMES:frame[f'lag{lag}_{e}']=frame[f'lag{lag}_{e}'].fillna(pd.Series(previous['current_'+e].to_numpy(),index=frame.index))
        frame[f'has_lag{lag}']=frame[f'lag{lag}_xgf'].notna().astype(float)
    sh=pd.read_parquet(OUT/'response/current/source_identity_features.parquet').set_index('player_id').reindex(frame.player_id)
    for c in SHIFT:frame[c]=sh[c].to_numpy()
    forecasts=np.full((len(frame),6),np.nan);macroforecasts=np.full((len(frame),6),np.nan);selections={}
    for role in ['F','D']:
        use=frame.position_group.eq(role);test=frame[use]
        spec=json.loads((OUT/'player'/f'model_{role}_{SEASONS[-1]}.json').read_text());family=spec['selected']['family'];selections[role]=family
        pred=predict_ridge(spec['macro'],test[BASE])
        macroforecasts[use]=pred.copy()
        if family!='m1':
            correction=spec['corrections'][family]
            if correction.get('available'):
                _,x,_=correction_features(test,test,family,correction['measurement_priors']);pred+=predict_ridge(correction,x)
        forecasts[use]=pred
    out=frame[['player_id','player_name','position_group','season_id','strict_minutes','tracked_minutes','provider_source_available']].copy()
    for j,e in enumerate(ENDNAMES):out['forecast_'+e+'_impact60']=forecasts[:,j]
    out['future_offense_goals60']=forecasts[:,2];out['future_defense_goals_prevented60']=forecasts[:,5]
    out['future_net_goal_impact60']=forecasts[:,2]+forecasts[:,5]
    out['macro_offense_goals60']=macroforecasts[:,2];out['macro_defense_goals_prevented60']=macroforecasts[:,5]
    out['macro_net_goal_impact60']=macroforecasts[:,2]+macroforecasts[:,5]
    out['measurement_correction_offense60']=forecasts[:,2]-macroforecasts[:,2]
    out['measurement_correction_defense60']=forecasts[:,5]-macroforecasts[:,5]
    out['measurement_correction_net_goals60']=out.future_net_goal_impact60-out.macro_net_goal_impact60
    out['selected_player_family']=out.position_group.map(selections).fillna('unknown position; no forecast')
    out['research_overall_pct']=out.future_net_goal_impact60.rank(method='average',pct=True)*100
    out['information_cutoff']='2026-09-17';out['model_status']='research only; development gate '+json.loads((OUT/'evaluation_summary.json').read_text())['development_gate']
    out.to_csv(OUT/'player_ratings.csv',index=False);out[out.position_group.isin(['F','D'])].sort_values('future_net_goal_impact60',ascending=False).to_csv(OUT/'research_overall.csv',index=False)
    querycols=['player_id','player_name','position_group',*BASE,*SHIFT,*[c+s for c in CHANNELS+RICH for s in ['', '_den']],
        *[f'hist{lag}_{c}'+s for lag in [1,2] for c in CHANNELS+RICH for s in ['', '_den']]]
    query=json.loads(frame[querycols].to_json(orient='records',double_precision=15))
    dump(OUT/'player_query_inputs.json',{str(r['player_id']):r for r in query})
    for role in ['F','D']:shutil.copy2(OUT/'player'/f'model_{role}_{SEASONS[-1]}.json',OUT/f'player_model_{role}.json')
    # Category anchors use training-source priors; no category display weights
    # influence the forecast. Source provider CA is separately identified.
    pri=transform_fit(m[m.season_id.lt(SEASONS[-1])],CHANNELS);values=transform(current,CHANNELS,pri)[:,::3]
    profiles=current.copy();profiles['information_cutoff']='2026-09-17'
    for cat,anchor in zip(CATS,ANCHORS):
        if anchor=='source_ca_suppression':val=current.source_ca_suppression.to_numpy()
        else:val=values[:,CHANNELS.index(anchor)]
        profiles[cat+' estimate']=val;profiles[cat+' percentile']=np.nan
        for role in ['F','D']:
            use=current.position_group.eq(role);profiles.loc[use,cat+' percentile']=pd.Series(val[use],index=profiles.index[use]).rank(method='average',pct=True)*100
        if anchor in CHANNELS:
            profiles[cat+' observed opportunities']=current[anchor+'_den'].to_numpy()
            profiles[cat+' evidence available']=current[anchor+'_den'].fillna(0).gt(0).to_numpy()
        else:profiles[cat+' evidence available']=current.source_ca_suppression.notna()
    profiles.to_csv(OUT/'player_category_profiles.csv',index=False)
    response=json.loads((OUT/'response/current/manifest.json').read_text());family=response['selected_family']
    root=OUT/'response/current';spec=json.loads((root/f'{family}.json').read_text())
    compact=frame[['player_id','player_name','position_group',*[c+s for c in CHANNELS for s in ['', '_den']]]]
    records=json.loads(compact.to_json(orient='records',double_precision=15))
    spec.update({'profiles':{str(r['player_id']):r for r in records},'as_of':'2026-09-17',
        'development_gate':json.loads((OUT/'evaluation_summary.json').read_text())['development_gate'],
        'production_admission':False,'overall_estimand':'separate future native provider GF+GA suppression, not this conditional response'})
    dump(OUT/'conditional_response.json',spec);shutil.copy2(root/f'{family}.npz',OUT/'conditional_response.npz')
    legacy=pd.read_parquet(PCV/'data/pcv/player_season_all_in_one_8cat_scores_tuned.parquet')
    wanted=[c for c in ['cat_physicality_z__adj','cat_power_play_z__adj','cat_penalty_kill_z__adj'] if c in legacy]
    companions=legacy[legacy.season_id.eq(SEASONS[-1])][['player_id','player_name',*wanted]].copy()
    companions['scope']='legacy display companions; excluded from v2 inference and Overall';companions.to_csv(OUT/'display_companions.csv',index=False)
    # Deterministic observed same-role environments; numerical contrasts are
    # conditional diagnostics rather than the published forecast score.
    risk=pd.read_parquet(DATA/'risk.parquet');risk=risk[risk.season.eq(SEASONS[-1])]
    known=set(current[current.position_group.isin(['F','D'])].player_id)
    ids=np.column_stack([np.stack(risk.home_players),np.stack(risk.away_players)])
    use=np.array([all(int(p) in known for p in row) for row in ids]);risk=risk[use]
    rng=np.random.default_rng(20260918);selected=rng.choice(len(risk),size=24,replace=False,p=risk.seconds.to_numpy()/risk.seconds.sum())
    params=dict(np.load(OUT/'conditional_response.npz',allow_pickle=False));contrasts=[];environments=[]
    lookup=current.set_index('player_id').position_group
    for env,row in enumerate(risk.iloc[selected].itertuples()):
        own=list(row.home_players);opp=list(row.away_players);environments.append({'environment':env,'game':int(row.game),'own':own,'opp':opp,'clock_start':float(row.t0)})
        for role in ['F','D']:
            slots=[int(p) for p in own if lookup.loc[p]==role]
            if not slots:continue
            remove=slots[0]
            for candidate in current[current.position_group.eq(role)].player_id:
                candidate=int(candidate)
                if candidate in own+opp:continue
                result=swap(spec,params,own,opp,remove,candidate,home=1,period=2,score=0,elapsed=3)
                contrasts.append({'player_id':candidate,'position_group':role,'environment':env,'displaced_player':remove,
                    **{k:v for k,v in result.items() if k.endswith('60')},'replacement_certified':False})
    pd.DataFrame(contrasts).to_csv(OUT/'lineup_contrasts.csv',index=False);dump(OUT/'reference_environments.json',environments)
    dump(OUT/'export_manifest.json',{'created':now(),'information_cutoff':'2026-09-17','source_completed_season':SEASONS[-1],
        'profiles':len(current),'overall_players':int(out.future_net_goal_impact60.notna().sum()),'unknown_position_profiles':int(current.position_group.eq('U').sum()),
        'player_family_by_role':selections,'conditional_response_family':family,'contrasts':len(contrasts),'no_2026_27_outcomes':True,
        'display_is_monotone_in_native_net_goal_forecast':True,'replacement_certified':False,'production_admission':False})
    log(f'exported {len(current)} profiles, {out.future_net_goal_impact60.notna().sum()} Overall forecasts, {len(contrasts)} conditional contrasts')

if __name__=='__main__':main()
