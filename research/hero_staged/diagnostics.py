"""Additional observed-outcome and roster-window checks; never select models."""
import json
import numpy as np
import pandas as pd
from .common import OUT,TASK,DATA,CACHE,TESTS,SEED,dump,log
from .measure import prepare as measurements,transform_fit,transform
from .response import prepare as response_rows,design
from .player import panel,ENDNAMES,fit_ridge,predict_ridge
from .report import statistics

def onice():
    path=CACHE/'onice_outcomes.parquet'
    if path.exists():return pd.read_parquet(path)
    r=pd.read_parquet(DATA/'risk.parquet');s=pd.read_parquet(DATA/'shots.parquet');parts=[]
    for season,risk in r.groupby('season'):
        ids=np.column_stack([np.stack(risk.home_players),np.stack(risk.away_players)]);players,inverse=np.unique(ids,return_inverse=True);inverse=inverse.reshape(ids.shape)
        seconds=np.bincount(inverse.ravel(),weights=np.repeat(risk.seconds.to_numpy(),10),minlength=len(players))
        shots=s[s.season.eq(season)];local=pd.Index(risk.interval_index).get_indexer(shots.interval_index)
        result={'player_id':players,'outcome_season':int(season),'actual_minutes':seconds/60}
        for head,column in [('shots',None),('xg','xg'),('goals','goal')]:
            y=np.zeros((len(risk),2));np.add.at(y,(local,shots.side.to_numpy(int)),1 if column is None else shots[column].to_numpy(float))
            own=np.column_stack([np.repeat(y[:,0,None],5,axis=1),np.repeat(y[:,1,None],5,axis=1)])
            against=np.column_stack([np.repeat(y[:,1,None],5,axis=1),np.repeat(y[:,0,None],5,axis=1)])
            gf=np.bincount(inverse.ravel(),weights=own.ravel(),minlength=len(players));ga=np.bincount(inverse.ravel(),weights=against.ravel(),minlength=len(players))
            assert np.isclose(gf.sum(),5*y.sum()) and np.isclose(gf.sum(),ga.sum())
            result['onice_'+head+'_for60']=gf/seconds*3600;result['onice_'+head+'_against60']=ga/seconds*3600
            result['onice_'+head+'_net60']=(gf-ga)/seconds*3600
        parts.append(pd.DataFrame(result))
    result=pd.concat(parts,ignore_index=True);result.to_parquet(path,index=False);return result

def future_onice():
    outcomes=onice();src=panel();oof=pd.read_parquet(OUT/'macro_oof_predictions.parquet')
    p=src.merge(oof,on=['player_id','season_id','position_group'],how='left',validate='one_to_one')
    p['outcome_season']=p.season_id+10001;p=p.merge(outcomes,on=['player_id','outcome_season'],how='inner',validate='one_to_one')
    p=p[p.eligible&p.actual_minutes.ge(300)&p[['oof_m1_'+e for e in ENDNAMES]].notna().all(axis=1)]
    pred=pd.read_parquet(OUT/'player_predictions.parquet').merge(outcomes,on=['player_id','outcome_season'],how='inner',validate='many_to_one')
    pred=pred[pred.actual_minutes.ge(300)]
    cols=['xgf','xga_suppression','gf','ga_suppression'];rows=[];allpred=[]
    for (role,source),g in pred.groupby(['position_group','season_id']):
        tr=p[p.position_group.eq(role)&p.season_id.lt(source)]
        if len(tr)<50:continue
        x=tr[['oof_m1_'+e for e in cols]].to_numpy();y=tr[['onice_xg_net60','onice_goals_net60']].to_numpy()
        calibration=fit_ridge(x,y,tr.season_id.to_numpy(),100,100)
        for family,q in g.groupby('family'):
            result=predict_ridge(calibration,q[['pred_'+e for e in cols]])
            for j,head in enumerate(['xg','goals']):rows.append({'source_season':int(source),'position_group':role,'family':family,'target':'actual_onice_'+head+'_net60',
                **statistics(q['onice_'+head+'_net60'],result[:,j]),'calibration_training_source_max':int(tr.season_id.max()),
                'interpretation':'observed environment forecast diagnostic; on-ice rates are not adjusted causal player credit'})
            out=q[['player_id','season_id','position_group','family','actual_minutes','onice_xg_net60','onice_goals_net60']].copy()
            out['forecast_onice_xg_net60']=result[:,0];out['forecast_onice_goals_net60']=result[:,1];allpred.append(out)
    pd.DataFrame(rows).to_csv(OUT/'future_onice_metrics.csv',index=False)
    pd.concat(allpred,ignore_index=True).to_parquet(OUT/'future_onice_predictions.parquet',index=False)

def roster():
    # Reuse the declared, retained v1 observational cohort; not a fresh test.
    windows=pd.read_csv(TASK/'outputs/structural_v1/roster_change_20_game_predictions.csv')
    windows=windows[windows.variant.eq('macro')].copy()
    m=measurements();frame=response_rows();r=pd.read_parquet(DATA/'risk.parquet')
    meta=r[['game','date','home_team','away_team']].drop_duplicates('game').sort_values(['date','game'])
    predictions=[]
    for season,ww in windows.groupby('season'):
        f=frame[frame.season.eq(season)].reset_index(drop=True)
        manifest=json.loads((OUT/'response'/str(season)/'manifest.json').read_text())
        selected=manifest['selected_family'];h=np.stack(f.home_players);a=np.stack(f.away_players)
        source=m[m.season_id.eq(season-10001)].copy()
        rolepri=transform_fit(m[m.season_id.lt(season)],['act_shots']);ids=np.unique(np.r_[h.ravel(),a.ravel()])
        actor=source.set_index('player_id').reindex(ids).reset_index();actor['position_group']=actor.position_group.fillna('U')
        activity=transform(actor,['act_shots'],rolepri)[:,0];activity=np.maximum(activity,1e-6)
        il=pd.Index(ids);hw=activity[il.get_indexer(h.ravel()).reshape(h.shape)];aw=activity[il.get_indexer(a.ravel()).reshape(a.shape)]
        for label,family in [('context','context'),('identities','identities'),('selected',selected)]:
            root=OUT/'response'/str(season);spec=json.loads((root/f'{family}.json').read_text());params=dict(np.load(root/f'{family}.npz',allow_pickle=False))
            d=design(f,m,family,priors=spec['priors'],scale=params['scale'],players=params['players'])
            expected=(d['exposure']*np.exp(d['X']@params['shots'])).reshape(-1,2)
            del d
            for w in ww.itertuples():
                games=meta[(meta.date.ge(w.first_date))&(meta.home_team.eq(w.new_team)|meta.away_team.eq(w.new_team))].head(20).game
                use=f.game.isin(games).to_numpy();home=(h==w.player_id).any(axis=1)&use;away=(a==w.player_id).any(axis=1)&use
                team_expected=float(expected[home,0].sum()+expected[away,1].sum())
                focal_expected=float((expected[home,0]*(hw[home]*(h[home]==w.player_id)).sum(axis=1)/hw[home].sum(axis=1)).sum()+
                    (expected[away,1]*(aw[away]*(a[away]==w.player_id)).sum(axis=1)/aw[away].sum(axis=1)).sum())
                observed=float(f.loc[home,'home_shots'].sum()+f.loc[away,'away_shots'].sum())
                assert np.isclose(observed,w.observed_team_shots)
                predictions.append({'player_id':int(w.player_id),'season':int(season),'first_date':w.first_date,'family':label,
                    'expected_team_shots':team_expected,'observed_team_shots':observed,'expected_focal_shots':focal_expected,
                    'observed_focal_shots':float(w.observed_focal_shots),'expected_teammate_shots':team_expected-focal_expected,
                    'observed_teammate_shots':observed-w.observed_focal_shots,'ratings_frozen':True,'conditional_observed_exposure':True,
                    'information_rule':'frozen preseason source profiles and response; fewer updates than v1 pre-window filter'})
        log(f'v2 frozen roster diagnostic {season}: {len(ww)} windows')
    table=pd.DataFrame(predictions);table.to_csv(OUT/'roster_window_predictions.csv',index=False)
    rows=[]
    for family,g in table.groupby('family'):
        for scope in ['team','focal']:
            observed=g[f'observed_{scope}_shots'].to_numpy();expected=g[f'expected_{scope}_shots'].to_numpy()
            rows.append({'family':family,'scope':scope,'windows':len(g),'expected_observed_ratio':float(expected.sum()/observed.sum()),
                'mae':float(np.mean(np.abs(expected-observed))),'count_nll':float(np.mean(expected-observed*np.log(np.maximum(expected,1e-12))))})
    pd.DataFrame(rows).to_csv(OUT/'roster_window_metrics.csv',index=False)
    dump(OUT/'diagnostic_manifest.json',{'frozen_roster_windows':len(windows),'players':int(windows.player_id.nunique()),
        'cohort_reused_from_inspected_v1':True,'information_rule':'all v2 variants frozen preseason; v1 used later pre-window filtering, so not equal-date head-to-head',
        'future_onice_interpretation':'past-calibrated observed rates; different outcome construction, shared games and environmental confounding remain',
        'selection_uses_diagnostics':False,'causal_or_replacement_transport_admitted':False})

def main():future_onice();roster()
if __name__=='__main__':main()
