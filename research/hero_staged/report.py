"""Matched magnitudes, elite checks, all-family losses and cluster uncertainty."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from .common import OUT,SEED,TESTS,ENDS,dump,now,log
from .player import ENDNAMES

def statistics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);valid=np.isfinite(y)&np.isfinite(p);y=y[valid];p=p[valid]
    if len(y)<3:return {'n':len(y),'r2':None,'rmse':None,'mae':None,'pearson':None,'spearman':None}
    var=np.sum((y-y.mean())**2);err=np.sum((p-y)**2)
    return {'n':len(y),'r2':float(1-err/var) if var>0 else 0,'rmse':float(np.sqrt(np.mean((p-y)**2))),
        'mae':float(np.mean(np.abs(p-y))),'pearson':float(pearsonr(y,p).statistic) if np.std(p)>1e-12 else 0,
        'spearman':float(spearmanr(y,p).statistic) if np.std(p)>1e-12 else 0}

def player_bootstrap(pred):
    families=list(pred.family.unique());keys=['player_id','season_id','position_group']
    ref=pred[pred.family.eq('m1')].sort_values(keys).reset_index(drop=True)
    population=np.unique(ref.player_id);idx=pd.Index(population).get_indexer(ref.player_id)
    rng=np.random.default_rng(SEED);B=1000
    counts=rng.multinomial(len(population),np.full(len(population),1/len(population)),size=B).astype(float)
    actual=ref[['actual_'+e for e in ENDNAMES]].to_numpy(float)
    aligned={}
    for family in families:
        q=pred[pred.family.eq(family)].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(q[keys],ref[keys])
        aligned[family]=q[['pred_'+e for e in ENDNAMES]].to_numpy(float)
    cells=sorted(set(zip(ref.position_group,ref.season_id)));boots={f:np.zeros(B) for f in families};points={f:0. for f in families}
    for role,season in cells:
        use=(ref.position_group.eq(role)&ref.season_id.eq(season)).to_numpy();yy=actual[use];ii=idx[use]
        def aggregate(v):
            if v.ndim==1:return np.bincount(ii,weights=v,minlength=len(population))
            return np.column_stack([np.bincount(ii,weights=c,minlength=len(population)) for c in v.T])
        nn=np.einsum('ij,j->i',counts,aggregate(np.ones(len(yy))),optimize=False)
        ysum=np.einsum('ij,jk->ik',counts,aggregate(yy),optimize=False)
        y2=np.einsum('ij,jk->ik',counts,aggregate(yy**2),optimize=False)
        sst=y2-ysum**2/nn[:,None];sst=np.maximum(sst,1e-12)
        for f,pp in aligned.items():
            error=(pp[use]-yy)**2;sse=np.einsum('ij,jk->ik',counts,aggregate(error),optimize=False)
            boots[f]+=np.mean(1-sse/sst,axis=1)/len(cells)
            points[f]+=float(np.mean(1-error.sum(axis=0)/np.sum((yy-yy.mean(axis=0))**2,axis=0)))/len(cells)
    candidate=['activity','execution','both','both_shift','full_measurements','nonlinear_measurements','persistent_measurements','selected']
    delta={f:boots[f]-boots['m1'] for f in candidate};gain={f:points[f]-points['m1'] for f in candidate}
    deviations=np.column_stack([delta[f]-gain[f] for f in candidate]);critical=float(np.quantile(np.max(np.abs(deviations),axis=1),.99))
    return {'resamples':B,'clusters':len(population),'scheme':'player blocks retaining all source seasons; six equally weighted F/D-season cells',
        'mean_endpoint_r2':points,'family_gain_vs_m1':{f:{'gain':gain[f],'simultaneous_99_interval':[gain[f]-critical,gain[f]+critical],
        'pointwise_99_interval':np.quantile(delta[f],[.005,.995]).tolist()} for f in candidate},
        'simultaneous_family':candidate,'conditional_development_uncertainty':True}

def main():
    pred=pd.read_parquet(OUT/'player_predictions.parquet');rows=[];elite=[];slice_rows=[]
    for (family,role,season),g in pred.groupby(['family','position_group','season_id']):
        for endpoint in ENDNAMES+['net_goals']:
            rows.append({'family':family,'position_group':role,'source_season':int(season),'outcome_season':int(season+10001),
                'endpoint':endpoint,**statistics(g['actual_'+endpoint],g['pred_'+endpoint])})
        y=g.actual_net_goals.to_numpy();p=g.pred_net_goals.to_numpy()
        true=np.argsort(-y,kind='stable');rank=np.argsort(-p,kind='stable')
        for k in [25,50]:
            k=min(k,len(g));overlap=len(set(true[:k])&set(rank[:k]));relevance=(pd.Series(y).rank(method='average').to_numpy()/len(y))**4
            discount=1/np.log2(np.arange(k)+2);ndcg=float(np.sum(relevance[rank[:k]]*discount)/np.sum(relevance[true[:k]]*discount))
            elite.append({'family':family,'position_group':role,'source_season':int(season),'top_k':k,'recall':overlap/k,'precision':overlap/k,'ndcg':ndcg,
                'target':'future native provider GF+GA suppression; noisy provider ranking, not known true talent'})
        groups={'under23':g.age_years.lt(23),'23to27':g.age_years.between(23,27,inclusive='left'),
            '27to32':g.age_years.between(27,32,inclusive='left'),'32plus':g.age_years.ge(32),
            'high_source_toi':g.eh_rapm_TOI.ge(900),'lower_source_toi':g.eh_rapm_TOI.lt(900),
            'team_change':g.target_team.notna()&g.eh_rapm_Team.ne(g.target_team),'stable_team':g.target_team.notna()&g.eh_rapm_Team.eq(g.target_team),
            'tracking_absent':g.tracked_minutes.fillna(0).eq(0),'tracking_present':g.tracked_minutes.fillna(0).gt(0)}
        for label,use in groups.items():
            for e in ENDNAMES:
                z=g[use];slice_rows.append({'family':family,'position_group':role,'source_season':int(season),'slice':label,'endpoint':e,**statistics(z['actual_'+e],z['pred_'+e])})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'player_metrics.csv',index=False)
    elite=pd.DataFrame(elite);elite.to_csv(OUT/'elite_metrics.csv',index=False)
    pd.DataFrame(slice_rows).to_csv(OUT/'player_slice_metrics.csv',index=False)
    boot=player_bootstrap(pred);dump(OUT/'player_cluster_uncertainty.json',boot)
    cells=metrics[metrics.endpoint.ne('net_goals')].groupby(['family','position_group','source_season']).r2.mean().unstack('family')
    wins=int((cells.selected>cells.m1).sum());position={}
    for role,g in cells.groupby(level=0):position[role]=float((g.selected-g.m1).mean())
    erecall=elite[(elite.top_k==25)&elite.family.isin(['selected','m1'])].pivot(index=['position_group','source_season'],columns='family',values='recall')
    elitegap=float((erecall.selected-erecall.m1).mean())
    comp=boot['family_gain_vs_m1']['selected'];rules={'gain_ge_005':comp['gain']>=.005,'wins_ge_4_of_6':wins>=4,
        'simultaneous_99_lower_above_zero':comp['simultaneous_99_interval'][0]>0,'position_preservation':min(position.values())>=-.005,
        'elite_top25_preservation':elitegap>=-.05}
    response=[];comparisons=[]
    for test in TESTS:
        root=OUT/'response'/str(test);sc=pd.read_parquet(root/'game_scores.parquet');selection=json.loads((root/'manifest.json').read_text())['selected_family']
        for (family,head),g in sc.groupby(['family','head']):
            response.append({'test_season':test,'family':family,'head':head,'games':len(g),'nll_per60':float(g.loss.sum()/g.seconds.sum()*3600),
                'expected_observed_ratio':float(g.expected.sum()/g.observed.sum()),'selected_in_inner':family==selection})
        for label,family in [(v,v) for v in ['identities','activity','execution','both']]+[('selected',selection)]:
            a=sc[sc.family.eq(family)&sc['head'].eq('shots')].set_index('game');b=sc[sc.family.eq('context')&sc['head'].eq('shots')].set_index('game')
            q=pd.DataFrame({'game':a.index,'test_season':test,'family':label,
                'loss_reduction':b.loss.reindex(a.index).to_numpy()-a.loss.to_numpy(),'baseline_loss':b.loss.reindex(a.index).to_numpy()})
            comparisons.append(q)
    pd.DataFrame(response).to_csv(OUT/'response_metrics.csv',index=False)
    comparison=pd.concat(comparisons,ignore_index=True);comparison.to_csv(OUT/'response_game_comparisons.csv',index=False)
    game_summary={};rng=np.random.default_rng(SEED)
    for family,g in comparison.groupby('family'):
        value=g.loss_reduction.to_numpy();baseline=g.baseline_loss.to_numpy()
        sample=rng.integers(0,len(g),size=(1000,len(g)))
        gains=100*np.sum(value[sample],axis=1)/np.sum(baseline[sample],axis=1)
        game_summary[family]={'games':len(g),'count_process_loss_gain_percent':float(100*value.sum()/baseline.sum()),
            'pointwise_game_block_99_interval_percent':np.quantile(gains,[.005,.995]).tolist()}
    dump(OUT/'response_cluster_uncertainty.json',{'comparisons':game_summary,'scope':'count process only; seconds convention; diagnostic pointwise intervals',
        'all_models_on_same_games':True,'not_a_player_talent_interval':True})
    summary={'created':now(),'primary':'mean six native-unit endpoint R2 over equally weighted F/D-season cells',
        'mean_endpoint_r2':boot['mean_endpoint_r2'],'selected_gain_vs_m1':comp,'cell_wins':wins,'cells':len(cells),
        'position_gain_vs_m1':position,'net_goal_top25_recall_gain':elitegap,'development_rules':rules,
        'development_gate':'pass' if all(rules.values()) else 'fail','production_admission':False,
        'independent_strict_5v5_player_talent_gate':'unproven; provider EV scope, shared outcome sources and vintage limits remain',
        'prospective_gate':'unproven; 2026-27 outcomes not accessed','replacement_transport_gate':'unproven',
        'selected_families':{p.stem:json.loads(p.read_text())['family'] for p in (OUT/'player').glob('selection_*.json')}}
    dump(OUT/'evaluation_summary.json',summary)
    display=cells.reset_index()[['position_group','source_season','m1','selected','activity','execution','both','both_shift','full_measurements','nonlinear_measurements','persistent_measurements','benchmark_hero_3yr']]
    table='| '+' | '.join(map(str,display.columns))+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'
    for values in display.itertuples(index=False,name=None):table+='| '+' | '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |\n'
    lines=['# Staged HERO challenger v2 — results','',f'**Development gate: {summary["development_gate"].upper()}. Production admission: false.**','',
        'The eight category vocabulary is preserved, with independently measured evidence, a frozen macro forecast and learned future-impact corrections. The historical iteration uses only earlier chronological folds; all outer seasons are inspected development data.','',
        f'Selected mean six-endpoint R²: {boot["mean_endpoint_r2"]["selected"]:.4f}; strong M1: {boot["mean_endpoint_r2"]["m1"]:.4f}. Gain: {comp["gain"]:+.5f}, simultaneous exploratory player-block 99% interval [{comp["simultaneous_99_interval"][0]:+.5f}, {comp["simultaneous_99_interval"][1]:+.5f}]. Cell wins: {wins}/{len(cells)}.','',table,'',
        f'F/D mean gains: {position}. Future native-goal top-25 recall gain: {elitegap:+.4f}. These are provider endpoint/ranking forecasts, not talent R² or causal credit.','',
        '## What the challenger produces','',
        'Source category profiles; six separate native contribution-rate endpoint forecasts; Overall equal to predicted GF impact plus GA suppression; current conditional team shot/xG/GF response and same-role swaps. Category percentiles are displays, not imposed value weights. The lineup response and future-provider Overall have different estimands and are not silently blended.','',
        '## Limits','',
        '- Exact strict-5v5 provider endpoint scope and original publication-time vintages are not independently certified. HERO three-year display is reconstructed, not an exact historical production snapshot.','- Previously inspected seasons provide development evidence only. No 2026–27 game outcomes were accessed.','- Measured activity/conditional execution has incomplete tracking, identities, difficulty and opportunity coverage. Named passer/recovery involvement is not personal completion probability.','- Additive player response coefficients are associations conditional on observed lineups and known context. Neither causal swaps nor replacement-team player quality is established.','- Bootstrap intervals condition on fitted models; source/provider estimation uncertainty and every team dependence are not included. Category conditional uncertainty is not a full talent confidence interval.','',
        'Read CHARTER.md, MODEL_SPEC.md, iteration_record.md, player_metrics.csv, elite_metrics.csv, response_metrics.csv and verification manifests.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    selection_rows=[]
    for p in sorted((OUT/'player').glob('selection_*.json')):
        s=json.loads(p.read_text());selection_rows.append(f'- {s["role"]}, source {s["source_season"]}: {s["family"]}; training targets complete through {s["max_training_target"]}.')
    count=sum(len(json.loads(p.read_text())) for p in (OUT/'response').glob('*/inner_trials.json'))
    trials=json.loads((OUT/'player_inner_trials.json').read_text())
    history=['# V2 finite historical iteration — completed','',f'Response inner settings executed: {count}. Player inner settings recorded: {len(trials)}. Every family and trial is retained, including zero-correction fallbacks when no earlier measurement history exists.','',
        'Macro iteration varies endpoint-specific ridge shrinkage and historical recency. Skill/value iteration compares activity, conditional execution, their combination, source-completed response effects, fuller A3Z evidence, a fixed nonlinear ensemble, and a fixed three-season measurement pool. Training-only chronological selection freezes each chosen setting before its later outcome is scored.','',*selection_rows,'',
        f'Outer player development gate: {summary["development_gate"]}. Selected mean endpoint R² {boot["mean_endpoint_r2"]["selected"]:.5f}; M1 {boot["mean_endpoint_r2"]["m1"]:.5f}. No model or blend was invented from these outer metrics.','',
        'The finite search stops here. Current fits legitimately include all completed outcomes through 2025–26; no 2026–27 game results exist in this run. Prospective and independent talent/transport admission remain unproven.']
    (OUT/'iteration_record.md').write_text('\n'.join(history)+'\n')
    log(f'v2 report: development gate={summary["development_gate"]}; selected gain={comp["gain"]:+.5f}')

if __name__=='__main__':main()
