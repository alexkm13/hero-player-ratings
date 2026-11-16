"""Common-row development metrics; never promotes a candidate."""
import json
import numpy as np
import pandas as pd
from .common import OUT,SEED,SEASONS,dump,now

MACRO=['shot_incidence','shooter_allocation','shot_quality','goal_conversion']

def game_comparison(wide,baseline):
    keep=wide[['joint',baseline]].dropna()
    diff=keep[baseline].to_numpy()-keep.joint.to_numpy()
    rng=np.random.default_rng(SEED);means=[]
    # A game includes both teams/all marks; never resample actor rows.
    for _ in range(2000):
        draw=rng.integers(0,len(diff),len(diff))
        means.append(float(diff[draw].sum()/keep[baseline].to_numpy()[draw].sum()*100))
    return {'baseline':baseline,'games':len(keep),
            'gain_percent':float(diff.sum()/keep[baseline].sum()*100),
            'game_block_gain_percent_interval_95':np.quantile(means,[.025,.975]).tolist(),
            'mean_game_loss_reduction':float(diff.mean())}

def main():
    pieces=[];cal=[];manifests=[]
    for seas in SEASONS[2:]:
        for variant in ['context','macro','joint']:
            path=OUT/'evaluation'/str(seas)/variant
            if not (path/'manifest.json').exists():raise RuntimeError(f'Incomplete evaluation: {path}')
            f=pd.read_parquet(path/'game_component_scores.parquet');f['test_season']=seas;pieces.append(f)
            c=pd.read_parquet(path/'calibration_bins.parquet');c['test_season']=seas;cal.append(c)
            manifests.append(json.loads((path/'manifest.json').read_text()))
    scores=pd.concat(pieces,ignore_index=True)
    macro=scores[scores.factor.isin(MACRO)]
    game=macro.groupby(['test_season','game','variant']).loss.sum().unstack('variant')
    assert game.notna().all().all(), 'Different game populations'
    counts=macro.groupby(['test_season','factor','variant']).n.sum().unstack('variant')
    assert counts.eq(counts.context,axis=0).all().all(), 'Different macro event rows'
    game.to_csv(OUT/'game_loss_comparison.csv')
    exposures=macro[macro.factor.eq('shot_incidence')].groupby(['test_season','variant']).exposure.sum()
    sums=macro.groupby(['test_season','variant']).loss.sum()
    rates=(sums/(exposures/3600)).rename('macro_nll_per_60_team_minutes').reset_index()
    rates.to_csv(OUT/'fold_metrics.csv',index=False)
    components=scores.groupby(['test_season','variant','factor']).agg(loss=('loss','sum'),n=('n','sum'),
                 brier=('brier','sum'),expected=('expected','sum'),observed=('observed','sum')).reset_index()
    components['loss_per_observation']=components.loss/components.n
    components['brier_per_observation']=components.brier/components.n
    incidence=components.factor.str.endswith('_incidence')
    allocation=components.factor.str.endswith('_allocation')
    components['observation_grain']=np.where(incidence,'team risk interval',np.where(allocation,'one actor choice','one binary mark'))
    components['selected_actor_mean_probability']=np.where(allocation,components.expected/components.n,np.nan)
    components.loc[incidence,'brier_per_observation']=np.nan
    components.loc[allocation,['expected','observed']]=np.nan
    components=components.drop(columns='brier')
    components.to_csv(OUT/'component_metrics.csv',index=False)
    position=macro[macro.position.isin(['F','D','U'])].groupby(['test_season','variant','factor','position']).agg(
        loss=('loss','sum'),n=('n','sum'),brier=('brier','sum')).reset_index()
    position['loss_per_observation']=position.loss/position.n
    position['brier_per_observation']=position.brier/position.n
    position.to_csv(OUT/'position_mark_metrics.csv',index=False)
    calibration=pd.concat(cal,ignore_index=True).groupby(['test_season','variant','factor','bin']).sum(numeric_only=True).reset_index()
    calibration['predicted']=calibration.predicted_sum/calibration.n
    calibration['observed']=calibration.observed_sum/calibration.n
    calibration.to_csv(OUT/'calibration.csv',index=False)
    pooled={b:game_comparison(game,b) for b in ['context','macro']}
    fold={str(s):{b:game_comparison(game.loc[s],b) for b in ['context','macro']} for s in SEASONS[2:]}
    best=min(['context','macro'],key=lambda b:float(game[b].sum()))
    comparison=pooled[best]
    fold_wins=sum(fold[str(s)][best]['gain_percent']>0 for s in SEASONS[2:])
    # Component/position guard is declared material at >1% loss regression.
    # Report every component so a pooled win cannot conceal a mark failure.
    pooledcomp=macro.groupby(['variant','factor']).loss.sum().unstack('variant')
    regressions=((pooledcomp.joint-pooledcomp[best])/pooledcomp[best]*100).to_dict()
    ps=macro[macro.position.isin(['F','D'])].groupby(['position','variant']).loss.sum().unstack('variant')
    preg=((ps.joint-ps[best])/ps[best]*100).to_dict()
    numerical=[]
    for m in manifests:
        training=m['training_manifest']['solver']
        if m['variant']!='context' and not training.get('final_success',False):numerical.append(f"{m['test_season']}/{m['variant']} training")
        if m['failed_updates']:numerical.append(f"{m['test_season']}/{m['variant']}: {m['failed_updates']} filter updates")
    event_pass=(comparison['gain_percent']>=1 and fold_wins>=2 and comparison['game_block_gain_percent_interval_95'][0]>0
                and max(regressions.values())<=1 and max(preg.values(),default=0)<=1 and not numerical)
    result={'created':now(),'status':'research only; no production promotion','best_event_baseline':best,
            'pooled':pooled,'folds':fold,'fold_wins_against_best':fold_wins,'common_macro_rows_verified':True,
            'component_regression_percent':regressions,'position_mark_regression_percent':preg,
            'numerical_failures':numerical,'event_gate':'pass' if event_pass else 'fail',
            'player_magnitude_incumbent_M1_RTV_gate':'unproven; different endpoints and forecast times',
            'elite_ranking_gate':'unproven; no independently matched player-impact test',
            'roster_change_gate':'diagnostic only; conditional observed exposure',
            'prospective_2026_27_gate':'unproven; outcomes not accessed'}
    dump(OUT/'evaluation_summary.json',result)
    lines=['# Structural talent challenger v1 — development evaluation','',
        'This implements a new marked-event response architecture while retaining the eight HERO categories. It is a research candidate, not an approved replacement for the incumbent.', '',
        '## Common-row event comparison','',
        '| Test season | Context NLL/60 | Macro-only NLL/60 | Joint NLL/60 | Joint gain vs macro |',
        '|---|---:|---:|---:|---:|']
    table=rates.pivot(index='test_season',columns='variant',values='macro_nll_per_60_team_minutes')
    for s,row in table.iterrows():lines.append(f"| {str(s)[:4]}–{str(s)[4:]} | {row.context:.3f} | {row.macro:.3f} | {row.joint:.3f} | {fold[str(s)]['macro']['gain_percent']:.3f}% |")
    lines+=['',f"Best eligible event baseline: **{best}**. Pooled joint loss improvement: **{comparison['gain_percent']:.3f}%**, game-block 95% interval **[{comparison['game_block_gain_percent_interval_95'][0]:.3f}%, {comparison['game_block_gain_percent_interval_95'][1]:.3f}%]**. Fold wins: {fold_wins}/3. Event gate: **{result['event_gate']}**.",'',
        'The loss uses seconds as the intensity clock, includes all zero-event exposure, and adds conditional actor/quality/conversion marks once. Its relative percent depends on this locked clock-unit convention. All candidates share the same macro population. A game, including both teams and all its marks, is the uncertainty block. These intervals do not include every source/parameter uncertainty.', '',
        '## Scientific limits and required gates','',
        '- A3Z coverage and identity linkage are incomplete. Its clock intensities model recorded events; they are not certified complete opportunity counts. Skating observes prior-season all-state maximum speed, not agility or strict-5v5 movement.',
        '- Eight skills and responsibility deviations are regularized latent quantities. Activity and skill remain partly confounded where independent execution marks are weak. Category names do not establish full construct validity.',
        '- Local multivariate curvature omits between-player and shared-parameter dependence. Conditional skill SDs are not full talent confidence intervals.',
        '- Annual training states and daily future filtering use different resolution; missing historical years are collapsed. Micro execution selects recorded non-shot roots, shot-assister timestamps are terminal shot times, and shared unmeasured context can correlate channels. The conditional likelihood uses a working independence approximation.',
        '- Future player impact magnitude, elite ranking, and matched incumbent/M1/RTV comparisons remain unproven. Lower event loss would not establish a better player Overall.',
        '- Observed lineup contrasts condition on exposure and fixed roles. Weak observed environments and low-usage reference players do not certify replacement-level teams or causal transport.',
        '- 2023–26 tests are development data. No 2026–27 outcomes were accessed. Prospective admission remains necessary.', '',
        '## Numerical status','',str(numerical) if numerical else 'All saved training fits and daily filter updates reported convergence.', '',
        '## Reproduction','',
        '`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -W error::RuntimeWarning -m research.hero_structural.run fit`',
        '`... -m hero_structural.run evaluate`', '`... -m hero_structural.run report`', '`... -m hero_structural.run export`','',
        'Read IMPLEMENTATION_MEMORY.md before resuming. See executable_config.json, source_audit.json, ingestion_audit.json, verification.json, fit/evaluation manifests and experiment_record.md.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
