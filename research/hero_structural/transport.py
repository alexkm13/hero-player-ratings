"""Frozen pre-change conditional next-20-team-game supply/allocation diagnostic.

Selected observational roster changes, not randomized swaps or causal effects.
All players' states freeze before the first new-team game. Future actual lineups
define conditional risk exposure, never a deployment prediction.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from .common import OUT,SEASONS,SEED,dump
from .factors import Canonical,StateIndex
from .run import load_fit,active_fit

def appearances(risk):
    rows=[]
    for side in ['home','away']:
        teamcol=side+'_team';players='h' if side=='home' else 'a'
        f=risk[['game','season','date',teamcol,'seconds',players]].explode(players)
        f=f.rename(columns={teamcol:'team',players:'player_id'})
        rows.append(f.groupby(['game','season','date','team','player_id']).seconds.sum().reset_index())
    return pd.concat(rows,ignore_index=True).sort_values(['player_id','date','game'])

def predict_window(risk,shots,index,state,betas,pid,team,variant):
    side=np.where(risk.home_team.eq(team),0,1)
    h=index.map(risk.season.to_numpy(),np.stack(risk.h));a=index.map(risk.season.to_numpy(),np.stack(risk.a))
    own=np.where(side[:,None]==0,h,a);opp=np.where(side[:,None]==0,a,h)
    si=index.map(int(risk.season.iloc[0]),pid);on=(own==si).any(axis=1)
    if not on.any():return None
    own=own[on];opp=opp[on];r=risk.iloc[np.flatnonzero(on)];side=side[on]
    b=betas['shot_incidence'];dims=[0] if variant=='macro' else [0,1,2,4,7]
    own_state=state[own]
    opp_state=state[opp]
    eta=np.einsum('ij,j->i',own_state.sum(axis=1)[:,dims],b[:len(dims)],optimize=False)-opp_state[:,:,5].sum(axis=1)+b[-2]+b[-1]*(1-side)
    mu=np.exp(eta)*r.seconds.to_numpy()
    b=betas['shooter_allocation'];logits=own_state[:,:,0]+own_state[:,:,8]+b[-1]*(index.role[own]=='D')
    p=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));share=(p*(own==si)).sum(axis=1)
    s=shots[shots.interval_index.isin(r.interval_index)].copy()
    sidemap=dict(zip(r.interval_index,side));s=s[s.side.eq(s.interval_index.map(sidemap))]
    focal=s.shooter_id.eq(pid).sum();team_n=len(s)
    return {'on_ice_minutes':float(r.seconds.sum()/60),'expected_team_shots':float(mu.sum()),
        'observed_team_shots':int(team_n),'expected_focal_shots':float((mu*share).sum()),
        'observed_focal_shots':int(focal),'expected_teammate_shots':float((mu*(1-share)).sum()),
        'observed_teammate_shots':int(team_n-focal),'expected_focal_share':float((mu*share).sum()/mu.sum()),
        'observed_focal_share':float(focal/team_n) if team_n else None}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--summarize-only',action='store_true',help='Resume summaries from completed window predictions')
    args=parser.parse_args()
    if args.summarize_only:
        summarize(pd.read_csv(OUT/'roster_change_20_game_predictions.csv'))
        return
    data=Canonical.load();apps=appearances(data.risk);changes=[]
    for pid,g in apps.groupby('player_id'):
        last=None
        for row in g.itertuples():
            if last is not None and row.team!=last.team and row.season in SEASONS[2:]:
                changes.append({'player_id':int(pid),'old_team':last.team,'new_team':row.team,
                    'first_game':int(row.game),'first_date':row.date,'season':int(row.season)})
            last=row
    results=[]
    for season in SEASONS[2:]:
        subset=data.subset(seasons=[season]);teamgames=pd.concat([
            subset.risk[['game','date','home_team']].drop_duplicates().rename(columns={'home_team':'team'}),
            subset.risk[['game','date','away_team']].drop_duplicates().rename(columns={'away_team':'team'})]).sort_values(['date','game'])
        cases=[x for x in changes if x['season']==season]
        for variant in ['macro','joint']:
            path=OUT/'evaluation'/str(season)/variant
            states=np.load(path/'pregame_states.npz');dates=states['dates'];means=states['state'];dateindex={str(d):i for i,d in enumerate(dates)}
            index=StateIndex(pd.read_parquet(path/'inventory.parquet'),data.meta)
            _,_,params=load_fit(active_fit(season,variant));betas={k[5:]:v for k,v in params.items() if k.startswith('beta_')}
            for case in cases:
                window=teamgames[teamgames.team.eq(case['new_team'])&teamgames.date.ge(case['first_date'])].head(20)
                if len(window)!=20:continue
                games=window.game.to_numpy();eligible=apps[apps.player_id.eq(case['player_id'])&apps.game.isin(games)]
                if eligible.team.ne(case['new_team']).any():continue
                frozen=means[dateindex[case['first_date']]].astype(float)
                r=subset.risk[subset.risk.game.isin(games)];s=subset.shots[subset.shots.game.isin(games)]
                values=predict_window(r,s,index,frozen,betas,case['player_id'],case['new_team'],variant)
                if values and values['on_ice_minutes']>=30:results.append({**case,'variant':variant,'team_games':20,
                    'focal_games':int(eligible.game.nunique()),'ratings_frozen':True,'conditional_observed_exposure':True,**values})
            print(f'transport {season}/{variant}: {len(cases)} possible changes',flush=True)
    table=pd.DataFrame(results);table.to_csv(OUT/'roster_change_20_game_predictions.csv',index=False)
    summarize(table)

def summarize(table):
    summaries=[]
    for variant,g in table.groupby('variant'):
        rec={'variant':variant,'windows':len(g),'players':g.player_id.nunique()}
        for component in ['team','focal','teammate']:
            expected=g['expected_'+component+'_shots'].to_numpy();observed=g['observed_'+component+'_shots'].to_numpy()
            # Count diagnostic, factorial constant omitted identically.
            rec[component+'_count_nll_mean']=float(np.mean(expected-observed*np.log(np.maximum(expected,1e-12))))
            rec[component+'_mae']=float(np.mean(np.abs(expected-observed)))
            rec[component+'_expected_observed_ratio']=float(expected.sum()/observed.sum())
        rec['share_mae']=float(np.nanmean(np.abs(g.expected_focal_share-g.observed_focal_share)))
        summaries.append(rec)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'roster_change_metrics.csv',index=False)
    # Cluster complete windows by focal player; repeated windows can overlap.
    keys=['player_id','season','first_game'];paired=table.pivot(index=keys,columns='variant',values=['expected_team_shots','expected_focal_shots','observed_team_shots','observed_focal_shots']).dropna()
    interval={};rng=np.random.default_rng(SEED)
    for component in ['team','focal']:
        expected_m=paired['expected_'+component+'_shots']['macro'];expected_j=paired['expected_'+component+'_shots']['joint']
        observed=paired['observed_'+component+'_shots']['joint']
        assert np.array_equal(observed,paired['observed_'+component+'_shots']['macro'])
        losses=pd.DataFrame({'player_id':paired.index.get_level_values('player_id'),
            'difference':((expected_m-observed*np.log(expected_m))-(expected_j-observed*np.log(expected_j))).to_numpy()})
        clustered=losses.groupby('player_id').difference.agg(['sum','count']);draw=[]
        for _ in range(2000):
            ids=rng.integers(0,len(clustered),len(clustered));draw.append(float(clustered.iloc[ids]['sum'].sum()/clustered.iloc[ids]['count'].sum()))
        interval[component]={'mean_window_nll_reduction':float(losses.difference.mean()),
            'player_block_95_interval':np.quantile(draw,[.025,.975]).tolist()}
    dump(OUT/'roster_change_summary.json',{'matched_windows':len(paired),'players':table.player_id.nunique(),
        'comparison':interval,'status':'diagnostic; no causal transport admission',
        'selection':'observed team change, complete next20 team games, >=30 focal on-ice minutes, no second team change inside window',
        'frozen_pre_first_date_states':True,'future_actual_lineups_are_exposure_only':True,
        'counts':summaries,'unproven':'entry/exit role adaptation, deployment, causal effects, incumbent player-impact comparison'})

if __name__=='__main__':main()
