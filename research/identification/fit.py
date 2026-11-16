"""Locked diagnostic ridge grid: conditional credit and design information."""
import json
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from research.identification.common import *

def solve(normal,rhs,k,names,player_penalty,system_penalty,inverse=False):
    penalty=np.r_[np.full(2*k,player_penalty),[0. if x=='intercept' else system_penalty if '_team' in x else 20. for x in names]]
    system=normal.copy();system.flat[::len(system)+1]+=penalty
    ch=cho_factor(system,lower=True,check_finite=True)
    beta=cho_solve(ch,rhs,check_finite=True)
    residual=np.einsum('ij,j->i',system,beta)-rhs
    err=float(np.linalg.norm(residual)/max(1.,np.linalg.norm(rhs)))
    assert err<1e-9,err
    covariance=cho_solve(ch,np.eye(len(system)),check_finite=True) if inverse else None
    return beta,covariance,{'relative_normal_equation_residual':err,'player_penalty':player_penalty,'system_penalty':system_penalty}

def predictions(players,names,betas,years,phase):
    frames=[];negative={name:[0.,0.] for name in betas}
    arr=np.column_stack(list(betas.values()));k=len(players)
    for year in years:
        q=prepare_ledger(year)
        keys=pd.MultiIndex.from_arrays([np.repeat(q.game.to_numpy(),2),np.tile([0,1],len(q))])
        unique=keys.unique().sort_values();index=unique.get_indexer(keys)
        count=np.zeros(len(unique));pred=np.zeros((len(unique),arr.shape[1]));seconds=np.zeros(len(unique))
        for start in range(0,len(q),30000):
            end=min(start+30000,len(q));xp,xn,y,h=design(q.iloc[start:end],players,names)
            rate=np.asarray(xp@arr[:2*k]+xn@arr[2*k:])
            loc=index[2*start:2*end]
            count+=np.bincount(loc,weights=y,minlength=len(unique))
            seconds+=np.bincount(loc,weights=h*3600,minlength=len(unique))
            for j,name in enumerate(betas):
                negative[name][0]+=float(h[rate[:,j]<0].sum());negative[name][1]+=float(h.sum())
                pred[:,j]+=np.bincount(loc,weights=np.maximum(rate[:,j],0)*h,minlength=len(unique))
        d=unique.to_frame(index=False);d.columns=['game','side']
        d['actual']=count;d['seconds']=seconds;d['season']=year;d['phase']=phase
        for j,name in enumerate(betas):d[name]=pred[:,j]
        frames.append(d)
        print('predictions',phase,year,'games',len(unique)//2,flush=True)
    return pd.concat(frames,ignore_index=True),{m:a/max(b,1e-15) for m,(a,b) in negative.items()}

def score(d,m):
    return {'team_rows':len(d),'games':int(d.game.nunique()),'count_mse':float(np.mean((d[m]-d.actual)**2)),
        'observed_predicted':float(d.actual.sum()/d[m].sum()),
        'prediction_correlation_to_baseline':float(np.corrcoef(d[m],d.p20_s20)[0,1]),
        'prediction_mae_to_baseline':float(np.mean(np.abs(d[m]-d.p20_s20)))}

def information(cov,normal,players,player_penalty,labels):
    k=len(players)
    netcov=cov[:k,:k]+cov[k:2*k,k:2*k]+cov[:k,k:2*k]+cov[k:2*k,:k]
    var=np.diag(netcov)
    info=1-player_penalty*var/2
    raw=np.diag(normal)[:k]
    hours=2/var-player_penalty
    assert np.min(info)>-1e-7 and np.max(info)<1+1e-7
    assert np.min(hours)>-1e-6 and np.max(hours-raw)<1e-5
    d=pd.DataFrame({'player_id':players,'net_design_information_fraction':np.clip(info,0,1),
        'conditional_net_hours':np.maximum(hours,0),'raw_hours':raw,
        'conditional_net_exposure_ratio':np.divide(np.maximum(hours,0),raw,out=np.zeros(k),where=raw>0)})
    pairs=pd.read_csv(OUT/'dominant_teammate_pairs.csv');pairs=pairs[pairs.window.eq('3year')].copy()
    ix=pd.Index(players);i=ix.get_indexer(pairs.player_id);j=ix.get_indexer(pairs.teammate_id)
    v_sum=var[i]+var[j]+2*netcov[i,j];v_diff=var[i]+var[j]-2*netcov[i,j]
    pairs['pair_sum_information_fraction']=1-player_penalty*v_sum/4
    pairs['pair_difference_information_fraction']=1-player_penalty*v_diff/4
    pairs['net_coefficient_covariance']=netcov[i,j]
    return d.merge(labels,on='player_id',validate='one_to_one'),pairs

def main():
    assert not (OUT/'fit_summary.json').exists(),'Do not overwrite fitted audit.'
    players=np.load(OUT/'players.npy');k=len(players);labels=pd.read_csv(OUT/'player_metadata.csv')
    pg=pd.read_parquet(PG);pg=pg[pg.season.isin(MAIN)].copy();pg['team']=pg.team.map(team)
    teams=sorted(pg.team.unique());team_seasons=sorted(set(zip(pg.team,pg.season)))
    names=system_names(teams,team_seasons)
    dump('design_metadata.json',{'players':players.tolist(),'nuisance_columns':names,'player_columns':2*k,'all_columns':2*k+len(names),'system_variables':'persistent franchise and team-season proxy; no observed coach effects'})
    stats={};audits=[]
    for year in MAIN:
        g,b,a=normal_stats(year,players,names);stats[year]=(g,b);audits.append(a)
        np.savez_compressed(OUT/f'normal_stats_{year}.npz',normal=g,rhs=b)
        print('normal stats',year,g.shape,a,flush=True)
    outputs=[];coefrows=[];fit_audit=[];info=None
    for phase,years in [('historical_source',MAIN[:2]),('current_3year',MAIN)]:
        g=sum(stats[y][0] for y in years);b=sum(stats[y][1] for y in years)
        betas={}
        for pp in PENALTIES:
            for sp in PENALTIES:
                name=f'p{int(pp)}_s{int(sp)}'
                beta,cov,a=solve(g,b,k,names,pp,sp,inverse=(phase=='current_3year' and pp==sp==20.))
                betas[name]=beta
                a.update(phase=phase,model=name,source_seasons=list(years));fit_audit.append(a)
                c=labels.copy();c['phase']=phase;c['model']=name
                c['offense_shots60']=beta[:k];c['suppression_shots60']=beta[k:2*k];c['net_shots60']=beta[:k]+beta[k:2*k]
                c['active_role_rank']=np.nan;c['active_role_pct']=np.nan
                active=c.active_card & c.position_group.isin(['F','D'])
                c.loc[active,'active_role_rank']=c.loc[active].groupby('position_group').net_shots60.rank(ascending=False,method='average')
                c.loc[active,'active_role_pct']=100*c.loc[active].groupby('position_group').net_shots60.rank(pct=True,method='average')
                coefrows.append(c)
                if cov is not None:
                    info,pairs=information(cov,g,players,pp,labels)
                    info.to_csv(OUT/'conditional_information.csv',index=False);pairs.to_csv(OUT/'pair_contrast_information.csv',index=False)
                    np.savez_compressed(OUT/'baseline_net_covariance.npz',players=players,net_covariance=cov[:k,:k]+cov[k:2*k,k:2*k]+cov[:k,k:2*k]+cov[k:2*k,:k])
                print('fit',phase,name,a['relative_normal_equation_residual'],flush=True)
        np.savez_compressed(OUT/f'coefficients_{phase}.npz',**betas)
        target_years=(MAIN[-1],) if phase=='historical_source' else MAIN
        pred,negative=predictions(players,names,betas,target_years,phase)
        outputs.append(pred)
        dump(f'process_{phase}.json',{'phase':phase,'models':{m:{**score(pred,m),'negative_rate_exposure_fraction':negative[m]} for m in betas}})
    coefs=pd.concat(coefrows,ignore_index=True);coefs.to_csv(OUT/'all_diagnostic_coefficients.csv',index=False)
    pred=pd.concat(outputs,ignore_index=True);pred.to_parquet(OUT/'team_game_predictions.parquet',index=False)
    current=coefs[coefs.phase.eq('current_3year')]
    ranks=current.groupby('player_id').agg(min_net=('net_shots60','min'),max_net=('net_shots60','max'),
        min_role_rank=('active_role_rank','min'),max_role_rank=('active_role_rank','max'),min_role_pct=('active_role_pct','min'),max_role_pct=('active_role_pct','max')).reset_index()
    ranks['net_sensitivity_range']=ranks.max_net-ranks.min_net
    ranks['role_percentile_sensitivity_range']=ranks.max_role_pct-ranks.min_role_pct
    ranks['role_rank_sensitivity_range']=ranks.max_role_rank-ranks.min_role_rank
    base=current[current.model.eq('p20_s20')][['player_id','net_shots60','offense_shots60','suppression_shots60','active_role_pct','active_role_rank']]
    ranks=ranks.merge(base,on='player_id',validate='one_to_one').merge(labels,on='player_id',validate='one_to_one')
    ranks.to_csv(OUT/'player_credit_sensitivity.csv',index=False)
    held=pred[pred.phase.eq('historical_source')]
    model_scores={m:score(held,m) for m in betas}
    similar=[]
    pivot=current.pivot(index='player_id',columns='model',values='active_role_pct')
    active_ids=labels[labels.active_card].player_id
    for i,m in enumerate(betas):
        for n in list(betas)[i+1:]:
            corr=float(np.corrcoef(held[m],held[n])[0,1])
            relative=abs(model_scores[m]['count_mse']/model_scores[n]['count_mse']-1)
            if relative<=.01 and corr>=.995:
                delta=(pivot.loc[active_ids,m]-pivot.loc[active_ids,n]).abs()
                similar.append({'model_a':m,'model_b':n,'heldout_prediction_correlation':corr,'relative_mse_difference':relative,
                    'median_player_percentile_difference':float(delta.median()),'p90_player_percentile_difference':float(delta.quantile(.9)),'max_player_percentile_difference':float(delta.max()),'players_difference_ge20':int(delta.ge(20).sum())})
    active=ranks[ranks.active_card]
    dump('fit_summary.json',{'scope':'sensitivity audit; conditional associations, not talent/causal rankings','normal_stat_audit':audits,
        'fits':fit_audit,'historical_heldout':model_scores,'process_similar_pairs':similar,
        'active_players':len(active),'role_percentile_range_quantiles':active.role_percentile_sensitivity_range.quantile([.1,.5,.9,1.]).to_dict(),
        'active_range_ge20':int(active.role_percentile_sensitivity_range.ge(20).sum()),
        'design_information_quantiles':info[info.active_card].net_design_information_fraction.quantile([.1,.5,.9]).to_dict()})
    print('fit audit complete',flush=True)

if __name__=='__main__':main()
