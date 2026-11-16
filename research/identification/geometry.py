"""Player-level separability support, without assuming it is causal skill."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from research.identification.common import *

def summarize(s,c,units,players):
    sec=np.diag(s).copy()*3600
    mate=s.copy()*3600;np.fill_diagonal(mate,0)
    top=mate.argmax(axis=1);shared=mate[np.arange(len(players)),top]
    p=np.divide(mate,4*sec[:,None],out=np.zeros_like(mate),where=sec[:,None]>0)
    op=np.divide(c*3600,5*sec[:,None],out=np.zeros_like(c),where=sec[:,None]>0)
    d=pd.DataFrame({'player_id':players,'minutes':sec/60,'dominant_teammate_id':np.where(sec>0,players[top],0),
        'dominant_teammate_minutes':shared/60,'dominant_teammate_fraction':np.divide(shared,sec,out=np.zeros_like(sec),where=sec>0),
        'minutes_apart_from_dominant_teammate':(sec-shared)/60,'teammates_60min':(mate>=3600).sum(axis=1),
        'opponents_60min':(c>=1).sum(axis=1),'effective_teammates':np.divide(1.,np.sum(p*p,axis=1),out=np.zeros(len(players)),where=np.sum(p*p,axis=1)>0),
        'effective_opponents':np.divide(1.,np.sum(op*op,axis=1),out=np.zeros(len(players)),where=np.sum(op*op,axis=1)>0)})
    u=units.groupby([f'p{i}' for i in range(5)],as_index=False).seconds.sum()
    ids=u[[f'p{i}' for i in range(5)]].to_numpy().ravel()
    x=pd.DataFrame({'player_id':ids,'seconds':np.repeat(u.seconds.to_numpy(),5)})
    x['seconds_sq']=x.seconds**2
    a=x.groupby('player_id').agg(unit_seconds=('seconds','sum'),dominant_unit_seconds=('seconds','max'),unit_seconds_sq=('seconds_sq','sum'),distinct_units=('seconds','size'))
    a['effective_units']=a.unit_seconds**2/a.unit_seconds_sq
    a['dominant_unit_fraction']=a.dominant_unit_seconds/a.unit_seconds
    d=d.merge(a.reset_index(),on='player_id',how='left',validate='one_to_one')
    check=d.minutes.gt(0)
    assert np.max(np.abs(d.loc[check,'unit_seconds']-d.loc[check,'minutes']*60))<1e-6
    ncomp,component=connected_components(sparse.csr_matrix(mate>=3600),directed=False)
    d['teammate_graph_component']=component
    pair=[]
    for i,j in enumerate(top):
        if sec[i]<=0:continue
        pair.append({'player_id':int(players[i]),'teammate_id':int(players[j]),'shared_minutes':float(shared[i]/60),
            'player_shared_fraction':float(shared[i]/sec[i]),'teammate_shared_fraction':float(shared[i]/sec[j]),
            'coice_cosine':float(shared[i]/np.sqrt(sec[i]*sec[j]))})
    return d,pd.DataFrame(pair),{'components_including_zero_exposure':ncomp,'components_among_exposed':int(d.loc[check,'teammate_graph_component'].nunique())}

def main():
    players=np.load(OUT/'players.npy');labels=pd.read_csv(OUT/'player_metadata.csv')
    pg=pd.read_parquet(PG);pg['canonical_team']=pg.team.map(team)
    units=pd.read_parquet(OUT/'annual_units.parquet')
    rows=[];pairs=[];report={}
    for label,years in [('3year',MAIN),('5year',YEARS)]:
        s=sum(np.load(OUT/f'geometry_{y}.npz')['same'] for y in years)
        c=sum(np.load(OUT/f'geometry_{y}.npz')['cross'] for y in years)
        d,p,graph=summarize(s,c,units[units.season.isin(years)],players)
        d=d.merge(labels,on='player_id',validate='one_to_one');d['window']=label
        hist=pg[pg.season.isin(years)]
        context=hist.groupby('player_id').agg(games=('game','nunique'),franchises=('canonical_team','nunique'),seasons=('season','nunique')).reset_index()
        d=d.merge(context,on='player_id',how='left',validate='one_to_one')
        lookup=labels.set_index('player_id').player_name
        d['dominant_teammate_name']=d.dominant_teammate_id.map(lookup)
        p['window']=label;p['player_name']=p.player_id.map(lookup);p['teammate_name']=p.teammate_id.map(lookup)
        # Novel partner exposure in the last year, conditional on four teammate slots.
        last=np.load(OUT/f'geometry_{years[-1]}.npz')['same'].copy();np.fill_diagonal(last,0)
        prior=s-last;np.fill_diagonal(prior,0)
        known=(prior>=1.)
        denom=last.sum(axis=1)
        novelty=np.divide(np.sum(last*~known,axis=1),denom,out=np.full(len(players),np.nan),where=denom>0)
        d['latest_year_new_partner_exposure_fraction']=novelty
        rows.append(d);pairs.append(p)
        active=d[d.active_card]
        report[label]={'active_players':len(active),'median_minutes':float(active.minutes.median()),
            'dominant_mate_fraction_quantiles':active.dominant_teammate_fraction.quantile([.1,.5,.9]).to_dict(),
            'apart_minutes_quantiles':active.minutes_apart_from_dominant_teammate.quantile([.1,.5,.9]).to_dict(),
            'active_under300_minutes_apart':int(active.minutes_apart_from_dominant_teammate.lt(300).sum()),
            'active_atleast80pct_with_one_mate':int(active.dominant_teammate_fraction.ge(.8).sum()),
            'active_single_franchise':int(active.franchises.eq(1).sum()),
            'median_effective_units':float(active.effective_units.median()),'graph':graph}
    pd.concat(rows,ignore_index=True).to_csv(OUT/'player_geometry.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(OUT/'dominant_teammate_pairs.csv',index=False)
    dump('geometry_summary.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
