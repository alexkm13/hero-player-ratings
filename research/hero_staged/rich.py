"""Audited A3Z player-season evidence complementing the timed root ledger."""
import numpy as np
import pandas as pd
from .common import CACHE,OUT,PRIOR,DATA,dump,sha
from .measure import prepare as timed

RICH=['act_full_shots','act_forecheck_pressures','act_dz_touches','act_zone_exits',
      'exec_full_entry','exec_full_denial','exec_full_retrieval','exec_exit_controlled','exec_carry_chance','exec_defender_danger']
SOURCE=PRIOR/'rtv_full_trait_repair_v2/traits/source_numerators_denominators.parquet'

def prepare():
    path=CACHE/'measurements_full_roleverified.parquet'
    if path.exists():return pd.read_parquet(path)
    result=timed();raw=pd.read_parquet(SOURCE)
    assert not raw.duplicated(['player_id','season_id']).any()
    original=list(result.columns)
    result=result.merge(raw,on=['player_id','season_id'],how='left',suffixes=('','_aggregate'),validate='one_to_one')
    # An absent current provider row does not erase an already observed
    # historical position/name. Resolve only from completed source or earlier,
    # matching v1's current export rule; never use later-year metadata.
    meta=pd.read_parquet(DATA/'player_meta.parquet');role_fills=0
    for season in result.season_id.unique():
        use=result.season_id.eq(season)
        last=meta[meta.season_id.le(season)].sort_values('season_id').groupby('player_id').tail(1).set_index('player_id')
        roles=result.loc[use,'player_id'].map(last.position_group)
        fill=result.loc[use,'position_group'].eq('U')&roles.isin(['F','D']);role_fills+=int(fill.sum())
        result.loc[roles.index[fill],'position_group']=roles[fill]
        names=result.loc[use,'player_id'].map(last.player_name)
        missing=result.loc[use,'player_name'].eq('')&names.notna();result.loc[names.index[missing],'player_name']=names[missing]
    invalid={}
    def channel(c,num,den,binary=False):
        num=pd.to_numeric(num,errors='coerce').to_numpy(float);den=pd.to_numeric(den,errors='coerce').to_numpy(float)
        bad=(num<0)|(den<0)|((num>den) if binary else False);invalid[c]=int(np.sum(bad))
        valid=np.isfinite(num)&np.isfinite(den)&(den>0)&~bad
        result[c+'_num']=np.where(valid,num,np.nan);result[c+'_den']=np.where(valid,den,0)
        result[c]=np.divide(num,den,out=np.full(len(num),np.nan),where=valid)
    for c,col in [('act_full_shots','shots'),('act_forecheck_pressures','forecheck_pressures'),('act_dz_touches','dz_puck_touches'),('act_zone_exits','zone_exits')]:
        channel(c,result[col],result.toi_5v5_min/60)
    for c,num,den in [('exec_full_denial','entry_denials','entry_defense_targets'),('exec_exit_controlled','controlled_exits','exit_resolution_attempts'),
        ('exec_carry_chance','carries_w_chances','entry_carries'),('exec_defender_danger','entry_carries_with_chance_against','entry_defense_targets')]:
        channel(c,result[num],result[den],True)
    channel('exec_full_entry',result.entry_attempts-result.failed_entries,result.entry_attempts,True)
    channel('exec_full_retrieval',result.retrieval_opportunities-result.botched_retrievals,result.retrieval_opportunities,True)
    result=result[list(original)+[c+s for c in RICH for s in ['', '_num','_den']]]
    result.to_parquet(path,index=False);result.to_parquet(OUT/'full_source_measurements.parquet',index=False)
    dump(OUT/'full_measurement_audit.json',{'source':str(SOURCE),'sha256':sha(SOURCE),'rows':len(result),'channels':RICH,
        'historical_position_carry_forward_fills':role_fills,'position_rule':'most recent completed source or earlier metadata; no later-year role assignment',
        'invalid_numerator_denominator_pairs_excluded':invalid,'source_vintage':'retained audited repair; original publication availability not certified',
        'interpretation':'recorded player-season A3Z evidence; opportunities and attempt difficulty remain imperfect; overlapping channels are correlated predictors, not independent likelihood evidence'})
    return result
