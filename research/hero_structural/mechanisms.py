"""Context-only comparators for recorded micro mechanisms, separate primary."""
import numpy as np
import pandas as pd
from .common import OUT,SEASONS,dump
from .factors import Canonical,StateIndex,build_factors
from .run import fit_context,factor_scores
from .model import D
from .report import MACRO

def main():
    data=Canonical.load();results=[];numerical=[]
    for test in SEASONS[2:]:
        train=data.subset(seasons=[s for s in SEASONS if s<test]);ti=StateIndex(train.state_inventory(),train.meta)
        factors,transform=build_factors(train,ti,include_skating=False)
        factors=[f for f in factors if f.name not in MACRO]
        betas,audits=fit_context(factors);beta={f.name:b for f,b in zip(factors,betas)}
        numerical.extend([{'test_season':test,**a} for a in audits])
        held=data.subset(seasons=[test]);index=StateIndex(held.state_inventory(),held.meta)
        scores=[]
        for date in sorted(held.risk.date.unique()):
            fs,_=build_factors(held.subset(date=date),index,include_skating=False)
            fs=[f for f in fs if f.name not in MACRO]
            day,_=factor_scores(fs,np.zeros((index.n,D)),[beta[f.name] for f in fs],'micro_context',str(date),index)
            scores.extend(day)
        baseline=pd.concat(scores,ignore_index=True)
        joint=pd.read_parquet(OUT/'evaluation'/str(test)/'joint'/'game_component_scores.parquet')
        joint=joint[~joint.factor.isin(MACRO)]
        for name,b in baseline.groupby('factor'):
            j=joint[joint.factor.eq(name)]
            assert b.n.sum()==j.n.sum()
            results.append({'test_season':test,'factor':name,'observations':int(b.n.sum()),
                'context_loss':float(b.loss.sum()),'joint_loss':float(j.loss.sum()),
                'gain_percent':float((b.loss.sum()-j.loss.sum())/b.loss.sum()*100),
                'context_brier':float(b.brier.sum()/b.n.sum()),'joint_brier':float(j.brier.sum()/j.n.sum())})
            if name.endswith('_incidence'):
                results[-1]['context_brier']=None;results[-1]['joint_brier']=None
        print(f'mechanism context baseline complete {test}',flush=True)
    table=pd.DataFrame(results);table.to_csv(OUT/'micro_mechanism_comparison.csv',index=False)
    dump(OUT/'micro_mechanism_audit.json',{'solvers':numerical,'scope':'recorded-event conditional mechanisms; source-selection caveat',
        'primary_event_gate_unchanged':True,'skating_forecast_not_evaluated':'prior physical snapshot assimilation is a measurement, not a held-out future skating prediction'})

if __name__=='__main__':main()
