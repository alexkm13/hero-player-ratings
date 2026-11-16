"""Replay a current player's six-head forecast from frozen raw source inputs."""
import argparse,json
from pathlib import Path
import pandas as pd
from functools import lru_cache
from .common import OUT
from .player import ENDNAMES,BASE,predict_ridge,correction_features

@lru_cache(maxsize=8)
def _inputs(root):return json.loads((Path(root)/'player_query_inputs.json').read_text())
@lru_cache(maxsize=8)
def _model(root,role):return json.loads((Path(root)/f'player_model_{role}.json').read_text())
@lru_cache(maxsize=8)
def _gate(root):return json.loads((Path(root)/'evaluation_summary.json').read_text())['development_gate']

def predict_player(root,pid):
    root=Path(root);inputs=_inputs(str(root));record=inputs.get(str(pid))
    if record is None:raise ValueError('player has no exported source evidence')
    role=record['position_group']
    if role not in ['F','D']:raise ValueError('unknown position; no compatible Overall forecast')
    model=_model(str(root),role);frame=pd.DataFrame([record])
    pred=predict_ridge(model['macro'],frame[BASE]);family=model['selected']['family']
    if family!='m1':
        correction=model['corrections'][family]
        if correction.get('available'):
            _,x,_=correction_features(frame,frame,family,correction['measurement_priors']);pred+=predict_ridge(correction,x)
    gate=_gate(str(root))
    return {'player_id':int(pid),'player_name':record['player_name'],'position_group':role,'family':family,
        'forecast':{e:float(pred[0,j]) for j,e in enumerate(ENDNAMES)},'future_net_goal_impact60':float(pred[0,2]+pred[0,5]),
        'as_of':'2026-09-17','outcome_season':20262027,'development_gate':gate,'production_admission':False,
        'scope':'future native provider endpoint forecast; exact strict-5v5, causal talent and replacement-team transport unproven'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,default=OUT);p.add_argument('--player',type=int,required=True)
    a=p.parse_args();print(json.dumps(predict_player(a.bundle,a.player),indent=2,allow_nan=False))

if __name__=='__main__':main()
