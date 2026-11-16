from pathlib import Path
import os,json,hashlib
from datetime import datetime,timezone
import numpy as np

TASK=Path(os.environ.get('HERO_TASK_ROOT',str(Path(__file__).resolve().parents[2])))
OUT=Path(os.environ.get('HERO_V2_OUTPUT_ROOT',str(TASK/'artifacts/staged')))
DATA=Path(os.environ.get('HERO_CANONICAL_ROOT',str(TASK/'data/canonical')))
CACHE=TASK/'artifacts/staged_cache'
PCV=Path(os.environ.get('HERO_PCV_ROOT',str(TASK/'pcv')))
PRIOR=Path(os.environ.get('HERO_PRIOR_ROOT',str(TASK/'data/previous_runs')))
SEASONS=[20212022,20222023,20232024,20242025,20252026]
TESTS=SEASONS[2:]
SEED=20260918
ENDS={'xgf':('eh_rapm_xGF/60',1),'cf':('eh_rapm_CF/60',1),'gf':('eh_rapm_GF/60',1),
      'xga_suppression':('eh_rapm_xGA/60',-1),'ca_suppression':('eh_rapm_CA/60',-1),'ga_suppression':('eh_rapm_GA/60',-1)}
CATS=['Shot Creation','Transition Offense','Playmaking','Finishing','Forechecking','Shot Suppression','Zone / Transition Defense','Skating']
def now():return datetime.now(timezone.utc).isoformat()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''):h.update(c)
    return h.hexdigest()
def finite(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite numeric array')
    return x
def log(s):print(s,flush=True)
