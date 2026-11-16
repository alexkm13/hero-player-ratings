from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata
import os

TASK = Path(os.environ.get('HERO_TASK_ROOT',str(Path(__file__).resolve().parents[2])))
PCV = Path(os.environ.get('HERO_PCV_ROOT',str(TASK/'pcv')))
PRIOR = Path(os.environ.get('HERO_PRIOR_ROOT',str(TASK/'data/previous_runs')))
DATA = Path(os.environ.get('HERO_CANONICAL_ROOT',str(TASK / 'data/canonical')))
OUT = Path(os.environ.get('HERO_OUTPUT_ROOT',str(TASK / 'artifacts/structural')))
CATEGORIES = ['Shot Creation','Transition Offense','Playmaking','Finishing',
              'Forechecking','Shot Suppression','Zone / Transition Defense','Skating']
SHORT = ['shot_creation','transition_offense','playmaking','finishing',
         'forechecking','shot_suppression','zone_transition_defense','skating']
SEASONS = [20212022,20222023,20232024,20242025,20252026]
SEED = 20260917

def dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,default=str)+'\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def now():return datetime.now(timezone.utc).isoformat()

def norm_name(value):
    if value is None:return ''
    x=unicodedata.normalize('NFKD',str(value).replace('\xa0',' ').lower())
    x=''.join(c for c in x if not unicodedata.combining(c))
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]','',x)).strip()

def team(value):
    x=re.sub(r'[^A-Z]','',str(value).upper())
    return {'TB':'TBL','LA':'LAK','NJ':'NJD','SJ':'SJS','MON':'MTL','PHX':'ARI'}.get(x,x)

def season_id(label):
    start=int(str(label)[:4]);return start*10000+start+1
