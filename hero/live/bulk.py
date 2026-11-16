"""Download declared public fastRhockey release assets once, never live scrape."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import requests

from hero.live.fetch import OUT, RAW

BASE='https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
ASSETS={f'{tag}/{prefix}_{s}.parquet' for tag,prefix in
        [('nhl_pbp_lite','play_by_play_lite'),('nhl_shifts','shifts'),('nhl_schedules','nhl_schedule')]
        for s in range(2022,2028)}


def run():
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    def get(asset):
        p=RAW/asset.split('/')[-1]
        if not p.exists():
            r=requests.get(f'{BASE}/{asset}',timeout=95)
            r.raise_for_status()
            assert r.content[:4]==r.content[-4:]==b'PAR1'
            p.write_bytes(r.content)
        with p.open('rb') as f:
            h=hashlib.file_digest(f,'sha256').hexdigest()
        print(p.name,p.stat().st_size,flush=True)
        return p.name,{'url':f'{BASE}/{asset}','sha256':h,'bytes':p.stat().st_size}
    with ThreadPoolExecutor(4) as e:
        result=dict(e.map(get,sorted(ASSETS)))
    (OUT/'bulk_manifest.json').write_text(json.dumps({'retrieval_utc':datetime.now(timezone.utc).isoformat(),
        'assets':result},indent=2)+'\n')


if __name__=='__main__':
    run()
