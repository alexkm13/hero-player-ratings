"""Finite unattended queue; declared historical iteration, no outer-score search."""
import argparse,os,time,subprocess,sys,json
from .common import OUT,CACHE,TESTS,dump,now,log

def main():
    p=argparse.ArgumentParser();p.add_argument('--await-response-pid',type=int);a=p.parse_args()
    CACHE.mkdir(parents=True,exist_ok=True)
    ready=[OUT/'response'/str(t)/'manifest.json' for t in TESTS]+[OUT/'response/current/manifest.json']
    if a.await_response_pid:
        dump(OUT/'pipeline_status.json',{'updated':now(),'stage':'await response','pid':a.await_response_pid})
        for i in range(720):
            if all(f.exists() for f in ready):break
            try:os.kill(a.await_response_pid,0)
            except ProcessLookupError:raise RuntimeError('response process exited before every declared fold completed; inspect its retained log')
            time.sleep(5)
        else:raise TimeoutError('response wait exceeded one hour; no automatic refit')
    else:
        with (CACHE/'response_queue.log').open('a') as f:subprocess.run([sys.executable,'-W','error::RuntimeWarning','-m','research.hero_staged.response'],stdout=f,stderr=subprocess.STDOUT,check=True)
    stages=['player','diagnostics','report','export']
    for i,stage in enumerate(stages):
        log(f'queue stage {stage}')
        dump(OUT/'pipeline_status.json',{'updated':now(),'stage':stage,'pid':os.getpid(),'status':'running'})
        path=CACHE/f'{i:02d}_{stage}.log'
        with path.open('a') as f:
            result=subprocess.run([sys.executable,'-W','error::RuntimeWarning','-m',f'research.hero_staged.{stage}'],stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            dump(OUT/'pipeline_status.json',{'updated':now(),'stage':stage,'status':'failed','returncode':result.returncode,'log':str(path)})
            raise RuntimeError(f'{stage} failed; retained log {path}')
    dump(OUT/'pipeline_status.json',{'updated':now(),'stage':'complete','status':'complete','production_admission':False})
    log('Finite v2 challenger pipeline complete; consult evaluation_summary.json for scientific disposition')

if __name__=='__main__':main()
