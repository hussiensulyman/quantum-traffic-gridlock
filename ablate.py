import json,numpy as np
from bench import bench
import os

out=json.load(open('ablation_R.json')) if os.path.exists('ablation_R.json') else {}

def row(tag,extra):
    if tag in out: return
    cs,cr,res=bench('scenario_R.json',extra=extra); L=[r[0] for r in res]; E=[r[1] for r in res]
    out[tag]=dict(min_lift=min(L),mean_lift=float(np.mean(L)),mean_enrich=float(np.mean(E)),min_enrich=min(E)); print(tag,{k:round(float(v),3) for k,v in out[tag].items()},flush=True); json.dump(out,open('ablation_R.json','w'),indent=1)

for d in (1,2,3,4): row(f'depth={d}',('--depth',str(d)))

for k in (0.0,0.5,1.5,3.0): row(f'k_ov={k}',('--k_ov',str(k)))

for i in ('ramp','random','zero'): row(f'init={i}',('--init',i))

json.dump(out,open('ablation_R.json','w'),indent=1)
