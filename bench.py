import json,subprocess,sys,numpy as np,itertools
from exp import *

def bench(scn,seeds=range(5),extra=()):
    s=json.load(open(scn)) if isinstance(scn,str) else scn
    c=costs(s); cs=float(c.min()); cr=float(c.mean()); order=np.lexsort((np.arange(256),c))[:13]
    elite={''.join(map(str,ALL[i])) for i in order}
    res=[]

    for sd in seeds:
        subprocess.run([sys.executable,'run.py','--scenario',scn,'--seed',str(sd),'--out',f'out_{sd}',*extra],check=True,capture_output=True)
        cnt=json.load(open(f'out_{sd}/counts.json'))
        EQ=sum(v*c[int(''.join(k),2)] for k,v in cnt.items())/4096
        en=sum(v for k,v in cnt.items() if k in elite)/4096/(13/256)
        res.append(((cr-EQ)/(cr-cs),en))

    return cs,cr,res

if __name__=="__main__":
    cs,cr,res=bench('scenario_R.json'); print(cs,cr); [print('lift %.3f enrich %.2f'%r) for r in res]
