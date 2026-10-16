import json, itertools, random, numpy as np
from scorer import evaluate

R=json.load(open('scenario_R.json'))
ALL=list(itertools.product([0,1],repeat=8))

def rand_scn(rng):
    r=lambda:[rng.randint(0,8) for _ in range(4)]

    return {"name":"r","arrivals":{"A":{"NS":r(),"EW":r()},"B":{"NS":r(),"EW":r()}},
     "service_capacity":rng.randint(4,6),"storage_capacity":rng.randint(5,8),
     "overflow_weight":rng.randint(2,6),"switch_weight":rng.choice([.25,.5,.75,1,1.25,1.5]),
     "link":{"from":"A","to":"B","phase":rng.choice(["NS","EW"]),"fraction":rng.choice([.4,.5,.6,.7,.8])}}

def costs(s): return np.array([evaluate(s,list(b))['cost'] for b in ALL])

if __name__=="__main__":
    c=costs(R); print(c.min(), [ ''.join(map(str,ALL[i])) for i in np.argsort(c)[:5]], c.mean())
