import json,sys,numpy as np
from exp import ALL,costs
from scorer import evaluate
from surr import build_qubo,eval_qubo

def greedy(s):
    c=s['service_capacity'];lk=s['link'];q={(n,p):0 for n in 'AB' for p in('NS','EW')};pend=[0]*5;bits={'A':[],'B':[]}
    for t in range(4):
        for n in 'AB':
            av={}
            for p in('NS','EW'):
                av[p]=q[(n,p)]+s['arrivals'][n][p][t]+(pend[t] if n=='B' and p==lk['phase'] else 0)
            g='NS' if av['NS']>=av['EW'] else 'EW'; bits[n].append(1 if g=='NS' else 0)
            for p in('NS','EW'):
                d=min(av[p],c) if p==g else 0; q[(n,p)]=av[p]-d
                if n=='A' and p==lk['phase']: pend[t+1]=int(lk['fraction']*d+0.5)
    return bits['A']+bits['B']

s=json.load(open(sys.argv[1])); cs=costs(s); f=lambda b:evaluate(s,list(b))['cost']
alt=[[1,0,1,0],[0,1,0,1],[1,1,1,1],[0,0,0,0]]

print('best alternating/constant:',min(f(a+a) for a in alt[:2]),'| enumeration optimum:',cs.min(),'| uniform random mean:',cs.mean())
print('greedy:',f(greedy(s)))

Q,o=build_qubo(s); i=int(np.argmin(eval_qubo(Q,o))); print('QUBO minimiser',ALL[i],'official cost',cs[i],'regret %.2f%%'%(100*(cs[i]-cs.min())/cs.min()))
