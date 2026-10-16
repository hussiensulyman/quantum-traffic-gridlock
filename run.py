#!/usr/bin/env python
"""Quantum Traffic Gridlock pipeline: python run.py --scenario s.json --seed 0 --out outdir
Scenario-independent: fluid-surrogate QUBO (surr.py) -> Ising -> QAOA (p layers, numpy statevector training,
Aer 4096-shot final sampling) -> qubo.json, circuit.qasm, counts.json, schedule.json."""
import argparse, json, itertools, os, numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, qasm3
from qiskit_aer import AerSimulator
from surr import build_qubo, N
K_OV=1.5; P_DEPTH=3; MAX_EVALS=300; SHOTS=4096

def to_ising(Q,off):
    # x=(1-z)/2 ; returns h (N), J dict, const
    h=np.zeros(N); J={}; c=off
    for i in range(N):
        h[i]+=-Q[i,i]/2; c+=Q[i,i]/2
        for j in range(i+1,N):
            if Q[i,j]!=0:
                J[(i,j)]=Q[i,j]/4; h[i]+=-Q[i,j]/4; h[j]+=-Q[i,j]/4; c+=Q[i,j]/4
    return h,J,c

def basis_costs(Q,off):
    X=np.array(list(itertools.product([0,1],repeat=N)))  # index bit order: q0 is MSB here
    return np.einsum('bi,ij,bj->b',X,Q,X)+off

def apply_rx_all(psi,beta):
    c,s=np.cos(beta),-1j*np.sin(beta)
    psi=psi.reshape([2]*N)

    for ax in range(N):
        a=np.take(psi,0,axis=ax); b=np.take(psi,1,axis=ax)
        psi=np.stack([c*a+s*b, s*a+c*b],axis=ax)

    return psi.reshape(-1)

def qaoa_state(params,diag_n,p):
    g,b=params[:p],params[p:]
    psi=np.full(2**N,1/np.sqrt(2**N),dtype=complex)

    for l in range(p):
        psi=psi*np.exp(-1j*g[l]*diag_n); psi=apply_rx_all(psi,b[l])

    return psi

def build_circuit(h,J,params,p,scale,measure=True):
    g,b=params[:p],params[p:]
    qc=QuantumCircuit(N,N)
    qc.h(range(N))

    for l in range(p):
        for i in range(N):
            if abs(h[i])>1e-12: qc.rz(float(2*g[l]*h[i]/scale),i)

        for (i,j),v in J.items():
            qc.cx(i,j); qc.rz(float(2*g[l]*v/scale),j); qc.cx(i,j)

        for i in range(N): qc.rx(float(2*b[l]),i)
    if measure: qc.measure(range(N),range(N))
    return qc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',required=True)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',required=True)
    ap.add_argument('--init',default='ramp',choices=['ramp','random','zero']); ap.add_argument('--depth',type=int,default=P_DEPTH); ap.add_argument('--k_ov',type=float,default=K_OV)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    s=json.load(open(a.scenario)); p=a.depth
    Q,off=build_qubo(s,k_ov=a.k_ov)
    h,J,c0=to_ising(Q,off)
    vals=basis_costs(Q,off)            # diagonal; q0 = leftmost bit of index
    scale=vals.max()-vals.min()        # normalise angle scale (scenario-independent rule)
    dn=(vals-vals.min())/scale
    rng=np.random.default_rng(a.seed)
    # linear-ramp (annealing-inspired) init + small seeded jitter
    ramp=(np.arange(p)+0.5)/p
    x0=np.concatenate([ramp*1.0*np.pi*0.8, (1-ramp)*0.4*np.pi/2*1.0])+rng.normal(0,0.05,2*p)

    if a.init=='random': x0=rng.uniform(0,np.pi,2*p)*np.concatenate([np.ones(p),0.5*np.ones(p)])
    elif a.init=='zero': x0=np.full(2*p,0.1)

    nev=[0]

    def obj(x):
        nev[0]+=1; psi=qaoa_state(x,dn,p); return float(np.real(np.sum(np.abs(psi)**2*dn)))
    res=minimize(obj,x0,method='COBYLA',options={'maxiter':MAX_EVALS-1,'rhobeg':0.3})
    params=res.x
    qc=build_circuit(h,J,params,p,scale)
    # consistency: circuit statevector vs numpy model (little-endian handled below)
    sim=AerSimulator(method='automatic')

    from qiskit import transpile
    tqc=transpile(qc,sim,seed_transpiler=a.seed)
    out=sim.run(tqc,shots=SHOTS,seed_simulator=a.seed).result().get_counts()
    # qiskit bitstring: leftmost char = highest clbit (c7) -> reverse so char i = qubit/bit i
    counts={}

    for k,v in out.items(): counts[k[::-1]]=counts.get(k[::-1],0)+v

    assert sum(counts.values())==SHOTS
    json.dump(dict(sorted(counts.items())),open(f'{a.out}/counts.json','w'),indent=1)
    open(f'{a.out}/circuit.qasm','w').write(qasm3.dumps(build_circuit(h,J,params,p,scale)))
    ranked=sorted(counts.items(),key=lambda kv:-kv[1])
    # most frequent sample (raw QAOA); also best-surrogate among sampled
    best_q=min(counts,key=lambda b:(float(np.array([int(c) for c in b])@Q@np.array([int(c) for c in b])+off),b))
    json.dump({"bit_order":"A slot1..4 then B slot1..4; 1=NS green, 0=EW green; counts.json keys use the same order (char i = bit i)",
      "most_frequent_sample":ranked[0][0],"best_surrogate_among_samples":best_q,
      "top5":[{"bits":b,"count":n} for b,n in ranked[:5]],"bits":ranked[0][0]},open(f'{a.out}/schedule.json','w'),indent=1)
    json.dump({"form":"minimize x^T Q x + offset, x in {0,1}^8 (Q upper-triangular)","Q":Q.tolist(),"offset":off,
      "variable_map":{f"x{i}":("A" if i<4 else "B")+f" slot {i%4+1}: 1=NS,0=EW" for i in range(N)},
      "auxiliary_variables":[],"ising":{"h":h.tolist(),"J":[[i,j,v] for (i,j),v in J.items()],"const":c0},
      "angle_scale":scale,"qaoa":{"p":p,"gamma_beta":params.tolist(),"objective_evals":nev[0],"optimizer":"COBYLA","seed":a.seed},
      "surrogate_params":{"k_ov":a.k_ov}},open(f'{a.out}/qubo.json','w'),indent=1)
    print('evals',nev[0],'E_surr(norm)',res.fun,'top',ranked[:3])

if __name__=='__main__': main()
