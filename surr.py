import numpy as np, itertools
N=8

def aff(c=0.0):
    v=np.zeros(N+1); v[0]=c; return v

def xb(i):
    v=aff(); v[1+i]=1; return v

def build_qubo(s, k_ov=0.5, clip=True):
    """Return (Q upper-triangular NxN, offset). cost ~ x^T Q x + offset, x in {0,1}^8.
    Rule: fluid queues, affine in bits; service = green * effective capacity;
    delay exact-linear; overflow hinge -> quadratic proxy (k_ov*w/S)*q^2; switches exact."""
    c=s['service_capacity']; S=s['storage_capacity']; w=s['overflow_weight']; ws=s['switch_weight']
    lk=s['link']; arr=s['arrivals']
    L=np.zeros(N+1); M=np.zeros((N+1,N+1))

    def g(n,p,t):
        b=xb((0 if n=='A' else 4)+t)
        return b if p=='NS' else aff(1)-b
    q={(n,p):aff() for n in 'AB' for p in ('NS','EW')}
    ref={k:0.0 for k in q}            # half-duty reference fluid, used only to cap service
    pend=[aff() for _ in range(5)]; pend_ref=[0.0]*5

    for t in range(4):
        for n in 'AB':
            for p in ('NS','EW'):
                a=arr[n][p][t]; inflow=aff(a); rin=float(a)
                if n=='B' and p==lk['phase']:
                    inflow=inflow+pend[t]; rin+=pend_ref[t]
                ceff=min(c,ref[(n,p)]+rin) if clip else c
                dep=g(n,p,t)*ceff
                q[(n,p)]=q[(n,p)]+inflow-dep
                L+=q[(n,p)]                                   # delay
                M+=(w*k_ov/S)*np.outer(q[(n,p)],q[(n,p)])     # overflow proxy
                rdep=min(c,ref[(n,p)]+rin)*0.5
                ref[(n,p)]=ref[(n,p)]+rin-rdep
                if n=='A' and p==lk['phase'] and t<3:
                    pend[t+1]=dep*lk['fraction']; pend_ref[t+1]=rdep*2*lk['fraction']*0.5*2/2
    # switches: x_t + x_{t-1} - 2 x_t x_{t-1}
    for off in (0,4):
        for t in range(1,4):
            i,j=off+t,off+t-1
            L[1+i]+=ws; L[1+j]+=ws; M[1+i,1+j]-=2*ws
    Q=np.zeros((N,N)); off=L[0]+M[0,0]

    for i in range(N):
        Q[i,i]+=L[1+i]+M[0,1+i]+M[1+i,0]+M[1+i,1+i]
        for j in range(i+1,N):
            Q[i,j]+=M[1+i,1+j]+M[1+j,1+i]
    return Q,off

def eval_qubo(Q,off):
    X=np.array(list(itertools.product([0,1],repeat=N)))
    return np.einsum('bi,ij,bj->b',X,Q,X)+off
