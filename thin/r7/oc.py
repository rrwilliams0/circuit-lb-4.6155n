"""Independent core (independent of the certificate code): types, C(m), D, kill, RB, keys, witness optimisers."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, itertools
import numpy as np
# vertices u v a b c d -> 0..5 ; internal edges
IE = [(0,1),(0,2),(0,3),(1,4),(1,5)]
def gen_types():
    T = set()
    for tv in (-1,0,1):
        for ta in (-1,0,1):
            for tb in (-1,0,1):
                for tc in range(tv-1,tv+2):
                    for td in range(tv-1,tv+2):
                        t = (0,tv,ta,tb,tc,td)
                        if min(tv,ta,tb) != -1: continue
                        if min(0,tc,td) != tv-1: continue
                        T.add(t)
    for tc in (1,2):
        for td in (1,2): T.add((0,1,1,1,tc,td))
    for ta in (0,1):
        for tb in (0,1): T.add((0,-1,ta,tb,0,0))
    return sorted(T)
TY = gen_types(); assert len(TY) == 123, len(TY)
NT = 123
IX = {t:i for i,t in enumerate(TY)}
GN = np.array([[math.sqrt(2.0)**(-x) for x in t] for t in TY])
tA=(0,1,-1,1,2,2); tB=(0,1,1,-1,2,2); tC=(0,-1,1,1,-2,0); tD=(0,-1,1,1,0,-2)
iA,iB,iC,iD = IX[tA],IX[tB],IX[tC],IX[tD]
FACE=[iA,iB,iC,iD]; OFF=[i for i in range(NT) if i not in FACE]
mT = np.zeros(NT); mT[iA]=mT[iB]=1/3; mT[iC]=mT[iD]=1/6
KAP = 4.478385650326709971336669
P = (1+2*math.sqrt(2))/4; S = (2+math.sqrt(2))/4
a = np.array([KAP,KAP,1,1,1,1.0]); K0 = 2*KAP*P+4*S; CT = np.array([P,P,S,S,S,S])
MTREE = 0.27658360688237347
BE = [(0,2),(0,3),(1,4),(1,5)]   # ua ub vc vd
PR = [(0,1),(2,3),(0,2),(0,3),(1,2),(1,3)]
def G(m): return GN.T @ (m[:,None]*GN)
def Ghat(m):
    g = G(m); d = np.sqrt(np.diag(g)); return g/np.outer(d,d)
def W(m):
    gh = Ghat(m); w = np.zeros((4,4))
    for i,(x,y) in enumerate(BE):
        for j,(z,u) in enumerate(BE):
            w[i,j] = gh[x,z]+gh[x,u]+gh[y,z]+gh[y,u]
    return w
def Cmat(m):
    w = W(m); d = np.sqrt(np.diag(w)); return w/np.outer(d,d)
def c6(m):
    C = Cmat(m); return np.array([C[i,j] for i,j in PR])
def full(c):
    M = np.eye(4)
    for q,(i,j) in enumerate(PR): M[i,j]=M[j,i]=c[q]
    return M
def lmin(c): return np.linalg.eigvalsh(full(c))[0]
def Dx(c): return c[2]*c[5]-c[3]*c[4]
def kill(c):
    c = c.copy(); D = Dx(c)
    if D > 0:
        dl = min(D/c[5], D/c[2])
        if D/c[5] <= D/c[2]: c[2] -= dl
        else: c[5] -= dl
    elif D < 0:
        D = -D; dl = min(D/c[4], D/c[3])
        if D/c[4] <= D/c[3]: c[3] -= dl
        else: c[4] -= dl
    else: dl = 0.0
    return c, dl
def rows(m):
    R = (np.sqrt(m)[:,None]*GN).T; return R/np.linalg.norm(R,axis=1,keepdims=True)
def keyvecs(m):
    A = rows(m); out = []
    for (x,y) in BE+[(0,1)]:
        v = A[x]+A[y]; out.append(v/np.linalg.norm(v))
    return np.array(out)
def RB(m):
    y = keyvecs(m)
    ku = [y[4]@y[0], y[4]@y[1], y[0]@y[1]]; kv = [y[4]@y[2], y[4]@y[3], y[2]@y[3]]
    return sum(math.acos(min(1.0,k)) for k in ku+kv)/(2*math.pi)
def sigma_ab(t): return (t[0],t[1],t[3],t[2],t[4],t[5])
def sigma_cd(t): return (t[0],t[1],t[2],t[3],t[5],t[4])
def sigma_uv(t):
    tv=t[1]; return (0, -tv, t[4]-tv, t[5]-tv, t[2]-tv, t[3]-tv)
