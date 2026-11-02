"""an earlier version library: 123-type cone, C(m), float Phi_F (numerical only). Own enumeration."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import itertools, math
import numpy as np
# vertex order u, v, a, b, c, d ; edges uv, ua, ub, vc, vd
VX = 'uvabcd'
EDG = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
def enum_types():
    out = []
    for t in itertools.product(range(-2, 3), repeat=5):
        t = (0,) + t
        if any(abs(t[x] - t[y]) > 1 for x, y in EDG): continue
        # generic source z not in {u, v}
        if min(t[1], t[2], t[3]) == -1 and min(t[0], t[4], t[5]) == t[1] - 1:
            out.append(('gen', t))
    for tc in (1, 2):
        for td in (1, 2): out.append(('z=u', (0, 1, 1, 1, tc, td)))
    for ta in (0, 1):
        for tb in (0, 1): out.append(('z=v', (0, -1, ta, tb, 0, 0)))
    return out
TY = enum_types()
TYPES = [t for _, t in TY]
assert len(set(TYPES)) == len(TYPES)
NT = len(TYPES)
GEN = np.array([[2.0 ** (-x / 2) for x in t] for t in TYPES])
B = np.zeros((4, 6))
for k, (x, y) in enumerate([(0, 2), (0, 3), (1, 4), (1, 5)]): B[k, x] = B[k, y] = 1
TREE = {(0, 1, -1, 1, 2, 2): 1 / 3, (0, 1, 1, -1, 2, 2): 1 / 3, (0, -1, 1, 1, -2, 0): 1 / 6, (0, -1, 1, 1, 0, -2): 1 / 6}
def mtree():
    m = np.zeros(NT)
    for t, w in TREE.items(): m[TYPES.index(t)] = w
    return m
def corr(K):
    d = np.sqrt(np.diag(K)); return K / np.outer(d, d)
def Ghat(m): return corr((GEN.T * m) @ GEN)
def Cmat(m): return corr(B @ Ghat(m) @ B.T)
# pairs of the 4x4: order ab, cd, ac, ad, bc, bd  (a=ua, b=ub, c=vc, d=vd)
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
def cvec(C): return np.array([C[p] for p in PAIRS])
# --- Phi_F by Monte-Carlo-free exact formula: E F = P(a!=b) + P(c!=d) + P(a=b!=c=d)
# Orthant probabilities via Plackett path (float, Gauss-Legendre), as in GT's m123lib (re-derived below).
from scipy.stats import multivariate_normal
def phi_mvn(r):
    """E F via scipy MVN CDF (independent float evaluator)."""
    C = np.eye(4)
    for (i, j), x in zip(PAIRS, r): C[i, j] = C[j, i] = x
    mv = multivariate_normal(mean=np.zeros(4), cov=C, allow_singular=True)
    # P(a=b!=c=d) = P(a,b<=0, c,d>0) + P(a,b>0, c,d<=0); P(X1<=0,X2<=0,X3>0,X4>0)=P(X1<=0,X2<=0,-X3<=0,-X4<=0)
    def orth(sg):
        S = np.diag(sg); M = S @ C @ S
        return multivariate_normal.cdf(np.zeros(4), mean=np.zeros(4), cov=M, allow_singular=True, abseps=1e-10, releps=1e-10, maxpts=4000000)
    p3 = orth([1, 1, -1, -1]) + orth([-1, -1, 1, 1])
    ab, cd = math.acos(r[0]) / math.pi, math.acos(r[1]) / math.pi
    return ab + cd + p3
XS, XW = np.polynomial.legendre.leggauss(64); XS = 0.5 * (XS + 1); XW = 0.5 * XW
COMP = dict(zip(PAIRS, [(2, 3), (0, 1), (1, 3), (1, 2), (0, 3), (0, 2)]))
def phiF(r):
    """Plackett path: cross correlations scaled by s from 0 to 1 (Gauss-Legendre)."""
    rr = dict(zip(PAIRS, r))
    R = {}
    for p in PAIRS:
        R[p] = R[p[::-1]] = (np.full_like(XS, rr[p]) if p in PAIRS[:2] else XS * rr[p])
    tot = 0.0
    for p in PAIRS[2:]:
        (i, j), (k, l) = p, COMP[p]
        a, b, c, d, e, f = R[i, j], R[k, l], R[k, i], R[l, i], R[k, j], R[l, j]
        om = 1 - a * a
        num = b * om - (c * d + e * f - a * (c * f + e * d))
        d1 = om - c * c - e * e + 2 * a * c * e; d2 = om - d * d - f * f + 2 * a * d * f
        rho = np.clip(num / np.sqrt(d1 * d2), -1, 1)
        g = (0.25 - np.arcsin(rho) / (2 * math.pi)) / np.sqrt(om) * rr[p]
        tot += float(XW @ g)
    sab, scd = math.asin(rr[0, 1]), math.asin(rr[2, 3])
    return 1 - (sab + scd) / math.pi + 2 * (0.25 + sab / (2 * math.pi)) * (0.25 + scd / (2 * math.pi)) - tot / math.pi
def f(m): return phiF(cvec(Cmat(m)))
