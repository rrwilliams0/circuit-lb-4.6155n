"""bk14_supcert.py: CERTIFIED (mpmath interval arithmetic, cover of the box by a 48 x 48 grid) upper bounds of the
profile maxima used by the sup constants of Delta_u, over |log d_a|, |log d_b| <= 0.29 (n_x = d_x^{-1/2})."""
import itertools
from mpmath import iv, mpf, mp
iv.dps = 25
from bk14_sides import PU, KU
S2 = iv.sqrt(2)
def g(t): return {-1: S2, 0: iv.mpf(1), 1: 1 / S2}[t]
def W(p, q): return p[0] * q[1] - p[1] * q[0]
LB = mpf('0.29'); NG = 48
I1 = [i for i, p in enumerate(PU) if p[0] == 1]; IM = [i for i, p in enumerate(PU) if p[0] == -1]; I0 = [i for i, p in enumerate(PU) if p[0] == 0]
IPI0 = PU.index((-1, 1, 1))
best = dict(a_m1=0, a_1=0, a2_0=0, phm1=0, ph1=0, wPh=0, wPhP0=0, w11=0, w10=0, pq=0)
edges = [-LB + 2 * LB * k / NG for k in range(NG + 1)]
def hi(x): return x.b
for i, j in itertools.product(range(NG), repeat=2):
    la = iv.mpf([edges[i], edges[i + 1]]); lb = iv.mpf([edges[j], edges[j + 1]])
    na, nb = iv.exp(-la / 2), iv.exp(-lb / 2)
    P = [(1 + na * g(p[1]), 1 + nb * g(p[2])) for p in PU]; P0 = P[IPI0]
    a = [W(P0, q) for q in P]
    ph = [(q[0] + q[1]) / (P0[0] + P0[1]) for q in P]
    Ph = [(q[0] - P0[0], q[1] - P0[1]) for q in P]
    absmax = lambda xs: max(max(abs(x.a), abs(x.b)) for x in xs)
    best['a_m1'] = max(best['a_m1'], absmax([a[k] for k in IM]))
    best['a_1'] = max(best['a_1'], absmax([a[k] for k in I1]))
    best['a2_0'] = max(best['a2_0'], absmax([a[k] for k in I0]) ** 2)
    best['phm1'] = max(best['phm1'], absmax([ph[k] - 1 for k in IM]))
    best['ph1'] = max(best['ph1'], absmax([ph[k] for k in I1]))
    best['wPh'] = max(best['wPh'], absmax([W(Ph[k], Ph[l]) for k in IM for l in IM]))
    best['wPhP0'] = max(best['wPhP0'], absmax([W(Ph[k], P[l]) for k in IM for l in I0]))
    best['w11'] = max(best['w11'], absmax([W(P[k], P[l]) for k in I1 for l in I1]))
    best['w10'] = max(best['w10'], absmax([W(P[k], P[l]) for k in I1 for l in I0]))
    best['pq'] = max(best['pq'], absmax([W(P[k], Ph[l]) for k in I1 for l in IM]))
for k, v in best.items(): print('%s <= %s' % (k, mp.nstr(v, 10)))
