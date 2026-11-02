"""bk12_exact.py: (T2) per-type first-order slopes of PB at G_T in 40-digit arithmetic (mpmath); the slopes along the
four tree types (should be 0 to ~1e-24) and the minimum off-face slope.
(T3) LP feasibility: is there a linear functional ell on the other G entries with ell.G(t) >= G_ac(t) for all 123
types and equality on the 4 tree types (a linear upper bound for the distance-3 entry G_ac tight on the tree face)?
Same for a lower bound, and for the alternating sum G_ac + G_bd - G_ad - G_bc (two-sided)."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, numpy as np, mpmath as mp
from scipy.optimize import linprog
from bk11_core import TYPES, NT, TREE_IDX, OFF_IDX, TIDX
mp.mp.dps = 40
sqrt2 = mp.sqrt(2); KAP = mp.mpf('4.478385650326709971336669')   # the 40-digit kappa
RHO = 2 * sqrt2 / 3; MU0 = (1 + 2 * sqrt2) / 4; ST = (2 + sqrt2) / 4; K0 = 2 * KAP * MU0 + 4 * ST


def gen(t): return [mp.mpf(2) ** (-mp.mpf(x) / 2) for x in t]


def Gof(m):
    G = mp.matrix(6, 6)
    for t, w in m:
        g = gen(t)
        for i in range(6):
            for j in range(6): G[i, j] += w * g[i] * g[j]
    return G


def PBofG(G):
    def gh(x, y): return G[x, y] / mp.sqrt(G[x, x] * G[y, y])
    u, v, a, b, c, d = range(6)
    e0 = gh(u, v); eua, eub, evc, evd = gh(u, a), gh(u, b), gh(v, c), gh(v, d)
    hab, hcd, huc, hud, hva, hvb = gh(a, b), gh(c, d), gh(u, c), gh(u, d), gh(v, a), gh(v, b)
    ca, cb, cc, cd = [mp.sqrt((1 + e) / 2) for e in (eua, eub, evc, evd)]
    Cab = (1 + eua + eub + hab) / (4 * ca * cb); Ccd = (1 + evc + evd + hcd) / (4 * cc * cd)
    pu = ca + cb + (e0 + huc) / (2 * cc) + (e0 + hud) / (2 * cd)
    pv = cc + cd + (e0 + hva) / (2 * ca) + (e0 + hvb) / (2 * cb)
    proj2 = (pu * pu + pv * pv - 2 * e0 * pu * pv) / (1 - e0 * e0)
    return (KAP - 1) * (Cab + Ccd) + proj2 / 2 - 2


TREE = [((0, 1, -1, 1, 2, 2), mp.mpf(1) / 3), ((0, 1, 1, -1, 2, 2), mp.mpf(1) / 3), ((0, -1, 1, 1, -2, 0), mp.mpf(1) / 6), ((0, -1, 1, 1, 0, -2), mp.mpf(1) / 6)]
GT = Gof(TREE); base = PBofG(GT)
print('PB(G_T) - K0 = %s' % mp.nstr(base - K0, 10))
h = mp.mpf('1e-12'); slopes = []
for t in TYPES:
    Gp = Gof(TREE + [(t, h)]); Gm = Gof(TREE + [(t, -h)])
    slopes.append(((PBofG(Gp) - PBofG(Gm)) / (2 * h), t))
tree_sl = [mp.nstr(s, 5) for s, t in slopes if TIDX[t] in TREE_IDX]
off = sorted([(s, t) for s, t in slopes if TIDX[t] not in TREE_IDX])
print('slopes along tree types:', tree_sl)
print('min off-face slope: %s at %s; 5th %s at %s' % (mp.nstr(off[0][0], 15), off[0][1], mp.nstr(off[4][0], 15), off[4][1]))
print('max off-face slope: %s at %s' % (mp.nstr(off[-1][0], 12), off[-1][1]))
print('all 119 off-face slopes (sorted, 8 digits):', [mp.nstr(s, 8) for s, t in off])

# (T3)
GEN = np.array([[2.0 ** (-x / 2) for x in t] for t in TYPES])
pairs = [(i, j) for i in range(6) for j in range(i, 6)]
GG = np.array([[GEN[t, i] * GEN[t, j] for (i, j) in pairs] for t in range(NT)])
LET = 'uvabcd'
names = [LET[i] + LET[j] for (i, j) in pairs]


def target(coefs): return sum(c * GG[:, names.index(n)] for n, c in coefs.items())


def feas(tgt, sense, excl):
    cols = [k for k, n in enumerate(names) if n not in excl]
    A = GG[:, cols]
    offl = [t for t in range(NT) if t not in TREE_IDX]
    if sense == 'upper': A_ub = -A[offl]; b_ub = -tgt[offl]
    else: A_ub = A[offl]; b_ub = tgt[offl]
    r = linprog(np.zeros(len(cols)), A_ub=A_ub, b_ub=b_ub, A_eq=A[TREE_IDX], b_eq=tgt[TREE_IDX], bounds=[(None, None)] * len(cols), method='highs')
    return r.status, (r.x if r.status == 0 else None), cols


for nm, coefs in (('G_ac', {'ac': 1}), ('alt = G_ac + G_bd - G_ad - G_bc', {'ac': 1, 'bd': 1, 'ad': -1, 'bc': -1})):
    for sense in ('upper', 'lower'):
        st, x, cols = feas(target(coefs), sense, set(coefs))
        print('(T3) %s bound for %s tight on the tree face, linear in the other entries: %s' % (sense, nm, 'FEASIBLE' if st == 0 else 'INFEASIBLE (status %d)' % st))
        if st == 0:
            print('      ell =', {names[c]: round(v, 5) for c, v in zip(cols, x) if abs(v) > 1e-9})
for nm, coefs in (('G_ac', {'ac': 1}), ('alt', {'ac': 1, 'bd': 1, 'ad': -1, 'bc': -1})):
    for sense in ('upper', 'lower'):
        st, x, cols = feas(target(coefs), sense, {'ac', 'ad', 'bc', 'bd'})
        print('(T3) %s bound for %s using only distance <= 2 entries: %s' % (sense, nm, 'FEASIBLE' if st == 0 else 'INFEASIBLE'))
