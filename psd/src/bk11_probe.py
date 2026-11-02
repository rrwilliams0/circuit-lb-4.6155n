"""bk11_probe.py: NUMERICAL probes.
(A) the two-dimensional projection bound: a.C >= (kappa-1)(k_ab + k_cd) + |proj_{span(a_u, a_v)} Y|^2/2 - 2 =: PB(m);
    is PB(m) >= K0 globally?  and PB - K0 - delta >= 0?
(A2) on {RB >= M_tree}: min lambda_min(C), max delta, max delta/lambda_min.
(B) the rule-ceiling region {d : R(d) >= M_tree}: radius along coordinate directions in log d.
(E) which generators lie in the span of the four tree generators (fibre of G_T)."""
import math, numpy as np
from bk11_core import *
from bk11_check import RB, key_corrs

rng = np.random.default_rng(5)
MTREE = 0.27658360688237347
RHO = 2 * math.sqrt(2) / 3


def PB(m):
    R = (np.sqrt(m)[:, None] * GEN).T; A = R / np.linalg.norm(R, axis=1, keepdims=True)
    ys = keys(m); Y = ys[0] + ys[1] + ys[2] + ys[3]
    au, av = A[0], A[1]; e0 = au @ av
    pu, pv = Y @ au, Y @ av
    proj2 = (pu * pu + pv * pv - 2 * e0 * pu * pv) / (1 - e0 * e0)
    c = Cv(m)
    return (KAPPA - 1) * (c[0] + c[1]) + proj2 / 2 - 2, Y @ Y - proj2


def sample(kind):
    if kind == 0: return rng.dirichlet(0.3 * np.ones(NT))
    if kind == 1:
        m = np.zeros(NT); k = rng.integers(1, 6); idx = rng.choice(NT, size=k, replace=False); m[idx] = rng.dirichlet(np.ones(k)); return m
    if kind == 2:
        m = mtree(); v = rng.normal(size=4); v -= v.mean(); m[TREE_IDX] += 10 ** rng.uniform(-3, -0.6) * v / np.abs(v).sum(); return np.maximum(m, 0)
    if kind == 3:
        m = mtree() * rng.uniform(0.3, 1); k = rng.integers(1, 4); idx = rng.choice(OFF_IDX, size=k, replace=False); m[idx] += rng.dirichlet(np.ones(k)) * 10 ** rng.uniform(-3, 0); return m
    t = rng.choice(OFF_IDX); l = 10 ** rng.uniform(-3, 0); m = (1 - l) * mtree(); m[t] += l; return m


if __name__ == '__main__':
    print('(A) projection bound PB(m) - K0 and PB - K0 - delta over samples')
    worst = (np.inf, None); worst2 = (np.inf, None); nneg = 0; nneg2 = 0; N = 4000
    for i in range(N):
        m = sample(i % 5); m /= m.sum()
        pb, lossY = PB(m); c = Cv(m); ck, d = kill(c)
        val = pb - K0; val2 = val - d
        if val < -1e-12: nneg += 1
        if val2 < -1e-12: nneg2 += 1
        if val < worst[0]: worst = (val, i % 5, offmass(m), imbalance(m), AVEC @ c - K0, lossY)
        if val2 < worst2[0]: worst2 = (val2, i % 5, offmass(m), imbalance(m), AVEC @ c - K0 - d, lossY)
    print('   PB - K0 < 0 at %d of %d; worst %s' % (nneg, N, worst))
    print('   PB - K0 - delta < 0 at %d of %d; worst %s' % (nneg2, N, worst2))
    print('(A2) on {RB >= M_tree}: lambda_min(C), delta')
    lmins = []; dels = []; ratio = 0; nrb = 0
    for i in range(6000):
        m = sample(i % 5); m /= m.sum()
        if RB(m) < MTREE: continue
        nrb += 1; c = Cv(m); ck, d = kill(c); lm = lam_min(c); lmins.append(lm); dels.append(d); ratio = max(ratio, d / lm)
    print('   points with RB >= M_tree: %d; min lambda_min(C) %.4f; max delta %.4f; max delta/lambda_min %.4f' % (nrb, min(lmins), max(dels), ratio))
    print('(B) rule ceiling R(d) = M_tree radius along coordinate directions (log d)')
    def mu(e): return ((sum(math.sqrt((1 + x) / 2) for x in e)) ** 2 - 3) / 6
    def Rd(d):      # d = (dv, da, db, dc, dd), du = 1
        dv, da, db, dc, dd = d
        ex = lambda x, y: (RHO / 2) * (math.sqrt(y / x) + math.sqrt(x / y))
        eu = [ex(1, dv), ex(1, da), ex(1, db)]; ev = [ex(dv, 1), ex(dv, dc), ex(dv, dd)]
        return (3 * math.acos(min(1, mu(eu))) + 3 * math.acos(min(1, mu(ev)))) / (2 * math.pi)
    print('   R(1) = %.6f, 2 kappa_E = %.6f' % (Rd([1, 1, 1, 1, 1]), 3 * math.acos((1 + 2 * math.sqrt(2)) / 4) / math.pi))
    for j, name in enumerate(['dv', 'da', 'db', 'dc', 'dd']):
        for sgn in (1, -1):
            lo, hi = 0.0, 2.0
            for _ in range(60):
                mid = (lo + hi) / 2; d = [1.0] * 5; d[j] = math.exp(sgn * mid)
                if Rd(d) >= MTREE: lo = mid
                else: hi = mid
            print('   %s %s: R(d) >= M_tree up to |log d_%s| = %.4f (ratio %.4f)' % (name, '+' if sgn > 0 else '-', name, lo, math.exp(sgn * lo)))
    # a diagonal direction: all five ratios equal
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if Rd([math.exp(mid)] * 5) >= MTREE: lo = mid
        else: hi = mid
    print('   all five equal: |log d| up to %.4f' % lo)
    print('(E) fibre of G_T: generators in the span of the four tree generators')
    T4 = GEN[TREE_IDX]; rank4 = np.linalg.matrix_rank(T4)
    inspan = [TYPES[t] for t in range(NT) if np.linalg.matrix_rank(np.vstack([T4, GEN[t]])) == rank4]
    print('   rank of tree generators', rank4, '; generators in their span:', inspan)
