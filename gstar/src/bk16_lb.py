"""bk16_lb.py : rigorous box bounds for the CKS* slice LP (generalised Mode 1 of Lemma DUAL).

For a box Z of z = (alpha, lam, d_a, d_b), a level [s_lo, s_hi] of the off-face mass, a cost c(z) (T1 vector) and constant
k(z) (T1), and multipliers (y_a, y_b) in R^2, eta_NS, eta_lo, eta_hi >= 0 (exact doubles), weak duality gives for every
m >= 0 in the slice with nsg.m <= nsh and s_lo <= OFF.m <= s_hi:

  c.m + k >= k + y_a d_a + y_b d_b - eta_NS nsh - eta_hi s_hi + eta_lo s_lo
             + sum_k mu_k min_{j in class k} (c_j - y_a A_aj - y_b A_bj + eta_NS nsg_j + (eta_hi - eta_lo) OFF_j).

Each piece is turned into a concave minorant over Z by the Lemma VERTEX (interval gradient, outward rounding).
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import linprog
from ivarith import IA, dn, up
import bk13_lb as L
from bk13_lb import T1, minorant, NP, KV, AA, AB, CLS
import bk16_model as M

YB = 1e4


class Box:
    def __init__(s, lo, hi, pm=False):
        s.lo = np.asarray(lo, float); s.hi = np.asarray(hi, float); s.zc = 0.5 * (s.lo + s.hi); s.pm = pm
        qf = M.quantities_pm if pm else M.quantities
        s.Qb = qf([T1.var(i, 4, s.lo[i], s.hi[i]) for i in range(4)])
        s.Qp = qf([T1.var(i, 4, s.zc[i], s.zc[i]) for i in range(4)], BC=s.Qb['BC'])
        s.dc = (float(s.fl(s.Qp['da'])), float(s.fl(s.Qp['db'])))

    def fl(s, t):
        """float (midpoint) of a point-evaluated T1/IA."""
        x = t.v if isinstance(t, T1) else t
        return 0.5 * (np.asarray(x.lo, float) + np.asarray(x.hi, float))


def dual_T1(Q, zc, spec):
    """a dual component: spec = v (constant double) or [v, [g0, g1, g2, g3]] (affine in z - zc, exact doubles)."""
    if not isinstance(spec, (list, tuple)): return float(spec)
    v, g = spec
    t = M.tconst(float(v))
    for i in range(4):
        if g[i] != 0.0: t = t + (Q['z'][i] - float(zc[i])) * float(g[i])
    return t


def mode1_vertices(bx, cost_p, k_p, cost_b, k_b, lev, dual):
    """16 vertex lower values of the Mode 1 bound with multipliers dual = (ya, yb, eNS, elo, ehi); each multiplier is a
    constant or an affine function of z (dual_T1).  The eta's must be >= 0 on the whole box (checked here)."""
    slo, shi = lev
    parts = []
    for Q, cost, k in ((bx.Qp, cost_p, k_p), (bx.Qb, cost_b, k_b)):
        ya, yb, eNS, elo, ehi = [dual_T1(Q, bx.zc, d) for d in dual]
        for e in (eNS, elo, ehi):
            elo_ = float(e.v.lo) if isinstance(e, T1) else e
            if not (elo_ >= 0): return np.full(16, -np.inf)
        f0 = k + Q['da'] * ya + Q['db'] * yb - Q['nsh'] * eNS
        f0 = f0 + (elo * slo if isinstance(elo, T1) else IA(elo) * slo) - (ehi * shi if isinstance(ehi, T1) else IA(ehi) * shi)
        cj = cost - (_vt(AA) * ya) - (_vt(AB) * yb) + Q['nsg'] * eNS + _vt(M.OFF) * (ehi - elo if isinstance(ehi, T1) or isinstance(elo, T1) else IA(ehi) - elo)
        parts.append((f0, cj, Q['mu']))
    (f0p, cjp, mup), (f0b, cjb, mub) = parts
    tot = minorant(f0p.v, f0b, bx.lo, bx.hi, bx.zc)
    for kk, idx in enumerate(CLS):
        fc = cjp[idx] * mup[kk]; fb = cjb[idx] * mub[kk]
        Lk = minorant(fc.v, fb, bx.lo, bx.hi, bx.zc)
        tot = dn(tot + Lk.min(axis=1))
    return tot


def _vt(arr):
    return M.tvecconst(np.asarray(arr, float))


def dual_slopes(bx, cost_p, dual_c, lev, tol=1e-9):
    """first-order variation of the optimal dual along z (least squares on the tight dual constraints at the centre)."""
    Q = bx.Qp
    c = bx.fl(cost_p); cg = 0.5 * (np.asarray(cost_p.g.lo) + np.asarray(cost_p.g.hi))       # (4, NP)
    nsg = bx.fl(Q['nsg']); ng = 0.5 * (np.asarray(Q['nsg'].g.lo) + np.asarray(Q['nsg'].g.hi))
    ya, yb, eNS, elo, ehi = dual_c
    # class duals from the tight rows: y_k = min_j (c_j - ya A_aj - yb A_bj + eNS nsg_j + (ehi - elo) OFF_j)
    red = c - ya * AA - yb * AB + eNS * nsg + (ehi - elo) * M.OFF
    tight = []
    for k_, idx in enumerate(CLS):
        mn = red[idx].min(); tight += [(int(j), k_) for j in idx if red[j] - mn <= 1e-9 * (1 + abs(mn))]
    act = [i for i, e in enumerate((eNS, elo, ehi)) if e > 0]
    nv = 3 + 2 + len(act)
    A = np.zeros((len(tight), nv))
    for r, (j, k_) in enumerate(tight):
        A[r, k_] = 1.0; A[r, 3] = AA[j]; A[r, 4] = AB[j]
        for t, i in enumerate(act):
            A[r, 5 + t] = [-nsg[j], M.OFF[j], -M.OFF[j]][i]
    out = np.zeros((5, 4))
    if len(tight) < 1: return out
    for d in range(4):
        rhs = np.array([cg[d, j] + (eNS * ng[d, j] if True else 0.0) for (j, k_) in tight])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        out[0, d], out[1, d] = sol[3], sol[4]
        for t, i in enumerate(act): out[2 + i, d] = sol[5 + t]
    return out


def affine_dual(bx, dual_c, slopes):
    """dual spec with slopes; eta components keep slopes only if they stay >= 0 on the box."""
    spec = []
    w = 0.5 * (bx.hi - bx.lo)
    for i in range(5):
        v = float(dual_c[i]); g = [float(x) for x in slopes[i]]
        if i >= 2:
            if v <= 0: spec.append(0.0); continue
            if v - sum(abs(gg) * ww for gg, ww in zip(g, w)) * 1.01 < 0: spec.append(v); continue
        spec.append([v, g])
    return spec


def float_lp_at(bx, cost_f, lev, mu, da, db):
    """bounded dual LP with the centre's cost and NS row but marginal mu and diagonals (da, db) of another point."""
    Q = bx.Qp
    nsg = bx.fl(Q['nsg']); nsh = float(bx.fl(Q['nsh']))
    slo, shi = lev
    A = np.zeros((NP, 8))
    for j in range(NP):
        A[j, int(KV[j]) + 1] = 1.0; A[j, 3] = AA[j]; A[j, 4] = AB[j]; A[j, 5] = -nsg[j]; A[j, 6] = M.OFF[j]; A[j, 7] = -M.OFF[j]
    obj = -np.array([mu[0], mu[1], mu[2], da, db, -nsh, slo, -shi])
    bnds = [(None, None)] * 3 + [(-YB, YB)] * 2 + [(0, YB)] * 3
    r = linprog(obj, A_ub=A, b_ub=np.asarray(cost_f), bounds=bnds, method='highs')
    if r.status != 0: return None, None
    y = r.x
    return (y[3], y[4], max(y[5], 0.0), max(y[6], 0.0), max(y[7], 0.0)), -r.fun


def float_lp(bx, cost_f, lev, maximize=False):
    """float dual multipliers (ya, yb, eNS, elo, ehi) for min cost.m over the slice-level set at the box centre.
    Solved as the bounded dual LP (always feasible; value large if the primal is infeasible)."""
    Q = bx.Qp
    mu = [bx.fl(x) for x in Q['mu']]; da, db = bx.dc
    nsg = bx.fl(Q['nsg']); nsh = float(bx.fl(Q['nsh']))
    c = -np.asarray(cost_f) if maximize else np.asarray(cost_f)
    slo, shi = lev
    # dual variables: y_cls (3), ya, yb, eNS, elo, ehi ; constraint per profile j:
    #   y_k + ya A_aj + yb A_bj - eNS nsg_j + elo OFF_j - ehi OFF_j <= c_j
    A = np.zeros((NP, 8))
    for j in range(NP):
        A[j, int(KV[j]) + 1] = 1.0; A[j, 3] = AA[j]; A[j, 4] = AB[j]; A[j, 5] = -nsg[j]; A[j, 6] = M.OFF[j]; A[j, 7] = -M.OFF[j]
    obj = -np.array([mu[0], mu[1], mu[2], da, db, -nsh, slo, -shi])
    bnds = [(None, None)] * 3 + [(-YB, YB)] * 2 + [(0, YB)] * 3
    r = linprog(obj, A_ub=A, b_ub=c, bounds=bnds, method='highs')
    if r.status != 0: return None, None
    y = r.x
    return (y[3], y[4], max(y[5], 0.0), max(y[6], 0.0), max(y[7], 0.0)), -r.fun


def float_primal(bx, cost_f, lev, cands=None):
    Q = bx.Qp
    mu = [bx.fl(x) for x in Q['mu']]; b = np.array([mu[0], mu[1], mu[2], bx.dc[0], bx.dc[1]])
    nsg = bx.fl(Q['nsg']); nsh = float(bx.fl(Q['nsh']))
    from bk13_core import A_EQ
    Aub = [nsg, M.OFF, -M.OFF]; bub = [nsh, lev[1], -lev[0]]
    r = linprog(cost_f, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=A_EQ, b_eq=b, bounds=[(0, None)] * NP, method='highs')
    if r.status != 0: return None, None
    return r.fun, r.x


def certified_min(bx, vp, vb, kp, kb, lev, maximize=False, dual=None, slopes=False):
    """certified lower bound (over the box) of v.m + k (or upper bound of v.m + k if maximize) on the slice-level set;
    returns (value, dual). value = +inf-like large if empty."""
    cf = bx.fl(vp)
    if dual is None:
        dual, val = float_lp(bx, cf, lev, maximize=maximize)
        if dual is None: return None, None
        if slopes:
            sl = dual_slopes(bx, -vp if maximize else vp, dual, lev)
            dual = affine_dual(bx, dual, sl)
    if maximize:
        V = mode1_vertices(bx, -vp, -kp, -vb, -kb, lev, dual)
        return -float(V.min()), dual
    V = mode1_vertices(bx, vp, kp, vb, kb, lev, dual)
    return float(V.min()), dual
