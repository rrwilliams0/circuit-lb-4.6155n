"""bk19_lb2.py : the generalised Mode 1 (bk16_lb) with an E^lin BAND: the slice-level set

    S_band(z) = { m >= 0 : class sums mu(z), G_aa = d_a, G_bb = d_b, s_lo <= OFF.m <= s_hi,
                  nsg.m <= nsh_c (E^lin <= e_hi),  nsg.m >= nsl (E^lin >= e_lo; absent if e_lo is None) },

where E^lin(m) = (nsc + nsg.m)/3 - mu0 is the Lemma-L lower bound of the mean key excess (the CKS* engine's NS row),
nsh_c = 3 min(c_Theta, mu0 + e_hi) - nsc, nsl = 3 (mu0 + e_lo) - nsc.  Multipliers (ya, yb, eNS, elo, ehi, eL), the last four
>= 0 on the whole box, each a constant or affine in z.  Weak duality (as bk16_lb, one more row):

  c.m + k >= k + ya d_a + yb d_b - eNS nsh_c + eL nsl - ehi s_hi + elo s_lo
             + sum_k mu_k min_{j in class k} (c_j - ya A_aj - yb A_bj + (eNS - eL) nsg_j + (ehi - elo) OFF_j).

Every piece is turned into a concave minorant over the box by the Lemma VERTEX (bk13_lb.minorant), exactly as
bk16_lb.mode1_vertices (Lemma MODE1-A).  LP solutions only produce candidate multipliers.
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import linprog
from ivarith import IA, dn, up
from bk13_lb import T1, minorant, NP, KV, AA, AB, CLS, VMASK, MU0
import bk16_model as M
import bk16_lb as B16

YB = 1e4
OFF = M.OFF


class Box:
    """u-star box (pm coordinates) with an E^lin band [e_lo, e_hi] (e_lo None: no lower row)."""
    def __init__(s, lo, hi, band):
        s.lo = np.asarray(lo, float); s.hi = np.asarray(hi, float); s.zc = 0.5 * (s.lo + s.hi); s.pm = True
        s.Qb = M.quantities_pm([T1.var(i, 4, s.lo[i], s.hi[i]) for i in range(4)])
        s.Qp = M.quantities_pm([T1.var(i, 4, s.zc[i], s.zc[i]) for i in range(4)], BC=s.Qb['BC'])
        s.band = band
        e_lo, e_hi = band
        capc = min(float(M.CTH.hi), float((MU0 + IA(e_hi)).hi))       # upper end: larger set, valid
        for Q in (s.Qb, s.Qp):
            Q['nshc'] = -Q['nsc'] + IA(capc) * 3.0
            Q['nsl'] = (-Q['nsc'] + (MU0 + IA(e_lo)) * 3.0) if e_lo is not None else None
        s.capc = capc
        s.dc = (float(s.fl(s.Qp['da'])), float(s.fl(s.Qp['db'])))
        s._pts = None

    def fl(s, t):
        x = t.v if isinstance(t, T1) else t
        return 0.5 * (np.asarray(x.lo, float) + np.asarray(x.hi, float))


def _lo_of(e):
    return float(e.v.lo) if isinstance(e, T1) else e


def mode1_vertices(bx, cost_p, k_p, cost_b, k_b, lev, dual):
    """16 vertex lower values; dual = (ya, yb, eNS, elo, ehi, eL), each constant or [v, [g0..g3]] (affine in z - zc)."""
    slo, shi = lev
    parts = []
    hasL = bx.Qp['nsl'] is not None
    for Q, cost, k in ((bx.Qp, cost_p, k_p), (bx.Qb, cost_b, k_b)):
        ya, yb, eNS, elo, ehi, eL = [B16.dual_T1(Q, bx.zc, d) for d in dual]
        for e in (eNS, elo, ehi, eL):
            if not (_lo_of(e) >= 0): return np.full(16, -np.inf)
        if not hasL and not (isinstance(eL, float) and eL == 0.0): return np.full(16, -np.inf)
        f0 = k + Q['da'] * ya + Q['db'] * yb - Q['nshc'] * eNS
        if hasL: f0 = f0 + Q['nsl'] * eL
        f0 = f0 + (elo * slo if isinstance(elo, T1) else IA(elo) * slo) - (ehi * shi if isinstance(ehi, T1) else IA(ehi) * shi)
        eN = eNS - eL if (isinstance(eNS, T1) or isinstance(eL, T1)) else IA(eNS) - eL
        eO = ehi - elo if (isinstance(ehi, T1) or isinstance(elo, T1)) else IA(ehi) - elo
        cj = cost - (_vt(AA) * ya) - (_vt(AB) * yb) + Q['nsg'] * eN + _vt(OFF) * eO
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


def _rows(bx):
    Q = bx.Qp
    nsg = bx.fl(Q['nsg']); nsh = float(bx.fl(Q['nshc']))
    nsl = float(bx.fl(Q['nsl'])) if Q['nsl'] is not None else None
    return nsg, nsh, nsl


def float_lp(bx, cost_f, lev, maximize=False, mu=None, da=None, db=None):
    """bounded dual LP at the centre (or at another marginal/diagonals): returns (dual6, value)."""
    Q = bx.Qp
    if mu is None:
        mu = [bx.fl(x) for x in Q['mu']]; da, db = bx.dc
    nsg, nsh, nsl = _rows(bx)
    c = -np.asarray(cost_f) if maximize else np.asarray(cost_f)
    slo, shi = lev
    A = np.zeros((NP, 9))
    for j in range(NP):
        A[j, int(KV[j]) + 1] = 1.0; A[j, 3] = AA[j]; A[j, 4] = AB[j]; A[j, 5] = -nsg[j]; A[j, 6] = OFF[j]; A[j, 7] = -OFF[j]
        A[j, 8] = nsg[j]
    obj = -np.array([mu[0], mu[1], mu[2], da, db, -nsh, slo, -shi, nsl if nsl is not None else 0.0])
    bnds = [(None, None)] * 3 + [(-YB, YB)] * 2 + [(0, YB)] * 3 + [(0, YB) if nsl is not None else (0, 0)]
    r = linprog(obj, A_ub=A, b_ub=c, bounds=bnds, method='highs')
    if r.status != 0: return None, None
    y = r.x
    return (y[3], y[4], max(y[5], 0.0), max(y[6], 0.0), max(y[7], 0.0), max(y[8], 0.0) if nsl is not None else 0.0), -r.fun


def float_primal(bx, cost_f, lev):
    Q = bx.Qp
    mu = [bx.fl(x) for x in Q['mu']]; b = np.array([mu[0], mu[1], mu[2], bx.dc[0], bx.dc[1]])
    nsg, nsh, nsl = _rows(bx)
    from bk13_core import A_EQ
    Aub = [nsg, OFF, -OFF]; bub = [nsh, lev[1], -lev[0]]
    if nsl is not None: Aub.append(-nsg); bub.append(-nsl)
    r = linprog(cost_f, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=A_EQ, b_eq=b, bounds=[(0, None)] * NP, method='highs')
    if r.status != 0: return None, None
    return r.fun, r.x


def dual_slopes(bx, cost_p, dual_c, lev):
    """least-squares slopes of the optimal dual along z (tight rows at the centre); a heuristic for candidates."""
    Q = bx.Qp
    c = bx.fl(cost_p); cg = 0.5 * (np.asarray(cost_p.g.lo) + np.asarray(cost_p.g.hi))
    nsg = bx.fl(Q['nsg']); ng = 0.5 * (np.asarray(Q['nsg'].g.lo) + np.asarray(Q['nsg'].g.hi))
    ya, yb, eNS, elo, ehi, eL = dual_c
    red = c - ya * AA - yb * AB + (eNS - eL) * nsg + (ehi - elo) * OFF
    tight = []
    for k_, idx in enumerate(CLS):
        mn = red[idx].min(); tight += [(int(j), k_) for j in idx if red[j] - mn <= 1e-9 * (1 + abs(mn))]
    act = [i for i, e in enumerate((eNS, elo, ehi, eL)) if e > 0]
    nv = 5 + len(act)
    A = np.zeros((len(tight), nv))
    for r_, (j, k_) in enumerate(tight):
        A[r_, k_] = 1.0; A[r_, 3] = AA[j]; A[r_, 4] = AB[j]
        for t, i in enumerate(act):
            A[r_, 5 + t] = [-nsg[j], OFF[j], -OFF[j], nsg[j]][i]
    out = np.zeros((6, 4))
    if len(tight) < 1: return out
    for d in range(4):
        rhs = np.array([cg[d, j] + (eNS - eL) * ng[d, j] for (j, k_) in tight])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        out[0, d], out[1, d] = sol[3], sol[4]
        for t, i in enumerate(act): out[2 + i, d] = sol[5 + t]
    return out


def affine_dual(bx, dual_c, slopes):
    spec = []
    w = 0.5 * (bx.hi - bx.lo)
    for i in range(6):
        v = float(dual_c[i]); g = [float(x) for x in slopes[i]]
        if i >= 2:
            if v <= 0: spec.append(0.0); continue
            if v - sum(abs(gg) * ww for gg, ww in zip(g, w)) * 1.01 < 0: spec.append(v); continue
        spec.append([v, g])
    return spec


def certified_min(bx, vp, vb, kp, kb, lev, maximize=False, dual=None, slopes=False):
    cf = bx.fl(vp)
    if dual is None:
        dual, val = float_lp(bx, cf, lev, maximize=maximize)
        if dual is None: return None, None
        dual = [float(x) for x in dual]
        if slopes:
            sl = dual_slopes(bx, -vp if maximize else vp, dual, lev)
            dual = affine_dual(bx, dual, sl)
    if maximize:
        V = mode1_vertices(bx, -vp, -kp, -vb, -kb, lev, dual)
        return -float(V.min()), dual
    V = mode1_vertices(bx, vp, kp, vb, kb, lev, dual)
    return float(V.min()), dual


def _mid(x):
    x = x.v if isinstance(x, T1) else x
    return 0.5 * (np.asarray(x.lo, float) + np.asarray(x.hi, float))


def _midg(x):
    return 0.5 * (np.asarray(x.g.lo, float) + np.asarray(x.g.hi, float))


def robust_dual(bx, cp, kp, lev, affine=True, YBb=1e4):
    """the robust affine dual (bk19_fast form) with the band row: maximise the minimum over the centre and the 16
    vertices of the first-order extrapolation of the Mode 1 bound.  Candidate only."""
    Q = bx.Qp; zc = bx.zc
    samples = [np.zeros(4)] + [np.where(v == 1, bx.hi, bx.lo) - zc for v in VMASK]
    S = len(samples)
    c0 = _mid(cp); cg = _midg(cp)
    k0 = float(_mid(kp)); kg = _midg(kp).reshape(4)
    ns0 = _mid(Q['nsg']); nsgg = _midg(Q['nsg'])
    nh0 = float(_mid(Q['nshc'])); nhg = _midg(Q['nshc']).reshape(4)
    hasL = Q['nsl'] is not None
    nl0 = float(_mid(Q['nsl'])) if hasL else 0.0; nlg = _midg(Q['nsl']).reshape(4) if hasL else np.zeros(4)
    mu0 = [float(_mid(m)) for m in Q['mu']]; mug = [_midg(m).reshape(4) for m in Q['mu']]
    da0, db0 = float(_mid(Q['da'])), float(_mid(Q['db'])); dag, dbg = _midg(Q['da']).reshape(4), _midg(Q['db']).reshape(4)
    slo, shi = lev
    ny = 6; nY = 24 if affine else 0
    nv = ny + nY + 1 + 3 * S
    it = ny + nY
    etas = (2, 3, 4, 5) if hasL else (2, 3, 4)
    nrows = S * (1 + NP) + (S - 1) * len(etas)
    A = np.zeros((nrows, nv)); rhs = np.zeros(nrows)
    KV1 = KV.astype(int) + 1
    r = 0
    for s, D in enumerate(samples):
        k_s = k0 + kg @ D
        bt = np.array([da0 + dag @ D, db0 + dbg @ D, -(nh0 + nhg @ D), slo, -shi, (nl0 + nlg @ D)])
        mu_s = [max(mu0[k] + mug[k] @ D, 0.0) for k in range(3)]
        A[r, it] = 1.0
        A[r, :ny] += -bt
        if affine: A[r, ny:ny + nY] += np.outer(-bt, D).ravel()
        for k in range(3): A[r, it + 1 + 3 * s + k] -= mu_s[k]
        rhs[r] = k_s; r += 1
        c_s = c0 + D @ cg; ns_s = ns0 + D @ nsgg
        At = np.stack([AA, AB, -ns_s, OFF, -OFF, ns_s], axis=1)
        A[r:r + NP, :ny] = At
        if affine: A[r:r + NP, ny:ny + nY] = (At[:, :, None] * D[None, None, :]).reshape(NP, nY)
        A[np.arange(r, r + NP), it + 1 + 3 * s + KV1] = 1.0
        rhs[r:r + NP] = c_s; r += NP
        if s > 0:
            for i in etas:
                A[r, i] = -1.0
                if affine: A[r, ny + 4 * i: ny + 4 * i + 4] = -D
                rhs[r] = -1e-9; r += 1
    obj = np.zeros(nv); obj[it] = -1.0
    bnds = [(-YBb, YBb)] * 2 + [(0, YBb)] * 3 + [(0, YBb) if hasL else (0, 0)] + [(-1e6, 1e6)] * nY + [(-10.0, 10.0)] + [(None, None)] * (3 * S)
    if affine and not hasL:
        for q in range(4): bnds[ny + 4 * 5 + q] = (0, 0)
    res = linprog(obj, A_ub=A, b_ub=rhs, bounds=bnds, method='highs')
    if res.status != 0: return None
    x = res.x
    spec = []
    hw = 0.5 * (bx.hi - bx.lo)
    for i in range(6):
        v = float(x[i])
        g = [float(x[ny + 4 * i + q]) for q in range(4)] if affine else [0.0] * 4
        if i >= 2:
            if i == 5 and not hasL: spec.append(0.0); continue
            need = float(sum(abs(gg) * ww for gg, ww in zip(g, hw)))
            if v < need * (1 + 1e-9) + 1e-14: v = need * (1 + 1e-9) + 1e-14
        spec.append([v, g] if any(gg != 0.0 for gg in g) else v)
    return spec
