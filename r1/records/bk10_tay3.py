"""bk10_tay3.py : rule TAY3, a rigorous theta-space second-order Taylor bound for Phi_4 over a theta-box, built on the
an independent program's own enclosure (indep/oia.py, indep/ocore.py, imported UNMODIFIED) with three changes that make every step
rigorous:
 (1) arcsin: an independent program's V.asin (numpy arcsin padded by 4 ulps, not verified) is replaced at import time by asin_verified:
     a numpy candidate moved outward and then CHECKED with an outward-rounded Taylor enclosure of sin (degree 25, Lagrange
     remainder |y|^27/27!), using monotonicity of sin on [-pi/2, pi/2]; lanes that fail the check get the trivial bounds
     -pi/2 / +pi/2 (rounded outward).
 (2) kappa, K0: the module constants of ocore are replaced by the checked enclosures (bk8_core.consts), the same
     ones bk9_bnb.py uses, so that TAY3 bounds the same function Phi_4 as the other rules.
 (3) the quadratic maximum: instead of a float KKT maximum, the bound is
        max_{|d_j| <= w_j} [Gm.d + 1/2 d^T Mm d]  <=  qc + sum_j max(+-) grad_j(dc) (d_j - dc_j)   (+ s/2 |w|^2)
     where Mm - s I (s = 0 if possible) is certified NEGATIVE DEFINITE in EXACT rational arithmetic (fractions; LDL^T of the
     float matrix), dc is any point of the box (the float KKT maximiser, used only as a choice), and qc, grad are evaluated in
     outward-rounded intervals.  No floating eigenvalue step is used anywhere in TAY3 (eigvalsh only CHOOSES the shift s).
Bound (Taylor with Lagrange remainder along the segment theta0 -> theta, inside the box whose image is certified PD):
     Phi_4(theta) <= Phi_4(theta0) + G.d + 1/2 d^T M(xi) d,   G in [Gm +- Grad],  M(xi) in [Mm +- Mrad] (hull over k^4 sub-boxes)
                  <= base + qmax(Gm, Mm) + Grad.w + 1/2 w^T Mrad w,
     base = upper end of the checked phi_point at c_dn = lower ends of the outward enclosure of C(theta0), certified PD,
     valid by lem:G(b) (c_dn <= C(theta0) entrywise, both PSD).  Closed if the total (rounded up) is <= Mt_lo."""
import os, sys, math, itertools
from fractions import Fraction as Fr
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'indep'))
import oia
from oia import V

# ---------------------------------------------------------------- (1) verified arcsin
def _enc_frac(q):
    f = float(q)
    lo = f if Fr(f) <= q else float(np.nextafter(f, -np.inf))
    hi = f if Fr(f) >= q else float(np.nextafter(f, np.inf))
    return V(lo, hi)
_INV_FACT = [_enc_frac(Fr(1, math.factorial(2*n + 1))) for n in range(13)]      # 1/(2n+1)!, n = 0..12
_REM27 = float(_enc_frac(Fr(1, math.factorial(27))).hi)
_HALFPI_HI = float(np.nextafter(np.nextafter(math.pi/2, np.inf), np.inf))      # > pi/2 (math.pi/2 is within 1 ulp)


def sin_enc(y):
    """outward enclosure of sin(y) for a float array y (any real y): Taylor polynomial of degree 25 in interval Horner form
    plus the Lagrange remainder |y|^27/27!."""
    y = np.asarray(y, float); Y = V(y); Z = Y*Y
    acc = _INV_FACT[12]
    for n in range(11, -1, -1):
        acc = _INV_FACT[n] - Z*acc
    s = Y*acc
    p = V(np.abs(y)); r = V(np.ones_like(y))
    for _ in range(27): r = r*p
    R = (r*_REM27).hi
    return V(np.nextafter(s.lo - R, -np.inf), np.nextafter(s.hi + R, np.inf))


def _asin_lo(x):
    """a <= arcsin(x) for every lane (x already clipped to [-1, 1])."""
    a0 = np.arcsin(x); out = np.full(x.shape, -_HALFPI_HI); done = x <= -1.0
    for k in range(10):
        a = a0 - 2.0**-50*16.0**k*(1.0 + np.abs(a0))
        ok = (~done) & ((a <= -_HALFPI_HI) | (sin_enc(a).hi <= x))
        out = np.where(ok, np.maximum(a, -_HALFPI_HI), out); done = done | ok
        if done.all(): break
    return out


def _asin_hi(x):
    a0 = np.arcsin(x); out = np.full(x.shape, _HALFPI_HI); done = x >= 1.0
    for k in range(10):
        a = a0 + 2.0**-50*16.0**k*(1.0 + np.abs(a0))
        ok = (~done) & ((a >= _HALFPI_HI) | (sin_enc(a).lo >= x))
        out = np.where(ok, np.minimum(a, _HALFPI_HI), out); done = done | ok
        if done.all(): break
    return out


def asin_verified(s):
    """enclosure of arcsin over the interval array s (true argument in [-1, 1], so clipping is valid; arcsin increasing).
    Correctness: for a in [-pi/2, pi/2], sin(a) <= x implies a <= arcsin(x) (sin strictly increasing there); a candidate
    a <= -pi/2 is trivially a lower bound.  Symmetrically for the upper end.  NaN lanes stay NaN (rejected downstream)."""
    lo = np.atleast_1d(np.asarray(s.lo, float)); hi = np.atleast_1d(np.asarray(s.hi, float))
    with np.errstate(invalid='ignore'):
        xl = np.clip(np.where(np.isnan(lo), -1.0, lo), -1.0, 1.0); xh = np.clip(np.where(np.isnan(hi), 1.0, hi), -1.0, 1.0)
        L = _asin_lo(xl); H = _asin_hi(xh)
    L = np.where(np.isnan(lo), np.nan, L); H = np.where(np.isnan(hi), np.nan, H)
    return V(L.reshape(np.shape(s.lo)), H.reshape(np.shape(s.hi)))


oia.V.asin = asin_verified                              # patched BEFORE ocore is imported (ocore evaluates at import)

import ocore as OCo
from bk8_core import consts
_K = consts()
OCo.KAP = V(float(_K['kappa'].lo), float(_K['kappa'].hi))
OCo.K0 = V(float(_K['K0'].lo), float(_K['K0'].hi))
MT_LO = _K['Mt_lo']


# ---------------------------------------------------------------- (3) exact negative-definiteness and the quadratic bound
def neg_def_exact(Mf):
    """True iff the float symmetric matrix Mf is negative definite, decided EXACTLY (Gaussian elimination of -Mf in
    rationals: all pivots positive)."""
    n = Mf.shape[0]
    A = [[-Fr(float(Mf[i, j])) for j in range(n)] for i in range(n)]
    for j in range(n):
        if A[j][j] <= 0: return False
        for i in range(j + 1, n):
            f = A[i][j]/A[j][j]
            if f != 0:
                for k in range(j, n): A[i][k] -= f*A[j][k]
    return True


def kkt_argmax(G, M, w):
    """float CHOICE of a maximiser of G.d + 1/2 d^T M d over |d_j| <= w_j (enumeration of faces)."""
    n = len(G); best = -np.inf; arg = np.zeros(n)
    for pat in itertools.product((-1, 0, 1), repeat=n):
        d = np.array([s*w[j] for j, s in enumerate(pat)], float); free = [j for j in range(n) if pat[j] == 0]
        if free:
            fx = [j for j in range(n) if pat[j] != 0]
            A = M[np.ix_(free, free)]; b = -(G[free] + (M[np.ix_(free, fx)] @ d[fx] if fx else 0.0))
            try: z = np.linalg.solve(A, b)
            except np.linalg.LinAlgError: continue
            if not np.all(np.isfinite(z)): continue
            d[free] = np.clip(z, -w[free], w[free])
        v = G @ d + 0.5*d @ M @ d
        if v > best: best = v; arg = d
    return arg


def _s(x): return float(np.ravel(x)[0])


def quad_bound(Gm, Mm, w):
    """rigorous upper bound of max_{|d_j|<=w_j} Gm.d + 1/2 d^T Mm d (Gm, Mm, w float arrays). Returns (bound, s)."""
    s = 0.0
    if neg_def_exact(Mm):
        Mq = Mm
    else:
        lam = np.linalg.eigvalsh(Mm)                      # float, used only to CHOOSE the shift
        s = float(max(lam.max(), 0.0))*(1 + 1e-6) + 1e-9*(1.0 + float(np.abs(lam).max()))
        for _ in range(8):
            if neg_def_exact(Mm - s*np.eye(4)): break
            s = 2.0*s + 1e-9
        else:
            return None, None
        Mq = Mm - s*np.eye(4)                             # float matrix, certified ND; d^T Mm d = d^T Mq d + s|d|^2 exactly
        Mq = np.array(Mq)                                 # (Mm - s I computed in floats; the identity is used with THIS Mq:
        # d^T Mm d - d^T Mq d = sum_i (Mm_ii - Mq_ii) d_i^2, and Mm_ii - Mq_ii is bounded by an interval below)
    dc = kkt_argmax(Gm, Mq, w)
    qc = V(0.0); grad = []
    for j in range(4):
        gj = V(float(Gm[j]))
        for k in range(4): gj = gj + V(float(Mq[j, k]))*float(dc[k])
        grad.append(gj)
        qc = qc + V(float(Gm[j]))*float(dc[j])
        for k in range(4): qc = qc + (V(float(Mq[j, k]))*float(dc[j]))*float(dc[k])*0.5
    tot = qc
    for j in range(4):
        a = grad[j]*(V(float(w[j])) - float(dc[j])); b = grad[j]*(V(-float(w[j])) - float(dc[j]))
        tot = tot + V(max(_s(a.hi), _s(b.hi)))
    if s > 0:
        for i in range(4):                                # + 1/2 sum_i (Mm_ii - Mq_ii) w_i^2, the exact diagonal difference
            dd = V(float(Mm[i, i])) - float(Mq[i, i])
            tot = tot + (V(float(w[i]))*float(w[i]))*V(max(_s(dd.hi), 0.0))*0.5
    return _s(tot.hi), s


# ---------------------------------------------------------------- the Hessian hull and TAY3
def M_hull(lo, hi, k):
    """hull of an independent program's interval Hessian of Phi_4 (J^T H J + sum_p g_p HC_p) over k^4 sub-boxes covering the box;
    None unless every sub-box image is certified PD and every entry is finite."""
    if k > 1: L, H = OCo.split_box(lo, hi, k)
    else: L, H = np.asarray(lo, float).reshape(4, 1), np.asarray(hi, float).reshape(4, 1)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        M, _, ok = OCo.M4_vec(L, H)
    fin = np.asarray(ok, bool).copy()
    for i in range(4):
        for j in range(4): fin &= np.isfinite(M[i][j].lo) & np.isfinite(M[i][j].hi)
    if not np.all(fin): return None
    lo_ = np.array([[np.min(M[i][j].lo) for j in range(4)] for i in range(4)])
    hi_ = np.array([[np.max(M[i][j].hi) for j in range(4)] for i in range(4)])
    lo_ = np.maximum(lo_, lo_.T); hi_ = np.minimum(hi_, hi_.T)      # the true Hessian is symmetric: intersect
    if np.any(lo_ > hi_): return None
    return lo_, hi_


STATS3 = {'calls': 0, 'hull_fail': 0, 'shift': 0, 'closed': 0, 'max_abs_Mm': 0.0}


def tay3(lo, hi, k, base_fn):
    """returns (closed, U). base_fn(th0, need) must return a RIGOROUS upper bound of Phi_4(th0) or None
    (need = Mt_lo - rest is the largest base value that still closes; base_fn may give up early if hopeless)."""
    STATS3['calls'] += 1
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    th0 = 0.5*(lo + hi)
    w = np.maximum(np.atleast_1d((V(hi) - V(th0)).hi), np.atleast_1d((V(th0) - V(lo)).hi))
    w = np.maximum(w, 0.0)                                # |theta_j - th0_j| <= w_j on the box
    Mh = M_hull(lo, hi, k)
    if Mh is None: STATS3['hull_fail'] += 1; return False, None
    Mlo, Mhi = Mh
    Mm = 0.5*(Mlo + Mhi); Mm = 0.5*(Mm + Mm.T)
    Mrad = np.maximum((V(Mhi) - V(Mm)).hi, (V(Mm) - V(Mlo)).hi)    # |M_ij - Mm_ij| <= Mrad_ij on the hull
    STATS3['max_abs_Mm'] = max(STATS3['max_abs_Mm'], float(np.abs(Mm).max()))
    t0 = [V(np.array([th0[j]])) for j in range(4)]
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        J = OCo.J4(t0); g = oia.grad_phi(OCo.C4(t0))
    G = []
    for j in range(4):
        acc = V(0.0)
        for p in range(6): acc = acc + J[p][j]*g[p]
        G.append(acc)
    Glo = np.array([_s(x.lo) for x in G]); Ghi = np.array([_s(x.hi) for x in G])
    if not (np.all(np.isfinite(Glo)) and np.all(np.isfinite(Ghi))): return False, None
    Gm = 0.5*(Glo + Ghi); Grad = np.maximum((V(Ghi) - V(Gm)).hi, (V(Gm) - V(Glo)).hi)
    qb, s = quad_bound(Gm, Mm, w)
    if qb is None: return False, None
    if s > 0: STATS3['shift'] += 1
    extra = V(0.0)
    for j in range(4): extra = extra + V(float(Grad[j]))*float(w[j])
    for i in range(4):
        for j in range(4): extra = extra + (V(float(Mrad[i, j]))*float(w[i]))*float(w[j])*0.5
    rest = _s((V(qb) + extra).hi)
    base = base_fn(th0, MT_LO - rest)
    if base is None: return False, None
    U = _s((V(base) + rest).hi)
    if U <= MT_LO: STATS3['closed'] += 1
    return U <= MT_LO, U
