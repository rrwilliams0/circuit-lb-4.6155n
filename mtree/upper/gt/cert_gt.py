"""cert.py: certified branch and bound for Lemma cert, f <= x9 on the normalised cone.
Usage: python cert.py RUN_DIR.  Writes RUN_DIR/log.jsonl (one line per box) and
RUN_DIR/summary.json, and resumes from RUN_DIR/ckpt.json.  Floating point (HiGHS,
SLSQP, eigh) only proposes multipliers, points and coefficients; every inequality
that closes a box is checked in interval arithmetic."""
import os, sys, json, math, time, functools, itertools
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import linprog, minimize
from mpmath import iv
from math import inf, sqrt

# ------------------------------------------------------------- interval arithmetic
def dn(x): return math.nextafter(x, -inf)
def up(x): return math.nextafter(x, inf)

class I:
    """Interval [lo,hi] of doubles: each operation is rounded to nearest (IEEE 754) and
    each endpoint is then moved one ulp outward, so the exact result is enclosed."""
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None):
        hi = lo if hi is None else hi
        if not lo <= hi: raise ArithmeticError('empty or NaN interval')
        s.lo, s.hi = lo, hi
    def __add__(s, o): o = cv(o); return I(dn(s.lo + o.lo), up(s.hi + o.hi))
    def __sub__(s, o): o = cv(o); return I(dn(s.lo - o.hi), up(s.hi - o.lo))
    def __rsub__(s, o): return cv(o) - s
    def __neg__(s): return I(-s.hi, -s.lo)
    def __mul__(s, o):
        o = cv(o); p = (s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi)
        return I(dn(min(p)), up(max(p)))
    def __truediv__(s, o):
        o = cv(o)
        if o.lo <= 0 <= o.hi: raise ZeroDivisionError
        p = (s.lo/o.lo, s.lo/o.hi, s.hi/o.lo, s.hi/o.hi)
        return I(dn(min(p)), up(max(p)))
    def __rtruediv__(s, o): return cv(o)/s
    __radd__, __rmul__ = __add__, __mul__
    def sqrt(s):
        if s.lo < 0: raise ArithmeticError('sqrt of negative')
        return I(max(0.0, dn(sqrt(s.lo))), up(sqrt(s.hi)))
    def mid(s): return 0.5*(s.lo + s.hi)
    def mag(s): return max(-s.lo, s.hi)              # max |x| on the interval

def cv(x): return x if isinstance(x, I) else I(float(x))
def isum(xs): return sum(xs, I(0.0))
def meet(x, y):
    return I(max(x.lo, y.lo), min(x.hi, y.hi)) if x.lo <= y.hi and y.lo <= x.hi else x
iv.prec = 80

@functools.lru_cache(maxsize=1 << 17)
def asin_b(x):
    """Doubles lo <= arcsin x <= hi for a double x in [-1,1], verified with the interval
    sine of mpmath (sin is increasing on [-pi/2, pi/2])."""
    if abs(x) >= 1: h = math.copysign(math.pi/2, x); return dn(h), up(h)
    y = math.asin(x); lo, hi = dn(y), up(y)
    while iv.sin(lo).b > x: lo = dn(lo)
    while iv.sin(hi).a < x: hi = up(hi)
    return lo, hi

def iasin(X): return I(asin_b(max(X.lo, -1.0))[0], asin_b(min(X.hi, 1.0))[1])
PI, S2 = I(dn(math.pi), up(math.pi)), I(2.0).sqrt()

# ------------------------------------------------------------- the cone (Section 6)
T18 = [(-1, -1, 2, 2), (-1, -1, 2, 0), (-1, -1, 0, 2), (1, 1, 0, -2), (1, 1, -2, 0),
       (1, 1, -2, -2)] + [(s, -s, u, v) for s in (1, -1) for u, v in
                          ((2, 2), (2, 0), (0, 2), (0, -2), (-2, 0), (-2, -2))]
ENT = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
PAIRS = ENT[4:]                                      # ab, cd, ac, ad, bc, bd
COMP = dict(zip(PAIRS, [(2, 3), (0, 1), (1, 3), (1, 2), (0, 3), (0, 2)]))
def gg(t, x, y):
    """g_t(x) g_t(y) = 2^{-(t_x+t_y)/2}, exact up to the enclosure of sqrt 2."""
    e = -(t[x] + t[y]); return I(2.0**(e//2))*S2 if e % 2 else I(2.0**(e//2))
GG = [[gg(t, x, y) for t in T18] for x, y in ENT]   # entries of g_t g_t^T, 10 x 18
PF = np.array([[g.mid() for g in row] for row in GG])
MMAX = [2.0**t[0] for t in T18]                      # K_aa = 1 gives m_t <= 1/g_t(a)^2
T = float(os.environ.get('GT_T', '0.4632551'))      # target (GT: from environment; default = checked run)
ROOT = ([1.0, 0.125, 0.125, dn(0.8), dn(0.8)] + [0.6285]*4, [4.0, 8.0, 8.0] + [1.0]*6)

# ------------------------------------------------------------- rows and duality
def row(c0, *kv):
    """The row e.K + c0 >= 0, with e given as (entry, coefficient) pairs."""
    e = np.zeros(10)
    for k, v in zip(kv[::2], kv[1::2]): e[k] += v
    return (False, e, c0)

def upper_row(k, t, dlo, dhi):
    """Lemma rows(b): c_i K_ii + c_j K_jj - K_ij >= 0 if r_ij <= t (4c_ic_j >= t^2)."""
    i, j = PAIRS[k]
    ci = up(0.5*t*sqrt((dlo[j] + dhi[j])/(dlo[i] + dhi[i])))
    cj = up((I(t)*I(t)/(4*I(ci))).hi)
    assert ci > 0 and cj > 0 and (4*I(ci)*I(cj)).lo >= (I(t)*I(t)).hi
    return row(0.0, i, ci, j, cj, 4 + k, -1.0)

def lower_rows(k, t, dlo, dhi):
    """Lemma rows(c): K_ij - A K_ii - B K_jj - A0 >= 0 if r_ij >= t.  Planes through
    three corners of the diagonal rectangle, moved down until valid at all corners."""
    i, j = PAIRS[k]
    pts = [(u, v) for u in sorted({dlo[i], dhi[i]}) for v in sorted({dlo[j], dhi[j]})]
    out = []
    for tri in itertools.combinations(pts, min(3, len(pts))):
        M = np.array([[1.0, u, v] for u, v in tri])
        y = [t*sqrt(u*v) for u, v in tri]
        A0, A, B = map(float, np.linalg.lstsq(M, y, rcond=None)[0])
        viol = max((A0 + A*I(u) + B*I(v) - t*(I(u)*I(v)).sqrt()).hi for u, v in pts)
        out.append(row(-dn(A0 - viol) if viol > 0 else -A0, 4 + k, 1.0, i, -A, j, -B))
    return out

def rows(lo, hi):
    """Lemma rows: rows (is_eq, e, c0), meaning e.K + c0 = 0 or >= 0, valid on S_Q."""
    dlo, dhi = [1.0] + lo[:3], [1.0] + hi[:3]
    R = [(True, row(0, 0, 1.0)[1], -1.0), row(-1.0, 1, 1.0), row(0.0, 3, 1.0, 2, -1.0),
         row(-1.0, 2, 1.0, 3, 1.0, 1, -1.0)]
    R += [row(-dlo[x], x, 1.0) for x in (1, 2, 3)]
    R += [row(dhi[x], x, -1.0) for x in (1, 2, 3)]
    for k in range(6):
        R += [upper_row(k, hi[3 + k], dlo, dhi)] + lower_rows(k, lo[3 + k], dlo, dhi)
    return R

def lp(R, q=None):
    """HiGHS on the rows (q = None: phase 1, min z with e.K + c0 + z >= 0).  It only
    proposes multipliers lam (>= 0 on inequalities).  Returns (status, min, lam, m)."""
    A = np.array([e for _, e, _ in R]) @ PF
    c0, eq = np.array([c for _, _, c in R]), np.array([E for E, _, _ in R])
    if q is None:
        A = np.hstack([A, (~eq)[:, None]*1.0])
        obj, bnd = np.r_[np.zeros(18), 1.0], [(0, None)]*18 + [(None, None)]
    else: obj, bnd = np.array([x.mid() for x in q]) @ PF, [(0, None)]*18
    res = linprog(obj, A_ub=-A[~eq], b_ub=c0[~eq], A_eq=A[eq], b_eq=-c0[eq], bounds=bnd,
                  method='highs')
    if res.status != 0: return res.status, None, None, None
    yi, ye = iter(-res.ineqlin.marginals), iter(res.eqlin.marginals)
    lam = [float(next(ye)) if E else max(0.0, float(next(yi))) for E, _, _ in R]
    return 0, res.fun, lam, res.x[:18]

def lagrange(q, R, lam):
    """Lemma dual: L with q.K >= L for every K in C18 with K_aa = 1 satisfying R."""
    c, L = list(q), I(0.0)
    for (E, e, c0), l in zip(R, lam):
        if l == 0: continue
        for k in np.nonzero(e)[0]: c[k] = c[k] - l*I(float(e[k]))
        L = L - l*I(c0)
    for t in range(18):
        rho = isum(c[k]*GG[k][t] for k in range(10))
        if rho.lo < 0: L = L + I(rho.lo)*MMAX[t]
    return L.lo

def infeasible(R):
    """Lemma dual, q = 0 (Farkas): True only if no K in C18, K_aa = 1, satisfies R."""
    st, z, lam, _ = lp(R)
    return st == 0 and z > 0 and lagrange([I(0.0)]*10, R, lam) > 0

def lower_bound(q, R):
    """Lemma dual: lower bound of q.K on S_Q (inf: certified empty; None: failed)."""
    st, _, lam, _ = lp(R, q)
    if st == 2: return inf if infeasible(R) else None
    return lagrange(q, R, lam) if st == 0 else None

def tighten(lo, hi, nbis=5):
    """Lemma tight: a box containing S_Q (None if S_Q is certified empty)."""
    lo, hi = list(lo), list(hi)
    R = rows(lo, hi)
    if infeasible(R): return None
    for x in (1, 2, 3):
        for sg in (1.0, -1.0):
            L = lower_bound([I(0.0)]*x + [I(sg)] + [I(0.0)]*(9 - x), R)
            if L == inf: return None
            if L is not None and sg > 0: lo[x - 1] = max(lo[x - 1], L)
            if L is not None and sg < 0: hi[x - 1] = min(hi[x - 1], -L)
        if lo[x - 1] > hi[x - 1]: return None
    dlo, dhi = [1.0] + lo[:3], [1.0] + hi[:3]
    for k in range(6):
        R, a, b = rows(lo, hi), lo[3 + k], hi[3 + k]
        for _ in range(nbis):                    # r_p > t if {r_p <= t} is infeasible
            t = 0.5*(a + b)
            if infeasible(R + [upper_row(k, t, dlo, dhi)]): a = t
            else: b = t
        lo[3 + k], b = a, hi[3 + k]
        for _ in range(nbis):                    # r_p < t if {r_p >= t} is infeasible
            t = 0.5*(a + b)
            if infeasible(R + lower_rows(k, t, dlo, dhi)): b = t
            else: a = t
        hi[3 + k] = b
    return lo, hi

# ------------------------------------------------------------- partial correlations
class A2:
    """Value, gradient and Hessian (intervals) in the six correlations; forward mode."""
    __slots__ = ('v', 'g', 'h')
    def __init__(s, v, g, h): s.v, s.g, s.h = v, g, h
    def __add__(s, o):
        if not isinstance(o, A2): return A2(s.v + o, s.g, s.h)
        return A2(s.v + o.v, [x + y for x, y in zip(s.g, o.g)],
                  [[x + y for x, y in zip(a, b)] for a, b in zip(s.h, o.h)])
    def __neg__(s): return s*(-1.0)
    def __sub__(s, o): return s + (-o)
    def __rsub__(s, o): return -s + o
    def __mul__(s, o):
        if not isinstance(o, A2):
            o = cv(o)
            return A2(s.v*o, [x*o for x in s.g], [[x*o for x in a] for a in s.h])
        return A2(s.v*o.v, [x*o.v + s.v*y for x, y in zip(s.g, o.g)],
                  [[s.h[i][j]*o.v + s.v*o.h[i][j] + s.g[i]*o.g[j] + s.g[j]*o.g[i]
                    for j in range(6)] for i in range(6)])
    __radd__, __rmul__ = __add__, __mul__
    def rsqrt(s):                                    # x^{-1/2}
        q = 1/s.v.sqrt(); f1, f2 = -0.5*q/s.v, 0.75*q/(s.v*s.v)
        return A2(q, [f1*x for x in s.g],
                  [[f1*s.h[i][j] + f2*s.g[i]*s.g[j] for j in range(6)]
                   for i in range(6)])

def var(x, k): return A2(x, [I(float(q == k)) for q in range(6)], [[I(0.0)]*6]*6)

def tt(fc, fb, D):
    """Taylor forms of the value and gradient over the box D + centre, from the centre
    (fc) and the Hessian over the box (fb), met with the natural enclosures (fb)."""
    sq = [I(0.0, up(max(x.lo*x.lo, x.hi*x.hi))) for x in D]
    v = fc.v + isum(g*d for g, d in zip(fc.g, D)) + 0.5*isum(fb.h[i][j]*(
        sq[i] if i == j else D[i]*D[j]) for i in range(6) for j in range(6))
    g = [fc.g[i] + isum(fb.h[i][j]*D[j] for j in range(6)) for i in range(6)]
    return A2(meet(v, fb.v), [meet(x, y) for x, y in zip(g, fb.g)], fb.h)

def corr(vals):
    """Map (i,j) and (j,i) to the entry of pair ij (vals in PAIRS order)."""
    return {q: v for p, v in zip(PAIRS, vals) for q in (p, p[::-1])}

def pcorr_parts(r, p):
    """Lemma deriv: varrho_p = num / sqrt(D1 D2), D1, D2 the 3x3 principal minors."""
    (i, j), (k, l) = p, COMP[p]
    a, b, c, d, e, f = r[i, j], r[k, l], r[k, i], r[l, i], r[k, j], r[l, j]
    om = 1 - a*a
    return (b*om - (c*d + e*f - a*(c*f + e*d)), om - c*c - e*e + 2*a*c*e,
            om - d*d - f*f + 2*a*d*f)

def fdot(p, rho):
    """Lemma angle: dF/dtheta_p, from an enclosure rho of varrho_p."""
    return (iasin(rho)/(2*PI) - (0.75 if p in PAIRS[:2] else 0.25))/PI

def pcorr_point(r, p):
    """An enclosure of varrho_p at a PD point."""
    n, d1, d2 = pcorr_parts(r, p); return meet(n/(d1*d2).sqrt(), I(-1.0, 1.0))

def pcorr_box(lo, hi):
    """Enclosures of varrho_p and of its gradient over the correlation box (Taylor
    forms); ([-1,1], None) if a minor is not certified positive."""
    c = [0.5*(a + b) for a, b in zip(lo, hi)]
    D = [I((I(a) - I(m)).lo, (I(b) - I(m)).hi) for a, b, m in zip(lo, hi, c)]
    rc = corr([var(I(m), k) for k, m in enumerate(c)])
    rb = corr([var(I(a, b), k) for k, (a, b) in enumerate(zip(lo, hi))])
    out = []
    for p in PAIRS:
        try:
            pc, pb = pcorr_parts(rc, p), pcorr_parts(rb, p)
            nb, d1, d2 = (tt(x, y, D) for x, y in zip(pc, pb))
            qc = pc[1]*pc[2]; qb = tt(qc, d1*d2, D)
            if qb.v.lo <= 0: raise ArithmeticError
            sc = qc.rsqrt(); rho = tt(pc[0]*sc, nb*tt(sc, qb.rsqrt(), D), D)
            out.append((meet(rho.v, I(-1.0, 1.0)), rho.g))
        except (ArithmeticError, ZeroDivisionError): out.append((I(-1.0, 1.0), None))
    return out

def chol_pd(lo, hi):
    """Positive definiteness: True only if C(rho) is PD for all rho in the box.  The
    radius s >= ||R||_F is taken about the float midpoint c itself (Weyl), and an
    interval Cholesky factorisation of C(c) - sI must have positive pivots."""
    c = [0.5*(a + b) for a, b in zip(lo, hi)]
    ss = 0.0
    for a, b, m in zip(lo, hi, c): x = max(up(b - m), up(m - a)); ss = up(ss + up(x*x))
    s, r, L = up(sqrt(up(2*ss))), corr([I(x) for x in c]), [[None]*4 for _ in range(4)]
    for j in range(4):
        d = 1 - I(s) - isum(L[j][k]*L[j][k] for k in range(j))
        if d.lo <= 0: return False
        L[j][j] = d.sqrt()
        for i in range(j + 1, 4):
            L[i][j] = (r[i, j] - isum(L[i][k]*L[j][k] for k in range(j)))/L[j][j]
    return True

# ------------------------------------------------------------- the cost at a point
class TS:
    """Truncated Taylor series sum_{k<=N} c_k t^k with interval coefficients.  The
    recurrences are exact, so a series at an interval base point encloses the Taylor
    coefficients at every point of the interval."""
    N = 8                                            # even (see F_point)
    def __init__(s, c): s.c = c
    def __add__(s, o): o = ts(o); return TS([x + y for x, y in zip(s.c, o.c)])
    def __neg__(s): return TS([-x for x in s.c])
    def __sub__(s, o): return s + (-ts(o))
    def __rsub__(s, o): return ts(o) - s
    def __mul__(s, o):
        if not isinstance(o, TS): o = cv(o); return TS([x*o for x in s.c])
        out, nz = [I(0.0)]*(TS.N + 1), [k for k, y in enumerate(o.c) if y.lo or y.hi]
        for j, x in enumerate(s.c):
            for k in (nz if x.lo or x.hi else []):
                if j + k <= TS.N: out[j + k] = out[j + k] + x*o.c[k]
        return TS(out)
    __radd__, __rmul__ = __add__, __mul__
    def inv(s):
        a, b = s.c, [1/s.c[0]]
        for k in range(1, TS.N + 1):
            b.append(-isum(a[j]*b[k - j] for j in range(1, k + 1))/a[0])
        return TS(b)
    def sqrt(s):
        a, b = s.c, [s.c[0].sqrt()]
        for k in range(1, TS.N + 1):
            b.append((a[k] - isum(b[j]*b[k - j] for j in range(1, k)))/(2*b[0]))
        return TS(b)
    def asin(s):
        u, w = s.c, (1 - s*s).sqrt().inv().c
        return TS([iasin(u[0])] + [isum(j*u[j]*w[k - j] for j in range(1, k + 1))/k
                                   for k in range(1, TS.N + 1)])
def ts(x): return x if isinstance(x, TS) else TS([cv(x)] + [I(0.0)]*TS.N)

def integrand(r, m, h=0.0):
    """Lemma path: the Taylor series in t of g(m + t), where g(s) is the integrand
    sum_x r_x (1/4 - arcsin varrho_x(C(s))/(2 pi)) / sqrt(1 - s^2 r_x^2).  For h > 0 the
    base point is any point of [m-h, m+h]: the polynomial parts (degree <= 3 < N) are
    expanded exactly at m and then shifted by [-h, h]."""
    s = TS([I(m), I(1.0)] + [I(0.0)]*(TS.N - 1))
    rs = corr([ts(x) if k < 2 else s*x for k, x in enumerate(r)])
    Tp = [I(-h**e, h**e) if e % 2 else I(0.0 if e else 1.0, h**e)
          for e in range(TS.N + 1)]
    def sh(P):                                       # h is a power of 2: Tp is exact
        return P if not h else TS([isum(math.comb(k, j)*P.c[k]*Tp[k - j] for k in
                                        range(j, TS.N + 1)) for j in range(TS.N + 1)])
    g = ts(0.0)
    for p, x in zip(PAIRS[2:], r[2:]):
        num, d1, d2 = map(sh, pcorr_parts(rs, p))
        om = sh(1 - rs[p]*rs[p])
        g = g + (0.25 - (num*(d1*d2).sqrt().inv()).asin()*(1/(2*PI)))*om.sqrt().inv()*x
    return g

def F_point(rho, tol=1e-6):
    """Lemma path: an enclosure of F(C(rho)) at a PD point.  On a piece [m-h, m+h] of
    [0,1], the Taylor polynomial at m is integrated exactly and the degree-N term is
    bounded over the piece; a piece whose remainder is wider than 2h tol is halved."""
    r, N = [I(x) for x in rho], TS.N
    sab, scd, tot = iasin(r[0]), iasin(r[1]), I(0.0)
    todo = [((k + 0.5)/8, 1/16) for k in range(8)]
    while todo:
        m, h = todo.pop()
        try:
            rem = integrand(r, m, h).c[N]*(I(2*h**(N + 1))/(N + 1))
            ok = rem.hi - rem.lo <= tol*2*h
        except (ArithmeticError, ZeroDivisionError): ok = False
        if not ok:
            if h < 2**-9: return None
            todo += [(m - h/2, h/2), (m + h/2, h/2)]; continue
        try: c = integrand(r, m).c
        except (ArithmeticError, ZeroDivisionError): return None
        tot = tot + rem + isum(c[j]*I(2*h**(j + 1))/(j + 1) for j in range(0, N, 2))
    return 1 - (sab + scd)/PI + 2*(0.25 + sab/(2*PI))*(0.25 + scd/(2*PI)) - tot/PI

# ------------------------------------------------------------- arcsine sums
def tau(r): return (1 - r*r).sqrt()/(4*r)
def ups(r): return (1 - 2*r*r)/(4*r*(1 - r*r).sqrt())

def psi(K, xi, al, dlo, dhi, sq, asin):
    """Psi of Lemma arcsin and its gradient at K (K_aa = 1), in floats or intervals."""
    v, g = 0.0, [0.0]*10
    for k, (i, j) in enumerate(PAIRS):
        r = K[4 + k]/sq(K[i]*K[j]); w = xi[k]*r/sq(1 - r*r)
        v, g[4 + k] = v + xi[k]*asin(r), g[4 + k] + w/K[4 + k]
        if i: g[i] = g[i] - w/(2*K[i])
        g[j] = g[j] - w/(2*K[j])
    for x in (1, 2, 3):
        v = v + al[x]*(K[x] - dlo[x])*(K[x] - dhi[x])
        g[x] = g[x] + al[x]*(2*K[x] - dlo[x] - dhi[x])
    return v, [0.0] + g[1:]

def arcsine_bound(xi, lo, hi, R):
    """Lemmas abb and arcsin, with P^- = d^-, P^+ = d^+: S <= sum_p xi_p arcsin r_p(K)
    on S_Q; 'empty' if S_Q is certified empty, None on failure."""
    assert min(xi) >= 0
    dlo, dhi, al, ell = [1.0] + lo[:3], [1.0] + hi[:3], [I(0.0)]*4, []
    for k, (i, j) in enumerate(PAIRS):
        ell.append(dn((I(lo[3 + k])*(I(dlo[i])*I(dlo[j])).sqrt()).lo*(1 - 1e-9)))
        rl = (I(ell[k])/(I(dhi[i])*I(dhi[j])).sqrt()).lo
        if not 0 < rl < 1: return None
        if i == 0:
            al[j] = al[j] + xi[k]*I(max(0.0, ups(I(rl)).hi))/(2*I(dlo[j])*I(dlo[j]))
        else:
            for x in (i, j):
                al[x] = al[x] + xi[k]*I(tau(I(rl)).hi)/(I(dlo[x])*I(dlo[x]))
    al = [a.hi for a in al]
    st, _, _, m0 = lp(R, [I(0.0)]*10)
    if st: return ('empty' if infeasible(R) else None) if st == 2 else None
    A, c0 = np.array([e for _, e, _ in R]) @ PF, np.array([c for _, _, c in R])
    eq = np.array([E for E, _, _ in R])
    def f(m):                                        # float Psi(K(m)), K_aa := 1
        K = PF @ m; K[0] = 1.0
        if min(K[1:4]) <= 0 or any(K[4 + k]**2 >= K[i]*K[j]
                                   for k, (i, j) in enumerate(PAIRS)):
            return 1e3, np.zeros(18)
        v, g = psi(K, xi, al, dlo, dhi, sqrt, math.asin)
        return v, PF.T @ np.array(g)
    cands = [m0]
    cons = [{'type': 'ineq', 'fun': lambda m: A[~eq] @ m + c0[~eq]},
            {'type': 'eq', 'fun': lambda m: A[eq] @ m + c0[eq]}]
    try: cands.insert(0, minimize(f, m0, jac=True, method='SLSQP', constraints=cons,
                                  bounds=[(0, None)]*18,
                                  options={'maxiter': 200, 'ftol': 1e-13}).x)
    except Exception: pass
    best = None
    for m in cands:                                  # K* must lie in the domain D
        K = PF @ np.maximum(m, 0); K = K/K[0]
        K = [1.0] + [min(max(K[x], dlo[x]), dhi[x]) for x in (1, 2, 3)] + list(K[4:])
        if any(K[4 + k] < ell[k] for k in range(6)): continue
        try:
            v, q = psi([I(float(x)) for x in K], xi, al, dlo, dhi, I.sqrt, iasin)
            L = lower_bound([cv(x) for x in q], R)
        except (ArithmeticError, ZeroDivisionError): continue
        if L == inf: return 'empty'
        if L is not None:
            S = (v - isum(cv(x)*I(float(y)) for x, y in zip(q, K)) + L).lo
            best = S if best is None else max(best, S)
    return best

# ------------------------------------------------------------- box bounds
def remainder(H, d):
    """Lemma box(c): Rem >= (1/2) max D^T M D over symmetric M in H and |D_p| <= d_p."""
    Hs = [[(H[p][q] + H[q][p])*0.5 for q in range(6)] for p in range(6)]
    lam, V = np.linalg.eigh(np.array([[x.mid() for x in a] for a in Hs]))
    E = [[Hs[p][q] - isum(I(V[p, k])*I(V[q, k])*I(lam[k]) for k in range(6))
          for q in range(6)] for p in range(6)]
    tot = isum(E[p][q].mag()*I(d[p])*I(d[q]) for p in range(6) for q in range(6))
    for k in (k for k in range(6) if lam[k] > 0):
        w = isum(abs(V[p, k])*I(d[p]) for p in range(6))
        tot = tot + w*w*lam[k]
    return (tot*0.5).hi

def taylor_bound(lo, hi, R):
    """Lemma box (b)-(d) about a PD point c of the correlation box (the centre, or the
    correlations of a cone point proposed by the LP).  Returns (value, mode)."""
    rlo, rhi = lo[3:], hi[3:]
    c = [0.5*(a + b) for a, b in zip(rlo, rhi)]
    if not chol_pd(c, c):
        K = PF @ lp(R, [I(0.0)]*10)[3]
        c = [min(max(K[4+k]/sqrt(K[i]*K[j]), a), b) for k, (i, j), a, b in
             zip(range(6), PAIRS, rlo, rhi)]
        if not chol_pd(c, c): return inf, 'fail'
    gc = [fdot(p, pcorr_point(corr([I(x) for x in c]), p)) for p in PAIRS]
    xi = [max(0.0, -g.mid()) for g in gc]
    S = arcsine_bound(xi, lo, hi, R)
    if S == 'empty': return None, 'empty'
    Fc = F_point(c)
    if S is None or Fc is None: return inf, 'fail'
    thc = [iasin(I(x)) for x in c]
    d = [max((iasin(I(b)) - t).hi, (t - iasin(I(a))).hi)
         for a, b, t in zip(rlo, rhi, thc)]
    base, box = Fc + isum(x*t for x, t in zip(xi, thc)) - S, pcorr_box(rlo, rhi)
    G = [(fdot(p, rho) + x).mag() for p, (rho, _), x in zip(PAIRS, box, xi)]
    best = ((base + isum(I(g)*e for g, e in zip(G, d))).hi, 'first')
    if chol_pd(rlo, rhi) and all(gr is not None for _, gr in box):
        try:
            H = [[gr[q]*(1 - I(rlo[q], rhi[q])*I(rlo[q], rhi[q])).sqrt()
                  /(2*PI*PI*(1 - rho*rho).sqrt()) for q in range(6)] for rho, gr in box]
            lin = isum((g + x).mag()*I(e) for g, x, e in zip(gc, xi, d))
            best = min(best, ((base + lin + remainder(H, d)).hi, 'second'))
        except (ArithmeticError, ZeroDivisionError): pass
    return best

def process(lo, hi):
    """One box: tightening (stage 1), monotone corner (2), Taylor bounds (3).  Returns
    the tightened box and (value, mode), with value None for an empty box."""
    tb = tighten(lo, hi)
    if tb is None: return lo, hi, (None, 'empty')
    lo, hi = tb
    F = F_point(lo[3:]) if chol_pd(lo[3:], lo[3:]) else None
    best = (inf, 'fail') if F is None else (F.hi, 'mono')
    if best[0] >= T:
        v = taylor_bound(lo, hi, rows(lo, hi))
        best = v if v[0] is None else min(best, v)
    return lo, hi, best

# ------------------------------------------------------------- the search
def split(lo, hi):
    """Stage 4: the widest scaled coordinate and its (double) midpoint."""
    w = [(hi[k] - lo[k])/(0.08*max(lo[k], 0.3) if k < 3 else 0.02) for k in range(9)]
    k = w.index(max(w))
    return k, 0.5*(lo[k] + hi[k])

def low_priority():
    """Run at BELOW_NORMAL priority (shared machine)."""
    if os.name == 'nt':
        import ctypes
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)

def work(job):
    """Process one box in a worker; any exception just leaves the box open."""
    t0 = time.process_time()
    try: lo, hi, res = process(*job)
    except Exception as e: lo, hi, res = *job, (inf, 'fail:' + type(e).__name__)
    return [float(x) for x in lo], [float(x) for x in hi], res, time.process_time() - t0

def main(run, nproc=int(os.environ.get('GT_NPROC', '3')), chunk=60):
    """Breadth-first search from ROOT; logs every box; checkpoints after every chunk."""
    from multiprocessing import Pool
    low_priority(); os.makedirs(run, exist_ok=True)
    ckf, logf = os.path.join(run, 'ckpt.json'), os.path.join(run, 'log.jsonl')
    st = {'nodes': 0, 'leaves': {}, 'max_leaf': None, 'cpu': 0.0, 'wall': 0.0,
          'next': 1, 'queue': [[0, -1, 0] + ROOT[0] + ROOT[1]]}
    if os.path.exists(ckf):
        st = json.load(open(ckf))
        lines = open(logf).readlines()[:st['nodes']]; open(logf, 'w').writelines(lines)
    log, t0, w0 = open(logf, 'a'), time.time(), st['wall']
    with Pool(nproc, initializer=low_priority) as pool:
        while st['queue']:
            jobs, st['queue'] = st['queue'][:chunk], st['queue'][chunk:]
            res = pool.map(work, [(j[3:12], j[12:]) for j in jobs], chunksize=1)
            for (nid, par, side, *_), (lo, hi, (v, mode), cpu) in zip(jobs, res):
                rec = {'id': nid, 'par': par, 'side': side, 'lo': lo, 'hi': hi,
                       'mode': mode, 'val': None if v in (None, inf) else v}
                st['nodes'] += 1; st['cpu'] += cpu
                if v is None or v < T:
                    st['leaves'][mode] = st['leaves'].get(mode, 0) + 1
                    if v is not None: st['max_leaf'] = max(st['max_leaf'] or v, v)
                else:
                    k, m = split(lo, hi)
                    if not lo[k] < m < hi[k]: sys.exit('box too small')
                    rec['cut'], n = [k, m], st['next']
                    st['queue'] += [[n, nid, 0] + lo + hi[:k] + [m] + hi[k + 1:],
                                    [n + 1, nid, 1] + lo[:k] + [m] + lo[k + 1:] + hi]
                    st['next'] += 2
                log.write(json.dumps(rec) + '\n')
            log.flush(); os.fsync(log.fileno()); st['wall'] = w0 + time.time() - t0
            json.dump(st, open(ckf + '.tmp', 'w')); os.replace(ckf + '.tmp', ckf)
            print(st['nodes'], len(st['queue']), st['leaves'], st['max_leaf'],
                  round(st['wall']), flush=True)
    del st['queue']
    json.dump(st, open(os.path.join(run, 'summary.json'), 'w'), indent=1)
    print('DONE', st)

if __name__ == '__main__': main(sys.argv[1])
