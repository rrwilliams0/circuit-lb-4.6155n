"""phifun.py : own rigorous enclosures of Phi_F and of its gradient/Hessian.
Phi_F(C) = P(s1 != s2) + P(s3 != s4) + P(s1 = s2 != s3 = s4),  s_i = 1[Z_i <= 0], Z ~ N(0, C), keys (ua, ub, vc, vd),
entries in the order (ab, cd, ac, ad, bc, bd).
Plackett: dPhi/dr_p = -(w_p - asin(rho_p)/(2 pi))/(pi sqrt(1 - r_p^2)), w = 3/4 (ab, cd), 1/4 (cross), rho_p the partial
correlation of the complementary pair given p (derived by an independent program from the mixed differences of F).
Path C(s) = I + s (C - I) (ALL six entries scaled; an earlier version's F_point scales only the cross entries):
    Phi_F(C) = 9/8 - sum_p w_p asin(r_p)/pi + (1/(2 pi^2)) int_0^1 h(s) ds,  h(s) = sum_p r_p asin(rho_p(s))/sqrt(1 - s^2 r_p^2).
The integral is enclosed piecewise by Taylor series in s with interval coefficients (exact recurrences), N = 8; the
degree-N coefficient is enclosed over the whole piece (series at the interval base point; Lagrange remainder)."""
import numpy as np, math
from ivarith import IA, dn, up, rsum, PI, asin

PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
COMP = {(0, 1): (2, 3), (2, 3): (0, 1), (0, 2): (1, 3), (0, 3): (1, 2), (1, 2): (0, 3), (1, 3): (0, 2)}
PIDX = {}
for _q, (_i, _j) in enumerate(PAIRS): PIDX[_i, _j] = PIDX[_j, _i] = _q
N = 8
W = np.array([0.75, 0.75, 0.25, 0.25, 0.25, 0.25])


def _stack(xs): return IA(np.array([x.lo for x in xs]), np.array([x.hi for x in xs]))


class TS:
    """Taylor series sum_{k<=N} c_k t^k; c = IA arrays of shape (N+1, M) (M lanes)."""
    __slots__ = ('c',)

    def __init__(s, c): s.c = c

    def __add__(s, o):
        if isinstance(o, TS): return TS(s.c + o.c)
        o = IA.of(o); lo = s.c.lo.copy(); hi = s.c.hi.copy(); c0 = s.c[0] + o
        lo[0] = c0.lo; hi[0] = c0.hi; return TS(IA(lo, hi))
    __radd__ = __add__

    def __neg__(s): return TS(-s.c)

    def __sub__(s, o): return s + (-o if isinstance(o, TS) else -IA.of(o))

    def __rsub__(s, o): return (-s) + o

    def scale(s, x):
        x = IA.of(x); return TS(s.c*IA(x.lo[None], x.hi[None]))

    def __mul__(s, o):
        if not isinstance(o, TS): return s.scale(o)
        a, b = s.c, o.c
        P = IA(a.lo[:, None, :], a.hi[:, None, :])*IA(b.lo[None, :, :], b.hi[None, :, :])
        lo = np.empty_like(a.lo); hi = np.empty_like(a.hi)
        for k in range(N + 1):
            j = np.arange(k + 1)
            t = rsum(IA(P.lo[j, k - j], P.hi[j, k - j]), 0); lo[k] = t.lo; hi[k] = t.hi
        return TS(IA(lo, hi))

    def inv(s):
        a = s.c; b = [a[0].ninv()]
        for k in range(1, N + 1):
            acc = a[1]*b[k - 1]
            for j in range(2, k + 1): acc = acc + a[j]*b[k - j]
            b.append(-(acc*b[0]))
        return TS(_stack(b))

    def sqrt(s):
        a = s.c; b0 = a[0].nsqrt(); b = [b0]; den = (b0*2.0).ninv()
        for k in range(1, N + 1):
            acc = a[k]
            for j in range(1, k): acc = acc - b[j]*b[k - j]
            b.append(acc*den)
        return TS(_stack(b))

    def asin(s):
        u = s.c
        w = (1.0 - s*s).sqrt().inv().c
        out = [asin(u[0])]
        for k in range(1, N + 1):
            acc = u[1]*w[k - 1]
            for j in range(2, k + 1): acc = acc + (u[j]*float(j))*w[k - j]
            out.append(acc*(1.0/k) if k in (1, 2, 4, 8) else acc/float(k))
        return TS(_stack(out))


def lanes(r):
    """per-pair constants: for p = (i,j) with complement (k,l): r_ij, r_kl, r_ki, r_li, r_kj, r_lj -> array (6, 6)."""
    R = lambda x, y: r[PIDX[x, y]]
    out = []
    for (i, j) in PAIRS:
        k, l = COMP[i, j]
        out.append((R(i, j), R(k, l), R(k, i), R(l, i), R(k, j), R(l, j)))
    return np.array(out, float).T


def _shift(T, h):
    """polynomial series at the point m -> enclosure of its coefficients at every base m + tau, |tau| <= h (h dyadic)."""
    if h == 0.0: return T
    c = T.c; lo = np.zeros_like(c.lo); hi = np.zeros_like(c.hi)
    for j in range(N + 1):
        acc = IA(c.lo[j], c.hi[j])
        for k in range(j + 1, N + 1):
            e = k - j
            tp = IA(-h**e, h**e) if e % 2 else IA(0.0, h**e)          # exact (h dyadic)
            acc = acc + IA(c.lo[k], c.hi[k])*float(math.comb(k, j))*tp
        lo[j] = acc.lo; hi[j] = acc.hi
    return TS(IA(lo, hi))


def h_series(r, ms, h):
    """Taylor coefficients of h at base points: if h == 0 at the points ms (array, exact doubles); if h > 0 enclosures
    valid for every base point in [m - h, m + h] (polynomial parts expanded exactly at m and shifted, then the
    non-polynomial recurrences in interval arithmetic).  Returns IA (N+1, P)."""
    P = len(ms)
    Q = lanes(r)
    rep = lambda v: IA(np.tile(v, P))                     # lane = piece*6 + pair
    rij, rkl, rki, rli, rkj, rlj = (rep(Q[t]) for t in range(6))
    M = 6*P
    sl = np.zeros((N + 1, M)); sl[0] = np.repeat(ms, 6); sl[1] = 1.0
    s = TS(IA(sl)); s2 = s*s; s3 = s2*s
    om = 1.0 - s2.scale(rij*rij)
    num = (s*om).scale(rkl) - s2.scale(rki*rli + rkj*rlj) + s3.scale(rij*(rki*rlj + rkj*rli))
    Dk = 1.0 - s2.scale(rij*rij + rki*rki + rkj*rkj) + s3.scale(rij*rki*rkj*2.0)
    Dl = 1.0 - s2.scale(rij*rij + rli*rli + rlj*rlj) + s3.scale(rij*rli*rlj*2.0)
    om, num, Dk, Dl = (_shift(T, h) for T in (om, num, Dk, Dl))
    rho = num*(Dk*Dl).sqrt().inv()
    term = (rho.asin()*om.sqrt().inv()).scale(rij)
    c = term.c
    return rsum(IA(c.lo.reshape(N + 1, P, 6), c.hi.reshape(N + 1, P, 6)), 2)


def _piece(Cp, Ci, q, h):
    t = IA(0.0)
    for j in range(0, N, 2):
        t = t + IA(Cp.lo[j, q], Cp.hi[j, q])*(IA(2.0*h**(j + 1))/float(j + 1))
    return t + IA(Ci.lo[N, q], Ci.hi[N, q])*(IA(2.0*h**(N + 1))/float(N + 1))


def _eval(r, ms, hs):
    try:
        assert np.all(hs == hs[0])
        return h_series(r, ms, float(hs[0])), h_series(r, ms, 0.0)
    except (ArithmeticError, ZeroDivisionError):
        return None


def phi_point(c, tol=1e-9, hmin=2.0**-11):
    """Rigorous enclosure (IA scalar) of Phi_F at the double 6-vector c, which must be certified PD by the caller
    (PD of C implies PD of C(s) for s in [0,1]).  None on failure."""
    r = np.asarray(c, float)
    todo = [((k + 0.5)/16.0, 1.0/32.0) for k in range(16)]
    tot = IA(0.0)
    while todo:
        ms = np.array([m for m, h in todo]); hs = np.array([h for m, h in todo])
        res = _eval(r, ms, hs)
        nxt = []
        for q, (m, h) in enumerate(todo):
            Ci, Cp = res if res is not None else (None, None); qq = q
            ok = Ci is not None and bool(np.isfinite(Ci.lo[N, q]) and np.isfinite(Ci.hi[N, q]))
            if ok:
                remw = (Ci.hi[N, qq] - Ci.lo[N, qq])*2*h**(N + 1)/(N + 1)
                ok = bool(np.isfinite(remw) and remw <= tol*2*h and np.all(np.isfinite(Cp.lo[:, qq])) and np.all(np.isfinite(Cp.hi[:, qq])))
            if not ok:
                if h/2 < hmin: return None
                nxt += [(m - h/2, h/2), (m + h/2, h/2)]; continue
            tot = tot + _piece(Cp, Ci, qq, h)
        todo = nxt
    A = asin(IA(r))
    lin = IA(0.0)
    for p in range(6): lin = lin + A[p]*float(W[p])
    return IA(9.0)/8.0 - lin/PI + tot/(PI*PI*2.0)


# ------------------------------------------------------------------ forward AD (value + gradient in the 6 entries)
class AD:
    __slots__ = ('v', 'g')                 # v: IA (L,), g: IA (L, 6)

    def __init__(s, v, g): s.v, s.g = v, g

    def __add__(s, o):
        if isinstance(o, AD): return AD(s.v + o.v, s.g + o.g)
        return AD(s.v + o, s.g)
    __radd__ = __add__

    def __neg__(s): return AD(-s.v, -s.g)

    def __sub__(s, o): return s + (-o if isinstance(o, AD) else -IA.of(o))

    def __rsub__(s, o): return (-s) + o

    def __mul__(s, o):
        if isinstance(o, AD):
            return AD(s.v*o.v, s.g*IA(o.v.lo[:, None], o.v.hi[:, None]) + o.g*IA(s.v.lo[:, None], s.v.hi[:, None]))
        o = IA.of(o)
        oc = IA(o.lo[:, None], o.hi[:, None]) if o.lo.ndim else o
        return AD(s.v*o, s.g*oc)
    __rmul__ = __mul__

    def chain(s, f, df): return AD(f, s.g*IA(df.lo[:, None], df.hi[:, None]))

    def inv(s): q = s.v.inv(); return s.chain(q, -(q*q))

    def sqrt(s): q = s.v.sqrt(); return s.chain(q, (q*2.0).inv())

    def asin(s):
        x = IA(np.clip(s.v.lo, -1, 1), np.clip(s.v.hi, -1, 1))
        return s.chain(asin(x), (1.0 - x*x).sqrt().inv())


def _meet(x, y):
    lo = np.maximum(x.lo, y.lo); hi = np.minimum(x.hi, y.hi)
    if np.any(lo > hi): raise ArithmeticError('empty meet')
    return IA(lo, hi)


def grad_ad(lo, hi):
    """dPhi/dr_p over the box [lo, hi], lanes = the six p.  Returns AD g with g.v the gradient enclosure (6,) and g.g the
    Hessian enclosure (row p = d/dr_q of dPhi/dr_p), valid at every matrix of the box.  Intermediate values (1 - r^2,
    numerator, 3x3 minors, their product, the partial correlation) are met with their mean-value forms
    q(c0) + [grad q over the box].(c - c0) about the float midpoint c0.  Raises if a minor is not certified positive."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    c0 = 0.5*(lo + hi)
    Dl6 = IA(dn(lo - c0), up(hi - c0))                    # c - c0 over the box (c0 a double inside the box)
    def build(L, H, ref):
        def pick(f):
            idx = [PIDX[f(i, j, *COMP[i, j])] for (i, j) in PAIRS]
            return AD(IA(L[idx], H[idx]), IA(np.eye(6)[idx]))
        def mv(x, k):
            if ref is None: return x
            xc = ref[k]
            t = xc.v
            for q in range(6): t = t + x.g[:, q]*Dl6[q]
            return AD(_meet(x.v, t), x.g)
        out = []
        rij = pick(lambda i, j, k, l: (i, j)); rkl = pick(lambda i, j, k, l: (k, l))
        rki = pick(lambda i, j, k, l: (k, i)); rli = pick(lambda i, j, k, l: (l, i))
        rkj = pick(lambda i, j, k, l: (k, j)); rlj = pick(lambda i, j, k, l: (l, j))
        om = mv(1.0 - rij*rij, 0); out.append(om)
        num = mv(rkl*om - (rki*rli + rkj*rlj) + rij*(rki*rlj + rkj*rli), 1); out.append(num)
        Dk = mv(om - (rki*rki + rkj*rkj) + rij*rki*rkj*2.0, 2); out.append(Dk)
        Dl = mv(om - (rli*rli + rlj*rlj) + rij*rli*rlj*2.0, 3); out.append(Dl)
        if np.any(Dk.v.lo <= 0) or np.any(Dl.v.lo <= 0) or np.any(om.v.lo <= 0): raise ArithmeticError('minor')
        DD = mv(Dk*Dl, 4); out.append(DD)
        rho = mv(num*DD.sqrt().inv(), 5); out.append(rho)
        g = (rho.asin()*(PI*2.0).inv() - IA(W))*(om.sqrt()*PI).inv()
        return out, g
    ref, _ = build(c0, c0, None)
    _, g = build(lo, hi, ref)
    return g


def det_minors_pos(C):
    """C: 4x4 nested list of IA scalars. True only if all leading principal minors are certified > 0 (Sylvester)."""
    a = C
    def det3(M): return (M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
                         + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]))
    m2 = a[0][0]*a[1][1] - a[0][1]*a[1][0]
    m3 = det3([row[:3] for row in a[:3]])
    m4 = IA(0.0)
    for col in range(4):
        sub = [[a[i][j] for j in range(4) if j != col] for i in range(1, 4)]
        t = a[0][col]*det3(sub)
        m4 = m4 + t if col % 2 == 0 else m4 - t
    return bool(a[0][0].lo > 0 and m2.lo > 0 and m3.lo > 0 and m4.lo > 0)


def mat(cv, diag):
    C = [[None]*4 for _ in range(4)]
    for i in range(4): C[i][i] = diag
    for q, (i, j) in enumerate(PAIRS): C[i][j] = C[j][i] = cv[q]
    return C


def pd_point(c): return det_minors_pos(mat([IA(float(x)) for x in c], IA(1.0)))


def pd_box(lo, hi):
    """True only if every matrix with entries in the box is PD: lambda_min(C) >= lambda_min(C0) - ||C - C0||_F (Weyl)."""
    c0 = [0.5*(a + b) for a, b in zip(lo, hi)]
    rr = [max(up(b - m), up(m - a)) for a, b, m in zip(lo, hi, c0)]
    s2 = IA(0.0)
    for x in rr: s2 = s2 + IA(x)*IA(x)*2.0
    s = s2.sqrt()
    return det_minors_pos(mat([IA(x) for x in c0], 1.0 - IA(s.hi)))
