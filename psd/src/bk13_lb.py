"""bk13_lb.py : rigorous lower bounds for the u-star slice LP over 4-dimensional boxes.
Coordinates z = (alpha, lam, sa, sb): marginal mu(alpha, lam) = (alpha(1-lam), lam, 2(1-alpha)(1-lam))/S,
S = (1-lam)(2-alpha) + lam; diagonal ratios d_a = sa^-2, d_b = sb^-2 (G_uu = 1).
LP(z) = const(p) + min { c(z).m : m >= 0, class sums mu, G_aa(m) = d_a, G_bb(m) = d_b }   (Lemma L objective).
Two families of valid lower bounds (weak duality, Section 2 of the report):
  Mode 1 (constant y = (ya, yb)):  LP >= const + ya d_a + yb d_b + sum_k mu_k min_{j in k} (c_j - ya A_aj - yb A_bj)
  Mode 2 (basis B):                LP >= F_B(z) + sum_k mu_k min(0, min_{j in k, j notin B} rc_j(z)),
                                   F_B = const + c_B . B^-1 b(z),  y(z) = B^-T c_B(z),  rc_j = c_j - A_j . y(z).
Each bound is turned into a concave piecewise-linear minorant on the box by the mean-value theorem with an interval
gradient (T1 forward-mode AD on ivarith.IA, outward rounding), so its minimum over the box is attained at a vertex."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, itertools
from fractions import Fraction as Fr
import numpy as np
from ivarith import IA, dn, up, rsum


def I(x):
    """outward enclosure of a real given as a Fraction, int, or decimal string."""
    if isinstance(x, str): x = Fr(x)
    x = Fr(x); f = float(x)
    lo = f if Fr(f) <= x else dn(f)
    hi = f if Fr(f) >= x else up(f)
    return IA(lo, hi)


def isqrt(n):
    """enclosure of sqrt(n) for a positive rational n, verified by squaring exactly."""
    f = math.sqrt(float(n)); lo, hi = dn(dn(f)), up(up(f))
    assert Fr(lo) ** 2 <= Fr(n) <= Fr(hi) ** 2
    return IA(lo, hi)


SQ2 = isqrt(2)
ISQ2 = IA(dn(1 / SQ2.hi), up(1 / SQ2.lo))
assert Fr(float(ISQ2.lo)) ** 2 <= Fr(1, 2) <= Fr(float(ISQ2.hi)) ** 2
RHO = SQ2 * 2.0 / 3.0
MU0 = (SQ2 * 2.0 + 1.0) * 0.25
D0 = RHO * 2.0 + 2.0
KAP = IA(I('4.478385650326169').lo, I('4.478385650327289').hi)    # the certified enclosure of kappa
R = (KAP - 1.0) / (MU0 * 4.0)
K1 = D0.inv() - (D0 * D0).inv() * 4.0
KP = (R + 1.0) * K1                                                   # (1 + r) K1
KC = D0.inv() * 2.0 + RHO * 16.0 / (D0 * D0) + R / D0 + R * RHO * 8.0 / (D0 * D0) - (R + 2.0) * MU0
KE = D0.inv() * 2.0 - (D0 * D0).inv() * 8.0


def profiles():
    out = []
    for k, ta, tb in itertools.product((-1, 0, 1), repeat=3):
        if min(k, ta, tb) == -1: out.append((k, ta, tb))
    out.append((1, 1, 1))
    return out


PU = profiles(); NP = 20
KV = np.array([p[0] for p in PU]); TAv = np.array([p[1] for p in PU]); TBv = np.array([p[2] for p in PU])


def g_of(t):   # enclosure of 2^{-t/2}
    return {-1: SQ2, 0: IA(1.0), 1: ISQ2}[int(t)]


def cat(lst):
    return IA(np.array([float(np.asarray(x.lo)) for x in lst]), np.array([float(np.asarray(x.hi)) for x in lst]))


GA = cat([g_of(t) for t in TAv]); GB = cat([g_of(t) for t in TBv]); GV = cat([g_of(t) for t in KV])
GAGB = cat([(g_of(a) * g_of(b)) if a + b != 0 else IA(1.0) for a, b in zip(TAv, TBv)])
AA = np.array([2.0 ** (-int(t)) for t in TAv]); AB = np.array([2.0 ** (-int(t)) for t in TBv])     # exact doubles
AFR = [[Fr(1) if KV[j] == k else Fr(0) for j in range(NP)] for k in (-1, 0, 1)] + \
      [[Fr(2) ** (-int(t)) for t in TAv], [Fr(2) ** (-int(t)) for t in TBv]]
CLS = [np.where(KV == k)[0] for k in (-1, 0, 1)]


# ------------------------------------------------------------------ first-order interval AD
def _b(x):  # add a leading axis to an IA
    return IA(np.asarray(x.lo)[None], np.asarray(x.hi)[None])


class T1:
    __slots__ = ('v', 'g')

    def __init__(s, v, g): s.v = v; s.g = g

    @staticmethod
    def var(i, n, lo, hi):
        g = np.zeros(n); g[i] = 1.0; return T1(IA(lo, hi), IA(g))

    def _l(s, o):
        if isinstance(o, T1): return o
        o = IA.of(o); return T1(o, IA(np.zeros((s.g.lo.shape[0],) + np.shape(o.lo))))

    def _al(s, o):
        sg, og = s.g, o.g
        ds, do = np.ndim(s.v.lo), np.ndim(o.v.lo)
        if ds < do: sg = IA(sg.lo.reshape(sg.lo.shape + (1,) * (do - ds)), sg.hi.reshape(sg.hi.shape + (1,) * (do - ds)))
        if do < ds: og = IA(og.lo.reshape(og.lo.shape + (1,) * (ds - do)), og.hi.reshape(og.hi.shape + (1,) * (ds - do)))
        return sg, og

    def __add__(s, o):
        o = s._l(o); sg, og = s._al(o); return T1(s.v + o.v, sg + og)
    __radd__ = __add__

    def __neg__(s): return T1(-s.v, -s.g)

    def __sub__(s, o):
        o = s._l(o); sg, og = s._al(o); return T1(s.v - o.v, sg - og)

    def __rsub__(s, o): return s._l(o) - s

    def __mul__(s, o):
        if not isinstance(o, T1):
            o = IA.of(o); return T1(s.v * o, s.g * _b(o))
        sg, og = s.g, o.g
        ds, do = np.ndim(s.v.lo), np.ndim(o.v.lo)
        if ds < do: sg = IA(sg.lo.reshape(sg.lo.shape + (1,) * (do - ds)), sg.hi.reshape(sg.hi.shape + (1,) * (do - ds)))
        if do < ds: og = IA(og.lo.reshape(og.lo.shape + (1,) * (ds - do)), og.hi.reshape(og.hi.shape + (1,) * (ds - do)))
        return T1(s.v * o.v, sg * _b(o.v) + og * _b(s.v))
    __rmul__ = __mul__

    def inv(s):
        xi = s.v.inv(); return T1(xi, -(s.g * _b(xi * xi)))

    def __truediv__(s, o):
        if not isinstance(o, T1): return s * IA.of(o).inv()
        return s * o.inv()

    def sqrt(s):
        r = s.v.sqrt(); return T1(r, s.g * _b(r.inv() * 0.5))

    def __getitem__(s, k): return T1(s.v[k], IA(s.g.lo[:, k], s.g.hi[:, k]))


def tsum(f):
    """sum of a vector T1 over its last axis (rigorous)."""
    return T1(rsum(f.v, 0), rsum(f.g, 1))


def tlin(M, f):
    """M (IA, k x m) applied to the vector T1 f (v shape (m,), g shape (n, m))."""
    v = rsum(M * IA(f.v.lo[None, :], f.v.hi[None, :]), 1)
    g = rsum(IA(M.lo[None], M.hi[None]) * IA(f.g.lo[:, None, :], f.g.hi[:, None, :]), 2)
    return T1(v, g)


def tstack(lst):
    v = IA(np.array([float(np.asarray(x.v.lo)) for x in lst]), np.array([float(np.asarray(x.v.hi)) for x in lst]))
    g = IA(np.stack([np.asarray(x.g.lo).reshape(-1) for x in lst], 1), np.stack([np.asarray(x.g.hi).reshape(-1) for x in lst], 1))
    return T1(v, g)


def tvec(t, arr):
    """scalar T1 times an IA vector -> vector T1."""
    return T1(t.v * arr, IA(t.g.lo[:, None], t.g.hi[:, None]) * IA(arr.lo[None, :], arr.hi[None, :]))


COORD = 'd'     # 'd': variables (alpha, lam, d_a, d_b);  's': (alpha, lam, s_a, s_b)


def evalz(z):
    """shared quantities for z = list of 4 T1 variables (alpha, lam, d_a, d_b) (or (alpha, lam, sa, sb) if COORD == 's')."""
    al, la, xa, xb = z
    if COORD == 'd':
        sa = xa.sqrt().inv(); sb = xb.sqrt().inv()
    else:
        sa, sb = xa, xb
    q = 1.0 - la
    S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    V = m1 * 2.0 + m0 + p1 * 0.5
    U = m1 * SQ2 + m0 + p1 * ISQ2
    w = V.sqrt().inv(); e0 = U * w
    const = e0 * KE + KC
    coefa = tvec(w * D0.inv(), GV) + KP                 # KP + gv w / D0   (vector over profiles)
    c = tvec(sa, GA) * coefa + tvec(sb, GB) * coefa + tvec(sa * sb * (R / D0), GAGB)
    if COORD == 'd': da, db = xa, xb
    else: da = (sa * sa).inv(); db = (sb * sb).inv()
    phi = tstack([sa, sb, w * sa, w * sb, sa * sb])
    return dict(phi=phi, mu=tstack([m1, m0, p1]), const=const, c=c, da=da, db=db, b=tstack([m1, m0, p1, da, db]), w=w, e0=e0)


class Shared:
    """the shared quantities as T1 over the box [lo, hi] and at the point zc."""
    def __init__(s, lo, hi, zc=None):
        s.lo = np.asarray(lo, float); s.hi = np.asarray(hi, float)
        s.zc = 0.5 * (s.lo + s.hi) if zc is None else np.asarray(zc, float)
        s.box = evalz([T1.var(i, 4, s.lo[i], s.hi[i]) for i in range(4)])
        s.pt = evalz([T1.var(i, 4, s.zc[i], s.zc[i]) for i in range(4)])


# ------------------------------------------------------------------ minorants
VMASK = np.array(list(itertools.product((0, 1), repeat=4)))


def vertices(lo, hi):
    return np.array([[hi[i] if VMASK[v, i] else lo[i] for i in range(4)] for v in range(16)])


def minorant(fc, fbox, lo, hi, zc):
    """vertex values (16, ...) of the concave minorant  f(zc) + sum_i min_{g in G_i} g (z_i - zc_i)."""
    flo = np.asarray(fc.lo)
    out = np.broadcast_to(flo, (16,) + flo.shape).copy()
    for i in range(4):
        dlo = IA(dn(lo[i] - zc[i]), up(lo[i] - zc[i])); dhi = IA(dn(hi[i] - zc[i]), up(hi[i] - zc[i]))
        G = IA(fbox.g.lo[i], fbox.g.hi[i])
        tl = np.asarray((G * dlo).lo); th = np.asarray((G * dhi).lo)
        sel = VMASK[:, i].reshape((16,) + (1,) * flo.ndim) == 1
        out = dn(out + np.where(sel, th, tl))
    return out


def mode1(sh, ya, yb):
    """Mode 1 vertex lower values for constant (ya, yb) (doubles, taken exactly)."""
    lo, hi, zc = sh.lo, sh.hi, sh.zc
    f0c = sh.pt['const'] + sh.pt['da'] * ya + sh.pt['db'] * yb
    f0b = sh.box['const'] + sh.box['da'] * ya + sh.box['db'] * yb
    tot = minorant(f0c.v, f0b, lo, hi, zc)
    shift = IA(AA) * ya + IA(AB) * yb
    for k, idx in enumerate(CLS):
        fc = (sh.pt['c'][idx] - shift[idx]) * sh.pt['mu'][k]
        fb = (sh.box['c'][idx] - shift[idx]) * sh.box['mu'][k]
        Lk = minorant(fc.v, fb, lo, hi, zc)
        tot = dn(tot + Lk.min(axis=1))
    return tot


_BINV = {}


def binv(basis):
    """exact rational inverse of the 5 x 5 basis matrix (rows: 3 class sums, G_aa, G_bb), as an IA matrix."""
    basis = tuple(sorted(int(j) for j in basis))
    if basis in _BINV: return _BINV[basis]
    if len(set(basis)) != 5: _BINV[basis] = None; return None
    n = 5
    Aug = [[AFR[r][j] for j in basis] + [Fr(int(i == r)) for i in range(n)] for r in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if Aug[r][col] != 0), None)
        if piv is None: _BINV[basis] = None; return None
        Aug[col], Aug[piv] = Aug[piv], Aug[col]
        pv = Aug[col][col]; Aug[col] = [x / pv for x in Aug[col]]
        for r in range(n):
            if r != col and Aug[r][col] != 0:
                f = Aug[r][col]; Aug[r] = [a - f * b for a, b in zip(Aug[r], Aug[col])]
    Inv = [row[n:] for row in Aug]
    lo = np.array([[float(I(x).lo) for x in row] for row in Inv]); hi = np.array([[float(I(x).hi) for x in row] for row in Inv])
    _BINV[basis] = (basis, IA(lo, hi), Inv)
    return _BINV[basis]


def mode2_parts(E, basis, BI):
    cB = E['c'][np.array(basis)]
    xB = tlin(BI, E['b'])                                  # B^-1 b
    y = tlin(IA(BI.lo.T.copy(), BI.hi.T.copy()), cB)       # B^-T c_B : (y_-1, y_0, y_1, y_a, y_b)
    F = tsum(cB * xB) + E['const']
    return F, y


def mode2(sh, basis, detail=False):
    """Mode 2 vertex lower values for a basis (5 profile indices); None if singular."""
    bi = binv(basis)
    if bi is None: return None
    basis, BI, _ = bi
    Fc, _ = mode2_parts(sh.pt, basis, BI)
    Fb, yb_ = mode2_parts(sh.box, basis, BI)
    L = minorant(Fc.v, Fb, sh.lo, sh.hi, sh.zc)
    yv = yb_.v; c = sh.box['c'].v
    rc = c - (IA(AA) * yv[3] + IA(AB) * yv[4])
    pen = 0.0
    for k, idx in enumerate(CLS):
        nb = np.array([j for j in idx if j not in basis])
        if len(nb) == 0: continue
        r = rc[nb] - yv[k]
        mn = min(0.0, float(r.lo.min()))
        if mn < 0: pen = dn(pen + dn(mn * float(sh.box['mu'].v.hi[k])))
    if detail: return dn(L + pen), pen, Fb
    return dn(L + pen)


# ------------------------------------------------------------------ Mode 2 in the phi-basis (less dependency)
# c_j = C_j . phi,  phi = (sa, sb, w sa, w sb, sa sb),  C_j = (ga KP, gb KP, ga gv/D0, gb gv/D0, (R/D0) ga gb)
def _Cmat():
    rows = []
    for j in range(NP):
        ga, gb, gv, gab = GA[j], GB[j], GV[j], GAGB[j]
        rows.append([ga * KP, gb * KP, ga * gv / D0, gb * gv / D0, gab * R / D0])
    lo = np.array([[float(np.asarray(x.lo)) for x in r] for r in rows]); hi = np.array([[float(np.asarray(x.hi)) for x in r] for r in rows])
    return IA(lo, hi)
CM = _Cmat()                     # 20 x 5
_BD = {}


def basis_data(basis):
    """M_B (5x5: F = const + phi^T M_B b), Y (5x5: y_r = Y_r . phi), Gamma (20x5: rc_j = Gamma_j . phi)."""
    bi = binv(basis)
    if bi is None: return None
    basis, BI, _ = bi
    if basis in _BD: return _BD[basis]
    CB = IA(CM.lo[list(basis)], CM.hi[list(basis)])            # 5 x 5 (basis column i, phi component r)
    # M[r, c] = sum_i CB[i, r] BI[i, c]
    M = rsum(IA(CB.lo[:, :, None], CB.hi[:, :, None]) * IA(BI.lo[:, None, :], BI.hi[:, None, :]), 0)
    # Y[q, r] = sum_i BI[i, q] CB[i, r]   (dual component q)
    Y = rsum(IA(BI.lo[:, :, None], BI.hi[:, :, None]) * IA(CB.lo[:, None, :], CB.hi[:, None, :]), 0)
    G = []
    for j in range(NP):
        k = int(KV[j]) + 1
        G.append(CM[j] - Y[k] - Y[3] * float(AA[j]) - Y[4] * float(AB[j]))
    Gam = IA(np.array([g.lo for g in G]), np.array([g.hi for g in G]))
    _BD[basis] = (basis, M, Y, Gam)
    return _BD[basis]


def touter(f, g):
    """outer product of vector T1s f (m,), g (k,) -> flattened (m*k,)."""
    m, k = f.v.lo.shape[0], g.v.lo.shape[0]
    fv = IA(np.repeat(f.v.lo, k), np.repeat(f.v.hi, k)); gv = IA(np.tile(g.v.lo, m), np.tile(g.v.hi, m))
    fg = IA(np.repeat(f.g.lo, k, axis=1), np.repeat(f.g.hi, k, axis=1)); gg = IA(np.tile(g.g.lo, (1, m)), np.tile(g.g.hi, (1, m)))
    return T1(fv * gv, fg * _b(gv) + gg * _b(fv))


def F2(E, M):
    pb = touter(E['phi'], E['b'])
    Mf = IA(M.lo.reshape(1, -1), M.hi.reshape(1, -1))
    r = tlin(Mf, pb)
    return T1(r.v[0], IA(r.g.lo[:, 0], r.g.hi[:, 0])) + E['const']


def mode2b(sh, basis, detail=False):
    bd = basis_data(basis)
    if bd is None: return None
    basis, M, Y, Gam = bd
    Fc = F2(sh.pt, M); Fb = F2(sh.box, M)
    L = minorant(Fc.v, Fb, sh.lo, sh.hi, sh.zc)
    ph = sh.box['phi'].v
    rc = rsum(Gam * IA(ph.lo[None, :], ph.hi[None, :]), 1)     # interval of rc_j over the box
    pen = 0.0
    for k, idx in enumerate(CLS):
        nb = np.array([j for j in idx if j not in basis])
        if len(nb) == 0: continue
        mn = min(0.0, float(rc.lo[nb].min()))
        if mn < 0: pen = dn(pen + dn(mn * float(sh.box['mu'].v.hi[k])))
    if detail: return dn(L + pen), pen, Fb, rc
    return dn(L + pen)
