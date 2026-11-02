"""hloc.py : CERTIFIED local bound for M_123 around the tree point, in the exact linear chart G (20 coordinates,
G_uu = 1).  For m in the simplex with |G(m) - G_T|_ij <= rho for all 20 coordinates:
   f(m) = Psi(G(m)) = Psi(G_T) + DPsi(G_T)[G(m) - G_T] + (1/2) D^T H(xi) D     (Taylor, xi on the segment, in the box)
   DPsi(G_T)[G(m) - G_T] = sum_t m_t D_t f(m_T) <= 0                         (Euler + GT Theorem II.4, re-verified)
   (1/2) D^T H(xi) D <= remainder(H_box, rho)                                (interval Hessian, eigen split)
so f(m) <= Psi(G_T)^hi + remainder.  Interval class I of the checked cert.py (round to nearest + 1 ulp outward)."""
import os, sys, math, json, time
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gt'))
import numpy as np
import cert_gt as CG
from cert_gt import I, isum, iasin, PI, F_point, chol_pd, meet, cv, dn, up
import lib

N = 20
IDX = [(i, j) for i in range(6) for j in range(i, 6) if (i, j) != (0, 0)]
Z = I(0.0)


class AD:
    """value, gradient, Hessian (intervals) in the N coordinates; forward mode, dense; g = None means a constant."""
    __slots__ = ('v', 'g', 'h')

    def __init__(s, v, g, h): s.v, s.g, s.h = v, g, h

    @staticmethod
    def const(x): return AD(cv(x), None, None)

    def __add__(s, o):
        o = o if isinstance(o, AD) else AD.const(o)
        if s.g is None and o.g is None: return AD(s.v + o.v, None, None)
        if s.g is None: return AD(s.v + o.v, o.g, o.h)
        if o.g is None: return AD(s.v + o.v, s.g, s.h)
        return AD(s.v + o.v, [x + y for x, y in zip(s.g, o.g)],
                  [[x + y for x, y in zip(p, q)] for p, q in zip(s.h, o.h)])
    __radd__ = __add__

    def __neg__(s): return s*(-1.0)

    def __sub__(s, o): return s + (-(o if isinstance(o, AD) else AD.const(o)))

    def __rsub__(s, o): return (-s) + o

    def __mul__(s, o):
        if not isinstance(o, AD): o = AD.const(o)
        if o.g is None:
            if s.g is None: return AD(s.v*o.v, None, None)
            return AD(s.v*o.v, [x*o.v for x in s.g], [[x*o.v for x in r] for r in s.h])
        if s.g is None: return o*s
        g = [x*o.v + s.v*y for x, y in zip(s.g, o.g)]
        n = len(s.g)
        h = [[s.h[i][j]*o.v + s.v*o.h[i][j] + s.g[i]*o.g[j] + s.g[j]*o.g[i] for j in range(n)] for i in range(n)]
        return AD(s.v*o.v, g, h)
    __rmul__ = __mul__

    def chain(s, f0, f1, f2):
        """phi(s) with phi = f0, phi' = f1, phi'' = f2, each an enclosure over the value interval s.v."""
        if s.g is None: return AD(f0, None, None)
        n = len(s.g)
        return AD(f0, [f1*x for x in s.g], [[f1*s.h[i][j] + f2*s.g[i]*s.g[j] for j in range(n)] for i in range(n)])

    def rsqrt(s):
        if s.v.lo <= 0: raise ArithmeticError('rsqrt')
        q = 1/s.v.sqrt(); return s.chain(q, -0.5*q/s.v, 0.75*q/(s.v*s.v))

    def acos(s):
        x = meet(s.v, I(-1.0, 1.0)); om = 1 - x*x
        if om.lo <= 0: raise ArithmeticError('acos')
        r = 1/om.sqrt()
        return s.chain(PI*0.5 - iasin(x), -r, -x*r*r*r)


def var(x, k, n=N):
    g = [Z]*n; g[k] = I(1.0)
    return AD(cv(x), g, [[Z]*n for _ in range(n)])


EDG = [(0, 2), (0, 3), (1, 4), (1, 5)]
PAIRS, COMP = lib.PAIRS, lib.COMP


def cmap(vals):
    """vals: 20 intervals (a box of G-coordinates) -> the six C entries (PAIRS order) as AD objects in 20 variables."""
    G = {(0, 0): AD.const(1.0)}
    for k, (i, j) in enumerate(IDX): G[i, j] = G[j, i] = var(vals[k], k)
    rs = [G[x, x].rsqrt() for x in range(6)]
    Gh = {}
    for x in range(6):
        for y in range(x, 6):
            Gh[x, y] = Gh[y, x] = AD.const(1.0) if x == y else G[x, y]*rs[x]*rs[y]

    def Wf(e, f):
        (a, b), (c, d) = EDG[e], EDG[f]
        return Gh[a, c] + Gh[a, d] + Gh[b, c] + Gh[b, d]
    W = {(e, f): Wf(e, f) for e in range(4) for f in range(e, 4)}
    rw = [W[e, e].rsqrt() for e in range(4)]
    return [W[e, f]*rw[e]*rw[f] for e, f in PAIRS]


def dphi_ad(cvals, cen):
    """cvals: 6 intervals (a correlation box, PAIRS order), cen: point enclosures inside it -> dPhi_F/dr_p (Lemma B) as AD
    objects in the 6 correlations; .v encloses the first and .g the second derivatives of Phi_F over the box.  Every
    intermediate quantity's value enclosure is met with its mean-value form q(cen) + sum_q [dq/dr_q]_box (r_q - cen_q)."""
    Dl6 = [I((I(c.lo) - m.hi).lo, (I(c.hi) - m.lo).hi) for c, m in zip(cvals, cen)]

    def mv(xb, xc):
        if xb.g is None: return xb
        return AD(meet(xb.v, xc.v + isum(g*d for g, d in zip(xb.g, Dl6))), xb.g, xb.h)

    def build(vals):
        C = {}
        for q, (e, f) in enumerate(PAIRS): C[e, f] = C[f, e] = var(vals[q], q, 6)
        out = {}
        for p in PAIRS:
            (i, j), (k, l) = p, COMP[p]
            rij, rkl, rki, rli, rkj, rlj = C[i, j], C[k, l], C[k, i], C[l, i], C[k, j], C[l, j]
            Nn = rki*rli + rkj*rlj - rij*(rki*rlj + rkj*rli)
            om = 1 - rij*rij
            Dk = om - (rki*rki + rkj*rkj - 2*rij*rki*rkj)
            Dl = om - (rli*rli + rlj*rlj - 2*rij*rli*rlj)
            out[p] = (om, Dk*Dl, rkl*om - Nn)
        return out
    B, Cc = build(cvals), build(cen)
    dP = []
    for p in PAIRS:
        om, DD, nu = (mv(x, y) for x, y in zip(B[p], Cc[p]))
        omc, DDc, nuc = Cc[p]
        if DD.v.lo <= 0 or om.v.lo <= 0 or DDc.v.lo <= 0: raise ArithmeticError('minor')
        rho = mv(nu*DD.rsqrt(), nuc*DDc.rsqrt())
        phi2 = om.rsqrt()*(1/(2*PI))
        ac = rho.acos()*(1/PI)
        dP.append(-1*phi2*(ac + 1.0) if p in PAIRS[:2] else -1*phi2*ac)
    return dP


KSPLIT = int(os.environ.get('HLOC_K', '1'))


def hull(a, b): return I(min(a.lo, b.lo), max(a.hi, b.hi))


def dphi_split(Cb, k):
    """dphi_ad over the correlation box Cb, as the hull of its values over the k^6 sub-boxes of a uniform grid (each
    sub-box with its own centre); the sub-boxes cover Cb, so the hull encloses the derivatives over Cb."""
    import itertools
    edges = []
    for c in Cb:
        pts = [c.lo] + [c.lo + (c.hi - c.lo)*i/k for i in range(1, k)] + [c.hi]
        edges.append([(pts[i], pts[i + 1]) for i in range(k)])
    out = None
    for cell in itertools.product(*edges):
        box = [I(a, b) for a, b in cell]
        cen = [I(0.5*(a + b)) for a, b in cell]
        cen = [meet(x, y) for x, y in zip(cen, box)]
        dP = dphi_ad(box, cen)
        if out is None: out = [AD(d.v, list(d.g), None) for d in dP]
        else:
            out = [AD(hull(o.v, d.v), [hull(x, y) for x, y in zip(o.g, d.g)], None) for o, d in zip(out, dP)]
    return out


def hess_box(lo, hi, cen):
    """Enclosures over the G-box [lo, hi] of Hess Psi (Psi = Phi_F o C, 20 coordinates).  cen: point enclosures (inside
    the box) used for the mean-value enclosure of the C-box.  Returns (H, C-box, Phi-Hessian enclosure)."""
    Cs = cmap([I(a, b) for a, b in zip(lo, hi)])
    Cc = cmap(cen)
    D = [I((I(a) - c.hi).lo, (I(b) - c.lo).hi) for a, b, c in zip(lo, hi, cen)]
    Cb = [meet(Cs[p].v, Cc[p].v + isum(Cs[p].g[k]*D[k] for k in range(N))) for p in range(6)]
    Clo, Chi = [c.lo for c in Cb], [c.hi for c in Cb]
    if not chol_pd(Clo, Chi): raise ArithmeticError('C box not certified PD')
    dP = dphi_split(Cb, KSPLIT)
    J = [[Cs[p].g[k] for k in range(N)] for p in range(6)]
    HP = [[(dP[p].g[q] + dP[q].g[p])*0.5 for q in range(6)] for p in range(6)]
    JHJ = [[isum(HP[p][q]*J[q][l] for q in range(6)) for l in range(N)] for p in range(6)]
    H = [[isum(J[p][k]*JHJ[p][l] + dP[p].v*Cs[p].h[k][l] for p in range(6)) for l in range(N)] for k in range(N)]
    return H, (Clo, Chi), HP


def remainder(H, d):
    """(1/2) max D^T M D over symmetric M in H and |D_p| <= d_p (cert.py Lemma box(c), with n = len(d))."""
    n = len(d)
    Hs = [[(H[p][q] + H[q][p])*0.5 for q in range(n)] for p in range(n)]
    lam, V = np.linalg.eigh(np.array([[x.mid() for x in a] for a in Hs]))
    E = [[Hs[p][q] - isum(I(V[p, k])*I(V[q, k])*I(lam[k]) for k in range(n)) for q in range(n)] for p in range(n)]
    tot = isum(E[p][q].mag()*I(d[p])*I(d[q]) for p in range(n) for q in range(n))
    for k in (k for k in range(n) if lam[k] > 0):
        w = isum(abs(V[p, k])*I(d[p]) for p in range(n))
        tot = tot + w*w*lam[k]
    return (tot*0.5).hi


def gp(e):
    """2^{e/2} as an interval (exact for even e)."""
    return I(2.0**(e//2))*CG.S2 if e % 2 else I(2.0**(e//2))


def GT_vals():
    """Enclosures of the 20 coordinates of G_T = G(m_T), sum m_T = 1 (masses 1/3, 1/3, 1/6, 1/6)."""
    out = []
    for (x, y) in IDX:
        s = Z
        for t, w in lib.TREE.items():
            s = s + (I(1.0)/3 if w > 0.3 else I(1.0)/6)*gp(-(t[x] + t[y]))
        out.append(s)
    return out


def psi_T_upper():
    """Upper bound of Phi_F(C_T), C_T = ((1+2 sqrt2)/4 (ab, cd), (2+sqrt2)/4 (four cross)).  Phi_F is nonincreasing in
    every correlation (Lemma G(b) of proof.md), so Phi_F(C_T) <= Phi_F(c) for doubles c <= C_T entrywise with C(c) PD."""
    S2 = CG.S2
    rab = (1 + 2*S2)/4; rx = (2 + S2)/4
    c = [rab.lo]*2 + [rx.lo]*4
    assert chol_pd(c, c)
    return F_point(c).hi, c


if __name__ == '__main__':
    t0 = time.time()
    M, c = psi_T_upper(); print('Phi_F(C_T) <=', repr(M), ' at doubles', c, flush=True)
    vT = GT_vals()
    lo = [v.lo for v in vT]; hi = [v.hi for v in vT]
    H, _, _ = hess_box(lo, hi, vT)
    Hm = np.array([[x.mid() for x in r] for r in H])
    try:
        Hfd = np.load('H_G.npy'); print('NUMERICAL check: max |H_ad - H_fd| at G_T:', np.abs(Hm - Hfd).max(), flush=True)
    except Exception as e: print('no fd file', e)
    print('eig of H(G_T) (mid):', np.round(np.linalg.eigvalsh((Hm + Hm.T)/2), 4))
    res = {}
    for rho in [float(x) for x in sys.argv[1:]]:
        lo = [dn(v.lo - rho) for v in vT]; hi = [up(v.hi + rho) for v in vT]
        try:
            H, (Clo, Chi), HP = hess_box(lo, hi, vT)
            # Taylor about the exact G_T (inside every [v.lo, v.hi]); |G - G_T|_k <= hi_k - lo_k =: d_k is a crude
            # but valid radius; the sharper one is rho + width(G_T enclosure):
            d = [up(rho + up(v.hi - v.lo)) for v in vT]
            R = remainder(H, d)
            res[rho] = (M + R, R)
            print('rho', rho, 'remainder', R, ' bound f <=', M + R, ' C box lo', np.round(Clo, 4), 'hi', np.round(Chi, 4),
                  round(time.time() - t0, 1), 's', flush=True)
        except ArithmeticError as e:
            print('rho', rho, 'FAIL', e, flush=True)
    json.dump({str(k): v for k, v in res.items()}, open('hloc_out.json', 'w'))
