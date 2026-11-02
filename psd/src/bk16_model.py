"""bk16_model.py : the slice-LP relaxation of the CKS* objective, in first-order interval AD (T1 on ivarith.IA).

Coordinates z = (alpha, lam, d_a, d_b) (Lemma COORD; d = diagonal ratios, G_uu = 1).  For a u-star m on the slice
S(z) = {m >= 0 : class sums mu(alpha, lam), G_aa = d_a, G_bb = d_b}, intersected with the Lemma NS row and a level
{s_lo <= Moff(m) <= s_hi} of the off-face mass, the objective

    F(m) = 4 mu0 s_u(m) - KILL Delta_u(m; mu)          (see proof.md)

is bounded below by  4 mu0 Lfun(m) - KILL DUB(m),  where Lfun is the Lemma-L objective (linear on the slice) and DUB is
an upper bound of Delta_u that is linear on the slice once the RANGES of six linear functionals over the slice-level set
are known (secants, McCormick planes, tangent planes; see bk16 report Section 2.3).  Everything that depends on z is a
T1 object (value enclosure + gradient enclosure over the box); quantities that are only needed as upper bounds over the
whole box ("box constants") are IA intervals with zero gradient.
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, itertools
import numpy as np
from ivarith import IA, dn, up, rsum
import bk13_lb as L
from bk13_lb import T1, I, SQ2, ISQ2, RHO, MU0, D0, R, KP, KC, KE, PU, NP, KV, GA, GB, GV, GAGB, AA, AB, AFR, CLS, \
    tsum, tlin, tstack, tvec, _b

# ------------------------------------------------------------------ constants (outward)
CTH = I('0.9595702342051')          # c_Theta (Lemma NS) rounded UP (mpmath 50 digits: 0.95957023420505315...)
KILL = I('0.08126848580')           # 1/(N0^4 pi_lo), pi_lo = 0.815 (Lemma PI): 0.0812684857993..., rounded UP
EPS1 = I('1.02')                    # 1 + eps
LAMM = I('0.6')
R22 = SQ2 * 2.0
# the certified profile maxima (bk14_supcert.py), used AS DEFINED in Delta_u's sup constants
PMa_m1 = I('1.5035305'); PMa_1 = I('1.5035305'); PMa2_0 = I('2.2606040'); PMphm1 = I('0.4537150'); PMph1 = I('1.4537150')
PMwPh = I('0.6803371'); PMwPhP0 = I('2.1798184')
DLO, DHI = 0.74898, 1.33514         # R7 ratio box |log d| <= 0.28903, widened outward
VLO, VHI = 0.81451, 1.22773         # |log V| <= 0.20516, widened outward

iA, iB, iC = PU.index((-1, 1, 1)), PU.index((1, -1, 1)), PU.index((1, 1, -1))
ij0, ij0p, ijp, ijm = PU.index((0, 1, -1)), PU.index((0, -1, 1)), PU.index((1, -1, -1)), PU.index((1, 1, 1))
FACE = [iA, iB, iC, ij0, ij0p]
OFF = np.array([0.0 if j in FACE else 1.0 for j in range(NP)])
OFFm1 = np.array([1.0 if (j not in FACE and KV[j] == -1) else 0.0 for j in range(NP)])
I1 = np.where(KV == 1)[0]; IM = np.where(KV == -1)[0]; I0 = np.where(KV == 0)[0]
BREF = tuple(sorted([iA, iB, iC, ij0, ijp]))
_bi = L.binv(BREF); BI = _bi[1]                    # exact rational inverse, enclosed
BIT = IA(BI.lo.T.copy(), BI.hi.T.copy())
AT = IA(np.array([[float(AFR[r][j]) for r in range(5)] for j in range(NP)]))   # 20 x 5, exact (0, 1, 1/2, 2)
NONB = np.array([0.0 if j in BREF else 1.0 for j in range(NP)])


def tconst(x, n=4):
    x = IA.of(x); return T1(x, IA(np.zeros((n,) + np.shape(x.lo))))


def tabs(f):
    """|f| for a T1 (vector or scalar): Clarke-gradient hull where the sign is not fixed (Lebourg mean value theorem)."""
    lo, hi = np.asarray(f.v.lo), np.asarray(f.v.hi)
    pos = lo >= 0; neg = hi <= 0
    vlo = np.where(pos, lo, np.where(neg, -hi, 0.0)); vhi = np.where(pos, hi, np.where(neg, -lo, np.maximum(-lo, hi)))
    glo, ghi = np.asarray(f.g.lo), np.asarray(f.g.hi)
    G_lo = np.where(pos, glo, np.where(neg, -ghi, np.minimum(glo, -ghi)))
    G_hi = np.where(pos, ghi, np.where(neg, -glo, np.maximum(ghi, -glo)))
    return T1(IA(vlo, vhi), IA(G_lo, G_hi))


def tpos(f):
    """max(f, 0) for a scalar T1 (gradient hull with 0 where the sign is not fixed)."""
    lo, hi = float(f.v.lo), float(f.v.hi)
    if lo >= 0: return f
    if hi <= 0: return tconst(0.0)
    glo, ghi = np.asarray(f.g.lo), np.asarray(f.g.hi)
    return T1(IA(0.0, hi), IA(np.minimum(glo, 0.0), np.maximum(ghi, 0.0)))


def tmax(lst):
    """max of scalar T1s: value hull of the max, gradient hull of the candidates that can be active (Lebourg MVT)."""
    mlo = max(float(f.v.lo) for f in lst); mhi = max(float(f.v.hi) for f in lst)
    act = [f for f in lst if float(f.v.hi) >= mlo]
    glo = np.min([np.asarray(f.g.lo) for f in act], axis=0); ghi = np.max([np.asarray(f.g.hi) for f in act], axis=0)
    return T1(IA(mlo, mhi), IA(glo, ghi))


def tdot(f, c):
    """T1 vector f (20) dotted with a constant float vector c (exact doubles)."""
    return tsum(f * IA(np.asarray(c, float)))


def tvecconst(arr):
    return tconst(IA(np.asarray(arr, float)))


def wedge(pa, pb, qa, qb):
    return pa * qb - pb * qa


def face_closed(cm1, cls1):
    """phi, Abar, w11 = w10 of the face star at a marginal with class sums (c_-1, ., c_1) (T1 scalars).
    The face star has d_a = d_b = dF = 1.25 - 0.75 c_-1, n = dF^{-1/2}; closed forms CERTIFIED by w/verify_face.py:
    Abar = (n/sqrt2)(1 + n/sqrt2), w11 = w10 = (n/sqrt2)(2 + 3 n/sqrt2), phi = c_1 (2 + (3/sqrt2) n)/(2 + sqrt2 n)."""
    n = (cm1 * (-0.75) + 1.25).sqrt().inv()
    t = n * ISQ2
    Abar = t * (t + 1.0)
    w = t * (t * 3.0 + 2.0)
    phi = cls1 * (n * (SQ2 * 1.5) + 2.0) / (n * SQ2 + 2.0)
    return phi, Abar, w, w


def thresholds_closed(cm1, cls1):
    phi, Abar, w11, w10 = face_closed(cm1, cls1)
    return dict(U=phi * Abar * EPS1, A=Abar * EPS1, phi=phi * EPS1, w11=w11 * EPS1, w10=w10 * EPS1)


def quantities(z, BC=None):
    """All z-dependent quantities.  BC: box constants (computed from the box evaluation and re-used at the point)."""
    al, la, da, db = z
    sa = da.sqrt().inv(); sb = db.sqrt().inv()
    q_ = 1.0 - la
    S = q_ * (2.0 - al) + la; Si = S.inv()
    m1 = al * q_ * Si; m0 = la * Si; p1 = (1.0 - al) * q_ * Si * 2.0
    mu = [m1, m0, p1]
    V = m1 * 2.0 + m0 + p1 * 0.5
    U = m1 * SQ2 + m0 + p1 * ISQ2
    w = V.sqrt().inv(); e0 = U * w; om = V.sqrt()
    # Lemma L objective (see proof.md)
    constL = e0 * KE + KC
    coefa = tvec(w * D0.inv(), GV) + KP
    cL = tvec(sa, GA) * coefa + tvec(sb, GB) * coefa + tvec(sa * sb * (R / D0), GAGB)
    # Lemma NS row (Lemma L form): nsg . m <= nsh
    k8 = D0.inv() * 2.0 - (D0 * D0).inv() * 8.0
    nsg = (tvec(sa, GA) + tvec(sb, GB)) * k8 + (tvec(sa, GA) + tvec(sb, GB)) * tvec(w * D0.inv(), GV) + tvec(sa * sb * D0.inv(), GAGB)
    nsc = e0 * 2.0 / D0 + D0.inv() * 3.0 - (e0 * 2.0 - RHO * 6.0) * (D0 * D0).inv() * 4.0
    nsh = -nsc + CTH * 3.0
    b = [m1, m0, p1, da, db]
    bT = tstack(b)
    # profile data
    Pa = tvec(sa, GA) + 1.0; Pb = tvec(sb, GB) + 1.0
    P0a, P0b = Pa[iA], Pb[iA]
    a = Pb * P0a - Pa * P0b                              # a_j = P0 ^ P_j
    ph = (Pa + Pb) / (P0a + P0b)
    absa = tabs(a)
    mask = lambda cls: np.array([1.0 if KV[j] == cls else 0.0 for j in range(NP)])
    M1v, M0v, Mm1 = mask(1), mask(0), mask(-1)
    xv = -(a * IA(M1v))                                  # X = sum_{U1} m (P ^ P0) = - sum_{U1} m a
    fv = ph * IA(M1v)
    kv = absa * IA(Mm1)
    a0v = a * a * IA(M0v)
    kpv = tabs((ph - 1.0) * IA(Mm1))
    p0v = ph * ph * IA(M0v)

    def reduce(v):
        lam = tlin(BIT, T1(IA(np.asarray(v.v.lo)[list(BREF)], np.asarray(v.v.hi)[list(BREF)]),
                           IA(np.asarray(v.g.lo)[:, list(BREF)], np.asarray(v.g.hi)[:, list(BREF)])))
        vt = v - tlin(AT, lam)
        vt = vt * IA(NONB)                               # exact zeros on the basis (identity B^T lam = v_B)
        vB = tsum(lam * bT)
        return vt, vB
    xt, XB = reduce(xv); ft, phiB = reduce(fv); at, A0B = reduce(a0v); zt, P0B = reduce(p0v)
    Dq = (al + 1.0) * q_ + la                            # q = mirror marginal = (mu_1/2, mu_0, 2 mu_-1)/V, V = Dq/S
    qm = [(1.0 - al) * q_ / Dq, la / Dq, al * q_ * 2.0 / Dq]
    Vq = S / Dq
    Tp = thresholds_closed(m1, p1); Tq = thresholds_closed(qm[0], qm[2])
    Sq = dict(K=qm[0] * PMa_m1, A0=qm[1] * PMa2_0, Zphi=qm[2] * PMph1 * qm[0] * PMa_m1 / Vq,
              ZX=qm[2] * PMa_1 * qm[0] * PMphm1 / Vq, ZR=qm[0] * qm[0] * PMwPh / Vq, ZRm=qm[0] * qm[1] * PMwPhP0 * SQ2 / Vq)
    Q = dict(z=z, nsc=nsc, mu=mu, V=V, om=om, w=w, constL=constL, cL=cL, nsg=nsg, nsh=nsh, b=b, da=da, db=db,
             xt=xt, XB=XB, ft=ft, phiB=phiB, at=at, A0B=A0B, zt=zt, P0B=P0B, ph=ph, sa=sa, sb=sb, kv=kv, kpv=kpv, p0v=p0v, Tp=Tp, Tq=Tq, Sq=Sq)
    if BC is None:
        # box constants (upper ends over the box); a, P are interval vectors over the box here
        pa, pb = Pa.v, Pb.v
        def wd(i, j): return pa[i] * pb[j] - pb[i] * pa[j]
        def mag(x): return float(np.max(np.maximum(-np.asarray(x.lo), np.asarray(x.hi))))
        Abar = max(mag(a.v[j]) for j in I1); Phibar = float(np.max(np.asarray(ph.v.hi)[I1]))
        w11 = max(mag(wd(i, j)) for i in I1 for j in I1 if i < j); w10 = max(mag(wd(i, j)) for i in I1 for j in I0)
        Pha = pa - pa[iA]; Phb = pb - pb[iA]
        def wdh(i, j): return Pha[i] * Phb[j] - Phb[i] * Pha[j]
        def wdpq(i, j): return pa[i] * Phb[j] - pb[i] * Pha[j]
        pq = np.zeros(NP)
        for j in IM:
            if j != iA: pq[j] = max(mag(wdpq(i, j)) for i in I1)
        WPh = max(mag(wdh(i, j)) for i in IM for j in IM if i != j)
        W00 = max(mag(wd(i, j)) for i in I0 for j in I0) ** 2
        Wm0 = max(mag(Pha[i] * pb[j] - Phb[i] * pa[j]) for i in IM for j in I0)
        BC = dict(Abar=up(Abar), Phibar=up(Phibar), w11=up(w11), w10=up(w10), pq=up(pq), WPh=up(WPh), W00=up(up(W00)), Wm0=up(Wm0))
    Q['BC'] = BC
    Q['yv'] = tvecconst(BC['pq'])                        # Ybar = sum pq_j m_j >= Y
    return Q


# ------------------------------------------------------------------ the functionals whose ranges are needed
RANGE_KEYS = ('xi', 'xi0', 'fi', 'K', 'Y', 'M1')


def functional(Q, key):
    if key == 'xi': return Q['xt']
    if key == 'xi0': return Q['at']
    if key == 'fi': return Q['ft']
    if key == 'K': return Q['kv']
    if key == 'Y': return Q['yv']
    if key == 'M1': return tvecconst(OFFm1)
    if key == 'zeta': return Q['zt']


def mc_upper(xv, x_lo, x_hi, yv, y_lo, y_hi, th):
    """McCormick upper bound of (x.m)(y.m) on the box of ranges, as th * plane1 + (1 - th) * plane2  -> (cost vector, const).
    plane1: x_hi y + y_lo x - x_hi y_lo;  plane2: y_hi x + x_lo y - x_lo y_hi.  th in {0, 0.5, 1} (exact weights)."""
    assert th in (0.0, 0.5, 1.0)
    t1, t2 = th, 1.0 - th
    c = yv * (IA(x_hi) * t1 + IA(x_lo) * t2) + xv * (IA(y_lo) * t1 + IA(y_hi) * t2)
    k = -(IA(x_hi) * y_lo * t1 + IA(x_lo) * y_hi * t2)
    return c, k


def main_cost(Q, rg, ch):
    """Cost (T1 vector) and constant (T1) of the lower-bounding linear objective, in units of 4 mu0 s.
    rg: dict key -> (lo, hi) certified ranges over the slice-level set (floats);
    ch: choice weights: 'xs' in [0,1] (secant vs abs form for xi^2), 'x0s' (same for xi0^2), 'mK', 'm0' (McCormick weights),
        'R' (list of 9 nonnegative weights summing to 1: tangent at 0 and at M1hi e_k for the 8 off-face class -1 profiles)."""
    V, om, BC = Q['V'], Q['om'], Q['BC']
    Tp, Tq, Sq = Q['Tp'], Q['Tq'], Q['Sq']
    xt, XB, ft, phiB, at, A0B, kv = Q['xt'], Q['XB'], Q['ft'], Q['phiB'], Q['at'], Q['A0B'], Q['kv']
    m0 = Q['mu'][1]
    xlo, xhi = rg['xi']; x0lo, x0hi = rg['xi0']; flo, fhi = rg['fi']; Klo, Khi = rg['K']; Ylo, Yhi = rg['Y']; M1hi = rg['M1'][1]
    Phib = IA(BC['Phibar'])
    cX = (om * R22 * 2.0).inv() * LAMM + (LAMM * Phib * Phib * 0.5 + 0.25)     # T1 scalar
    c = tvecconst(np.zeros(NP)); k = tconst(0.0)
    # ---- face + AM-GM X^2 parts: cX X^2, X = XB + xi,  xi^2 <= xs*secant + (1-xs)*abs-form
    Xi = max(-xlo, xhi, 0.0)
    xs = ch['xs']; assert xs in (0.0, 1.0)
    sq_c = xt * ((IA(xlo) + xhi) * xs) + tabs(xt) * (Xi * (1.0 - xs)); sq_k = -(IA(xlo) * xhi) * xs
    c = c + (xt * (XB * 2.0) + sq_c) * cX; k = k + (XB * XB + sq_k) * cX
    # ---- J own: Tq.U K + Tq.A/V (phiB K + (ft.m) K) + [phiB A0B + phiB xi0 + A0B ft.m + (ft.m) xi0]/(om R22) + om Tq.phi (A0B + xi0)/R22
    cfK, kfK = mc_upper(ft, flo, fhi, kv, Klo, Khi, ch['mK'])
    cf0, kf0 = mc_upper(ft, flo, fhi, at, x0lo, x0hi, ch['m0'])
    TqAV = Tq['A'] / V
    c = c + kv * Tq['U'] + (kv * phiB + cfK) * TqAV
    k = k + TqAV * kfK
    c = c + (at * phiB + ft * A0B + cf0) / (om * R22) + at * (om * Tq['phi'] / R22)
    k = k + (phiB * A0B + kf0) / (om * R22) + om * Tq['phi'] * A0B / R22
    # ---- AM-GM: K^2/(2 lamM) (secant), om A0^2/(2 R22 lamM), A0 = A0B + xi0, xi0^2 secant/abs
    c = c + kv * ((IA(Klo) + Khi) / (LAMM * 2.0)); k = k - (IA(Klo) * Khi) / (LAMM * 2.0)
    Xi0 = max(-x0lo, x0hi, 0.0); x0s = ch['x0s']; assert x0s in (0.0, 1.0)
    s0c = at * ((IA(x0lo) + x0hi) * x0s) + tabs(at) * (Xi0 * (1.0 - x0s)); s0k = -(IA(x0lo) * x0hi) * x0s
    fac = om / (R22 * LAMM * 2.0)
    c = c + (at * (A0B * 2.0) + s0c) * fac; k = k + (A0B * A0B + s0k) * fac
    # ---- M own: Tq.A |X| Kp / V + |X| Phi0 / (R22 om),  |X| <= |XB| + Xi
    aX = tabs(XB) + Xi
    c = c + (Q['kpv'] * TqAV + Q['p0v'] / (om * R22)) * aX
    # ---- det E: Tq.w11 R / V + R0/4 + Y^2 + (sqrt2/V) Tq.w10 Rm0
    wR = Tq['w11'] / V * BC['WPh']
    th = [float(x) for x in ch['R']]
    assert len(th) == 9 and min(th) >= 0 and float(IA(np.array(th)).lo.sum()) <= 1.0 and sum(th) <= 1.0
    offk = [j for j in range(NP) if OFFm1[j] > 0]
    tl = np.zeros(NP); th_ = np.zeros(NP); tk = IA(0.0)
    for wgt, j in zip(th[1:], offk):
        e = IA(wgt) * 2.0 * M1hi; tl[j] = e.lo; th_[j] = e.hi; tk = tk + IA(wgt) * M1hi * M1hi
    cvec = IA(OFFm1) * M1hi - IA(tl, th_)
    c = c + T1(cvec, IA(np.zeros((4, NP)))) * wR; k = k + wR * tk
    k = k + m0 * m0 * (BC['W00'] / 4.0)
    c = c + Q['yv'] * (IA(Ylo) + Yhi); k = k - IA(Ylo) * Yhi
    c = c + tvecconst(OFFm1) * (m0 * Tq['w10'] * SQ2 / V * BC['Wm0'])
    # ---- hinges
    Abar = IA(BC['Abar'])
    h = tpos(phiB * Abar + Abar * fhi - Tp['U']) * Sq['K']
    h = h + tpos(-Tp['A'] + Abar) * (Sq['Zphi'] + Sq['ZX'])
    h = h + tpos(phiB + fhi - Tp['phi']) * Sq['A0'] / (om * R22)
    h = h + tpos(-Tp['w11'] + IA(BC['w11'])) * Sq['ZR'] + tpos(-Tp['w10'] + IA(BC['w10'])) * Sq['ZRm']
    k = k + h
    # ---- total, units of 4 mu0 s
    F4 = MU0 * 4.0
    cost = Q['cL'] * F4 - c * KILL
    const = Q['constL'] * F4 - k * KILL
    return cost, const


def quantities_pm(z, BC=None):
    """z = (alpha, lam, dp, dm): d_a = (tau + dp + dm)/2, d_b = (tau + dp - dm)/2, tau = 5/2 - (3/2) mu_-1 (Lemma KINK)."""
    al, la, dp, dm = z
    q_ = 1.0 - la; S = q_ * (2.0 - al) + la
    m1 = al * q_ / S
    tau = m1 * (-1.5) + 2.5
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    Q = quantities([al, la, da, db], BC=BC)
    Q['z'] = list(z)
    return Q
