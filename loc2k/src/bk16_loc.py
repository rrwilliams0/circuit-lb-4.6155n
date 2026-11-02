"""bk16_loc.py : the local relaxation near the tree marginal (region Q x D_loc) and Lemma KINK-K.

LOCAL COST.  For stars m on the slice with Moff(m) <= s1 (level [0, s1]) and the Lemma NS row, with certified ranges
(over that set) Xi_x >= |xi|, Xi_0 >= |xi0|, Fb >= |ft.m|, Khi >= K, Yhi >= Y, M1hi >= M1, Z >= |zeta|, the kill satisfies
Delta_u(m) <= DL(m) := (linear in m, see local_cost), provided every hinge argument is certified <= 0 on the box, and DL is
EXACT-AT-THE-FACE: at m4(p, t) (support A, B, C, j0 of the basis B+) every range-dependent term vanishes and DL(m4) is the
smooth function Dface(p, t) of bk16_loc2 (used by Lemma LOC2-K).  Units: |D(W)|; the objective is 4 mu0 Lfun - KILL DL.

KINK-K.  Mode 2 with the basis B+ = {A, B, C, j0, jp} (d_+ >= 0) or B- = {A, B, C, j0, jm} (d_+ <= 0) and the shifted dual
y' = y_B + eta (u), u = (-base_k on the class duals, +1, +1 on the two d-rows), eta <= 0 a constant of the box:
   c.m = y'.b + sum_j rc'_j m_j,   rc'_j = rc_j - eta delta_j,   y'.b = F_B(z) + eta d_+,
delta_j = (A_aj + A_bj) - base_{k(j)} (base = 1, 5/2, 5/2 for classes -1, 0, 1).  If every rc'_j >= 0 except j0' (penalty
P0 mu_0) and d/dd_+ (F_B + eta d_+) >= 0 (<= 0 on the minus side), then LP >= F_B(p, 0, d_-) - P0 mu_0 = h_K(p, t(d)) - P0 mu_0.
"""
import numpy as np
from ivarith import IA, dn, up, rsum
import bk13_lb as L
from bk13_lb import T1, tlin, tsum, PU, NP, KV, AA, AB, AFR
import bk16_model as M
from bk16_model import tabs, tconst, tvecconst, LAMM, R22, SQ2, KILL, MU0

GAMMA = 0.05
iA, iB, iC, ij0, ij0p, ijp, ijm = M.iA, M.iB, M.iC, M.ij0, M.ij0p, M.ijp, M.ijm
U0x = np.array([1.0 if (KV[j] == 0 and j != ij0) else 0.0 for j in range(NP)])     # class 0 except j0
LOCAL_KEYS = ('xi', 'xi0', 'fi', 'K', 'Y', 'M1', 'zeta')
BASE = {-1: 1.0, 0: 2.5, 1: 2.5}
DELTA = np.array([AA[j] + AB[j] - BASE[int(KV[j])] for j in range(NP)])            # exact (0.5, 1, 1.5, 2, 3, -1.5, ...)
BPLUS = tuple(sorted((iA, iB, iC, ij0, ijp))); BMINUS = tuple(sorted((iA, iB, iC, ij0, ijm)))


def phibar_smooth(Q):
    """ph(jp) as the smooth Phibar; returns None unless jp is certified to attain max_{U1} ph on the box."""
    ph = Q['ph']
    others = [j for j in M.I1 if j != ijp]
    if float(ph.v.lo[ijp]) >= max(float(ph.v.hi[j]) for j in others): return ph[ijp]
    return None


def centered(fp, fb, bx):
    """mean-value (centred) enclosure over the box: f(Z) in f(zc) + sum_i G_i [lo_i - zc_i, hi_i - zc_i],
    fp = T1 at the centre (value enclosure), fb = T1 over the box (gradient enclosure)."""
    tot = IA(np.asarray(fp.v.lo), np.asarray(fp.v.hi))
    for i in range(4):
        D = IA(dn(bx.lo[i] - bx.zc[i]), up(bx.hi[i] - bx.zc[i]))
        G = IA(np.asarray(fb.g.lo)[i], np.asarray(fb.g.hi)[i])
        tot = tot + G * D
    # intersect with the natural enclosure
    lo = np.maximum(np.asarray(tot.lo), np.asarray(fb.v.lo)); hi = np.minimum(np.asarray(tot.hi), np.asarray(fb.v.hi))
    return IA(lo, hi)


def hinge_args(Q, Fb):
    BC = Q['BC']; Tp = Q['Tp']; Abar = IA(BC['Abar'])
    return [Q['phiB'] * Abar + Abar * Fb - Tp['U'], -Tp['A'] + Abar, Q['phiB'] + Fb - Tp['phi'],
            -Tp['w11'] + IA(BC['w11']), -Tp['w10'] + IA(BC['w10'])]


def hinges_inactive(bx, Fb):
    hp = hinge_args(bx.Qp, Fb); hb = hinge_args(bx.Qb, Fb)
    return all(float(centered(a, b_, bx).hi) <= 0 for a, b_ in zip(hp, hb))


def local_cost(Q, rg, check_hinges=False):
    """(cost T1 vector, const T1) of 4 mu0 Lfun - KILL DL, or None if a hinge is not certified inactive / Phibar not smooth."""
    V, om, BC, Tq = Q['V'], Q['om'], Q['BC'], Q['Tq']
    xt, XB, ft, phiB, at, A0B, kv, zt, P0B = Q['xt'], Q['XB'], Q['ft'], Q['phiB'], Q['at'], Q['A0B'], Q['kv'], Q['zt'], Q['P0B']
    m0 = Q['mu'][1]
    Xi = max(abs(rg['xi'][0]), abs(rg['xi'][1])); Xi0 = max(abs(rg['xi0'][0]), abs(rg['xi0'][1]))
    Fb = max(abs(rg['fi'][0]), abs(rg['fi'][1])); Khi = max(rg['K'][1], 0.0); Yhi = max(rg['Y'][1], 0.0)
    M1hi = max(rg['M1'][1], 0.0); Z = max(abs(rg['zeta'][0]), abs(rg['zeta'][1]))
    if check_hinges:
        for h in hinge_args(Q, Fb):
            if float(h.v.hi) > 0: return None
    Phib = phibar_smooth(Q)
    if Phib is None: return None
    cX = (om * R22 * 2.0).inv() * (LAMM + GAMMA) + Phib * Phib * (LAMM * 0.5) + 0.25
    # X^2 <= XB^2 + 2 XB xi + Xi |xt|.m
    X2c = xt * (XB * 2.0) + tabs(xt) * Xi; X2k = XB * XB
    c = X2c * cX; k = X2k * cX
    TqAV = Tq['A'] / V
    # J own
    c = c + kv * Tq['U'] + (kv * phiB + tabs(ft) * Khi) * TqAV
    c = c + (at * phiB + ft * A0B + tabs(ft) * Xi0) / (om * R22) + at * (om * Tq['phi'] / R22)
    k = k + phiB * A0B / (om * R22) + om * Tq['phi'] * A0B / R22
    # AM-GM
    c = c + kv * (Khi / (LAMM * 2.0)).hi if False else c + kv * (IA(Khi) / (LAMM * 2.0))
    fac = om / (R22 * LAMM * 2.0)
    c = c + (at * (A0B * 2.0) + tabs(at) * Xi0) * fac; k = k + A0B * A0B * fac
    # M own: Tq.A (|XB| + Xi) Kp / V  and  |X| Phi0/(R22 om) <= (GAMMA X^2 + Phi0^2/GAMMA)/(2 R22 om)  [GAMMA X^2 is in cX]
    c = c + Q['kpv'] * (TqAV * (tabs(XB) + Xi))
    g2 = (om * R22 * (2.0 * GAMMA)).inv()
    c = c + (zt * (P0B * 2.0) + tabs(zt) * Z) * g2; k = k + P0B * P0B * g2
    # det E: Tq.w11 R/V <= Tq.w11/V Wmax M1hi M1 ; R0/4 <= (3/4) W00 mu0 sum_{U0 \ j0} m ; Y^2 <= Yhi Y ; Rm0 <= Wm0 mu0 M1
    c = c + tvecconst(M.OFFm1) * (Tq['w11'] / V * (BC['WPh'] * M1hi)) if M1hi > 0 else c
    c = c + tvecconst(U0x) * (m0 * (BC['W00'] * 0.75))
    c = c + Q['yv'] * Yhi
    c = c + tvecconst(M.OFFm1) * (m0 * Tq['w10'] * SQ2 / V * BC['Wm0'])
    F4 = MU0 * 4.0
    return Q['cL'] * F4 - c * KILL, Q['constL'] * F4 - k * KILL


def mode2_parts(Q, cost, const, basis):
    """F_B (T1), rc (T1 vector over all profiles) for the basis (sorted tuple)."""
    bi = L.binv(basis); _, BI, _ = bi
    b = M.tstack(Q['b']) if hasattr(M, 'tstack') else L.tstack(Q['b'])
    cB = T1(IA(np.asarray(cost.v.lo)[list(basis)], np.asarray(cost.v.hi)[list(basis)]),
            IA(np.asarray(cost.g.lo)[:, list(basis)], np.asarray(cost.g.hi)[:, list(basis)]))
    xB = tlin(BI, b)
    y = tlin(IA(BI.lo.T.copy(), BI.hi.T.copy()), cB)            # (y_-1, y_0, y_1, y_a, y_b)
    F = tsum(cB * xB) + const
    AT = M.AT
    rc = cost - tlin(AT, y)
    return F, rc, xB


def kink_check(lo, hi, side, s1, rec=None):
    """Lemma KINK-K on one box of (alpha, lam, dp, dm) (dp >= 0 if side > 0, <= 0 if side < 0).
    Returns dict(ok, rcmin, P0, slope, eta, ranges, duals) ; with rec the recorded range duals and eta are re-used."""
    import bk16_lb as B
    bx = B.Box(lo, hi, pm=True); lev = (0.0, s1); zero = tconst(0.0)
    rg = {}; rd = {}
    for key in LOCAL_KEYS:
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        d1 = rec['rd'][key][0] if rec else None; d2 = rec['rd'][key][1] if rec else None
        a, d1 = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1, slopes=True)
        b_, d2 = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2, slopes=True)
        if a is None or b_ is None: return dict(ok=False, why='range')
        rg[key] = (a, b_); rd[key] = (d1, d2)
    Fb = max(abs(rg['fi'][0]), abs(rg['fi'][1]))
    if not hinges_inactive(bx, Fb): return dict(ok=False, why='hinge', rg=rg)
    lcb = local_cost(bx.Qb, rg); lcp = local_cost(bx.Qp, rg)
    if lcb is None or lcp is None: return dict(ok=False, why='phibar', rg=rg)
    basis = BPLUS if side > 0 else BMINUS
    F, rcb, xB = mode2_parts(bx.Qb, lcb[0], lcb[1], basis)
    Fp, rcp_, _ = mode2_parts(bx.Qp, lcp[0], lcp[1], basis)
    rcI = centered(rcp_, rcb, bx)
    rc = T1(rcI, rcb.g)
    rclo = np.asarray(rc.v.lo)
    if rec is not None: eta = rec['eta']
    elif side > 0:
        cand = [rclo[j] / DELTA[j] for j in range(NP) if j not in basis and DELTA[j] > 0 and rclo[j] < 0]
        eta = float(dn(min(cand))) * 1.0000001 if cand else 0.0
        eta = min(eta, 0.0)
    else:
        eta = 0.0
    # rc' = rc - eta delta  (eta exact double)
    rcp = rc.v - IA(DELTA) * eta
    nb = [j for j in range(NP) if j not in basis and j != ij0p]
    rcmin = float(np.asarray(rcp.lo)[nb].min())
    if side > 0 and ijp in basis:
        rcmin = min(rcmin, float(dn(-1.5 * eta)) if eta < 0 else 0.0) if False else rcmin
    P0 = max(0.0, -float(np.asarray(rcp.lo)[ij0p]))
    # slope along dp (variable index 2) of F + eta dp
    g = IA(F.g.lo[2], F.g.hi[2]) + eta
    slope = float(g.lo) if side > 0 else -float(g.hi)
    ok = rcmin >= 0 and slope >= 0
    return dict(ok=bool(ok), rcmin=rcmin, P0=P0, slope=slope, eta=eta, rd=rd, rg={k_: list(v) for k_, v in rg.items()})
