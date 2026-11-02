"""bk17_fbp.py : the basic value F_B+(z) of the local relaxation for the basis B+ = {A, B, C, j0, jp}, in second-order
interval AD (Tay2), z = (alpha, lam, dp, dm) (pm coordinates of Lemma KINK).

On B+ every range-dependent and box-constant-dependent term of the local cost (bk16_loc.local_cost) has coefficient exactly 0
(xt, at, ft, zt vanish on BREF = B+ by construction; kv, kpv vanish on A since a(A) = P0 ^ P_A = 0 and ph(A) = 1; OFFm1, U0x
and the Ybar weights pq vanish on A, j0 and class 1).  Hence, exactly,
    F_B+(z) = 4 mu0 Lfun(x_B) - KILL Dloc(x_B),  x_B = B+^{-1} b(z),
    Dloc(m) = cX X^2 + phi A0/(om R22) + om Tq.phi A0/R22 + om A0^2/(2 R22 lamM) + Phi0^2/(2 GAMMA R22 om),
the face function of bk16_loc2.h_kill with the jp column added.  Used for the slope condition of Lemma KINK-K on side +:
a centred enclosure of dF_B+/ddp (point gradient at the centre + interval Hessian over the box)."""
import numpy as np
from ivarith import IA, dn, up
from tay2 import Tay2
import bk13_lb as L
from bk13_lb import SQ2, ISQ2, RHO, MU0, D0, R, PU
import bk16_model as M
import bk16_loc as LC
from bk16_loc import GAMMA

BP = LC.BPLUS                                     # sorted indices
_, BIp, _ = L.binv(BP)
GVEC = {j: [IA(1.0), L.g_of(PU[j][0]), L.g_of(PU[j][1]), L.g_of(PU[j][2])] for j in BP}
CLASS = {j: int(PU[j][0]) for j in BP}


def fbplus(al, la, dp, dm, parts=False):
    q = 1.0 - la; S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    tau = m1 * (-1.5) + 2.5
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    b = [m1, m0, p1, da, db]
    xs = {}
    for r, j in enumerate(BP):
        acc = None
        for k in range(5):
            c = IA(BIp.lo[r, k], BIp.hi[r, k])
            if c.lo == 0 and c.hi == 0: continue
            t = b[k] * c; acc = t if acc is None else acc + t
        xs[j] = acc
    ms = [(xs[j], GVEC[j]) for j in BP]

    def G(x, y):
        acc = None
        for m, g in ms:
            term = m * (g[x] * g[y]); acc = term if acc is None else acc + term
        return acc
    sd = [G(0, 0).sqrt(), G(1, 1).sqrt(), G(2, 2).sqrt(), G(3, 3).sqrt()]

    def gh(x, y): return G(x, y) / (sd[x] * sd[y])
    e0, eua, eub, hva, hvb, hab = gh(0, 1), gh(0, 2), gh(0, 3), gh(1, 2), gh(1, 3), gh(2, 3)
    iD = D0.inv(); c4 = (D0 * D0).inv() * 4.0; r2 = RHO * 2.0

    def lL(e1, e2, hh): return (e1 + e2 + hh + 1.0) * iD - (e1 + e2 - r2) * c4
    Lf = lL(eua, eub, hab) * R + lL(e0, eua, hva) + lL(e0, eub, hvb) - (R + 2.0) * MU0
    na = G(2, 2).sqrt().inv(); nb = G(3, 3).sqrt().inv()
    V = G(1, 1); om = V.sqrt()

    def P(j): g = GVEC[j]; return (na * g[2] + 1.0, nb * g[3] + 1.0)
    iA, iB, iC, ij0, ijp = M.iA, M.iB, M.iC, M.ij0, M.ijp
    P0 = P(iA)

    def wedge(p, r): return p[0] * r[1] - p[1] * r[0]
    s0 = P0[0] + P0[1]
    X = None; phi = None
    for j in (iB, iC, ijp):
        Pj = P(j); t = xs[j] * wedge(P0, Pj); X = t if X is None else X + t
        u = xs[j] * (Pj[0] + Pj[1]) / s0; phi = u if phi is None else phi + u
    X = -X
    PJ = P(ij0); aj0 = wedge(P0, PJ); A0 = xs[ij0] * aj0 * aj0
    phj0 = (PJ[0] + PJ[1]) / s0; Phi0 = xs[ij0] * phj0 * phj0
    PP = P(ijp); Phib = (PP[0] + PP[1]) / s0
    Dq = (al + 1.0) * q + la
    qm1 = (1.0 - al) * q / Dq
    n = (qm1 * (-0.75) + 1.25).sqrt().inv()
    qm1c = al * q * 2.0 / Dq
    Tqphi = qm1c * (n * (SQ2 * 1.5) + 2.0) / (n * SQ2 + 2.0) * M.EPS1
    R22 = SQ2 * 2.0
    cX = (om * R22 * 2.0).inv() * (M.LAMM + GAMMA) + Phib * Phib * (M.LAMM * 0.5) + 0.25
    D = cX * X * X + phi * A0 / (om * R22) + om * Tqphi * A0 / R22 + om * A0 * A0 / (R22 * M.LAMM * 2.0) \
        + Phi0 * Phi0 / (om * R22 * (2.0 * GAMMA))
    h = Lf * (MU0 * 4.0) - D * M.KILL
    if parts: return h, xs
    return h


def slope_lo(elo, ehi, var=2):
    """certified lower bound of dF_B+/dz_var over the box [elo, ehi]: gradient at the centre (point Tay2) plus the
    interval Hessian row over the box times the half-widths (mean value theorem for the gradient)."""
    elo = np.asarray(elo, float); ehi = np.asarray(ehi, float); zc = 0.5 * (elo + ehi)
    tp = fbplus(*[Tay2.var(i, 4, zc[i], zc[i]) for i in range(4)])
    tb = fbplus(*[Tay2.var(i, 4, elo[i], ehi[i]) for i in range(4)])
    g = IA(tp.g.lo[var], tp.g.hi[var])
    for i in range(4):
        Dz = IA(dn(elo[i] - zc[i]), up(ehi[i] - zc[i]))
        g = g + IA(tb.h.lo[var, i], tb.h.hi[var, i]) * Dz
    # also intersect with the natural interval gradient
    return max(float(g.lo), float(tb.g.lo[var])), min(float(g.hi), float(tb.g.hi[var]))
