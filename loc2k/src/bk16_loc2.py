"""bk16_loc2.py : h_K(p, t) = 4 mu0 Lfun(m4(p, t)) - KILL Dface(p, t) in second-order interval AD (Tay2), and
Lemma LOC2-K:  Psi_K(a, lam, t, t') = h_K(1/2 + a, lam, t) + h_K(1/2 - a, lam, t') - 2 P0 lam / S_min >= 0 on K, = 0 only at 0;
single star h_K - P0 mu_0 >= -tau0 on Q x T.   Units: 4 mu0 s (the slack units of bk16_model).

Dface(p, t) = cX X^2 + phi A0/(om R22) + om Tq.phi A0/R22 + om A0^2/(2 R22 lamM) + Phi0^2/(2 GAMMA R22 om) evaluated at m4,
cX = 1/4 + lamM Phibar^2/2 + (lamM + GAMMA)/(2 R22 om), Phibar = ph(jp).  It equals the local cost DL of bk16_loc at m4."""
import numpy as np
from ivarith import IA, dn, up
from tay2 import Tay2
import bk13_lb as L
from bk13_lb import SQ2, ISQ2, RHO, MU0, D0, R
import bk16_model as M
from bk16_loc import GAMMA


def gv(pr):
    k, ta, tb = pr; return [IA(1.0), L.g_of(k), L.g_of(ta), L.g_of(tb)]


PA, PJ0, PB, PC = gv((-1, 1, 1)), gv((0, 1, -1)), gv((1, -1, 1)), gv((1, 1, -1))
GJP = gv((1, -1, -1))


def h_kill(al, la, t, parts=False):
    q = 1.0 - la; S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    mB = p1 * 0.5 + t; mC = p1 * 0.5 - t
    ms = [(m1, PA), (m0, PJ0), (mB, PB), (mC, PC)]

    def G(x, y):
        acc = None
        for m, g in ms:
            term = m * (g[x] * g[y]); acc = term if acc is None else acc + term
        return acc
    Guu = G(0, 0)
    sd = [Guu.sqrt(), G(1, 1).sqrt(), G(2, 2).sqrt(), G(3, 3).sqrt()]

    def gh(x, y): return G(x, y) / (sd[x] * sd[y])
    e0, eua, eub, hva, hvb, hab = gh(0, 1), gh(0, 2), gh(0, 3), gh(1, 2), gh(1, 3), gh(2, 3)
    iD = D0.inv(); c4 = (D0 * D0).inv() * 4.0; r2 = RHO * 2.0

    def lL(e1, e2, hh): return (e1 + e2 + hh + 1.0) * iD - (e1 + e2 - r2) * c4
    Lf = lL(eua, eub, hab) * R + lL(e0, eua, hva) + lL(e0, eub, hvb) - (R + 2.0) * MU0
    # kill at m4 (G_uu = 1 since the masses sum to 1)
    na = G(2, 2).sqrt().inv(); nb = G(3, 3).sqrt().inv()
    V = G(1, 1); om = V.sqrt()

    def P(g): return (na * g[2] + 1.0, nb * g[3] + 1.0)
    P0 = P(PA); PBv = P(PB); PCv = P(PC); PJ = P(PJ0); PP = P(GJP)

    def wedge(p, r): return p[0] * r[1] - p[1] * r[0]
    s0 = P0[0] + P0[1]
    X = -(mB * wedge(P0, PBv) + mC * wedge(P0, PCv))
    phi = mB * (PBv[0] + PBv[1]) / s0 + mC * (PCv[0] + PCv[1]) / s0
    aj0 = wedge(P0, PJ); A0 = m0 * aj0 * aj0
    phj0 = (PJ[0] + PJ[1]) / s0; Phi0 = m0 * phj0 * phj0
    Phib = (PP[0] + PP[1]) / s0
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
    if parts: return h, Lf, D, X
    return h


def psi_tay(lo, hi):
    a, la, t, tp = [Tay2.var(i, 4, lo[i], hi[i]) for i in range(4)]
    return h_kill(a + 0.5, la, t) + h_kill(0.5 - a, la, tp)


def single_tay(lo, hi):
    a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
    return h_kill(a + 0.5, la, t)
