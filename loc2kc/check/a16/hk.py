"""hk.py : own model of h_K(p, t) (Lemma LOC2-K), written from the Delta_u (version 2) restricted to
m4(p, t) with the local replacements, independent of bk16_loc2.py / tay2.py / ivarith.py.
Generic in the number type: mp.mpf (high precision floats) or oad.D2 (rigorous mpmath.iv second-order AD).
Units: 4 mu0 s (h = 4 mu0 Lfun(m4) - KILL * D(m4))."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'a13'))
import mpmath as mp
from mpmath import iv

PA, PJ0, PB, PC, PJP = (-1, 1, 1), (0, 1, -1), (1, -1, 1), (1, 1, -1), (1, -1, -1)


def consts_mp(dps=50, kill='0.08126848580'):
    mp.mp.dps = dps
    s2 = mp.sqrt(2); rho = 2 * s2 / 3; mu0 = (1 + 2 * s2) / 4; D0 = 2 + 2 * rho
    KAP = mp.mpf('4.478385650326709971336669'); R = (KAP - 1) / (4 * mu0)
    return dict(s2=s2, rho=rho, mu0=mu0, D0=D0, R=R, kill=mp.mpf(kill), eps1=mp.mpf('1.02'), lamM=mp.mpf('0.6'),
                gam=mp.mpf('0.05'), one=mp.mpf(1), sqrt=mp.sqrt)


def consts_iv(kill='0.08126848580'):
    iv.prec = 90
    s2 = iv.sqrt(iv.mpf(2)); rho = 2 * s2 / 3; mu0 = (1 + 2 * s2) / 4; D0 = 2 + 2 * rho
    KAP = iv.mpf(['4.478385650326169', '4.478385650327289']); R = (KAP - 1) / (4 * mu0)
    return dict(s2=s2, rho=rho, mu0=mu0, D0=D0, R=R, kill=iv.mpf(kill), eps1=iv.mpf('1.02'), lamM=iv.mpf('0.6'),
                gam=iv.mpf('0.05'), one=iv.mpf(1), sqrt=lambda x: x.sqrt() if hasattr(x, 'h') or hasattr(x, 'g') else iv.sqrt(x))


def gen(pr, s2):
    return [s2 ** 0, s2 ** (-pr[0]), s2 ** (-pr[1]), s2 ** (-pr[2])]


def W(p, q): return p[0] * q[1] - p[1] * q[0]


def rec(x): return x.inv() if hasattr(x, 'h') else 1 / x


def h_K(al, la, t, C, parts=False):
    s2, sqrt = C['s2'], C['sqrt']
    q = 1 - la; S = q * (2 - al) + la
    m1 = al * q / S; m0 = la / S; p1 = 2 * (1 - al) * q / S
    mB = p1 * 0.5 + t; mC = p1 * 0.5 - t
    masses = [(m1, PA), (m0, PJ0), (mB, PB), (mC, PC)]
    g = {pr: gen(pr, s2) for pr in (PA, PJ0, PB, PC, PJP)}

    def G(x, y):
        acc = None
        for m, pr in masses:
            term = m * (g[pr][x] * g[pr][y]); acc = term if acc is None else acc + term
        return acc
    d = [sqrt(G(i, i)) for i in range(4)]

    def gh(x, y): return G(x, y) / (d[x] * d[y])
    e0, eua, eub, hva, hvb, hab = gh(0, 1), gh(0, 2), gh(0, 3), gh(1, 2), gh(1, 3), gh(2, 3)
    D0, rho = C['D0'], C['rho']

    def l(e1, e2, hh): return (1 + e1 + e2 + hh) / D0 - (e1 + e2 - 2 * rho) * (4 / (D0 * D0))
    Lf = l(eua, eub, hab) * C['R'] + l(e0, eua, hva) + l(e0, eub, hvb) - (C['R'] + 2) * C['mu0']
    # ---- kill part (own derivation): u-star quantities in the frame G_uu = 1 (sum m = 1)
    na = rec(d[2]); nb = rec(d[3]); V = G(1, 1); om = d[1]

    def P(pr): return (1 + na * g[pr][2], 1 + nb * g[pr][3])
    P0 = P(PA); s0 = P0[0] + P0[1]

    def ph(pr):
        p = P(pr); return (p[0] + p[1]) / s0
    F = (mB * P(PB)[0] + mC * P(PC)[0], mB * P(PB)[1] + mC * P(PC)[1])
    X = W(F, P0)                                        # X = F ^ P0
    phi = mB * ph(PB) + mC * ph(PC)
    aj0 = W(P0, P(PJ0)); A0 = m0 * aj0 * aj0
    Phi0 = m0 * ph(PJ0) * ph(PJ0)
    Phib = ph(PJP)
    # thresholds at the mirror marginal q = (mu_1/2, mu_0, 2 mu_-1)/V: face star there, built from its definition
    Vm = 2 * m1 + m0 + p1 * 0.5
    qm1, q0, qp1 = p1 * 0.5 / Vm, m0 / Vm, 2 * m1 / Vm
    dF = qm1 * 0.5 + (qp1 * 0.5) * 2.5 + (q0 * 0.5) * 2.5     # G_aa = G_bb of the face star
    nF = rec(sqrt(dF))

    def PF(pr): return (1 + nF * g[pr][2], 1 + nF * g[pr][3])
    sF = PF(PA)[0] + PF(PA)[1]
    phiF = (qp1 * 0.5) * ((PF(PB)[0] + PF(PB)[1]) / sF) + (qp1 * 0.5) * ((PF(PC)[0] + PF(PC)[1]) / sF)
    Tqphi = phiF * C['eps1']
    R22 = 2 * s2; L = C['lamM']; gam = C['gam']
    cX = 0.25 + Phib * Phib * (L * 0.5) + rec(om * (2 * R22)) * (L + gam)
    D = cX * X * X + phi * A0 / (om * R22) + om * Tqphi * A0 / R22 + om * A0 * A0 / (2 * R22 * L) + Phi0 * Phi0 / (om * (2 * gam * R22))
    h = Lf * (4 * C['mu0']) - D * C['kill']
    if parts: return dict(h=h, Lf=Lf, D=D, X=X, A0=A0, Phi0=Phi0, phi=phi, Tqphi=Tqphi, cX=cX, V=V, Phib=Phib, phiF=phiF)
    return h
