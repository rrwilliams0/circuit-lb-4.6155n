"""bk13_L3.py: the coupled local inequality (L3) on Q.
h(p, t) = Lfun(m4(p, t)),  m4 = mu_-1 e_A + mu_0 e_j0 + (mu_1/2 + t) e_B + (mu_1/2 - t) e_C,  p = (1/2 + a, lam).
Psi(a, lam, t, t') = h((1/2 + a, lam), t) + h((1/2 - a, lam), t').
Second-order Taylor at 0 with Lagrange remainder; Psi(0) = 0 and dPsi/da = dPsi/dt = dPsi/dt' = 0 at 0 EXACTLY (proved in the
report by the tree identity and the a <-> b symmetry); dPsi/dlam(0) and the Hessian over the box by interval AD (Tay2)."""
import sys, json, math, itertools
import numpy as np
from ivarith import IA, dn, up
from tay2 import Tay2
import bk13_lb as L
from bk13_lb import SQ2, ISQ2, RHO, MU0, D0, R

def gvec(pr):
    k, ta, tb = pr; return [IA(1.0), L.g_of(k), L.g_of(ta), L.g_of(tb)]
PA, PJ0, PB, PC = gvec((-1, 1, 1)), gvec((0, 1, -1)), gvec((1, -1, 1)), gvec((1, 1, -1))

def h_tay(al, la, t):
    """al, la, t: Tay2 objects. Returns Tay2 of h."""
    q = 1.0 - la; S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    ms = [(m1, PA), (m0, PJ0), (p1 * 0.5 + t, PB), (p1 * 0.5 - t, PC)]
    def G(x, y):
        acc = None
        for m, g in ms:
            term = m * (g[x] * g[y])
            acc = term if acc is None else acc + term
        return acc
    Guu = G(0, 0)
    sd = [Guu.sqrt(), G(1, 1).sqrt(), G(2, 2).sqrt(), G(3, 3).sqrt()]
    def gh(x, y): return G(x, y) / (sd[x] * sd[y])
    e0, eua, eub, hva, hvb, hab = gh(0, 1), gh(0, 2), gh(0, 3), gh(1, 2), gh(1, 3), gh(2, 3)
    iD = D0.inv(); c4 = (D0 * D0).inv() * 4.0; r2 = RHO * 2.0
    def lL(e1, e2, hh): return (e1 + e2 + hh + 1.0) * iD - (e1 + e2 - r2) * c4
    return lL(eua, eub, hab) * R + lL(e0, eua, hva) + lL(e0, eub, hvb) - (R + 2.0) * MU0

def psi_tay(lo, hi):
    n = 4
    a, la, t, tp = [Tay2.var(i, n, lo[i], hi[i]) for i in range(n)]
    return h_tay(a + 0.5, la, t) + h_tay(0.5 - a, la, tp)

def single_tay(lo, hi):
    n = 3
    a, la, t = [Tay2.var(i, n, lo[i], hi[i]) for i in range(n)]
    return h_tay(a + 0.5, la, t)

def specnorm_bound(rad):
    return float(np.sqrt((rad ** 2).sum()))   # Frobenius >= spectral

if __name__ == '__main__':
    ra, rl, T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03, 0.005, (-0.035, 0.036)
    P0 = 3.934e-3 * 1.01        # from bk13_L2 (W+ = 0.01, W- = 0.1), rounded up
    # (i) gradient at 0 (point evaluation)
    z0 = np.zeros(4)
    P = psi_tay(z0, z0)
    print('Psi(0) enclosure', P.v.lo, P.v.hi); print('grad(0) lo', P.g.lo, '\n        hi', P.g.hi)
    glam = float(P.g.lo[1])
    # (ii) Hessian over the box
    lo = np.array([0.0, 0.0, T[0], T[0]]); hi = np.array([ra, rl, T[1], T[1]])
    B = psi_tay(lo, hi)
    H = B.h; Hmid = 0.5 * (H.lo + H.hi); Hrad = up(0.5 * (H.hi - H.lo))
    print('Hessian mid\n', np.round(Hmid, 4)); print('Hessian rad\n', np.round(Hrad, 4))
    X = [0, 2, 3]
    lam_mid = float(np.linalg.eigvalsh(Hmid[np.ix_(X, X)]).min())
    cX = lam_mid - specnorm_bound(Hrad[np.ix_(X, X)]) - 1e-12
    hX = float(np.sqrt(sum(max(abs(H.lo[1, i]), abs(H.hi[1, i])) ** 2 for i in X)))
    hll = max(abs(H.lo[1, 1]), abs(H.hi[1, 1]))
    Smin = (1 - rl) * (2 - 0.5 - ra) + 0.0
    G = glam - 2 * P0 / Smin - 0.5 * hll * rl
    print('c_X (min eig over the interval Hessian, x = (a,t,t\')) >= %.5f ; |H_lam,x| <= %.5f ; |H_lamlam| <= %.4f' % (cX, hX, hll))
    print('G = 2 dh/dlam - 2 P0/Smin - |H_ll| rl/2 = %.5f ; need rl <= 2 cX G / hX^2 = %.5f (rl = %.4f)' % (G, 2 * cX * G / hX ** 2, rl))
    ok = cX > 0 and G > 0 and rl <= 2 * cX * G / hX ** 2
    print('L3 (coupled, origin box) CLOSED' if ok else 'L3 NOT closed')
    # (iii) single-star lower bound on Q x T: h - P0 mu0 >= -tau0
    lo3 = np.array([-ra, 0.0, T[0]]); hi3 = np.array([ra, rl, T[1]])
    z = np.zeros(3); S0 = single_tay(z, z); S1 = single_tay(lo3, hi3)
    g = S0.g; Hs = S1.h
    # h(z) >= h(0) + g.z + 1/2 z^T H z  ; crude: min over box of linear part + sum_il min(H_il z_i z_l)
    lin = sum(min(float(g.lo[i]) * lo3[i], float(g.lo[i]) * hi3[i], float(g.hi[i]) * lo3[i], float(g.hi[i]) * hi3[i]) for i in range(3))
    quad = 0.0
    for i in range(3):
        for l in range(3):
            prods = [x * y for x in (lo3[i], hi3[i]) for y in (lo3[l], hi3[l])] if i != l else [0.0, max(lo3[i] ** 2, hi3[i] ** 2)]
            quad += 0.5 * min(float(Hs.lo[i, l]) * min(prods), float(Hs.lo[i, l]) * max(prods), float(Hs.hi[i, l]) * min(prods), float(Hs.hi[i, l]) * max(prods))
    single = float(S0.v.lo) + lin + quad - P0 * rl / Smin
    print('single star on Q x T: h - P0 mu0 >= %.3e (crude)' % single)
    json.dump(dict(ra=ra, rl=rl, T=T, P0=P0, glam=glam, cX=cX, hX=hX, hll=hll, G=G, ok=bool(ok), single=single,
                   Hlo=H.lo.tolist(), Hhi=H.hi.tolist(), g0lo=P.g.lo.tolist(), g0hi=P.g.hi.tolist()), open('L3_cert.json', 'w'), indent=1)
