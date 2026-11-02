"""bk14_lpk.py: slice LP with the linear part of the kill subtracted (prototype).  NUMERICAL.
tauLP(p) = min_d [ const + min_{slice} (c - kill_lin) . m ] (s units), with kill_lin per profile from Delta_u's own terms."""
import numpy as np, math, sys
from scipy.optimize import linprog, minimize
import bk13_core as K
from bk14_delta import Kill, mirror_mu, Vof, face_star
from bk14_sym import ustar_scal
from bk14_sides import wedge
RHO = 2 * math.sqrt(2) / 3; N0 = math.sqrt(2 + 2 * RHO); MU0 = K.MU0
MT = 0.27658360688237347; KE2 = 0.28070193727164739; CT = math.cos(2 * math.pi * (MT - KE2 / 2) / 3)
kap = 1 / (N0 ** 4 * 0.80)
PU = K.PU
eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05; x0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
KL = Kill(eps=eps, x0=x0, x1=x0)
g = lambda t: 2.0 ** (-t / 2)
def kill_lin(mu, sa, sb):
    """per-profile linear kill coefficients (s units) on the slice (sa = d_a^-1/2, sb = d_b^-1/2)."""
    P = np.array([[1 + sa * g(p[1]), 1 + sb * g(p[2])] for p in PU]); P0 = np.array([1 + sa * g(1), 1 + sb * g(1)])
    a = wedge(P0[None, :], P); ph = (P[:, 0] + P[:, 1]) / (P0[0] + P0[1])
    V = Vof(mu); w = math.sqrt(V); ms = mirror_mu(mu); Ts = KL.thresholds(ms)
    iA, iB = PU.index((1, -1, 1)), PU.index((1, 1, -1))
    phi_face = mu[2] * (ph[iA] + ph[iB]) / 2
    c = np.zeros(len(PU))
    for j, p in enumerate(PU):
        if p[0] == -1: c[j] = (Ts['U'] + Ts['A'] * phi_face / V + x0) * abs(a[j])
        if p[0] == 0: c[j] = (phi_face / w + w * Ts['phi'] + w * x0) * a[j] ** 2 / (2 * math.sqrt(2))
    return kap * c / (4 * MU0)
A_UB = None
def LPk(mu, sa, sb, withN=True):
    c, const = K.lp_data(mu, sa, sb)
    c2 = c - kill_lin(mu, sa, sb)
    b = np.array([mu[0], mu[1], mu[2], sa ** -2, sb ** -2])
    # N-star constraint, Lemma-L form: l_a + l_b + l_ab <= 3 c_Theta  (l linear on the slice): l-sum = constL + cL . m
    V, U = K.VU(mu); e0 = U / math.sqrt(V); iv = 1 / math.sqrt(V)
    D0 = K.D0
    # l_a = (1 + e0 + eua + hva)/D0 - 4/D0^2 (e0 + eua - 2 rho); eua = sa G_ua, hva = iv sa G_va; l_ab = (1 + eua + eub + hab)/D0 - 4/D0^2 (eua + eub - 2 rho)
    GA_, GB_, GV_ = K.GA, K.GB, K.GV
    cL = (sa * GA_ + sb * GB_) * (2 / D0 - 8 / D0 ** 2) + iv * GV_ * (sa * GA_ + sb * GB_) / D0 + sa * sb * GA_ * GB_ / D0
    constL = 3 / D0 + 2 * e0 / D0 - (4 / D0 ** 2) * (2 * e0 - 6 * RHO)
    kw = {}
    if withN: kw = dict(A_ub=cL[None, :], b_ub=[3 * CT - constL])
    r = linprog(c2, A_eq=K.A_EQ, b_eq=b, bounds=[(0, None)] * len(PU), method='highs', **kw)
    if r.status != 0: return None, None
    return r.fun + const, r.x
def tauLP(mu, withN=True):
    best = (np.inf, None)
    for z0 in ([0, 0], [0.05, -0.05], [-0.05, 0.05], [0.1, 0.1], [-0.1, -0.1]):
        f = lambda z: (LPk(mu, math.exp(z[0]), math.exp(z[1]), withN)[0] or 9.0)
        r = minimize(f, np.array(z0, float), method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-13, maxiter=2000))
        if r.fun < best[0]: best = (r.fun, np.exp(r.x))
    return best
if __name__ == '__main__':
    pts = [(0.5, 0.0), (0.49, 0.0), (0.48, 0.0), (0.47, 0.0), (0.45, 0.0), (0.42, 0.0), (0.4, 0.0), (0.37, 0.0),
           (0.5, 0.001), (0.5, 0.005), (0.48, 0.005), (0.5, 0.02), (0.45, 0.02), (0.5, 0.1), (0.4, 0.1), (0.5, 0.3)]
    for (al, la) in pts:
        mu = K.mu_of(al, la); ms = mirror_mu(mu)
        t1, d1 = tauLP(mu); t2, d2 = tauLP(ms)
        s1 = K.sigmaLP(mu)[0] if False else None
        print('p=(%.3f,%.3f) logV %.3f: tauLP(p) %.4e  tauLP(p^s) %.4e  sum %.4e  d %s %s' % (al, la, math.log(Vof(mu)), t1, t2, t1 + t2, np.round(d1, 4), np.round(d2, 4)), flush=True)
