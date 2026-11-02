"""bk14_delta.py: Delta_u(m_u; p) - the u-share of the separable kill bound (|D(W)| units).  NUMERICAL prototype.
p is the t_v-marginal (mu_-1, mu_0, mu_1), sum 1; the mirror marginal is mirror_mu(p)."""
import numpy as np, math, itertools
from bk14_sym import *
from bk14_sides import PU, KU, wedge
S2 = math.sqrt(2); R22 = 2 * S2
def mirror_mu(mu): w = np.array([mu[2] / 2, mu[1], 2 * mu[0]]); return w / w.sum()
def Vof(mu): return 2 * mu[0] + mu[1] + mu[2] / 2
def face_star(mu):
    m = np.zeros(len(PU))
    m[PU.index((-1, 1, 1))] = mu[0]
    m[PU.index((1, -1, 1))] = m[PU.index((1, 1, -1))] = mu[2] / 2
    m[PU.index((0, 1, -1))] = m[PU.index((0, -1, 1))] = mu[1] / 2
    return m
# crude profile maxima over the ratio box |log d_a|, |log d_b| <= LB (n = d^{-1/2})
LB = 0.29
def prof_max():
    best = dict(a_m1=0, a_1=0, a2_0=0, phm1=0, ph1=0, wPh=0, wPhP0=0, w11=0, w10=0, ph0sq=0, pq=0)
    for la, lb in itertools.product(np.linspace(-LB, LB, 13), repeat=2):
        na, nb = math.exp(-la / 2), math.exp(-lb / 2)
        g = lambda t: 2.0 ** (-t / 2)
        P = np.array([[1 + na * g(p[1]), 1 + nb * g(p[2])] for p in PU]); P0 = P[PU.index((-1, 1, 1))]
        a = wedge(P0[None, :], P); ph = (P[:, 0] + P[:, 1]) / (P0[0] + P0[1]); Ph = P - P0
        c1, cm, c0 = KU == 1, KU == -1, KU == 0
        i1, im, i0 = np.where(c1)[0], np.where(cm)[0], np.where(c0)[0]
        best['a_m1'] = max(best['a_m1'], np.abs(a[cm]).max()); best['a_1'] = max(best['a_1'], np.abs(a[c1]).max())
        best['a2_0'] = max(best['a2_0'], (a[c0] ** 2).max()); best['phm1'] = max(best['phm1'], np.abs(ph[cm] - 1).max())
        best['ph1'] = max(best['ph1'], ph[c1].max()); best['ph0sq'] = max(best['ph0sq'], (ph[c0] ** 2).max())
        best['wPh'] = max(best['wPh'], max(abs(wedge(Ph[i], Ph[j])) for i in im for j in im))
        best['wPhP0'] = max(best['wPhP0'], max(abs(wedge(Ph[i], P[j])) for i in im for j in i0))
        best['w11'] = max(best['w11'], max(abs(wedge(P[i], P[j])) for i in i1 for j in i1))
        best['w10'] = max(best['w10'], max(abs(wedge(P[i], P[j])) for i in i1 for j in i0))
        best['pq'] = max(best['pq'], max(abs(wedge(P[i], Ph[j])) for i in i1 for j in im))
    return best
PM_GRID = None  # float grid version prof_max() kept for reference
# CERTIFIED upper bounds (bk14_supcert.py, mpmath intervals, 48x48 cover of the box), rounded up:
PM = dict(a_m1=1.5035305, a_1=1.5035305, a2_0=2.2606040, phm1=0.4537150, ph1=1.4537150, wPh=0.6803371, wPhP0=2.1798184, w11=3.6574014, w10=3.6574014, pq=2.1798184)
def sups(q):
    """crude sups over stars at marginal q (any masses, ratios in the box)."""
    Vq = Vof(q)
    return dict(K=q[0] * PM['a_m1'], A0=q[1] * PM['a2_0'], Zphi=q[2] * PM['ph1'] * q[0] * PM['a_m1'] / Vq,
                ZX=q[2] * PM['a_1'] * q[0] * PM['phm1'] / Vq, ZR=q[0] ** 2 * PM['wPh'] / Vq, ZRm=S2 * q[0] * q[1] * PM['wPhP0'] / Vq)
class Kill:
    def __init__(self, eps=0.05, x0=0.01, x1=0.01):
        self.eps, self.x0, self.x1 = eps, x0, x1
    def thresholds(self, mu):
        f = ustar_scal(face_star(mu)); e = 1 + self.eps
        return dict(U=e * f['phi'] * f['Abar'], A=e * f['Abar'], phi=e * f['phi'], w11=e * f['w11'], w10=e * f['w10'])
    def delta_u(self, m_, mu, parts=False):
        s = ustar_scal(m_); ms = mirror_mu(mu); V = Vof(mu); w = math.sqrt(V)
        Tp, Ts = self.thresholds(mu), self.thresholds(ms); Sp = sups(ms)
        x0, x1 = self.x0, self.x1
        own = (s['X'] ** 2 / 4 + Ts['U'] * s['K'] + Ts['A'] * s['phi'] * s['K'] / V + x0 * s['K']
               + (s['phi'] / w + w * Ts['phi'] + w * x1) * s['A0'] / R22 + Ts['A'] * abs(s['X']) * s['Kp'] / V
               + abs(s['X']) * s['Phi0'] / (R22 * w) + Ts['w11'] * s['R'] / V + s['R0'] / 4 + s['Y'] ** 2 + S2 / V * Ts['w10'] * s['Rm0'])
        pos = lambda z: max(z, 0.0)
        hin = (pos(s['phi'] * s['Abar'] - Tp['U']) * Sp['K'] + pos(s['Abar'] - Tp['A']) * (Sp['Zphi'] + Sp['ZX'])
               + pos(s['phi'] - Tp['phi']) * Sp['A0'] / (R22 * w) + pos(abs(s['X']) * s['Phib'] - x0) * Sp['K']
               + pos(abs(s['X']) - x1) * Sp['A0'] / (R22 * w) + pos(s['w11'] - Tp['w11']) * Sp['ZR'] + pos(s['w10'] - Tp['w10']) * Sp['ZRm'])
        return (own, hin) if parts else own + hin
if __name__ == '__main__':
    print(PM)
    K = Kill()
    mu = np.array([1 / 3, 0, 2 / 3])
    print('thresholds at tree', K.thresholds(mu)); print('sups at tree', sups(mu))
    print('Delta_u at face star', K.delta_u(face_star(mu), mu, parts=True))
