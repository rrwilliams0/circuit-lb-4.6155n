"""bk14_tau2.py: tau(p) = inf over u-stars at marginal p (ratio box, N-star) of 4 mu0 s_u - kappa Delta_u (exact formulas);
coupled sum tau(p) + tau(p^sigma).  NUMERICAL (local optimisation from several starts: upper bounds on the infimum)."""
import numpy as np, math, sys
from scipy.optimize import minimize
from bk14_fast import *
import bk14_delta as BD
N0 = math.sqrt(2 + 2 * RHO)
MT = 0.27658360688237347; KE2 = 0.28070193727164739; CT = math.cos(2 * math.pi * (MT - KE2 / 2) / 3)
PILO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.80
kap = 1 / (N0 ** 4 * PILO)
eps = float(sys.argv[1]); x0 = float(sys.argv[2])
from bk14_d2 import Kill2
KL2 = Kill2(eps=eps, lamM=x0)
TH = {}
def thr(mu):
    key = tuple(np.round(mu, 12))
    if key not in TH:
        f = scal(BD.face_star(mu)); e = 1 + eps
        TH[key] = dict(U=e * f['phi'] * f['Abar'], A=e * f['Abar'], phi=e * f['phi'], w11=e * f['w11'], w10=e * f['w10'])
    return TH[key]
def delta_u_v1(s, mu):
    ms = BD.mirror_mu(mu); V = BD.Vof(mu); w = math.sqrt(V)
    Tp, Ts = thr(mu), thr(ms); Sp = BD.sups(ms)
    own = (s['X'] ** 2 / 4 + Ts['U'] * s['K'] + Ts['A'] * s['phi'] * s['K'] / V + x0 * s['K']
           + (s['phi'] / w + w * Ts['phi'] + w * x0) * s['A0'] / R22 + Ts['A'] * abs(s['X']) * s['Kp'] / V
           + abs(s['X']) * s['Phi0'] / (R22 * w) + Ts['w11'] * s['R'] / V + s['R0'] / 4 + s['Y'] ** 2 + S2 / V * Ts['w10'] * s['Rm0'])
    pos = lambda z: max(z, 0.0)
    hin = (pos(s['phi'] * s['Abar'] - Tp['U']) * Sp['K'] + pos(s['Abar'] - Tp['A']) * (Sp['Zphi'] + Sp['ZX'])
           + pos(s['phi'] - Tp['phi']) * Sp['A0'] / (R22 * w) + pos(abs(s['X']) * s['Phib'] - x0) * Sp['K']
           + pos(abs(s['X']) - x0) * Sp['A0'] / (R22 * w) + pos(s['w11'] - Tp['w11']) * Sp['ZR'] + pos(s['w10'] - Tp['w10']) * Sp['ZRm'])
    return own + hin
def delta_u(s, mu): return KL2.delta_u(s, mu)
CLS = [np.where(KU == k)[0] for k in (-1, 0, 1)]
def build(x, mu):
    m = np.zeros(len(PU))
    for j in range(3):
        idx = CLS[j]; w = np.exp(x[idx] - x[idx].max()); m[idx] = mu[j] * w / w.sum()
    return m
def obj(x, mu, pen=1e4):
    m = build(x, mu); s = scal(m)
    v = 4 * MU0 * s_u(s) - kap * delta_u(s, mu)
    ka, kb, kab = kvals_G(s['G']); G = s['G']
    v += pen * (max(0, ka + kb + kab - 3 * CT) ** 2 + max(0, abs(math.log(G[2, 2])) - 0.289) ** 2 + max(0, abs(math.log(G[3, 3])) - 0.289) ** 2)
    return v
def tau(mu, rng, nst=6):
    best = (np.inf, None)
    f = BD.face_star(mu)
    starts = [np.log(f + 1e-6)]
    for _ in range(nst):
        starts.append(np.log(f + 1e-6) + rng.normal(scale=rng.choice([0.5, 2, 4]), size=len(PU)))
    for x0_ in starts:
        r = minimize(lambda x: obj(x, mu), x0_, method='L-BFGS-B', options=dict(maxiter=3000, maxfun=60000))
        r = minimize(lambda x: obj(x, mu), r.x, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-9, fatol=1e-14))
        if r.fun < best[0]: best = (r.fun, build(r.x, mu))
    return best
if __name__ == '__main__':
    import bk13_core as K13
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    pts = [(0.5, 0.0), (0.495, 0.0), (0.49, 0.0), (0.48, 0.0), (0.47, 0.0), (0.45, 0.0), (0.42, 0.0), (0.4, 0.0), (0.37, 0.0),
           (0.5, 0.002), (0.49, 0.002), (0.5, 0.005), (0.48, 0.005), (0.5, 0.01), (0.47, 0.01), (0.5, 0.03), (0.45, 0.03), (0.5, 0.06), (0.5, 0.1)]
    LOG = open(sys.argv[5], 'w') if len(sys.argv) > 5 else sys.stdout
    for (al, la) in pts:
        mu = K13.mu_of(al, la); ms = BD.mirror_mu(mu)
        if abs(math.log(BD.Vof(mu))) > 0.2052: print('p', (al, la), 'outside V-range', file=LOG, flush=True); continue
        t1, m1 = tau(mu, rng); t2, m2 = tau(ms, rng)
        s1 = scal(m1); s2 = scal(m2)
        print('p=(%.3f,%.3f): tau(p) %.3e [4mu0 s %.3e kD %.3e]  tau(p^s) %.3e [4mu0 s %.3e]  sum %.3e' % (al, la, t1, 4 * MU0 * s_u(s1), kap * delta_u(s1, mu), t2, 4 * MU0 * s_u(s2), t1 + t2), file=LOG, flush=True)
