"""bk13_core.py : own float implementation of the u-star problem.
u-profiles (k, ta, tb) = (t_v, t_a, t_b), generator g = (1, 2^{-k/2}, 2^{-ta/2}, 2^{-tb/2}) for rows (u, v, a, b).
Marginal coordinates p = (alpha, lam): mu = (alpha (1-lam), lam, 2 (1-alpha)(1-lam)) / S  (mirror: alpha -> 1 - alpha).
Exact star slack s_u(m); Lemma-L objective Lfun(m); slice LP(mu, sa, sb) with sa = d_a^{-1/2}, sb = d_b^{-1/2}.
NUMERICAL (floating point)."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, itertools
import numpy as np
from scipy.optimize import linprog, minimize

KAPPA = 4.478385650326709971336669
RHO = 2 * math.sqrt(2) / 3
MU0 = (1 + 2 * math.sqrt(2)) / 4
R = (KAPPA - 1) / (4 * MU0)
D0 = 2 + 2 * RHO
K1 = 1 / D0 - 4 / D0 ** 2

def profiles():
    out = []
    for k, ta, tb in itertools.product((-1, 0, 1), repeat=3):
        if min(k, ta, tb) == -1: out.append((k, ta, tb))
    out.append((1, 1, 1))
    return out
PU = profiles(); NP = len(PU); assert NP == 20
KV = np.array([p[0] for p in PU]); TA = np.array([p[1] for p in PU]); TB = np.array([p[2] for p in PU])
GV = 2.0 ** (-KV / 2); GA = 2.0 ** (-TA / 2); GB = 2.0 ** (-TB / 2)
GEN = np.vstack([np.ones(NP), GV, GA, GB]).T
CLS = [np.where(KV == k)[0] for k in (-1, 0, 1)]
STAR_T = [PU.index((-1, 1, 1)), PU.index((1, -1, 1)), PU.index((1, 1, -1))]

def mu_of(alpha, lam):
    m = np.array([alpha * (1 - lam), lam, 2 * (1 - alpha) * (1 - lam)])
    return m / m.sum()

def p_of(mu):
    N = mu[0] + mu[2] / 2; lam = mu[1] / (mu[1] + N); return np.array([mu[0] / N if N > 0 else 0.5, lam])

def mirror_mu(mu):
    w = np.array([mu[2] / 2, mu[1], 2 * mu[0]]); return w / w.sum()

def VU(mu):
    V = mu[0] * 2 + mu[1] + mu[2] / 2; U = mu[0] * math.sqrt(2) + mu[1] + mu[2] / math.sqrt(2)
    return V, U

def ghat(m):
    G = (GEN.T * m) @ GEN; d = np.sqrt(np.diag(G)); return G / np.outer(d, d), G

def kc(e1, e2, h): return (1 + e1 + e2 + h) / (2 * math.sqrt((1 + e1) * (1 + e2)))

def s_exact(m):
    Gh, _ = ghat(m)
    e0, eua, eub, hva, hvb, hab = Gh[0, 1], Gh[0, 2], Gh[0, 3], Gh[1, 2], Gh[1, 3], Gh[2, 3]
    return R * kc(eua, eub, hab) + kc(e0, eua, hva) + kc(e0, eub, hvb) - (R + 2) * MU0

def lL(e1, e2, h): return (1 + e1 + e2 + h) / D0 - (4 / D0 ** 2) * (e1 + e2 - 2 * RHO)

def Lfun(m):
    Gh, _ = ghat(m)
    e0, eua, eub, hva, hvb, hab = Gh[0, 1], Gh[0, 2], Gh[0, 3], Gh[1, 2], Gh[1, 3], Gh[2, 3]
    return R * lL(eua, eub, hab) + lL(e0, eua, hva) + lL(e0, eub, hvb) - (R + 2) * MU0

def lp_data(mu, sa, sb):
    """objective c (per profile) and constant for the slice LP at marginal mu (sum 1), d_a = sa^-2, d_b = sb^-2."""
    V, U = VU(mu); e0 = U / math.sqrt(V); iv = 1 / math.sqrt(V)
    const = 2 * (1 + e0) / D0 - (8 / D0 ** 2) * (e0 - 2 * RHO) + R * (1 / D0 + 8 * RHO / D0 ** 2) - (R + 2) * MU0
    c = (1 + R) * K1 * (sa * GA + sb * GB) + (iv / D0) * GV * (sa * GA + sb * GB) + (R / D0) * sa * sb * GA * GB
    return c, const

A_EQ = np.vstack([(KV == k).astype(float) for k in (-1, 0, 1)] + [GA ** 2, GB ** 2])

def LP(mu, sa, sb, full=False):
    c, const = lp_data(mu, sa, sb)
    b = np.array([mu[0], mu[1], mu[2], sa ** -2, sb ** -2])
    r = linprog(c, A_eq=A_EQ, b_eq=b, bounds=[(0, None)] * NP, method='highs')
    if r.status != 0: return (None, None, None) if full else None
    if full: return r.fun + const, r.x, r.eqlin.marginals
    return r.fun + const

def fiber_build(mu, x):
    m = np.zeros(NP)
    for j, idx in enumerate(CLS):
        w = x[idx] ** 2; s = w.sum()
        if mu[j] > 0 and s > 0: m[idx] = mu[j] * w / s
    return m

def min_fiber(fun, mu, nst=10, rng=None, x0s=None):
    rng = rng or np.random.default_rng(0)
    best = (np.inf, None)
    starts = list(x0s or [])
    while len(starts) < nst: starts.append(np.sqrt(rng.dirichlet(np.ones(NP) * rng.choice([0.3, 1, 3]))))
    for x0 in starts:
        f = lambda x: fun(fiber_build(mu, x))
        def fg(x):
            v = f(x); g = np.zeros_like(x); e = 1e-7
            for i in range(len(x)):
                xp = x.copy(); xp[i] += e; g[i] = (f(xp) - v) / e
            return v, g
        r = minimize(fg, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=400))
        if r.fun < best[0]: best = (r.fun, fiber_build(mu, r.x))
    return best

def tree_start(mu):
    x = np.full(NP, 1e-3)
    for i in STAR_T: x[i] = 1.0
    x[PU.index((0, -1, 1))] = 1.0; x[PU.index((0, 1, -1))] = 1.0
    return x

def sigmaLP(mu, nst=6, rng=None):
    """min over d of LP(mu, d): start from the minimiser of Lfun over the fiber, polish with Nelder-Mead on (log sa, log sb)."""
    v, m = min_fiber(Lfun, mu, nst=nst, rng=rng, x0s=[tree_start(mu)])
    _, G = ghat(m); sa0, sb0 = G[2, 2] ** -0.5, G[3, 3] ** -0.5
    f = lambda z: (LP(mu, math.exp(z[0]), math.exp(z[1])) or 9.0)
    r = minimize(f, np.log([sa0, sb0]), method='Nelder-Mead', options=dict(xatol=1e-9, fatol=1e-13, maxiter=800))
    return min(v, r.fun), np.exp(r.x), m
