"""bk12_star.py: the two-star structure of PB.
u-profiles P_u = {(t_v, t_a, t_b)} and v-profiles P_v = {(t_v, t_c, t_d)} occurring in the 123 types; a type is a pair
with the same t_v (plus the 8 special types, which are pairs too).  The 11 entries of PB split: u-side entries depend
on the u-marginal m_u only, v-side on m_v only; both share the t_v-marginal mu = (mu_-1, mu_0, mu_1) (G_uu = sum mu).
(S0) check: PB(m) = PB(m_u, m_v) (reassembled from marginals), on random m.
(S1) an earlier version's linear chain, coupled: s_u(m_u) = r k_ab + k_a + k_b - (r + 2) mu0, r = (kappa - 1)/(4 mu0);
     sigma_u(mu) = min s_u over m_u with t_v-marginal mu; sigma_v(mu) likewise; is sigma_u + sigma_v >= 0 for all mu?
     Also reproduce the single-star minimum (an earlier version: -6.9e-6).
(S2) exact PB - K0 minimised over the fibre product (m_u, m_v with equal marginals) - is it >= 0?
NUMERICAL."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, sys, itertools, numpy as np
from scipy.optimize import minimize
from bk11_core import TYPES, NT, KAPPA, K0, P_T, S_T, mtree, TIDX
from bk12_pb import PB_from11
RHO = 2 * math.sqrt(2) / 3; MU0 = P_T; R = (KAPPA - 1) / (4 * MU0)
PU = sorted(set((t[1], t[2], t[3]) for t in TYPES)); PV = sorted(set((t[1], t[4], t[5]) for t in TYPES))
print('u-profiles', len(PU), 'v-profiles', len(PV), 'r =', R)
GU = np.array([[1.0] + [2.0 ** (-x / 2) for x in pr] for pr in PU])     # columns u, v, a, b
GV = np.array([[1.0, 2.0 ** (-pr[0] / 2), 2.0 ** (-pr[1] / 2), 2.0 ** (-pr[2] / 2)] for pr in PV])   # u, v, c, d
TVU = np.array([pr[0] for pr in PU]); TVV = np.array([pr[0] for pr in PV])


def marg(m, tv): return np.array([m[tv == k].sum() for k in (-1, 0, 1)])


def side(m, Gg):
    """Gram of (u, v, x, y) rows for a star with masses m over profiles; returns cosines e0, e_ux, e_uy, h_xy, h_vx, h_vy
    for the u-star (x, y = a, b); for the v-star call with the roles swapped (see vside)."""
    G = (Gg.T * m) @ Gg; d = np.sqrt(np.diag(G)); Gh = G / np.outer(d, d)
    return Gh


def entries_u(mu_):
    Gh = side(mu_, GU)   # order u v a b
    return dict(e0=Gh[0, 1], eua=Gh[0, 2], eub=Gh[0, 3], hab=Gh[2, 3], hva=Gh[1, 2], hvb=Gh[1, 3])


def entries_v(mv_):
    Gh = side(mv_, GV)   # order u v c d
    return dict(e0=Gh[0, 1], evc=Gh[1, 2], evd=Gh[1, 3], hcd=Gh[2, 3], huc=Gh[0, 2], hud=Gh[0, 3])


def PB_uv(mu_, mv_):
    E = entries_u(mu_); E.update(entries_v(mv_)); return PB_from11(E)[0]


def kcorr(e1, e2, h): return (1 + e1 + e2 + h) / (2 * math.sqrt((1 + e1) * (1 + e2)))


def s_u(mu_):
    E = entries_u(mu_)
    return R * kcorr(E['eua'], E['eub'], E['hab']) + kcorr(E['e0'], E['eua'], E['hva']) + kcorr(E['e0'], E['eub'], E['hvb']) - (R + 2) * MU0


def s_v(mv_):
    E = entries_v(mv_)
    return R * kcorr(E['evc'], E['evd'], E['hcd']) + kcorr(E['e0'], E['evc'], E['huc']) + kcorr(E['e0'], E['evd'], E['hud']) - (R + 2) * MU0


def split(m):
    mu_ = np.zeros(len(PU)); mv_ = np.zeros(len(PV))
    for i, t in enumerate(TYPES):
        mu_[PU.index((t[1], t[2], t[3]))] += m[i]; mv_[PV.index((t[1], t[4], t[5]))] += m[i]
    return mu_, mv_


rng = np.random.default_rng(4)
from bk12_pb import PB
worst = 0
for i in range(50):
    m = rng.dirichlet(0.3 * np.ones(NT)); mu_, mv_ = split(m)
    worst = max(worst, abs(PB(m)[0] - PB_uv(mu_, mv_)))
print('(S0) PB(m) vs PB(m_u, m_v): max difference %.2e' % worst)
mT = mtree(); muT, mvT = split(mT); print('    tree marginal', marg(muT, TVU), 's_u(T) = %.2e s_v(T) = %.2e PB_uv - K0 = %.2e' % (s_u(muT), s_v(mvT), PB_uv(muT, mvT) - K0))


def min_star(fun, Gg, tv, mu_target, nst=12):
    """minimise fun over star masses with the given t_v-marginal; parametrise by conditional weights within classes."""
    cls = {k: np.where(tv == k)[0] for k in (-1, 0, 1)}
    best = (np.inf, None)
    for it in range(nst):
        x0 = np.sqrt(rng.dirichlet(np.ones(len(tv)) * rng.choice([0.3, 1, 3])))
        def build(x):
            m = np.zeros(len(tv))
            for j, k in enumerate((-1, 0, 1)):
                idx = cls[k]; w = x[idx] ** 2; s = w.sum()
                if s > 0 and mu_target[j] > 0: m[idx] = mu_target[j] * w / s
            return m
        def f(x): return fun(build(x))
        def fg(x):
            v = f(x); g = np.zeros_like(x); e = 1e-7
            for i in range(len(x)):
                xp = x.copy(); xp[i] += e; g[i] = (f(xp) - v) / e
            return v, g
        r = minimize(fg, x0, jac=True, method='L-BFGS-B', options=dict(maxiter=300))
        if r.fun < best[0]: best = (r.fun, build(r.x))
    return best


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'S1'
    if mode == 'S1':
        # single-star minimum, free marginal
        best = (np.inf, None)
        for it in range(40):
            m0 = rng.dirichlet(np.ones(len(PU)) * rng.choice([0.2, 0.5, 1.0]))
            def f(x): m = x * x; return s_u(m)
            def fg(x):
                v = f(x); g = np.zeros_like(x); e = 1e-7
                for i in range(len(x)):
                    xp = x.copy(); xp[i] += e; g[i] = (f(xp) - v) / e
                return v, g
            r = minimize(fg, np.sqrt(m0), jac=True, method='L-BFGS-B')
            if r.fun < best[0]: best = (r.fun, r.x ** 2 / np.sum(r.x ** 2))
        print('(S1) single u-star min of s_u (free marginal): %.4e at marginal %s' % (best[0], np.round(marg(best[1], TVU), 5)))
        print('     top profiles:', [(PU[i], round(best[1][i], 4)) for i in np.argsort(-best[1])[:5]])
        # coupled: grid over marginals mu = (x, y, 1 - x - y) near the tree marginal (1/3, 0, 2/3) and broadly
        print('(S1) coupled: sigma_u(mu) + sigma_v(mu) over marginals')
        pts = [(1 / 3, 0.0)]
        for s in (0.002, 0.01, 0.03, 0.1):
            for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                x = 1 / 3 + s * math.cos(ang); y = max(0.0, s * math.sin(ang))
                if x > 0 and x + y < 1: pts.append((x, y))
        for _ in range(25):
            x = rng.uniform(0.05, 0.95); y = rng.uniform(0, 0.95 - x); pts.append((x, y))
        worst = (np.inf, None)
        for (x, y) in pts:
            mu_t = np.array([x, y, 1 - x - y])
            su = min_star(s_u, GU, TVU, mu_t)[0]; sv = min_star(s_v, GV, TVV, mu_t)[0]
            tot = su + sv
            print('     mu = (%.4f, %.4f, %.4f): sigma_u %.3e sigma_v %.3e sum %.3e' % (x, y, 1 - x - y, su, sv, tot), flush=True)
            if tot < worst[0]: worst = (tot, (x, y))
        print('     worst coupled sum: %.4e at mu = %s' % worst)
    else:
        # (S2) exact PB over the fibre product: joint minimisation over (x_u, x_v) with shared marginal variables
        print('(S2) min of PB - K0 over the fibre product (m_u, m_v with equal t_v-marginals)')
        clsU = {k: np.where(TVU == k)[0] for k in (-1, 0, 1)}; clsV = {k: np.where(TVV == k)[0] for k in (-1, 0, 1)}
        nU, nV = len(PU), len(PV)
        def build(z):
            w = z[:3] ** 2; mu_t = w / w.sum(); xu = z[3:3 + nU]; xv = z[3 + nU:]
            mu_ = np.zeros(nU); mv_ = np.zeros(nV)
            for j, k in enumerate((-1, 0, 1)):
                wu = xu[clsU[k]] ** 2; wv = xv[clsV[k]] ** 2
                mu_[clsU[k]] = mu_t[j] * wu / wu.sum(); mv_[clsV[k]] = mu_t[j] * wv / wv.sum()
            return mu_, mv_
        def f(z): mu_, mv_ = build(z); return PB_uv(mu_, mv_) - K0
        def fg(z):
            v = f(z); g = np.zeros_like(z); e = 1e-7
            for i in range(len(z)):
                zp = z.copy(); zp[i] += e; g[i] = (f(zp) - v) / e
            return v, g
        best = (np.inf, None)
        for it in range(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
            z0 = np.sqrt(np.concatenate([rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(nU) * rng.choice([0.2, 1])), rng.dirichlet(np.ones(nV) * rng.choice([0.2, 1]))]))
            r = minimize(fg, z0 + 1e-3, jac=True, method='L-BFGS-B', options=dict(maxiter=500))
            mu_, mv_ = build(r.x)
            print('     start %2d: PB - K0 = %.3e  marginal %s' % (it, r.fun, np.round(marg(mu_, TVU), 4)), flush=True)
            if r.fun < best[0]: best = (r.fun, r.x)
        print('     fibre-product min: %.4e' % best[0])
