"""bk14_scan.py: single-star scan tau(p) (version 2 bound, exact s_u, N-star penalty) along lines of marginals,
more starts, and the coupled sums tau(p) + tau(p^sigma).  NUMERICAL (upper bounds on infima)."""
import numpy as np, math, sys
sys.argv = ['x', '0.05', '1.0', sys.argv[1] if len(sys.argv) > 1 else '0.80', '0'] + sys.argv[2:]
import bk14_tau2 as T
from bk14_fast import scal, s_u, MU0, PU
import bk14_delta as BD, bk13_core as K13
LOG = open(sys.argv[5], 'w')
rng = np.random.default_rng(5)
def tau_many(mu, nst):
    best = (np.inf, None); f = BD.face_star(mu)
    starts = [np.log(f + 1e-6)]
    for k in range(nst):
        x = np.log(f + 1e-6)
        j = rng.integers(len(PU)); x[j] = np.log(rng.uniform(0.02, 0.2))
        x += rng.normal(scale=0.3, size=len(PU)); starts.append(x)
    for x0 in starts:
        from scipy.optimize import minimize
        r = minimize(lambda x: T.obj(x, mu), x0, method='L-BFGS-B', options=dict(maxiter=3000, maxfun=60000))
        r = minimize(lambda x: T.obj(x, mu), r.x, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-9, fatol=1e-14))
        if r.fun < best[0]: best = (r.fun, T.build(r.x, mu))
    return best
pts = []
for lam in (0.0, 0.01, 0.05, 0.1):
    for al in np.arange(0.32, 0.665, 0.02):
        pts.append((round(al, 3), lam))
res = {}
for (al, lam) in pts:
    mu = K13.mu_of(al, lam)
    t, m = tau_many(mu, 14); s = scal(m)
    res[(al, lam)] = t
    print('%.3f %.3f tau %.4e  4mu0 s %.4e  kD %.4e  offmass %.3f' % (al, lam, t, 4 * MU0 * s_u(s), T.kap * T.delta_u(s, mu), m[[i for i in range(len(PU)) if PU[i] not in [(-1, 1, 1), (1, -1, 1), (1, 1, -1), (0, 1, -1), (0, -1, 1)]]].sum()), file=LOG, flush=True)
print('done', file=LOG, flush=True)
