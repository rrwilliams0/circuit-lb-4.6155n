"""bk14_far.py: far-star search: stars with off-face class -1 mass on (-1,-1,1) or (-1,1,-1) and face imbalance, at the
N-star boundary; minimise 4 mu0 s_u - kappa Delta_u (version 2) from such seeds.  NUMERICAL."""
import numpy as np, math, sys, itertools, os
pilo = sys.argv[1]; LOGF = sys.argv[2]; lamM = sys.argv[3] if len(sys.argv) > 3 else '1.0'; eps = sys.argv[4] if len(sys.argv) > 4 else '0.05'
sys.argv = ['x', eps, lamM, pilo, '0']
import bk14_tau2 as T
from bk14_fast import scal, s_u, MU0, PU
import bk14_delta as BD, bk13_core as K13
from scipy.optimize import minimize
LOG = open(LOGF, 'w')
def seed(mu, y, prof, t):
    m = BD.face_star(mu).copy(); i0 = PU.index((-1, 1, 1)); m[i0] -= y; m[PU.index(prof)] += y
    m[PU.index((1, -1, 1))] += t * mu[2]; m[PU.index((1, 1, -1))] -= t * mu[2]
    return np.log(np.maximum(m, 1e-7))
for al in [float(a) for a in os.environ.get("FAR_AL", "0.46,0.48,0.49,0.50,0.51,0.52,0.53,0.54,0.56,0.58,0.60").split(",")]:
    for lam in [float(a) for a in os.environ.get("FAR_LAM", "0.0,0.01").split(",")]:
        mu = K13.mu_of(al, lam); best = (np.inf, None)
        for y, prof, t in itertools.product([0.06, 0.1, 0.14], [(-1, -1, 1), (-1, 1, -1)], [-0.3, -0.15, 0.15, 0.3]):
            x0 = seed(mu, y, prof, t)
            r = minimize(lambda x: T.obj(x, mu), x0, method='L-BFGS-B', options=dict(maxiter=3000, maxfun=60000))
            r = minimize(lambda x: T.obj(x, mu), r.x, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-9, fatol=1e-14))
            if r.fun < best[0]: best = (r.fun, T.build(r.x, mu))
        s = scal(best[1]); m = best[1]
        off = sum(m[i] for i in range(len(PU)) if PU[i] not in [(-1, 1, 1), (1, -1, 1), (1, 1, -1), (0, 1, -1), (0, -1, 1)])
        print('%.3f %.3f far-tau %.4e  4mu0 s %.4e  kD %.4e  off %.3f  ratio kD/slack %.4f' % (al, lam, best[0], 4 * MU0 * s_u(s), T.kap * T.delta_u(s, mu), off, T.kap * T.delta_u(s, mu) / max(4 * MU0 * s_u(s), 1e-12)), file=LOG, flush=True)
print('done', file=LOG, flush=True)
