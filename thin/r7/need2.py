"""Item 6: is there a cone point in N = {RB >= M_tree} with |log d_v| > 0.145?  penalty ascent of |log d_v|."""
import math, numpy as np
from scipy.optimize import minimize
import oc
def dv(m): g = oc.G(m); return g[1,1]/g[0,0]
rng = np.random.default_rng(3)
best = (0, None)
for trial in range(12):
    z0 = np.log(oc.mT + 1e-3*rng.uniform(0, 1, 123) + 1e-12)
    sgn = 1 if trial % 2 == 0 else -1
    def obj(z):
        m = np.exp(z - z.max()); m /= m.sum()
        rb = oc.RB(m); pen = max(0.0, oc.MTREE + 1e-6 - rb)
        return -sgn*math.log(dv(m)) + 1e4*pen
    r = minimize(obj, z0, method='Powell', options={'maxiter': 20000, 'xtol': 1e-6, 'ftol': 1e-10})
    m = np.exp(r.x - r.x.max()); m /= m.sum()
    ld = math.log(dv(m)); rb = oc.RB(m)
    if rb >= oc.MTREE and abs(ld) > abs(best[0]): best = (ld, m)
    print('trial %d sign %+d: log d_v %.4f RB %.8f (M_tree %.8f)' % (trial, sgn, ld, rb, oc.MTREE), flush=True)
m = best[1]; g = oc.G(m); d = np.diag(g)/g[0,0]
print('best |log d_v| in N: %.4f ; log d = %s ; RB - M_tree = %.2e' % (abs(best[0]), np.round(np.log(d[1:]), 4), oc.RB(m)-oc.MTREE))
np.save('need2_best.npy', m)
