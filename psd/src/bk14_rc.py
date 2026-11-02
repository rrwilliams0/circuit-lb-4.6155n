import numpy as np, math
import bk13_core as K
from bk14_sides import PU as PU2
PU = K.PU
A_, B_, C_ = PU.index((-1, 1, 1)), PU.index((1, -1, 1)), PU.index((1, 1, -1))
J0, J0p, JP, JM = PU.index((0, 1, -1)), PU.index((0, -1, 1)), PU.index((1, -1, -1)), PU.index((1, 1, 1))
mu = K.mu_of(0.5, 0.0); sa = sb = 1.0
c, const = K.lp_data(mu, sa, sb)
for name, Bs in (('B+', [A_, B_, C_, J0, JP]), ('B-', [A_, B_, C_, J0, JM])):
    Bm = K.A_EQ[:, Bs]; y = np.linalg.solve(Bm.T, c[Bs]); rc = c - K.A_EQ.T @ y
    print(name, 'reduced costs (s units) at p_T, d=1:')
    for j in np.argsort(rc): print('   ', PU[j], '%.5f' % rc[j], ' x4mu0 = %.5f' % (4 * K.MU0 * rc[j]))
