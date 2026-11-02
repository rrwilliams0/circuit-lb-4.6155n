"""bk14_ymax.py: NUMERICAL. On star slices with the N-star constraint (l_a + l_b + l_ab <= 3 c_Theta, Lemma NS + Lemma L),
the largest off-face mass (classes -1 and 1 off the face profiles; and class 0 = mu_0) over a grid of (p, d)."""
import numpy as np, math, itertools
from scipy.optimize import linprog
import bk13_core as K
from bk14_lpk import CT, RHO
PU = K.PU; FACE = [PU.index((-1, 1, 1)), PU.index((1, -1, 1)), PU.index((1, 1, -1))]
def slice_rows(mu, sa, sb):
    V, U = K.VU(mu); e0 = U / math.sqrt(V); iv = 1 / math.sqrt(V); D0 = K.D0
    cL = (sa * K.GA + sb * K.GB) * (2 / D0 - 8 / D0 ** 2) + iv * K.GV * (sa * K.GA + sb * K.GB) / D0 + sa * sb * K.GA * K.GB / D0
    constL = 3 / D0 + 2 * e0 / D0 - (4 / D0 ** 2) * (2 * e0 - 6 * RHO)
    return cL, constL
best = (0, None); feas = 0; tot = 0
for al, la in itertools.product(np.linspace(0.3, 0.7, 21), [0, 0.005, 0.02, 0.05, 0.1, 0.2]):
    mu = K.mu_of(al, la)
    for lda, ldb in itertools.product(np.linspace(-0.29, 0.29, 15), repeat=2):
        sa, sb = math.exp(-lda / 2), math.exp(-ldb / 2); tot += 1
        cL, constL = slice_rows(mu, sa, sb)
        c = np.array([0.0 if (j in FACE or K.KV[j] == 0) else -1.0 for j in range(len(PU))])
        b = np.array([mu[0], mu[1], mu[2], sa ** -2, sb ** -2])
        r = linprog(c, A_ub=cL[None, :], b_ub=[3 * CT - constL], A_eq=K.A_EQ, b_eq=b, bounds=[(0, None)] * len(PU), method='highs')
        if r.status == 0:
            feas += 1
            if -r.fun > best[0]: best = (-r.fun, (al, la, lda, ldb))
print('feasible slices %d of %d; max off-face (non-class-0) mass %.4f at %s' % (feas, tot, best[0], best[1]))
