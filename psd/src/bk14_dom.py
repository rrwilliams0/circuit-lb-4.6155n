"""bk14_dom.py: NUMERICAL map of the CKS domain: marginals p = (alpha, lam) for which some slice (d in the 0.29-box) is
feasible together with the N-star row l_a + l_b + l_ab <= 3 c_Theta (Lemma NS + Lemma L)."""
import numpy as np, math, itertools
from scipy.optimize import linprog
import bk13_core as K
from bk14_ymax import slice_rows, CT
PU = K.PU
res = {}
for al in np.linspace(0.30, 0.70, 41):
    for la in [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        mu = K.mu_of(al, la); ok = False
        for lda, ldb in itertools.product(np.linspace(-0.29, 0.29, 21), repeat=2):
            sa, sb = math.exp(-lda / 2), math.exp(-ldb / 2)
            cL, constL = slice_rows(mu, sa, sb)
            b = np.array([mu[0], mu[1], mu[2], sa ** -2, sb ** -2])
            r = linprog(np.zeros(len(PU)), A_ub=cL[None, :], b_ub=[3 * CT - constL], A_eq=K.A_EQ, b_eq=b, bounds=[(0, None)] * len(PU), method='highs')
            if r.status == 0: ok = True; break
        res[(round(al, 3), la)] = ok
for la in [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
    als = [al for (al, l), ok in res.items() if l == la and ok]
    print('lam %.3f: feasible alpha in [%s, %s] (%d grid points)' % (la, min(als) if als else '-', max(als) if als else '-', len(als)))
