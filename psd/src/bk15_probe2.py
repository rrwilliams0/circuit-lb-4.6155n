"""on N, max of |beta_i| + delta (the quantity of Obligation O-the-1), max c^2, and a pointwise sanity check of the
chain: alpha >= alpha_lo, k in [k_lo, k_hi], e <= e_max.  NUMERICAL."""
import sys, math, numpy as np
sys.path.insert(0, '.')
from bk11_core import *
from bk11_check import RB, MTREE, key_corrs
from bk11_probe import sample, rng
rec = []
for i in range(40000):
    m = sample(i % 5); m /= m.sum()
    if RB(m) < MTREE: continue
    c = Cv(m); ck, d = kill(c)
    k, kp, ac, ad, bc, bd = c
    b1 = ac + ad - bc - bd; b2 = ac - ad + bc - bd; sig = ac + ad + bc + bd
    ys = keys(m)  # ua ub vc vd uv
    R = (np.sqrt(m)[:, None] * GEN).T; A = R / np.linalg.norm(R, axis=1, keepdims=True)
    z = A[0] - A[1]; z /= np.linalg.norm(z)
    al = (ys[0] + ys[1]) @ z; be = -(ys[2] + ys[3]) @ z
    G = Gmat(m); e = corr(G); emx = max(e[0, 1], e[0, 2], e[0, 3], e[1, 4], e[1, 5])
    ku, kv = key_corrs(m)
    rec.append((abs(b1) + d, abs(b2) + d, sig**2 / (4 * (1 + k) * (1 + kp)), al, be, min(ku + kv), max(ku + kv), emx, d))
R = np.array(rec); print('N points', len(R))
for j, nm in enumerate('|b1|+delta |b2|+delta c2 alpha_u alpha_v k_min k_max e_max delta'.split()):
    print('%-11s min %.5f max %.5f' % (nm, R[:, j].min(), R[:, j].max()))
