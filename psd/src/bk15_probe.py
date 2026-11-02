"""the probe: PSD criterion of the kill in (c^2, g1, g2) form on N.  NUMERICAL."""
import sys, math, numpy as np
sys.path.insert(0, '.')
from bk11_core import *
from bk11_check import RB, MTREE, key_corrs
from bk11_probe import sample, rng
def stats(c):
    # c order: ab cd ac ad bc bd
    k, kp = c[0], c[1]; ac, ad, bc, bd = c[2:6]
    sig = ac + ad + bc + bd; b1 = ac + ad - bc - bd; b2 = ac - ad + bc - bd; alt = ac - ad - bc + bd
    c2 = sig**2 / (4 * (1 + k) * (1 + kp)); g1 = b1 / (2 * math.sqrt((1 - k) * (1 + kp))); g2 = b2 / (2 * math.sqrt((1 + k) * (1 - kp)))
    return k, kp, sig, b1, b2, alt, c2, g1, g2
recs = []
for i in range(40000):
    m = sample(i % 5); m /= m.sum()
    if RB(m) < MTREE: continue
    c = Cv(m); ck, d = kill(c)
    k, kp, sig, b1, b2, alt, c2, g1, g2 = stats(ck)
    F = c2 + g1**2 + g2**2 + g1**2 * g2**2 / c2
    k0, kp0, sig0, b10, b20, alt0, c20, g10, g20 = stats(c)
    recs.append((F, c2, abs(g1), abs(g2), 1 - k, 1 - kp, d, lam_min(ck), lam_min(c), abs(b10), abs(b20), abs(alt0), c20, offmass(m), float(np.linalg.norm(np.atleast_1d(imbalance(m)))), i % 5))
R = np.array(recs); print('N points', len(R))
names = 'F c2 |g1| |g2| 1-k 1-kp delta lmin(kill) lmin(C) |b1| |b2| |alt| c2pre off imb kind'.split()
for j, nm in enumerate(names[:-1]):
    print('%-10s min %.5f max %.5f' % (nm, R[:, j].min(), R[:, j].max()))
j = R[:, 0].argmax(); print('argmax F:', dict(zip(names, np.round(R[j], 5))))
j = R[:, 7].argmin(); print('argmin lmin kill:', dict(zip(names, np.round(R[j], 5))))
