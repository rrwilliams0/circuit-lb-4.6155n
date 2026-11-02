"""bk16_margx.py : CERTIFIED marginal exclusion.  Every marginal (alpha, lam) in [0,1]^2 outside the window
W = [A0, A1] x [0, L1] fails V in [VLO, VHI] or e_0 <= E0MAX (Lemma E0), so it is not the marginal of a point of N.
Adaptive bisection in interval arithmetic; prints the number of boxes and any failure."""
import sys, numpy as np
import bk16_bnb as BB
A0, A1, L1 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
def excl(lo, hi): return BB.marg_excluded([lo[0], lo[1], 1, 1], [hi[0], hi[1], 1, 1])
roots = [([0.0, L1], [A0, 1.0]), ([A0, L1], [A1, 1.0]), ([A1, L1], [1.0, 1.0]), ([0.0, 0.0], [A0, L1]), ([A1, 0.0], [1.0, L1])]
n = 0; fails = []
stack = [(np.array(a), np.array(b), 0) for a, b in roots]
while stack:
    lo, hi, d = stack.pop()
    if excl(lo, hi): n += 1; continue
    if d > 40: fails.append((lo.tolist(), hi.tolist())); continue
    i = int(np.argmax(hi - lo)); mid = 0.5 * (lo[i] + hi[i])
    h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
    stack += [(lo, h1, d + 1), (l2, hi, d + 1)]
print('window', (A0, A1, L1), 'excluded boxes', n, 'failures', len(fails), fails[:3])
