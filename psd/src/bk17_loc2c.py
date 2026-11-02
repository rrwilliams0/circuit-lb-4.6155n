"""bk17_loc2c.py : Lemma LOC2-K(c), a per-cell refinement of LOC2-K(b).
For each of the 12 alpha cells C_i = [0.47 + 0.005 i, 0.475 + 0.005 i] of Q (a = alpha - 1/2) it certifies
      h_K(p, t) - P0 mu_0(p) >= -c_i      for p in C_i x [0, 0.005], t in T = [-0.027, 0.029],
by the second-order Taylor bound (bk13_L3b.taylor_lb) on a grid of sub-boxes (mu_0 <= lam / S_min).  h_K is evaluated with
the kappa of bk17_cfg (pi_lo = 0.82); h_K(0.82) >= h_K(0.815) (Dface >= 0), so H_new >= H_old.  Output: loc2c_boxes.jsonl
(every sub-box with its lower bound) and loc2c.json (the c_i, rounded up).  CERTIFIED (interval arithmetic)."""
import sys, json, itertools, time
import numpy as np
import bk17_cfg
from ivarith import IA, dn, up
from tay2 import Tay2
from bk13_L3b import taylor_lb
import bk16_loc2 as L2
P0 = 0.02; RL = 0.005; T = (-0.027, 0.029)
SMIN = float(dn(dn((1 - RL) * (2 - 0.5 - 0.0300001))))
NA, NL, NT = int(sys.argv[1]) if len(sys.argv) > 1 else 4, 4, 28


def h3(lo, hi):
    a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
    return L2.h_kill(a + 0.5, la, t)


out = open('loc2c_boxes.jsonl', 'w'); cs = []
t0 = time.time()
for i in range(12):
    a0 = 0.47 + 0.005 * i - 0.5; a1 = a0 + 0.005
    if i == 0: a0 = 0.47 - 0.5
    if i == 11: a1 = 0.53 - 0.5
    ga = np.linspace(a0, a1, NA + 1); gl = np.linspace(0, RL, NL + 1); gt = np.linspace(T[0], T[1], NT + 1)
    ga[0], ga[-1] = a0, a1; gt[0], gt[-1] = T
    worst = 1.0
    for ia, il, it in itertools.product(range(NA), range(NL), range(NT)):
        lo = np.array([ga[ia], gl[il], gt[it]]); hi = np.array([ga[ia + 1], gl[il + 1], gt[it + 1]])
        lb, _ = taylor_lb(h3, lo, hi); lb = float(dn(lb - float(up(P0 * hi[2 - 1] / SMIN))))
        worst = min(worst, lb)
        out.write(json.dumps(dict(cell=i, lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n')
    c = max(0.0, float(up(-worst)))
    cs.append(c)
    print('cell %2d alpha [%.3f, %.3f]: min lb %.4e -> c = %.4e  (%.0f s)' % (i, a0 + 0.5, a1 + 0.5, worst, c, time.time() - t0), flush=True)
out.close()
json.dump(dict(c=cs, P0=P0, T=T, SMIN=SMIN, NA=NA, NL=NL, NT=NT), open('loc2c.json', 'w'))
