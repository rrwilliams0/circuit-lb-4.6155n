"""validity and pointwise ratio of Delta_u version 2 on N (and validity on random cone points with ratios in the box)."""
import numpy as np, math, sys
from bk11_core import *
from bk11_check import RB
import bk11_probe, bk12_star as S
from bk14_dec import decomp
from bk14_fast import scal, MU0
from bk14_sym import mirror_star
from bk14_d2 import Kill2
from bk14_sides import KU
MTREE = 0.27658360688237347; RHO = 2 * math.sqrt(2) / 3; N0 = math.sqrt(2 + 2 * RHO)
kap = 1 / (N0 ** 4 * float(sys.argv[3]))
KL = Kill2(eps=float(sys.argv[1]), lamM=float(sys.argv[2]))
rng = np.random.default_rng(17); bk11_probe.rng = rng
bad = 0; n = 0; worst = (0,)
for i in range(20000):
    m = bk11_probe.sample(i % 5); m /= m.sum()
    if RB(m) < MTREE: continue
    n += 1
    mu_, mv_ = S.split(m); mu = np.array([mu_[KU == k].sum() for k in (-1, 0, 1)])
    su, ss = scal(mu_), scal(mirror_star(mv_))
    import bk14_delta as BD
    du = KL.delta_u(su, mu); dv = KL.delta_u(ss, BD.mirror_mu(mu))
    D, T, X = decomp(m)
    if abs(D) > (du + dv) * (1 + 1e-9) + 1e-14: bad += 1
    sl = 4 * S.MU0 * (S.s_u(mu_) + S.s_v(mv_)); r = kap * (du + dv) / max(sl, 1e-12)
    if r > worst[0]: worst = (r, kap * (du + dv), sl, m[OFF_IDX].sum())
print('N points', n, 'violations', bad, 'max ratio %.4f (val %.3e slack %.3e off %.3f)' % worst)
