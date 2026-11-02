"""bk16_loc2verify.py: replay of the LOC2-K record (every Taylor box recomputed; origin box recomputed)."""
import sys, json, numpy as np
from ivarith import dn, up
from tay2 import Tay2
from bk13_L3b import taylor_lb
import bk16_loc2 as L2
fn = sys.argv[1]
recs = [json.loads(l) for l in open(fn)]
sys.argv = ['x', str(recs[0]['P0']), '1.5e-4', str(recs[0]['eps0'])]
import bk16_loc2run as R
o = recs[0]; ok, info = R.origin_ok(o['eps0'], o['epl'], (-o['eps0'], o['eps0']), o['P0'], o['Smin'])
print('origin', ok, info['cX'], info['G'], info['need'])
def h3(lo, hi):
    a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
    return L2.h_kill(a + 0.5, la, t)
bad = 0; mn = {'psi': 1, 'single': 1}; n = 0
for r in recs[1:]:
    lo, hi = np.array(r['lo']), np.array(r['hi'])
    if r['kind'] == 'psi':
        lb, _ = taylor_lb(L2.psi_tay, lo, hi); lb = float(dn(lb - float(up(2 * o['P0'] * hi[1] / o['Smin']))))
        if lb < 0: bad += 1
    elif r['kind'] == 'single':
        lb, _ = taylor_lb(h3, lo, hi); lb = float(dn(lb - float(up(o['P0'] * hi[1] / o['Smin']))))
        if lb < -1.5e-4: bad += 1
    else: bad += 1; continue
    mn[r['kind']] = min(mn[r['kind']], lb); n += 1
print('replayed', n, 'boxes, failures', bad, 'min', mn)

from fractions import Fraction as Fr
def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= Fr(b) - Fr(a)
    return v
e = o['eps0']; ra, rl, T = R.RA, R.RL, R.T
vpsi = sum(vol(r['lo'], r['hi']) for r in recs[1:] if r['kind'] == 'psi') + vol([0, 0, -e, -e], [e, min(rl, e), e, e])
print('psi volume + origin box == K volume:', vpsi == vol([0, 0, T[0], T[0]], [ra, rl, T[1], T[1]]))
vs = sum(vol(r['lo'], r['hi']) for r in recs[1:] if r['kind'] == 'single')
print('single volume == domain volume:', vs == vol([-ra, 0, T[0]], [ra, rl, T[1]]))
