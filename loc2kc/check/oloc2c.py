"""the independent check: Lemma LOC2-K(c).  Own second-order Taylor lower bound of h_K(1/2 + a, lam, t) - P0 lam_hi/S_min
(own h_K = an independent program's hk.py, own AD = oad.D2 on mpmath.iv, KILL of pi_lo = 0.82), on
 (1) N sampled records of run/loc2c/loc2c_boxes.jsonl (plus the 12 per-cell minimisers), checked >= -c_cell;
 (2) exact tiling of each cell by its 448 records (exact rational volume + grid structure);
 (3) the 11 GAP slivers between consecutive LOC2-K(c) cells (a in (a1_j, a0_{j+1}), width about 4e-18), which the mirror of the run's
     alpha cell 11 - j reaches: certify >= -c_j on [a1_j - 1e-9, a0_{j+1} + 1e-9] x [0, 0.005] x T (4 x 28 sub-boxes)."""
import os, sys, json, random, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'a16'))
from fractions import Fraction as Fr
from mpmath import iv, mpf
iv.prec = 90
import hk
from oad import D2
C = hk.consts_iv(kill='0.08077294626')
P0 = iv.mpf('0.02')
SMIN = (1 - iv.mpf('0.005')) * (2 - iv.mpf('0.5300001'))      # S >= (1 - lam)(2 - alpha) on Q (S = that + lam)
SMINlo = SMIN.a


def own_lb(lo, hi):
    c = [(lo[i] + hi[i]) / 2 for i in range(3)]
    def f(L, H):
        a, la, t = [D2.var(i, 3, mpf(L[i]), mpf(H[i])) for i in range(3)]
        return hk.h_K(a + 0.5, la, t, C)
    P = f(c, c); B = f(lo, hi)
    D = [iv.mpf([mpf(lo[i]) - mpf(c[i]), mpf(hi[i]) - mpf(c[i])]) for i in range(3)]
    tot = P.v
    for i in range(3): tot = tot + P.g[i] * D[i]
    for i in range(3):
        for l in range(3):
            pr = D[i] * D[l] if i != l else iv.mpf([0, (D[i] * D[i]).b])
            tot = tot + B.h[i][l] * pr * 0.5
    tot = tot - P0 * iv.mpf(hi[1]) / iv.mpf(SMINlo)
    return tot.a


base = sys.argv[1]; N = int(sys.argv[2])
rs = [json.loads(l) for l in open(os.path.join(base, 'loc2c_boxes.jsonl'))]
cc = json.load(open(os.path.join(base, 'loc2c.json')))['c']
t0 = time.time()
# (2) tiling
T = (-0.027, 0.029); badtile = 0
for i in range(12):
    a0 = 0.47 + 0.005 * i - 0.5; a1 = a0 + 0.005
    if i == 0: a0 = 0.47 - 0.5
    if i == 11: a1 = 0.53 - 0.5
    b = [r for r in rs if r['cell'] == i]
    V = sum(((Fr(r['hi'][0]) - Fr(r['lo'][0])) * (Fr(r['hi'][1]) - Fr(r['lo'][1])) * (Fr(r['hi'][2]) - Fr(r['lo'][2])) for r in b), Fr(0))
    Vc = (Fr(a1) - Fr(a0)) * Fr(0.005) * (Fr(T[1]) - Fr(T[0]))
    keys = set((r['lo'][0], r['lo'][1], r['lo'][2]) for r in b)
    inside = all(Fr(a0) <= Fr(r['lo'][0]) and Fr(r['hi'][0]) <= Fr(a1) and 0 <= r['lo'][1] and r['hi'][1] <= 0.005 and T[0] <= r['lo'][2] and r['hi'][2] <= T[1] for r in b)
    # grid: the distinct coordinates form a product grid; disjointness = distinct lower corners on a product grid with matching upper corners
    ga = sorted(set(r['lo'][0] for r in b)); gl = sorted(set(r['lo'][1] for r in b)); gt = sorted(set(r['lo'][2] for r in b))
    grid_ok = len(keys) == len(b) == len(ga) * len(gl) * len(gt)
    if not (V == Vc and inside and grid_ok): badtile += 1; print('tiling problem cell', i, V == Vc, inside, grid_ok)
    if max(-min(r['lb'] for r in b), 0.0) > cc[i]: badtile += 1; print('c_i below the recorded min, cell', i)
print('(2) tiling: cells with a problem %d (exact volumes, product grid, c_i >= -min recorded lb)' % badtile, flush=True)
# (1) sample
random.seed(23)
cellmin = {}
for r in rs:
    if r['cell'] not in cellmin or r['lb'] < cellmin[r['cell']]['lb']: cellmin[r['cell']] = r
pick = list(cellmin.values()) + random.sample(rs, N)
bad = 0; worst = 1.0; maxdiff = 0.0
for r in pick:
    lb = own_lb(r['lo'], r['hi']); need = -cc[r['cell']]
    maxdiff = max(maxdiff, abs(float(lb) - r['lb']))
    worst = min(worst, float(lb) - need)
    if lb < need: bad += 1; print('own LOC2c FAIL', r, float(lb), flush=True)
print('(1) %d records (12 cell minimisers + %d random): own failures %d, min own margin over -c_cell %.3e, max |own lb - recorded lb| %.2e, %.0f s'
      % (len(pick), N, bad, worst, maxdiff, time.time() - t0), flush=True)
# (3) gaps
gl = [0.0, 0.00125, 0.0025, 0.00375, 0.005]
gt = [T[0] + (T[1] - T[0]) * k / 28 for k in range(29)]; gt[0], gt[-1] = T
for j in range(11):
    a1 = 0.47 + 0.005 * j - 0.5 + 0.005 if j > 0 else (0.47 - 0.5) + 0.005
    a0n = 0.47 + 0.005 * (j + 1) - 0.5
    lo_a, hi_a = a1 - 1e-9, a0n + 1e-9
    w = 1.0
    for il in range(4):
        for it in range(28):
            w = min(w, float(own_lb([lo_a, gl[il], gt[it]], [hi_a, gl[il + 1], gt[it + 1]])))
    print('(3) gap after cell %2d: a in [%.18f, %.18f], own min lb %.4e, need >= -c_%d = %.4e : %s' % (j, lo_a, hi_a, w, j, -cc[j], 'OK' if w >= -cc[j] else 'FAIL'), flush=True)
print('total %.0f s' % (time.time() - t0))
