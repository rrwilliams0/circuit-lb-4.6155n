"""bk13_cover.py: coverage of the domain [0,1] x [0,1] x [1/2,2]^2 by the union of the closed records of several runs.
Checks: every box lies in the domain; no two boxes overlap (interiors disjoint; exact float comparisons, done per root cell
of the grid alpha x lam x d); the exact rational volumes add up to 9/4.  Together: the boxes tile the domain (a finite union
of closed boxes of full measure in a closed box is the whole box)."""
import sys, json
from fractions import Fraction as Fr
from collections import defaultdict
import numpy as np

files = sys.argv[1:]
abp = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.47, 0.53, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
lbp = [0.0, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
dg = list(np.linspace(0.5, 2.0, 7))
grids = [abp, lbp, dg, dg]
groups = defaultdict(list); vol = Fr(0); n = 0; bad = 0
for f in files:
    for line in open(f):
        r = json.loads(line); lo, hi = r['lo'], r['hi']; n += 1
        if not (0 <= lo[0] and hi[0] <= 1 and 0 <= lo[1] and hi[1] <= 1 and all(0.5 <= lo[i] and hi[i] <= 2 for i in (2, 3))):
            print('outside', lo, hi); bad += 1
        if any(not lo[i] < hi[i] for i in range(4)): print('degenerate', lo, hi); bad += 1
        v = Fr(1)
        for a, b in zip(lo, hi): v *= Fr(b) - Fr(a)
        vol += v
        key = tuple(int(np.searchsorted(grids[i], lo[i], side='right') - 1) for i in range(4))
        groups[key].append(lo + hi)
over = 0; maxg = 0
for key, L in groups.items():
    A = np.array(L); m = len(A); maxg = max(maxg, m)
    lo, hi = A[:, :4], A[:, 4:]
    for s in range(0, m, 2000):
        l1, h1 = lo[s:s + 2000, None, :], hi[s:s + 2000, None, :]
        ov = np.all((l1 < hi[None]) & (lo[None] < h1), axis=2)
        idx = np.arange(s, min(s + 2000, m))
        ov[np.arange(len(idx)), idx] = False
        over += int(ov.sum())
print('boxes %d in %d root cells (largest %d); outside/degenerate %d; overlapping ordered pairs %d' % (n, len(groups), maxg, bad, over))
print('total volume == 9/4 exactly: %s (float %.15f)' % (vol == Fr(9, 4), float(vol)))
