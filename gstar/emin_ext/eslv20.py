"""independent re-derivation: re-derivation of the two eslv_out.json entries that the records use ("12,6,hi" and "5,6,lo"), by the
an independent program's construction (eslv.py): every Emin leaf of the table cell touching that alpha edge is extended outward by
1e-12 and re-certified with its recorded dual (the leaf(): bk16_lb.mode1_vertices on the NS functional; 'E' with the
recorded Farkas dual; 'Xd' recomputed).  Also checks that the edge leaves tile the cell's face at that edge (exact, so the
extended leaves cover the 1e-12 slab).  Run from any folder.  Output: eslv20_out.json in the current folder."""
import os, sys, json
from fractions import Fraction as F
OWN = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(OWN, '..', 'src'))
sys.argv = [sys.argv[0], 'replay']
import numpy as np
import bk19_emin as EM
ABP = [0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.47, 0.485, 0.5, 0.515, 0.53, 0.54, 0.56, 0.58, 0.6, 0.65, 0.7]
LBP = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
T = {}
for fn in ('emin_A.json', 'emin_B.json'):
    for r in json.load(open(os.path.join(OWN, '..', 'emin', fn))): T[(ABP.index(r['cell'][0]), LBP.index(r['cell'][2]))] = r
D = 1e-12
NEED = [(12, 6, 'hi'), (5, 6, 'lo')]
OUT = {}
for c, j, side in NEED:
    r = T[(c, j)]; em = r['emin_lo']; a0, a1 = r['cell'][0], r['cell'][1]; n = 0; worst = 1e9; fails = 0; facevol = F(0)
    kinds = {}
    for x in r['leaves']:
        lo, hi = np.array(x['lo']), np.array(x['hi'])
        if side == 'lo' and lo[0] != a0: continue
        if side == 'hi' and hi[0] != a1: continue
        facevol += (F(hi[1]) - F(lo[1])) * (F(hi[2]) - F(lo[2])) * (F(hi[3]) - F(lo[3]))
        if side == 'lo': lo[0] = a0 - D
        else: hi[0] = a1 + D
        n += 1; kinds[x['k']] = kinds.get(x['k'], 0) + 1
        if x['k'] == 'Xd':
            ok = EM.d_outside(lo, hi)
        elif x['k'] == 'E':
            ok = EM.leaf(lo, hi, ed=x['ed']) is not None
        else:
            q = EM.leaf(lo, hi, dual=x['d']); v = EM.elin(q['lb']); worst = min(worst, v); ok = v >= em
        if not ok:
            fails += 1
            if x['k'] != 'L': print('E/Xd leaf not closed on extension', x['k'], x['lo'], x['hi'], flush=True)
    root = r['leaves'][0]
    face = (F(LBP[j + 1]) - F(LBP[j])) * (F(0.99) - F(-0.79)) * (F(0.59) - F(0.0))
    OUT['%d,%d,%s' % (c, j, side)] = dict(emin=em, sliver_min=min(worst, em), nleaves=n, below_table=fails, kinds=kinds,
                                         face_volume_exact=(facevol == face))
    print('cell %2d row %d side %s: Emin_lo %.12e, %d edge leaves %s extended by %.0e, min elin %.12e, L below table %d, '
          'edge leaves tile the face (exact volume): %s' % (c, j, side, em, n, kinds, D, worst, fails, facevol == face), flush=True)
json.dump(OUT, open('eslv20_out.json', 'w'), indent=0)   # written to the current folder
