"""bk13_verify.py: replay of a B&B record (no LP solving, no float choices re-made).
For every closed record: recompute the rigorous vertex lower values of the recorded candidates (Mode 1 with the recorded
doubles (ya, yb); Mode 2 with the recorded basis) over the recorded box, form sum_c theta_c (V_c - target) with outward
rounding, and check >= 0 at all 16 vertices.  For 'loc' records: recheck inside_Dloc.  Coverage: the boxes must be pairwise
distinct, lie in the domain, and their exact (rational) volumes must add up to the volume of the domain
[0,1] x [0,1] x [1/2,2]^2 = 9/4.  Usage: python bk13_verify.py <tag> [stride offset]"""
import sys, json
from fractions import Fraction as Fr
import numpy as np
from ivarith import dn
from bk13_lb import Shared, mode1, mode2b
from bk13_bnb import inside_Dloc, in_Q, outside_Q, TAU0, TG

tag = sys.argv[1]
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
vol = Fr(0); seen = set(); nrec = 0; bad = 0; nchk = 0; minlb = {'G': 1.0, 'L1': 1.0}; kinds = {}
for i, line in enumerate(open(tag + '_closed.jsonl')):
    r = json.loads(line); lo = np.array(r['lo']); hi = np.array(r['hi'])
    key = tuple(r['lo'] + r['hi'])
    if key in seen: print('DUPLICATE', key); bad += 1; continue
    seen.add(key); nrec += 1; kinds[r['k']] = kinds.get(r['k'], 0) + 1
    if not (lo[0] >= 0 and hi[0] <= 1 and lo[1] >= 0 and hi[1] <= 1 and lo[2] >= 0.5 and hi[3] <= 2 and lo[3] >= 0.5 and hi[2] <= 2):
        print('OUTSIDE DOMAIN', key); bad += 1
    v = Fr(1)
    for a, b in zip(r['lo'], r['hi']): v *= Fr(b) - Fr(a)
    vol += v
    if i % stride != off: continue
    nchk += 1
    if r['k'] == 'loc':
        if not (in_Q(lo, hi) and inside_Dloc(lo, hi)): print('LOC FAIL', key); bad += 1
        continue
    target = TAU0 if r['k'] == 'L1' else TG
    if r['k'] == 'L1' and not in_Q(lo, hi): print('L1 box not in Q', key); bad += 1
    if r['k'] == 'G' and not outside_Q(lo, hi): print('G box meets Q', key); bad += 1
    if r['t'] != target: print('TARGET MISMATCH', key); bad += 1
    sh = Shared(lo, hi)
    s = np.zeros(16)
    for c, th in zip(r['c'], r['th']):
        if c[0] == 'm1': V = mode1(sh, float(c[1]), float(c[2]))
        else: V = mode2b(sh, tuple(c[1]))
        V = dn(V - target)
        s = dn(s + dn(th * V))
    lb = float(s.min())
    minlb[r['k']] = min(minlb[r['k']], lb)
    if not (lb >= 0): print('BOUND FAIL', key, lb); bad += 1
print('records %d (kinds %s), checked %d (stride %d offset %d), failures %d' % (nrec, kinds, nchk, stride, off, bad))
print('min verified margin above target: G %.3e  L1 %.3e' % (minlb['G'], minlb['L1']))
print('total volume %s = %.12f ; domain volume 9/4 ; equal: %s' % (vol if vol.denominator < 10 ** 6 else '...', float(vol), vol == Fr(9, 4)))
