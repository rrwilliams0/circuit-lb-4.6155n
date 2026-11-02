"""bk17_gtotal.py: overall closed fraction of G* (all levels) from the G runs with disjoint selections."""
import os, sys, json, subprocess
from fractions import Fraction as Fr
sys.argv = [sys.argv[0], 'G']
import bk17_run as RUN
def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= (Fr(b) - Fr(a))
    return v
roots = RUN.roots_G(); tot = sum(vol(a, b) for a, b in roots) * len(RUN.LEVELS)
runs = [('GB1', 2), ('GB2', 2), ('GC', 1), ('GD', 1)]
closed = Fr(0); kinds = {}; nrec = 0
for tag, W in runs:
    for wk in range(W):
        base = '../run/%s_w%d' % (tag, wk)
        if not os.path.exists(base + '.state.json'): continue
        st = json.load(open(base + '.state.json'))
        data = open(base + '.jsonl', 'rb').read()[:st['roff']].decode()
        for l in data.splitlines():
            r = json.loads(l); v = vol(r['lo'], r['hi']); closed += v; nrec += 1; kinds[r['k']] = kinds.get(r['k'], Fr(0)) + v
print('G* records %d, closed volume fraction %.5f of the whole G* domain (window minus Q, dm >= 0, 2 levels)' % (nrec, float(closed / tot)))
for k, v in sorted(kinds.items()): print('  %s %.5f' % (k, float(v / tot)))
