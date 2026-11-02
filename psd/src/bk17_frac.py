"""bk17_frac.py : coverage bookkeeping for a (possibly unfinished) bk17_run.
Usage: python bk17_frac.py <domain Q|G> <tag> <W> [SEL]
For every root cell x level: closed volume (records), pending volume (queue of the checkpoint + open file), never-started
roots.  Volumes are in the coordinates of the run (alpha, lam, dp, dm); the identity closed + pending + unstarted = total is
checked with exact rational arithmetic (Fractions of the doubles).  Prints totals per kind and the pending set summary."""
import sys, os, json
from fractions import Fraction as Fr
DOM = sys.argv[1]; tag = sys.argv[2]; W = int(sys.argv[3])
if len(sys.argv) > 4: os.environ['BK17_SEL'] = sys.argv[4]
sys.argv = [sys.argv[0], DOM]
import bk17_run as RUN


def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= (Fr(b) - Fr(a))
    return v


roots = RUN.roots_Q() if DOM == 'Q' else RUN.roots_G()
sel = os.environ.get('BK17_SEL')
if sel:
    a0, a1, l0, l1 = map(float, sel.split(','))
    roots = [r for r in roots if r[0][0] >= a0 and r[1][0] <= a1 and r[0][1] >= l0 and r[1][1] <= l1]
total = sum(vol(lo, hi) for lo, hi in roots) * len(RUN.LEVELS)
closed = Fr(0); kinds = {}; pend = Fr(0); npend = 0; started = set(); nrec = 0
for wk in range(W):
    base = '%s_w%d' % (tag, wk)
    if not os.path.exists(base + '.state.json'): continue
    st = json.load(open(base + '.state.json'))
    with open(base + '.jsonl', 'rb') as f: data = f.read()[:st['roff']].decode()
    for l in data.splitlines():
        r = json.loads(l); v = vol(r['lo'], r['hi']); closed += v; nrec += 1
        kinds[r['k']] = kinds.get(r['k'], Fr(0)) + v
    for a, b, l, d in st['queue']:
        pend += vol(a, b); npend += 1
    with open(base + '_open.jsonl', 'rb') as f: odata = f.read()[:st['ooff']].decode()
    for l in odata.splitlines():
        r = json.loads(l); pend += vol(r['lo'], r['hi']); npend += 1
print('records %d; closed fraction %.6f; pending (queue + open) fraction %.6f in %d boxes; unaccounted %.6f'
      % (nrec, float(closed / total), float(pend / total), npend, float((total - closed - pend) / total)))
for k, v in sorted(kinds.items()): print('  kind %s: %.6f' % (k, float(v / total)))
