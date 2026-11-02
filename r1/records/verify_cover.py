"""verify_cover.py : coverage walker for the records.
Reads the roots (the 17,326 open boxes of bnb9_RVB.json) and the record files rec_*.jsonl.  Every root must be resolved:
a box is resolved if it has a closure record (rule in CLOSE), or a SPLIT record whose two children (bisection of coordinate j
at m, exactly as bk10_run does it) are both resolved.  Prints counts per rule, unresolved boxes (written to
open_remaining.json if any), and checks that no record is a closure of a box that was also split (consistency).
usage: python verify_cover.py rec_W0.jsonl rec_W1.jsonl [...]"""
import sys, json
from collections import Counter
CLOSE = {'XY', 'INF-PSD', 'INF-K', 'LOC', 'CORNER', 'TAY2', 'TAY1', 'TAY3'}
roots = json.load(open('bnb9_RVB.json'))['open']
rec = {}; dup = 0; conflict = 0
for f in sys.argv[1:]:
    for line in open(f):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except ValueError: continue                  # a torn last line after a crash
        k = (tuple(d['lo']), tuple(d['hi']))
        if k in rec:
            dup += 1
            if (rec[k]['r'] in CLOSE) != (d['r'] in CLOSE) and 'OPEN' not in (rec[k]['r'], d['r']): conflict += 1
            if rec[k]['r'] in CLOSE: continue            # keep a closure if there is one
        rec[k] = d
cnt = Counter(); unresolved = []
stack = [(tuple(a), tuple(b), 0) for a, b in roots]
maxdepth = 0
while stack:
    lo, hi, dep = stack.pop(); maxdepth = max(maxdepth, dep)
    d = rec.get((lo, hi))
    if d is None or d['r'] == 'OPEN': unresolved.append([list(lo), list(hi)]); cnt['UNRESOLVED'] += 1; continue
    if d['r'] in CLOSE: cnt[d['r']] += 1; continue
    if d['r'] == 'SPLIT':
        j, m = d['j'], d['m']
        lo2 = list(lo); hi2 = list(hi); lo2[j] = m; hi2[j] = m
        if not (lo[j] < m < hi[j]): unresolved.append([list(lo), list(hi)]); cnt['BADSPLIT'] += 1; continue
        cnt['SPLIT'] += 1
        stack.append((lo, tuple(hi2), dep + 1)); stack.append((tuple(lo2), hi, dep + 1))
        continue
    unresolved.append([list(lo), list(hi)]); cnt['OTHER'] += 1
print('roots %d, records %d (duplicates %d, closure/split conflicts %d), max split depth %d' % (len(roots), len(rec), dup, conflict, maxdepth))
print('resolution tree: %s' % dict(cnt))
print('unresolved boxes: %d' % len(unresolved))
json.dump(dict(tag='remaining', boxes=0, stats={}, time=0.0, open=unresolved), open('open_remaining.json', 'w'))
