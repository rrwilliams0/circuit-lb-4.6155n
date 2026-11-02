r"""bk19_gcover.py : exact cover of the G* domain by closed records and pending boxes.

Usage (in S\run):  python ..\src\bk19_gcover.py <bk17 tags (comma list)> <bk19 tags (comma list)> [list]
Every root-level (root of bk17_run.roots_G() x level) must be exactly the union of the leaves of a bisection tree (midpoint
splits, and in alpha also the breakpoints 1/2 +- A0G) whose leaves are
  * closed records: the files <tag>_w<k>.jsonl up to the offset of the checkpoint <tag>_w<k>.state.json, and the files
    <tag>_w<k>.jsonl up to the offset of the checkpoint (bk19_grun format);
  * pending boxes: the checkpoint queues (only for the jobs not yet re-processed is this relevant: their boxes are the
    jobs of the job files), the worker stacks (current job), the jobs never claimed, and open (depth-limit) boxes.
A the queue box is NOT counted when the job list contains it (then the job's own status decides).
Prints leaves / gaps / overlaps / stuck, the number of fully closed root-levels, the pending boxes and their volume share.
With 'list', writes the pending boxes to gcover_pending.json.
"""
import sys, os, json, glob, bisect
from fractions import Fraction as Fr
from collections import defaultdict
t17 = [t for t in sys.argv[1].split(',') if t]; t19 = [t for t in sys.argv[2].split(',') if t]
want_list = len(sys.argv) > 3 and sys.argv[3] == 'list'
os.environ['BK17_GPM'] = '1'; os.environ['BK17_GLEVELS'] = '0:0.01,0.01:1'
sys.argv = [sys.argv[0], 'G']
import numpy as np
import bk17_run as RUN
import bk17_verify as V
V.DOM = 'G'
sys.setrecursionlimit(20000)
G = [RUN.ABP, RUN.LBP, RUN.GDP, RUN.GDM]


def rootkey(lo):
    return tuple(bisect.bisect_right(G[i], lo[i]) - 1 for i in range(4))


def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= Fr(b) - Fr(a)
    return v


items = defaultdict(list); nrec = 0; pend = []
# records that FAILED the replay (bk19_verify.py, the check fix) are not counted; their boxes are re-done as jobs of tag H3
EXCL = set()
if os.path.exists('excluded_records.json'):
    for base, lo, hi, lev in json.load(open('excluded_records.json')): EXCL.add((base, tuple(lo), tuple(hi), tuple(lev)))
nexcl = 0
kinds = defaultdict(int)
jobboxes = set()
for tag in t19:
    jobs = json.load(open(tag + '_jobs.json'))
    for a, b, lev, d in jobs: jobboxes.add((tuple(a), tuple(b), tuple(lev)))
for tag in t17:
    for st_f in sorted(glob.glob('%s_w*.state.json' % tag)):
        base = st_f[:-len('.state.json')]
        st = json.load(open(st_f))
        data = open(base + '.jsonl', 'rb').read()[:st['roff']].decode()
        for l in data.splitlines():
            r = json.loads(l); nrec += 1; kinds[r['k']] += 1
            items[(rootkey(r['lo']), tuple(r['lev']))].append((tuple(r['lo']), tuple(r['hi']), 'R'))
        for a, b, lev, d in st['queue']:
            if (tuple(a), tuple(b), tuple(lev)) in jobboxes: continue
            pend.append((a, b, lev))
for tag in t19:
    jobs = json.load(open(tag + '_jobs.json'))
    claimed_cur = set(); done = set()
    for st_f in sorted(glob.glob('%s_w*.state.json' % tag)):
        base = st_f[:-len('.state.json')]
        st = json.load(open(st_f))
        data = open(base + '.jsonl', 'rb').read()[:st['roff']].decode()
        for l in data.splitlines():
            r = json.loads(l)
            if (base, tuple(r['lo']), tuple(r['hi']), tuple(r['lev'])) in EXCL: nexcl += 1; continue
            nrec += 1; kinds[r['k']] += 1
            items[(rootkey(r['lo']), tuple(r['lev']))].append((tuple(r['lo']), tuple(r['hi']), 'R'))
        od = open(base + '_open.jsonl', 'rb').read()[:st['ooff']].decode() if os.path.exists(base + '_open.jsonl') else ''
        for l in od.splitlines():
            r = json.loads(l); pend.append((r['lo'], r['hi'], r['lev']))
        done |= set(st['done'])
        if st['cur'] is not None:
            claimed_cur.add(st['cur'])
            for a, b, lev, d in st['stack']: pend.append((a, b, lev))
    for i, (a, b, lev, d) in enumerate(jobs):
        if i in done or i in claimed_cur: continue
        if os.path.exists(os.path.join('claims', '%s_%d.claim' % (tag, i))):
            print('WARNING orphan claim (claimed, not done, not current in any checkpoint):', tag, i)
        pend.append((a, b, lev))
pvol = Fr(0)
for a, b, lev in pend:
    items[(rootkey(a), tuple(lev))].append((tuple(a), tuple(b), 'P')); pvol += vol(a, b)
stats = dict(leaves=0, gaps=0, overlaps=0, stuck=0, gapboxes=[], stuckboxes=[])
full = 0; withp = 0; total = Fr(0); ROWS = defaultdict(lambda: [0, 0])
for lo, hi in RUN.roots_G():
    lo = tuple(float(x) for x in lo); hi = tuple(float(x) for x in hi)
    for lev in RUN.LEVELS:
        it = items.get((rootkey(lo), tuple(lev)), [])
        total += vol(lo, hi)
        V.walk(lo, hi, it, stats)
        g0 = stats['gaps'] + stats['overlaps'] + stats['stuck']
        if any(x[2] == 'P' for x in it): withp += 1; ROWS[lo[1]][1] += 1
        else: full += 1; ROWS[lo[1]][0] += 1
print('excluded (replay-failed) records', nexcl)
print('records %d %s, pending %d; leaves %d gaps %d overlaps %d stuck %d' % (nrec, dict(kinds), len(pend), stats['leaves'], stats['gaps'], stats['overlaps'], stats['stuck']))
print('root-levels fully closed %d, with pending boxes %d (of %d); pending volume fraction %.6f' % (full, withp, 2 * len(RUN.roots_G()), float(pvol / total)))
for g in stats['gapboxes'][:5]: print('  gap', g)
for g in stats['stuckboxes'][:5]: print('  stuck', g)
for lam0 in sorted(ROWS): print('  lam row from %.4f: root-levels closed %d, with pending %d' % (lam0, ROWS[lam0][0], ROWS[lam0][1]))
if want_list: json.dump(pend, open('gcover_pending.json', 'w'))
