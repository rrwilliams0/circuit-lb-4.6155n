"""bk17_coverq.py : exact cover check of domain Q, including the pending boxes of the checkpoints.
Every root cell x level of roots_Q must be exactly the union of the leaves of a bisection tree whose leaves are closed records
or pending boxes (queue entries of the checkpoints); the walker uses exact float comparisons (bk17_verify.walk).
Reports: leaves, gaps, overlaps, stuck; root-levels fully closed (no pending box); pending volume fraction.
Usage: python bk17_coverq.py <tags...>   (record files <tag>_w<k>.jsonl, checkpoints <tag>_w<k>.state.json)"""
import sys, os, json, bisect, glob
from fractions import Fraction as Fr
from collections import defaultdict
tags = sys.argv[1:]
sys.argv = [sys.argv[0], 'Q']
import numpy as np
import bk17_run as RUN
import bk17_verify as V
V.DOM = 'Q'
A = list(np.round(np.linspace(0.47, 0.53, 13), 10)); A[0], A[-1] = 0.47, 0.53
L = [0.0, 0.0025, 0.005]
DP = [-0.55, -0.28, -0.14, -0.07, -0.035, -0.01, -0.005, 0.0, 0.005, 0.01, 0.035, 0.07, 0.14, 0.28, 0.72]
DM = [-0.59, -0.3, -0.16, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.16, 0.3, 0.59]
G = [A, L, DP, DM]
def rootkey(lo):
    return tuple(bisect.bisect_right(G[i], lo[i]) - 1 for i in range(4))
items = defaultdict(list); npend = 0; nrec = 0; pendvol = Fr(0)
def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= Fr(b) - Fr(a)
    return v
for tag in tags:
    for st_f in sorted(glob.glob('%s_w*.state.json' % tag)):
        base = st_f[:-len('.state.json')]
        st = json.load(open(st_f))
        data = open(base + '.jsonl', 'rb').read()[:st['roff']].decode()
        for l in data.splitlines():
            r = json.loads(l); nrec += 1
            items[(rootkey(r['lo']), tuple(r['lev']))].append((tuple(r['lo']), tuple(r['hi']), 'R'))
        for a, b, lev, d in st['queue']:
            npend += 1; pendvol += vol(a, b)
            items[(rootkey(a), tuple(lev))].append((tuple(a), tuple(b), 'P'))
        od = open(base + '_open.jsonl', 'rb').read()[:st['ooff']].decode()
        for l in od.splitlines():
            r = json.loads(l); npend += 1; pendvol += vol(r['lo'], r['hi'])
            items[(rootkey(r['lo']), tuple(r['lev']))].append((tuple(r['lo']), tuple(r['hi']), 'P'))
stats = dict(leaves=0, gaps=0, overlaps=0, stuck=0, gapboxes=[], stuckboxes=[])
full = 0; withp = 0; total = Fr(0)
for idx in np.ndindex(12, 2, 14, 14):
    lo = tuple(float(G[i][idx[i]]) for i in range(4)); hi = tuple(float(G[i][idx[i] + 1]) for i in range(4))
    for lev in RUN.LEVELS:
        it = items.get((tuple(idx), tuple(lev)), [])
        total += vol(lo, hi)
        V.walk(lo, hi, it, stats)
        if any(x[2] == 'P' for x in it): withp += 1
        else: full += 1
print('records %d, pending %d; leaves %d gaps %d overlaps %d stuck %d' % (nrec, npend, stats['leaves'], stats['gaps'], stats['overlaps'], stats['stuck']))
print('root-levels fully closed %d, with pending boxes %d (of %d); pending volume fraction %.6f' % (full, withp, 9408, float(pendvol / total)))
for g in stats['gapboxes'][:5]: print('  gap', g)
for g in stats['stuckboxes'][:5]: print('  stuck', g)
