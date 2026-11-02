r"""bk19_omkjobs.py : job list for bk19_orun.py.  Usage: python bk19_omkjobs.py <tag> <fams e.g. UV> <bands e.g. 0-24>
Roots: every marginal cell of bk17_run's alpha x lam grid over the WIN window [0.3, 0.7] x [0, 0.25] (Q included) times the whole
d-box dp in [-0.79, 0.99], dm in [0, 0.59] (covers the R7 square on the window; dm >= 0 by Lemma SYM), times the two levels
[0, 0.01], [0.01, 1]; times families and bands.  Order: band, then |alpha - 1/2| + lam."""
import sys, json, os
os.environ['BK17_GPM'] = '1'
tag, fams, bands = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], 'G']
import bk17_run as RUN
b0, b1 = map(int, bands.split('-'))
jobs = []
for k in range(b0, b1 + 1):
    cells = []
    for i in range(len(RUN.ABP) - 1):
        for j in range(len(RUN.LBP) - 1):
            cells.append(([RUN.ABP[i], RUN.LBP[j], -0.79, 0.0], [RUN.ABP[i + 1], RUN.LBP[j + 1], 0.99, 0.59]))
    cells.sort(key=lambda c: abs(0.5 * (c[0][0] + c[1][0]) - 0.5) + c[0][1])
    for f in fams:
        for lo, hi in cells:
            for lev in ([0.0, 0.01], [0.01, 1.0]):
                jobs.append([f, k, lo, hi, lev, 0])
json.dump(jobs, open(tag + '_jobs.json', 'w'))
print('jobs', len(jobs))
