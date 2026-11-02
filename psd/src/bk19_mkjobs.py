r"""bk19_mkjobs.py : the job list of G* for bk19_grun.py.
Usage: python bk19_mkjobs.py <tag>    (in S\run; reads the G checkpoints copied there)
Jobs = (a) every pending box of the G checkpoints listed in STARTED (queue entries, at their depth), and
       (b) every root-level (root of bk17_run.roots_G() x level) that belongs to no the worker in STARTED.
the workers (selection, W, worker): their root-levels are accounted for by their records (up to the checkpoint offset)
plus their queue.  GE (still running at job-creation time) is excluded from (b); its leftovers become a later job file.
Order: by lam (low first), then by |alpha - 1/2| (small first)."""
import sys, os, json
os.environ['BK17_GPM'] = '1'; os.environ['BK17_GLEVELS'] = '0:0.01,0.01:1'
tag = sys.argv[1]
sys.argv = [sys.argv[0], 'G']
import bk17_run as RUN
STARTED = [('GB1', '0.3,0.7,0.1,0.25', 2, [0, 1]), ('GB2', '0.3,0.7,0.05,0.1', 2, [0]), ('GC', '0.44,0.56,0,0.005', 1, [0]),
           ('GD', '0.44,0.56,0.005,0.01', 1, [0]), ('GE', '0.44,0.56,0.01,0.02', 1, [0])]
roots = RUN.roots_G()
started = set(); jobs = []
for t, sel, W, wks in STARTED:
    a0, a1, l0, l1 = map(float, sel.split(','))
    rs = [r for r in roots if r[0][0] >= a0 and r[1][0] <= a1 and r[0][1] >= l0 and r[1][1] <= l1]
    for wk in wks:
        for i, r in enumerate(rs):
            if i % W == wk: started.add((tuple(r[0]), tuple(r[1])))
        if t == 'GE': continue
        st = json.load(open('%s_w%d.state.json' % (t, wk)))
        for a, b, lev, d in st['queue']: jobs.append([a, b, lev, d])
        print(t, wk, 'pending', len(st['queue']))
nb = 0
for lo, hi in roots:
    if (tuple(lo), tuple(hi)) in started: continue
    for lev in RUN.LEVELS:
        jobs.append([list(lo), list(hi), list(lev), 0]); nb += 1
print('unstarted root-levels', nb, 'started roots', len(started), 'of', len(roots))
jobs.sort(key=lambda j: (j[0][1], abs(0.5 * (j[0][0] + j[1][0]) - 0.5), j[0][0], j[0][2], j[0][3], j[2][0]))
json.dump(jobs, open(tag + '_jobs.json', 'w'))
print('jobs', len(jobs))
