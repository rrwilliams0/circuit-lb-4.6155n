r"""bk19_hprog.py: progress of G* jobs by volume (alpha, lam, dp, dm) x levels: done jobs, in-progress, unclaimed."""
import json, os, glob, sys
def vol(a, b):
    v = 1.0
    for x, y in zip(a, b): v *= y - x
    return v
for tag in sys.argv[1:]:
    jobs = json.load(open(tag + '_jobs.json'))
    done = set(); cur = set(); cpu = 0
    for f in glob.glob(tag + '_w*.state.json'):
        s = json.load(open(f)); done |= set(s['done']); cpu += s['stats']['cpu']
        if s['cur'] is not None: cur.add(s['cur'])
    V = [vol(j[0], j[1]) for j in jobs]; T = sum(V)
    vd = sum(V[i] for i in done); vc = sum(V[i] for i in cur)
    print('%s jobs %d done %d (vol %.4f) current %d (vol %.4f) cpu %.2f h' % (tag, len(jobs), len(done), vd / T, len(cur), vc / T, cpu / 3600))
