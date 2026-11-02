r"""bk23_cover.py : bk22_cover.py with per-family check counts (targets.json in the working folder: len(U), len(V)).
bk22_cover.py : exact cover of the PL certificate.
 (a) Roots: for each family and each level, the marginal cells of <tag>_jobs.json are exactly the cells of the grid ABP x LBP
     (every pair once), and the d-box dp in [-0.79, 0.99], dm in [0, 0.59] contains the R7 square of every marginal of the
     window (interval check on a fine grid of marginal cells; dm >= 0 by Lemma SYM), the two levels tile [0, 1].
 (b) Per root job and per band k = 0..24: the records certifying band k (mask of 'X', 'Xd', 'E', 'E2', 'ER', 'O' records)
     tile the root: walking the bisection tree given by the recorded paths (each step '<dim>a' = lower half, '<dim>b' = upper
     half, mid = 0.5 (lo + hi) in doubles exactly as the driver), every node is either certifying for k or split with both
     children covered; every record's box equals its path's box (exact double equality).  Uncovered nodes are gaps.
 (c) The open set: depth-limit boxes (<tag>_w*_open.jsonl), stacks of unfinished claimed jobs, unclaimed jobs.
Usage: python bk22_cover.py <tag> [list]
"""
import sys, os, json, glob, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src'))
tag = sys.argv[1]
TGJ = json.load(open(os.environ.get('BK23_TARGETS', 'targets.json'))); NCH = {f: len(TGJ[f]) for f in 'UV'}
J = json.load(open(tag + '_jobs.json')); jobs = J['jobs']; ABP = J['ABP']; LBP = J['LBP']

# ---------------- (a) roots
ok_a = True
for f in 'UV':
    for lev in ([0.01, 1.0], [0.0, 0.01]):
        cells = sorted((tuple(j[1][:2]), tuple(j[2][:2])) for j in jobs if j[0] == f and j[3] == lev)
        want = sorted(((ABP[i], LBP[q]), (ABP[i + 1], LBP[q + 1])) for i in range(len(ABP) - 1) for q in range(len(LBP) - 1))
        if cells != want: ok_a = False; print('ROOT GRID MISMATCH', f, lev)
for j in jobs:
    if j[1][2:] != [-0.79, 0.0] or j[2][2:] != [0.99, 0.59]: ok_a = False; print('D-BOX MISMATCH', j)
from ivarith import IA
import bk16_model as M
import bk19_orun as R
# R7 square: d_a, d_b in [DLO, DHI]; dp = d_a + d_b - tau, dm = d_a - d_b; check on 400 x 250 sub-cells of the window
amin, amax, lmin, lmax = ABP[0], ABP[-1], LBP[0], LBP[-1]
na, nl = 400, 250; worst = [1e9, -1e9, -1e9]
for i in range(na):
    for q in range(nl):
        al = IA(amin + (amax - amin) * i / na, amin + (amax - amin) * (i + 1) / na)
        la = IA(lmin + (lmax - lmin) * q / nl, lmin + (lmax - lmin) * (q + 1) / nl)
        tau = R.tau_of(al, la)
        dpl = float((IA(M.DLO) * 2.0 - tau).lo); dph = float((IA(M.DHI) * 2.0 - tau).hi)
        worst[0] = min(worst[0], dpl); worst[1] = max(worst[1], dph)
dmh = float((IA(M.DHI) - IA(M.DLO)).hi)
print('(a) R7 square over the window: dp in [%.5f, %.5f] (root d-box [-0.79, 0.99]), |dm| <= %.5f (root [0, 0.59], SYM)' % (worst[0], worst[1], dmh))
if not (worst[0] >= -0.79 and worst[1] <= 0.99 and dmh <= 0.59): ok_a = False
print('(a) window [%g, %g] x [%g, %g], %d marginal cells, %d roots: %s' % (amin, amax, lmin, lmax, (len(ABP) - 1) * (len(LBP) - 1), len(jobs), 'OK' if ok_a else 'FAILED'))

# ---------------- (b) per job, per band
cert = {}      # job -> path -> set of bands
boxes = {}     # job -> path -> (lo, hi)
nrec = 0; dup = 0; geo_bad = 0
for fn in sorted(glob.glob(tag + '_w*.jsonl')):
    if fn.endswith('_open.jsonl'): continue
    for l in open(fn):
        r = json.loads(l); nrec += 1
        jb = r['job']; p = r['path']
        s = cert.setdefault(jb, {}).setdefault(p, set())
        for k in r['mask']:
            if k in s: dup += 1
            s.add(k)
        boxes.setdefault(jb, {})[p] = (r['lo'], r['hi'])
openb = {}
for fn in sorted(glob.glob(tag + '_w*_open.jsonl')):
    for l in open(fn):
        r = json.loads(l); openb.setdefault(r['job'], {})[r['path']] = r['mask']
done = set(int(os.path.basename(f)[len(tag) + 1:-5]) for f in glob.glob(os.path.join('claims', tag + '_*.done')))
claimed = set(int(os.path.basename(f)[len(tag) + 1:-6]) for f in glob.glob(os.path.join('claims', tag + '_*.claim')))


def path_box(root_lo, root_hi, p):
    lo = np.array(root_lo, float); hi = np.array(root_hi, float)
    for t in range(0, len(p), 2):
        i = int(p[t]); mid = 0.5 * (lo[i] + hi[i])
        if p[t + 1] == 'a': hi = hi.copy(); hi[i] = mid
        else: lo = lo.copy(); lo[i] = mid
    return lo, hi


complete = []; gaps_tot = 0; overlap_tot = 0; incomplete = {}
for jb in range(len(jobs)):
    f, rlo, rhi, lev = jobs[jb]
    C = cert.get(jb, {}); Ob = openb.get(jb, {})
    # split dims from all known paths
    dim = {}
    bad_dim = 0
    for p in list(C.keys()) + list(Ob.keys()):
        for t in range(0, len(p), 2):
            pre = p[:t]; d = p[t]
            if dim.setdefault(pre, d) != d: bad_dim += 1
    for p, (lo, hi) in boxes.get(jb, {}).items():
        a, b = path_box(rlo, rhi, p)
        if a.tolist() != lo or b.tolist() != hi: geo_bad += 1
    gaps = []; opens = 0; overl = 0

    def walk(p, need):
        have = C.get(p, set())
        rest = [k for k in need if k not in have]
        if p in Ob:
            gaps.append((p, rest, 'open'))
            return
        if not rest: return
        if p not in dim:
            gaps.append((p, rest, 'gap')); return
        d = dim[p]
        walk(p + d + 'a', rest); walk(p + d + 'b', rest)
    walk('', list(range(NCH[f])))
    # overlaps: a band certified at a node and again at a descendant
    for p, s in C.items():
        for t in range(0, len(p), 2):
            if C.get(p[:t], set()) & s: overl += 1
    overlap_tot += overl
    if not gaps and bad_dim == 0: complete.append(jb)
    else: incomplete[jb] = dict(gaps=len(gaps), open=sum(1 for g in gaps if g[2] == 'open'), bad_dim=bad_dim, done=jb in done, claimed=jb in claimed)
    gaps_tot += len(gaps)
print('(b) records %d; duplicate band certificates at one node %d; overlaps (band certified at a node and a descendant) %d; '
      'record boxes != path boxes: %d' % (nrec, dup, overlap_tot, geo_bad))
print('(b) root jobs fully covered (all checks of their family: U %d, V %d): %d of %d' % (NCH['U'], NCH['V'], len(complete), len(jobs)))
print('(b) covered jobs:', complete)
nd = [j for j in incomplete if incomplete[j]['done']]
print('(b) jobs marked done but not covered: %d %s' % (len(nd), nd[:20]))
print('(c) open: incomplete jobs %d (claimed %d, unclaimed %d); depth-limit open boxes %d' % (
    len(incomplete), sum(1 for j in incomplete if j in claimed), sum(1 for j in incomplete if j not in claimed), sum(len(v) for v in openb.values())))
if len(sys.argv) > 2 and sys.argv[2] == 'list':
    json.dump(dict(incomplete=incomplete, open_boxes=openb), open(tag + '_open_set.json', 'w'))
    print('wrote', tag + '_open_set.json')
print('COVER', 'COMPLETE' if (ok_a and not incomplete and geo_bad == 0) else 'INCOMPLETE')
