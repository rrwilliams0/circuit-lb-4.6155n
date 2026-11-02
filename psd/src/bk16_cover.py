"""bk16_cover.py : exact cover check by a bisection walker (exact float comparisons; the B&B splits at the float
midpoint, or at 1/2 +- A0G in alpha).  For each root cell and each level, the records inside the cell must be exactly the
leaves of a bisection tree of the cell: 0 gaps, 0 overlaps.  Usage:
   python bk16_cover.py <roots: G|Q1|Q2|OK2|KINKp|KINKm|DELEG:<file>> <levels comma list> <records files...>"""
import sys, json
import numpy as np
sys.setrecursionlimit(10000)
import bk16_bnb as BB
import bk16_model as M


def walk(cell_lo, cell_hi, recs, stats, depth=0):
    if not recs: stats['gaps'] += 1; stats['gapboxes'].append((cell_lo, cell_hi)); return
    if len(recs) == 1 and recs[0][0] == cell_lo and recs[0][1] == cell_hi: stats['leaves'] += 1; return
    for r in recs:
        if r[0] == cell_lo and r[1] == cell_hi: stats['overlaps'] += 1; return
    for i in range(4):
        cands = [0.5 * (cell_lo[i] + cell_hi[i])]
        if i == 0:
            for x in (0.5 - BB.A0G, 0.5 + BB.A0G):
                if cell_lo[0] < x < cell_hi[0]: cands.insert(0, x)
        for mid in cands:
            left = [r for r in recs if r[1][i] <= mid]; right = [r for r in recs if r[0][i] >= mid]
            if len(left) + len(right) == len(recs) and left and right:
                h1 = list(cell_hi); h1[i] = mid; l2 = list(cell_lo); l2[i] = mid
                walk(cell_lo, tuple(h1), left, stats, depth + 1); walk(tuple(l2), cell_hi, right, stats, depth + 1); return
    stats['stuck'] += 1; stats['stuckboxes'].append((cell_lo, cell_hi, len(recs)))


def main():
    rootspec = sys.argv[1]; levels = [float(x) for x in sys.argv[2].split(',')]
    recs = []
    for fn in sys.argv[3:]:
        for l in open(fn):
            r = json.loads(l)
            if 'DELEGATE' in r: continue
            lev = tuple(r.get('lev', (levels[0], levels[-1])))
            recs.append((tuple(r['lo']), tuple(r['hi']), lev, r.get('side', 0)))
    if rootspec == 'G': roots = BB.roots_grid(BB.ABP, BB.LBP, 2)
    elif rootspec == 'Q1': roots = BB.roots_Q1()
    elif rootspec == 'Q2': roots = BB.roots_Q2()
    elif rootspec == 'OK2': roots = BB.roots_grid([0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7], [0.0, 0.02, 0.05, 0.1, 0.25], 2)
    elif rootspec.startswith('DELEG:'):
        roots = [(json.loads(l)['lo'], json.loads(l)['hi']) for l in open(rootspec[6:]) if 'DELEGATE' in json.loads(l)]
    elif rootspec in ('KINKp', 'KINKm'):
        import itertools
        import bk16_kink2 as K
        side = 1 if rootspec == 'KINKp' else -1
        dpr = (0.0, K.WP) if side > 0 else (-K.WP, 0.0)
        n = (12, 2, 2, 8); rng_ = [K.QA, K.QL, dpr, (-K.WM, K.WM)]
        grids = [np.linspace(r[0], r[1], k + 1) for r, k in zip(rng_, n)]
        for g, r in zip(grids, rng_): g[0], g[-1] = r[0], r[1]
        roots = [([grids[i][idx[i]] for i in range(4)], [grids[i][idx[i] + 1] for i in range(4)]) for idx in itertools.product(*[range(k) for k in n])]
        recs = [r for r in recs if r[3] == side]
    stats = dict(leaves=0, gaps=0, overlaps=0, stuck=0, gapboxes=[], stuckboxes=[])
    levs = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]
    used = 0
    for lev in levs:
        R = [r for r in recs if r[2] == lev]
        for lo, hi in roots:
            lo, hi = tuple(float(x) for x in lo), tuple(float(x) for x in hi)
            inside = [r for r in R if all(r[0][i] >= lo[i] and r[1][i] <= hi[i] for i in range(4))]
            used += len(inside)
            walk(lo, hi, inside, stats)
    print('records %d, used %d; leaves %d gaps %d overlaps %d stuck %d' % (len(recs), used, stats['leaves'], stats['gaps'], stats['overlaps'], stats['stuck']))
    for g in stats['gapboxes'][:5]: print('  gap', g)
    for g in stats['stuckboxes'][:5]: print('  stuck', g)


if __name__ == '__main__':
    main()
