"""bk17_verify.py : replay of bk17_run records (no LP solve) and exact cover check.

Usage: python bk17_verify.py <domain Q|G> replay <stride> <offset> <records...>
       python bk17_verify.py <domain Q|G> cover <records...>
Replay: 'X' (marginal excluded, recomputed), 'Xd' (d outside the R7 square, recomputed), 'E'/'E2' (recorded duals,
certified bound > 0), 'G' (bk17_box.replay_G: ranges re-certified on the recorded range box from the recorded duals,
candidates' Mode 1 vertex values recomputed, theta combination >= 0), 'K' (bk17_kink.kink_check2 re-run with the recorded
range duals and eta; must pass with P0 <= 0.02, the box must lie in D_loc on one side, level [0, S1]).
Any other kind, a missing kind, or a record flagged 'fail' is a FAILURE (no DELEGATE-type pass-through).
Cover: for each root cell and level the records inside the cell must be exactly the leaves of a bisection tree of the cell
(midpoint splits; in domain G also the splits at 1/2 +- A0G in alpha): 0 gaps, 0 overlaps, 0 stuck.
"""
import os, sys, json
DOM = sys.argv[1]
sys.argv = [sys.argv[0], DOM] + sys.argv[2:]
import numpy as np
import bk17_run as RUN          # sets BK16_MODE and imports bk17_cfg (kappa, tau0)
import bk17_box as XB
import bk16_bnb as BB
sys.setrecursionlimit(20000)


def replay(r):
    lo, hi = np.array(r['lo']), np.array(r['hi'])
    k = r.get('k')
    if r.get('fail'): return False, -9
    if k == 'X': return (DOM == 'G' and BB.marg_excluded(lo, hi)), 1.0
    if k == 'Xd': return (RUN.PM and RUN.d_outside_R7(lo, hi)), 1.0
    if k in ('E', 'E2'): return XB.replay_E(r, RUN.PM)
    if k == 'G':
        if DOM == 'G' and not BB.outside_Q(lo, hi): return False, -5
        if DOM == 'Q' and not BB.in_Q(lo, hi): return False, -5
        if DOM == 'Q':
            tq = r.get('tq', RUN.TAU0)
            i = RUN.q_cell(lo, hi)
            if i is None: return False, -11
            need = RUN.TAU0 if RUN.TQ is None else max(0.0, float(RUN.TQ[11 - i]))
            if not (tq >= need): return False, -12
            return XB.replay_G(r, RUN.PM, DOM, tq)
        return XB.replay_G(r, RUN.PM, DOM, RUN.TAU0)
    if k == 'K':
        import bk17_kink as K2
        if DOM != 'Q' or tuple(r['lev']) != (0.0, RUN.S1) or not RUN.inside_Dloc_pm(lo, hi): return False, -6
        side = 1 if lo[2] >= 0 else -1
        if side != r['side']: return False, -7
        res = K2.kink_check2(lo, hi, side, RUN.S1, rec=r)
        return bool(res['ok'] and res['P0'] <= RUN.P0MAX), min(res.get('rcmin', -1), res.get('slope', -1))
    return False, -8


def walk(cell_lo, cell_hi, recs, stats):
    if not recs: stats['gaps'] += 1; stats['gapboxes'].append((cell_lo, cell_hi)); return
    if len(recs) == 1 and recs[0][0] == cell_lo and recs[0][1] == cell_hi: stats['leaves'] += 1; return
    for r in recs:
        if r[0] == cell_lo and r[1] == cell_hi: stats['overlaps'] += 1; return
    for i in range(4):
        cands = [0.5 * (cell_lo[i] + cell_hi[i])]
        if i == 0 and DOM == 'G':
            for x in (0.5 - BB.A0G, 0.5 + BB.A0G):
                if cell_lo[0] < x < cell_hi[0]: cands.insert(0, x)
        for mid in cands:
            left = [r for r in recs if r[1][i] <= mid]; right = [r for r in recs if r[0][i] >= mid]
            if len(left) + len(right) == len(recs) and left and right:
                h1 = list(cell_hi); h1[i] = mid; l2 = list(cell_lo); l2[i] = mid
                walk(cell_lo, tuple(h1), left, stats); walk(tuple(l2), cell_hi, right, stats); return
    stats['stuck'] += 1; stats['stuckboxes'].append((cell_lo, cell_hi, len(recs)))


def cover(files):
    recs = []
    for fn in files:
        for l in open(fn):
            r = json.loads(l)
            recs.append((tuple(r['lo']), tuple(r['hi']), tuple(r['lev'])))
    roots = RUN.roots_Q() if DOM == 'Q' else RUN.roots_G()
    sel = os.environ.get('BK17_SEL')
    if sel:
        a0, a1, l0, l1 = map(float, sel.split(','))
        roots = [r for r in roots if r[0][0] >= a0 and r[1][0] <= a1 and r[0][1] >= l0 and r[1][1] <= l1]
    stats = dict(leaves=0, gaps=0, overlaps=0, stuck=0, gapboxes=[], stuckboxes=[])
    used = 0; vol = 0.0; gapvol = {}
    from collections import defaultdict
    byroot = defaultdict(list)
    for r in recs:
        byroot[r[2]].append(r)
    for lev in RUN.LEVELS:
        R = byroot.get(tuple(lev), [])
        for lo, hi in roots:
            lo, hi = tuple(float(x) for x in lo), tuple(float(x) for x in hi)
            inside = [r for r in R if all(r[0][i] >= lo[i] and r[1][i] <= hi[i] for i in range(4))]
            used += len(inside)
            g0 = stats['gaps']; nb = len(stats['gapboxes'])
            walk(lo, hi, inside, stats)
    print('records %d, used %d; leaves %d gaps %d overlaps %d stuck %d' % (len(recs), used, stats['leaves'], stats['gaps'], stats['overlaps'], stats['stuck']), flush=True)
    return stats


if __name__ == '__main__':
    mode = sys.argv[2]
    if mode == 'replay':
        stride, off = int(sys.argv[3]), int(sys.argv[4]); files = sys.argv[5:]
        bad = 0; n = 0; kinds = {}; minlb = {}; i = -1
        for fn in files:
            for l in open(fn):
                i += 1
                r = json.loads(l); kk = r.get('k'); kinds[kk] = kinds.get(kk, 0) + 1
                if i % stride != off: continue
                if os.path.exists('replay.stop'): print('STOPPED by replay.stop at record', i, flush=True); break
                n += 1
                try:
                    ok, lb = replay(r)
                except Exception as ex:
                    ok, lb = False, -10
                minlb[kk] = min(minlb.get(kk, 1e9), lb)
                if not ok: bad += 1; print('FAIL', fn, i, kk, r['lo'], r['hi'], r['lev'], lb, flush=True)
        print('records %d kinds %s checked %d (stride %d off %d) failures %d min margins %s' % (i + 1, kinds, n, stride, off, bad, minlb), flush=True)
    else:
        st = cover(sys.argv[3:])
        for g in st['gapboxes'][:10]: print('  gap', g)
        for g in st['stuckboxes'][:10]: print('  stuck', g)
