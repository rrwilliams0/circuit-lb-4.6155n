r"""bk19_emin.py : certified lower bounds Emin_lo(P) of the least Lemma-L excess on marginal cells P.

For a marginal cell P = [a0, a1] x [l0, l1] (alpha, lam):
   Emin_lo(P) <= inf { E^lin(m) : m a u-star with marginal in P, ratios in the R7 square, NS row },
   E^lin(m) = (nsc + nsg.m)/3 - mu0  (Lemma L; E(m) >= E^lin(m), the check/the).
Method: best-first branch and bound over (alpha, lam, dp, dm) (pm coordinates; dm >= 0 suffices by Lemma SYM, E^lin and the
slice are symmetric under a <-> b) with level [0, 1]; each leaf is
   'Xd' (d outside the R7 square), 'E' (slice set empty: certified Farkas bound > 0), or
   'L'  (certified Mode 1 lower bound lb of nsc + nsg.m on the box, recorded dual),
and Emin_lo(P) = min over 'L' leaves of lb/3 - mu0 (rounded down).  The leaves tile P x [-0.79, 0.99] x [0, 0.59].
Usage:  python bk19_emin.py run <out.json> <maxleaves> <cells...: a0,a1,l0,l1>       (one output file per call)
        python bk19_emin.py replay <files...>                                          (re-certify every leaf, exact tiling)
"""
import os, sys, json, heapq, time
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
from fractions import Fraction as Fr
import bk17_cfg
from ivarith import IA, dn, up
import bk16_model as M
import bk16_lb as B
from bk13_lb import T1, MU0
import bk19_fast
import bk17_box as XB
bk19_fast.install()
LEV = (0.0, 1.0)
DP0, DP1, DM0, DM1 = -0.79, 0.99, 0.0, 0.59
TW = np.array([0.01, 0.01, 0.05, 0.05])


def tau_of(al, la):
    q = 1.0 - la; S = q * (2.0 - al) + la
    return 2.5 - (al * q * S.inv()) * 1.5


def d_outside(lo, hi):
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); tau = tau_of(al, la)
    dp = IA(lo[2], hi[2]); dm = IA(lo[3], hi[3])
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    return any(float(d.hi) < M.DLO or float(d.lo) > M.DHI for d in (da, db))


def leaf(lo, hi, dual=None, ed=None):
    if d_outside(lo, hi): return dict(k='Xd')
    bx = B.Box(lo, hi, pm=True); bx._pts = None
    zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if ed is not None:
        V = B.mode1_vertices(bx, zv, zero, zv, zero, LEV, ed); return dict(k='E', ed=ed, lb=float(V.min())) if float(V.min()) > 0 else None
    if dual is None:
        e = XB.emptiness(bx, LEV)
        if e is not None: return e
    vp, vb = bx.Qp['nsg'], bx.Qb['nsg']; kp, kb = bx.Qp['nsc'], bx.Qb['nsc']
    if dual is not None:
        V = B.mode1_vertices(bx, vp, kp, vb, kb, LEV, dual); return dict(k='L', d=dual, lb=float(V.min()))
    best = None
    lb, d = B.certified_min(bx, vp, vb, kp, kb, LEV, slopes=True)
    if lb is not None: best = (lb, d)
    for aff in (True, False):
        sp = XB.robust_dual(bx, vp, kp, LEV, affine=aff)
        if sp is None: continue
        V = B.mode1_vertices(bx, vp, kp, vb, kb, LEV, sp)
        if best is None or float(V.min()) > best[0]: best = (float(V.min()), sp)
    return dict(k='L', d=best[1], lb=best[0])


def elin(lb):
    return float(dn(dn(lb / 3.0) - float(MU0.hi)))


def run_cell(cell, maxleaves):
    a0, a1, l0, l1 = cell
    root = (np.array([a0, l0, DP0, DM0]), np.array([a1, l1, DP1, DM1]))
    heap = []; closed = []; cnt = 0
    def push(lo, hi):
        nonlocal cnt
        r = leaf(lo, hi); cnt += 1
        if r is None or r['k'] != 'L': closed.append(dict(lo=lo.tolist(), hi=hi.tolist(), **(r or {}))); return
        heapq.heappush(heap, (r['lb'], cnt, lo.tolist(), hi.tolist(), r))
    push(*root)
    t0 = time.time()
    while heap and len(heap) + len(closed) < maxleaves:
        lbv, _, lo, hi, r = heap[0]
        lo, hi = np.array(lo), np.array(hi)
        w = (hi - lo) / TW; i = int(np.argmax(w))
        if w[i] < 1e-3: break
        heapq.heappop(heap)
        mid = 0.5 * (lo[i] + hi[i]); h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        push(lo, h1); push(l2, hi)
    leaves = closed + [dict(lo=lo, hi=hi, **r) for (_, _, lo, hi, r) in heap]
    Ls = [x['lb'] for x in leaves if x['k'] == 'L']
    em = elin(min(Ls)) if Ls else 1.0
    return dict(cell=list(cell), emin_lo=em, nleaves=len(leaves), leaves=leaves, sec=time.time() - t0)


def vol(lo, hi):
    v = Fr(1)
    for a, b in zip(lo, hi): v *= Fr(b) - Fr(a)
    return v


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        out, maxl = sys.argv[2], int(sys.argv[3])
        res = json.load(open(out)) if os.path.exists(out) else []
        have = set(tuple(r['cell']) for r in res)
        cells = sys.argv[4:]
        if len(cells) == 1 and cells[0].endswith('.json'): cells = [','.join(map(str, c)) for c in json.load(open(cells[0]))]
        for c in cells:
            if os.path.exists(out + '.stop'): print('stopped by stop file', flush=True); break
            cell = tuple(float(x) for x in c.split(','))
            if cell in have: continue
            r = run_cell(cell, maxl); res.append(r)
            print('cell %s Emin_lo %.6e leaves %d %.0fs' % (cell, r['emin_lo'], r['nleaves'], r['sec']), flush=True)
            json.dump(res, open(out + '.tmp', 'w')); os.replace(out + '.tmp', out)
    else:
        bad = 0; n = 0
        for fn in sys.argv[2:]:
            for r in json.load(open(fn)):
                a0, a1, l0, l1 = r['cell']; tot = vol([a0, l0, DP0, DM0], [a1, l1, DP1, DM1])
                s = Fr(0); mins = []
                for x in r['leaves']:
                    lo, hi = np.array(x['lo']), np.array(x['hi']); s += vol(lo, hi); n += 1
                    if not (lo[0] >= a0 and hi[0] <= a1 and lo[1] >= l0 and hi[1] <= l1 and lo[2] >= DP0 and hi[2] <= DP1 and lo[3] >= DM0 and hi[3] <= DM1): bad += 1; continue
                    if x['k'] == 'Xd':
                        if not d_outside(lo, hi): bad += 1
                    elif x['k'] == 'E':
                        q = leaf(lo, hi, ed=x['ed'])
                        if q is None: bad += 1
                    else:
                        q = leaf(lo, hi, dual=x['d']); mins.append(q['lb'])
                em = elin(min(mins)) if mins else 1.0
                ok = (s == tot) and em >= r['emin_lo']
                if not ok: bad += 1
                print('cell %s recorded %.6e replayed %.6e volume %s' % (r['cell'], r['emin_lo'], em, 'exact' if s == tot else 'MISMATCH'), flush=True)
        print('leaves %d failures %d' % (n, bad))
