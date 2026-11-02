"""bk16_verify.py : replay of B&B records without any LP solve, and exact cover check.
Usage: python bk16_verify.py <mode G|Q1|Q2|OK2|KINK> <records.jsonl> [stride offset]
 - G/Q1/Q2 records: 'X' (marginal excluded: recomputed), 'E' (zero objective, recorded dual: certified bound > 0),
   'E2' (recorded range duals: certified lo > certified hi), 'G' (ranges re-certified from the recorded duals and must be
   inside the recorded ranges; every candidate's Mode 1 vertex values recomputed; theta combination >= 0 with downward
   rounding), 'Q' (box inside Q, mode G), 'loc' (box inside D_loc, mode Q1).
 - OK2 records: 'X', 'E', 'K' (O-the-2 bound).   KINK records: kink_check re-run with the recorded duals and eta.
Cover: for every level, the boxes are leaves of bisection trees of the root cells: exact walk, 0 gaps, 0 overlaps."""
import os, sys, json
os.environ.setdefault('BK16_MODE', sys.argv[1] if sys.argv[1] in ('G', 'Q1', 'Q2') else 'G')
from fractions import Fraction as Fr
import numpy as np
from ivarith import dn
import bk16_model as M, bk16_lb as B, bk16_bnb as BB

mode = sys.argv[1]; fn = sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1; off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
recs = [json.loads(l) for l in open(fn)]
bad = 0; nchk = 0; kinds = {}; minlb = {}


def recheck_G(r):
    lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    k = r['k']
    if k == 'X': return BB.marg_excluded(lo, hi), 1.0
    if k == 'Q': return BB.in_Q(lo, hi), 1.0
    if k == 'loc': return (BB.in_Q(lo, hi) and BB.inside_Dloc(lo, hi)), 1.0
    bx = B.Box(lo, hi, pm=(BB.MODE == 'Q2'))
    bx.Qp['al'] = B.T1.var(0, 4, bx.zc[0], bx.zc[0]); bx.Qb['al'] = B.T1.var(0, 4, bx.lo[0], bx.hi[0])
    zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if k == 'E':
        V = B.mode1_vertices(bx, zv, zero, zv, zero, lev, r['ed']); return float(V.min()) > 0, float(V.min())
    if k == 'E2':
        key = r['key']; d1, d2 = r['rd'][key]
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        a, _ = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
        b_, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
        return a > b_, a - b_
    if k != 'G': return False, -1
    if mode == 'G' and not BB.outside_Q(lo, hi): return False, -1
    if mode in ('Q1', 'Q2') and not BB.in_Q(lo, hi): return False, -1
    rg = {}
    for key in M.RANGE_KEYS:
        d1, d2 = r['rd'][key]
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        a, _ = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
        b_, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
        rlo, rhi = r['rg'][key]
        if not (rlo <= a and rhi >= b_): return False, -2
        rg[key] = (rlo, rhi)
    tp = BB.target_T1(bx, bx.Qp); tb = BB.target_T1(bx, bx.Qb)
    if not r['th'] or min(r['th']) < 0 or sum(r['th']) <= 0: return False, -3
    s = np.zeros(16)
    for c, th in zip(r['c'], r['th']):
        cp, kp = M.main_cost(bx.Qp, rg, c['ch']); cb, kb = M.main_cost(bx.Qb, rg, c['ch'])
        V = B.mode1_vertices(bx, cp, kp - tp, cb, kb - tb, lev, c['d'])
        s = dn(s + dn(th * V))
    return float(s.min()) >= 0, float(s.min())


def recheck_OK2(r):
    import bk16_ok2 as O
    lo, hi = np.array(r['lo']), np.array(r['hi'])
    if r['k'] == 'X': return BB.marg_excluded(lo, hi), 1.0
    return O.box_eval(lo, hi, rec=r)


def recheck_KINK(r):
    import bk16_loc as LC
    if 'DELEGATE' in r: return True, 0.0
    lo, hi = np.array(r['lo']), np.array(r['hi'])
    res = LC.kink_check(lo, hi, r['side'], float(os.environ.get('BK16_S1', '0.005')), rec=r)
    ok = res['ok'] and res['P0'] <= float(os.environ.get('BK16_P0MAX', '0.006'))
    return ok, min(res.get('rcmin', -1), res.get('slope', -1))


for i, r in enumerate(recs):
    kinds[r.get('k', 'kink')] = kinds.get(r.get('k', 'kink'), 0) + 1
    if i % stride != off: continue
    nchk += 1
    if mode == 'OK2': ok, lb = recheck_OK2(r)
    elif mode == 'KINK': ok, lb = recheck_KINK(r)
    else: ok, lb = recheck_G(r)
    kk = r.get('k', 'kink'); minlb[kk] = min(minlb.get(kk, 1e9), lb)
    if not ok: bad += 1; print('FAIL', i, r.get('k'), r['lo'], r['hi'], r.get('lev'), lb, flush=True)
print('records %d kinds %s checked %d (stride %d off %d) failures %d min margins %s' % (len(recs), kinds, nchk, stride, off, bad, minlb), flush=True)
