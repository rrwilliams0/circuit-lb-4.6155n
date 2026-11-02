"""bk16_ok2.py : certificate of O-the-2 in the form

    K(m) <= C_K (Lbar(m) - mu0) + c0      for every u-star m with ratios in the R7 box satisfying the Lemma NS row,

K = sum_{U_-1} m |a(pi)| (the first-order kill mass), Lbar = (l_a + l_b + l_ab)/3 (Lemma L, linear on the slice).
Since Lbar <= kbar (Lemma L), this gives K <= C_K E + c0 with E = kbar - mu0.  Branch and bound over (alpha, lam, d_a, d_b)
on [0,1]^2 x R7 box; each box closed by a certified Mode 1 bound of  (C_K/3) (nsc + nsg.m) - C_K mu0 - K(m) >= -c0, or by
emptiness of the slice set (Farkas-type certified bound of the zero objective > 0), or 'X' (V outside [VLO, VHI])."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import sys, json, time, math
import numpy as np
from ivarith import IA, dn
import bk16_model as M, bk16_lb as B
from bk16_bnb import marg_excluded, roots_grid, split_dim, theta_combine
CK = float(os.environ.get('BK16_CK', '60')); C0 = float(os.environ.get('BK16_C0', '1e-4'))
LEV = (0.0, 1.0)


def obj(Q):
    cost = Q['nsg'] * (CK / 3.0) - Q['kv']
    const = Q['nsc'] * (CK / 3.0) - M.MU0 * CK + C0
    return cost, const


def box_eval(lo, hi, rec=None):
    bx = B.Box(lo, hi); zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if rec is None:
        dual, val = B.float_lp(bx, np.zeros(M.NP), LEV)
        if dual is not None and val > 1e-7:
            V = B.mode1_vertices(bx, zv, zero, zv, zero, LEV, dual)
            if float(V.min()) > 0: return dict(k='E', d=list(map(float, dual)), lb=float(V.min())), bx
        cp, kp = obj(bx.Qp); cb, kb = obj(bx.Qb)
        dual, _ = B.float_lp(bx, bx.fl(cp), LEV)
        if dual is None: return dict(k='F', lb=-1.0), bx
        dual = B.affine_dual(bx, dual, B.dual_slopes(bx, cp, dual, LEV))
        V = B.mode1_vertices(bx, cp, kp, cb, kb, LEV, dual)
        if float(V.min()) < 0 and np.all((bx.hi - bx.lo) <= np.array([0.02, 0.02, 0.03, 0.03])):
            # edge boxes: optimal duals at feasible sample points (any dual is valid)
            import bk16_bnb as BB
            bx._pts = None; cf = bx.fl(cp)
            for (muv, dav, dbv, okp) in BB.sample_points(bx, LEV):
                if not okp: continue
                dd, vv = B.float_lp_at(bx, cf, LEV, muv, dav, dbv)
                if dd is None: continue
                dd = [float(x) for x in dd]
                V2 = B.mode1_vertices(bx, cp, kp, cb, kb, LEV, dd)
                if float(V2.min()) > float(V.min()): V, dual = V2, dd
                if float(V.min()) >= 0: break
        bx.vbest = V
        return dict(k='K', d=dual, lb=float(V.min())), bx
    if rec['k'] == 'E':
        V = B.mode1_vertices(bx, zv, zero, zv, zero, LEV, rec['d']); return float(V.min()) > 0, float(V.min())
    cp, kp = obj(bx.Qp); cb, kb = obj(bx.Qb)
    V = B.mode1_vertices(bx, cp, kp, cb, kb, LEV, rec['d']); return float(V.min()) >= 0, float(V.min())


def run(tag, roots, t_budget=1e9, maxdepth=50):
    qf = tag + '_queue.json'
    if os.path.exists(qf):
        queue = [(np.array(a), np.array(b), d) for a, b, d in json.load(open(qf))]
        out = open(tag + '_closed.jsonl', 'a'); openf = open(tag + '_open.jsonl', 'a'); print('resuming', len(queue), flush=True)
    else:
        out = open(tag + '_closed.jsonl', 'w'); openf = open(tag + '_open.jsonl', 'w')
        queue = [(np.array(a, float), np.array(b, float), 0) for a, b in roots]
    st = dict(processed=0, K=0, E=0, X=0, open=0, minlb=1.0); t0 = time.time(); last = t0
    while queue and time.time() - t0 < t_budget and not os.path.exists(tag + '.stop'):
        lo, hi, dep = queue.pop(); st['processed'] += 1
        base = dict(lo=lo.tolist(), hi=hi.tolist())
        if marg_excluded(lo, hi):
            st['X'] += 1; out.write(json.dumps(dict(base, k='X')) + '\n'); continue
        try:
            rec, bx = box_eval(lo, hi)
        except (ZeroDivisionError, ArithmeticError):
            rec, bx = dict(k='F', lb=-1.0), None
        if rec['k'] in ('E', 'K') and rec['lb'] >= 0:
            st[rec['k']] += 1
            if rec['k'] == 'K': st['minlb'] = min(st['minlb'], rec['lb'])
            out.write(json.dumps(dict(base, **rec)) + '\n'); continue
        if dep >= maxdepth:
            st['open'] += 1; openf.write(json.dumps(dict(base, lb=rec['lb'])) + '\n'); continue
        i = split_dim(lo, hi, bx) if (bx is not None and dep % 3 != 2) else int(np.argmax((hi - lo) / np.array([1.0, 1.0, 0.6, 0.6])))
        mid = 0.5 * (lo[i] + hi[i]); h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        queue.append((l2, hi, dep + 1)); queue.append((lo, h1, dep + 1))
        if time.time() - last > 60:
            last = time.time(); out.flush(); print('%6.0fs %s queue %d' % (time.time() - t0, st, len(queue)), flush=True)
    out.close(); openf.close()
    json.dump([(a.tolist(), b.tolist(), d) for a, b, d in queue], open(qf, 'w'))
    print('DONE' if not queue else 'STOPPED', tag, st, 'queue', len(queue), 'CK', CK, 'c0', C0, flush=True)


if __name__ == '__main__':
    tag = sys.argv[1]
    ABP = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]; LBP = [0.0, 0.02, 0.05, 0.1, 0.25]
    run(tag, roots_grid(ABP, LBP, 2), t_budget=float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
