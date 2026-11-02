"""bk13_bnb.py : branch and bound over 4-dimensional boxes (alpha, lam, d_a, d_b) for the u-star slice LP.
Goal G (p outside Q):  LP(p, d) >= TG for all d in [1/2, 2]^2      (single star, strict).
Goal L1 (p in Q):      LP(p, d) >= TAU0 for all d outside D_loc(p) (boxes inside D_loc(p) for every p are skipped; L2/L3 cover them).
Per box: candidates = Mode 1 (constant dual y) and Mode 2 (basis, phi-form) from dual LPs at the centre (and, on failure, at
corners); the best convex combination theta is found by a small LP and re-verified in outward rounding:
    sum_c theta_c (V_c(v) - target) >= 0 at all 16 vertices v  ==>  max_c (bound_c - target) >= 0 on the box  ==>  LP >= target.
Records: every closed box (with its candidates and theta) in <tag>_closed.jsonl; the open queue in <tag>_queue.json."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import sys, json, time, math, itertools, argparse
import numpy as np
from scipy.optimize import linprog
from ivarith import IA, dn, up
import bk13_lb as L
from bk13_lb import Shared, mode1, mode2b, basis_data, NP, PU, AFR, AA, AB, KV, vertices

QA = (0.47, 0.53); QL = (0.0, 0.005)
WP, WM = 0.01, 0.1
TAU0 = 1e-4
TG = 1e-6
YB = float(os.environ.get("BK13_YB", "1000"))

# float data for the dual LPs (choice of candidates only; not part of the proof)
import bk13_core as C


def float_c(p, d):
    mu = C.mu_of(p[0], p[1]); c, const = C.lp_data(mu, d[0] ** -0.5, d[1] ** -0.5); return mu, c, const


A_UB = np.zeros((NP, 5))
for j in range(NP):
    A_UB[j, int(KV[j]) + 1] = 1.0; A_UB[j, 3] = AA[j]; A_UB[j, 4] = AB[j]


def dual_lp(p, d):
    mu, c, const = float_c(p, d)
    b = np.array([mu[0], mu[1], mu[2], d[0], d[1]])
    r = linprog(-b, A_ub=A_UB, b_ub=c, bounds=[(None, None)] * 3 + [(-YB, YB)] * 2, method='highs')
    if r.status != 0: return None
    y = r.x; slack = c - A_UB @ y
    basis = None
    if abs(y[3]) < YB * 0.999 and abs(y[4]) < YB * 0.999:
        order = np.argsort(slack); B = []
        for j in order:
            if slack[j] > 1e-7: break
            B.append(int(j))
            M = A_UB[B].T
            if np.linalg.matrix_rank(M) < len(B): B.pop()
            if len(B) == 5: break
        if len(B) == 5: basis = tuple(sorted(B))
    return dict(y=y, basis=basis, val=-r.fun + const)


def theta_combine(V):
    """V: (ncand, 16) rigorous lower values minus target. Returns (theta, verified lower bound of the combination)."""
    nc = V.shape[0]
    best_single = int(np.argmax(V.min(axis=1)))
    if V[best_single].min() >= 0:
        th = np.zeros(nc); th[best_single] = 1.0
        return th, float(V[best_single].min())
    # maximise t s.t. sum_c th_c V[c, v] >= t, sum th = 1, th >= 0
    cobj = np.zeros(nc + 1); cobj[-1] = -1.0
    Aub = np.hstack([-V.T, np.ones((16, 1))]); bub = np.zeros(16)
    Aeq = np.zeros((1, nc + 1)); Aeq[0, :nc] = 1.0
    r = linprog(cobj, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * nc + [(None, None)], method='highs')
    if r.status != 0:
        th = np.zeros(nc); th[best_single] = 1.0; return th, float(V[best_single].min())
    th = np.maximum(r.x[:nc], 0.0)
    # verified: s_v = sum_c th_c V[c, v] with downward rounding (th >= 0 exact doubles)
    s = np.zeros(16)
    for c in range(nc):
        if th[c] > 0: s = dn(s + dn(th[c] * V[c]))
    return th, float(s.min())


def in_Q(lo, hi):
    return lo[0] >= QA[0] and hi[0] <= QA[1] and lo[1] >= QL[0] and hi[1] <= QL[1]


def outside_Q(lo, hi):
    return hi[0] <= QA[0] or lo[0] >= QA[1] or lo[1] >= QL[1]


def inside_Dloc(lo, hi):
    """True if for every (p, d) in the box, |d_a + d_b - tau(p)| <= WP and |d_a - d_b| <= WM, tau = 5/2 - (3/2) mu_-1(p)."""
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); q = 1.0 - la
    S = q * (2.0 - al) + la; m1 = al * q * S.inv()
    tau = 2.5 - m1 * 1.5
    dp = IA(lo[2], hi[2]) + IA(lo[3], hi[3]) - tau
    dm = IA(lo[2], hi[2]) - IA(lo[3], hi[3])
    return float(dp.lo) >= -WP and float(dp.hi) <= WP and float(dm.lo) >= -WM and float(dm.hi) <= WM


def box_bound(lo, hi, target, extra=False):
    sh = Shared(lo, hi)
    zc = sh.zc
    cands = []; vals = []
    pts = [zc]
    if extra:
        for v in vertices(lo, hi)[::3]: pts.append(0.5 * (v + zc))
        for v in vertices(lo, hi)[::5]: pts.append(v)
    seenB = set(); seenY = []
    for z in pts:
        dl = dual_lp(z[:2], z[2:])
        if dl is None: continue
        ya, yb = float(dl['y'][3]), float(dl['y'][4])
        if not any(abs(ya - a) < 1e-9 and abs(yb - b) < 1e-9 for a, b in seenY):
            seenY.append((ya, yb))
            vals.append(mode1(sh, ya, yb)); cands.append(('m1', ya, yb))
        if dl['basis'] is not None and dl['basis'] not in seenB:
            seenB.add(dl['basis'])
            v2 = mode2b(sh, dl['basis'])
            if v2 is not None: vals.append(v2); cands.append(('m2', list(dl['basis'])))
    # intermediate duals between a feasible-slice dual and an infeasibility (bounded-ray) dual: useful on boxes that
    # straddle the boundary of the feasible d-region
    yin = [y for y in seenY if max(abs(y[0]), abs(y[1])) < YB * 0.999]
    yout = [y for y in seenY if max(abs(y[0]), abs(y[1])) >= YB * 0.999]
    for yf in yin[:2]:
        for yi in yout[:2]:
            for sfrac in (0.002, 0.006, 0.02, 0.06, 0.2):
                ya, yb = yf[0] + sfrac * (yi[0] - yf[0]), yf[1] + sfrac * (yi[1] - yf[1])
                vals.append(mode1(sh, ya, yb)); cands.append(('m1', ya, yb))
    if not vals: return None, None, -1.0, sh
    V = dn(np.array(vals) - target)
    V = np.where(np.isfinite(V), V, -1e9)
    th, lb = theta_combine(V)
    sh.vbest = V[int(np.argmax(V.min(axis=1)))]
    return cands, th, lb, sh


def split_dim(lo, hi, sh):
    vb = getattr(sh, 'vbest', None)
    if vb is not None and np.all(np.isfinite(vb)):
        from bk13_lb import VMASK
        infl = np.array([np.mean(np.abs(vb[VMASK[:, i] == 1] - vb[VMASK[:, i] == 0])) for i in range(4)])
        if infl.max() > 0: return int(np.argmax(infl + 1e-12 * (hi - lo)))
    return split_dim_old(lo, hi, sh)


def split_dim_old(lo, hi, sh):
    E = sh.box
    # sensitivity of the objective: use |grad const| + |grad c| . mu + |y| . |grad d| as a crude scale
    w = hi - lo
    gc = np.abs(np.asarray(E['c'].g.hi)).sum(axis=1) * 0.5 + np.abs(np.asarray(E['const'].g.hi)) + np.array([0, 0, 0.3, 0.3])
    sc = gc * w
    return int(np.argmax(sc))


def run(tag, roots, maxboxes, maxdepth=60, t_budget=1e9):
    qf = tag + '_queue.json'; cf = tag + '_closed.jsonl'
    if os.path.exists(qf):
        st = json.load(open(qf)); queue = [(np.array(a), np.array(b), d) for a, b, d in st['queue']]; stats = st['stats']
        print('resuming', tag, 'queue', len(queue), stats, flush=True)
    else:
        queue = [(np.array(a, float), np.array(b, float), 0) for a, b in roots]
        stats = dict(closed=0, skipped_local=0, processed=0, open_final=0, minlb=1.0, minlb_G=1.0, minlb_L1=1.0, maxdepth=0)
    out = open(cf, 'a'); openf = open(tag + '_open.jsonl', 'a')
    t0 = time.time(); last = t0
    while queue and stats['processed'] < maxboxes and time.time() - t0 < t_budget:
        lo, hi, dep = queue.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        if in_Q(lo, hi):
            if inside_Dloc(lo, hi): stats['skipped_local'] += 1; out.write(json.dumps(dict(k='loc', lo=lo.tolist(), hi=hi.tolist())) + '\n'); continue
            target = TAU0; goal = 'L1'
        elif outside_Q(lo, hi):
            target = TG; goal = 'G'
        else:
            raise RuntimeError('box straddles Q: %s %s' % (lo, hi))
        cands, th, lb, sh = box_bound(lo, hi, target)
        if lb < 0 and np.max((hi - lo) / np.array([1, 1, 1.5, 1.5])) < 0.12:
            cands, th, lb, sh = box_bound(lo, hi, target, extra=True)
        if lb >= 0:
            stats['closed'] += 1
            stats['minlb'] = min(stats['minlb'], lb)
            stats['minlb_' + goal] = min(stats['minlb_' + goal], lb)
            keep = [i for i in range(len(th)) if th[i] > 0]
            out.write(json.dumps(dict(k=goal, lo=lo.tolist(), hi=hi.tolist(), t=target, c=[cands[i] for i in keep],
                                      th=[float(th[i]) for i in keep], lb=lb)) + '\n')
            continue
        if dep >= maxdepth:
            stats['open_final'] += 1; openf.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), lb=lb, goal=goal)) + '\n'); continue
        i = split_dim(lo, hi, sh) if dep % 3 != 2 else int(np.argmax((hi - lo) / np.array([1.0, 1.0, 1.5, 1.5])))
        mid = 0.5 * (lo[i] + hi[i])
        if goal == 'G' or goal == 'L1':
            pass
        h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        queue.append((l2, hi, dep + 1)); queue.append((lo, h1, dep + 1))
        if time.time() - last > 60:
            last = time.time(); out.flush(); openf.flush()
            json.dump(dict(queue=[(a.tolist(), b.tolist(), d) for a, b, d in queue], stats=stats), open(qf, 'w'))
            print('%7.0fs processed %d closed %d local %d queue %d open %d minlb G %.2e L1 %.2e depth %d' % (
                time.time() - t0, stats['processed'], stats['closed'], stats['skipped_local'], len(queue), stats['open_final'],
                stats['minlb_G'], stats['minlb_L1'], stats['maxdepth']), flush=True)
    out.close(); openf.close()
    json.dump(dict(queue=[(a.tolist(), b.tolist(), d) for a, b, d in queue], stats=stats), open(qf, 'w'))
    print('DONE' if not queue else 'STOPPED', tag, stats, 'queue', len(queue), flush=True)


def roots_grid(alpha_bp, lam_bp, nd):
    dg = np.linspace(0.5, 2.0, nd + 1)
    roots = []
    for i in range(len(alpha_bp) - 1):
        for j in range(len(lam_bp) - 1):
            for k in range(nd):
                for l in range(nd):
                    roots.append(([alpha_bp[i], lam_bp[j], dg[k], dg[l]], [alpha_bp[i + 1], lam_bp[j + 1], dg[k + 1], dg[l + 1]]))
    return roots


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('--alpha', default=None); ap.add_argument('--lam', default=None)
    ap.add_argument('--nd', type=int, default=6); ap.add_argument('--max', type=int, default=10 ** 9); ap.add_argument('--time', type=float, default=1e9)
    ap.add_argument('--fromopen', default=None)
    a = ap.parse_args()
    if a.fromopen:
        rs = [json.loads(l) for l in open(a.fromopen)]
        run(a.tag, [(r['lo'], r['hi']) for r in rs], a.max, t_budget=a.time); sys.exit(0)
    abp = [float(x) for x in a.alpha.split(',')] if a.alpha else sorted(set(list(np.round(np.linspace(0, 1, 21), 6)) + [0.47, 0.53]))
    lbp = [float(x) for x in a.lam.split(',')] if a.lam else [0.0, 0.005, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    abp = [x for x in abp if not (QA[0] < x < QA[1])]
    run(a.tag, roots_grid(abp, lbp, a.nd), a.max, t_budget=a.time)
