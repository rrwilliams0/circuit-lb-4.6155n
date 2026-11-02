"""bk16_bnb.py : branch and bound for CKS* outside the local box Q (goal G*):

   for every z = (alpha, lam, d_a, d_b) with (alpha, lam) outside int Q, d in the R7 box, and every u-star m on the slice
   S(z) satisfying the Lemma NS row:   F(m) = 4 mu0 s_u(m) - KILL Delta_u(m; mu) >= g(alpha),

   g(alpha) = -sgn(a) GS max(|a| - A0G, 0),  a = alpha - 1/2  (antisymmetric: g(p) + g(p^sigma) = 0).

Boxes are 4-dimensional boxes times a level [s_lo, s_hi] of the off-face mass.  A box-level is closed when
 'E'  the slice-level set is empty on the whole box (certified Farkas-type Mode 1 bound > 0 of the zero objective, or
      certified range lo > certified range hi), or
 'X'  V(alpha, lam) lies outside [VLO, VHI] on the whole box (not a marginal of N), or
 'G'  the theta-combination of certified Mode 1 bounds (bk16_lb) minus g is >= 0 at all 16 vertices.
Every record keeps the certified ranges (with their duals) and the candidates, so the replay re-certifies everything.
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import sys, json, time, math
import numpy as np
from scipy.optimize import linprog
from ivarith import IA, dn, up
from bk13_lb import T1, VMASK
import bk16_model as M
import bk16_lb as B

QA = (0.47, 0.53); QL = (0.0, 0.005)
GS = 0.045; A0G = 0.015
BLEND = True
LMAX_SPLIT = int(os.environ.get("BK16_LSPLIT", "5"))


MODE = os.environ.get('BK16_MODE', 'G')            # 'G' (outside Q), 'Q1' (Q, d outside D_loc), 'Q2' (Q x D_loc, pm coords)
TAU0 = 1e-4                                         # units 4 mu0 s
WP, WM = 0.01, 0.08


def inside_Dloc(lo, hi):
    """True if for every (p, d) of the box (alpha, lam, d_a, d_b): |d_a + d_b - tau(p)| <= WP and |d_a - d_b| <= WM."""
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); q = 1.0 - la
    S = q * (2.0 - al) + la; m1 = al * q * S.inv()
    tau = 2.5 - m1 * 1.5
    dp = IA(lo[2], hi[2]) + IA(lo[3], hi[3]) - tau
    dm = IA(lo[2], hi[2]) - IA(lo[3], hi[3])
    return float(dp.lo) >= -WP and float(dp.hi) <= WP and float(dm.lo) >= -WM and float(dm.hi) <= WM


def target_T1(bx, Q):
    """g(alpha) as a T1 (affine on boxes that do not straddle 1/2 +- A0G); TAU0 in the modes Q1, Q2."""
    if MODE in ('Q1', 'Q2'): return M.tconst(TAU0)
    al = Q['al']
    lo, hi = bx.lo[0], bx.hi[0]
    if lo >= 0.5 + A0G: return (al - (0.5 + A0G)) * (-GS)
    if hi <= 0.5 - A0G: return (-al + (0.5 - A0G)) * GS
    if lo >= 0.5 - A0G and hi <= 0.5 + A0G: return M.tconst(0.0)
    raise RuntimeError('box straddles a breakpoint of g')


def zero_T1(): return M.tconst(0.0)


def in_Q(lo, hi): return lo[0] >= QA[0] and hi[0] <= QA[1] and lo[1] >= QL[0] and hi[1] <= QL[1]


def outside_Q(lo, hi): return hi[0] <= QA[0] or lo[0] >= QA[1] or lo[1] >= QL[1]


E0MAX = 0.95267118        # e_max = 2 (sqrt(6 c_Theta + 3) - N0)^2 - 1 = 0.952671176..., rounded up (Lemma E0, report 2.2)


def marg_excluded(lo, hi):
    """True if no marginal of the box can be the marginal of a point of N: V outside [VLO, VHI] or
    e_0 = U/sqrt(V) > E0MAX on the whole box (Lemma E0)."""
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); q = 1.0 - la
    S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    V = m1 * 2.0 + m0 + p1 * 0.5
    if float(V.lo) > M.VHI or float(V.hi) < M.VLO: return True
    if float(V.lo) <= 0: return False
    U = m1 * M.SQ2 + m0 + p1 * M.ISQ2
    e0 = U / V.sqrt()
    return float(e0.lo) > E0MAX


def V_range(lo, hi):
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); q = 1.0 - la
    S = q * (2.0 - al) + la; Si = S.inv()
    m1 = al * q * Si; m0 = la * Si; p1 = (1.0 - al) * q * Si * 2.0
    return m1 * 2.0 + m0 + p1 * 0.5


CHOICES = [dict(xs=1.0, x0s=1.0, mK=0.5, m0=0.5), dict(xs=0.0, x0s=0.0, mK=0.5, m0=0.5)]
OFFK = [j for j in range(M.NP) if M.OFFm1[j] > 0]


def sample_points(bx, lev):
    """float data (mu, da, db, feasible?) at the centre, the 16 vertices and 16 half-way points (cached on the box)."""
    if getattr(bx, '_pts', None) is not None: return bx._pts
    qf = M.quantities_pm if MODE == 'Q2' else M.quantities
    pts = []
    for zv in [bx.zc] + [np.where(v == 1, bx.hi, bx.lo) for v in VMASK] + [0.5 * (bx.zc + np.where(v == 1, bx.hi, bx.lo)) for v in VMASK]:
        Qv = qf([T1.var(i, 4, zv[i], zv[i]) for i in range(4)], BC=bx.Qb['BC'])
        muv = [float(bx.fl(x)) for x in Qv['mu']]; dav, dbv = float(bx.fl(Qv['da'])), float(bx.fl(Qv['db']))
        d0, v0 = B.float_lp_at(bx, np.zeros(M.NP), lev, muv, dav, dbv)
        pts.append((muv, dav, dbv, d0 is not None and v0 <= 1e-9))
    bx._pts = pts
    return pts


def ranges(bx, lev, rec=None):
    """certified ranges; with rec (a record) the recorded duals are re-used (replay).  Candidate duals: the centre's
    optimal dual (affine in z) and, when that bound is loose (edge boxes), optimal duals at feasible sample points."""
    zero = zero_T1(); rg = {}; duals = {}
    for key in M.RANGE_KEYS:
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        if rec:
            d1, d2 = rec['rd'][key]
            lo_, _ = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
            hi_, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
            rg[key] = (lo_, hi_); duals[key] = (d1, d2); continue
        res = []
        for mx in (False, True):
            best, bd = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, slopes=True)
            if best is None: return None, None
            cf = bx.fl(vp)
            dc, fv = B.float_lp(bx, -cf if mx else cf, lev)
            fv = (-fv if mx else fv) if dc is not None else None
            loose = fv is None or abs(best - fv) > 0.05 * abs(fv) + 1e-3
            if loose:
                for (muv, dav, dbv, ok) in sample_points(bx, lev):
                    if not ok: continue
                    dd, vv = B.float_lp_at(bx, -cf if mx else cf, lev, muv, dav, dbv)
                    if dd is None: continue
                    dd = [float(x) for x in dd]
                    val, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, dual=dd)
                    if (mx and val < best) or ((not mx) and val > best): best, bd = val, dd
            res.append((best, bd))
        rg[key] = (res[0][0], res[1][0]); duals[key] = (res[0][1], res[1][1])
    return rg, duals


def box_eval(lo, hi, lev):
    bx = B.Box(lo, hi, pm=(MODE == 'Q2')); bx._pts = None
    bx.Qp['al'] = T1.var(0, 4, bx.zc[0], bx.zc[0]); bx.Qb['al'] = T1.var(0, 4, bx.lo[0], bx.hi[0])
    # 1. emptiness of the slice-level set
    zero = zero_T1()
    dual, val = B.float_lp(bx, np.zeros(M.NP), lev)
    if dual is not None and val > 1e-7:
        V = B.mode1_vertices(bx, M.tvecconst(np.zeros(M.NP)), zero, M.tvecconst(np.zeros(M.NP)), zero, lev, dual)
        if float(V.min()) > 0:
            return dict(k='E', ed=list(map(float, dual)), lb=float(V.min())), bx
    # 2. ranges
    rg, rd = ranges(bx, lev)
    if rg is None: return dict(k='F', lb=-1.0), bx
    for key, (a, b) in rg.items():
        if a > b:                                        # empty: certified lower bound above certified upper bound
            return dict(k='E2', key=key, rd={key: rd[key]}, lb=a - b), bx
    # 3. main candidates
    tp = target_T1(bx, bx.Qp); tb = target_T1(bx, bx.Qb)
    cands = []; vals = []
    for ch0 in CHOICES:
        Rset = [[1.0] + [0.0] * 8]
        cp, kp = M.main_cost(bx.Qp, rg, dict(ch0, R=Rset[0]))
        fv, x = B.float_primal(bx, bx.fl(cp), lev)
        if x is not None:
            mm = x[OFFK]; M1 = mm.sum()
            if M1 > 0.3 * rg['M1'][1] and M1 > 0:
                wv = [0.0] + list(np.round(mm / M1, 6)); s = sum(wv)
                if s > 1: wv = [w / s * (1 - 1e-12) for w in wv]
                Rset.append([float(w) for w in wv])
        for Rw in Rset:
            ch = dict(ch0, R=Rw)
            cp, kp = M.main_cost(bx.Qp, rg, ch); cb, kb = M.main_cost(bx.Qb, rg, ch)
            kp2 = kp - tp; kb2 = kb - tb
            dual, _ = B.float_lp(bx, bx.fl(cp), lev)
            if dual is None: continue
            dual = B.affine_dual(bx, dual, B.dual_slopes(bx, cp, dual, lev))
            V = B.mode1_vertices(bx, cp, kp2, cb, kb2, lev, dual)
            fv_, _x = B.float_primal(bx, bx.fl(cp), lev)
            if fv_ is not None: bx.fcent = max(getattr(bx, 'fcent', -1e9), fv_ + float(bx.fl(kp2)))
            vals.append(V); cands.append(dict(ch=ch, d=dual))
    if not vals: return dict(k='F', lb=-1.0), bx
    Vm = np.array(vals); Vm = np.where(np.isfinite(Vm), Vm, -1e9)
    th, lb = theta_combine(Vm)
    if lb < 0 and BLEND and np.all((bx.hi - bx.lo) <= np.array([0.02, 0.02, 0.03, 0.03])):
        # edge boxes: blends of a dual of a FEASIBLE point of the box with Farkas-type duals of infeasible points
        ch = cands[int(np.argmax(Vm.min(axis=1)))]['ch']
        cp, kp = M.main_cost(bx.Qp, rg, ch); cb, kb = M.main_cost(bx.Qb, rg, ch); kp2 = kp - tp; kb2 = kb - tb
        cf = bx.fl(cp)
        feas, infe = [], []
        for (muv, dav, dbv, okp) in sample_points(bx, lev):
            dd, vv = B.float_lp_at(bx, cf, lev, muv, dav, dbv)
            if dd is None: continue
            (feas if okp else infe).append(dd)
            if len(feas) >= 3 and len(infe) >= 3: break
        for df in feas[:3]:
            dfl = [float(x) for x in df]
            V = B.mode1_vertices(bx, cp, kp2, cb, kb2, lev, dfl); vals.append(V); cands.append(dict(ch=ch, d=dfl))
            for di in infe[:3]:
                for sfrac in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
                    dd = [float(dfl[i] + sfrac * (di[i] - dfl[i])) for i in range(5)]
                    dd = [max(x, 0.0) if i >= 2 else x for i, x in enumerate(dd)]
                    V = B.mode1_vertices(bx, cp, kp2, cb, kb2, lev, dd)
                    vals.append(V); cands.append(dict(ch=ch, d=dd))
        Vm = np.array(vals); Vm = np.where(np.isfinite(Vm), Vm, -1e9)
    th, lb = theta_combine(Vm)
    keep = [i for i in range(len(th)) if th[i] > 0]
    bx.vbest = Vm[int(np.argmax(Vm.min(axis=1)))]
    return dict(k='G', rd=rd, rg={k_: list(v) for k_, v in rg.items()}, c=[cands[i] for i in keep], th=[float(th[i]) for i in keep], lb=lb), bx


def theta_combine(V):
    nc = V.shape[0]
    bi = int(np.argmax(V.min(axis=1)))
    if V[bi].min() >= 0:
        th = np.zeros(nc); th[bi] = 1.0; return th, float(V[bi].min())
    cobj = np.zeros(nc + 1); cobj[-1] = -1.0
    Aub = np.hstack([-V.T, np.ones((16, 1))]); Aeq = np.zeros((1, nc + 1)); Aeq[0, :nc] = 1.0
    r = linprog(cobj, A_ub=Aub, b_ub=np.zeros(16), A_eq=Aeq, b_eq=[1.0], bounds=[(0, None)] * nc + [(None, None)], method='highs')
    if r.status != 0:
        th = np.zeros(nc); th[bi] = 1.0; return th, float(V[bi].min())
    th = np.maximum(r.x[:nc], 0.0)
    s = np.zeros(16)
    for c in range(nc):
        if th[c] > 0: s = dn(s + dn(th[c] * V[c]))
    return th, float(s.min())


def level_split(lev):
    slo, shi = lev
    a = max(slo, 0.002); b = min(shi, 0.16)
    if b <= 2.0 * max(slo, 0.004): return None
    s = math.sqrt(a * b)
    if s <= slo or s >= shi: return None
    return s


def split_dim(lo, hi, bx):
    vb = getattr(bx, 'vbest', None)
    w = hi - lo
    if vb is not None and np.all(np.isfinite(vb)):
        infl = np.array([np.mean(np.abs(vb[VMASK[:, i] == 1] - vb[VMASK[:, i] == 0])) for i in range(4)])
        if infl.max() > 0: return int(np.argmax(infl + 1e-12 * w))
    return int(np.argmax(w / np.array([1.0, 1.0, 0.6, 0.6])))


def run(tag, roots, maxboxes, maxdepth=48, t_budget=1e9):
    qf = tag + '_queue.json'; cf = tag + '_closed.jsonl'
    if os.path.exists(qf):
        st = json.load(open(qf)); queue = [(np.array(a), np.array(b), tuple(l), d, ls) for a, b, l, d, ls in st['queue']]; stats = st['stats']
        print('resuming', tag, len(queue), stats, flush=True)
    else:
        LV = [float(x) for x in os.environ.get('BK16_LEVELS', '0,1').split(',')]
        queue = [(np.array(a, float), np.array(b, float), (LV[i], LV[i + 1]), 0, 0) for a, b in roots for i in range(len(LV) - 1)]
        stats = dict(processed=0, G=0, E=0, X=0, Q=0, open=0, minlb=1.0, maxdepth=0, levsplit=0)
    out = open(cf, 'a'); openf = open(tag + '_open.jsonl', 'a')
    t0 = time.time(); last = t0
    while queue and stats['processed'] < maxboxes and time.time() - t0 < t_budget and not os.path.exists(tag + '.stop'):
        lo, hi, lev, dep, ls = queue.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        base = dict(lo=lo.tolist(), hi=hi.tolist(), lev=list(lev))
        if in_Q(lo, hi):
            stats['Q'] += 1; out.write(json.dumps(dict(base, k='Q')) + '\n'); continue
        if not outside_Q(lo, hi): raise RuntimeError('straddles Q')
        if marg_excluded(lo, hi):
            stats['X'] += 1; out.write(json.dumps(dict(base, k='X')) + '\n'); continue
        try:
            rec, bx = box_eval(lo, hi, lev)
        except (ZeroDivisionError, ArithmeticError):
            rec, bx = dict(k='F', lb=-1.0), None
        if rec['k'] in ('E', 'E2', 'G') and rec['lb'] >= 0:
            k = 'E' if rec['k'] == 'E2' else rec['k']
            stats[k] += 1
            if rec['k'] == 'G': stats['minlb'] = min(stats['minlb'], rec['lb'])
            out.write(json.dumps(dict(base, **rec)) + '\n'); continue
        if dep >= maxdepth:
            stats['open'] += 1; openf.write(json.dumps(dict(base, lb=rec['lb'])) + '\n'); continue
        fc = getattr(bx, 'fcent', None) if bx is not None else None
        need_level = (fc is not None and fc < 0.25 * abs(float(rec['lb'])) + 1e-6)
        s = level_split(lev) if (ls < LMAX_SPLIT and need_level) else None
        if s is not None:
            stats['levsplit'] += 1
            queue.append((lo, hi, (s, lev[1]), dep + 1, ls + 1)); queue.append((lo, hi, (lev[0], s), dep + 1, ls + 1))
        else:
            wm = (hi - lo)[:2] / np.array([0.02, 0.02])
            if rec['k'] in ('F',) or wm.max() > 1.0 or bx is None: i = int(np.argmax(wm)) if wm.max() > 1.0 else int(np.argmax((hi - lo) / np.array([1.0, 1.0, 0.6, 0.6])))
            elif dep % 3 != 2: i = split_dim(lo, hi, bx)
            else: i = int(np.argmax((hi - lo) / np.array([1.0, 1.0, 0.6, 0.6])))
            mid = 0.5 * (lo[i] + hi[i])
            if i == 0 and lo[0] < 0.5 - A0G < hi[0]: mid = 0.5 - A0G
            if i == 0 and lo[0] < 0.5 + A0G < hi[0]: mid = 0.5 + A0G
            h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
            queue.append((l2, hi, lev, dep + 1, ls)); queue.append((lo, h1, lev, dep + 1, ls))
        if time.time() - last > 60:
            last = time.time(); out.flush(); openf.flush()
            json.dump(dict(queue=[(a.tolist(), b.tolist(), list(l), d, s_) for a, b, l, d, s_ in queue], stats=stats), open(qf, 'w'))
            print('%7.0fs %s queue %d' % (time.time() - t0, stats, len(queue)), flush=True)
    out.close(); openf.close()
    json.dump(dict(queue=[(a.tolist(), b.tolist(), list(l), d, s_) for a, b, l, d, s_ in queue], stats=stats), open(qf, 'w'))
    print('DONE' if not queue else 'STOPPED', tag, stats, 'queue', len(queue), flush=True)


def roots_grid(abp, lbp, nd):
    dbp = [M.DLO + (M.DHI - M.DLO) * k / nd for k in range(nd + 1)]; dbp[-1] = M.DHI
    out = []
    for i in range(len(abp) - 1):
        for j in range(len(lbp) - 1):
            for p in range(nd):
                for q in range(nd):
                    out.append(([abp[i], lbp[j], dbp[p], dbp[q]], [abp[i + 1], lbp[j + 1], dbp[p + 1], dbp[q + 1]]))
    return out


ABP = [0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.47, 0.485, 0.5, 0.515, 0.53, 0.54, 0.56, 0.58, 0.6, 0.65, 0.7]   # window W of bk16_margx
LBP = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]

def roots_Q1(nd=8):
    out = []
    for r in roots_grid([0.47, 0.485, 0.5, 0.515, 0.53], [0.0, 0.0025, 0.005], nd): out.append(r)
    return out


def roots_Q2(n=(4, 2, 2, 4)):
    import itertools
    rng_ = [(0.47, 0.53), (0.0, 0.005), (-WP, WP), (-WM, WM)]
    grids = [np.linspace(r[0], r[1], k + 1) for r, k in zip(rng_, n)]
    for g, r in zip(grids, rng_): g[0], g[-1] = r[0], r[1]
    return [([grids[i][idx[i]] for i in range(4)], [grids[i][idx[i] + 1] for i in range(4)]) for idx in itertools.product(*[range(k) for k in n])]


if __name__ == '__main__' and MODE in ('Q1', 'Q2'):
    tag = sys.argv[1]; tb = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    roots = roots_Q1() if MODE == 'Q1' else roots_Q2()
    rf = os.environ.get('BK16_ROOTS_FILE')
    if rf:
        roots = [(json.loads(l)['lo'], json.loads(l)['hi']) for l in open(rf) if 'DELEGATE' in json.loads(l)]
    print('mode', MODE, 'roots', len(roots), flush=True)
    run(tag, roots, 10 ** 9, t_budget=tb)
elif __name__ == '__main__':
    tag = sys.argv[1]; arange = (float(sys.argv[2]), float(sys.argv[3]))
    maxb = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    tb = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
    lmax = float(os.environ.get('BK16_LMAX', '1.0'))
    roots = [r for r in roots_grid(ABP, LBP, 2) if r[0][0] >= arange[0] and r[1][0] <= arange[1] and r[1][1] <= lmax]
    print('roots', len(roots), flush=True)
    run(tag, roots, maxb, t_budget=tb)
