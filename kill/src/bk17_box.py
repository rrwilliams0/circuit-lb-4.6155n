"""bk17_box.py : box evaluation for the CKS* certificate (goals G*, Q1, Q2 and the direct closure 'D' of KINK-K).

Same mathematics as the bk16_bnb.box_eval (Lemma REL relaxation, Lemma MODE1-A bounds, theta-combination), with:
  * coordinates (alpha, lam, d_a, d_b) or pm coordinates (alpha, lam, dp, dm) (pm=True; Lemma KINK);
  * the target as a parameter: tau0 (domain Q) or the antisymmetric g(alpha) (domain G);
  * RANGE INHERITANCE (speed-up, sound): ranges certified on a box P at a level L are valid on every sub-box at a
    sub-level, because the slice-level set of a sub-box is a subset of P's.  A record closed with inherited ranges keeps
    P ('rbox') and its level ('rlev'), and the replay re-certifies the ranges on P from the recorded duals;
  * cheaper range certification (at most NSAMP feasible sample-point duals when the centre dual is loose).
Every bound is a certified Mode 1 bound (weak duality for every point of the box, outward rounding); duals are only
candidates, so heuristics affect speed, never validity.
"""
import os
import numpy as np
from ivarith import IA, dn, up
from bk13_lb import T1, VMASK
import bk16_model as M
import bk16_lb as B
import bk16_bnb as BB

NSAMP = int(os.environ.get('BK17_NSAMP', '0'))
GS, A0G = BB.GS, BB.A0G


def target_G(bx, Q, lo, hi):
    al = Q['al']
    if lo[0] >= 0.5 + A0G: return (al - (0.5 + A0G)) * (-GS)
    if hi[0] <= 0.5 - A0G: return (-al + (0.5 - A0G)) * GS
    if lo[0] >= 0.5 - A0G and hi[0] <= 0.5 + A0G: return M.tconst(0.0)
    raise RuntimeError('box straddles a breakpoint of g')


def target_value(dom, bx, Q, lo, hi, tau0):
    if dom == 'Q': return M.tconst(tau0)
    return target_G(bx, Q, lo, hi)


def make_box(lo, hi, pm):
    bx = B.Box(lo, hi, pm=pm); bx._pts = None
    bx.Qp['al'] = T1.var(0, 4, bx.zc[0], bx.zc[0]); bx.Qb['al'] = T1.var(0, 4, bx.lo[0], bx.hi[0])
    return bx


def sample_points(bx, lev, pm):
    if bx._pts is not None: return bx._pts
    qf = M.quantities_pm if pm else M.quantities
    pts = []
    for zv in [bx.zc] + [np.where(v == 1, bx.hi, bx.lo) for v in VMASK] + [0.5 * (bx.zc + np.where(v == 1, bx.hi, bx.lo)) for v in VMASK]:
        Qv = qf([T1.var(i, 4, zv[i], zv[i]) for i in range(4)], BC=bx.Qb['BC'])
        muv = [float(bx.fl(x)) for x in Qv['mu']]; dav, dbv = float(bx.fl(Qv['da'])), float(bx.fl(Qv['db']))
        d0, v0 = B.float_lp_at(bx, np.zeros(M.NP), lev, muv, dav, dbv)
        pts.append((muv, dav, dbv, d0 is not None and v0 <= 1e-9))
    bx._pts = pts
    return pts


def ranges(bx, lev, pm, rec=None):
    zero = M.tconst(0.0); rg = {}; duals = {}
    for key in M.RANGE_KEYS:
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        if rec is not None:
            d1, d2 = rec[key]
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
                for aff in (True, False):
                    sp = robust_dual(bx, -vp if mx else vp, zero, lev, affine=aff)
                    if sp is None: continue
                    val, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, dual=sp)
                    if (mx and val < best) or ((not mx) and val > best): best, bd = val, sp
                loose = abs(best - fv) > 0.05 * abs(fv) + 1e-3 if fv is not None else True
            if loose:
                ntry = 0
                for (muv, dav, dbv, ok) in sample_points(bx, lev, pm):
                    if not ok: continue
                    if ntry >= NSAMP: break
                    ntry += 1
                    dd, vv = B.float_lp_at(bx, -cf if mx else cf, lev, muv, dav, dbv)
                    if dd is None: continue
                    dd = [float(x) for x in dd]
                    val, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, dual=dd)
                    if (mx and val < best) or ((not mx) and val > best): best, bd = val, dd
            res.append((best, bd))
        rg[key] = (res[0][0], res[1][0]); duals[key] = (res[0][1], res[1][1])
    return rg, duals


def emptiness(bx, lev):
    zero = M.tconst(0.0)
    dual, val = B.float_lp(bx, np.zeros(M.NP), lev)
    if dual is not None and val > 1e-7:
        zv = M.tvecconst(np.zeros(M.NP))
        V = B.mode1_vertices(bx, zv, zero, zv, zero, lev, dual)
        if float(V.min()) > 0:
            return dict(k='E', ed=list(map(float, dual)), lb=float(V.min()))
        for aff in (True, False):
            sp = robust_dual(bx, zv, zero, lev, affine=aff)
            if sp is None: continue
            V = B.mode1_vertices(bx, zv, zero, zv, zero, lev, sp)
            if float(V.min()) > 0:
                return dict(k='E', ed=sp, lb=float(V.min()))
    return None


def main_bound(bx, lev, rg, tp, tb, blend=True):
    OFFK = BB.OFFK
    cands = []; vals = []
    done = False
    for ch0 in BB.CHOICES:
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
            if float(V.min()) >= 0: done = True; break
        if done: break
    if not vals: return None
    Vm = np.array(vals); Vm = np.where(np.isfinite(Vm), Vm, -1e9)
    th, lb = BB.theta_combine(Vm)
    if lb < 0:
        for ch0 in BB.CHOICES:
            ch = dict(ch0, R=[1.0] + [0.0] * 8)
            cp, kp = M.main_cost(bx.Qp, rg, ch); cb, kb = M.main_cost(bx.Qb, rg, ch)
            for aff in (True, False):
                sp = robust_dual(bx, cp, kp - tp, lev, affine=aff)
                if sp is None: continue
                V = B.mode1_vertices(bx, cp, kp - tp, cb, kb - tb, lev, sp)
                vals.append(V); cands.append(dict(ch=ch, d=sp))
        Vm = np.array(vals); Vm = np.where(np.isfinite(Vm), Vm, -1e9)
        th, lb = BB.theta_combine(Vm)
    if lb < 0 and blend and np.all((bx.hi - bx.lo) <= np.array([0.02, 0.02, 0.03, 0.03])):
        ch = cands[int(np.argmax(Vm.min(axis=1)))]['ch']
        cp, kp = M.main_cost(bx.Qp, rg, ch); cb, kb = M.main_cost(bx.Qb, rg, ch); kp2 = kp - tp; kb2 = kb - tb
        cf = bx.fl(cp)
        feas, infe = [], []
        for (muv, dav, dbv, okp) in sample_points(bx, lev, bx.pm):
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
        th, lb = BB.theta_combine(Vm)
    keep = [i for i in range(len(th)) if th[i] > 0]
    bx.vbest = Vm[int(np.argmax(Vm.min(axis=1)))]
    return dict(c=[cands[i] for i in keep], th=[float(th[i]) for i in keep], lb=lb)


def box_eval(lo, hi, lev, pm, dom, tau0, inh=None):
    """returns (rec, bx, rng) ; rng = (ranges, duals, box_lo, box_hi, lev) to pass to children (inheritance)."""
    bx = make_box(lo, hi, pm)
    e = emptiness(bx, lev)
    if e is not None: return e, bx, inh
    tp = target_value(dom, bx, bx.Qp, lo, hi, tau0); tb = target_value(dom, bx, bx.Qb, lo, hi, tau0)
    if inh is not None:
        rg, rd, rlo, rhi, rlev = inh
        mb = main_bound(bx, lev, rg, tp, tb, blend=False)
        if mb is not None and mb['lb'] >= 0:
            return dict(k='G', rd=rd, rg={k_: list(v) for k_, v in rg.items()}, rbox=[list(rlo), list(rhi)], rlev=list(rlev), **mb), bx, inh
    rg, rd = ranges(bx, lev, pm)
    if rg is None: return dict(k='F', lb=-1.0), bx, inh
    for key, (a, b) in rg.items():
        if a > b: return dict(k='E2', key=key, rd={key: rd[key]}, lb=a - b), bx, inh
    new = (rg, rd, [float(x) for x in lo], [float(x) for x in hi], tuple(lev))
    mb = main_bound(bx, lev, rg, tp, tb, blend=True)
    if mb is None: return dict(k='F', lb=-1.0), bx, new
    return dict(k='G', rd=rd, rg={k_: list(v) for k_, v in rg.items()}, **mb), bx, new


def replay_G(r, pm, dom, tau0):
    """re-certify a G record (no LP): ranges from the recorded duals on rbox (default: the box) must lie inside the
    recorded ranges; every candidate's Mode 1 vertex values recomputed; the theta combination >= 0."""
    lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    rlo, rhi = (np.array(r['rbox'][0]), np.array(r['rbox'][1])) if 'rbox' in r else (lo, hi)
    rlev = tuple(r.get('rlev', lev))
    if not (np.all(rlo <= lo) and np.all(rhi >= hi) and rlev[0] <= lev[0] and rlev[1] >= lev[1]): return False, -4
    rb = make_box(rlo, rhi, pm)
    rg_c, _ = ranges(rb, rlev, pm, rec=r['rd'])
    rg = {}
    for key in M.RANGE_KEYS:
        a, b_ = rg_c[key]; rl, rh = r['rg'][key]
        if not (rl <= a and rh >= b_): return False, -2
        rg[key] = (rl, rh)
    bx = make_box(lo, hi, pm)
    tp = target_value(dom, bx, bx.Qp, lo, hi, tau0); tb = target_value(dom, bx, bx.Qb, lo, hi, tau0)
    if not r['th'] or min(r['th']) < 0 or sum(r['th']) < 0.999999: return False, -3
    s = np.zeros(16)
    for c, th in zip(r['c'], r['th']):
        cp, kp = M.main_cost(bx.Qp, rg, c['ch']); cb, kb = M.main_cost(bx.Qb, rg, c['ch'])
        V = B.mode1_vertices(bx, cp, kp - tp, cb, kb - tb, lev, c['d'])
        s = dn(s + dn(th * V))
    return float(s.min()) >= 0, float(s.min())


def replay_E(r, pm):
    lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    bx = make_box(lo, hi, pm); zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if r['k'] == 'E':
        V = B.mode1_vertices(bx, zv, zero, zv, zero, lev, r['ed']); return float(V.min()) > 0, float(V.min())
    key = r['key']; d1, d2 = r['rd'][key]
    vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
    a, _ = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
    b_, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
    return a > b_, a - b_


# ---------------------------------------------------------------------------------------------------------------------
# ROBUST AFFINE DUAL (speed-up; validity comes only from the Mode 1 certification of the returned dual).
def _mid(x):
    x = x.v if isinstance(x, T1) else x
    return 0.5 * (np.asarray(x.lo, float) + np.asarray(x.hi, float))


def _midg(x):
    return 0.5 * (np.asarray(x.g.lo, float) + np.asarray(x.g.hi, float))


def robust_dual(bx, cp, kp, lev, affine=True, YB=1e4):
    """A dual (ya, yb, eNS, elo, ehi), affine in z - zc, that maximises the minimum over the 16 vertices and the centre
    of the first-order (point-gradient) extrapolation of the Mode 1 bound of cp.m + kp.  Returns a dual spec or None."""
    from scipy.optimize import linprog
    from bk13_lb import AA, AB, KV
    Q = bx.Qp; zc = bx.zc
    samples = [np.zeros(4)] + [np.where(v == 1, bx.hi, bx.lo) - zc for v in VMASK]
    S = len(samples)
    c0 = _mid(cp); cg = _midg(cp)                    # (20,), (4, 20)
    k0 = float(_mid(kp)); kg = _midg(kp).reshape(4)
    ns0 = _mid(Q['nsg']); nsgg = _midg(Q['nsg'])
    nh0 = float(_mid(Q['nsh'])); nhg = _midg(Q['nsh']).reshape(4)
    mu0 = [float(_mid(m)) for m in Q['mu']]; mug = [_midg(m).reshape(4) for m in Q['mu']]
    da0, db0 = float(_mid(Q['da'])), float(_mid(Q['db'])); dag, dbg = _midg(Q['da']).reshape(4), _midg(Q['db']).reshape(4)
    slo, shi = lev
    ny = 5; nY = 20 if affine else 0
    nv = ny + nY + 1 + 3 * S
    it = ny + nY
    def iv(k, s): return it + 1 + 3 * s + k
    rows = []; rhs = []
    def yrow(i, D, coef, row):
        row[i] += coef
        if affine:
            for q in range(4): row[ny + 4 * i + q] += coef * D[q]
    for s, D in enumerate(samples):
        k_s = k0 + kg @ D
        bt = [da0 + dag @ D, db0 + dbg @ D, -(nh0 + nhg @ D), slo, -shi]
        mu_s = [mu0[k] + mug[k] @ D for k in range(3)]
        if min(mu_s) < 0: mu_s = [max(m, 0.0) for m in mu_s]
        row = np.zeros(nv); row[it] = 1.0
        for i in range(5): yrow(i, D, -bt[i], row)
        for k in range(3): row[iv(k, s)] -= mu_s[k]
        rows.append(row); rhs.append(k_s)
        c_s = c0 + D @ cg; ns_s = ns0 + D @ nsgg
        for j in range(M.NP):
            k = int(KV[j]) + 1
            row = np.zeros(nv); row[iv(k, s)] = 1.0
            At = [AA[j], AB[j], -ns_s[j], M.OFF[j], -M.OFF[j]]
            for i in range(5): yrow(i, D, At[i], row)
            rows.append(row); rhs.append(c_s[j])
        if s > 0:
            for i in (2, 3, 4):
                row = np.zeros(nv); yrow(i, D, -1.0, row); rows.append(row); rhs.append(-1e-9)
    obj = np.zeros(nv); obj[it] = -1.0
    bnds = [(-YB, YB)] * 2 + [(0, YB)] * 3 + [(-1e6, 1e6)] * nY + [(-10.0, 10.0)] + [(None, None)] * (3 * S)
    r = linprog(obj, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bnds, method='highs')
    if r.status != 0: return None
    x = r.x
    spec = []
    hw = 0.5 * (bx.hi - bx.lo)
    for i in range(5):
        v = float(x[i])
        g = [float(x[ny + 4 * i + q]) for q in range(4)] if affine else [0.0] * 4
        if i >= 2:
            need = float(sum(abs(gg) * ww for gg, ww in zip(g, hw)))
            if v < need * (1 + 1e-9) + 1e-14: v = need * (1 + 1e-9) + 1e-14
        spec.append([v, g] if any(gg != 0.0 for gg in g) else v)
    return spec
