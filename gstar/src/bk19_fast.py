"""bk19_fast.py : speed-ups of the box evaluation that change NO mathematics.

Import AFTER bk17_box (or through bk19_grun).  It replaces three helpers of bk17_box by faster equivalents:
  * sample_points: the float data (mu, d_a, d_b) of the 33 sample points are computed from closed float formulas instead
    of 33 full interval evaluations of bk16_model.quantities (the data only seed candidate duals);
  * robust_dual: the same LP (bit-identical matrix, built with numpy instead of a Python loop);
  * ranges: identical, except that the sample-point loop is skipped when NSAMP == 0 (in the 33 sample LPs were
    solved and then discarded).
Every bound that enters a record is still produced and certified by bk16_lb.mode1_vertices / certified_min, so these
helpers only change which candidate multipliers are tried and how fast.  The replay (bk17_verify) is unchanged.
"""
import numpy as np
import bk16_model as M
import bk16_lb as B
import bk17_box as XB
from bk13_lb import VMASK


def _floats_at(zv, pm):
    al, la = float(zv[0]), float(zv[1])
    q_ = 1.0 - la; S = q_ * (2.0 - al) + la
    m1 = al * q_ / S; m0 = la / S; p1 = (1.0 - al) * q_ * 2.0 / S
    if pm:
        tau = 2.5 - 1.5 * m1
        da = 0.5 * (tau + zv[2] + zv[3]); db = 0.5 * (tau + zv[2] - zv[3])
    else:
        da, db = float(zv[2]), float(zv[3])
    return [m1, m0, p1], float(da), float(db)


def sample_points(bx, lev, pm):
    if bx._pts is not None: return bx._pts
    pts = []
    for zv in [bx.zc] + [np.where(v == 1, bx.hi, bx.lo) for v in VMASK] + [0.5 * (bx.zc + np.where(v == 1, bx.hi, bx.lo)) for v in VMASK]:
        muv, dav, dbv = _floats_at(zv, pm)
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
            if loose and XB.NSAMP > 0:
                ntry = 0
                for (muv, dav, dbv, ok) in sample_points(bx, lev, pm):
                    if not ok: continue
                    if ntry >= XB.NSAMP: break
                    ntry += 1
                    dd, vv = B.float_lp_at(bx, -cf if mx else cf, lev, muv, dav, dbv)
                    if dd is None: continue
                    dd = [float(x) for x in dd]
                    val, _ = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, dual=dd)
                    if (mx and val < best) or ((not mx) and val > best): best, bd = val, dd
            res.append((best, bd))
        rg[key] = (res[0][0], res[1][0]); duals[key] = (res[0][1], res[1][1])
    return rg, duals


_mid, _midg = XB._mid, XB._midg


def robust_dual(bx, cp, kp, lev, affine=True, YB=1e4):
    """Same LP as bk17_box.robust_dual, matrix built with numpy (identical entries)."""
    from scipy.optimize import linprog
    from bk13_lb import AA, AB, KV
    Q = bx.Qp; zc = bx.zc
    samples = [np.zeros(4)] + [np.where(v == 1, bx.hi, bx.lo) - zc for v in VMASK]
    S = len(samples)
    c0 = _mid(cp); cg = _midg(cp)
    k0 = float(_mid(kp)); kg = _midg(kp).reshape(4)
    ns0 = _mid(Q['nsg']); nsgg = _midg(Q['nsg'])
    nh0 = float(_mid(Q['nsh'])); nhg = _midg(Q['nsh']).reshape(4)
    mu0 = [float(_mid(m)) for m in Q['mu']]; mug = [_midg(m).reshape(4) for m in Q['mu']]
    da0, db0 = float(_mid(Q['da'])), float(_mid(Q['db'])); dag, dbg = _midg(Q['da']).reshape(4), _midg(Q['db']).reshape(4)
    slo, shi = lev
    ny = 5; nY = 20 if affine else 0
    nv = ny + nY + 1 + 3 * S
    it = ny + nY
    NPn = M.NP
    nrows_s = 1 + NPn
    nrows = S * nrows_s + (S - 1) * 3
    A = np.zeros((nrows, nv)); rhs = np.zeros(nrows)
    KV1 = KV.astype(int) + 1
    OFF = M.OFF
    r = 0
    for s, D in enumerate(samples):
        k_s = k0 + kg @ D
        bt = np.array([da0 + dag @ D, db0 + dbg @ D, -(nh0 + nhg @ D), slo, -shi])
        mu_s = [mu0[k] + mug[k] @ D for k in range(3)]
        if min(mu_s) < 0: mu_s = [max(m, 0.0) for m in mu_s]
        # objective-cut row
        A[r, it] = 1.0
        A[r, :ny] += -bt
        if affine: A[r, ny:ny + nY] += np.outer(-bt, D).ravel()
        for k in range(3): A[r, it + 1 + 3 * s + k] -= mu_s[k]
        rhs[r] = k_s; r += 1
        # profile rows
        c_s = c0 + D @ cg; ns_s = ns0 + D @ nsgg
        At = np.stack([AA, AB, -ns_s, OFF, -OFF], axis=1)          # (NP, 5)
        blk = slice(r, r + NPn)
        A[blk, :ny] = At
        if affine: A[blk, ny:ny + nY] = (At[:, :, None] * D[None, None, :]).reshape(NPn, nY)
        A[np.arange(r, r + NPn), it + 1 + 3 * s + KV1] = 1.0
        rhs[blk] = c_s; r += NPn
        if s > 0:
            for i in (2, 3, 4):
                A[r, i] = -1.0
                if affine: A[r, ny + 4 * i: ny + 4 * i + 4] = -D
                rhs[r] = -1e-9; r += 1
    obj = np.zeros(nv); obj[it] = -1.0
    bnds = [(-YB, YB)] * 2 + [(0, YB)] * 3 + [(-1e6, 1e6)] * nY + [(-10.0, 10.0)] + [(None, None)] * (3 * S)
    res = linprog(obj, A_ub=A, b_ub=rhs, bounds=bnds, method='highs')
    if res.status != 0: return None
    x = res.x
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


def install():
    XB._orig = dict(sample_points=XB.sample_points, robust_dual=XB.robust_dual, ranges=XB.ranges)
    XB.sample_points = sample_points
    XB.robust_dual = robust_dual
    XB.ranges = ranges


# ---------------------------------------------------------------------------------------------------------------------
# OPTIONAL heuristic (env BK19_SKIPINF=1): a box whose CENTRE slice-level set is infeasible (float LP) and which is not
# certified empty is split at once, without computing ranges (such boxes straddle the edge of the feasible region, where
# ranges are loose).  Speed only: the box is not closed, its children are processed normally.
import os as _os
SKIPINF = _os.environ.get('BK19_SKIPINF', '0') == '1'


def box_eval(lo, hi, lev, pm, dom, tau0, inh=None):
    bx = XB.make_box(lo, hi, pm)
    zero = M.tconst(0.0)
    dual, val = B.float_lp(bx, np.zeros(M.NP), lev)
    cinf = dual is not None and val > 1e-7
    e = XB.emptiness(bx, lev) if cinf else None
    if e is not None: return e, bx, inh
    if SKIPINF and cinf: return dict(k='S', lb=-1.0), bx, inh
    tp = XB.target_value(dom, bx, bx.Qp, lo, hi, tau0); tb = XB.target_value(dom, bx, bx.Qb, lo, hi, tau0)
    if inh is not None:
        rg, rd, rlo, rhi, rlev = inh
        # note: Ybar uses the box constants of the box it is evaluated on; an inherited (parent) range [Ylo, Yhi]
        # is valid for the child's Ybar only as [min(Ylo, 0), Yhi] (0 <= Ybar_child <= Ybar_parent <= Yhi).
        rg = dict(rg); rg['Y'] = (min(rg['Y'][0], 0.0), rg['Y'][1])
        mb = XB.main_bound(bx, lev, rg, tp, tb, blend=False)
        if mb is not None and mb['lb'] >= 0:
            return dict(k='G', rd=rd, rg={k_: list(v) for k_, v in rg.items()}, rbox=[list(rlo), list(rhi)], rlev=list(rlev), **mb), bx, inh
    rg, rd = XB.ranges(bx, lev, pm)
    if rg is None: return dict(k='F', lb=-1.0), bx, inh
    for key, (a, b) in rg.items():
        if a > b: return dict(k='E2', key=key, rd={key: rd[key]}, lb=a - b), bx, inh
    new = (rg, rd, [float(x) for x in lo], [float(x) for x in hi], tuple(lev))
    mb = XB.main_bound(bx, lev, rg, tp, tb, blend=True)
    if mb is None: return dict(k='F', lb=-1.0), bx, new
    return dict(k='G', rd=rd, rg={k_: list(v) for k_, v in rg.items()}, **mb), bx, new


def install2():
    install()
    XB._orig['box_eval'] = XB.box_eval
    XB.box_eval = box_eval
