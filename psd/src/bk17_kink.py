"""bk17_kink.py : Lemma KINK-K box check with the REPAIRED slope condition.

the kink_check certifies d/dd_+ (F_B + eta d_+) >= 0 only on the box itself.  The reduction (Lemma KINK-K) needs it on the
whole segment from the kink line: F_B(p, d_+, d_-) + eta d_+ >= F_B(p, 0, d_-) for every (p, d) of the box, i.e. the slope
condition on the EXTENDED box  [alpha] x [lam] x [0, dp_hi] x [dm]  (side +) or  [dp_lo, 0]  (side -), with the SAME
function F_B (the box's local cost: its ranges and box constants) and the SAME eta.  kink_check2 does exactly this.
Everything else (ranges, hinges, centred reduced costs, eta, P0) is the kink_check, unchanged.
"""
import numpy as np
from ivarith import IA, dn, up
from bk13_lb import T1, NP
import bk16_model as M
import bk16_loc as LC
import bk16_lb as B


def kink_check2(lo, hi, side, s1, rec=None, slopes=True):
    bx = B.Box(lo, hi, pm=True); lev = (0.0, s1); zero = M.tconst(0.0)
    rg = {}; rd = {}
    for key in LC.LOCAL_KEYS:
        vp = M.functional(bx.Qp, key); vb = M.functional(bx.Qb, key)
        d1 = rec['rd'][key][0] if rec else None; d2 = rec['rd'][key][1] if rec else None
        a, d1 = B.certified_min(bx, vp, vb, zero, zero, lev, dual=d1, slopes=slopes)
        b_, d2 = B.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2, slopes=slopes)
        if a is None or b_ is None: return dict(ok=False, why='range')
        rg[key] = (a, b_); rd[key] = (d1, d2)
    Fb = max(abs(rg['fi'][0]), abs(rg['fi'][1]))
    if not LC.hinges_inactive(bx, Fb): return dict(ok=False, why='hinge', rg=rg)
    lcb = LC.local_cost(bx.Qb, rg); lcp = LC.local_cost(bx.Qp, rg)
    if lcb is None or lcp is None: return dict(ok=False, why='phibar', rg=rg)
    basis = LC.BPLUS if side > 0 else LC.BMINUS
    F, rcb, xB = LC.mode2_parts(bx.Qb, lcb[0], lcb[1], basis)
    Fp, rcp_, _ = LC.mode2_parts(bx.Qp, lcp[0], lcp[1], basis)
    rcI = LC.centered(rcp_, rcb, bx)
    rclo = np.asarray(rcI.lo)
    if rec is not None: eta = rec['eta']
    elif side > 0:
        cand = [rclo[j] / LC.DELTA[j] for j in range(NP) if j not in basis and LC.DELTA[j] > 0 and rclo[j] < 0]
        eta = float(dn(min(cand))) * 1.0000001 if cand else 0.0
        eta = min(eta, 0.0)
    else:
        eta = 0.0
    rcp = rcI - IA(LC.DELTA) * eta
    nb = [j for j in range(NP) if j not in basis and j != LC.ij0p]
    rcmin = float(np.asarray(rcp.lo)[nb].min())
    P0 = max(0.0, -float(np.asarray(rcp.lo)[LC.ij0p]))
    # slope on the extended box (same rg, same box constants BC, same eta)
    elo, ehi = np.array(lo, float), np.array(hi, float)
    if side > 0: elo[2] = 0.0
    else: ehi[2] = 0.0
    Qe = M.quantities_pm([T1.var(i, 4, elo[i], ehi[i]) for i in range(4)], BC=bx.Qb['BC'])
    lce = LC.local_cost(Qe, rg)
    if lce is None: return dict(ok=False, why='phibar_ext', rg=rg)
    Fe, _, _ = LC.mode2_parts(Qe, lce[0], lce[1], basis)
    g = IA(Fe.g.lo[2], Fe.g.hi[2]) + eta
    slope = float(g.lo) if side > 0 else -float(g.hi)
    if side > 0 and slope < 0:
        # second-order centred enclosure of dF_B+/ddp on the extended box (bk17_fbp; F_B+ is range-free)
        import bk17_fbp as FP
        slo, _ = FP.slope_lo(elo, ehi)
        slope = max(slope, float(dn(slo + eta)))
    ok = rcmin >= 0 and slope >= 0
    return dict(ok=bool(ok), why='' if ok else ('rc' if rcmin < 0 else 'slope'), rcmin=rcmin, P0=P0, slope=slope, eta=eta,
                rd=rd, rg={k_: list(v) for k_, v in rg.items()})
