r"""bk19_verify.py : bk17_verify.py with applied to the replay of 'G' records.

For an inherited record ('rbox' present) the range of Ybar used in the replay is [min(Ylo, 0), Yhi] (Ylo, Yhi the recorded
range of the parent's Ybar, re-certified on rbox as before), and the replay checks pq(child) <= pq(parent) componentwise
(box constants of Ybar), so that 0 <= Ybar_child <= Ybar_parent <= Yhi.  Everything else is bk17_verify unchanged.
Usage: python bk19_verify.py <Q|G> replay <stride> <offset> <records...>   (as bk17_verify.py)
"""
import sys, os
import numpy as np
import bk17_verify as V            # parses sys.argv[1] (domain)
import bk17_box as XB
import bk16_model as M
import bk16_lb as B
from ivarith import dn


def replay_G(r, pm, dom, tau0):
    lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    inh = 'rbox' in r
    rlo, rhi = (np.array(r['rbox'][0]), np.array(r['rbox'][1])) if inh else (lo, hi)
    rlev = tuple(r.get('rlev', lev))
    if not (np.all(rlo <= lo) and np.all(rhi >= hi) and rlev[0] <= lev[0] and rlev[1] >= lev[1]): return False, -4
    rb = XB.make_box(rlo, rhi, pm)
    rg_c, _ = XB.ranges(rb, rlev, pm, rec=r['rd'])
    rg = {}
    for key in M.RANGE_KEYS:
        a, b_ = rg_c[key]; rl, rh = r['rg'][key]
        if not (rl <= a and rh >= b_): return False, -2
        rg[key] = (rl, rh)
    bx = XB.make_box(lo, hi, pm)
    if inh:
        rg['Y'] = (min(rg['Y'][0], 0.0), rg['Y'][1])
        if not np.all(np.asarray(bx.Qb['BC']['pq']) <= np.asarray(rb.Qb['BC']['pq'])): return False, -13
    tp = XB.target_value(dom, bx, bx.Qp, lo, hi, tau0); tb = XB.target_value(dom, bx, bx.Qb, lo, hi, tau0)
    if not r['th'] or min(r['th']) < 0 or sum(r['th']) < 0.999999: return False, -3
    s = np.zeros(16)
    for c, th in zip(r['c'], r['th']):
        cp, kp = M.main_cost(bx.Qp, rg, c['ch']); cb, kb = M.main_cost(bx.Qb, rg, c['ch'])
        Vv = B.mode1_vertices(bx, cp, kp - tp, cb, kb - tb, lev, c['d'])
        s = dn(s + dn(th * Vv))
    return float(s.min()) >= 0, float(s.min())


XB.replay_G = replay_G
_J = None


def replay(r):
    """bk17_verify.replay plus Lemma CAP records: 'XJ' (marginal cell excluded by the certified Emin table) and records
    carrying 'cth' (the capped NS row; cth must be >= c_Theta - W_LO Emin_lo(mirror cell), replayed with that row)."""
    global _J
    if r.get('k') == 'XJ' or 'cth' in r:
        if _J is None:
            import bk19_joint as J; J.table(); _J = J
        i, j = _J.root_of(r['lo'], r['hi'])
        if r.get('k') == 'XJ': return (V.DOM == 'G' and list(r['cell']) == [i, j] and _J.excluded(i, j)), 1.0
        if not (r['cth'] >= _J.cth_cap(i, j)): return False, -14
        from ivarith import IA
        M.CTH = IA(r['cth'])
        try:
            return V.replay(r)
        finally:
            M.CTH = _J.CTH0
    if V.DOM == 'Q' and r.get('k') == 'G' and r.get('tq', 0.0) < 0:
        # Theorem CS-K'': only cell C_11 may carry a negative target, and only >= -1.5e-4 (see bk19_qrun.T11)
        import bk17_run as RUN, bk16_bnb as BB
        lo, hi = r['lo'], r['hi']
        if not BB.in_Q(lo, hi) or RUN.q_cell(lo, hi) != 11 or r['tq'] < -1.5e-4: return False, -15
        return XB.replay_G(r, RUN.PM, 'Q', r['tq'])
    return V.replay(r)

if __name__ == '__main__':
    mode = sys.argv[2]
    assert mode == 'replay'
    import json
    stride, off = int(sys.argv[3]), int(sys.argv[4]); files = sys.argv[5:]
    bad = 0; n = 0; kinds = {}; minlb = {}; i = -1; ninh = 0
    for fn in files:
        for l in open(fn):
            i += 1
            r = json.loads(l); kk = r.get('k'); kinds[kk] = kinds.get(kk, 0) + 1
            if i % stride != off: continue
            if os.path.exists('replay.stop'): print('STOPPED by replay.stop at record', i, flush=True); break
            n += 1; ninh += ('rbox' in r)
            try:
                ok, lb = replay(r)
            except Exception as ex:
                ok, lb = False, -10
            minlb[kk] = min(minlb.get(kk, 1e9), lb)
            if not ok: bad += 1; print('FAIL', fn, i, kk, r['lo'], r['hi'], r['lev'], lb, flush=True)
    print('records %d kinds %s checked %d (inherited %d) (stride %d off %d) failures %d min margins %s' % (i + 1, kinds, n, ninh, stride, off, bad, minlb), flush=True)
