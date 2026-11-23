"""bk16_loc2verify.py: replay of the LOC2-K record (every Taylor box recomputed; origin box recomputed; exact disjoint tiling checked)."""
import sys, json, math
from fractions import Fraction as Fr


def check_exact_tiling(dom_lo, dom_hi, boxes):
    """Verify that boxes form an exact pairwise interior-disjoint cover of [dom_lo, dom_hi] in exact rationals."""
    dim = len(dom_lo)
    dlo = [Fr(x) for x in dom_lo]; dhi = [Fr(x) for x in dom_hi]
    def vol(lo, hi):
        v = Fr(1)
        for a, b in zip(lo, hi): v *= b - a
        return v
    dom_vol = vol(dlo, dhi)
    malformed = 0; out_of_bounds = 0; overlaps = 0; total_vol = Fr(0)
    exact = []
    for lo, hi in boxes:
        if len(lo) != dim or len(hi) != dim or any(not math.isfinite(x) for x in lo) or any(not math.isfinite(x) for x in hi):
            malformed += 1; continue
        lf = [Fr(x) for x in lo]; hf = [Fr(x) for x in hi]
        if any(a >= b for a, b in zip(lf, hf)):
            malformed += 1; continue
        if any(lf[d] < dlo[d] or hf[d] > dhi[d] for d in range(dim)):
            out_of_bounds += 1
        total_vol += vol(lf, hf)
        exact.append((lf, hf))
    exact.sort(key=lambda b: b[0][0])
    for i in range(len(exact)):
        a_lo, a_hi = exact[i]
        for j in range(i + 1, len(exact)):
            b_lo, b_hi = exact[j]
            if b_lo[0] >= a_hi[0]: break
            if all(max(a_lo[d], b_lo[d]) < min(a_hi[d], b_hi[d]) for d in range(dim)):
                overlaps += 1
    vol_ok = (total_vol == dom_vol)
    ok = (malformed == 0 and out_of_bounds == 0 and overlaps == 0 and vol_ok)
    return ok, dict(vol_ok=vol_ok, overlaps=overlaps, out_of_bounds=out_of_bounds, malformed=malformed)


if __name__ == '__main__':
    import numpy as np
    from ivarith import dn, up
    from tay2 import Tay2
    from bk13_L3b import taylor_lb
    import bk16_loc2 as L2

    fn = sys.argv[1]
    recs = [json.loads(l) for l in open(fn)]
    o = recs[0]
    sys.argv = ['x', str(o['P0']), '1.5e-4', str(o['eps0'])]
    import bk16_loc2run as R
    header_ok = (
        o.get('kind') == 'origin'
        and o['P0'] == 0.02
        and o['Smin'] <= R.SMIN
        and o['eps0'] > 0
        and o['epl'] == min(R.RL, o['eps0'])
    )
    ok, info = R.origin_ok(o['eps0'], o['epl'], (-o['eps0'], o['eps0']), o['P0'], o['Smin'])
    print('origin', bool(ok and header_ok), info['cX'], info['G'], info['need'])

    def h3(lo, hi):
        a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
        return L2.h_kill(a + 0.5, la, t)

    bad = 0; mn = {'psi': 1, 'single': 1}; n = 0
    for r in recs[1:]:
        lo, hi = np.array(r['lo']), np.array(r['hi'])
        if r['kind'] == 'psi':
            lb, _ = taylor_lb(L2.psi_tay, lo, hi); lb = float(dn(lb - float(up(2 * o['P0'] * hi[1] / o['Smin']))))
            if lb < 0: bad += 1
        elif r['kind'] == 'single':
            lb, _ = taylor_lb(h3, lo, hi); lb = float(dn(lb - float(up(o['P0'] * hi[1] / o['Smin']))))
            if lb < -1.5e-4: bad += 1
        else: bad += 1; continue
        mn[r['kind']] = min(mn[r['kind']], lb); n += 1
    print('replayed', n, 'boxes, failures', bad, 'min', mn)

    e = o['eps0']; ra, rl, T = R.RA, R.RL, R.T
    psi_boxes = [(r['lo'], r['hi']) for r in recs[1:] if r['kind'] == 'psi'] + [([0, 0, -e, -e], [e, min(rl, e), e, e])]
    single_boxes = [(r['lo'], r['hi']) for r in recs[1:] if r['kind'] == 'single']
    psi_ok, psi_st = check_exact_tiling([0, 0, T[0], T[0]], [ra, rl, T[1], T[1]], psi_boxes)
    single_ok, single_st = check_exact_tiling([-ra, 0, T[0]], [ra, rl, T[1]], single_boxes)
    print('psi volume + origin box == K volume:', psi_st['vol_ok'])
    print('single volume == domain volume:', single_st['vol_ok'])
    print('psi tiling valid:', psi_ok, psi_st)
    print('single tiling valid:', single_ok, single_st)
    if not (header_ok and ok and bad == 0 and psi_ok and single_ok):
        sys.exit(1)
