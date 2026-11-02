"""bk13_L2.py: local structure check (L2) on Q x D_loc.
Coordinates (alpha, lam, dp, dm): d_a = (tau(p) + dp + dm)/2, d_b = (tau(p) + dp - dm)/2, tau = 5/2 - (3/2) mu_-1.
Bases B+ = {A, B, C, j0, (1,-1,-1)}, B- = {A, B, C, j0, (1,1,1)}, j0 = (0,1,-1).
Checks on every sub-box: (a) all nonbasic reduced costs >= 0 except j0' = (0,-1,1); records P0 = max(0, -rc_j0');
(b) dF_{B+}/d dp >= 0 on dp in [0, W+];  dF_{B-}/d dp <= 0 on dp in [-W+, 0].   CERTIFIED (interval arithmetic)."""
import sys, json, itertools
import bk13_lb as L
from bk13_lb import *
A_, B_, C_ = PU.index((-1, 1, 1)), PU.index((1, -1, 1)), PU.index((1, 1, -1))
J0, J0p, JP, JM = PU.index((0, 1, -1)), PU.index((0, -1, 1)), PU.index((1, -1, -1)), PU.index((1, 1, 1))
BP = tuple(sorted((A_, B_, C_, J0, JP))); BM = tuple(sorted((A_, B_, C_, J0, JM)))

def evalq(lo, hi):
    al, la, dp, dm = [T1.var(i, 4, lo[i], hi[i]) for i in range(4)]
    q = 1.0 - la; S = q * (2.0 - al) + la; m1 = al * q * S.inv()
    tau = 2.5 - m1 * 1.5
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    return L.evalz([al, la, da, db])

def check(lo, hi, side):
    E = evalq(lo, hi)
    basis, M, Y, Gam = basis_data(BP if side > 0 else BM)
    ph = E['phi'].v
    rc = rsum(Gam * IA(ph.lo[None, :], ph.hi[None, :]), 1)
    nb = [j for j in range(NP) if j not in basis and j != J0p]
    rcmin = float(rc.lo[nb].min()); jworst = PU[nb[int(np.argmin(rc.lo[nb]))]]
    P0 = max(0.0, -float(rc.lo[J0p]))
    F = F2(E, M)
    gdp = F.g[2]
    mono = float(gdp.lo) if side > 0 else -float(gdp.hi)
    return rcmin, jworst, P0, mono

def run(Wp, Wm, Qa=(0.47, 0.53), Ql=(0.0, 0.005), n=(6, 2, 2, 6), maxdepth=6):
    res = dict(rcmin=1.0, P0=0.0, mono=1.0, boxes=0, fails=[])
    for side in (+1, -1):
        dpr = (0.0, Wp) if side > 0 else (-Wp, 0.0)
        rng_ = [Qa, Ql, dpr, (-Wm, Wm)]
        grids = [np.linspace(r[0], r[1], k + 1) for r, k in zip(rng_, n)]
        stack = []
        for idx in itertools.product(*[range(k) for k in n]):
            lo = np.array([grids[i][idx[i]] for i in range(4)]); hi = np.array([grids[i][idx[i] + 1] for i in range(4)])
            stack.append((lo, hi, 0))
        while stack:
            lo, hi, dep = stack.pop()
            rcmin, jw, P0, mono = check(lo, hi, side)
            if rcmin >= 0 and mono > 0:
                res['boxes'] += 1; res['rcmin'] = min(res['rcmin'], rcmin); res['P0'] = max(res['P0'], P0); res['mono'] = min(res['mono'], mono)
                continue
            if dep >= maxdepth:
                res['fails'].append((side, lo.tolist(), hi.tolist(), rcmin, jw, mono)); continue
            w = (hi - lo) / np.array([Qa[1] - Qa[0], Ql[1] - Ql[0], Wp, 2 * Wm])
            i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
            h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
            stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
    return res

if __name__ == '__main__':
    for Wp, Wm in [(0.004, 0.06), (0.01, 0.1), (0.02, 0.15)]:
        r = run(Wp, Wm)
        print('W+ %.3f W- %.3f: boxes %d  min rc (others) %.5f  P0 %.3e  min |dF/ddp| %.5f  fails %d' % (Wp, Wm, r['boxes'], r['rcmin'], r['P0'], r['mono'], len(r['fails'])), flush=True)
        for f in r['fails'][:3]: print('   fail', f)
