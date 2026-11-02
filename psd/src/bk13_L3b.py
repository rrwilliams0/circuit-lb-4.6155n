"""bk13_L3b.py: branch and bound for (L3): Psi(a, lam, t, t') - pen >= 0 on [0, ra] x [0, rl] x T x T, and
(L3s): h(p, t) - P0 mu_0 >= -tau0 on [-ra, ra] x [0, rl] x T.  Box bounds: second-order Taylor at the box centre
(value and gradient by point interval AD, Hessian by interval AD over the box); the origin box by the exact expansion
of bk13_L3 (Psi(0) = 0, zero gradient except dPsi/dlam).  Records every closed box in L3_boxes.jsonl.  CERTIFIED."""
import sys, json, time
import numpy as np
from ivarith import IA, dn, up
from tay2 import Tay2
from bk13_L3 import h_tay, psi_tay, specnorm_bound

def taylor_lb(fun, lo, hi):
    """lower bound of fun over the box by f(c) + min g.d + 1/2 sum_il min H_il d_i d_l (all interval-rigorous)."""
    n = len(lo); c = 0.5 * (lo + hi)
    P = fun(c, c); B = fun(lo, hi)
    dl = lo - c; dh = hi - c                    # exact? c rounded; use outward intervals
    D = [IA(dn(dl[i]), up(dh[i])) for i in range(n)]
    tot = IA(float(P.v.lo), float(P.v.hi))
    for i in range(n):
        tot = tot + IA(float(P.g.lo[i]), float(P.g.hi[i])) * D[i]
    for i in range(n):
        for l in range(n):
            Hil = IA(float(B.h.lo[i, l]), float(B.h.hi[i, l]))
            prod = (D[i] * D[i]) if i == l else (D[i] * D[l])
            if i == l: prod = IA(0.0, float(prod.hi))
            tot = tot + Hil * prod * 0.5
    return float(tot.lo), P

def origin_ok(ra, rl, T, P0, Smin):
    lo = np.array([0.0, 0.0, T[0], T[0]]); hi = np.array([ra, rl, T[1], T[1]])
    z0 = np.zeros(4); P = psi_tay(z0, z0); glam = float(P.g.lo[1])
    H = psi_tay(lo, hi).h
    Hmid = 0.5 * (H.lo + H.hi); Hrad = up(0.5 * (H.hi - H.lo)); X = [0, 2, 3]
    cX = float(np.linalg.eigvalsh(Hmid[np.ix_(X, X)]).min()) - specnorm_bound(Hrad[np.ix_(X, X)]) - 1e-12
    hX = float(np.sqrt(sum(max(abs(H.lo[1, i]), abs(H.hi[1, i])) ** 2 for i in X)))
    hll = max(abs(H.lo[1, 1]), abs(H.hi[1, 1]))
    G = glam - 2 * P0 / Smin - 0.5 * hll * rl
    ok = cX > 0 and G > 0 and rl <= 2 * cX * G / hX ** 2
    ok = bool(ok)
    return ok, dict(glam=glam, cX=cX, hX=hX, hll=hll, G=G, need=2 * cX * G / max(hX, 1e-300) ** 2)

if __name__ == '__main__':
    ra, rl = 0.03, 0.005; T = (-0.035, 0.036); P0 = 3.934e-3 * 1.01; tau0 = 1e-4
    Smin = (1 - rl) * (2 - 0.5 - ra)
    eps0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002
    out = open('L3_boxes.jsonl', 'w')
    # origin box
    epl = min(rl, eps0)
    ok, info = origin_ok(eps0, epl, (-eps0, eps0), P0, Smin)
    print('origin box a<=%.4f lam<=%.4f |t|,|t\'|<=%.4f:' % (eps0, epl, eps0), ok, info, flush=True)
    out.write(json.dumps(dict(kind='origin', ok=ok, eps0=eps0, epl=epl, **info)) + '\n')
    if not ok: sys.exit(1)
    def inside_origin(lo, hi):
        return lo[0] >= 0 and hi[0] <= eps0 and lo[1] >= 0 and hi[1] <= epl and all(lo[i] >= -eps0 and hi[i] <= eps0 for i in (2, 3))
    # coupled B&B
    t0 = time.time()
    grid = [np.linspace(0, ra, 16), np.linspace(0, rl, 3), np.linspace(T[0], T[1], 8), np.linspace(T[0], T[1], 8)]
    # make eps0 a grid line: insert breakpoints
    grid[0] = np.unique(np.concatenate([grid[0], [eps0]])); grid[1] = np.unique(np.concatenate([grid[1], [epl]]))
    grid[2] = np.unique(np.concatenate([grid[2], [-eps0, eps0]])); grid[3] = np.unique(np.concatenate([grid[3], [-eps0, eps0]]))
    import itertools
    stack = []
    for idx in itertools.product(*[range(len(g) - 1) for g in grid]):
        lo = np.array([grid[i][idx[i]] for i in range(4)]); hi = np.array([grid[i][idx[i] + 1] for i in range(4)])
        stack.append((lo, hi, 0))
    nclosed = 0; nopen = 0; worst = 1.0; scale = np.array([ra, rl, T[1] - T[0], T[1] - T[0]])
    while stack:
        lo, hi, dep = stack.pop()
        if inside_origin(lo, hi): nclosed += 1; continue
        lb, P = taylor_lb(psi_tay, lo, hi)
        lb -= 2 * P0 * hi[1] / Smin
        if lb >= 0:
            nclosed += 1; worst = min(worst, lb); out.write(json.dumps(dict(kind='psi', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        if dep > 14: nopen += 1; out.write(json.dumps(dict(kind='OPEN', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        w = (hi - lo) / scale; i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
        h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
    print('coupled B&B: closed %d open %d, min lb %.3e, %.0f s' % (nclosed, nopen, worst, time.time() - t0), flush=True)
    # single star bound h - P0 mu0 >= -tau0
    def h3(lo, hi):
        a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
        return h_tay(a + 0.5, la, t)
    stack = []
    g3 = [np.linspace(-ra, ra, 13), np.linspace(0, rl, 2), np.linspace(T[0], T[1], 8)]
    for idx in itertools.product(*[range(len(g) - 1) for g in g3]):
        stack.append((np.array([g3[i][idx[i]] for i in range(3)]), np.array([g3[i][idx[i] + 1] for i in range(3)]), 0))
    n1 = 0; o1 = 0; w1 = 1.0
    while stack:
        lo, hi, dep = stack.pop()
        lb, _ = taylor_lb(h3, lo, hi); lb -= P0 * hi[1] / Smin
        if lb >= -tau0: n1 += 1; w1 = min(w1, lb); out.write(json.dumps(dict(kind='single', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        if dep > 12: o1 += 1; out.write(json.dumps(dict(kind='OPEN1', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        w = (hi - lo) / np.array([2 * ra, rl, T[1] - T[0]]); i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
        h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
    print('single-star bound h - P0 mu0 >= -tau0: closed %d open %d, min lb %.3e' % (n1, o1, w1))
    out.close()
