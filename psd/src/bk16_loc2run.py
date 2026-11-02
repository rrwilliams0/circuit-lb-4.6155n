"""bk16_loc2run.py : Lemma LOC2-K by second-order Taylor branch and bound (adapted from the bk13_L3b.py).
(a) Psi_K(a, lam, t, t') - 2 P0 lam / S_min >= 0 on K = [0, RA] x [0, RL] x T x T, equality only at 0;
(b) h_K(p, t) - P0 mu_0(p) >= -TAU0 on [-RA, RA] x [0, RL] x T.   Units 4 mu0 s.  CERTIFIED (interval arithmetic)."""
import os, sys, json, time, itertools
import numpy as np
from ivarith import IA, dn, up
from tay2 import Tay2
from bk13_L3b import taylor_lb
from bk13_L3 import specnorm_bound
import bk16_loc2 as L2
RA = 0.0300001; RL = 0.005; T = (-0.027, 0.029)
P0 = float(sys.argv[1]); TAU0 = float(sys.argv[2]); EPS0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.002
SMIN = float(dn(dn((1 - RL) * (2 - 0.5 - RA))))       # lower bound of S on Q (S = (1-lam)(2-alpha)+lam >= (1-lam)(2-alpha))


def origin_ok(ra, rl, T_, P0, Smin):
    lo = np.array([0.0, 0.0, T_[0], T_[0]]); hi = np.array([ra, rl, T_[1], T_[1]])
    z0 = np.zeros(4); P = L2.psi_tay(z0, z0); glam = float(P.g.lo[1])
    H = L2.psi_tay(lo, hi).h
    Hmid = 0.5 * (H.lo + H.hi); Hrad = up(0.5 * (H.hi - H.lo)); X = [0, 2, 3]
    cX = float(np.linalg.eigvalsh(Hmid[np.ix_(X, X)]).min()) - specnorm_bound(Hrad[np.ix_(X, X)]) - 1e-12
    hX = float(np.sqrt(sum(max(abs(H.lo[1, i]), abs(H.hi[1, i])) ** 2 for i in X)))
    hll = max(abs(H.lo[1, 1]), abs(H.hi[1, 1]))
    G = glam - 2 * P0 / Smin - 0.5 * hll * rl
    ok = bool(cX > 0 and G > 0 and rl <= 2 * cX * G / hX ** 2)
    return ok, dict(glam=glam, cX=cX, hX=hX, hll=hll, G=G, need=2 * cX * G / max(hX, 1e-300) ** 2,
                    psi0=[float(P.v.lo), float(P.v.hi)], grad0=[[float(x) for x in P.g.lo], [float(x) for x in P.g.hi]])


if __name__ == '__main__':
    out = open('loc2K_boxes.jsonl', 'w')
    eps0 = EPS0; epl = min(RL, eps0)
    ok, info = origin_ok(eps0, epl, (-eps0, eps0), P0, SMIN)
    print('origin box a<=%.4f lam<=%.4f |t|,|t\'|<=%.4f:' % (eps0, epl, eps0), ok, info, flush=True)
    out.write(json.dumps(dict(kind='origin', ok=ok, eps0=eps0, epl=epl, P0=P0, Smin=SMIN, **info)) + '\n')
    if not ok: sys.exit(1)

    def inside_origin(lo, hi):
        return lo[0] >= 0 and hi[0] <= eps0 and lo[1] >= 0 and hi[1] <= epl and all(lo[i] >= -eps0 and hi[i] <= eps0 for i in (2, 3))
    t0 = time.time()
    grid = [np.linspace(0, RA, 16), np.linspace(0, RL, 3), np.linspace(T[0], T[1], 8), np.linspace(T[0], T[1], 8)]
    grid[0] = np.unique(np.concatenate([grid[0], [eps0]])); grid[1] = np.unique(np.concatenate([grid[1], [epl]]))
    grid[2] = np.unique(np.concatenate([grid[2], [-eps0, eps0]])); grid[3] = np.unique(np.concatenate([grid[3], [-eps0, eps0]]))
    stack = []
    for idx in itertools.product(*[range(len(g) - 1) for g in grid]):
        stack.append((np.array([grid[i][idx[i]] for i in range(4)]), np.array([grid[i][idx[i] + 1] for i in range(4)]), 0))
    nclosed = 0; nopen = 0; worst = 1.0; scale = np.array([RA, RL, T[1] - T[0], T[1] - T[0]])
    while stack:
        lo, hi, dep = stack.pop()
        if inside_origin(lo, hi): nclosed += 1; continue
        lb, P = taylor_lb(L2.psi_tay, lo, hi)
        lb = float(dn(lb - float(up(2 * P0 * hi[1] / SMIN))))
        if lb >= 0:
            nclosed += 1; worst = min(worst, lb); out.write(json.dumps(dict(kind='psi', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        if dep > 16: nopen += 1; out.write(json.dumps(dict(kind='OPEN', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        w = (hi - lo) / scale; i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
        h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
    print('coupled B&B: closed %d open %d, min lb %.3e, %.0f s' % (nclosed, nopen, worst, time.time() - t0), flush=True)

    def h3(lo, hi):
        a, la, t = [Tay2.var(i, 3, lo[i], hi[i]) for i in range(3)]
        return L2.h_kill(a + 0.5, la, t)
    stack = []
    g3 = [np.linspace(-RA, RA, 13), np.linspace(0, RL, 2), np.linspace(T[0], T[1], 8)]
    for idx in itertools.product(*[range(len(g) - 1) for g in g3]):
        stack.append((np.array([g3[i][idx[i]] for i in range(3)]), np.array([g3[i][idx[i] + 1] for i in range(3)]), 0))
    n1 = 0; o1 = 0; w1 = 1.0
    while stack:
        lo, hi, dep = stack.pop()
        lb, _ = taylor_lb(h3, lo, hi); lb = float(dn(lb - float(up(P0 * hi[1] / SMIN))))
        if lb >= -TAU0: n1 += 1; w1 = min(w1, lb); out.write(json.dumps(dict(kind='single', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        if dep > 14: o1 += 1; out.write(json.dumps(dict(kind='OPEN1', lo=lo.tolist(), hi=hi.tolist(), lb=lb)) + '\n'); continue
        w = (hi - lo) / np.array([2 * RA, RL, T[1] - T[0]]); i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
        h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
        stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
    print('single-star bound h_K - P0 mu0 >= -TAU0 (%.1e): closed %d open %d, min lb %.3e' % (TAU0, n1, o1, w1), flush=True)
    out.close()
