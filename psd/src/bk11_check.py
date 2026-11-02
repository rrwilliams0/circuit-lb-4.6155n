"""bk11_check.py: (a) float sanity check of Theorem LOCAL's bound at random points of the ball (the certified lower
bound must lie below the true slack); (b) the rule bound RB(m) on the kill-PSD failure set: is RB <= M_tree there?"""
import sys, json, math
import numpy as np
from bk11_core import *

rng = np.random.default_rng(7)
MTREE = 0.27658360688237347


def key_corrs(m):
    ys = keys(m)   # ua ub vc vd uv
    ku = [ys[4] @ ys[0], ys[4] @ ys[1], ys[0] @ ys[1]]
    kv = [ys[4] @ ys[2], ys[4] @ ys[3], ys[2] @ ys[3]]
    return ku, kv


def RB(m):
    ku, kv = key_corrs(m)
    return (sum(math.acos(min(1, k)) for k in ku) + sum(math.acos(min(1, k)) for k in kv)) / (2 * math.pi)


if __name__ == '__main__':
    rec = json.load(open('local_cert.json'))
    o = [r for r in rec if r['r'] == 3e-4][0]
    mu1, cF, cc, c1, r = o['mu1'], o['cF'], o['cc'], o['c1'], o['r']
    print('(a) Theorem LOCAL constants at r = %g: mu1 %.6f cF %.6f cc %.4f c1 %.4f rO %.3e' % (r, mu1, cF, cc, c1, o['rO']))
    mT = mtree(); worst = np.inf; nviol = 0
    for trial in range(3000):
        # random v_F (sum 0) and v_O >= 0 with |v_F|_1 + |v_O|_1 <= r
        vF = rng.normal(size=4); vF -= vF.mean(); vF /= np.abs(vF).sum()
        k = rng.integers(0, 6); vO = np.zeros(NT)
        if k > 0:
            idx = rng.choice(OFF_IDX, size=k, replace=False); vO[idx] = rng.dirichlet(np.ones(k))
        share = rng.uniform(0, 1); tot = r * rng.uniform(0.2, 1.0)
        if k == 0: share = 1.0
        v = np.zeros(NT); v[TREE_IDX] = share * tot * vF; v += (1 - share) * tot * vO
        m = mT + v
        s, info = slack(m)
        x = np.linalg.norm(v[TREE_IDX]); y = v[OFF_IDX].sum()
        lb = y * (mu1 - c1 * y) + cF * x * x - cc * x * y
        assert info['killpsd']
        if s < lb: nviol += 1
        worst = min(worst, s - lb)
    print('   3000 points: certified lower bound exceeded the true slack at', nviol, 'points; min(slack - bound) = %.3e' % worst)
    # (b) rule bound on the failure set
    print('(b) rule bound on kill-PSD failure points')
    fails = []; maxRB = -1; maxf_pt = None; n = 0
    for trial in range(4000):
        kind = trial % 4
        if kind == 0:
            m = np.zeros(NT); idx = rng.choice(NT, size=2, replace=False); m[idx] = rng.dirichlet(np.ones(2))
        elif kind == 1:
            m = np.zeros(NT); k = rng.integers(2, 6); idx = rng.choice(NT, size=k, replace=False); m[idx] = rng.dirichlet(np.ones(k))
        elif kind == 2:
            m = np.zeros(NT); m[TREE_IDX] = rng.dirichlet(0.5 * np.ones(4))
        else:
            m = rng.dirichlet(0.02 * np.ones(NT))
        c = Cv(m); ck, d = kill(c)
        if lam_min(ck) >= -1e-12: continue
        n += 1; rb = RB(m)
        if rb > maxRB: maxRB = rb; maxf_pt = (kind, m[np.nonzero(m)], c, lam_min(c), rb)
        fails.append((rb, kind, lam_min(c), (c >= CT - 1e-15).all()))
    print('   failure points', n, '; max RB on them %.5f (M_tree %.5f); number with RB > M_tree:' % (maxRB, MTREE), sum(1 for f in fails if f[0] > MTREE))
    print('   kinds of RB>M_tree points:', sorted(set(f[1] for f in fails if f[0] > MTREE)), '; of those, dominated (C >= C_T):', sum(1 for f in fails if f[0] > MTREE and f[3]))
    print('   max-RB point:', maxf_pt)
