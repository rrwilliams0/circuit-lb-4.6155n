"""bk11_core.py : own float implementation of the 123-type cone, C(m), the cross determinant D, the kill delta,
the exact delta* (rank-one dominated witness with the ellipse (PSD) constraint), key vectors, and the slack.
NUMERICAL (floating point).  Conventions as proof.md: vertices u v a b c d = 0..5;
keys (ua, ub, vc, vd) = 0..3; C entries in the order (ab, cd, ac, ad, bc, bd)."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import itertools, math
import numpy as np

EDG = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
E4 = [(0, 2), (0, 3), (1, 4), (1, 5)]
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
KAPPA = 4.478385650326709971336669   # the 40-digit value (CERTIFIED interval [4.478385650326169, 4.478385650327289])
P_T = (1 + 2 * math.sqrt(2)) / 4
S_T = (2 + math.sqrt(2)) / 4
AVEC = np.array([KAPPA, KAPPA, 1, 1, 1, 1])
K0 = 2 * KAPPA * P_T + 4 * S_T
CT = np.array([P_T, P_T, S_T, S_T, S_T, S_T])


def enum_types():
    out = []
    for t in itertools.product(range(-3, 4), repeat=5):
        t = (0,) + t
        if any(abs(t[x] - t[y]) > 1 for x, y in EDG): continue
        if min(t[1], t[2], t[3]) == -1 and min(t[0], t[4], t[5]) == t[1] - 1:
            out.append(t)
    for tc in (1, 2):
        for td in (1, 2): out.append((0, 1, 1, 1, tc, td))
    for ta in (0, 1):
        for tb in (0, 1): out.append((0, -1, ta, tb, 0, 0))
    assert len(set(out)) == len(out)
    return out


TYPES = enum_types(); NT = len(TYPES); assert NT == 123
TIDX = {t: i for i, t in enumerate(TYPES)}
GEN = np.array([[2.0 ** (-x / 2) for x in t] for t in TYPES])
B = np.zeros((4, 6))
for k, (x, y) in enumerate(E4): B[k, x] = B[k, y] = 1.0
TREE = {(0, 1, -1, 1, 2, 2): 1 / 3, (0, 1, 1, -1, 2, 2): 1 / 3, (0, -1, 1, 1, -2, 0): 1 / 6, (0, -1, 1, 1, 0, -2): 1 / 6}
TREE_IDX = [TIDX[t] for t in TREE]
OFF_IDX = [i for i in range(NT) if i not in TREE_IDX]
TA, TB, TC, TD = TIDX[(0, 1, -1, 1, 2, 2)], TIDX[(0, 1, 1, -1, 2, 2)], TIDX[(0, -1, 1, 1, -2, 0)], TIDX[(0, -1, 1, 1, 0, -2)]


def mtree():
    m = np.zeros(NT)
    for t, w in TREE.items(): m[TIDX[t]] = w
    return m


def Gmat(m): return (GEN.T * m) @ GEN


def corr(K):
    d = np.sqrt(np.diag(K)); return K / np.outer(d, d)


def Wmat(m): return B @ corr(Gmat(m)) @ B.T


def Cmat(m): return corr(Wmat(m))


def cvec(C): return np.array([C[p] for p in PAIRS])


def Cv(m): return cvec(Cmat(m))


def Dcross(c): return c[2] * c[5] - c[3] * c[4]       # ac*bd - ad*bc


def kill(c):
    """the single-entry kill: returns (c', delta).  Lower the larger partner's complement so that D becomes 0."""
    c = c.copy(); D = Dcross(c)
    if D == 0: return c, 0.0
    if D > 0:
        if c[5] >= c[2]:            # partner bd is larger: lower ac by D/bd
            d = D / c[5]; c[2] -= d
        else:
            d = D / c[2]; c[5] -= d
    else:
        D = -D
        if c[4] >= c[3]:            # partner bc is larger: lower ad by |D|/bc
            d = D / c[4]; c[3] -= d
        else:
            d = D / c[3]; c[4] -= d
    return c, d


def full(c):
    C = np.eye(4)
    for q, (i, j) in enumerate(PAIRS): C[i, j] = C[j, i] = c[q]
    return C


def lam_min(c): return np.linalg.eigvalsh(full(c))[0]


def ellipse_q(x, al):
    """al^T P^{-1} al for P = [[1, x],[x, 1]]."""
    return (al[0] ** 2 - 2 * x * al[0] * al[1] + al[1] ** 2) / (1 - x * x)


def rank1_max(c, ngrid=1201, nth=300):
    """max of sum(R) over R = alpha gamma^T >= 0 with R <= K entrywise and the PSD (ellipse) constraint
    q_u(alpha) q_v(gamma) <= 1 (pair blocks of c kept).  Returns (best sum, beta, (alpha, gamma))."""
    x, y = c[0], c[1]
    Kac, Kad, Kbc, Kbd = c[2], c[3], c[4], c[5]
    best = -1.0; bestinfo = None
    betas = np.exp(np.linspace(math.log(0.02), math.log(50), ngrid))
    betas = np.concatenate([betas, [Kbc / Kac, Kbd / Kad]])
    L = np.linalg.cholesky(np.array([[1, y], [y, 1]]))
    ths = np.linspace(0, math.pi / 2, nth)
    for beta in betas:
        al = np.array([1.0, beta]); qu = ellipse_q(x, al)
        gcm = min(Kac, Kbc / beta); gdm = min(Kad, Kbd / beta)
        cap = 1.0 / qu
        g = np.array([gcm, gdm]); val = 0.0; garg = g
        if ellipse_q(y, g) <= cap:
            val = gcm + gdm
        else:
            for th in ths:
                gg = math.sqrt(cap) * (L @ np.array([math.cos(th), math.sin(th)]))
                gg = np.minimum(gg, [gcm, gdm])
                if ellipse_q(y, gg) <= cap * (1 + 1e-12) and gg[0] + gg[1] > val:
                    val = gg[0] + gg[1]; garg = gg
            for fix in (0, 1):
                gfix = [gcm, gdm][fix]
                disc = (y * gfix) ** 2 - (gfix ** 2 - cap * (1 - y * y))
                if disc >= 0:
                    g1 = min(y * gfix + math.sqrt(disc), [gcm, gdm][1 - fix])
                    gg = np.array([gfix, g1]) if fix == 0 else np.array([g1, gfix])
                    if g1 >= 0 and gfix + g1 > val and ellipse_q(y, gg) <= cap * (1 + 1e-12):
                        val = gfix + g1; garg = gg
        val *= (1 + beta)
        if val > best:
            best = val; bestinfo = (beta, al.copy(), garg.copy())
    return best, bestinfo


def delta_star(c):
    best, info = rank1_max(c)
    return c[2:].sum() - best, info


def slack(m, exact=False):
    """a.C(m) - K0 - delta*(m) and components; delta* = delta where the kill is PSD (Lemma MIN) unless exact."""
    c = Cv(m); ck, d = kill(c)
    psd = lam_min(ck) >= -1e-12
    if psd and not exact:
        ds = d
    else:
        ds, _ = delta_star(c)
    return AVEC @ c - K0 - ds, dict(c=c, D=Dcross(c), delta=d, killpsd=psd, dstar=ds, cut=AVEC @ c - K0, lmin=lam_min(c))


# ----- key vectors
def keys(m):
    """the five unit keys y_ua, y_ub, y_vc, y_vd, y_uv as rows (in R^123); Gram of the first four = C(m)."""
    R = (np.sqrt(m)[:, None] * GEN).T
    A = R / np.linalg.norm(R, axis=1, keepdims=True)
    ys = []
    for (x, y) in E4 + [(0, 1)]:
        v = A[x] + A[y]; ys.append(v / np.linalg.norm(v))
    return np.array(ys)


def offmass(m): return m[OFF_IDX].sum() / m.sum()


def imbalance(m):
    mm = m / m.sum()
    return abs(mm[TA] - mm[TB]), abs(mm[TC] - mm[TD])


if __name__ == '__main__':
    mT = mtree(); c = Cv(mT)
    print('C_T', c, 'vs', CT, 'max err', np.abs(c - CT).max())
    print('K0', K0, 'a.C_T', AVEC @ c)
    rng = np.random.default_rng(1)
    for _ in range(3):
        m = rng.dirichlet(0.3 * np.ones(NT)); ys = keys(m); Yu = ys[0] + ys[1]; Yv = ys[2] + ys[3]
        cc = Cv(m)
        lhs = AVEC @ cc + 2 * KAPPA; rhs = KAPPA / 2 * (Yu @ Yu + Yv @ Yv) + Yu @ Yv
        print('D1', lhs, rhs, 'Gram check', np.abs(ys[:4] @ ys[:4].T - full(cc)).max())
    for _ in range(5):
        m = rng.dirichlet(0.3 * np.ones(NT)); cc = Cv(m); ck, d = kill(cc); ds, b = delta_star(cc)
        print('D3 delta %.6e delta* %.6e killPSD lmin %.3e D %.3e beta %.4f' % (d, ds, lam_min(ck), Dcross(cc), b[0]))
    # a failure-set point: (e_A + e_C)/2
    m = np.zeros(NT); m[TA] = 0.5; m[TC] = 0.5
    s, info = slack(m, exact=True)
    print('(eA+eC)/2: slack', s, info)
