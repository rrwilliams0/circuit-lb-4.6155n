"""bk7_core.py : own implementation of the 123-type cone, C(m), Phi_F, its closed-form Plackett gradient,
the analytic Jacobian dC/dm, and high-precision (mpmath) twins.  NUMERICAL (floating point / mpmath), no intervals.

Conventions (proof.md): vertices u v a b c d = 0..5; internal edges uv ua ub vc vd;
boundary keys (ua, ub, vc, vd) = 0..3; C entries in the order (ab, cd, ac, ad, bc, bd).
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import itertools, math
import numpy as np
import mpmath as mp

VX = 'uvabcd'
EDG = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]          # internal edges
E4 = [(0, 2), (0, 3), (1, 4), (1, 5)]                   # boundary edges ua ub vc vd
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]   # ab cd ac ad bc bd (as pairs of keys)
COMP = {(0, 1): (2, 3), (2, 3): (0, 1), (0, 2): (1, 3), (0, 3): (1, 2), (1, 2): (0, 3), (1, 3): (0, 2)}
WGT = np.array([0.75, 0.75, 0.25, 0.25, 0.25, 0.25])    # Plackett weights w_p (3/4 pair, 1/4 cross)
NAMES = ['ab', 'cd', 'ac', 'ad', 'bc', 'bd']


def enum_types():
    """T123 from the definition: generic (E),(Ru),(Rv) with t_u = 0, plus the 8 special types."""
    out = []
    for t in itertools.product(range(-3, 4), repeat=5):       # wide range; (E) and the rules cut it down
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


TYPES = enum_types()
NT = len(TYPES)
assert NT == 123, NT
TIDX = {t: i for i, t in enumerate(TYPES)}
GEN = np.array([[2.0 ** (-x / 2) for x in t] for t in TYPES])        # generators g_t (rows)
B = np.zeros((4, 6))
for k, (x, y) in enumerate(E4): B[k, x] = B[k, y] = 1.0
TREE = {(0, 1, -1, 1, 2, 2): 1 / 3, (0, 1, 1, -1, 2, 2): 1 / 3, (0, -1, 1, 1, -2, 0): 1 / 6, (0, -1, 1, 1, 0, -2): 1 / 6}
TREE_IDX = [TIDX[t] for t in TREE]


def mtree():
    m = np.zeros(NT)
    for t, w in TREE.items(): m[TIDX[t]] = w
    return m


# ------------------------------------------------------------------ float geometry
def Gmat(m): return (GEN.T * m) @ GEN


def corr(K):
    d = np.sqrt(np.diag(K)); return K / np.outer(d, d)


def Cmat(m): return corr(B @ corr(Gmat(m)) @ B.T)


def cvec(C): return np.array([C[p] for p in PAIRS])


def diag_ratios(m):
    G = Gmat(m); return np.diag(G)[1:] / G[0, 0]


def dC_dm(m):
    """analytic Jacobian: returns J of shape (6, NT), J[p, t] = dC_p/dm_t (C in the PAIRS order)."""
    G = Gmat(m); dg = np.diag(G); sd = np.sqrt(dg)
    Gh = G / np.outer(sd, sd)
    W = B @ Gh @ B.T; dw = np.diag(W); sw = np.sqrt(dw)
    C = W / np.outer(sw, sw)
    J = np.zeros((6, NT))
    for t in range(NT):
        P = np.outer(GEN[t], GEN[t])                      # dG/dm_t
        dP = np.diag(P)
        dGh = P / np.outer(sd, sd) - 0.5 * Gh * (dP[:, None] / dg[:, None] + dP[None, :] / dg[None, :])
        dW = B @ dGh @ B.T; ddW = np.diag(dW)
        dC = dW / np.outer(sw, sw) - 0.5 * C * (ddW[:, None] / dw[:, None] + ddW[None, :] / dw[None, :])
        J[:, t] = [dC[p] for p in PAIRS]
    return J


# ------------------------------------------------------------------ Phi_F (float): Walsh form + Plackett path for Q
def partial_corr(C, i, j, k, l):
    """correlation of (Z_k, Z_l) given (Z_i, Z_j) under C (4x4)."""
    S = C[np.ix_([i, j], [i, j])]; Sinv = np.linalg.inv(S)
    A = C[np.ix_([k, l], [i, j])]
    R = C[np.ix_([k, l], [k, l])] - A @ Sinv @ A.T
    return R[0, 1] / math.sqrt(R[0, 0] * R[1, 1])


def grad_phi(C):
    """closed-form gradient of Phi_F in the six entries (Section 2.5 / phifun.py):
       dPhi/dr_p = -(w_p - asin(rho_p)/(2 pi))/(pi sqrt(1 - r_p^2)), rho_p = partial corr of the complementary pair given p."""
    g = np.zeros(6)
    for q, (i, j) in enumerate(PAIRS):
        k, l = COMP[(i, j)]
        r = C[i, j]; rho = partial_corr(C, i, j, k, l)
        g[q] = -(WGT[q] - math.asin(rho) / (2 * math.pi)) / (math.pi * math.sqrt(1 - r * r))
    return g


_XS, _XW = np.polynomial.legendre.leggauss(160); _XS = 0.5 * (_XS + 1); _XW = 0.5 * _XW


def Q4(C):
    """P[all four > 0] by the Plackett path C(s) = I + s (C - I), dQ/dC_ij = phi_2(0,0;C_ij) (1/4 + asin(rho_kl|ij)/(2 pi))."""
    tot = 0.0
    for s, w in zip(_XS, _XW):
        Cs = np.eye(4) + s * (C - np.eye(4))
        acc = 0.0
        for (i, j) in PAIRS:
            k, l = COMP[(i, j)]
            r = Cs[i, j]; rho = partial_corr(Cs, i, j, k, l)
            acc += C[i, j] * (1.0 / (2 * math.pi * math.sqrt(1 - r * r))) * (0.25 + math.asin(rho) / (2 * math.pi))
        tot += w * acc
    return 1.0 / 16 + tot


def phiF(C):
    s = lambda i, j: (2 / math.pi) * math.asin(C[i, j])
    return 1 - (s(0, 1) + s(2, 3)) / 2 - (s(0, 2) + s(0, 3) + s(1, 2) + s(1, 3)) / 4 + 2 * Q4(C)


def f(m): return phiF(Cmat(m))


def grad_f(m):
    """df/dm_t for all t (closed form: chain rule with grad_phi and dC_dm)."""
    return grad_phi(Cmat(m)) @ dC_dm(m)


# ------------------------------------------------------------------ mpmath twins (high precision)
def mp_setprec(d): mp.mp.dps = d


def mp_G(m):
    G = mp.zeros(6, 6)
    for t in range(NT):
        if m[t] == 0: continue
        g = [mp.power(2, mp.mpf(-x) / 2) for x in TYPES[t]]
        for x in range(6):
            for y in range(6): G[x, y] += m[t] * g[x] * g[y]
    return G


def mp_corr(K):
    n = K.rows; d = [mp.sqrt(K[i, i]) for i in range(n)]
    R = mp.zeros(n, n)
    for i in range(n):
        for j in range(n): R[i, j] = K[i, j] / (d[i] * d[j])
    return R


def mp_C(m):
    Gh = mp_corr(mp_G(m)); Bm = mp.matrix(B.tolist())
    return mp_corr(Bm * Gh * Bm.T)


def mp_partial(C, i, j, k, l):
    S = mp.matrix([[C[i, i], C[i, j]], [C[j, i], C[j, j]]]); Si = S ** -1
    A = mp.matrix([[C[k, i], C[k, j]], [C[l, i], C[l, j]]])
    R = mp.matrix([[C[k, k], C[k, l]], [C[l, k], C[l, l]]]) - A * Si * A.T
    return R[0, 1] / mp.sqrt(R[0, 0] * R[1, 1])


def mp_grad_phi(C):
    g = []
    for q, (i, j) in enumerate(PAIRS):
        k, l = COMP[(i, j)]
        r = C[i, j]; rho = mp_partial(C, i, j, k, l)
        w = mp.mpf(3) / 4 if q < 2 else mp.mpf(1) / 4
        g.append(-(w - mp.asin(rho) / (2 * mp.pi)) / (mp.pi * mp.sqrt(1 - r * r)))
    return g


def mp_Q4(C):
    I4 = mp.eye(4)

    def integrand(s):
        Cs = I4 + s * (C - I4); acc = mp.mpf(0)
        for (i, j) in PAIRS:
            k, l = COMP[(i, j)]
            r = Cs[i, j]; rho = mp_partial(Cs, i, j, k, l)
            acc += C[i, j] / (2 * mp.pi * mp.sqrt(1 - r * r)) * (mp.mpf(1) / 4 + mp.asin(rho) / (2 * mp.pi))
        return acc
    return mp.mpf(1) / 16 + mp.quad(integrand, [0, 0.5, 0.9, 1])


def mp_phiF(C):
    s = lambda i, j: 2 / mp.pi * mp.asin(C[i, j])
    return 1 - (s(0, 1) + s(2, 3)) / 2 - (s(0, 2) + s(0, 3) + s(1, 2) + s(1, 3)) / 4 + 2 * mp_Q4(C)


def mp_cvec(C): return [C[p] for p in PAIRS]


def mp_f(m): return mp_phiF(mp_C(m))


def mp_grad_f(m, h=None):
    """df/dm_t at mp precision: closed-form grad_phi times a central-difference Jacobian of C (step h, exact to O(h^2))."""
    if h is None: h = mp.mpf(10) ** (-(mp.mp.dps // 3))
    C0 = mp_C(m); g = mp_grad_phi(C0); out = []
    for t in range(NT):
        mp_ = list(m); mm_ = list(m); mp_[t] += h; mm_[t] -= h
        cp = mp_cvec(mp_C(mp_)); cm = mp_cvec(mp_C(mm_))
        out.append(sum(g[q] * (cp[q] - cm[q]) / (2 * h) for q in range(6)))
    return out


if __name__ == '__main__':
    import gc4_lib
    assert set(TYPES) == set(gc4_lib.TYPES), 'type sets differ from the reference'
    print('types: 123, identical set to the reference lib.py')
    m0 = mtree(); C0 = Cmat(m0)
    p = (1 + 2 * math.sqrt(2)) / 4; s_ = (2 + math.sqrt(2)) / 4
    print('C_T entries', cvec(C0), ' expected p,s =', p, s_)
    print('Ghat tree', corr(Gmat(m0))[0, 1], 2 * math.sqrt(2) / 3, corr(Gmat(m0))[2, 3], 5 / 6, corr(Gmat(m0))[2, 4], 1 / math.sqrt(2))
    print('Q4(C_T) =', repr(Q4(C0)), ' reference 0.3702816309621984812264')
    print('M_tree  =', repr(phiF(C0)), ' reference 0.2765836068823734710228')
    g = grad_phi(C0); print('grad Phi_F(C_T) =', g, ' ratio', g[0] / g[2], ' (reference: 4.478385650)')
    import gc4_lib as L
    print('cross-check phiF vs scipy Genz at tree:', L.phi_mvn(cvec(C0)))
    mp_setprec(30)
    print('mp M_tree =', mp_phiF(mp_C(m0)))
