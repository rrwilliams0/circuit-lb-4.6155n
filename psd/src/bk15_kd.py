"""independent numerical check of the Lemma KD (own implementation from the definitions in proof.md)."""
import sys, numpy as np
sys.path.insert(0, '.')
from bk11_core import *
rng = np.random.default_rng(15)
T = np.array(TYPES)  # columns u v a b c d
def wedge(p, q): return p[0] * q[1] - p[1] * q[0]
worst = 0; worstT = 0
for trial in range(300):
    kind = trial % 3
    if kind == 0: m = rng.dirichlet(0.3 * np.ones(NT))
    elif kind == 1: m = mtree(); m[rng.choice(OFF_IDX, 3)] += rng.uniform(0, 0.05, 3); m /= m.sum()
    else: m = np.zeros(NT); m[rng.choice(NT, 4, replace=False)] = rng.dirichlet(np.ones(4))
    G = Gmat(m); n = 1 / np.sqrt(np.diag(G))      # n_u = 1
    g = GEN                                        # g_t(x) = 2^{-t_x/2}
    P = np.stack([1 + n[2] * g[:, 2], 1 + n[3] * g[:, 3]], 1); Q = np.stack([n[1] * g[:, 1] + n[4] * g[:, 4], n[1] * g[:, 1] + n[5] * g[:, 5]], 1)
    W = (P * m[:, None]).T @ Q; DW = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
    i0 = [i for i, t in enumerate(TYPES) if tuple(t[1:4]) == (-1, 1, 1)][0]; i1 = [i for i, t in enumerate(TYPES) if (t[1], t[4], t[5]) == (1, 2, 2)][0]
    P0, Q1 = P[i0], Q[i1]; nu = np.array([-1, 1]) / (P0[0] + P0[1]); nup = np.array([-1, 1]) / (Q1[0] + Q1[1])
    tv = T[:, 1]; c1, cm, c0 = tv == 1, tv == -1, tv == 0
    a = np.array([wedge(P0, p) for p in P]); b = np.array([wedge(Q1, q) for q in Q])
    F = (m[c1, None] * P[c1]).sum(0); Gv = (m[cm, None] * Q[cm]).sum(0)
    Xu = wedge(F, P0); Xv = wedge(Q1, Gv); phi = wedge(F, nu); psi = wedge(Gv, nup)
    Pt = P.copy(); Pt[cm] -= P0; Qt = Q.copy(); Qt[c1] -= Q1
    J = (m * a * b).sum(); Mu = (m * np.array([wedge(p, nu) for p in Pt]) * b).sum(); Mv = (m * a * np.array([wedge(q, nup) for q in Qt])).sum()
    E = (Pt * m[:, None]).T @ Qt; detE = E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]
    rhs = Xu * Xv + (phi + psi) * J + Xu * Mu - Xv * Mv + detE
    worst = max(worst, abs(DW - rhs)); worstT = max(worstT, abs(DW - Dcross(Cv(m)) * np.prod([np.linalg.norm(x) for x in [0]]) ) if False else worstT)
    # also check D(W) against D(C) times the four key norms
    Wm = Wmat(m); Nn = np.sqrt(np.diag(Wm))[[0, 1, 2, 3]]  # keys ua ub vc vd
    worstT = max(worstT, abs(DW - Dcross(Cv(m)) * Nn[0] * Nn[1] * Nn[2] * Nn[3]))
print('Lemma KD identity: max |D(W) - RHS| over 300 points = %.2e;  max |D(W) - D(C) N_a N_b N_c N_d| = %.2e' % (worst, worstT))
