"""bk14_dec.py: the kill decomposition (Lemma KD), NUMERICAL check.
W_xz = sum_t m_t P_x(t) Q_z(t), P = (n_u + n_a g_a, n_u + n_b g_b), Q = (n_v g_v + n_c g_c, n_v g_v + n_d g_d).
D(W) = X_u X_v + (phi + psi) J + X_u M_u - X_v M_v + det E."""
import numpy as np, math
from bk11_core import *
def wedge(p, q): return p[0] * q[1] - p[1] * q[0]
def PQ(m):
    G = Gmat(m); n = 1 / np.sqrt(np.diag(G))
    g = GEN  # columns u v a b c d
    P = np.stack([n[0] * g[:, 0] + n[2] * g[:, 2], n[0] * g[:, 0] + n[3] * g[:, 3]], 1)
    Q = np.stack([n[1] * g[:, 1] + n[4] * g[:, 4], n[1] * g[:, 1] + n[5] * g[:, 5]], 1)
    return P, Q, n
TV = np.array([t[1] for t in TYPES])
def ref_PQ(n):
    # P_0 = P(pi0), pi0 = (-1,1,1) u-profile ; Q_1 = Q(e), e = (1,2,2) v-profile
    g = lambda x: 2.0 ** (-x / 2)
    P0 = np.array([n[0] + n[2] * g(1), n[0] + n[3] * g(1)])
    Q1 = np.array([n[1] * g(1) + n[4] * g(2), n[1] * g(1) + n[5] * g(2)])
    return P0, Q1
def decomp(m):
    P, Q, n = PQ(m); P0, Q1 = ref_PQ(n)
    W = (P.T * m) @ Q
    F = (P[TV == 1].T * m[TV == 1]).sum(1); Gv = (Q[TV == -1].T * m[TV == -1]).sum(1)
    nu = np.array([-1.0, 1.0]) / (P0[0] + P0[1]); nup = np.array([-1.0, 1.0]) / (Q1[0] + Q1[1])
    assert abs(wedge(P0, nu) - 1) < 1e-12 and abs(wedge(Q1, nup) - 1) < 1e-12
    Pt = P.copy(); Qt = Q.copy()
    Pt[TV == -1] -= P0; Qt[TV == 1] -= Q1
    a = np.array([wedge(P0, p) for p in P]); b = np.array([wedge(Q1, q) for q in Q])
    Xu = wedge(F, P0); Xv = wedge(Q1, Gv); phi = wedge(F, nu); psi = wedge(Gv, nup)
    J = (m * a * b).sum(); Mu = (m * np.array([wedge(p, nu) for p in Pt]) * b).sum(); Mv = (m * a * np.array([wedge(q, nup) for q in Qt])).sum()
    E = (Pt.T * m) @ Qt; detE = np.linalg.det(E)
    D = np.linalg.det(W)
    terms = dict(XX=Xu * Xv, J=(phi + psi) * J, XM=Xu * Mu, XMv=-Xv * Mv, dE=detE)
    return D, terms, dict(Xu=Xu, Xv=Xv, phi=phi, psi=psi, Jraw=J, Mu=Mu, Mv=Mv, n=n, W=W)
if __name__ == '__main__':
    rng = np.random.default_rng(3)
    for i in range(5):
        m = rng.dirichlet(0.3 * np.ones(NT))
        D, T, X = decomp(m)
        # check W against bk11 Wmat
        Wb = Wmat(m)
        Wx = np.array([[Wb[0, 2], Wb[0, 3]], [Wb[1, 2], Wb[1, 3]]])
        print('W err %.1e  D %.6e  sum terms %.6e' % (np.abs(Wx - X['W']).max(), D, sum(T.values())))
    D, T, X = decomp(mtree()); print('tree', D, T, X['phi'], X['psi'])
