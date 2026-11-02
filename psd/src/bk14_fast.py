"""bk14_fast.py: vectorised u-star scalars, s_u, and Delta_u (same formulas as bk14_sym / bk14_delta).  NUMERICAL."""
import numpy as np, math
from bk14_sides import PU, KU
S2 = math.sqrt(2); R22 = 2 * S2
TU = np.array(PU, float); GV_ = 2.0 ** (-TU[:, 0] / 2); GA_ = 2.0 ** (-TU[:, 1] / 2); GB_ = 2.0 ** (-TU[:, 2] / 2)
GEN = np.vstack([np.ones(len(PU)), GV_, GA_, GB_]).T
I1, IM, I0 = np.where(KU == 1)[0], np.where(KU == -1)[0], np.where(KU == 0)[0]
IPI0 = PU.index((-1, 1, 1)); IA_, IB_ = PU.index((1, -1, 1)), PU.index((1, 1, -1))
RHO = 2 * S2 / 3; MU0 = (1 + 2 * S2) / 4; KAPPA = 4.478385650326709971336669; RR = (KAPPA - 1) / (4 * MU0)
def W(p, q): return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
def prof_consts(na, nb):
    P = np.stack([1 + na * GA_, 1 + nb * GB_], 1); P0 = P[IPI0]
    a = W(P0[None, :], P); ph = (P[:, 0] + P[:, 1]) / (P0[0] + P0[1]); Ph = P - P0
    w11 = np.abs(W(P[I1][:, None, :], P[I1][None, :, :])).max()
    w10 = np.abs(W(P[I1][:, None, :], P[I0][None, :, :])).max()
    pq = np.abs(W(P[I1][:, None, :], Ph[IM][None, :, :])).max(0)
    WPh = np.abs(W(Ph[IM][:, None, :], Ph[IM][None, :, :]))
    W00 = W(P[I0][:, None, :], P[I0][None, :, :]) ** 2
    Wm0 = np.abs(W(Ph[IM][:, None, :], P[I0][None, :, :]))
    return P, P0, a, ph, w11, w10, pq, WPh, W00, Wm0
def scal(m_):
    m = m_ / m_.sum(); G = (GEN.T * m) @ GEN
    V = G[1, 1]; na = G[2, 2] ** -0.5; nb = G[3, 3] ** -0.5
    P, P0, a, ph, w11, w10, pq, WPh, W00, Wm0 = prof_consts(na, nb)
    F = (P[I1].T * m[I1]).sum(1)
    mm, m0 = m[IM], m[I0]
    return dict(phi=(m[I1] * ph[I1]).sum(), X=W(F, P0), Abar=np.abs(a[I1]).max(), K=(mm * np.abs(a[IM])).sum(), A0=(m0 * a[I0] ** 2).sum(),
                Kp=(mm * np.abs(ph[IM] - 1)).sum(), Phib=ph[I1].max(), Phi0=(m0 * ph[I0] ** 2).sum(), V=V, w11=w11, w10=w10,
                Y=(mm * pq).sum(), R=mm @ WPh @ mm, R0=m0 @ W00 @ m0, Rm0=mm @ Wm0 @ m0, G=G)
def kvals_G(G):
    d = np.sqrt(np.diag(G)); Gh = G / np.outer(d, d)
    e0, eua, eub, hva, hvb, hab = Gh[0, 1], Gh[0, 2], Gh[0, 3], Gh[1, 2], Gh[1, 3], Gh[2, 3]
    kc = lambda e1, e2, h: (1 + e1 + e2 + h) / (2 * math.sqrt((1 + e1) * (1 + e2)))
    return kc(e0, eua, hva), kc(e0, eub, hvb), kc(eua, eub, hab)
def s_u(s):
    ka, kb, kab = kvals_G(s['G']); return RR * kab + ka + kb - (RR + 2) * MU0
