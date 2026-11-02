"""bk14_sides.py: u-side and v-side quantities of the kill decomposition, from the star masses (joint frame, any scale).
u-profiles PU = (t_v, t_a, t_b), v-profiles PV = (t_v, t_c, t_d) (bk12_star order).  NUMERICAL helper."""
import numpy as np, math
import itertools
S2 = math.sqrt(2)
def wedge(p, q): return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
from bk11_core import TYPES
PU = sorted(set((t[1], t[2], t[3]) for t in TYPES)); PV = sorted(set((t[1], t[4], t[5]) for t in TYPES))
TU = np.array(PU, float); TVv = np.array(PV, float)
GU = 2.0 ** (-TU / 2); GVv = 2.0 ** (-TVv / 2)     # columns (v, a, b) and (v, c, d)
KU = TU[:, 0].astype(int); KV = TVv[:, 0].astype(int)
I_PI0 = PU.index((-1, 1, 1)); I_E = PV.index((1, 2, 2))
FACE_U = [PU.index(p) for p in ((-1, 1, 1), (1, -1, 1), (1, 1, -1))]; FACE_V = [PV.index(p) for p in ((1, 2, 2), (-1, -2, 0), (-1, 0, -2))]
def uside(mu_):
    m = mu_ / mu_.sum()
    V = m @ GU[:, 0] ** 2; na = (m @ GU[:, 1] ** 2) ** -0.5; nb = (m @ GU[:, 2] ** 2) ** -0.5
    P = np.stack([1 + na * GU[:, 1], 1 + nb * GU[:, 2]], 1); P0 = P[I_PI0]
    a = wedge(P0[None, :], P); phi_p = (P[:, 0] + P[:, 1]) / (P0[0] + P0[1])
    c1 = KU == 1; F = (P[c1].T * m[c1]).sum(1)
    return dict(m=m, V=V, na=na, nb=nb, nv=V ** -0.5, P=P, P0=P0, a=a, phip=phi_p, Xu=wedge(F, P0), phi=(m[c1] * phi_p[c1]).sum())
def vside(mv_):
    m = mv_ / mv_.sum()
    V = m @ GVv[:, 0] ** 2; nv = V ** -0.5; nc = (m @ GVv[:, 1] ** 2) ** -0.5; nd = (m @ GVv[:, 2] ** 2) ** -0.5
    Q = np.stack([nv * GVv[:, 0] + nc * GVv[:, 1], nv * GVv[:, 0] + nd * GVv[:, 2]], 1); Q1 = Q[I_E]
    b = wedge(Q1[None, :], Q); psi_p = (Q[:, 0] + Q[:, 1]) / (Q1[0] + Q1[1])
    cm = KV == -1; G = (Q[cm].T * m[cm]).sum(1)
    return dict(m=m, V=V, nc=nc, nd=nd, nv=nv, Q=Q, Q1=Q1, b=b, psip=psi_p, Xv=wedge(Q1, G), psi=(m[cm] * psi_p[cm]).sum())
