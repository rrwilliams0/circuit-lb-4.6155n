"""bk14_bound.py: the separable kill bound (joint frame, sum m = 1).  NUMERICAL prototype.
scal_u(m_u) / scal_v(m_v): the u- and v-side scalars.  pieces(su, sv): the bound on |D(W)| term by term, each
term split into its u-share and v-share (hinges with thresholds TH and sup constants SUP)."""
import numpy as np, math
from bk14_sides import *
def scal_u(mu_):
    U = uside(mu_); m, a, P, P0, ph = U['m'], U['a'], U['P'], U['P0'], U['phip']
    c1, cm, c0 = KU == 1, KU == -1, KU == 0
    Ph = P - P0                                     # P-hat for class -1
    w11 = max(abs(wedge(P[i], P[j])) for i in np.where(c1)[0] for j in np.where(c1)[0])
    w10 = max(abs(wedge(P[i], P[j])) for i in np.where(c1)[0] for j in np.where(c0)[0])
    pq = np.array([max(abs(wedge(P[i], Ph[j])) for i in np.where(c1)[0]) for j in range(len(m))])   # used on class -1
    Ru = sum(m[i] * m[j] * abs(wedge(Ph[i], Ph[j])) for i in np.where(cm)[0] for j in np.where(cm)[0])
    R0u = sum(m[i] * m[j] * wedge(P[i], P[j]) ** 2 for i in np.where(c0)[0] for j in np.where(c0)[0])
    R0m = sum(m[i] * m[j] * abs(wedge(Ph[i], P[j])) for i in np.where(cm)[0] for j in np.where(c0)[0])
    return dict(phi=U['phi'], X=U['Xu'], Abar=np.abs(a[c1]).max(), K=(m[cm] * np.abs(a[cm])).sum(), A0=(m[c0] * a[c0] ** 2).sum(),
                Kp=(m[cm] * np.abs(ph[cm] - 1)).sum(), Phib=ph[c1].max(), Phi0=(m[c0] * ph[c0] ** 2).sum(), V=U['V'],
                w11=w11, w10=w10, Yu=(m[cm] * pq[cm]).sum(), Ru=Ru, R0=R0u, Rm0=R0m)
def scal_v(mv_):
    Vv = vside(mv_); m, b, Q, Q1, ps = Vv['m'], Vv['b'], Vv['Q'], Vv['Q1'], Vv['psip']
    c1, cm, c0 = KV == 1, KV == -1, KV == 0
    Qh = Q - Q1
    w11 = max(abs(wedge(Q[i], Q[j])) for i in np.where(cm)[0] for j in np.where(cm)[0])
    w10 = max(abs(wedge(Q[i], Q[j])) for i in np.where(cm)[0] for j in np.where(c0)[0])
    pq = np.array([max(abs(wedge(Qh[j], Q[i])) for i in np.where(cm)[0]) for j in range(len(m))])
    Rv = sum(m[i] * m[j] * abs(wedge(Qh[i], Qh[j])) for i in np.where(c1)[0] for j in np.where(c1)[0])
    R0v = sum(m[i] * m[j] * wedge(Q[i], Q[j]) ** 2 for i in np.where(c0)[0] for j in np.where(c0)[0])
    R0m = sum(m[i] * m[j] * abs(wedge(Qh[i], Q[j])) for i in np.where(c1)[0] for j in np.where(c0)[0])
    return dict(phi=Vv['psi'], X=-Vv['Xv'], Abar=np.abs(b[cm]).max(), K=(m[c1] * np.abs(b[c1])).sum(), A0=(m[c0] * b[c0] ** 2).sum(),
                Kp=(m[c1] * np.abs(ps[c1] - 1)).sum(), Phib=ps[cm].max(), Phi0=(m[c0] * ps[c0] ** 2).sum(), V=Vv['V'],
                w11=w11, w10=w10, Yu=(m[c1] * pq[c1]).sum(), Ru=Rv, R0=R0v, Rm0=R0m)
# NOTE: scal_v returns the v-side quantities under the u-side names (the mirror dictionary):
#   phi<->psi, X <-> -X_v, Abar <-> Bbar(class -1), K <-> L (class 1), A0 <-> B0, Kp <-> L', Phib <-> Psibar, Phi0 <-> Psi0,
#   w11 <-> omega^v_{-1}, w10, Yu <-> Yv, Ru <-> Rv, R0, Rm0.
def joint_terms(su, sv):
    """the bound on |D(W)| as a sum of products, before splitting (for validity checks)."""
    V = su['V']; th0 = 1 / math.sqrt(2 * V)
    T = {}
    T['F'] = abs(su['X'] * sv['X'])
    J = su['Abar'] * sv['K'] + sv['Abar'] * su['K'] + 0.5 * (th0 * su['A0'] + sv['A0'] / th0)
    T['J'] = (su['phi'] + sv['phi']) * J
    Mu = su['Phib'] * sv['K'] + sv['Abar'] * su['Kp'] + 0.5 * (th0 * su['Phi0'] + sv['A0'] / th0)
    Mv = sv['Phib'] * su['K'] + su['Abar'] * sv['Kp'] + 0.5 * (th0 * su['A0'] + sv['Phi0'] / th0)
    T['M'] = abs(su['X']) * Mu + abs(sv['X']) * Mv
    E = 0.5 * su['w11'] * sv['Ru'] + 0.5 * sv['w11'] * su['Ru'] + 0.5 * math.sqrt(su['R0'] * sv['R0']) \
        + su['Yu'] * sv['Yu'] + su['w10'] * sv['Rm0'] + sv['w10'] * su['Rm0']
    T['E'] = E
    return T
