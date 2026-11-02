"""bk14_sym.py: normalised (u-frame) star scalars, the mirror star of a v-star, and the symmetric joint bound
B(m) >= |D(W)| written in u-star and mirror-star scalars.  NUMERICAL prototype."""
import numpy as np, math
from bk14_sides import *
import bk12_star as S
def ustar_scal(m_):
    """scalars of a u-star (masses on PU, any scale; normalised inside)."""
    U = uside(m_); m, a, P, P0, ph = U['m'], U['a'], U['P'], U['P0'], U['phip']
    c1, cm, c0 = KU == 1, KU == -1, KU == 0
    Ph = P - P0
    i1, im, i0 = np.where(c1)[0], np.where(cm)[0], np.where(c0)[0]
    w11 = max(abs(wedge(P[i], P[j])) for i in i1 for j in i1)
    w10 = max(abs(wedge(P[i], P[j])) for i in i1 for j in i0)
    pq = np.array([max(abs(wedge(P[i], Ph[j])) for i in i1) for j in range(len(m))])
    Ru = sum(m[i] * m[j] * abs(wedge(Ph[i], Ph[j])) for i in im for j in im)
    R0 = sum(m[i] * m[j] * wedge(P[i], P[j]) ** 2 for i in i0 for j in i0)
    Rm0 = sum(m[i] * m[j] * abs(wedge(Ph[i], P[j])) for i in im for j in i0)
    return dict(phi=U['phi'], X=U['Xu'], Abar=np.abs(a[c1]).max(), K=(m[cm] * np.abs(a[cm])).sum(), A0=(m[c0] * a[c0] ** 2).sum(),
                Kp=(m[cm] * np.abs(ph[cm] - 1)).sum(), Phib=ph[c1].max(), Phi0=(m[c0] * ph[c0] ** 2).sum(), V=U['V'],
                w11=w11, w10=w10, Y=(m[cm] * pq[cm]).sum(), R=Ru, R0=R0, Rm0=Rm0, mu=np.array([m[KU == k].sum() for k in (-1, 0, 1)]))
# mirror: v-profile (tv, tc, td) -> u-profile (-tv, tc - tv, td - tv), mass factor 2^{-tv}
MIR = np.zeros((len(PU), len(PV)))
for j, (tv, tc, td) in enumerate(PV):
    MIR[PU.index((-tv, tc - tv, td - tv)), j] = 2.0 ** (-tv)
def mirror_star(mv_): return MIR @ mv_
def joint_B(su, ss):
    V = su['V']; w = math.sqrt(V); r22 = 2 * math.sqrt(2)
    T = {}
    T['F'] = abs(su['X'] * ss['X']) / 2
    T['J'] = (su['phi'] * su['Abar'] * ss['K'] + ss['phi'] * ss['Abar'] * su['K'] + V * ss['phi'] * su['Abar'] * ss['K']
              + su['phi'] * ss['Abar'] * su['K'] / V + (su['phi'] / w + w * ss['phi']) * (su['A0'] + ss['A0']) / r22)
    T['M'] = (abs(su['X']) * su['Phib'] * ss['K'] + abs(ss['X']) * ss['Phib'] * su['K'] + abs(su['X']) * ss['Abar'] * su['Kp'] / V
              + V * abs(ss['X']) * su['Abar'] * ss['Kp'] + abs(su['X']) * (su['Phi0'] + ss['A0']) / (r22 * w) + w * abs(ss['X']) * (su['A0'] + ss['Phi0']) / r22)
    T['E'] = (su['w11'] * V * ss['R'] + ss['w11'] * su['R'] / V + (su['R0'] + ss['R0']) / 4 + su['Y'] ** 2 + ss['Y'] ** 2
              + math.sqrt(2) * V * su['w10'] * ss['Rm0'] + math.sqrt(2) / V * ss['w10'] * su['Rm0'])
    return T
