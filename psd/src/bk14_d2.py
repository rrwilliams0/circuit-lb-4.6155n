"""bk14_d2.py: Delta_u version 2 (the mixed terms split by AM-GM instead of hinges).  Uses the fast scalars (bk14_fast)
and the CERTIFIED profile maxima (bk14_delta.PM).  delta_u(s, mu) in |D(W)| units; s = scal(star masses)."""
import numpy as np, math
from bk14_fast import scal, S2, R22
import bk14_delta as BD
class Kill2:
    def __init__(s, eps=0.05, lamM=1.0):
        s.eps, s.lamM, s.TH = eps, lamM, {}
    def thr(s, mu):
        key = tuple(np.round(mu, 13))
        if key not in s.TH:
            f = scal(BD.face_star(mu)); e = 1 + s.eps
            s.TH[key] = dict(U=e * f['phi'] * f['Abar'], A=e * f['Abar'], phi=e * f['phi'], w11=e * f['w11'], w10=e * f['w10'])
        return s.TH[key]
    def terms(s, sc, mu):
        ms = BD.mirror_mu(mu); V = BD.Vof(mu); w = math.sqrt(V); L = s.lamM
        Tp, Ts = s.thr(mu), s.thr(ms); Sp = BD.sups(ms)
        X, K, A0 = sc['X'], sc['K'], sc['A0']; aX = abs(X)
        pos = lambda z: max(z, 0.0)
        return dict(
            face=X * X / 4,
            J_own=Ts['U'] * K + Ts['A'] * sc['phi'] * K / V + (sc['phi'] / w + w * Ts['phi']) * A0 / R22,
            M_amgm=L * sc['Phib'] ** 2 * X * X / 2 + K * K / (2 * L) + L * X * X / (2 * R22 * w) + w * A0 * A0 / (2 * R22 * L),
            M_own=Ts['A'] * aX * sc['Kp'] / V + aX * sc['Phi0'] / (R22 * w),
            E_own=Ts['w11'] * sc['R'] / V + sc['R0'] / 4 + sc['Y'] ** 2 + S2 / V * Ts['w10'] * sc['Rm0'],
            hinge=(pos(sc['phi'] * sc['Abar'] - Tp['U']) * Sp['K'] + pos(sc['Abar'] - Tp['A']) * (Sp['Zphi'] + Sp['ZX'])
                   + pos(sc['phi'] - Tp['phi']) * Sp['A0'] / (R22 * w) + pos(sc['w11'] - Tp['w11']) * Sp['ZR'] + pos(sc['w10'] - Tp['w10']) * Sp['ZRm']))
    def delta_u(s, sc, mu): return sum(s.terms(sc, mu).values())
