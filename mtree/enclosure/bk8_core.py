"""bk8_core.py : the rank-one-cross family R1 in the 5 coordinates theta = (x, y, S, tu, tv):
   C_ab = x, C_cd = y, cross block K = alpha gamma^T with alpha = (S/2 + tu, S/2 - tu), gamma = (S/2 + tv, S/2 - tv),
   so that alpha_a + alpha_b = gamma_c + gamma_d = S and sum of the four cross entries = S^2.
Interval constants: kappa = g_ab/g_ac at C_T, c = -g_ac, K0 = a.C_T, lower bound of M_tree; float helpers."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, time
import numpy as np
from ivarith import IA, dn, up, PI, asin
import phifun as AP

PAIRS = AP.PAIRS   # ab cd ac ad bc bd

# ---------------------------------------------------------------- theta -> C (works for floats and IA scalars)
def C_of(th):
    x, y, S, tu, tv = th
    aa = S*0.5 + tu; ab_ = S*0.5 - tu; gc = S*0.5 + tv; gd = S*0.5 - tv
    return [x, y, aa*gc, aa*gd, ab_*gc, ab_*gd]

def J_of(th):
    """6x5 Jacobian dC_p/dtheta_j (entries floats or IA)."""
    x, y, S, tu, tv = th
    aa = S*0.5 + tu; ab_ = S*0.5 - tu; gc = S*0.5 + tv; gd = S*0.5 - tv
    z = 0.0
    return [[1.0, z, z, z, z],
            [z, 1.0, z, z, z],
            [z, z, (gc + aa)*0.5, gc, aa],
            [z, z, (gd + aa)*0.5, gd, -aa],
            [z, z, (gc + ab_)*0.5, -gc, ab_],
            [z, z, (gd + ab_)*0.5, -gd, -ab_]]

# constant Hessians d^2 C_p / dtheta^2 (5x5 floats, exact dyadics)
def _H(sS, stu, stv, stt):
    M = np.zeros((5, 5)); M[2, 2] = 0.5
    M[2, 3] = M[3, 2] = stu*0.5; M[2, 4] = M[4, 2] = stv*0.5; M[3, 4] = M[4, 3] = stt
    return M
HESS_C = [np.zeros((5, 5)), np.zeros((5, 5)), _H(1, 1, 1, 1), _H(1, 1, -1, -1), _H(1, -1, 1, -1), _H(1, -1, -1, 1)]

def q_of(dl):
    """quadratic part: C(th0 + dl) - C(th0) - J0 dl = 1/2 dl^T HESS_C dl (exact identity); dl floats or IA list."""
    out = []
    for Hm in HESS_C:
        acc = IA(0.0) if isinstance(dl[0], IA) else 0.0
        for i in range(5):
            for j in range(5):
                if Hm[i, j] != 0.0: acc = acc + dl[i]*dl[j]*float(Hm[i, j])
        out.append(acc*0.5)
    return out

# ---------------------------------------------------------------- constants
S2 = IA(2.0).sqrt()
P_T = (1.0 + S2*2.0)*0.25          # (1 + 2 sqrt2)/4
S_T = (2.0 + S2)*0.25              # (2 + sqrt2)/4
SS_T = S_T.sqrt()*2.0              # S at the tree = 2 sqrt(s)

def cT_box():
    lo = [P_T.lo, P_T.lo] + [S_T.lo]*4; hi = [P_T.hi, P_T.hi] + [S_T.hi]*4
    return np.array(lo), np.array(hi)

_CONST = {}
def consts():
    """kappa, c, K0 (= a.C_T), M_tree lower/upper bounds, all IA scalars; cached."""
    if _CONST: return _CONST
    lo, hi = cT_box()
    g = AP.grad_ad(lo, hi)              # gradient enclosure at C_T (box of width ~1 ulp)
    gv = g.v
    g_ab = IA(gv.lo[0], gv.hi[0]); g_cd = IA(gv.lo[1], gv.hi[1]); g_ac = IA(gv.lo[2], gv.hi[2])
    kappa = g_ab*g_ac.inv()             # both negative -> positive ratio
    c = -g_ac
    K0 = kappa*P_T*2.0 + S_T*4.0
    # M_tree lower bound: Phi_F(C_T) >= Phi_F(c_up) >= phi_point(c_up).lo  (monotone; c_up >= C_T entrywise)
    cup = list(hi); assert AP.pd_point(cup)
    e_up = AP.phi_point(cup)
    cdn = list(lo); assert AP.pd_point(cdn)
    e_dn = AP.phi_point(cdn)
    _CONST.update(kappa=kappa, c=c, K0=K0, Mt_lo=float(e_up.lo), Mt_hi=float(e_dn.hi), g_T=gv, H_T=g.g,
                  g_ab=g_ab, g_cd=g_cd, g_ac=g_ac)
    return _CONST

def cut_h(th, k=None):
    """h(theta) = a.C(theta) - a.C_T = kappa (x + y - 2p) + (S^2 - 4 s); floats or IA."""
    K = consts(); kap = K['kappa']
    x, y, S, tu, tv = th
    if isinstance(x, IA) or isinstance(S, IA):
        return kap*(x + y - P_T*2.0) + (S*S - S_T*4.0)
    return float(kap.mid())*(x + y - 2*float(P_T.mid())) + (S*S - 4*float(S_T.mid()))

def theta_T():
    return [float(P_T.mid()), float(P_T.mid()), float(SS_T.mid()), 0.0, 0.0]

# ---------------------------------------------------------------- float Phi_F (for exploration only)
import bk7_core as B7
def phiF_float(c6):
    C = np.eye(4)
    for q, (i, j) in enumerate(PAIRS): C[i, j] = C[j, i] = c6[q]
    return B7.phiF(C)

def Cmat(c6):
    C = np.eye(4)
    for q, (i, j) in enumerate(PAIRS): C[i, j] = C[j, i] = c6[q]
    return C

if __name__ == '__main__':
    t0 = time.time(); K = consts(); t1 = time.time()
    print('consts in %.2fs' % (t1 - t0))
    for k in ('kappa', 'c', 'K0'): print(k, K[k].lo, K[k].hi)
    print('M_tree in [%.17g, %.17g]' % (K['Mt_lo'], K['Mt_hi']))
    thT = theta_T(); cT = C_of(thT); print('theta_T', thT, ' C(theta_T)', cT)
    print('cut_h(theta_T) =', cut_h(thT))
    # timings
    lo, hi = cT_box(); t0 = time.time(); e = AP.phi_point(list(hi)); print('phi_point: %.3fs width %.2e' % (time.time() - t0, e.hi - e.lo))
    for w in (1e-4, 1e-3, 3e-3):
        t0 = time.time(); g = AP.grad_ad(lo - w, hi + w); dt = time.time() - t0
        Hw = (g.g.hi - g.g.lo).max(); gw = (g.v.hi - g.v.lo).max()
        print('grad_ad half-width %g: %.3fs, grad width %.2e, Hess width %.2e' % (w, dt, gw, Hw))
    # J^T H J at the tree (floats): eigenvalues
    J = np.array(J_of(thT), float); H = 0.5*(K['H_T'].mid() + K['H_T'].mid().T)
    gT = K['g_T'].mid()
    M1 = sum(gT[p]*HESS_C[p] for p in range(6))
    Mloc = 2*float(K['c'].mid())*np.outer(np.eye(5)[2], np.eye(5)[2]) + J.T @ H @ J + M1
    print('eig H_T:', np.round(np.linalg.eigvalsh(H), 5))
    print('eig J^T H J:', np.round(np.linalg.eigvalsh(J.T @ H @ J), 5))
    print('eig M_loc(tree):', np.round(np.linalg.eigvalsh(Mloc), 5))
    print('max |M_loc - J^T H J|:', np.abs(Mloc - J.T @ H @ J).max())
