"""bk12_pb.py : the two-dimensional projection bound PB(m) of the Observation PROJ2, computed robustly
(sup over unit w in span(a_u, a_v) of <Y, w>^2, via the Gram matrix; e_0 = 1 handled as a 1-dim span), the exact
identity a.C - K0 = (PB - K0) + |Y - P Y|^2/2, and global minimisation of PB - K0 over the simplex.  NUMERICAL."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import math, sys, numpy as np
from bk11_core import *

RHO = 2 * math.sqrt(2) / 3
C0 = math.sqrt((1 + RHO) / 2)


def rows_keys(m):
    R = (np.sqrt(m)[:, None] * GEN).T
    A = R / np.linalg.norm(R, axis=1, keepdims=True)
    Gh = A @ A.T
    ys = []
    for (x, y) in E4 + [(0, 1)]:
        v = A[x] + A[y]; ys.append(v / np.linalg.norm(v))
    return A, Gh, np.array(ys)


def eleven(Gh):
    """the 11 Ghat entries at graph distance <= 2: e0=uv, ua, ub, vc, vd, ab, cd, uc, ud, va, vb"""
    return dict(e0=Gh[0, 1], eua=Gh[0, 2], eub=Gh[0, 3], evc=Gh[1, 4], evd=Gh[1, 5], hab=Gh[2, 3], hcd=Gh[4, 5],
                huc=Gh[0, 4], hud=Gh[0, 5], hva=Gh[1, 2], hvb=Gh[1, 3])


def PB_from11(E):
    """PB as a closed form in the 11 entries (proof.md, with C_ab, C_cd written out)."""
    e0 = E['e0']
    ca, cb, cc, cd = [math.sqrt((1 + E[k]) / 2) for k in ('eua', 'eub', 'evc', 'evd')]
    Cab = (1 + E['eua'] + E['eub'] + E['hab']) / (4 * ca * cb)
    Ccd = (1 + E['evc'] + E['evd'] + E['hcd']) / (4 * cc * cd)
    pu = ca + cb + (e0 + E['huc']) / (2 * cc) + (e0 + E['hud']) / (2 * cd)
    pv = cc + cd + (e0 + E['hva']) / (2 * ca) + (e0 + E['hvb']) / (2 * cb)
    if 1 - e0 * e0 > 1e-14:
        proj2 = (pu * pu + pv * pv - 2 * e0 * pu * pv) / (1 - e0 * e0)
    else:
        proj2 = pu * pu
    return (KAPPA - 1) * (Cab + Ccd) + proj2 / 2 - 2, dict(Cab=Cab, Ccd=Ccd, pu=pu, pv=pv, proj2=proj2)


def PB(m):
    """robust: projection of Y onto span(a_u, a_v) by least squares; returns PB, loss |Y - PY|^2/2, a.C - K0."""
    A, Gh, ys = rows_keys(m)
    Y = ys[:4].sum(axis=0)
    S = np.vstack([A[0], A[1]]).T          # 123 x 2
    coef, *_ = np.linalg.lstsq(S, Y, rcond=1e-10)
    PY = S @ coef
    proj2 = PY @ PY
    c = cvec(ys[:4] @ ys[:4].T)
    pb = (KAPPA - 1) * (c[0] + c[1]) + proj2 / 2 - 2
    return pb, (Y @ Y - proj2) / 2, AVEC @ c - K0


def pbslack(x):
    m = x * x
    return PB(m)[0] - K0


if __name__ == '__main__':
    mT = mtree()
    pb, loss, cut = PB(mT); print('tree: PB - K0 = %.3e, loss %.3e, cut %.3e' % (pb - K0, loss, cut))
    A, Gh, ys = rows_keys(mT); print('closed form at tree', PB_from11(eleven(Gh))[0] - K0)
    rng = np.random.default_rng(12)
    # identity check and closed form check
    worst = 0
    for i in range(200):
        m = rng.dirichlet(0.3 * np.ones(NT)); pb, loss, cut = PB(m)
        A, Gh, ys = rows_keys(m); pb2 = PB_from11(eleven(Gh))[0]
        worst = max(worst, abs(pb + loss - cut - K0), abs(pb - pb2))
    print('identity a.C - K0 = PB - K0 + loss and closed form: max err %.2e' % worst)
    # single generator exactly
    for t in (0, 5, 50, 122):
        m = np.zeros(NT); m[t] = 1
        pb, loss, cut = PB(m); print('single generator', TYPES[t], 'PB - K0 = %.6f (2kappa+4-K0 = %.6f), loss %.2e' % (pb - K0, 2 * KAPPA + 4 - K0, loss))
    # e0 = 1 mixtures: all types with t_v = 1
    for tv in (-1, 1):
        idx = [i for i, t in enumerate(TYPES) if t[1] == tv]
        for k in range(3):
            m = np.zeros(NT); m[idx] = rng.dirichlet(np.ones(len(idx)))
            A, Gh, ys = rows_keys(m); pb, loss, cut = PB(m)
            print('e0 = %.12f mixture (t_v = %d, %d types): PB - K0 = %.6f, cut %.6f' % (Gh[0, 1], tv, len(idx), pb - K0, cut))
