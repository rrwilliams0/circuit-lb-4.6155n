"""Independent chart code: constants, theta4 -> C, its Jacobian and Hessians (own derivation), interval gradient and
Hessian of Phi_F over a C-box (own dual-interval AD), interval Hessian of Phi_4 over a theta-box."""
import math
import numpy as np
from oia import V, D, grad_phi, PI
SQ2 = V(2.0).sqrt()
P = (1.0 + SQ2*2.0)*0.25          # (1 + 2 sqrt 2)/4
S = (2.0 + SQ2)*0.25              # (2 + sqrt 2)/4
_cT = [P, P, S, S, S, S]
_g = grad_phi(_cT)
KAP = (-_g[0])/(-_g[2])           # kappa = g_ab/g_ac (both negative; divide the positive negations)
CC = -_g[2]                        # c
K0 = KAP*P*2.0 + S*4.0
def Vs(lo, hi): return V(np.atleast_1d(np.asarray(lo, float)), np.atleast_1d(np.asarray(hi, float)))
SIG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]      # ac ad bc bd

def S_cut(x, y): return (K0 - KAP*(x + y)).sqrt()
def C4(th):
    x, y, tu, tv = th; Sc = S_cut(x, y); h = Sc*0.5
    return [x, y] + [(h + tu*float(su))*(h + tv*float(sv)) for su, sv in SIG]
def J4(th):
    x, y, tu, tv = th; Sc = S_cut(x, y); h = Sc*0.5; Sx = -(KAP/(Sc*2.0))
    rows = [[V(1.0), V(0.0), V(0.0), V(0.0)], [V(0.0), V(1.0), V(0.0), V(0.0)]]
    for su, sv in SIG:
        A = h + tu*float(su); B = h + tv*float(sv)
        dx = (A + B)*0.5*Sx
        rows.append([dx, dx, B*float(su), A*float(sv)])
    return rows
def HC4(th):
    x, y, tu, tv = th; Sc = S_cut(x, y); h = Sc*0.5
    Sx = -(KAP/(Sc*2.0)); Sxx = -(KAP.sq()/(Sc*Sc*Sc*4.0))
    Z = V(0.0); out = [[[Z]*4 for _ in range(4)] for _ in range(2)]
    for su, sv in SIG:
        A = h + tu*float(su); B = h + tv*float(sv)
        hxx = Sx.sq()*0.5 + (A + B)*0.5*Sxx; hxu = Sx*(0.5*su); hxv = Sx*(0.5*sv); huv = V(float(su*sv))
        out.append([[hxx, hxx, hxu, hxv], [hxx, hxx, hxu, hxv], [hxu, hxu, Z, huv], [hxv, hxv, huv, Z]])
    return out

def gH_box(clo, chi):
    """interval gradient g (6) and Hessian H (6x6) of Phi_F over the C-box [clo, chi] (natural extension via duals)."""
    n = 6
    cd = []
    for q in range(n):
        dv = [V(0.0)]*n; dv[q] = V(1.0)
        cd.append(D(Vs(clo[q], chi[q]), dv))
    gd = grad_phi(cd)
    g = [x.v for x in gd]
    H = [[gd[p].d[q] for q in range(n)] for p in range(n)]
    # symmetrise by intersection (true Hessian is symmetric)
    for p in range(n):
        for q in range(p + 1, n):
            a, b = H[p][q], H[q][p]
            lo = np.maximum(a.lo, b.lo); hi = np.minimum(a.hi, b.hi)
            H[p][q] = H[q][p] = V(lo, hi)
    return g, H

def pd_box_own(clo, chi):
    """sufficient test: every matrix in the box is PD (Gershgorin-free: leading minors as intervals via Schur steps)."""
    from oia import V as _V
    M = [[None]*4 for _ in range(4)]
    pairs = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
    for i in range(4): M[i][i] = _V(1.0)
    for q, (i, j) in enumerate(pairs): M[i][j] = M[j][i] = Vs(clo[q], chi[q])
    # interval Cholesky (sufficient for PD of every member)
    L = [[None]*4 for _ in range(4)]
    for j in range(4):
        s = M[j][j]
        for k in range(j): s = s - L[j][k].sq()
        ok = s.lo > 0
        if not np.any(ok): return ok
        s = V(np.where(ok, s.lo, 1.0), np.where(ok, s.hi, 1.0)); okall = ok if j == 0 else (okall & ok)
        L[j][j] = s.sqrt()
        for i in range(j + 1, 4):
            t = M[i][j]
            for k in range(j): t = t - L[i][k]*L[j][k]
            L[i][j] = t/L[j][j]
    return okall

def M4_box(lo, hi):
    """interval Hessian of Phi_4 over the theta-box: J^T H J + sum_p g_p HC_p (chain rule)."""
    th = [Vs(lo[j], hi[j]) for j in range(4)]
    Cb = C4(th); clo = [float(z.lo[0]) for z in Cb]; chi = [float(z.hi[0]) for z in Cb]
    if not pd_box_own(clo, chi): return None
    g, H = gH_box(clo, chi)
    J = J4(th); HC = HC4(th)
    M = [[None]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = V(0.0)
            for p in range(6):
                for q in range(6): acc = acc + J[p][i]*H[p][q]*J[q][j]
            for p in range(6): acc = acc + g[p]*HC[p][i][j]
            M[i][j] = acc
    return M, (clo, chi)

def split_box(lo, hi, k):
    """split a theta-box into k^4 equal sub-boxes; returns arrays (4, k^4)."""
    import itertools
    grids = [np.linspace(lo[j], hi[j], k + 1) if hi[j] > lo[j] else np.array([lo[j], hi[j]]) for j in range(4)]
    L = []; H = []
    for idx in itertools.product(*[range(len(g_) - 1) for g_ in grids]):
        L.append([grids[j][idx[j]] for j in range(4)]); H.append([grids[j][idx[j] + 1] for j in range(4)])
    L = np.array(L).T; H = np.array(H).T
    for j in range(4):   # outermost sub-boxes keep the exact outer bounds
        L[j][L[j] < lo[j]] = lo[j]; H[j][H[j] > hi[j]] = hi[j]
    return L, H

def M4_vec(L, H):
    """vectorised over sub-boxes (columns of L, H). returns (M, g, ok): M 4x4 of V arrays, ok mask (image box PD and
    all operations defined)."""
    th = [V(L[j], H[j]) for j in range(4)]
    Cb = C4(th); clo = [z.lo for z in Cb]; chi = [z.hi for z in Cb]
    ok = pd_box_own(clo, chi) | pd_box_mid(clo, chi)
    # neutralise non-PD sub-boxes (they are reported via ok) by collapsing them to the box centre value of a safe point
    g, Hs = gH_box(clo, chi)
    J = J4(th); HC = HC4(th)
    M = [[None]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = V(0.0)
            for p in range(6):
                for q in range(6): acc = acc + J[p][i]*Hs[p][q]*J[q][j]
            for p in range(6): acc = acc + g[p]*HC[p][i][j]
            M[i][j] = acc
    return M, g, ok

def hull_M(M, ok):
    lo = np.array([[np.min(M[i][j].lo[ok]) for j in range(4)] for i in range(4)])
    hi = np.array([[np.max(M[i][j].hi[ok]) for j in range(4)] for i in range(4)])
    return lo, hi

def pd_box_mid(clo, chi):
    """sufficient test that every matrix in the box is PD: with mid and radius r_q (outward), rho = Frobenius norm of the
    symmetric radius matrix (rounded up); then every member is >= mid - rho I (Weyl), and mid - rho I is checked PD by the
    interval Cholesky of that single matrix (pd_box_own on a point box). Vectorised."""
    clo = [np.asarray(v, float) for v in clo]; chi = [np.asarray(v, float) for v in chi]
    mid = [0.5*(a + b) for a, b in zip(clo, chi)]
    rad = [np.maximum(np.nextafter(b - m, np.inf), np.nextafter(m - a, np.inf)) for a, b, m in zip(clo, chi, mid)]
    rho = np.nextafter(np.sqrt(np.nextafter(2.0*sum(r*r for r in rad), np.inf)), np.inf)*(1 + 1e-15)
    # scale: mid - rho I  ~  (1 - rho) * correlation-like matrix with off-diagonals mid/(1 - rho); test PD of
    # [[1 - rho, mid], ...] directly by an interval Cholesky with diagonal 1 - rho
    from oia import V as _V
    pairs = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
    M = [[None]*4 for _ in range(4)]
    for i in range(4): M[i][i] = _V(np.nextafter(1.0 - rho, -np.inf), np.nextafter(1.0 - rho, -np.inf))
    for q, (i, j) in enumerate(pairs): M[i][j] = M[j][i] = _V(mid[q], mid[q])
    L = [[None]*4 for _ in range(4)]; okall = None
    for j in range(4):
        s = M[j][j]
        for k in range(j): s = s - L[j][k].sq()
        ok = s.lo > 0
        okall = ok if okall is None else (okall & ok)
        s = _V(np.where(ok, s.lo, 1.0), np.where(ok, s.hi, 1.0))
        L[j][j] = s.sqrt()
        for i in range(j + 1, 4):
            t = M[i][j]
            for k in range(j): t = t - L[i][k]*L[j][k]
            L[i][j] = t/L[j][j]
    return okall
