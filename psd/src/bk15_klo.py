"""budget-aware lower bounds of the star key correlations on N (grid; NUMERICAL draft)."""
import math, numpy as np
rho = 2 * math.sqrt(2) / 3; mu0 = (1 + 2 * math.sqrt(2)) / 4; MTREE = 0.27658360688237347
Theta = 2 * math.pi * MTREE - 3 * math.acos(mu0); cT = math.cos(Theta / 3)
N0 = math.sqrt(2 + 2 * rho); Sig = 2 * math.sqrt(6 * cT + 3)
LB, LBV = 0.28903, 0.20516
def elo(r): return (rho / 2) * (math.sqrt(r) + 1 / math.sqrt(r))
def Nf(e): return math.sqrt(2 + 2 * e)
def hmcc(e1, d1, e2, d2): return ((e1 * math.sqrt(d1) + e2 * math.sqrt(d2)) / math.sqrt(2) - 0.5) / math.sqrt(d1 * d2)
def kf(e1, e2, h): return (1 + e1 + e2 + h) / (Nf(e1) * Nf(e2))
n = 61; best = {}
# k_a = key(ua) vs key(uv): edges uv (ratio d_v) and ua (ratio d_a), common neighbour u (G_uu = 1); third edge ub has N_b >= N0
# k_ab: edges ua, ub, common neighbour u; third edge uv has N_0 >= N0
# k_c (v-star): edges vu (ratio 1/d_v relative to v), vc (ratio d_c/d_v); same form after rescaling G_vv = 1: identical to k_a with box (LBV, LB). k_cd same as k_ab.
for name, (L1, L2) in dict(k_a=(LBV, LB), k_ab=(LB, LB)).items():
    bk = 9
    for l1 in np.linspace(-L1, L1, n):
        for l2 in np.linspace(-L2, L2, n):
            d1, d2 = math.exp(l1), math.exp(l2); e1lo, e2lo = elo(d1), elo(d2)
            if Nf(e1lo) + Nf(e2lo) + N0 > Sig: continue
            for e1 in np.linspace(e1lo, Nf(e1lo) and ((Sig - N0 - Nf(e2lo))**2 / 2 - 1), 7):
                e2hi = (Sig - N0 - Nf(e1))**2 / 2 - 1
                if e2hi < e2lo: continue
                for e2 in np.linspace(e2lo, e2hi, 7):
                    k = kf(e1, e2, hmcc(e1, d1, e2, d2))
                    if k < bk: bk = k; arg = (l1, l2, e1, e2)
    best[name] = bk; print('%s lower bound %.6f at log d = (%.4f, %.4f), e = (%.5f, %.5f)' % (name, bk, *arg))
klo = min(best.values()); khi = 3 * cT - 2 * klo
print('k_lo %.6f  k_hi %.6f  1-k_hi %.6f   h_hi = k_hi N0^2 - 1 - 2rho = %.6f' % (klo, khi, 1 - khi, khi * N0**2 - 1 - 2 * rho))
