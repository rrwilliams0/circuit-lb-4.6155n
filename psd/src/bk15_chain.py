"""crude interval chain for O-the-2 (floats; grid maxima).  NUMERICAL draft of the proof constants."""
import math, itertools, numpy as np
rho = 2 * math.sqrt(2) / 3; mu0 = (1 + 2 * math.sqrt(2)) / 4; MTREE = 0.27658360688237347
Theta = 2 * math.pi * MTREE - 3 * math.acos(mu0); cT = math.cos(Theta / 3)
N0 = math.sqrt(2 + 2 * rho); Sig = 2 * math.sqrt(6 * cT + 3)
Nmax = Sig - 2 * N0; emax = Nmax**2 / 2 - 1
print('Theta %.6f cT %.7f 3cT-3mu0 %.6f Sig %.6f N0 %.6f Nmax %.6f emax %.6f' % (Theta, cT, 3 * cT - 3 * mu0, Sig, N0, Nmax, emax))
LB, LBV = 0.28903, 0.20516
def elo(r): return (rho / 2) * (math.sqrt(r) + 1 / math.sqrt(r))
# k_a lower bound: min over e0, eua in [elo(.), emax], dv, da in box, with h_va >= MCC normalised
def klo_scan(n=41):
    best = 9
    for lv in np.linspace(-LBV, LBV, n):
        for la in np.linspace(-LB, LB, n):
            dv, da = math.exp(lv), math.exp(la)
            if elo(dv) > emax or elo(da) > emax: continue
            for e0 in np.linspace(elo(dv), emax, 5):
                for ea in np.linspace(elo(da), emax, 5):
                    h = ((e0 * math.sqrt(dv) + ea * math.sqrt(da)) / math.sqrt(2) - 0.5) / math.sqrt(dv * da)
                    k = (1 + e0 + ea + h) / math.sqrt((2 + 2 * e0) * (2 + 2 * ea))
                    if k < best: best = k; arg = (lv, la, e0, ea, h)
    return best, arg
# k_ab lower bound: both ends are a, b with ratios in the 0.289 box relative to u (G_uu = 1), common neighbour u
def kablo_scan(n=41):
    best = 9
    for la in np.linspace(-LB, LB, n):
        for lb in np.linspace(-LB, LB, n):
            da, db = math.exp(la), math.exp(lb)
            if elo(da) > emax or elo(db) > emax: continue
            for ea in np.linspace(elo(da), emax, 5):
                for eb in np.linspace(elo(db), emax, 5):
                    h = ((ea * math.sqrt(da) + eb * math.sqrt(db)) / math.sqrt(2) - 0.5) / math.sqrt(da * db)
                    k = (1 + ea + eb + h) / math.sqrt((2 + 2 * ea) * (2 + 2 * eb))
                    if k < best: best = k; arg = (la, lb, ea, eb, h)
    return best, arg
ka_lo, arg1 = klo_scan(); kab_lo, arg2 = kablo_scan()
print('k_a lower %.6f at %s' % (ka_lo, np.round(arg1, 4))); print('k_ab lower %.6f at %s' % (kab_lo, np.round(arg2, 4)))
k_lo = min(ka_lo, kab_lo); k_hi = 3 * cT - 2 * k_lo
print('k_lo %.6f  k_hi %.6f  (1-k)_min %.6f' % (k_lo, k_hi, 1 - k_hi))
# h ranges: h_va >= MCC min over box; h_va <= k_hi N0^2 - 1 - 2 rho
h_lo = min(((e0 * math.sqrt(dv) + ea * math.sqrt(da)) / math.sqrt(2) - 0.5) / math.sqrt(dv * da)
           for lv in np.linspace(-LBV, LBV, 41) for la in np.linspace(-LB, LB, 41) for dv in [math.exp(lv)] for da in [math.exp(la)]
           if elo(dv) <= emax and elo(da) <= emax for e0 in [elo(dv)] for ea in [elo(da)])
h_hi = k_hi * N0**2 - 1 - 2 * rho
print('h_lo %.6f h_hi %.6f' % (h_lo, h_hi))
# Ghat ranges over the box with h <= h_hi (upper) / lower bound
Hhi = -9; Hlo = 9
for lv in np.linspace(-LBV, LBV, 21):
    for lx in np.linspace(-LB, LB, 21):
        for lz in np.linspace(-LB, LB, 21):
            dv, dx, dz = math.exp(lv), math.exp(lx), math.exp(lz + lv)
            hi = ((3 / math.sqrt(2)) * (h_hi * math.sqrt(dz) + h_hi * math.sqrt(dv * dx)) - (2 * math.sqrt(2) / 3) * (1 + dv) - (math.sqrt(2) / 3) * (dx + dz)) / math.sqrt(dx * dz)
            lo = ((math.sqrt(2) / 12) * (1 + dv) + (math.sqrt(2) / 6) * (dx + dz)) / math.sqrt(dx * dz)
            Hhi = max(Hhi, hi); Hlo = min(Hlo, lo)
print('Ghat range [%.5f, %.5f]' % (Hlo, Hhi))
W_hi = emax + 2 * h_hi + Hhi; W_lo = rho + 2 * h_lo + Hlo
C_hi = W_hi / N0**2; C_lo = W_lo / Nmax**2
print('W range [%.5f, %.5f]  C_xz range [%.5f, %.5f]' % (W_lo, W_hi, C_lo, C_hi))
sig_hi = 4 * C_hi; diff_hi = C_hi - C_lo
beta_hi = 2 * diff_hi; alt_hi = 2 * diff_hi
D_hi = (sig_hi * alt_hi + beta_hi**2) / 4; delta_hi = D_hi / C_lo
c2_hi = sig_hi**2 / (4 * (1 + k_lo)**2); c2_lo = (4 * C_lo)**2 / (4 * (1 + k_hi)**2)
g_hi = (beta_hi + delta_hi) / (2 * math.sqrt((1 - k_hi) * (1 + k_lo)))
F = c2_hi + 2 * g_hi**2 + g_hi**4 / c2_lo
print('sigma_hi %.4f beta_hi %.4f alt_hi %.4f delta_hi %.4f c2_hi %.4f c2_lo %.4f g_hi %.4f  F_hi %.4f' % (sig_hi, beta_hi, alt_hi, delta_hi, c2_hi, c2_lo, g_hi, F))
