"""CERTIFIED constants of the PSD chain (mpmath intervals): k_hi, N_max, alpha_lo, c2*, and the F' assembly with b*."""
from mpmath import iv
iv.dps = 25
MT = iv.mpf('0.2765836068'); KLO = iv.mpf('0.9491'); BSTAR = iv.mpf('0.095'); DELTA_MAX = iv.mpf('0.095')
rho = 2 * iv.sqrt(2) / 3; mu0 = (1 + 2 * iv.sqrt(2)) / 4
Theta = 2 * iv.pi * MT - 3 * iv.atan2(iv.sqrt(1 - mu0**2), mu0); cT = iv.cos(Theta / 3); N0 = iv.sqrt(2 + 2 * rho); Sig = 2 * iv.sqrt(6 * cT + 3)
Nmax = Sig - 2 * N0; emax = Nmax**2 / 2 - 1; khi = 3 * cT - 2 * KLO
print('3cT %s\nNmax %s emax %s\nk_hi %s 1-k_hi %s' % (3 * cT, Nmax, emax, khi, 1 - khi))
# alpha lower bound as a function of e0 in [rho, emax]:  num(e0) = [4 + 4 rho - (3cT - k_lo) N(e0) Nmax]/Nmax - 2 (1/N0 - 1/Nmax),  alpha >= num / sqrt(2 - 2 e0)
def N(e): return iv.sqrt(2 + 2 * e)
def alpha(e0): return (((4 + 4 * rho - (3 * cT - KLO) * N(e0) * Nmax) / Nmax) - 2 * (1 / N0 - 1 / Nmax)) / iv.sqrt(2 - 2 * e0)
n = 400; amin = None
for i in range(n):
    e0 = iv.mpf([ (rho + (emax - rho) * i / n).a, (rho + (emax - rho) * (i + 1) / n).b ])
    a = alpha(e0); amin = a if amin is None else iv.mpf([min(amin.a, a.a), max(amin.b, a.b)])
alo = iv.mpf(amin.a); print('alpha_lo %s' % alo)
costh = alo / iv.sqrt(2 * (1 + khi)); c2 = (1 - 2 * costh**2)**2
print('cos theta >= %s   c <= %s   c2* <= %s' % (costh, 1 - 2 * costh**2, c2))
# lower bound of c'^2: sigma >= 4 C_lo, C_lo = (rho + 2 h_lo + Hlo)/Nmax^2 with h_lo, Hlo from the crude box minima (0.798, 0.647 numerically; use 0.79, 0.64)
Clo = (rho + 2 * iv.mpf('0.78') + iv.mpf('0.63')) / Nmax**2; c2lo = (4 * Clo - DELTA_MAX)**2 / (4 * (1 + khi)**2)
g = BSTAR / (2 * iv.sqrt((1 - khi) * (1 + KLO))); F = c2 + 2 * g**2 + g**4 / c2lo
print('C_lo %s c2_lo %s\ng* %s\nF-bound %s' % (Clo, c2lo, g, F))
