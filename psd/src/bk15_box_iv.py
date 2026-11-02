"""CERTIFIED box minima: h_lo = min of the normalised MCC bound (e at the edge-lemma minimum) over the ratio box,
H_lo = min of the D3 lower bound (normalised) over the ratio box.  mpmath intervals on a grid."""
from mpmath import iv
iv.dps = 20
LB = iv.mpf('0.28903'); LBV = iv.mpf('0.20516'); rho = 2 * iv.sqrt(2) / 3; s2 = iv.sqrt(2)
def elo(d): return (rho / 2) * (iv.sqrt(d) + 1 / iv.sqrt(d))
def sub(L, n, i): return iv.mpf([(-L + 2 * L * i / n).a, (-L + 2 * L * (i + 1) / n).b])
n = 60; hmin = None
# MCC at common neighbour w (normalised G_ww = 1): h >= [(e1 sqrt d1 + e2 sqrt d2)/sqrt2 - 1/2]/sqrt(d1 d2); ratio boxes (LB, LB) contain (LBV, LB)
for i in range(n):
    for j in range(n):
        d1, d2 = iv.exp(sub(LB, n, i)), iv.exp(sub(LB, n, j))
        h = ((elo(d1) * iv.sqrt(d1) + elo(d2) * iv.sqrt(d2)) / s2 - iv.mpf(1) / 2) / iv.sqrt(d1 * d2)
        hmin = h.a if hmin is None else min(hmin, h.a)
print('h_lo >= %s' % hmin)
Hmin = None; n = 30
for i in range(n):
    for j in range(n):
        for k in range(n):
            dv = iv.exp(sub(LBV, n, i)); dx = iv.exp(sub(LB, n, j)); dz = dv * iv.exp(sub(LB, n, k))
            H = ((s2 / 12) * (1 + dv) + (s2 / 6) * (dx + dz)) / iv.sqrt(dx * dz)
            Hmin = H.a if Hmin is None else min(Hmin, H.a)
print('H_lo >= %s' % Hmin)
