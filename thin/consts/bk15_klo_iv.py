"""CERTIFIED lower bound of every star key correlation on N (mpmath interval branch and bound, 4 variables).
k = (1 + e1 + e2 + h)/(N1 N2), h = MCC bound [(e1 sqrt d1 + e2 sqrt d2)/sqrt2 - 1/2]/sqrt(d1 d2), N_i = sqrt(2 + 2 e_i),
over |log d_i| <= LB, e_i >= (rho/2)(sqrt d_i + 1/sqrt d_i), N1 + N2 <= Sig* - N0 (Lemma NS + star lemma), M_tree lower end."""
import sys, time
from mpmath import iv, mp
iv.dps = 25
MT = iv.mpf('0.2765836068'); LB = iv.mpf('0.28903'); TARGET = iv.mpf(sys.argv[1] if len(sys.argv) > 1 else '0.94925')
rho = 2 * iv.sqrt(2) / 3; mu0 = (1 + 2 * iv.sqrt(2)) / 4
Theta = 2 * iv.pi * MT - 3 * iv.atan2(iv.sqrt(1 - mu0**2), mu0); cT = iv.cos(Theta / 3); N0 = iv.sqrt(2 + 2 * rho); Sig = 2 * iv.sqrt(6 * cT + 3)
emax = (Sig - 2 * N0)**2 / 2 - 1
print('cT', cT, 'Sig', Sig, 'emax', emax)
def N(e): return iv.sqrt(2 + 2 * e)
def kbox(l1, l2, e1, e2):
    d1, d2 = iv.exp(l1), iv.exp(l2)
    h = ((e1 * iv.sqrt(d1) + e2 * iv.sqrt(d2)) / iv.sqrt(2) - iv.mpf(1) / 2) / iv.sqrt(d1 * d2)
    return (1 + e1 + e2 + h) / (N(e1) * N(e2))
def elo(l): d = iv.exp(l); return (rho / 2) * (iv.sqrt(d) + 1 / iv.sqrt(d))
def hi(x): return iv.mpf(x.b)
def lo(x): return iv.mpf(x.a)
root = (iv.mpf([-LB.b, LB.b]), iv.mpf([-LB.b, LB.b]), iv.mpf([rho.a, emax.b]), iv.mpf([rho.a, emax.b]))
stack = [root]; closed = infeas = 0; worst = None; t0 = time.time(); nmax = 0
while stack:
    b = stack.pop(); l1, l2, e1, e2 = b
    # feasibility pruning
    if lo(N(e1) + N(e2)) > hi(Sig - N0) or hi(e1) < lo(elo(l1)) or hi(e2) < lo(elo(l2)):
        infeas += 1; continue
    k = kbox(l1, l2, e1, e2)
    if lo(k) >= TARGET: closed += 1; continue
    widths = [hi(x) - lo(x) for x in b]
    if max(widths) < iv.mpf('1e-4'):
        print('FAIL: box', b, 'k', k); sys.exit(1)
    j = max(range(4), key=lambda i: widths[i]); m = (lo(b[j]) + hi(b[j])) / 2
    for part in (iv.mpf([lo(b[j]).a, m.b]), iv.mpf([m.a, hi(b[j]).b])):
        nb = list(b); nb[j] = part; stack.append(tuple(nb))
    nmax = max(nmax, len(stack))
print('CERTIFIED: k >= %s on the relaxed star set; closed %d boxes, infeasible %d, %.1fs' % (TARGET, closed, infeas, time.time() - t0))
