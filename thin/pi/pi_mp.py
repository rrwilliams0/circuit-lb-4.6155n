"""pi_mp.py: re-close sampled leaves of pi_check.py in mpmath.iv (40 digits), same bound written again in mpmath.
Usage: python pi_mp.py sample.json target"""
import sys, json
from mpmath import mp, iv, mpf
iv.dps = 40; mp.dps = 50
S2 = iv.sqrt(2); RHO = 2 * S2 / 3
M_LO = mpf('0.2765836068')
MU0 = (1 + 2 * mp.sqrt(2)) / 4
CTH = mp.cos((2 * mp.pi * M_LO - 3 * mp.acos(MU0)) / 3)
SIG = iv.sqrt(3 + 6 * iv.mpf(CTH)).b + mpf('1e-40')


def E(lo, hi):
    r = mpf(1) if lo <= 1 <= hi else (lo if lo > 1 else hi)
    r = iv.mpf(r); return (RHO / 2 * (r + 1 / r)).a


def s(e): return iv.sqrt((1 + iv.mpf(e)) / 2).a


def bound(lo, hi):
    X = [iv.mpf([mpf(lo[k]), mpf(hi[k])]) for k in range(5)]
    xv, xa, xb, yc, yd = X
    ev, ea, eb = E(lo[0], hi[0]), E(lo[1], hi[1]), E(lo[2], hi[2]); ec, ed = E(lo[3], hi[3]), E(lo[4], hi[4])
    sv, sa, sb, sc, sd = s(ev), s(ea), s(eb), s(ec), s(ed)
    if sv + sa + sb > SIG or sv + sc + sd > SIG: return None
    Nhi = {}
    for x, so in (('a', sb), ('b', sa), ('c', sd), ('d', sc)):
        t = (iv.mpf(SIG) - sv - so).b; Nhi[x] = 2 * min(mpf(1), max(t, mpf(0)))
    dv = xv * xv; sx = {'a': xa, 'b': xb}; ex = {'a': ea, 'b': eb}; sz = {'c': xv * yc, 'd': xv * yd}; ez = {'c': ec, 'd': ed}
    best = None
    for x in 'ab':
        for z in 'cd':
            mvx = (ev * xv + ex[x] * sx[x]) / S2 - mpf(1) / 2
            hvx = max(mpf(0), mvx.a) / (xv * sx[x]).b
            muz = (ev * xv + ez[z] * xv * sz[z]) / S2 - dv / 2
            huz = max(mpf(0), muz.a) / sz[z].b
            gxz = ((S2 / 12) * (1 + dv) + (S2 / 6) * (sx[x] * sx[x] + sz[z] * sz[z])) / (sx[x] * sz[z])
            W = iv.mpf(ev) + iv.mpf(hvx) + iv.mpf(huz) + gxz
            c = (iv.mpf(W.a) / (iv.mpf(Nhi[x]) * iv.mpf(Nhi[z]))).a
            best = c if best is None else min(best, c)
    return best


d = json.load(open(sys.argv[1])); target = mpf(sys.argv[2])
fails = 0; nout = 0; mn = mpf(9); maxdiff = 0
for lo, hi, bf in d['sample']:
    b = bound(lo, hi)
    if b is None: nout += 1; continue
    if b < target: fails += 1; continue
    mn = min(mn, b); maxdiff = max(maxdiff, abs(float(b) - bf))
print('%d sampled leaves, mpmath.iv: closed by bound %d, proved outside N %d, failures %d, min bound %s, max |mp - float bound| (closed ones) %.2e' % (len(d['sample']), len(d['sample']) - fails - nout, nout, fails, mp.nstr(mn, 12), maxdiff))
