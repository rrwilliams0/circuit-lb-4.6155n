"""constant_check.py : the arithmetic of Theorem 1 from the certified upper end of M_tree.
  python -B constant_check.py [M_up]     (default M_up = 0.27658360966404055, the double printed by mtree/upper/hloc.py)
Exact Fractions of the double M_up; 2 kappa_E = (3/pi) arccos((1 + 2 sqrt 2)/4) enclosed with mpmath intervals at 200 bits
(arccos bracketed by interval cosines).  Prints: 2 kappa_E > M_up (so min(2 kappa_E, M_123) = M_tree <= M_up, Corollary A1),
1 + 1/M_up exactly and to 30 digits, and whether it is >= 4.6155432392 (the constant, rounded down)."""
import sys
from fractions import Fraction as F
from mpmath import mp, iv, mpf
mup_s = sys.argv[1] if len(sys.argv) > 1 else '0.27658360966404055'
x = F(float(mup_s))                                  # the exact rational value of the double
print('M_up = double(%s) = %s/%s exactly' % (mup_s, x.numerator, x.denominator))
mp.prec = 200; iv.prec = 200
p = (1 + 2 * iv.sqrt(2)) / 4                         # interval enclosing (1 + 2 sqrt2)/4
th = mp.acos(mp.mpf(p.mid)); e = mpf(2) ** -180
a, b = th - e, th + e                                # cos is decreasing on [0, pi]: need cos(a) >= p_hi and cos(b) <= p_lo
assert iv.cos(iv.mpf(a)).a >= p.b and iv.cos(iv.mpf(b)).b <= p.a
k2 = 3 * iv.mpf([a, b]) / iv.pi                      # 2 kappa_E = (3/pi) arccos(p)
lo_, hi_ = mpf(k2.a.a), mpf(k2.b.b)
print('2 kappa_E in [%s, %s]' % (mp.nstr(lo_, 45), mp.nstr(hi_, 45)))
k2lo = F(int(lo_ * 2 ** 220), 2 ** 220) - F(1, 2 ** 200)
print('2 kappa_E > M_up:', k2lo > x, ' (difference > %s)' % float(k2lo - x))
c = 1 + 1 / x
q = c.numerator * 10 ** 30 // c.denominator
print('1 + 1/M_up = %s/%s = %d.%s... (truncated)' % (c.numerator, c.denominator, q // 10 ** 30, str(q % 10 ** 30).zfill(30)))
print('1 + 1/M_up >= 4.6155432392 (10 decimals rounded down):', c >= F('4.6155432392'))
print('1 + 1/M_up >= 63829180483/13829180483 (= 1 + 1/0.27658360966):', c >= F(63829180483, 13829180483))
