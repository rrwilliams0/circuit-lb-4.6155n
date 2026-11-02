"""bk14_iv.py: minimal vectorised interval arithmetic with outward rounding (np.nextafter after every operation).
Intervals are pairs of numpy arrays (lo, hi).  Only what the certificates below need: +, -, *, /, sqrt, constants.
Constants are entered as enclosures computed in mpmath at 40 digits and widened by one ulp."""
import numpy as np
from mpmath import mp, mpf
mp.dps = 40
NINF, PINF = -np.inf, np.inf
def dn(x): return np.nextafter(x, NINF)
def up(x): return np.nextafter(x, PINF)
class I:
    __slots__ = ('lo', 'hi')
    def __init__(s, lo, hi=None):
        s.lo = np.asarray(lo, float); s.hi = np.asarray(lo if hi is None else hi, float)
    @staticmethod
    def const(x):   # x: mpmath number or string
        x = mpf(x); f = float(x); return I(dn(f), up(f))
    def __add__(s, o):
        o = o if isinstance(o, I) else I(o); return I(dn(s.lo + o.lo), up(s.hi + o.hi))
    __radd__ = __add__
    def __neg__(s): return I(-s.hi, -s.lo)
    def __sub__(s, o):
        o = o if isinstance(o, I) else I(o); return I(dn(s.lo - o.hi), up(s.hi - o.lo))
    def __rsub__(s, o): return I(o) - s
    def __mul__(s, o):
        o = o if isinstance(o, I) else I(o)
        c = np.stack([s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi])
        return I(dn(c.min(0)), up(c.max(0)))
    __rmul__ = __mul__
    def inv(s):
        assert np.all(s.lo > 0) or np.all(s.hi < 0)
        return I(dn(1.0 / s.hi), up(1.0 / s.lo))
    def __truediv__(s, o):
        o = o if isinstance(o, I) else I(o); return s * o.inv()
    def __rtruediv__(s, o): return I(o) * s.inv()
    def sqrt(s):
        assert np.all(s.lo >= 0); return I(dn(np.sqrt(s.lo)), up(np.sqrt(s.hi)))
    def pos(s): return I(np.maximum(s.lo, 0.0), np.maximum(s.hi, 0.0))
    def __getitem__(s, k): return I(s.lo[k], s.hi[k])
