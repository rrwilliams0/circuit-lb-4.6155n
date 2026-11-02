"""ivarith.py : own interval arithmetic on numpy arrays of doubles.
Every +,-,*,/,sqrt is IEEE round-to-nearest (numpy float64, SSE2) followed by one ulp outward (np.nextafter), so the exact
real result is enclosed.  Constants given as Python floats are taken as EXACT doubles (only pass exactly representable ones).
arcsin: numpy/math candidate moved outward, then VERIFIED with mpmath's interval sine (sin increasing on [-pi/2, pi/2])."""
import numpy as np, math
from mpmath import iv
iv.prec = 80
INF = np.inf


def dn(x): return np.nextafter(x, -INF)
def up(x): return np.nextafter(x, INF)


class IA:
    __slots__ = ('lo', 'hi')

    def __init__(s, lo, hi=None):
        s.lo = np.asarray(lo, float); s.hi = s.lo if hi is None else np.asarray(hi, float)

    @staticmethod
    def of(x): return x if isinstance(x, IA) else IA(x)

    def __add__(s, o):
        o = IA.of(o); return IA(dn(s.lo + o.lo), up(s.hi + o.hi))
    __radd__ = __add__

    def __neg__(s): return IA(-s.hi, -s.lo)

    def __sub__(s, o):
        o = IA.of(o); return IA(dn(s.lo - o.hi), up(s.hi - o.lo))

    def __rsub__(s, o): return IA.of(o) - s

    def __mul__(s, o):
        o = IA.of(o)
        a, b, c, d = s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi
        return IA(dn(np.minimum(np.minimum(a, b), np.minimum(c, d))), up(np.maximum(np.maximum(a, b), np.maximum(c, d))))
    __rmul__ = __mul__

    def inv(s):
        if np.any((s.lo <= 0) & (s.hi >= 0)): raise ZeroDivisionError('inv')
        return IA(dn(1.0/s.hi), up(1.0/s.lo))

    def __truediv__(s, o): return s*IA.of(o).inv()

    def __rtruediv__(s, o): return IA.of(o)*s.inv()

    def sqrt(s):
        if np.any(s.lo < 0): raise ArithmeticError('sqrt of negative')
        return IA(np.maximum(0.0, dn(np.sqrt(s.lo))), up(np.sqrt(s.hi)))

    def nsqrt(s):
        """sqrt with NaN in the lanes where the argument is not certified >= 0 (lane-wise failure, no exception)."""
        bad = ~(s.lo >= 0)
        with np.errstate(invalid='ignore'):
            lo = np.where(bad, np.nan, np.maximum(0.0, dn(np.sqrt(np.where(bad, 1.0, s.lo)))))
            hi = np.where(bad, np.nan, up(np.sqrt(np.where(bad, 1.0, s.hi))))
        return IA(lo, hi)

    def ninv(s):
        """1/x with NaN in the lanes where 0 is not certified outside the interval."""
        bad = ~((s.lo > 0) | (s.hi < 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            lo = np.where(bad, np.nan, dn(1.0/np.where(bad, 1.0, s.hi)))
            hi = np.where(bad, np.nan, up(1.0/np.where(bad, 1.0, s.lo)))
        return IA(lo, hi)

    def __getitem__(s, k): return IA(s.lo[k], s.hi[k])

    def mag(s): return np.maximum(-s.lo, s.hi)

    def mid(s): return 0.5*(s.lo + s.hi)


def hull(a, b): return IA(np.minimum(a.lo, b.lo), np.maximum(a.hi, b.hi))


def rsum(x, axis):
    """rigorous sum of an interval array along an axis: float sum plus a bound on the rounding error
    (|fl(sum) - sum| <= gamma_{n-1} sum|x_i| for any summation order; we use 2 n 2^-52 >= 4 gamma_{n-1})."""
    n = x.lo.shape[axis]
    sl = x.lo.sum(axis=axis); sh = x.hi.sum(axis=axis)
    el = up(up(np.abs(x.lo).sum(axis=axis)*(2.0*n*2.0**-52)) + 1e-300)
    eh = up(up(np.abs(x.hi).sum(axis=axis)*(2.0*n*2.0**-52)) + 1e-300)
    return IA(dn(sl - el), up(sh + eh))


PI = IA(dn(math.pi), up(math.pi))


def _asin_lo(x):
    if not math.isfinite(x): return math.nan
    if x <= -1.0: return dn(dn(-math.pi/2))
    if x >= 1.0: return dn(dn(math.pi/2))
    y = dn(dn(math.asin(x)))
    while iv.sin(iv.mpf(y)).b > x: y = dn(y)
    return y


def _asin_hi(x):
    if not math.isfinite(x): return math.nan
    if x >= 1.0: return up(up(math.pi/2))
    if x <= -1.0: return up(up(-math.pi/2))
    y = up(up(math.asin(x)))
    while iv.sin(iv.mpf(y)).a < x: y = up(y)
    return y


def asin(X):
    """enclosure of arcsin over X (clipped to [-1,1], valid because the true argument lies in [-1,1]; arcsin increasing)."""
    with np.errstate(invalid='ignore'):
        lo = np.clip(X.lo, -1.0, 1.0); hi = np.clip(X.hi, -1.0, 1.0)
    L = np.array([_asin_lo(float(v)) for v in lo.ravel()]).reshape(lo.shape)
    H = np.array([_asin_hi(float(v)) for v in hi.ravel()]).reshape(hi.shape)
    return IA(L, H)
