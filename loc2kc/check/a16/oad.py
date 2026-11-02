"""oad.py : own rigorous arithmetic on mpmath.iv intervals (independent of ivarith / tay2 / bk13_lb).
D1: value + gradient (first-order forward AD); D2: value + gradient + Hessian (second-order forward AD).
Evaluated with interval inputs, the components enclose the value/derivatives at every point of the box."""
from mpmath import iv, mpf
iv.prec = 90

def I(x):
    """exact enclosure of a decimal string / int / float (floats taken exactly)."""
    if isinstance(x, str): return iv.mpf(x)          # iv.mpf of a string encloses the decimal
    return iv.mpf(x)

def ivhull(lo, hi): return iv.mpf([lo, hi])

class D1:
    __slots__ = ('v', 'g')
    def __init__(s, v, g): s.v = v; s.g = g
    @staticmethod
    def var(i, n, lo, hi):
        g = [iv.mpf(0)] * n; g[i] = iv.mpf(1); return D1(iv.mpf([lo, hi]), g)
    @staticmethod
    def const(c, n): return D1(c if isinstance(c, type(iv.mpf(0))) else iv.mpf(c), [iv.mpf(0)] * n)
    def _c(s, o): return o if isinstance(o, D1) else D1.const(o, len(s.g))
    def __add__(s, o):
        o = s._c(o); return D1(s.v + o.v, [a + b for a, b in zip(s.g, o.g)])
    __radd__ = __add__
    def __neg__(s): return D1(-s.v, [-a for a in s.g])
    def __sub__(s, o):
        o = s._c(o); return D1(s.v - o.v, [a - b for a, b in zip(s.g, o.g)])
    def __rsub__(s, o): return s._c(o) - s
    def __mul__(s, o):
        if not isinstance(o, D1):
            o = iv.mpf(o) if not isinstance(o, type(iv.mpf(0))) else o
            return D1(s.v * o, [a * o for a in s.g])
        return D1(s.v * o.v, [a * o.v + b * s.v for a, b in zip(s.g, o.g)])
    __rmul__ = __mul__
    def f1(s, f, fp):   # chain rule with f(v), f'(v)
        d = fp(s.v); return D1(f(s.v), [a * d for a in s.g])
    def inv(s): return s.f1(lambda x: 1 / x, lambda x: -1 / (x * x))
    def __truediv__(s, o):
        if isinstance(o, D1): return s * o.inv()
        return s * (1 / iv.mpf(o))
    def __rtruediv__(s, o): return s.inv() * o
    def sqrt(s): return s.f1(iv.sqrt, lambda x: 1 / (2 * iv.sqrt(x)))

class D2:
    __slots__ = ('v', 'g', 'h')
    def __init__(s, v, g, h): s.v = v; s.g = g; s.h = h
    @staticmethod
    def var(i, n, lo, hi):
        z = iv.mpf(0)
        g = [z] * n; g[i] = iv.mpf(1); return D2(iv.mpf([lo, hi]), g, [[z] * n for _ in range(n)])
    @staticmethod
    def const(c, n):
        z = iv.mpf(0); return D2(iv.mpf(c) if not isinstance(c, type(z)) else c, [z] * n, [[z] * n for _ in range(n)])
    def _c(s, o): return o if isinstance(o, D2) else D2.const(o, len(s.g))
    def __add__(s, o):
        o = s._c(o); n = len(s.g)
        return D2(s.v + o.v, [s.g[i] + o.g[i] for i in range(n)], [[s.h[i][j] + o.h[i][j] for j in range(n)] for i in range(n)])
    __radd__ = __add__
    def __neg__(s):
        n = len(s.g); return D2(-s.v, [-x for x in s.g], [[-s.h[i][j] for j in range(n)] for i in range(n)])
    def __sub__(s, o): return s + (-s._c(o))
    def __rsub__(s, o): return s._c(o) + (-s)
    def __mul__(s, o):
        n = len(s.g)
        if not isinstance(o, D2):
            o = iv.mpf(o) if not isinstance(o, type(iv.mpf(0))) else o
            return D2(s.v * o, [x * o for x in s.g], [[s.h[i][j] * o for j in range(n)] for i in range(n)])
        return D2(s.v * o.v, [s.g[i] * o.v + o.g[i] * s.v for i in range(n)],
                  [[s.h[i][j] * o.v + o.h[i][j] * s.v + s.g[i] * o.g[j] + s.g[j] * o.g[i] for j in range(n)] for i in range(n)])
    __rmul__ = __mul__
    def f2(s, f, fp, fpp):
        n = len(s.g); d1 = fp(s.v); d2 = fpp(s.v)
        return D2(f(s.v), [x * d1 for x in s.g], [[s.h[i][j] * d1 + s.g[i] * s.g[j] * d2 for j in range(n)] for i in range(n)])
    def inv(s): return s.f2(lambda x: 1 / x, lambda x: -1 / (x * x), lambda x: 2 / (x * x * x))
    def __truediv__(s, o):
        if isinstance(o, D2): return s * o.inv()
        return s * (1 / iv.mpf(o))
    def sqrt(s): return s.f2(iv.sqrt, lambda x: 1 / (2 * iv.sqrt(x)), lambda x: -1 / (4 * x * iv.sqrt(x)))
