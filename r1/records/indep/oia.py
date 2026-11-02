"""Independent interval arithmetic (independent of ivarith / phifun / cert_gt / the code).
V: interval arrays (numpy lo, hi), outward rounding by nextafter after every correctly-rounded IEEE op (+,-,*,/,sqrt);
asin padded by 4 ulps + 1e-300 (numpy arcsin is not guaranteed correctly rounded, observed error <= 1 ulp).
D: forward-mode dual numbers over V (value + gradient list) for interval Hessians."""
import math
import numpy as np
INF = np.inf
def _dn(x): return np.nextafter(x, -INF)
def _up(x): return np.nextafter(x, INF)
class V:
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        lo = np.asarray(lo, float); hi = lo if hi is None else np.asarray(hi, float)
        self.lo = lo; self.hi = hi
    @staticmethod
    def of(x): return x if isinstance(x, V) else V(x)
    def __add__(s, o):
        o = V.of(o); return V(_dn(s.lo + o.lo), _up(s.hi + o.hi))
    __radd__ = __add__
    def __neg__(s): return V(-s.hi, -s.lo)
    def __sub__(s, o):
        o = V.of(o); return V(_dn(s.lo - o.hi), _up(s.hi - o.lo))
    def __rsub__(s, o): return V.of(o) - s
    def __mul__(s, o):
        o = V.of(o)
        p = [s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi]
        lo = np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))
        hi = np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3]))
        return V(_dn(lo), _up(hi))
    __rmul__ = __mul__
    def inv(s):
        # every denominator in this code is TRULY positive (PD matrices); an interval reaching <= 0 is overestimation:
        # enclose 1/x for x in (0, hi] as [1/hi, +inf); hi <= 0 gives nan (invalid, caught by isfinite checks).
        with np.errstate(divide='ignore', invalid='ignore'):
            lo = np.where(s.hi > 0, _dn(1.0/s.hi), np.nan)
            hi = np.where(s.lo > 0, _up(1.0/s.lo), np.where(s.hi > 0, np.inf, np.nan))
        return V(lo, hi)
    def __truediv__(s, o): return s*V.of(o).inv()
    def __rtruediv__(s, o): return V.of(o)*s.inv()
    def sqrt(s):
        # arguments of sqrt are TRULY nonnegative here; clip overestimated lower ends at 0; hi < 0 gives nan.
        with np.errstate(invalid='ignore'):
            return V(_dn(np.sqrt(np.maximum(s.lo, 0.0))).clip(0.0), _up(np.sqrt(s.hi)))
    def sq(s):
        lo = np.where(s.lo > 0, s.lo*s.lo, np.where(s.hi < 0, s.hi*s.hi, 0.0))
        hi = np.maximum(s.lo*s.lo, s.hi*s.hi)
        return V(_dn(lo), _up(hi))
    def asin(s):
        lo = np.clip(s.lo, -1.0, 1.0); hi = np.clip(s.hi, -1.0, 1.0)
        a = np.arcsin(lo); b = np.arcsin(hi)
        for _ in range(4): a = _dn(a); b = _up(b)
        return V(a - 1e-300, b + 1e-300)
    def mag(s): return np.maximum(np.abs(s.lo), np.abs(s.hi))
    def mid(s): return 0.5*(s.lo + s.hi)
    def rad(s): return np.maximum(_up(s.hi - s.mid()), _up(s.mid() - s.lo))
    def __repr__(s): return 'V(%s, %s)' % (s.lo, s.hi)

PI = V(_dn(math.pi), _up(math.pi))         # math.pi is within 1 ulp of pi
def vsum(xs):
    acc = V(0.0)
    for x in xs: acc = acc + x
    return acc

class D:
    """dual over V: value v, derivative list d (len n)."""
    __slots__ = ('v', 'd')
    def __init__(self, v, d): self.v = v; self.d = d
    @staticmethod
    def const(x, n): return D(V.of(x), [V(0.0)]*n)
    def _o(s, o): return o if isinstance(o, D) else D(V.of(o), [V(0.0)]*len(s.d))
    def __add__(s, o): o = s._o(o); return D(s.v + o.v, [a + b for a, b in zip(s.d, o.d)])
    __radd__ = __add__
    def __neg__(s): return D(-s.v, [-a for a in s.d])
    def __sub__(s, o): o = s._o(o); return D(s.v - o.v, [a - b for a, b in zip(s.d, o.d)])
    def __rsub__(s, o): return s._o(o) - s
    def __mul__(s, o):
        if not isinstance(o, D):
            o = V.of(o); return D(s.v*o, [a*o for a in s.d])
        return D(s.v*o.v, [a*o.v + s.v*b for a, b in zip(s.d, o.d)])
    __rmul__ = __mul__
    def inv(s):
        iv = s.v.inv(); f = -(iv*iv)
        return D(iv, [a*f for a in s.d])
    def __truediv__(s, o):
        if not isinstance(o, D): return s*V.of(o).inv()
        return s*o.inv()
    def sqrt(s):
        r = s.v.sqrt(); f = (r*2.0).inv()
        return D(r, [a*f for a in s.d])
    def asin(s):
        f = (1.0 - s.v.sq()).sqrt().inv()
        return D(s.v.asin(), [a*f for a in s.d])
    def sq(s): return D(s.v.sq(), [a*s.v*2.0 for a in s.d])

def _sqrt(x): return x.sqrt()
def _asin(x): return x.asin()

# ------------------------------------------------------------ Phi_F gradient (Plackett), own derivation from proof.md 2.5
# entries order (ab, cd, ac, ad, bc, bd); keys a=0 b=1 c=2 d=3
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]
IDX = {}
for q, (i, j) in enumerate(PAIRS): IDX[(i, j)] = q; IDX[(j, i)] = q
COEF = [0.75, 0.75, 0.25, 0.25, 0.25, 0.25]

def rget(c, i, j): return c[IDX[(i, j)]]

def partial_corr(c, i, j, k, l):
    """rho_{kl | ij} for a 4x4 unit-diagonal matrix with off-diagonal list c (V or D entries)."""
    r = rget(c, i, j); rki = rget(c, k, i); rkj = rget(c, k, j); rli = rget(c, l, i); rlj = rget(c, l, j); rkl = rget(c, k, l)
    # dependency-reduced but algebraically identical forms (1 - r^2 = (1 - r)(1 + r)):
    #   A_kk = 1 - (r_ki - r_kj)^2/((1-r)(1+r)) - 2 r_ki r_kj/(1+r)
    #   A_kl = r_kl - (r_ki - r_kj)(r_li - r_lj)/((1-r)(1+r)) - (r_ki r_lj + r_kj r_li)/(1+r)
    om = 1.0 - r; op = 1.0 + r; den = om*op
    dk = rki - rkj; dl = rli - rlj
    Akk = 1.0 - dk.sq()/den - rki*rkj*2.0/op
    All = 1.0 - dl.sq()/den - rli*rlj*2.0/op
    Akl = rkl - dk*dl/den - (rki*rlj + rkj*rli)/op
    return Akl/(Akk*All).sqrt()

def grad_phi(c):
    """gradient of Phi_F wrt the six off-diagonal entries (list of V or D)."""
    out = []
    for q, (i, j) in enumerate(PAIRS):
        k, l = [t for t in range(4) if t not in (i, j)]
        rho = partial_corr(c, i, j, k, l)
        r = c[q]
        num = COEF[q] - rho.asin()/(PI*2.0) if not isinstance(rho, D) else (rho.asin()*(-1.0)*(PI*2.0).inv() + COEF[q])
        out.append(-(num/(((1.0 - r)*(1.0 + r)).sqrt()*PI)))
    return out
