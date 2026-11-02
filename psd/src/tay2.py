"""tay2.py : second-order interval automatic differentiation on top of the interval class ivarith.IA.
A Tay2 object carries an enclosure of (value, gradient, Hessian) of a function of n variables over a box.
Rules (all exact calculus, each evaluated in outward-rounded interval arithmetic):
  (f+g) = (f+g, f'+g', f''+g'');  (f g) = (fg, f'g + g'f, f''g + g''f + f'g'^T + g'f'^T);
  (1/x) = (1/x, -x'/x^2, 2 x'x'^T/x^3 - x''/x^2);  sqrt(x) = (s, x'/(2s), x''/(2s) - x'x'^T/(4 s x)) with s = sqrt(x).
Variables are seeded with value interval, unit gradient, zero Hessian.  The enclosure is valid for every point of the box
(the Hessian entries are interval functions evaluated over the box), hence usable in a Taylor bound with Lagrange remainder."""
import numpy as np
from ivarith import IA, dn, up


def _outer(a, b):
    """outer product of two IA vectors."""
    return IA(a.lo[:, None], a.hi[:, None]) * IA(b.lo[None, :], b.hi[None, :])


class Tay2:
    __slots__ = ('v', 'g', 'h', 'n')

    def __init__(s, v, g, h):
        s.v = v; s.g = g; s.h = h; s.n = g.lo.shape[0]

    @staticmethod
    def const(c, n):
        return Tay2(IA.of(c), IA(np.zeros(n)), IA(np.zeros((n, n))))

    @staticmethod
    def var(i, n, lo, hi):
        g = np.zeros(n); g[i] = 1.0
        return Tay2(IA(lo, hi), IA(g), IA(np.zeros((n, n))))

    def _lift(s, o):
        return o if isinstance(o, Tay2) else Tay2.const(o, s.n)

    def __add__(s, o):
        o = s._lift(o); return Tay2(s.v + o.v, s.g + o.g, s.h + o.h)
    __radd__ = __add__

    def __neg__(s): return Tay2(-s.v, -s.g, -s.h)

    def __sub__(s, o):
        o = s._lift(o); return Tay2(s.v - o.v, s.g - o.g, s.h - o.h)

    def __rsub__(s, o): return s._lift(o) - s

    def __mul__(s, o):
        if not isinstance(o, Tay2):
            c = IA.of(o); return Tay2(s.v * c, s.g * c, s.h * c)
        return Tay2(s.v * o.v, s.g * o.v + o.g * s.v, s.h * o.v + o.h * s.v + _outer(s.g, o.g) + _outer(o.g, s.g))
    __rmul__ = __mul__

    def inv(s):
        x = s.v; xi = x.inv(); xi2 = xi * xi; xi3 = xi2 * xi
        return Tay2(xi, -(s.g * xi2), _outer(s.g, s.g) * (xi3 * 2.0) - s.h * xi2)

    def __truediv__(s, o):
        if not isinstance(o, Tay2):
            return s * IA.of(o).inv()
        return s * o.inv()

    def sqrt(s):
        r = s.v.sqrt(); ri = r.inv(); half = ri * 0.5
        return Tay2(r, s.g * half, s.h * half - _outer(s.g, s.g) * (ri * s.v.inv() * 0.25))


# ---------------------------------------------------------------- the cone functions in G-space (21 variables)
VX = 6
UT = [(x, y) for x in range(VX) for y in range(x, VX)]      # 21 upper-triangular index pairs (x <= y)
UIDX = {p: i for i, p in enumerate(UT)}


def gidx(x, y): return UIDX[(min(x, y), max(x, y))]


E4 = [(0, 2), (0, 3), (1, 4), (1, 5)]
PAIRS = [(0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3)]


def cone_functions(lo, hi, Dir=None):
    """Given a 21-entry box [lo, hi] for G (upper-triangular order UT), return Tay2 objects for the six entries of C,
    in the PAIRS order (ab, cd, ac, ad, bc, bd).  Without Dir the 21 entries are the variables.  With Dir (an IA matrix
    21 x k) the variables are k coordinates s with G = G_box + Dir s: the enclosure is then of the s-derivatives at every
    point of the box (the box entries act as interval constants, the s-gradient of entry i is row i of Dir)."""
    if Dir is None:
        n = 21
        Gv = [Tay2.var(i, n, lo[i], hi[i]) for i in range(n)]
    else:
        n = Dir.lo.shape[1]
        Gv = [Tay2(IA(lo[i], hi[i]), IA(Dir.lo[i], Dir.hi[i]), IA(np.zeros((n, n)))) for i in range(21)]
    G = lambda x, y: Gv[gidx(x, y)]
    sd = [G(x, x).sqrt() for x in range(VX)]
    sdi = [z.inv() for z in sd]
    Gh = {}
    for x in range(VX):
        for y in range(VX):
            if x == y: Gh[(x, y)] = Tay2.const(1.0, n)
            elif y > x: Gh[(x, y)] = G(x, y) * sdi[x] * sdi[y]
            else: Gh[(x, y)] = Gh[(y, x)]
    W = {}
    for e, (a, b) in enumerate(E4):
        for f, (c, d) in enumerate(E4):
            if f < e: W[(e, f)] = W[(f, e)]; continue
            W[(e, f)] = Gh[(a, c)] + Gh[(a, d)] + Gh[(b, c)] + Gh[(b, d)]
    swi = [W[(e, e)].sqrt().inv() for e in range(4)]
    return [W[(i, j)] * swi[i] * swi[j] for (i, j) in PAIRS]


def Gbox_of_m(m_lo, m_hi, GEN_IA):
    """entrywise interval of G(m) = sum_t m_t g_t g_t^T for m in the box [m_lo, m_hi] (m_lo >= 0), GEN_IA a list of
    IA generator vectors; returns (lo, hi) arrays of length 21."""
    n = 21; lo = np.zeros(n); hi = np.zeros(n)
    acc = [IA(0.0) for _ in range(n)]
    for t, g in enumerate(GEN_IA):
        mt = IA(m_lo[t], m_hi[t])
        for i, (x, y) in enumerate(UT):
            acc[i] = acc[i] + mt * g[x] * g[y]
    return np.array([a.lo for a in acc]), np.array([a.hi for a in acc])
