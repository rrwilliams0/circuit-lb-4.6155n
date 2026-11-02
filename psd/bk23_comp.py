r"""bk23_comp.py : EXACT check of the composition obligation (C-LR) of Theorem LR (lower-row checks).

Input: a targets file (json) with
  U = [[e_lo, L, t0, t1], ...]   (e_lo = null for the check without a lower row; L the check's L constant)
  V = [[e_lo, t0, t1], ...]
Check j certifies  F(s) <= ell_j(E^lin(s)) := t0_j + t1_j (E^lin(s) - c_j)  for every star s of the family's domain with
E^lin(s) >= e_lo_j (c_j := e_lo_j, or 0 if e_lo_j is null).  Everything below is exact (Fractions of the file's doubles).
 1. t1_j >= 0 for every check; at least one check of each family has e_lo = null.
 2. Grid G = {0, B} U {e_lo_j in (0, B)} (both families).  For each grid interval I = [g_i, g_{i+1}] and each family, a
    piece p(I) is chosen among the checks ACTIVE on I (e_lo null or e_lo <= g_i); T~(x) := ell_{p(I)}(x) on I.  Then
    T~(x) >= min{ell_j(x) : j active at x} for every x in [0, B] (at a grid point both adjacent pieces are active).
 3. For every cell I x J of G x G and its polygon P = (I x J) intersected with {x + w y <= B}: the four quantities
    T~U(x), T~V(y), T~U(y), T~V(x) are affine on P; the value Phi'(T~U(x) + T~V(y), T~U(y) + T~V(x)) is computed exactly at
    every vertex of P; all four values must lie in [0, 0.15] at every vertex (then on P, and Phi' is convex there), so the max
    over P is the max over its vertices.  Output: the max over all cells, which must be < 1 (in units of Phi_0).
Usage: python bk23_comp.py <targets.json>
"""
import sys, json, hashlib
from fractions import Fraction as Fr

B = Fr('0.0024634531'); W = Fr('0.97147')
KHI = 3 * Fr('0.9595702342051') - 2 * Fr('0.9491')
D2 = 4 * (1 - KHI) * (1 + Fr('0.9491')); C2S = Fr('0.8643473'); C2L = Fr('0.61536')
PHI0 = D2 * (1 - C2S)


def Phi(a, b): return a * a + b * b + a * a * b * b / (D2 * C2L)


def load(TG):
    U = [(None if r[0] is None else Fr(r[0]), Fr(r[2]), Fr(r[3])) for r in TG['U']]
    V = [(None if r[0] is None else Fr(r[0]), Fr(r[1]), Fr(r[2])) for r in TG['V']]
    for fam in (U, V):
        assert any(e is None for e, _, _ in fam), 'a check without lower row is required'
        for e, t0, t1 in fam: assert t1 >= 0
    return U, V


def ell(p, x):
    e, t0, t1 = p
    return t0 + t1 * (x - (e if e is not None else 0))


def compose(U, V, verbose=True):
    G = sorted(set([Fr(0), B] + [e for fam in (U, V) for e, _, _ in fam if e is not None and 0 < e < B]))
    def pick(fam, a, b):
        act = [p for p in fam if p[0] is None or p[0] <= a]
        return min(act, key=lambda p: max(ell(p, a), ell(p, b)))
    PU = [pick(U, G[i], G[i + 1]) for i in range(len(G) - 1)]
    PV = [pick(V, G[i], G[i + 1]) for i in range(len(G) - 1)]
    worst = Fr(0); arg = None; vmax = Fr(0); vmin = Fr(1); nv = 0
    for i in range(len(G) - 1):
        for j in range(len(G) - 1):
            xa, xb, ya, yb = G[i], G[i + 1], G[j], G[j + 1]
            if xa + W * ya > B: continue
            # polygon vertices of [xa, xb] x [ya, yb] cut by x + W y <= B
            cand = [(xa, ya), (xb, ya), (xa, yb), (xb, yb)]
            verts = [(x, y) for x, y in cand if x + W * y <= B]
            for y in (ya, yb):
                x = B - W * y
                if xa <= x <= xb: verts.append((x, y))
            for x in (xa, xb):
                y = (B - x) / W
                if ya <= y <= yb: verts.append((x, y))
            for x, y in set(verts):
                vals = (ell(PU[i], x), ell(PV[j], y), ell(PU[j], y), ell(PV[i], x))
                for v in vals:
                    assert 0 <= v <= Fr(15, 100), ('value outside [0, 0.15]', float(v))
                vmax = max(vmax, max(vals)); vmin = min(vmin, min(vals)); nv += 1
                r = Phi(vals[0] + vals[1], vals[2] + vals[3]) / PHI0
                if r > worst: worst, arg = r, (x, y, vals)
    if verbose:
        print('grid %d points, %d vertex evaluations, values in [%.6f, %.6f]' % (len(G), nv, float(vmin), float(vmax)))
        x, y, v = arg
        print("(C-LR): max Phi'/Phi_0 = %.7f at x = %.7f, y = %.7f, (TU(x), TV(y), TU(y), TV(x)) = (%.6f, %.6f, %.6f, %.6f)"
              % (float(worst), float(x), float(y), *[float(t) for t in v]))
    return worst, arg


if __name__ == '__main__':
    raw = open(sys.argv[1], 'rb').read()
    print('targets', sys.argv[1], 'md5', hashlib.md5(raw).hexdigest())
    U, V = load(json.loads(raw))
    w, _ = compose(U, V)
    print('(C-LR) HOLDS' if w < 1 else '(C-LR) FAILS')
