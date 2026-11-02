"""pi_check.py : depth-first interval branch and bound for Lemma PI: on N every cross entry C_xz >= target.
Variables (square roots of the ratios, so that only +, -, *, /, sqrt occur, all correctly rounded in IEEE and widened by one
ulp outward after every operation): x_v = sqrt(d_v), x_a = sqrt(d_a), x_b = sqrt(d_b), y_c = sqrt(d_c/d_v), y_d = sqrt(d_d/d_v).
Box: |log d_v| <= 0.21, the other four |log| <= 0.29 (contains the box).
Bound (Lemma PI as stated, re-derived by an independent program):
  e_xy >= E(r) = (rho/2)(r + 1/r), r = sqrt of the edge ratio; exact minimum over an interval (convex, minimum at r = 1);
  out of N if s(e_uv)+s(e_ua)+s(e_ub) > Sigma* or s(e_uv)+s(e_vc)+s(e_vd) > Sigma*  (s(e) = sqrt((1+e)/2));
  N_a = 2 s(e_ua) <= 2 min(1, Sigma* - s(e_uv) - s(e_ub)), likewise b, c, d;
  W_xz >= e_uv + max(0, MCC_vx)/sqrt(d_v d_x) + max(0, MCC_uz)/sqrt(d_z) + D3_xz/sqrt(d_x d_z),
  MCC_vx = (e_uv sqrt d_v + e_ux sqrt d_x)/sqrt2 - 1/2, MCC_uz = (e_uv sqrt d_v + e_vz sqrt(d_v d_z))/sqrt2 - d_v/2,
  D3_xz = (sqrt2/12)(1 + d_v) + (sqrt2/6)(d_x + d_z);  C_xz >= W_xz^lo / (N_x^hi N_z^hi).
Usage: python pi_check.py target out.json [nsample]"""
import sys, time, json
import numpy as np
from mpmath import mp, mpf
mp.dps = 50
DN = lambda x: np.nextafter(x, -np.inf)
UPp = lambda x: np.nextafter(x, np.inf)


def cl(x): xf = float(x); return (float(DN(xf)), float(UPp(xf)))   # enclosure of an mpmath constant


RHO = cl(2 * mp.sqrt(2) / 3); IS2 = cl(1 / mp.sqrt(2)); C12 = cl(mp.sqrt(2) / 12); C6 = cl(mp.sqrt(2) / 6)
M_LO = mpf('0.2765836068')   # the value, below the certified lower end 0.27658360687864 (larger N)
MU0 = (1 + 2 * mp.sqrt(2)) / 4
TH = 2 * mp.pi * M_LO - 3 * mp.acos(MU0)
CTH = mp.cos(TH / 3)
SIG = cl(mp.sqrt(3 + 6 * CTH))[1]          # upper end of Sigma*


# interval helpers on (lo, hi) arrays, outward rounded
def add(a, b): return DN(a[0] + b[0]), UPp(a[1] + b[1])
def mul(a, b):   # all operands here are >= 0 on the lower side? not assumed: general
    c = np.stack([a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]); return DN(c.min(0)), UPp(c.max(0))
def div(a, b):
    assert np.all(b[0] > 0); c = np.stack([a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1]]); return DN(c.min(0)), UPp(c.max(0))
def csub(a, c): return DN(a[0] - c), UPp(a[1] - c)
def sqrt_(a): return DN(np.sqrt(np.maximum(a[0], 0))), UPp(np.sqrt(a[1]))
def pos(a): return np.maximum(a[0], 0.0), np.maximum(a[1], 0.0)


def Elo(lo, hi):
    """lower bound of (rho/2)(r + 1/r) over [lo, hi] (lo > 0): value at the point of [lo, hi] nearest to 1."""
    r = np.clip(1.0, lo, hi)
    inv = DN(1.0 / r)          # 1/r rounded down
    sm = DN(r + inv)
    return DN(DN(RHO[0] * 0.5) * sm)


def slo(e): return DN(np.sqrt(DN(DN(1.0 + e) * 0.5)))     # lower bound of s(e) from a lower bound of e


def bound(lo, hi):
    xv, xa, xb, yc, yd = [(lo[:, k], hi[:, k]) for k in range(5)]
    ev, ea, eb = Elo(*xv), Elo(*xa), Elo(*xb); ec, ed = Elo(*yc), Elo(*yd)
    sv, sa, sb, sc, sd = slo(ev), slo(ea), slo(eb), slo(ec), slo(ed)
    out = (DN(DN(sv + sa) + sb) > SIG) | (DN(DN(sv + sc) + sd) > SIG)
    Nhi = {}
    for x, so in (('a', sb), ('b', sa), ('c', sd), ('d', sc)):
        t = UPp(UPp(SIG - sv) - so)
        Nhi[x] = UPp(2.0 * np.minimum(np.maximum(t, 0.0), 1.0))
    dv = mul(xv, xv); evI = (ev, ev); one = (np.ones_like(ev), np.ones_like(ev))
    sx = {'a': xa, 'b': xb}; exI = {'a': (ea, ea), 'b': (eb, eb)}
    sz = {'c': mul(xv, yc), 'd': mul(xv, yd)}; ezI = {'c': (ec, ec), 'd': (ed, ed)}    # sqrt(d_z) = x_v y_z
    best = None
    for x in 'ab':
        for z in 'cd':
            mcc_vx = csub(mul(add(mul(evI, xv), mul(exI[x], sx[x])), IS2), 0.5)
            h_vx = div(pos(mcc_vx), mul(xv, sx[x]))
            mcc_uz = add(mul(add(mul(evI, xv), mul(ezI[z], mul(xv, sz[z]))), IS2), (-0.5 * dv[1], -0.5 * dv[0]))
            mcc_uz = (DN(mcc_uz[0]), UPp(mcc_uz[1]))
            h_uz = div(pos(mcc_uz), sz[z])
            dx = mul(sx[x], sx[x]); dz = mul(sz[z], sz[z])
            d3 = add(mul(C12, add(dv, one)), mul(C6, add(dx, dz)))
            g_xz = div(d3, mul(sx[x], sz[z]))
            Wlo = DN(DN(DN(ev + h_vx[0]) + h_uz[0]) + g_xz[0])
            den = UPp(Nhi[x] * Nhi[z])
            c = DN(Wlo / den)
            best = c if best is None else np.minimum(best, c)
    return best, out


def run(target, out_json, nsample=0, batch=50000, maxdepth=64, seed=5):
    t0 = time.time(); rng = np.random.default_rng(seed)
    L = [0.105, 0.145, 0.145, 0.145, 0.145]
    lo0 = np.array([cl(mp.exp(-mpf(l)))[0] for l in L]); hi0 = np.array([cl(mp.exp(mpf(l)))[1] for l in L])
    stack = [(lo0[None, :], hi0[None, :], 0)]
    closed = nout = total = 0; worst = 9.0; maxd = 0; peak = 0
    samp = []; seen = 0
    while stack:
        lo, hi, dep = stack.pop()
        if len(lo) > batch:
            stack.append((lo[batch:], hi[batch:], dep)); lo, hi = lo[:batch], hi[:batch]
        b, out = bound(lo, hi); total += len(lo)
        ok = (~out) & (b >= target)
        closed += int(ok.sum()); nout += int(out.sum())
        if ok.any():
            worst = min(worst, float(b[ok].min()))
            if nsample:   # reservoir sample of closed leaves
                idx = np.where(ok)[0]
                for i in idx:
                    seen += 1
                    if len(samp) < nsample: samp.append((lo[i].tolist(), hi[i].tolist(), float(b[i])))
                    else:
                        j = rng.integers(0, seen)
                        if j < nsample: samp[j] = (lo[i].tolist(), hi[i].tolist(), float(b[i]))
        rest = ~(ok | out)
        if not rest.any(): continue
        if dep >= maxdepth:
            print('FAIL: depth limit, %d open boxes' % rest.sum(), flush=True); return False
        lo, hi = lo[rest], hi[rest]
        w = hi / lo; k = np.argmax(w, axis=1); idx = np.arange(len(k))
        mid = 0.5 * (lo[idx, k] + hi[idx, k])
        hi1 = hi.copy(); hi1[idx, k] = mid; lo2 = lo.copy(); lo2[idx, k] = mid
        stack.append((np.concatenate([lo, lo2]), np.concatenate([hi1, hi]), dep + 1)); maxd = max(maxd, dep + 1)
        peak = max(peak, sum(len(s[0]) for s in stack))
    res = dict(target=target, processed=total, closed=closed, outside=nout, min_bound=worst, maxdepth=maxd, peak_stack_boxes=peak,
               secs=time.time() - t0, c_Theta=mp.nstr(CTH, 20), Sigma_hi=SIG, nsample=len(samp))
    print(res, flush=True)
    json.dump(dict(res=res, sample=samp), open(out_json, 'w'))
    return True


if __name__ == '__main__':
    run(float(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
