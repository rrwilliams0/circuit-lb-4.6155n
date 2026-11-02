"""bk14_pilo2.py: CERTIFIED lower bound pi_lo of every cross entry C_xz on N (Lemma PI).  Vectorised interval B&B
(bk14_iv: outward rounding after every operation).  Variables: r_v = sqrt(d_v), r_a = sqrt(d_a), r_b, r_c = sqrt(d_c/d_v),
r_d = sqrt(d_d/d_v); box |log d_v| <= 0.21, the other four |log| <= 0.29 (contains the box).  Region: the two
star constraints of Lemma NS (s(e_uv)+s(e_ua)+s(e_ub) <= Sigma*, same at v, with Lemma 2.2's lower bounds)."""
import sys, time, json
import numpy as np
from mpmath import mp, mpf
from bk14_iv import I, dn, up
mp.dps = 50
MT_LO = mpf('0.2765836068')
MU0m = (1 + 2 * mp.sqrt(2)) / 4
TH = 2 * mp.pi * MT_LO - 3 * mp.acos(MU0m)
CT = mp.cos(TH / 3)
SIG = mp.sqrt(3 + 6 * CT)
SIGHI = float(SIG) * (1 + 1e-14)
RHO = I.const(2 * mp.sqrt(2) / 3); S2 = I.const(mp.sqrt(2)); HALF = I(0.5)
C12 = I.const(mp.sqrt(2) / 12); C6 = I.const(mp.sqrt(2) / 6)
def ef(r): return RHO * HALF * (r + r.inv())
def sv(e): return ((e + 1.0) * HALF).sqrt()
def bound(lo, hi):
    """lo, hi: (n, 5) arrays of box corners in the r-variables.  Returns (lower bound of min cross entry, out-of-region mask)."""
    rv, ra, rb, rc, rd = [I(lo[:, k], hi[:, k]) for k in range(5)]
    ev, ea, eb, ec, ed = ef(rv), ef(ra), ef(rb), ef(rc), ef(rd)
    s_v, s_a, s_b, s_c, s_d = sv(ev), sv(ea), sv(eb), sv(ec), sv(ed)
    out = ((s_v + s_a + s_b).lo > SIGHI) | ((s_v + s_c + s_d).lo > SIGHI)
    def ebar(s_o):   # upper bound of the third edge cosine: s(e) <= Sig - s_v^lo - s_o^lo, e = 2 s^2 - 1
        t = up(up(SIGHI - s_v.lo) - s_o.lo); t = np.maximum(t, 0.0)
        e = up(up(2.0 * up(t * t)) - 1.0); return np.minimum(e, 1.0)
    eb_ = {'a': ebar(s_b), 'b': ebar(s_a), 'c': ebar(s_d), 'd': ebar(s_c)}
    dv = rv * rv; rx = {'a': ra, 'b': rb}; ex = {'a': ea, 'b': eb}
    rz = {'c': rv * rc, 'd': rv * rd}; ez = {'c': ec, 'd': ed}           # sqrt(d_z) = r_v r_z'
    res = None
    for x in 'ab':
        for z in 'cd':
            Gvx = (ev * rv + ex[x] * rx[x]) / S2 - HALF                      # G_uv >= e0 r_v, G_ux >= e_ux r_x
            hvx = Gvx.pos() / (rv * rx[x])
            Guz = (ev * rv + ez[z] * rv * rz[z]) / S2 - dv * HALF          # G_vz >= e_vz sqrt(d_v d_z)
            huz = Guz.pos() / rz[z]
            Gxz = (C12 * (dv + 1.0) + C6 * (rx[x] * rx[x] + rz[z] * rz[z])) / (rx[x] * rz[z])
            W = ev + hvx + huz + Gxz
            den = (I(2.0) + I(2.0 * eb_[x], up(2.0 * eb_[x]))) * (I(2.0) + I(2.0 * eb_[z], up(2.0 * eb_[z])))
            c = W / den.sqrt()
            res = c.lo if res is None else np.minimum(res, c.lo)
    return res, out
def run(target, out_json, batch=100000, maxboxes=4e8, maxdepth=60):
    """depth-first in batches (bounded memory)."""
    t0 = time.time()
    L = np.array([0.105, 0.145, 0.145, 0.145, 0.145])
    lo0 = np.array([dn(x) for x in np.exp(-L)]); hi0 = np.array([up(x) for x in np.exp(L)])
    stack = [(lo0[None, :], hi0[None, :], 0)]; closed = nout = 0; worst = 9.0; total = 0; maxd = 0
    while stack:
        lo, hi, dep = stack.pop()
        if len(lo) > batch:
            stack.append((lo[batch:], hi[batch:], dep)); lo, hi = lo[:batch], hi[:batch]
        b, out = bound(lo, hi); total += len(lo)
        ok = (~out) & (b >= target)
        closed += int(ok.sum()); nout += int(out.sum())
        if ok.any(): worst = min(worst, float(b[ok].min()))
        rest = ~(ok | out)
        if not rest.any(): continue
        if dep >= maxdepth or total > maxboxes:
            print('FAIL at depth', dep, 'total', total, flush=True); return False
        lo, hi = lo[rest], hi[rest]; maxd = max(maxd, dep + 1)
        w = (np.log(hi) - np.log(lo)); k = np.argmax(w, axis=1); idx = np.arange(len(k))
        mid = np.sqrt(lo[idx, k] * hi[idx, k])
        hi1 = hi.copy(); hi1[idx, k] = mid; lo2 = lo.copy(); lo2[idx, k] = mid
        stack.append((np.concatenate([lo, lo2]), np.concatenate([hi1, hi]), dep + 1))
        if total % 5000000 < len(lo): print('processed %d closed %d out %d depth %d stack %d' % (total, closed, nout, dep, len(stack)), flush=True)
    res = dict(target=target, processed=total, closed=closed, outside=nout, min_bound=worst, maxdepth=maxd, secs=time.time() - t0, c_Theta=str(mp.nstr(CT, 20)))
    print(res, flush=True); json.dump(res, open(out_json, 'w'))
    return True
if __name__ == '__main__':
    run(float(sys.argv[1]), sys.argv[2])
