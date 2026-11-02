"""Independent check of the U and V upper linear forms (bk19_obk.u_cost / v_cost) against the definitions.
Own code: the 123-type points, Gram matrices, the u-star / v-star / mirror maps, the f, q, q*, q^ref, F_c, M*, nu_c, U_L,
R_v and R_v^slice in ORIGINAL coordinates, own slice coordinates z, own re-implementation of the upper form.
The engine (byte copy of S\\\src, md5-identical to S\\the and S\\the) is called only to obtain its values.
Delta_u: the Kill2 (eps 0.02, lamM 0.6), the definition Theorem SEP uses (bk14_d2 / bk14_fast, read-only use).
modes:  point <seed> <n>      per-star comparison (U all 16 x 2 x 3 forms, V), own vs engine, form >= true
        e2e <seed> <n>        end-to-end Theorem BOUND with the engine's forms at full 123-type points
        sym <seed> <n>        Lemma SYM: x vs the a<->b swap
        box <seed> <n>        T1 enclosures of the engine's u/v cost vectors on small boxes vs own values at inner points
"""
import os, sys, math, itertools, json
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'src'); sys.path.insert(0, SRC)
os.environ['BK19_TARGETS'] = '__none__.json'
import numpy as np
import bk19_orun as R
import bk19_obk as O, bk19_lb2 as L2, bk16_model as M, bk16_bnb as BB
from bk13_lb import PU as EPU
import bk14_fast as F14, bk14_d2 as D2, bk11_core as C11

# ------------------------------------------------------------------ own constants (floats; mpmath not needed at 1e-9)
S2 = math.sqrt(2.0); RHO = 2 * S2 / 3; MU0 = (1 + 2 * S2) / 4
EMAXT = 0.9526712; N0 = math.sqrt(2 + 2 * RHO); NMAX = math.sqrt(2 + 2 * EMAXT)
RST = (1 / N0 + 1 / NMAX) / 2; EPR = (1 / N0 - 1 / NMAX) / 2
CNLO, CNHI = 1 / NMAX ** 3, 1 / N0 ** 3
FBAR = 0.48287; KAPPA = 0.08077294626; LBOX = 0.28903
GHV = {1: 2 ** -0.5, -1: 2 ** 0.5, 0: 1.0}
GH = {1: 1.0, -1: 3.0, 0: 0.5 * (3 / S2 + 2 * S2)}
g = lambda t: 2.0 ** (-t / 2)

# own profile list (u-star): (t_v, t_a, t_b) with min = -1, plus (1,1,1)
PU = [p for p in itertools.product((-1, 0, 1), repeat=3) if min(p) == -1] + [(1, 1, 1)]
assert PU == list(EPU), 'profile order differs from the engine'
NP = 20
K = np.array([p[0] for p in PU]); TAv = np.array([p[1] for p in PU]); TBv = np.array([p[2] for p in PU])
SWAP = np.array([PU.index((p[0], p[2], p[1])) for p in PU])          # a <-> b
P14 = np.array([PU.index(tuple(p)) for p in F14.PU])                 # bk14 order -> own order
assert len(F14.PU) == 20
KILL2 = D2.Kill2(eps=0.02, lamM=0.6)


def classes(x):
    return np.array([x[K == k].sum() for k in (-1, 0, 1)])


def zof(x):
    """own slice coordinates of a u-star x (sum 1): (alpha, lam, dp, dm)."""
    m1, m0, p1 = classes(x)
    al = 2 * m1 / (2 * m1 + p1); la = m0 * (2 - al) / (1 + m0 * (1 - al))
    da = (x * 2.0 ** (-TAv)).sum(); db = (x * 2.0 ** (-TBv)).sum()
    tau = 2.5 - 1.5 * m1
    return np.array([al, la, da + db - tau, da - db])


def gram_u(x):
    gv = np.stack([np.ones(NP), g(K), g(TAv), g(TBv)], 1)
    return (gv.T * x) @ gv                                             # (u, v, a, b)


def ustar(x, L):
    """the u-star quantities from the Gram matrix (own)."""
    G = gram_u(x); V = G[1, 1]; nv = V ** -0.5; na = G[2, 2] ** -0.5; nb = G[3, 3] ** -0.5
    eua = G[0, 2] * na; eub = G[0, 3] * nb                            # G_uu = 1
    ra, rb = (2 + 2 * eua) ** -0.5, (2 + 2 * eub) ** -0.5
    Pa = 1 + na * g(TAv); Pb = 1 + nb * g(TBv)
    f = ra * Pa - rb * Pb
    Fc = {c: (x * f)[K == c].sum() for c in (1, -1, 0)}
    qs = {c: nv * RST * (2 * GHV[c] + GH[c]) for c in (1, -1, 0)}
    nu = {c: nv * (2 * GHV[c] + GH[c]) * EPR + 0.2932 * nv * GH[c] / 2 * L for c in (1, -1, 0)}
    Ms = sum(qs[c] * Fc[c] for c in Fc)
    return dict(G=G, V=V, nv=nv, na=na, nb=nb, eua=eua, eub=eub, ra=ra, rb=rb, Pa=Pa, Pb=Pb, f=f, Fc=Fc, qs=qs, nu=nu, Ms=Ms)


def delta_true(x):
    m14 = x[P14]                                                       # bk14 order
    return KAPPA * KILL2.delta_u(F14.scal(m14), classes(x))


def true_s(x, L, s):
    u = ustar(x, L)
    return s[0] * u['Ms'] + s[1] * u['nu'][1] * u['Fc'][1] + s[2] * u['nu'][-1] * u['Fc'][-1] + s[3] * u['nu'][0] * u['Fc'][0]


def UL_true(x, L):
    u = ustar(x, L)
    return abs(u['Ms']) + sum(abs(u['Fc'][c]) * u['nu'][c] for c in u['Fc']) + delta_true(x)


def vstar_of_mirror(xs):
    """the v-star (profiles (t_v, t_c, t_d), u-normalisation) whose mirror is the u-star xs."""
    Vs = (xs * 2.0 ** (-K)).sum()
    prof = [(-k, a - k, b - k) for (k, a, b) in PU]
    mv = xs * 2.0 ** (-K) / Vs
    return prof, mv


def Rv_slice_orig(xs):
    """the check R_v^slice in ORIGINAL v-star coordinates (own), and the true R_v (actual r_c, r_d), and r ranges ok."""
    prof, mv = vstar_of_mirror(xs)
    tv = np.array([p[0] for p in prof]); tc = np.array([p[1] for p in prof]); td = np.array([p[2] for p in prof])
    gv = np.stack([np.ones(NP), g(tv), g(tc), g(td)], 1)
    G = (gv.T * mv) @ gv                                               # (u, v, c, d); G_uu = sum mv = 1
    Gvv, Gcc, Gdd = G[1, 1], G[2, 2], G[3, 3]
    nv = Gvv ** -0.5; elc = (Gcc / Gvv) ** -0.5; eld = (Gdd / Gvv) ** -0.5      # e^{-l_c/2}, e^{-l_d/2}
    lc, ld = math.log(Gcc / Gvv), math.log(Gdd / Gvv)
    ghat = np.array([GH[int(t)] for t in tv])
    X = g(tc) - ghat / 2; Y = g(td) - ghat / 2
    best = np.zeros(NP)
    for rc in (1 / NMAX, 1 / N0):
        for rd in (1 / NMAX, 1 / N0):
            best = np.maximum(best, np.abs(nv * (rc * elc * X + rd * eld * Y)))
    Rs = FBAR * (mv * best).sum()
    evc = G[1, 2] / math.sqrt(Gvv * Gcc); evd = G[1, 3] / math.sqrt(Gvv * Gdd)
    rc, rd = (2 + 2 * evc) ** -0.5, (2 + 2 * evd) ** -0.5
    # q(pi') - q^ref (the 1.1), with the actual r's
    nc, nd = Gcc ** -0.5, Gdd ** -0.5
    q = rc * (nv * g(tv) + nc * g(tc)) + rd * (nv * g(tv) + nd * g(td))
    ghv = np.array([GHV[int(t)] for t in tv])
    qref = nv * (ghv * (rc + rd) + (ghat / 2) * (elc * rc + eld * rd))
    Rtrue = FBAR * (mv * np.abs(q - qref)).sum()
    # fine-grid sup over the r-box (corner claim)
    grid = np.linspace(1 / NMAX, 1 / N0, 41)
    sup = np.zeros(NP)
    for a in grid:
        for b in grid:
            sup = np.maximum(sup, np.abs(nv * (a * elc * X + b * eld * Y)))
    Rgrid = FBAR * (mv * sup).sum()
    rok = (1 / NMAX <= rc <= 1 / N0) and (1 / NMAX <= rd <= 1 / N0)
    return dict(Rs=Rs, Rtrue=Rtrue, Rgrid=Rgrid, rok=rok, lc=lc, ld=ld, nv=nv)


# ------------------------------------------------------------------ engine wrappers
def mid(a): return 0.5 * (np.asarray(a.lo, float) + np.asarray(a.hi, float))


def eng_box(z, w=None):
    z = np.asarray(z, float)
    lo, hi = (z, z) if w is None else (z - w, z + w)
    return L2.Box(lo, hi, (None, 0.0025))


def ranges_at(Q, x, rng=None, wide=0.0):
    rg = {}; cache = {}
    for key in tuple(M.RANGE_KEYS) + O.EXTRA_U:
        fv = O.functional_ext(Q, key, cache)
        v = float(mid(fv.v) @ x)
        if rng is None or wide == 0.0:
            e1 = e2 = 1e-12 * (1 + abs(v))
        else:
            e1, e2 = rng.uniform(0, wide, 2)
        rg[key] = (v - e1, v + e2)
    return rg


def eng_val(cost, const, x):
    return float(mid(cost.v) @ x + mid(const.v))


def eng_U_forms(Q, rg, x, L):
    out = {}
    uv = O.u_vectors(Q)
    for ch0 in BB.CHOICES:
        ch = dict(ch0, R=[1.0] + [0.0] * 8)
        cK, kK = O.dub(Q, rg, ch); dubv = eng_val(cK, kK, x)
        for s in itertools.product((1.0, -1.0), repeat=4):
            for th in (1.0, 0.0, 0.5):
                c, k = O.u_cost(Q, rg, ch, s, L, 0.0, th, uv)
                out[(ch0['xs'], s, th)] = (-eng_val(c, k, x), dubv, c)
    return out


def eng_V_form(Q, rg, x):
    ch = dict(BB.CHOICES[0], R=[1.0] + [0.0] * 8)
    c, k = O.v_cost(Q, rg, ch, 0.0, 0.0, 0.0)
    cK, kK = O.dub(Q, rg, ch)
    return -eng_val(c, k, x), eng_val(cK, kK, x)


def own_U_form(Q, rg, x, L, s, th, dubv, zown):
    """independent re-implementation of the upper linear form (the 4.3 / bk19_obk docstring), from own slice quantities."""
    al, la, dp, dm = zown
    q_ = 1 - la; S = q_ * (2 - al) + la; m1 = al * q_ / S; m0 = la / S; p1 = 2 * (1 - al) * q_ / S
    V = 2 * m1 + m0 + p1 / 2; nv = V ** -0.5
    tau = 2.5 - 1.5 * m1; da = (tau + dp + dm) / 2; db = (tau + dp - dm) / 2; sa, sb = da ** -0.5, db ** -0.5
    Pa = 1 + sa * g(TAv); Pb = 1 + sb * g(TBv); D = Pa - Pb; Sp = (Pa + Pb) / 2
    sc = np.array([{1: s[1], -1: s[2], 0: s[3]}[int(k)] for k in K])
    qs = np.array([nv * RST * (2 * GHV[int(k)] + GH[int(k)]) for k in K])
    nu = np.array([nv * (2 * GHV[int(k)] + GH[int(k)]) * EPR + 0.2932 * nv * GH[int(k)] / 2 * L for k in K])
    kap = s[0] * qs + sc * nu
    ea = (max(rg['eA'][0], RHO), min(rg['eA'][1], EMAXT)); eb = (max(rg['eB'][0], RHO), min(rg['eB'][1], EMAXT))
    r = lambda e: (2 + 2 * e) ** -0.5
    rlo = (r(ea[1]) + r(eb[1])) / 2; rhi = (r(ea[0]) + r(eb[0])) / 2; rmid = (rlo + rhi) / 2; rrad = (rhi - rlo) / 2
    DM = EMAXT - RHO
    Dl = min(DM, max(abs(rg['dl'][0]), abs(rg['dl'][1])))
    xr = rg['dl'] if s[0] > 0 else (-rg['dl'][1], -rg['dl'][0]); xlo, xhi = max(xr[0], -DM), min(xr[1], DM)
    ylo, yhi = rg['Ys']
    xv = s[0] * (sb * g(TBv) - sa * g(TAv)); yv = qs * Sp
    X, Yv = xv @ x, yv @ x
    mc = th * (xhi * Yv + ylo * X - xhi * ylo) + (1 - th) * (yhi * X + xlo * Yv - xlo * yhi)
    cmid = (CNLO + CNHI) / 2; crad = (CNHI - CNLO) / 2
    lin = (x * (kap * rmid * D + np.abs(kap) * rrad * np.abs(D) + nu * CNHI * Dl * Sp)).sum()
    coef = kap * rmid * D + np.abs(kap) * rrad * np.abs(D) + nu * CNHI * Dl * Sp + cmid * (th * (xhi * yv + ylo * xv) + (1 - th) * (yhi * xv + xlo * yv))
    return lin + cmid * mc + crad * Dl * max(abs(ylo), abs(yhi)) + dubv, coef


def own_V_slice_mirror(zown, x):
    """independent R_v^slice in mirror coordinates from independent z (cross-check of the engine's VX, VY, VW layout)."""
    al, la, dp, dm = zown
    q_ = 1 - la; S = q_ * (2 - al) + la; m1 = al * q_ / S; m0 = la / S; p1 = 2 * (1 - al) * q_ / S
    V = 2 * m1 + m0 + p1 / 2; w = V ** -0.5
    tau = 2.5 - 1.5 * m1; da = (tau + dp + dm) / 2; db = (tau + dp - dm) / 2; sa, sb = da ** -0.5, db ** -0.5
    X = np.array([g(a - k) - GH[-k] / 2 for (k, a, b) in PU]); Y = np.array([g(b - k) - GH[-k] / 2 for (k, a, b) in PU])
    best = np.zeros(NP)
    for rc in (1 / NMAX, 1 / N0):
        for rd in (1 / NMAX, 1 / N0):
            best = np.maximum(best, np.abs(rc * sa * X + rd * sb * Y))
    return FBAR * (x * 2.0 ** (-K) * w * best).sum(), FBAR * 2.0 ** (-K) * w * best


# ------------------------------------------------------------------ star samplers
FACE = {(-1, 1, 1): 'm', (1, -1, 1): 'p', (1, 1, -1): 'p', (0, 1, -1): 'z', (0, -1, 1): 'z'}


def rand_ustar(rng, off=None):
    al = rng.uniform(0.3, 0.7); la = rng.uniform(0, 0.25)
    q_ = 1 - la; S = q_ * (2 - al) + la; mu = {-1: al * q_ / S, 0: la / S, 1: 2 * (1 - al) * q_ / S}
    x = np.zeros(NP)
    off = rng.uniform(0, 0.3) if off is None else off
    for c in (-1, 0, 1):
        idx = [j for j in range(NP) if K[j] == c]
        fidx = [j for j in idx if PU[j] in FACE]
        oidx = [j for j in idx if PU[j] not in FACE]
        wf = rng.dirichlet(np.ones(len(fidx)) * 3.0) if len(fidx) > 1 else np.ones(1)
        o = off * rng.uniform(0, 1) if oidx else 0.0
        wo = rng.dirichlet(np.ones(len(oidx)) * 0.3) if oidx else np.zeros(0)
        x[fidx] += mu[c] * (1 - o) * wf
        if oidx: x[oidx] += mu[c] * o * wo
    return x / x.sum()


def thin_ok(x):
    u = ustar(x, 0.0); G = u['G']
    la_, lb_ = math.log(G[2, 2]), math.log(G[3, 3])
    return (RHO <= u['eua'] <= EMAXT and RHO <= u['eub'] <= EMAXT and abs(la_) <= LBOX and abs(lb_) <= LBOX)


def mirror_ustar_of_point(m):
    """u-star of m^sigma (20 masses, sum 1) and u-star of m."""
    xu = np.zeros(NP); xs = np.zeros(NP)
    for i, t in enumerate(C11.TYPES):
        _, tv, ta, tb, tc, td = t
        xu[PU.index((tv, ta, tb))] += m[i]
        xs[PU.index((-tv, tc - tv, td - tv))] += m[i] * 2.0 ** (-tv)
    return xu / xu.sum(), xs / xs.sum()


def point_true(m):
    c = C11.Cv(m); D = C11.Dcross(c)
    part = max(c[2], c[5]) if D >= 0 else max(c[3], c[4])
    delta = abs(D) / part
    b1 = abs(c[2] + c[3] - c[4] - c[5]) + delta; b2 = abs(c[2] - c[3] + c[4] - c[5]) + delta
    return b1, b2, delta, min(c[2:]), part


# ------------------------------------------------------------------ modes
def mode_point(seed, n):
    rng = np.random.default_rng(seed)
    st = dict(n=0, thin=0, maxrelU=0.0, maxrelV=0.0, minUgap=1e9, minVgap=1e9, minDUBgap=1e9, maxcoef=0.0, maxVmir=0.0,
              cornergrid=0.0, Rtrue_viol=0, rcases=0, Lmax=0.0)
    i = 0
    while st['n'] < n:
        i += 1
        x = rand_ustar(rng)
        z = zof(x)
        Gq = gram_u(x)
        if max(abs(math.log(Gq[2, 2])), abs(math.log(Gq[3, 3]))) > 0.5: continue
        wide = [0.0, 0.0, 0.02, 0.2][i % 4]
        bx = eng_box(z); Q = bx.Qp
        rg = ranges_at(Q, x, rng, wide)
        if not R.derived(rg): continue
        L = [0.0, O.Lcut(rng.uniform(0, 0.0024)), 0.57806, rng.uniform(0, 0.6)][i % 4]
        st['n'] += 1
        ok = thin_ok(x); st['thin'] += ok
        dt = delta_true(x)
        forms = eng_U_forms(Q, rg, x, L)
        for (xs_, s, th), (fe, dubv, c) in forms.items():
            fo, coef = own_U_form(Q, rg, x, L, s, th, dubv, z)
            st['maxrelU'] = max(st['maxrelU'], abs(fe - fo) / (1e-3 + abs(fe)))
            # coefficient vectors: engine cost = -(cU + cK); own coef = cU
            if xs_ == 1.0:
                cK, kK = O.dub(Q, rg, dict(BB.CHOICES[0], R=[1.0] + [0.0] * 8))
                cU_eng = -mid(c.v) - mid(cK.v)
                st['maxcoef'] = max(st['maxcoef'], float(np.max(np.abs(cU_eng - coef))))
            if ok:
                st['minUgap'] = min(st['minUgap'], fe - (true_s(x, L, s) + dt))
                st['minDUBgap'] = min(st['minDUBgap'], dubv - dt)
        # V
        fv, dubv = eng_V_form(Q, rg, x)
        rv = Rv_slice_orig(x)
        rmir, _ = own_V_slice_mirror(z, x)
        st['maxrelV'] = max(st['maxrelV'], abs((fv - dubv) - rv['Rs']) / (1e-6 + rv['Rs']))
        st['maxVmir'] = max(st['maxVmir'], abs(rmir - rv['Rs']) / (1e-6 + rv['Rs']))
        st['cornergrid'] = max(st['cornergrid'], rv['Rgrid'] - rv['Rs'])
        if rv['rok'] and max(abs(rv['lc']), abs(rv['ld'])) <= LBOX:
            st['rcases'] += 1
            if rv['Rtrue'] > rv['Rs'] * (1 + 1e-12) + 1e-15: st['Rtrue_viol'] += 1
        st['minVgap'] = min(st['minVgap'], fv - (rv['Rs'] + dt))
    print('POINT seed %d tried %d' % (seed, i), json.dumps(st, default=float), flush=True)


def eng_U_best(xu, L):
    z = zof(xu); bx = eng_box(z); Q = bx.Qp; rg = ranges_at(Q, xu)
    assert R.derived(rg)
    forms = eng_U_forms(Q, rg, xu, L)
    # U form = max over sign patterns of (min over the valid candidates (choice, th) of that pattern)
    best = {}
    for (xs_, s, th), (fe, dubv, c) in forms.items():
        best[s] = min(best.get(s, 1e9), fe)
    return max(best.values()), Q, rg


def eng_V_best(xs):
    z = zof(xs); bx = eng_box(z); Q = bx.Qp; rg = ranges_at(Q, xs)
    vals = []
    for ch0 in BB.CHOICES:
        ch = dict(ch0, R=[1.0] + [0.0] * 8)
        c, k = O.v_cost(Q, rg, ch, 0.0, 0.0, 0.0); vals.append(-eng_val(c, k, xs))
    return min(vals)


def rand_point(rng):
    m = C11.mtree()
    t = rng.uniform(0, float(os.environ.get('E2E_TMAX', '0.12')))
    kind = rng.integers(0, 3)
    if kind == 0:
        r = rng.dirichlet(np.ones(C11.NT) * 0.2)
    elif kind == 1:
        r = np.zeros(C11.NT); idx = rng.choice(C11.NT, 6, replace=False); r[idx] = rng.dirichlet(np.ones(6))
    else:
        r = m.copy(); r[C11.TREE_IDX] *= rng.uniform(0.5, 1.5, 4)
        idx = rng.choice(C11.NT, 3, replace=False); r[idx] += rng.uniform(0, 0.05, 3); r /= r.sum()
    return (1 - t) * m + t * r


def mode_e2e(seed, n):
    rng = np.random.default_rng(seed)
    st = dict(n=0, tried=0, viol1=0, viol2=0, minslack1=1e9, minslack2=1e9, maxb1=0.0, maxratio=0.0)
    while st['n'] < n:
        st['tried'] += 1
        m = rand_point(rng)
        b1, b2, delta, cmin, part = point_true(m)
        xu, xs = mirror_ustar_of_point(m)
        if not (thin_ok(xu) and thin_ok(xs)) or part < 0.82: continue
        uu = ustar(xu, 0.0); us = ustar(xs, 0.0)
        if max(np.abs(uu['f']).max(), np.abs(us['f']).max()) > FBAR: continue
        rv_u = Rv_slice_orig(xs); rv_s = Rv_slice_orig(xu)          # v-star of m (mirror xs); v-star of m^sigma (mirror xu)
        if not (rv_u['rok'] and rv_s['rok']): continue
        # SEP's precondition (ratios in the 0.29-box) is part of thin_ok; delta <= kappa(Du + Ds) is checked too
        dsum = delta_true(xu) + delta_true(xs)
        if delta > dsum * (1 + 1e-12): print('SEP fails at a sampled point (not an engine issue)', delta, dsum); continue
        Lu = abs(rv_u['lc']) + abs(rv_u['ld']); Ls = abs(rv_s['lc']) + abs(rv_s['ld'])
        Ub, _, _ = eng_U_best(xu, Lu); Vb = eng_V_best(xs)
        Us, _, _ = eng_U_best(xs, Ls); Vu = eng_V_best(xu)
        st['n'] += 1
        s1 = Ub + Vb - b1; s2 = Us + Vu - b2
        st['minslack1'] = min(st['minslack1'], s1); st['minslack2'] = min(st['minslack2'], s2)
        st['viol1'] += s1 < 0; st['viol2'] += s2 < 0
        st['maxb1'] = max(st['maxb1'], b1); st['maxratio'] = max(st['maxratio'], b1 / (Ub + Vb), b2 / (Us + Vu))
    print('E2E seed %d' % seed, json.dumps(st, default=float), flush=True)


def e2e_one(m):
    b1, b2, delta, cmin, part = point_true(m)
    xu, xs = mirror_ustar_of_point(m)
    hyp = dict(thin=bool(thin_ok(xu) and thin_ok(xs)), part=part >= 0.82)
    uu = ustar(xu, 0.0); us = ustar(xs, 0.0)
    hyp['fbar'] = bool(max(np.abs(uu['f']).max(), np.abs(us['f']).max()) <= FBAR)
    rv_u = Rv_slice_orig(xs); rv_s = Rv_slice_orig(xu)
    hyp['rok'] = bool(rv_u['rok'] and rv_s['rok'])
    hyp['sep'] = bool(delta <= (delta_true(xu) + delta_true(xs)) * (1 + 1e-12))
    Lu = abs(rv_u['lc']) + abs(rv_u['ld']); Ls = abs(rv_s['lc']) + abs(rv_s['ld'])
    Ub, _, _ = eng_U_best(xu, Lu); Vb = eng_V_best(xs); Us, _, _ = eng_U_best(xs, Ls); Vu = eng_V_best(xu)
    return b1, b2, Ub + Vb, Us + Vu, hyp


def mode_adv(seed, n):
    """end-to-end at the stored adversarial maximisers (opt_*.npy, sorted-tuple order)."""
    import glob
    order = sorted(C11.TYPES)
    fs = sorted(glob.glob(os.path.join(HERE, 'opt', 'opt_*.npy')))
    for fn in fs:
        X = np.load(fn)
        X = X.reshape(-1, X.shape[-1]) if X.ndim > 1 else X.reshape(1, -1)
        for row in X[:5]:
            if row.shape[0] != 123: print('skip', os.path.basename(fn), row.shape); break
            m = np.zeros(C11.NT)
            for i, t in enumerate(order): m[C11.TIDX[t]] = row[i]
            m = np.maximum(m, 0); m /= m.sum()
            b1, b2, A, A2, hyp = e2e_one(m)
            print('ADV %s b1 %.5f <= %.5f (%s)  b2 %.5f <= %.5f (%s)  hyp %s' % (os.path.basename(fn), b1, A, b1 <= A, b2, A2, b2 <= A2, hyp), flush=True)


def elin_eng(Q, x):
    return float((mid(Q['nsc'].v) + mid(Q['nsg'].v) @ x) / 3 - MU0)


def mode_sym(seed, n):
    rng = np.random.default_rng(seed)
    st = dict(n=0, maxdz=0.0, maxdE=0.0, maxdUL=0.0, maxdV=0.0, maxdD=0.0, maxdRs=0.0, maxdUform=0.0, maxdVform=0.0,
              maxdnsg=0.0, setrows=0.0, Elin_le_E=0, minEgap=1e9)
    for _ in range(n):
        x = rand_ustar(rng); y = x[SWAP]
        z = zof(x); zy = zof(y)
        st['maxdz'] = max(st['maxdz'], float(np.max(np.abs(zy - np.array([z[0], z[1], z[2], -z[3]])))))
        L = rng.uniform(0, 0.57806)
        st['maxdUL'] = max(st['maxdUL'], abs(UL_true(x, L) - UL_true(y, L)))
        st['maxdD'] = max(st['maxdD'], abs(delta_true(x) - delta_true(y)))
        st['maxdRs'] = max(st['maxdRs'], abs(Rv_slice_orig(x)['Rs'] - Rv_slice_orig(y)['Rs']))
        # engine at z and at the exactly mirrored z' = (al, la, dp, -dm)
        zp = np.array([z[0], z[1], z[2], -z[3]])
        bx = eng_box(z); bxp = eng_box(zp); Q, Qp = bx.Qp, bxp.Qp
        Ex, Ey = elin_eng(Q, x), elin_eng(Qp, y)
        st['maxdE'] = max(st['maxdE'], abs(Ex - Ey))
        st['maxdnsg'] = max(st['maxdnsg'], float(np.max(np.abs(mid(Qp['nsg'].v)[SWAP] - mid(Q['nsg'].v)))))
        # slice rows: class sums, G_aa, G_bb, OFF
        A = lambda Qq, xx: np.array([classes(xx)[0], classes(xx)[1], classes(xx)[2], (xx * 2.0 ** (-TAv)).sum(),
                                     (xx * 2.0 ** (-TBv)).sum(), M.OFF @ xx])
        ra = A(Q, x); rb = A(Qp, y)
        st['setrows'] = max(st['setrows'], float(np.max(np.abs(rb[[0, 1, 2, 4, 3, 5]] - ra))))
        # engine forms with point ranges at each star
        rgx = ranges_at(Q, x); rgy = ranges_at(Qp, y)
        if R.derived(rgx) and R.derived(rgy):
            fx = eng_U_forms(Q, rgx, x, L); fy = eng_U_forms(Qp, rgy, y, L)
            for (xs_, s, th), v in fx.items():
                sn = tuple(-a for a in s)
                st['maxdUform'] = max(st['maxdUform'], abs(v[0] - fy[(xs_, sn, th)][0]))
                st['maxdUnoDUB'] = max(st.get('maxdUnoDUB', 0.0), abs((v[0] - v[1]) - (fy[(xs_, sn, th)][0] - fy[(xs_, sn, th)][1])))
                if xs_ == 1.0: st['maxdUform_xs1'] = max(st.get('maxdUform_xs1', 0.0), abs(v[0] - fy[(xs_, sn, th)][0]))
            vx_, vy_ = eng_V_form(Q, rgx, x), eng_V_form(Qp, rgy, y)
            st['maxdVform'] = max(st['maxdVform'], abs(vx_[0] - vy_[0])); st['maxdVnoDUB'] = max(st.get('maxdVnoDUB', 0.0), abs((vx_[0] - vx_[1]) - (vy_[0] - vy_[1])))
        # E^lin <= E (Lemma L), own E from the key correlations of the u-star
        Gm = gram_u(x); d = np.sqrt(np.diag(Gm)); H = Gm / np.outer(d, d)
        kc = lambda e1, e2, h: (1 + e1 + e2 + h) / (2 * math.sqrt((1 + e1) * (1 + e2)))
        Et = (kc(H[0, 1], H[0, 2], H[1, 2]) + kc(H[0, 1], H[0, 3], H[1, 3]) + kc(H[0, 2], H[0, 3], H[2, 3])) / 3 - MU0
        st['minEgap'] = min(st['minEgap'], Et - Ex); st['Elin_le_E'] += Et >= Ex
        st['n'] += 1
    print('SYM seed %d' % seed, json.dumps(st, default=float), flush=True)


def mode_box(seed, n):
    """engine T1 enclosures on small boxes contain own values at inner points (U coefficients of cU, V's cV, nsg)."""
    rng = np.random.default_rng(seed)
    st = dict(n=0, outU=0, outV=0, outN=0, maxexcess=0.0)
    for _ in range(n):
        x = rand_ustar(rng); z = zof(x)
        w = np.array([rng.uniform(0, 0.01), rng.uniform(0, 0.005), rng.uniform(0, 0.02), rng.uniform(0, 0.02)])
        bx = eng_box(z, w); Qb = bx.Qb
        rg = ranges_at(bx.Qp, x, rng, 0.05)
        if not R.derived(rg): continue
        L = rng.uniform(0, 0.57806); s = tuple(rng.choice([1.0, -1.0], 4)); th = float(rng.choice([0.0, 0.5, 1.0]))
        ch = dict(BB.CHOICES[0], R=[1.0] + [0.0] * 8)
        c, k = O.u_cost(Qb, rg, ch, s, L, 0.0, th, O.u_vectors(Qb)); cK, kK = O.dub(Qb, rg, ch)
        lo = -np.asarray(c.v.hi) - np.asarray(cK.v.hi); hi = -np.asarray(c.v.lo) - np.asarray(cK.v.lo)   # encloses cU
        cv, kv = O.v_cost(Qb, rg, ch, 0.0, 0.0, 0.0)
        vlo = -np.asarray(cv.v.hi) - np.asarray(cK.v.hi); vhi = -np.asarray(cv.v.lo) - np.asarray(cK.v.lo)
        def own_vecs(zi):
            _, coef = own_U_form(None, rg, np.ones(NP) / NP, L, s, th, 0.0, zi)
            _, vco = own_V_slice_mirror(zi, np.ones(NP) / NP)
            al, la, dp, dm = zi; q_ = 1 - la; S = q_ * (2 - al) + la; m1 = al * q_ / S; m0 = la / S; p1 = 2 * (1 - al) * q_ / S
            V = 2 * m1 + m0 + p1 / 2; ww = V ** -0.5; tau = 2.5 - 1.5 * m1
            da = (tau + dp + dm) / 2; db = (tau + dp - dm) / 2; sa, sb = da ** -0.5, db ** -0.5
            D0 = 2 * RHO + 2; k8 = 2 / D0 - 8 / D0 ** 2; sab = sa * g(TAv) + sb * g(TBv)
            nsg = sab * k8 + sab * ww * g(K) / D0 + sa * sb * np.array([g(a) * g(b) if a + b != 0 else 1.0 for a, b in zip(TAv, TBv)]) / D0
            return coef, vco, nsg
        gU = (-np.asarray(c.g.hi) - np.asarray(cK.g.hi), -np.asarray(c.g.lo) - np.asarray(cK.g.lo))
        gV = (-np.asarray(cv.g.hi) - np.asarray(cK.g.hi), -np.asarray(cv.g.lo) - np.asarray(cK.g.lo))
        gN = (np.asarray(Qb['nsg'].g.lo), np.asarray(Qb['nsg'].g.hi))
        for _p in range(6):
            zi = z + rng.uniform(-0.98, 0.98, 4) * w
            coef, vco, nsg = own_vecs(zi)
            tol = 1e-12
            eU = np.maximum(lo - coef, coef - hi).max(); eV = np.maximum(vlo - vco, vco - vhi).max()
            eN = np.maximum(np.asarray(Qb['nsg'].v.lo) - nsg, nsg - np.asarray(Qb['nsg'].v.hi)).max()
            st['outU'] += eU > tol; st['outV'] += eV > tol; st['outN'] += eN > tol
            st['maxexcess'] = max(st['maxexcess'], eU, eV, eN)
            # gradient enclosures (central differences inside the box)
            for d_ in range(4):
                if w[d_] < 1e-6: continue
                h = min(1e-6, 0.01 * w[d_])
                e_ = np.zeros(4); e_[d_] = h
                a1, b1, c1 = own_vecs(zi + e_); a2, b2, c2 = own_vecs(zi - e_)
                for (glo_, ghi_), fd in ((gU, (a1 - a2) / (2 * h)), (gV, (b1 - b2) / (2 * h)), (gN, (c1 - c2) / (2 * h))):
                    tl = 1e-5 * (1 + np.abs(fd))
                    ex = np.maximum(glo_[d_] - fd - tl, fd - ghi_[d_] - tl).max()
                    st['gout'] = st.get('gout', 0) + int(ex > 0); st['gmax'] = max(st.get('gmax', -1e9), float(ex))
        st['n'] += 1
    print('BOX seed %d' % seed, json.dumps(st, default=float), flush=True)


def mode_const(seed, n):
    """engine constants against 40-digit values (mpmath), outward directions."""
    import mpmath as mp
    mp.mp.dps = 40
    rho = 2 * mp.sqrt(2) / 3; mu0 = (1 + 2 * mp.sqrt(2)) / 4
    cth = mp.mpf('0.95957023420505315737'); emax = mp.mpf('0.95267117628053096636')   #  (40 digits)
    n0 = mp.sqrt(2 + 2 * rho); nmax = mp.sqrt(2 + 2 * emax)
    out = []
    def chk(name, eng, exact, direction):          # direction 'ge': engine value must be >= exact; 'le': <=
        ok = (mp.mpf(eng) >= exact) if direction == 'ge' else (mp.mpf(eng) <= exact)
        out.append((name, float(eng), float(exact), direction, bool(ok)))
    chk('R_LO <= 1/N_max', O.R_LO, 1 / nmax, 'le'); chk('R_HI >= 1/N0', O.R_HI, 1 / n0, 'ge')
    chk('CN_LO <= 1/N_max^3', O.CN_LO, 1 / nmax ** 3, 'le'); chk('CN_HI >= 1/N0^3', O.CN_HI, 1 / n0 ** 3, 'ge')
    chk('DMAXG >= e_max - rho', O.DMAXG, emax - rho, 'ge'); chk('EMAX.lo >= e_max', float(O.EMAX.lo), emax, 'ge')
    chk('EPSR >= (1/N0 - 1/N_max)/2', float(O.EPSR.lo), (1 / n0 - 1 / nmax) / 2, 'ge')
    chk('BJ.lo >= c_Theta - mu0', float(O.BJ.lo), cth - mu0, 'ge')
    chk('CTH.hi >= c_Theta', float(M.CTH.hi), cth, 'ge')
    chk('WJ.hi <= 0.9714701', float(O.WJ.hi), mp.mpf('0.9714701'), 'le')
    chk('KILL >= 1/(N0^4 0.82)', float(M.KILL.lo), 1 / (n0 ** 4 * mp.mpf('0.82')), 'ge')
    chk('REF.lo >= e^{0.144515}/(2 N0)', float(O.REF.lo), mp.e ** (mp.mpf('0.28903') / 2) / (2 * n0), 'ge')
    chk('FBAR = 0.48287', float(O.FBAR.lo), mp.mpf('0.48287'), 'ge')
    chk('GH0 hi >= 7 sqrt2/4', float(np.asarray(O.GH[0].hi)), 7 * mp.sqrt(2) / 4, 'ge'); chk('GH0 lo <= 7 sqrt2/4', float(np.asarray(O.GH[0].lo)), 7 * mp.sqrt(2) / 4, 'le')
    # Lcut(E) >= L(E) = min(0.57806, sqrt(2 (B - E)/(w c_Psi))) with B = c_Theta - mu0, w = 0.9714701
    B = cth - mu0; worst = 1e9
    for i in range(251):
        E = mp.mpf(i) * mp.mpf('0.00001')
        if E > B: break
        Lt = min(mp.mpf('0.57806'), mp.sqrt(2 * (B - E) / (mp.mpf('0.9714701') * mp.mpf('0.029385'))))
        worst = min(worst, float(mp.mpf(O.Lcut(float(E))) - Lt))
    out.append(('min Lcut(E) - L(E) over E = 0..B step 1e-5', worst, 0.0, 'ge', worst >= 0))
    for o in out: print('CONST', o)
    print('CONST all ok:', all(o[-1] for o in out), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'const': mode_const(0, 0); sys.exit()
    mode, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    dict(point=mode_point, e2e=mode_e2e, sym=mode_sym, box=mode_box, adv=mode_adv)[mode](seed, n)
