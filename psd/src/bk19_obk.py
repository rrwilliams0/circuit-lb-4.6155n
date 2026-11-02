"""bk19_obk.py : box evaluation for O-the-1 (the Theorem BOUND, Lemma CELL in the check form), banded form.

For a u-star slice box (pm coordinates z = (alpha, lam, dp, dm)), an off-face level [s_lo, s_hi] and an E^lin band
[e_lo, e_hi] (band k = [k 1e-4, (k+1) 1e-4]; band 0 has no lower row), certify on S_band(z) (bk19_lb2):

  family U, band k:  for every sign pattern s = (s0, s_1, s_-1, s_0) in {+-1}^4,
        s0 M* + sum_c s_c |.|-free F_c (a_c + b_c L(E_k)) + kappa Delta_u  <=  TU[k]          (UPPER bound, linear form)
     -> Lemma CELL's Ubar_k <= TU[k] if TU is nondecreasing (bands partition the stars by E^lin; L decreases in E).
  family V, band k (on MIRROR u-star slices, i.e. v-stars):  R_v^slice + kappa Delta_u <= t0_k + t1_k (E^lin - e_lo_k)
     (band 0: <= t0_0) -> Vbar(c) <= TV(c) for the piecewise linear increasing TV.

Linear upper forms (all coefficients T1 functions of z, or box constants from certified ranges):
  F_c = rbar D_c + (r_a - r_b) S+_c,  D_c = sum_{U_c} m (P_a - P_b),  S+_c = sum_{U_c} m (P_a + P_b)/2,
  rbar = (r_a + r_b)/2 in [rbar_lo, rbar_hi] (from certified ranges of e_ua = sa GA.m, e_ub = sb GB.m, intersected with
  THIN's [rho, e_max]; r(e) = (2 + 2e)^{-1/2} decreasing), r_a - r_b = c_N delta, delta = e_ub - e_ua, c_N in
  [1/N_max^3, 1/N0^3];  kappa_c := s0 q*_c + s_c nu_c,  nu_c = a_c + b_c L(E_k) >= 0;
  sum_c kappa_c F_c <= sum_j m_j [kappa_j rmid D_j + |kappa_j| rrad |P_a - P_b|_j + nu_j cNhi Delta S+_j]
                      + s0 cNmid McCormick(delta, Y*) + cNrad Delta |Y*|max,   Y* = sum_c q*_c S+_c,
  Delta = max |delta| on the box (certified range, intersected with [-(e_max - rho), e_max - rho]).
  Delta_u <= DUB (Lemma REL; bk16_model.main_cost's kill part).  R_v^slice in mirror coordinates:
  m_v(pi') = 2^{-k} m^sigma(pi^sigma)/V^sigma, n_v = (V^sigma)^{1/2}, e^{-l_c/2} = sa^sigma, so
  R_v^slice = fbar sum_j m_j 2^{-k_j} (V)^{-1/2} max_{r_c, r_d corners} |r_c sa X_j + r_d sb Y_j|,  X_j = g_c - ghat/2.
"""
import os, math
import numpy as np
from ivarith import IA, dn, up
from bk13_lb import T1, I, SQ2, ISQ2, RHO, MU0, NP, KV, TAv, TBv, GA, GB, CLS, VMASK, tvec, tsum
import bk16_model as M
import bk16_bnb as BB
import bk19_lb2 as L2

# ------------------------------------------------------------------ constants (outward)
EMAX = I('0.9526712')                       # THIN: every internal edge cosine <= 0.9526712 on N
N0 = (RHO * 2.0 + 2.0).sqrt(); NMAX = (EMAX * 2.0 + 2.0).sqrt()
R_LO = float((NMAX.inv()).lo); R_HI = float((N0.inv()).hi)
RSTAR = (N0.inv() + NMAX.inv()) * 0.5; EPSR = IA(float(((N0.inv() - NMAX.inv()) * 0.5).hi))
CN_LO = float((NMAX * NMAX * NMAX).inv().lo); CN_HI = float((N0 * N0 * N0).inv().hi)
DMAXG = float((EMAX - RHO).hi)
FBAR = I('0.48287')                          # Lemma FBAR (constant interval-certified by bk19_fbar.py)
BJ = I('0.0024634531')                       # >= c_Theta - mu0 = 0.00246345302
WJ = I('0.97147'); CPSI = I('0.029385'); LMAX = I('0.57806'); REF = I('0.2932')
GHV = {1: ISQ2, -1: SQ2, 0: IA(1.0)}
GH = {1: IA(1.0), -1: IA(3.0), 0: (SQ2 * 1.5 + SQ2 * 2.0) * 0.5}     # (3/sqrt2 + 2 sqrt2)/2 = (1.5 sqrt2 + 2 sqrt2)/2
KILL = None                                  # set from bk17_cfg at import of the driver


def Lcut(Ek):
    """L(E) = min(0.57806, sqrt(2 (B - E)/(w c_Psi))) rounded UP (larger L = weaker, valid)."""
    x = (IA(float(BJ.hi)) - IA(Ek)) * 2.0
    x = IA(float(x.hi)) / IA(float(WJ.lo)) / IA(float(CPSI.lo))
    return min(float(LMAX.hi), float(x.sqrt().hi))


def _cvec(f):
    return IA(np.array([float(np.asarray(f(int(KV[j]), int(TAv[j]), int(TBv[j])).lo)) for j in range(NP)]),
              np.array([float(np.asarray(f(int(KV[j]), int(TAv[j]), int(TBv[j])).hi)) for j in range(NP)]))


GQ = _cvec(lambda c, a, b: GHV[c] * 2.0 + GH[c])                    # 2 ghat_v(c) + ghat(c)
GHO = _cvec(lambda c, a, b: GH[c] * 0.5)                               # ghat(c)/2
MASK = {c: np.array([1.0 if KV[j] == c else 0.0 for j in range(NP)]) for c in (1, -1, 0)}


def gpow(t):  # 2^{-t/2} enclosure for integer t in [-2, 2]
    return {-2: IA(2.0), -1: SQ2, 0: IA(1.0), 1: ISQ2, 2: IA(0.5)}[int(t)]


# V-family profile data: mirror profile (k, a', b') = (-t_v, t_c - t_v, t_d - t_v): t_v = -k, t_c = a' - k, t_d = b' - k.
VX = _cvec(lambda k, a, b: gpow(a - k) - GH[-k] * 0.5)
VY = _cvec(lambda k, a, b: gpow(b - k) - GH[-k] * 0.5)
VW = _cvec(lambda k, a, b: FBAR * {1: IA(0.5), 0: IA(1.0), -1: IA(2.0)}[k])     # fbar 2^{-k}


def vtmax(lst):
    """elementwise max of vector T1s (value hull; gradient hull of the candidates that can be active)."""
    los = np.array([np.asarray(f.v.lo) for f in lst]); his = np.array([np.asarray(f.v.hi) for f in lst])
    mlo = los.max(axis=0); mhi = his.max(axis=0)
    act = his >= mlo[None, :]
    glo = np.min(np.where(act[:, None, :], np.array([np.asarray(f.g.lo) for f in lst]), np.inf), axis=0)
    ghi = np.max(np.where(act[:, None, :], np.array([np.asarray(f.g.hi) for f in lst]), -np.inf), axis=0)
    return T1(IA(mlo, mhi), IA(glo, ghi))


# ------------------------------------------------------------------ functionals for ranges
def u_vectors(Q):
    sa, sb, w = Q['sa'], Q['sb'], Q['w']
    eA = tvec(sa, GA); eB = tvec(sb, GB)
    Pa = eA + 1.0; Pb = eB + 1.0
    qs = tvec(w, GQ * RSTAR)                                            # q*_c per profile
    Sp = (Pa + Pb) * 0.5
    return dict(eA=eA, eB=eB, dl=eB - eA, Ys=qs * Sp, D=Pa - Pb, Sp=Sp, qs=qs)


EXTRA_U = ('eA', 'eB', 'dl', 'Ys')


def functional_ext(Q, key, cache):
    assert key in M.RANGE_KEYS or key in EXTRA_U
    if key in M.RANGE_KEYS: return M.functional(Q, key)
    if id(Q) not in cache: cache[id(Q)] = u_vectors(Q)
    return cache[id(Q)][key]


def dub(Q, rg, ch):
    """KILL * DUB as (cost T1 vector, const T1): bk16_model.main_cost with its Lfun part switched off."""
    save = M.MU0
    try:
        M.MU0 = IA(0.0)
        c, k = M.main_cost(Q, rg, ch)
    finally:
        M.MU0 = save
    return -c, -k


def u_cost(Q, rg, ch, sgn, Lk, T, th, uv):
    """cost, const of  T - [upper linear form of U]  (to be certified >= 0)."""
    s0 = sgn[0]; sc = sgn[1] * MASK[1] + sgn[2] * MASK[-1] + sgn[3] * MASK[0]
    w = Q['w']
    nu = tvec(w, GQ * EPSR) + tvec(w, GHO * REF) * Lk                   # a_c + b_c L
    kap = uv['qs'] * float(s0) + nu * IA(sc)
    rlo, rhi = rg['_rbar']; rmid = 0.5 * (rlo + rhi); rrad = up(max(rhi - rmid, rmid - rlo))
    Dl = rg['_Dl']
    cU = kap * uv['D'] * rmid + M.tabs(kap) * M.tabs(uv['D']) * rrad + nu * uv['Sp'] * up(CN_HI * Dl)
    xlo, xhi = rg['dl'] if s0 > 0 else (-rg['dl'][1], -rg['dl'][0])
    xlo = max(xlo, -DMAXG); xhi = min(xhi, DMAXG)
    ylo, yhi = rg['Ys']
    mc_c, mc_k = M.mc_upper(uv['dl'] * float(s0), xlo, xhi, uv['Ys'], ylo, yhi, th)
    cmid = 0.5 * (CN_LO + CN_HI); crad = up(max(CN_HI - cmid, cmid - CN_LO))
    cU = cU + mc_c * cmid
    kU = mc_k * cmid + IA(up(crad * Dl * max(abs(ylo), abs(yhi))))
    cK, kK = dub(Q, rg, ch)
    return -(cU + cK), (M.tconst(T) - kU) - kK


def v_cost(Q, rg, ch, t0, t1, elo):
    """cost, const of  t0 + t1 (E^lin - elo) - [R_v^slice + KILL DUB]  (mirror u-star slice)."""
    sa, sb, w = Q['sa'], Q['sb'], Q['w']
    cors = []
    for rc in (R_LO, R_HI):
        for rd in (R_LO, R_HI):
            cors.append(M.tabs(tvec(sa, VX * rc) + tvec(sb, VY * rd)))
    rho = vtmax(cors)
    cV = rho * tvec(w, VW)
    cK, kK = dub(Q, rg, ch)
    cost = -(cV + cK)
    const = M.tconst(t0) - kK
    if t1 != 0.0:
        cost = cost + Q['nsg'] * (IA(t1) / 3.0)
        const = const + (Q['nsc'] / 3.0 - MU0 - IA(elo)) * IA(t1)
    return cost, const
