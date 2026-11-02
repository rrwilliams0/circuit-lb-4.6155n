"""Item 6: (i) rule-ceiling thresholds in EDGE log-ratio coordinates (mpmath), monotonicity => box; (ii) resampling:
1000 points of N = {RB >= M_tree}: lambda_min(C), delta; 1000 kill-failure points: max RB; plus hill-climbs."""
import math, numpy as np, mpmath as mp
import oc
mp.mp.dps = 40
rho = 2*mp.sqrt(2)/3; MT_LO = mp.mpf('0.27658360687864')
def mu(e): return ((sum(mp.sqrt((1+x)/2) for x in e))**2 - 3)/6
def ecos(l): return rho*mp.cosh(l/2)        # (rho/2)(sqrt r + 1/sqrt r), r = e^l
def R(luv, lua, lub, lvc, lvd):
    mu_u = min(1, mu([ecos(luv), ecos(lua), ecos(lub)])); mu_v = min(1, mu([ecos(luv), ecos(lvc), ecos(lvd)]))
    return 3*(mp.acos(mu_u) + mp.acos(mu_v))/(2*mp.pi)
print('R(0) = %s, 2 kappa_E = %s' % (mp.nstr(R(0,0,0,0,0), 15), mp.nstr(3*mp.acos((1+2*mp.sqrt(2))/4)/mp.pi, 15)))
Tuv = mp.findroot(lambda l: R(l,0,0,0,0) - MT_LO, 0.14); Tua = mp.findroot(lambda l: R(0,l,0,0,0) - MT_LO, 0.29)
print('edge thresholds (R = M_tree lower end): uv %s, ua (= ub = vc = vd by symmetry) %s' % (mp.nstr(Tuv, 10), mp.nstr(Tua, 10)))
print('check vc direction:', mp.nstr(mp.findroot(lambda l: R(0,0,0,l,0) - MT_LO, 0.29), 10))
print('R at (0.1450,0,...) = %s ; R at (0,0.2891,0,...) = %s ; M_tree lower %s' % (mp.nstr(R(0.1450,0,0,0,0), 15), mp.nstr(R(0,0.2891,0,0,0), 15), mp.nstr(MT_LO, 15)))
print('all five ratios equal t: d_v=d_a=d_b=e^t, d_c=d_d=e^t => edges uv,ua,ub at t, vc,vd at 0:', mp.nstr(mp.findroot(lambda l: R(l,l,l,0,0) - MT_LO, 0.1), 10))
# monotonicity spot check: R nonincreasing in each |l_e| (random)
rng = np.random.default_rng(1); bad = 0
for _ in range(300):
    l = rng.uniform(-0.4, 0.4, 5); j = rng.integers(5); l2 = l.copy(); l2[j] *= 1.2
    if R(*l2) > R(*l) + mp.mpf('1e-30'): bad += 1
print('monotonicity violations in 300 random tests:', bad)
