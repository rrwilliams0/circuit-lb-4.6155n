"""maximise the kill-PSD criterion F' over the RELAXED entrywise description of N (edge, MCC, D3, star lemma, RB).
Variables: d_v d_a d_b d_c d_d, e0 eua eub evc evd, hva hvb hab huc hud hcd, Hac Had Hbc Hbd (normalised Ghat). NUMERICAL."""
import sys, math, numpy as np
from scipy.optimize import minimize
rho = 2 * math.sqrt(2) / 3; mu0 = (1 + 2 * math.sqrt(2)) / 4; MTREE = 0.27658360688237347
TH = 2 * math.pi * MTREE; LB = 0.29; LBV = 0.21
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
def unpack(x):
    d = dict(v=x[0], a=x[1], b=x[2], c=x[3], dd=x[4]); e = dict(uv=x[5], ua=x[6], ub=x[7], vc=x[8], vd=x[9])
    h = dict(va=x[10], vb=x[11], ab=x[12], uc=x[13], ud=x[14], cd=x[15]); H = dict(ac=x[16], ad=x[17], bc=x[18], bd=x[19])
    return d, e, h, H
def quantities(x):
    d, e, h, H = unpack(x)
    N = {k: math.sqrt(2 + 2 * v) for k, v in e.items()}
    k = dict(a=(1 + e['uv'] + e['ua'] + h['va']) / (N['uv'] * N['ua']), b=(1 + e['uv'] + e['ub'] + h['vb']) / (N['uv'] * N['ub']),
             ab=(1 + e['ua'] + e['ub'] + h['ab']) / (N['ua'] * N['ub']), c=(1 + e['uv'] + e['vc'] + h['uc']) / (N['uv'] * N['vc']),
             dd=(1 + e['uv'] + e['vd'] + h['ud']) / (N['uv'] * N['vd']), cd=(1 + e['vc'] + e['vd'] + h['cd']) / (N['vc'] * N['vd']))
    C = {}
    for xx, nx, hx in (('a', 'ua', 'va'), ('b', 'ub', 'vb')):
        for zz, nz, hz in (('c', 'vc', 'uc'), ('d', 'vd', 'ud')):
            C[xx + zz] = (e['uv'] + h[hx] + h[hz] + H[xx + zz]) / (N[nx] * N[nz])
    return d, e, h, H, N, k, C
def crit(x, ret=False):
    d, e, h, H, N, k, C = quantities(x)
    ac, ad, bc, bd = C['ac'], C['ad'], C['bc'], C['bd']; D = ac * bd - ad * bc
    if D > 0:
        if bd >= ac: ac -= D / bd
        else: bd -= D / ac
    else:
        if bc >= ad: ad -= -D / bc
        else: bc -= -D / ad
    kk, kp = k['ab'], k['cd']
    sig = ac + ad + bc + bd; b1 = ac + ad - bc - bd; b2 = ac - ad + bc - bd
    c2 = sig**2 / (4 * (1 + kk) * (1 + kp)); g1 = b1 / (2 * math.sqrt((1 - kk) * (1 + kp))); g2 = b2 / (2 * math.sqrt((1 + kk) * (1 - kp)))
    F = c2 + g1**2 + g2**2 + g1**2 * g2**2 / c2
    if ret: return F, c2, g1, g2, kk, kp, sig, abs(D)
    return F
def cons(x):
    d, e, h, H, N, k, C = quantities(x); out = []
    # edge: e >= (rho/2)(sqrt r + 1/sqrt r)
    for key, (p, q) in dict(uv=('v', 1.0), ua=('a', 1.0), ub=('b', 1.0), vc=('c', 'v'), vd=('dd', 'v')).items():
        r = d[p] / (d[q] if isinstance(q, str) else 1.0)
        out.append(e[key] - (rho / 2) * (math.sqrt(r) + 1 / math.sqrt(r)))
    # MCC: G_xz >= (G_wx + G_wz)/sqrt2 - G_ww/2  (G_uu = 1, G_xx = d_x, G_xy = e sqrt(d_x d_y))
    G = dict(u=1.0, v=d['v'], a=d['a'], b=d['b'], c=d['c'], d=d['dd'])
    Ge = dict(uv=e['uv'] * math.sqrt(G['v']), ua=e['ua'] * math.sqrt(G['a']), ub=e['ub'] * math.sqrt(G['b']),
              vc=e['vc'] * math.sqrt(G['v'] * G['c']), vd=e['vd'] * math.sqrt(G['v'] * G['d']))
    Gh = dict(va=h['va'] * math.sqrt(G['v'] * G['a']), vb=h['vb'] * math.sqrt(G['v'] * G['b']), ab=h['ab'] * math.sqrt(G['a'] * G['b']),
              uc=h['uc'] * math.sqrt(G['c']), ud=h['ud'] * math.sqrt(G['d']), cd=h['cd'] * math.sqrt(G['c'] * G['d']))
    out += [Gh['va'] - ((Ge['uv'] + Ge['ua']) / math.sqrt(2) - G['u'] / 2), Gh['vb'] - ((Ge['uv'] + Ge['ub']) / math.sqrt(2) - G['u'] / 2),
            Gh['ab'] - ((Ge['ua'] + Ge['ub']) / math.sqrt(2) - G['u'] / 2), Gh['uc'] - ((Ge['uv'] + Ge['vc']) / math.sqrt(2) - G['v'] / 2),
            Gh['ud'] - ((Ge['uv'] + Ge['vd']) / math.sqrt(2) - G['v'] / 2), Gh['cd'] - ((Ge['vc'] + Ge['vd']) / math.sqrt(2) - G['v'] / 2)]
    out += [1 - v for v in h.values()] + [1 - v for v in e.values()]
    # D3
    for xx in 'ab':
        for zz in 'cd':
            Gxz = H[xx + zz] * math.sqrt(G[xx] * G[zz]); Guz = Gh['u' + zz]; Gvx = Gh['v' + xx]
            lo = (math.sqrt(2) / 12) * (G['u'] + G['v']) + (math.sqrt(2) / 6) * (G[xx] + G[zz])
            hi = (3 / math.sqrt(2)) * (Guz + Gvx) - (2 * math.sqrt(2) / 3) * (G['u'] + G['v']) - (math.sqrt(2) / 3) * (G[xx] + G[zz])
            out += [Gxz - lo, hi - Gxz]
    # star lemma: sum k >= 3 mu(e); RB >= M_tree
    def mu(e1, e2, e3): return ((sum(math.sqrt((1 + t) / 2) for t in (e1, e2, e3)))**2 - 3) / 6
    out.append(k['a'] + k['b'] + k['ab'] - 3 * mu(e['uv'], e['ua'], e['ub']))
    out.append(k['c'] + k['dd'] + k['cd'] - 3 * mu(e['uv'], e['vc'], e['vd']))
    out.append(sum(math.acos(min(1, v)) for v in k.values()) - TH)
    return np.array(out)
lo = np.array([math.exp(-LBV)] + [math.exp(-LB)] * 4 + [rho] * 5 + [0.75] * 6 + [0.5] * 4)
hi = np.array([math.exp(LBV)] + [math.exp(LB)] * 4 + [0.96] * 5 + [1.0] * 6 + [1.1] * 4)
# the ratio box: d_c, d_d are relative to d_v -> handle via bounds on d_c/d_v: use plain box on d_c with |log d_c| <= 0.5 and constrain in cons? keep simple: box 0.29 on d_c/d_v added as constraint
def cons2(x):
    d = unpack(x)[0]; return np.array([LB - abs(math.log(d['c'] / d['v'])), LB - abs(math.log(d['dd'] / d['v']))])
lo[3] = lo[4] = math.exp(-LB - LBV); hi[3] = hi[4] = math.exp(LB + LBV)
best = (-1, None); tried = 0
x0T = np.array([1.0] * 5 + [rho] * 5 + [5 / 6] * 6 + [1 / math.sqrt(2)] * 4)
for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 40):
    x0 = x0T + rng.normal(size=20) * np.array([0.1] * 5 + [0.004] * 5 + [0.02] * 6 + [0.05] * 4)
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
    try:
        r = minimize(lambda x: -crit(x), x0, method='SLSQP', bounds=list(zip(lo, hi)),
                     constraints=[dict(type='ineq', fun=cons), dict(type='ineq', fun=cons2)], options=dict(maxiter=400, ftol=1e-10))
    except Exception as ex:
        continue
    if r.success and cons(r.x).min() > -1e-7 and cons2(r.x).min() > -1e-7:
        tried += 1
        if -r.fun > best[0]: best = (-r.fun, r.x)
print('feasible local maxima found:', tried, ' best F\' = %.5f' % best[0])
if best[1] is not None:
    F, c2, g1, g2, kk, kp, sig, D = crit(best[1], True)
    print('F %.4f c2 %.4f g1 %.4f g2 %.4f k %.4f kp %.4f sigma %.4f |D| %.5f' % (F, c2, g1, g2, kk, kp, sig, D))
    print('x =', np.round(best[1], 4).tolist())
