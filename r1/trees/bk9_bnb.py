"""bk9_bnb.py : CERTIFIED margin-first branch and bound for (R-i') on the reduced 4-parameter region
   R4 = {theta = (x, y, tu, tv) : XMIN <= x <= y <= 1, tu, tv in [0, 0.3], S := S_cut(x, y) = sqrt(K0 - kappa (x + y)),
         C(theta) := C'(x, y, S, tu, tv) PSD, cross entries <= 1},
claim: Phi_F(C(theta)) <= M_tree on R4.  (Lemma S-RED of the report: on the cut-feasible part of R1, Phi is nonincreasing
in S at fixed (x, y, tu, tv) by lem:G(b), so the supremum sits on the cut boundary S = S_cut; the three symmetries of Lemma
DOM give tu, tv >= 0 and x <= y.)  PSD on R1 (Lemma R1a + the x = 1 analysis, Lemma DOM'):
   q_u q_v <= 1,  q_u = S^2/(2(1+x)) + 2 tu^2/(1-x)  (for x = 1: feasible only if tu = 0, then q_u = S^2/4), same for v.
Closure rules (first that applies): XY, INF-PSD (lower ends of q_u q_v > 1), INF-K (a cross entry > 1 on the box),
LOC (box inside B0, Lemma LOC-4D certified at startup), TAY2 / TAY1 (second/first-order Taylor in theta about the box
centre, unconstrained, with all enclosures outward-rounded), CORNER (lem:G(b) at the entrywise lower corner of the image
box, shrunk toward I to be PD; checked point enclosures phi_point / F_point).  Queue: largest estimated margin first."""
import os, sys, math, time, json, heapq
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
from bk8_core import *
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gt'))
import cert_gt as CG

K = consts(); KAP = K['kappa']; CC = K['c']; K0 = K['K0']; MT_LO = K['Mt_lo']
XMIN = float(((K0 - KAP - 4.0)*KAP.inv()).lo)
TMAX = 0.3
STATS = {}; TIME = {'phi': 0.0, 'grad': 0.0}
def F(z): return float(z)
def ia(lo, hi): return IA(np.float64(lo), np.float64(hi))
def stat(k): STATS[k] = STATS.get(k, 0) + 1

# ------------------------------------------------------------------ the map theta4 -> C and its derivatives (IA or float)
def S_of(x, y):
    if isinstance(x, IA) or isinstance(y, IA): return (K0 - KAP*(IA.of(x) + IA.of(y))).sqrt()
    return math.sqrt(F(K0.mid()) - F(KAP.mid())*(x + y))

def C4(th):
    x, y, tu, tv = th; S = S_of(x, y)
    return C_of([x, y, S, tu, tv])

SIGNS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]          # ac ad bc bd : (s_u, s_v)

def J4(th):
    """6x4 Jacobian dC_p/dtheta_j, theta = (x, y, tu, tv); entries IA if th is IA."""
    x, y, tu, tv = th; S = S_of(x, y)
    isI = isinstance(S, IA)
    half = S*0.5
    Sx = -(KAP*(S*2.0).inv()) if isI else -F(KAP.mid())/(2*S)
    rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    for su, sv in SIGNS:
        A = half + tu*float(su); Bv = half + tv*float(sv)
        dS = (A + Bv)*0.5
        rows.append([dS*Sx, dS*Sx, Bv*float(su), A*float(sv)])
    return rows

def H4(th):
    """six 4x4 Hessians d^2 C_p / dtheta^2 (IA entries); pair entries have zero Hessian."""
    x, y, tu, tv = th; S = S_of(x, y); half = S*0.5
    Sx = -(KAP*(S*2.0).inv()); Sxx = -(KAP*KAP*(S*S*S*4.0).inv())
    Z = IA(0.0)
    out = [[[Z]*4 for _ in range(4)], [[Z]*4 for _ in range(4)]]
    for su, sv in SIGNS:
        A = half + tu*float(su); Bv = half + tv*float(sv)
        hxx = Sx*Sx*0.5 + (A + Bv)*0.5*Sxx
        hxu = Sx*(0.5*su); hxv = Sx*(0.5*sv); huv = IA(float(su*sv))
        M = [[hxx, hxx, hxu, hxv], [hxx, hxx, hxu, hxv], [hxu, hxu, Z, huv], [hxv, hxv, huv, Z]]
        out.append(M)
    return out

def theta_T4(): t = theta_T(); return [t[0], t[1], 0.0, 0.0]

# ------------------------------------------------------------------ checked point enclosure
def phi_up(c):
    t0 = time.time()
    e = AP.phi_point(list(c), tol=1e-7)
    if e is not None: TIME['phi'] += time.time() - t0; return F(e.hi)
    try:
        if CG.chol_pd(list(c), list(c)): e = CG.F_point(list(c)); TIME['phi'] += time.time() - t0; return float(e.hi)
    except Exception: pass
    TIME['phi'] += time.time() - t0
    return None

def shrink_pd(lo):
    ev = np.linalg.eigvalsh(Cmat(lo)).min()
    t = 0.0 if ev > 0 else -ev/(1 - ev)
    for k in range(6):
        t2 = min(1.0, t*(1 + 1e-6) + 1e-12*(k + 1) + 1e-9*k)
        lo2 = np.array([F(((1.0 - t2)*IA(float(v))).lo) for v in lo])
        if AP.pd_point(list(lo2)): return lo2, t2
        t = t2*1.5 + 1e-9
    return None

def float_margin(c):
    """NUMERICAL (filter only, never enters a bound): MT_LO - Phi_F(c) in floating point; -1 if not evaluable."""
    try:
        if max(c) >= 1 or min(c) <= 0 or np.linalg.eigvalsh(Cmat(c)).min() < 1e-9: return -1.0
        return MT_LO - phiF_float(c)
    except Exception: return -1.0

def corner(lo6):
    r = shrink_pd(lo6)
    if r is None: return None
    lo2, tneed = r
    for t in sorted(set([tneed, 1e-3, 3e-3, 1e-2, 3e-2])):
        if t < tneed: continue
        lo3 = np.array([F(((1.0 - t)*IA(float(v))).lo) for v in lo6]) if t != tneed else lo2
        if not AP.pd_point(list(lo3)): continue
        if float_margin(lo3) < 1e-7: continue          # hopeless: the certified enclosure is wider than 1e-11 only
        u = phi_up(lo3)
        if u is not None:
            return 'CORNER' if u <= MT_LO else None
    return None

# ------------------------------------------------------------------ Lemma LOC-4D
def Mloc4(r0):
    thI = [IA(dn(P_T.lo - r0), up(P_T.hi + r0)), IA(dn(P_T.lo - r0), up(P_T.hi + r0)), IA(-r0, r0), IA(-r0, r0)]
    Cb = C4(thI); lo = np.array([F(z.lo) for z in Cb]); hi = np.array([F(z.hi) for z in Cb])
    assert AP.pd_box(lo, hi)
    g = AP.grad_ad(lo, hi)
    Hb = [[IA(g.g.lo[p, q], g.g.hi[p, q]) for q in range(6)] for p in range(6)]
    Hb = [[(Hb[p][q] + Hb[q][p])*0.5 for q in range(6)] for p in range(6)]
    gb = [IA(g.v.lo[p], g.v.hi[p]) for p in range(6)]
    J = J4(thI); HC = H4(thI)
    M = [[IA(0.0)]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = IA(0.0)
            for p in range(6):
                for q in range(6):
                    acc = acc + IA.of(J[p][i])*Hb[p][q]*IA.of(J[q][j])
            for p in range(6): acc = acc + gb[p]*HC[p][i][j]
            M[i][j] = acc
    mid = np.array([[F(M[i][j].mid()) for j in range(4)] for i in range(4)]); mid = 0.5*(mid + mid.T)
    rad = np.array([[F(max(M[i][j].hi - mid[i, j], mid[i, j] - M[i][j].lo)) for j in range(4)] for i in range(4)])
    lam = np.linalg.eigvalsh(mid)
    bound = lam.max() + 1e-12 + math.sqrt((rad**2).sum())*(1 + 1e-12)
    return lam, rad, bound

# ------------------------------------------------------------------ closure attempt
def try_close(lo, hi):
    if lo[0] > hi[1]: return 'XY'
    thI = [ia(lo[j], hi[j]) for j in range(4)]
    x, y, tu, tv = thI
    S2 = K0 - KAP*(x + y)
    if S2.lo <= 0: return None
    # INF-PSD: lower ends of q_u, q_v
    def q_lo(xI, tI):
        q = IA(F(S2.lo))*(IA(2.0)*(1.0 + IA(F(xI.hi)))).inv()
        if xI.lo < 1.0: q = q + (IA(F(tI.lo))*IA(F(tI.lo))*2.0)*(1.0 - IA(F(xI.lo))).inv()
        return IA(F(q.lo))
    if (q_lo(x, tu)*q_lo(y, tv)).lo > 1.0: return 'INF-PSD'
    S = S2.sqrt()
    if ((IA(F(S.lo))*0.5 + F(tu.lo))*(IA(F(S.lo))*0.5 + F(tv.lo))).lo > 1.0: return 'INF-K'
    if np.all(lo >= B0_LO) and np.all(hi <= B0_HI): return 'LOC'
    Cb = C4(thI); clo = np.array([F(z.lo) for z in Cb]); chi = np.array([F(z.hi) for z in Cb])
    pdb = False
    try: pdb = AP.pd_box(clo, chi)
    except (ArithmeticError, ZeroDivisionError): pdb = False
    if pdb:
        r = taylor(lo, hi, thI, clo, chi)
        if r is not None: return r
    return corner(clo)

def taylor(lo, hi, thI, clo, chi):
    th0 = 0.5*(lo + hi); w = 0.5*(hi - lo)
    th0I = [IA(float(t)) for t in th0]
    CI = C4(th0I); c0 = np.array([F(z.mid()) for z in CI])
    if not AP.pd_point(list(c0)): return None
    eps0 = [CI[p] - float(c0[p]) for p in range(6)]
    t0 = time.time()
    try:
        g0a = AP.grad_ad(c0, c0); gBa = AP.grad_ad(clo, chi)
    except (ArithmeticError, ZeroDivisionError):
        TIME['grad'] += time.time() - t0; return None
    TIME['grad'] += time.time() - t0
    g0 = [IA(g0a.v.lo[p], g0a.v.hi[p]) for p in range(6)]
    gB = [IA(gBa.v.lo[p], gBa.v.hi[p]) for p in range(6)]
    HB = [[IA(gBa.g.lo[p, q], gBa.g.hi[p, q]) for q in range(6)] for p in range(6)]
    HB = [[(HB[p][q] + HB[q][p])*0.5 for q in range(6)] for p in range(6)]
    u = phi_up(c0)
    if u is None: return None
    base = IA(u)
    for p in range(6): base = base + g0[p]*eps0[p]
    base = F(base.hi)
    J0 = J4(th0I)                                                  # IA (tight, exact centre)
    G = [IA(0.0)]*4
    for j in range(4):
        acc = IA(0.0)
        for p in range(6): acc = acc + IA.of(J0[p][j])*g0[p]
        G[j] = acc
    HC = H4(thI)                                                   # over the box
    # second-order remainder of C over the box: |C(theta) - C(theta0) - J0 d| <= R_p = 1/2 sum |HC_p,ij| w_i w_j
    R = np.zeros(6)
    for p in range(2, 6):
        acc = IA(0.0)
        for i in range(4):
            for j in range(4): acc = acc + HC[p][i][j].mag()*float(w[i]*w[j])
        R[p] = F((acc*0.5).hi) + F(eps0[p].mag())
    R[0] = F(eps0[0].mag()); R[1] = F(eps0[1].mag())
    Jw = np.array([F(sum((IA.of(J0[p][j]).mag()*float(w[j]) for j in range(4)), IA(0.0)).hi) for p in range(6)])
    Hmag = np.array([[F(HB[p][q].mag()) for q in range(6)] for p in range(6)])
    cross = float(Jw @ Hmag @ R) + 0.5*float(R @ Hmag @ R)
    # M = HessL + J0^T HB J0 with HessL = sum_p g0_p HC_p
    M = [[None]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = IA(0.0)
            for p in range(6): acc = acc + g0[p]*HC[p][i][j]
            for p in range(6):
                Jpi = J0[p][i]
                if isinstance(Jpi, float) and Jpi == 0.0: continue
                for q in range(6):
                    Jqj = J0[q][j]
                    if isinstance(Jqj, float) and Jqj == 0.0: continue
                    acc = acc + IA.of(Jpi)*HB[p][q]*IA.of(Jqj)
            M[i][j] = acc
    for i in range(4):
        for j in range(i + 1, 4):
            s = (M[i][j] + M[j][i])*0.5; M[i][j] = s; M[j][i] = s
    Mmid = np.array([[F(M[i][j].mid()) for j in range(4)] for i in range(4)]); Mmid = 0.5*(Mmid + Mmid.T)
    lam, V = np.linalg.eigh(Mmid)
    lneg = np.minimum(lam, 0.0) - 1e-9
    Mneg = V @ np.diag(lneg) @ V.T; Mneg = 0.5*(Mneg + Mneg.T)
    if np.linalg.eigvalsh(Mneg).max() > -5e-10: return None
    Mpos = V @ np.diag(np.maximum(lam, 0.0)) @ V.T; Mpos = 0.5*(Mpos + Mpos.T)
    slop = 0.0
    for i in range(4):
        for j in range(4):
            rij = M[i][j] - float(Mneg[i, j]) - float(Mpos[i, j])
            slop += F(rij.mag())*w[i]*w[j]*0.5
    pen = 1e-12*float(w @ w)
    for k in range(4):
        if lam[k] > 0: pen += 0.5*lam[k]*(np.abs(V[:, k]) @ w)**2
    # concave part: q(d) = G.d + 1/2 d^T Mneg d over |d_j| <= w_j; bound q(dc) + sum_j max_{d_j} grad_j (d_j - dc_j)
    Gm = np.array([F(g.mid()) for g in G])
    try: dc = np.linalg.solve(Mneg, -Gm)
    except np.linalg.LinAlgError: dc = np.zeros(4)
    dc = np.clip(dc, -w, w)
    qc = IA(0.0); grad = []
    for j in range(4):
        gj = G[j]
        for k in range(4): gj = gj + float(Mneg[j, k])*float(dc[k])
        grad.append(gj)
        qc = qc + G[j]*float(dc[j])
        for k in range(4): qc = qc + float(0.5*Mneg[j, k]*dc[j]*dc[k])
    conc = qc
    for j in range(4):
        a = grad[j]*float(w[j] - dc[j]); b = grad[j]*float(-w[j] - dc[j])
        conc = conc + IA(max(a.hi, b.hi))
    U2 = base + F(conc.hi) + slop + pen + cross
    if U2 <= MT_LO: return 'TAY2'
    # TAY1
    U1 = base
    for p in range(6): U1 += F((gB[p] - g0[p]).mag())*(Jw[p] + R[p])
    for j in range(4): U1 += F(G[j].mag())*w[j]
    for i in range(4):
        for j in range(4):
            acc = IA(0.0)
            for p in range(6): acc = acc + g0[p]*HC[p][i][j]
            U1 += 0.5*F(acc.mag())*w[i]*w[j]
    if U1 <= MT_LO: return 'TAY1'
    return None

# ------------------------------------------------------------------ queue
def margin_est(lo, hi):
    """NUMERICAL ordering key: the float margin of the (shrunk) lower corner of the image box, i.e. what CORNER would
    see; boxes certified infeasible by the cheap tests get the key 2 (processed first, closed at once)."""
    try:
        thI = [ia(lo[j], hi[j]) for j in range(4)]
        Cb = C4(thI); clo = np.array([F(z.lo) for z in Cb])
        if lo[0] > hi[1]: return 2.0
        ev = np.linalg.eigvalsh(Cmat(clo)).min()
        t = 0.0 if ev > 0 else -ev/(1 - ev) + 1e-6
        c = (1 - t)*clo
        fm = float_margin(c)
        return 2.0 if fm == -1.0 else fm            # unevaluable corner: far / infeasible box, process first (cheap)
    except Exception: return 2.0

def split_coord(lo, hi):
    th = 0.5*(lo + hi); w = hi - lo
    J = np.array([[F(IA.of(z).mid()) if isinstance(z, IA) else z for z in row] for row in J4(list(th))], float)
    score = np.linalg.norm(J, axis=0)*w
    return int(np.argmax(score))

ORDER = 'margin'
SCALE = np.array([1.0, 1.0, 1.85, 1.85])
def key_of(lo, hi):
    if ORDER == 'volume': return -float(np.prod((hi - lo)*SCALE))      # largest boxes first (breadth-first)
    return -margin_est(lo, hi)

def run_boxes(boxes, tag, maxtime, wmin=1e-7):
    heap = []; cnt = 0
    for lo, hi in boxes:
        lo = np.array(lo, float); hi = np.array(hi, float)
        heapq.heappush(heap, (key_of(lo, hi), cnt, lo, hi)); cnt += 1
    t0 = time.time(); n = 0; last = t0; openb = []
    while heap:
        key, _, lo, hi = heapq.heappop(heap); n += 1
        try: r = try_close(lo, hi)
        except (ArithmeticError, ZeroDivisionError, AssertionError, ValueError) as ex:
            r = None; stat('ERR')
            if STATS['ERR'] <= 5: print('[%s] ERR %r at %s %s' % (tag, ex, lo.tolist(), hi.tolist()), flush=True)
        if r is not None: stat(r); continue
        w = hi - lo
        if w.max() < 2*wmin: stat('OPEN'); openb.append((lo.tolist(), hi.tolist())); continue
        j = split_coord(lo, hi); m = 0.5*(lo[j] + hi[j])
        lo2 = lo.copy(); hi2 = hi.copy(); lo2[j] = m; hi2[j] = m
        for a, b in ((lo, hi2), (lo2, hi)):
            heapq.heappush(heap, (key_of(a, b), cnt, a, b)); cnt += 1
        stat('SPLIT')
        if time.time() - last > 60:
            last = time.time()
            print('[%s] %.0fs boxes %d heap %d stats %s phi %.0fs grad %.0fs next-key %.4f' % (tag, last - t0, n, len(heap), STATS, TIME['phi'], TIME['grad'], -key), flush=True)
        if time.time() - t0 > maxtime:
            print('[%s] TIME LIMIT; heap %d' % (tag, len(heap)), flush=True)
            openb += [(a.tolist(), b.tolist()) for _, _, a, b in heap]; break
    dt = time.time() - t0
    out = dict(tag=tag, boxes=n, stats=STATS, open=openb, time=dt, r0=R0, Mt_lo=MT_LO, xmin=XMIN, kappa=[F(KAP.lo), F(KAP.hi)], K0=[F(K0.lo), F(K0.hi)], loc_lambda_max=LMAX)
    json.dump(out, open('bnb9_%s.json' % tag, 'w'))
    print('[%s] DONE %.0fs boxes %d stats %s open %d' % (tag, dt, n, STATS, len(openb)), flush=True)
    return out

R0 = 1e-4; LMAX = None; B0_LO = B0_HI = None
def setup(r0):
    global R0, LMAX, B0_LO, B0_HI
    R0 = r0; lam, rad, LMAX = Mloc4(r0)
    print('LOC-4D r0 %g: eig(mid) %s ||rad||_F %.4f lambda_max <= %.4f' % (r0, np.round(lam, 4), math.sqrt((rad**2).sum()), LMAX), flush=True)
    assert LMAX < 0
    B0_LO = np.array([F(P_T.lo) - r0, F(P_T.lo) - r0, -r0, -r0]); B0_HI = np.array([F(P_T.hi) + r0, F(P_T.hi) + r0, r0, r0])

if __name__ == '__main__':
    mode = sys.argv[1]; maxtime = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    r0 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
    setup(r0)
    print('XMIN %.12f Mt_lo %.17g' % (XMIN, MT_LO), flush=True)
    if mode == 'check':
        th = [0.9, 0.95, 0.03, 0.02]
        J = np.array([[F(IA.of(z).mid()) for z in row] for row in J4(th)]); h = 1e-6
        Jfd = np.zeros((6, 4))
        for j in range(4):
            tp = list(th); tm = list(th); tp[j] += h; tm[j] -= h
            Jfd[:, j] = (np.array(C4(tp)) - np.array(C4(tm)))/(2*h)
        print('max |J4 - fd| =', np.abs(J - Jfd).max())
        HI = H4([IA(float(t)) for t in th])
        err = 0
        for p in range(6):
            for i in range(4):
                for j in range(4):
                    tpp = list(th); tpm = list(th); tmp = list(th); tmm = list(th)
                    tpp[i] += h; tpp[j] += h; tpm[i] += h; tpm[j] -= h; tmp[i] -= h; tmp[j] += h; tmm[i] -= h; tmm[j] -= h
                    fd = (C4(tpp)[p] - C4(tpm)[p] - C4(tmp)[p] + C4(tmm)[p])/(4*h*h)
                    err = max(err, abs(F(HI[p][i][j].mid()) - fd))
        print('max |H4 - fd| =', err)
        thT = theta_T4(); thTI = [IA(float(t)) for t in thT]
        HC = H4(thTI); g = K['g_T'].mid()
        Ha = np.array([[sum(g[p]*F(HC[p][i][j].mid()) for p in range(6)) for j in range(4)] for i in range(4)])
        print('sum_p g_p Hess C_p at tree (should be ~0):', np.abs(Ha).max())
        for r in (1e-5, 3e-5, 2e-4, 3e-4, 5e-4):
            lam, rad, b = Mloc4(r); print('  r0 %g: eig(mid) %s lambda_max <= %.4f' % (r, np.round(lam, 4), b))
    elif mode == '2D':
        run_boxes([([XMIN, XMIN, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0])], '2D', maxtime)
    elif mode == 'test':
        r = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
        tT = theta_T4(); run_boxes([([tT[0] - r, tT[1] - r, 0.0, 0.0], [tT[0] + r, tT[1] + r, r, r])], 'test', maxtime)
    elif mode == 'A':
        run_boxes([([XMIN, XMIN, 0.0, 0.0], [1.0, F(P_T.hi), TMAX, TMAX])], 'A', maxtime)
    elif mode == 'B':
        run_boxes([([XMIN, F(P_T.lo), 0.0, 0.0], [1.0, 1.0, TMAX, TMAX])], 'B', maxtime)
    elif mode == 'full':
        run_boxes([([XMIN, XMIN, 0.0, 0.0], [1.0, 1.0, TMAX, TMAX])], 'full', maxtime)
    elif mode == 'R':
        d = json.load(open(sys.argv[4])); run_boxes(d['open'], 'R' + d['tag'], maxtime)
    elif mode == 'RV':
        ORDER = 'volume'; d = json.load(open(sys.argv[4])); run_boxes(d['open'], 'RV' + d['tag'], maxtime)
