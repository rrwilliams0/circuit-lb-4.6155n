"""bk19_olag.py : O-the-1 in LAGRANGIAN form (no E^lin bands): shared job pool, resumable; replay.

Usage:  python bk19_olag.py <tag> <worker k> <budget s>      (folder with <tag>_jobs.json and lagpts.json)
        python bk19_olag.py replay <stride> <offset> <records...>

lagpts.json: {"UP": [[L_i, lam_i, K_i], ...], "VP": [[mu_i, K_i], ...]}.  On every u-star slice box (pm coordinates, the
standard slice-level set with the NS row, E^lin <= B) the job's family certifies
   family 'U':  for every point i and every sign pattern s:  K_i + lam_i E^lin - [upper linear form of U_{L_i}] >= 0,
   family 'V':  (mirror u-star slices) for every point i:     K_i + mu_i E^lin - [upper linear form of V] >= 0,
i.e. Phi(L_i, lam_i) := sup_s [U_{L_i}(s) - lam_i E^lin(s)] <= K_i and Psi(mu_i) := sup_s [V(s) - mu_i E^lin(s)] <= K_i over
the stars of N (the upper linear forms are valid for stars of N: THIN, FBAR, REL; bk19_obk).
Then (Lemma LAG, report): Ubar_j <= Phi(L_j, lam) + lam E_{j+1} for every lam >= 0, Phi is convex in (L, lam); Vbar(c) <= min_i
(K_i + mu_i c).  Closures 'X', 'Xd', 'E', 'E2', 'O' as bk19_orun.py (job jobs: [fam, -1, lo, hi, lev, depth]).
"""
import os, sys, json, time, itertools
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
import bk19_orun as R
from bk19_orun import M, BB, L2, O, IA, dn
LP = json.load(open(os.environ.get('BK19_LAGPTS', 'lagpts.json'))) if os.path.exists(os.environ.get('BK19_LAGPTS', 'lagpts.json')) else None
BAND = (None, float(O.BJ.hi))           # E^lin <= B: the NS row itself (capc = min(c_Theta, mu0 + B) = c_Theta)


def objectives(fam):
    if fam == 'U': return [(i, s) for i in range(len(LP['UP'])) for s in R.SIGNS]
    return [(i, None) for i in range(len(LP['VP']))]


def cand_costs(fam, Q, rg, obj, ch, th):
    i, s = obj
    if fam == 'U':
        Lk, lam, K = LP['UP'][i]
        uv = O.u_vectors(Q)
        cp, kp = O.u_cost(Q, rg, ch, s, Lk, K, th, uv)
        return cp + Q['nsg'] * (IA(lam) / 3.0), kp + (Q['nsc'] / 3.0 - O.MU0) * IA(lam)
    mu, K = LP['VP'][i]
    return O.v_cost(Q, rg, ch, K, mu, 0.0)


def bound_one(bx, lev, fam, rg, obj, quick):
    vals = []; cands = []
    ths = R.THS if fam == 'U' else (None,)
    for ch in R.choices(bx, lev, rg):
        for th in ths:
            cp, kp = cand_costs(fam, bx.Qp, rg, obj, ch, th); cb, kb = cand_costs(fam, bx.Qb, rg, obj, ch, th)
            dual, _ = L2.float_lp(bx, bx.fl(cp), lev)
            if dual is None: continue
            dual = [float(x) for x in dual]
            dual = L2.affine_dual(bx, dual, L2.dual_slopes(bx, cp, dual, lev))
            V = L2.mode1_vertices(bx, cp, kp, cb, kb, lev, dual)
            vals.append(V); cands.append(dict(ch=ch, th=th, d=dual))
            if float(V.min()) >= 0: return dict(c=[cands[-1]], th=[1.0], lb=float(V.min()))
    if not vals: return None
    Vm = np.where(np.isfinite(np.array(vals)), np.array(vals), -1e9)
    th_, lb = BB.theta_combine(Vm)
    if lb < 0 and not quick:
        best = int(np.argmax(Vm.min(axis=1))); c0 = cands[best]
        cp, kp = cand_costs(fam, bx.Qp, rg, obj, c0['ch'], c0['th']); cb, kb = cand_costs(fam, bx.Qb, rg, obj, c0['ch'], c0['th'])
        for aff in (True, False):
            sp = L2.robust_dual(bx, cp, kp, lev, affine=aff)
            if sp is None: continue
            V = L2.mode1_vertices(bx, cp, kp, cb, kb, lev, sp)
            vals.append(V); cands.append(dict(ch=c0['ch'], th=c0['th'], d=sp))
        Vm = np.where(np.isfinite(np.array(vals)), np.array(vals), -1e9)
        th_, lb = BB.theta_combine(Vm)
    keep = [i for i in range(len(th_)) if th_[i] > 0]
    bx.vbest = Vm[int(np.argmax(Vm.min(axis=1)))]
    return dict(c=[cands[i] for i in keep], th=[float(th_[i]) for i in keep], lb=lb)


def all_bounds(bx, lev, fam, rg, quick=False):
    res = []; worst = 1e9
    for obj in objectives(fam):
        r = bound_one(bx, lev, fam, rg, obj, quick)
        if r is None: return None, -1.0
        res.append(r); worst = min(worst, r['lb'])
        if r['lb'] < 0: return res, r['lb']
    return res, worst


def process(fam, lo, hi, lev, inh):
    if BB.marg_excluded(lo, hi): return dict(k='X'), None, None
    if R.d_outside_R7(lo, hi): return dict(k='Xd'), None, None
    bx = L2.Box(lo, hi, BAND)
    e = R.emptiness(bx, lev)
    if e is not None: return e, None, bx
    if inh is not None:
        rg, rd, rlo, rhi = inh
        res, lb = all_bounds(bx, lev, fam, rg, quick=True)
        if res is not None and lb >= 0:
            return dict(k='O', rd=rd, rg={a: list(b) for a, b in rg.items() if not a.startswith('_')}, rbox=[list(rlo), list(rhi)], sg=res, lb=lb), None, bx
    rg, rd = R.ranges(bx, lev, R.keys_of(fam))
    if rg is None: return dict(k='F', lb=-1.0, fail=True), None, bx
    for key, (a, b) in rg.items():
        if a > b: return dict(k='E2', key=key, rd={key: rd[key]}, lb=a - b), None, bx
    if fam == 'U' and not R.derived(rg): return dict(k='F', lb=-1.0, fail=True), None, bx
    new = (rg, rd, [float(x) for x in lo], [float(x) for x in hi])
    res, lb = all_bounds(bx, lev, fam, rg)
    if res is not None and lb >= 0:
        return dict(k='O', rd=rd, rg={a: list(b) for a, b in rg.items() if not a.startswith('_')}, sg=res, lb=lb), None, bx
    return dict(k='O', fail=True, lb=lb), new, bx


def replay(r):
    fam = r['fam']; lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    kk = r.get('k')
    if r.get('fail'): return False, -9
    if kk == 'X': return BB.marg_excluded(lo, hi), 1.0
    if kk == 'Xd': return R.d_outside_R7(lo, hi), 1.0
    bx = L2.Box(lo, hi, BAND)
    zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if kk == 'E':
        V = L2.mode1_vertices(bx, zv, zero, zv, zero, lev, r['ed']); return float(V.min()) > 0, float(V.min())
    if kk == 'E2':
        key = r['key']; d1, d2 = r['rd'][key]; cache = {}
        vp = O.functional_ext(bx.Qp, key, cache); vb = O.functional_ext(bx.Qb, key, cache)
        a, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
        b_, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
        return a > b_, a - b_
    if kk != 'O': return False, -8
    rlo, rhi = (np.array(r['rbox'][0]), np.array(r['rbox'][1])) if 'rbox' in r else (lo, hi)
    if not (np.all(rlo <= lo) and np.all(rhi >= hi)): return False, -4
    rb = L2.Box(rlo, rhi, BAND)
    rgc, _ = R.ranges(rb, lev, R.keys_of(fam), rec=r['rd'])
    rg = {}
    for key in R.keys_of(fam):
        a, b_ = rgc[key]; rl, rh = r['rg'][key]
        if not (rl <= a and rh >= b_): return False, -2
        rg[key] = (rl, rh)
    if fam == 'U' and not R.derived(rg): return False, -3
    objs = objectives(fam)
    if len(r['sg']) != len(objs): return False, -5
    worst = 1e9
    for obj, rr in zip(objs, r['sg']):
        if not rr['th'] or min(rr['th']) < 0 or sum(rr['th']) < 0.999999: return False, -6
        tot = np.zeros(16)
        for c, th in zip(rr['c'], rr['th']):
            cp, kp = cand_costs(fam, bx.Qp, rg, obj, c['ch'], c['th']); cb, kb = cand_costs(fam, bx.Qb, rg, obj, c['ch'], c['th'])
            V = L2.mode1_vertices(bx, cp, kp, cb, kb, lev, c['d'])
            tot = dn(tot + dn(th * V))
        worst = min(worst, float(tot.min()))
    return worst >= 0, worst


def main(tag, wk, budget):
    jobs = json.load(open(tag + '_jobs.json'))
    os.makedirs('claims', exist_ok=True)
    base = '%s_w%d' % (tag, wk)
    stf = base + '.state.json'; rf = base + '.jsonl'; of = base + '_open.jsonl'
    if os.path.exists(stf):
        st = json.load(open(stf))
        for fn, off in ((rf, st['roff']), (of, st['ooff'])):
            if os.path.exists(fn):
                with open(fn, 'r+b') as f: f.truncate(off)
        cur = st['cur']; stack = [(np.array(a), np.array(b), tuple(l), d, None) for a, b, l, d in st['stack']]
        done = st['done']; stats = st['stats']
    else:
        cur = None; stack = []; done = []
        stats = dict(processed=0, closed=0, open=0, cpu=0.0, minlb={}, kinds={}, maxdepth=0)
    out = open(rf, 'a'); openf = open(of, 'a')
    t0 = time.time(); c0 = time.process_time(); last = t0; cpu_base = stats['cpu']

    def checkpoint():
        out.flush(); openf.flush()
        stats['cpu'] = cpu_base + time.process_time() - c0
        st = dict(cur=cur, stack=[(a.tolist(), b.tolist(), list(l), d) for a, b, l, d, _ in stack], done=done, stats=stats,
                  roff=out.tell(), ooff=openf.tell())
        tmp = stf + '.tmp'; json.dump(st, open(tmp, 'w')); os.replace(tmp, stf)
    nexti = 0; stopped = False
    while True:
        if time.time() - t0 > budget or os.path.exists(tag + '.stop') or os.path.exists(base + '.stop'): stopped = True; break
        if cur is None:
            while nexti < len(jobs) and not R.claim(tag, nexti): nexti += 1
            if nexti >= len(jobs): break
            cur = nexti; fam, kb, a, b, l, d = jobs[cur]
            stack = [(np.array(a, float), np.array(b, float), tuple(l), d, None)]
            checkpoint()
        if not stack:
            done.append(cur); fin = cur; cur = None; checkpoint()
            open(os.path.join('claims', '%s_%d.done' % (tag, fin)), 'w').close()
            continue
        fam = jobs[cur][0]
        lo, hi, lev, dep, inh = stack.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        try:
            rec, ninh, bx = process(fam, lo, hi, lev, inh)
        except (ZeroDivisionError, ArithmeticError, ValueError, FloatingPointError) as ex:
            rec, ninh, bx = dict(k='F', lb=-1.0, fail=True, err=str(ex)[:80]), None, None
        if not rec.get('fail'):
            kk = rec['k']; stats['closed'] += 1; stats['kinds'][kk] = stats['kinds'].get(kk, 0) + 1
            if 'lb' in rec: stats['minlb'][kk] = min(stats['minlb'].get(kk, 1e9), rec['lb'])
            out.write(json.dumps(dict(fam=fam, band=-1, lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), **rec)) + '\n')
        else:
            if dep >= R.MAXDEPTH:
                stats['open'] += 1
                openf.write(json.dumps(dict(fam=fam, band=-1, lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), lb=rec.get('lb'))) + '\n')
            else:
                i = R.split(lo, hi, bx)
                mid = 0.5 * (lo[i] + hi[i])
                h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
                stack.append((l2, hi, lev, dep + 1, ninh)); stack.append((lo, h1, lev, dep + 1, ninh))
        if time.time() - last > 60:
            last = time.time(); checkpoint()
            print('%7.0fs job %s stack %d done %d %s' % (time.time() - t0, cur, len(stack), len(done), json.dumps(stats)), flush=True)
    checkpoint(); out.close(); openf.close()
    print(('STOPPED' if stopped else 'DONE'), base, 'job', cur, 'done', len(done), json.dumps(stats), flush=True)
    if not stopped: open(base + '.done', 'w').close()


if __name__ == '__main__':
    if sys.argv[1] == 'replay':
        stride, off = int(sys.argv[2]), int(sys.argv[3]); files = sys.argv[4:]
        bad = n = 0; i = -1; kinds = {}; minlb = {}
        for fn in files:
            for l in open(fn):
                i += 1
                if i % stride != off: continue
                if os.path.exists('replay.stop'): break
                r = json.loads(l); n += 1; kk = r.get('k'); kinds[kk] = kinds.get(kk, 0) + 1
                try: ok, lb = replay(r)
                except Exception as ex: ok, lb = False, -10
                minlb[kk] = min(minlb.get(kk, 1e9), lb)
                if not ok: bad += 1; print('FAIL', fn, i, kk, r['fam'], r['lo'], r['hi'], r['lev'], lb, flush=True)
        print('checked %d kinds %s failures %d min margins %s' % (n, kinds, bad, minlb), flush=True)
    else:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
