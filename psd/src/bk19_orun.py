"""bk19_orun.py : branch and bound for O-the-1 (banded families, bk19_obk / bk19_lb2), shared job pool.

Usage:  python bk19_orun.py <tag> <worker k> <budget s>        (folder with <tag>_jobs.json and targets.json)
        python bk19_orun.py replay <stride> <offset> <records...>   (no LP: re-certify every record)
Jobs: [fam ('U'|'V'), band k, lo, hi, lev, depth].  Box closures:
  'X'  marginal excluded (bk16_bnb.marg_excluded: V outside R7's V-range or Lemma E0),  'Xd' d outside the R7 square,
  'E'  slice-band-level set empty (certified Farkas Mode 1 bound > 0),  'E2' a certified range with lo > hi,
  'O'  family bound certified: U: for all 16 sign patterns, T - (upper linear form) >= 0 at all 16 vertices;
       V: t0 + t1 (E^lin - e_lo) - (upper linear form) >= 0.
Files, checkpoints, claims and stop files as bk19_grun.py.
"""
import os, sys, json, time, itertools
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
os.environ['BK16_MODE'] = 'Q2'
import numpy as np
import bk17_cfg
from ivarith import IA, dn, up
import bk16_model as M
import bk16_bnb as BB
import bk19_lb2 as L2
import bk19_obk as O
from bk13_lb import T1
O.KILL = M.KILL
LK = [O.Lcut(k * 1e-4) for k in range(25)]
SIGNS = [s for s in itertools.product((1.0, -1.0), repeat=4)]
THS = (1.0, 0.0, 0.5)
MAXDEPTH = 36
TW = np.array([0.005, 0.0025, 0.01, 0.02])
TG = json.load(open(os.environ.get('BK19_TARGETS', 'targets.json'))) if os.path.exists(os.environ.get('BK19_TARGETS', 'targets.json')) else None


def band_of(k):
    return (None if k == 0 else round(k * 1e-4, 10), round((k + 1) * 1e-4, 10))


def tau_of(al, la):
    q = 1.0 - la; S = q * (2.0 - al) + la
    return 2.5 - (al * q * S.inv()) * 1.5


def d_outside_R7(lo, hi):
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); tau = tau_of(al, la)
    dp = IA(lo[2], hi[2]); dm = IA(lo[3], hi[3])
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    for d in (da, db):
        if float(d.hi) < M.DLO or float(d.lo) > M.DHI: return True
    return False


# ------------------------------------------------------------------ ranges
def _r_of(e):   # r(e) = (2 + 2e)^{-1/2} as an interval
    return (IA(e) * 2.0 + 2.0).sqrt().inv()


def derived(rg):
    ea = (max(rg['eA'][0], float(O.RHO.lo)), min(rg['eA'][1], float(O.EMAX.hi)))
    eb = (max(rg['eB'][0], float(O.RHO.lo)), min(rg['eB'][1], float(O.EMAX.hi)))
    if ea[0] > ea[1] or eb[0] > eb[1]: return False
    lo_ = float(((_r_of(ea[1]) + _r_of(eb[1])) * 0.5).lo); hi_ = float(((_r_of(ea[0]) + _r_of(eb[0])) * 0.5).hi)
    rg['_rbar'] = (lo_, hi_)
    rg['_Dl'] = min(O.DMAXG, max(abs(rg['dl'][0]), abs(rg['dl'][1])))
    return True


def ranges(bx, lev, keys, rec=None):
    zero = M.tconst(0.0); rg = {}; duals = {}; cache = {}
    for key in keys:
        vp = O.functional_ext(bx.Qp, key, cache); vb = O.functional_ext(bx.Qb, key, cache)
        if rec is not None:
            d1, d2 = rec[key]
            lo_, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
            hi_, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
            rg[key] = (lo_, hi_); duals[key] = (d1, d2); continue
        res = []
        for mx in (False, True):
            best, bd = L2.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, slopes=True)
            if best is None: return None, None
            cf = bx.fl(vp)
            dc, fv = L2.float_lp(bx, -cf if mx else cf, lev)
            fv = (-fv if mx else fv) if dc is not None else None
            loose = fv is None or abs(best - fv) > 0.05 * abs(fv) + 1e-4
            if loose:
                for aff in (True, False):
                    sp = L2.robust_dual(bx, -vp if mx else vp, zero, lev, affine=aff)
                    if sp is None: continue
                    val, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, maximize=mx, dual=sp)
                    if (mx and val < best) or ((not mx) and val > best): best, bd = val, sp
                    if fv is not None and abs(best - fv) <= 0.05 * abs(fv) + 1e-4: break
            res.append((best, bd))
        rg[key] = (res[0][0], res[1][0]); duals[key] = (res[0][1], res[1][1])
    return rg, duals


def emptiness(bx, lev):
    zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    dual, val = L2.float_lp(bx, np.zeros(M.NP), lev)
    if dual is not None and val > 1e-7:
        d = [float(x) for x in dual]
        V = L2.mode1_vertices(bx, zv, zero, zv, zero, lev, d)
        if float(V.min()) > 0: return dict(k='E', ed=d, lb=float(V.min()))
        for aff in (True, False):
            sp = L2.robust_dual(bx, zv, zero, lev, affine=aff)
            if sp is None: continue
            V = L2.mode1_vertices(bx, zv, zero, zv, zero, lev, sp)
            if float(V.min()) > 0: return dict(k='E', ed=sp, lb=float(V.min()))
    return None


# ------------------------------------------------------------------ objectives
def cand_costs(fam, k, Q, rg, sgn, ch, th):
    if fam == 'U':
        uv = O.u_vectors(Q)
        return O.u_cost(Q, rg, ch, sgn, LK[k], TG['TU'][k], th, uv)
    t0, t1, elo = TG['TV'][k]
    return O.v_cost(Q, rg, ch, t0, t1, elo)


def choices(bx, lev, rg):
    out = []
    for ch0 in BB.CHOICES:
        out.append(dict(ch0, R=[1.0] + [0.0] * 8))
    return out


def bound_one(bx, lev, fam, k, rg, sgn, quick):
    """theta-combined certified bound for one objective (one sign pattern).  Returns dict(c=..., th=..., lb=...)."""
    vals = []; cands = []
    ths = THS if fam == 'U' else (None,)
    for ch in choices(bx, lev, rg):
        for th in ths:
            cp, kp = cand_costs(fam, k, bx.Qp, rg, sgn, ch, th); cb, kb = cand_costs(fam, k, bx.Qb, rg, sgn, ch, th)
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
        cp, kp = cand_costs(fam, k, bx.Qp, rg, sgn, c0['ch'], c0['th']); cb, kb = cand_costs(fam, k, bx.Qb, rg, sgn, c0['ch'], c0['th'])
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


def all_bounds(bx, lev, fam, k, rg, quick=False):
    sg = SIGNS if fam == 'U' else [None]
    res = []; worst = 1e9
    for s in sg:
        r = bound_one(bx, lev, fam, k, rg, s, quick)
        if r is None: return None, -1.0
        res.append(r); worst = min(worst, r['lb'])
        if r['lb'] < 0: return res, r['lb']
    return res, worst


def keys_of(fam):
    return tuple(M.RANGE_KEYS) + (O.EXTRA_U if fam == 'U' else ())


def process(fam, k, lo, hi, lev, inh):
    if BB.marg_excluded(lo, hi): return dict(k='X'), None, None
    if d_outside_R7(lo, hi): return dict(k='Xd'), None, None
    bx = L2.Box(lo, hi, band_of(k))
    e = emptiness(bx, lev)
    if e is not None: return e, None, bx
    if inh is not None:
        rg, rd, rlo, rhi = inh
        res, lb = all_bounds(bx, lev, fam, k, rg, quick=True)
        if res is not None and lb >= 0:
            return dict(k='O', rd=rd, rg={a: list(b) for a, b in rg.items() if not a.startswith('_')}, rbox=[list(rlo), list(rhi)], sg=res, lb=lb), None, bx
    rg, rd = ranges(bx, lev, keys_of(fam))
    if rg is None: return dict(k='F', lb=-1.0, fail=True), None, bx
    for key, (a, b) in rg.items():
        if a > b: return dict(k='E2', key=key, rd={key: rd[key]}, lb=a - b), None, bx
    if fam == 'U' and not derived(rg): return dict(k='F', lb=-1.0, fail=True), None, bx
    new = (rg, rd, [float(x) for x in lo], [float(x) for x in hi])
    res, lb = all_bounds(bx, lev, fam, k, rg)
    if res is not None and lb >= 0:
        return dict(k='O', rd=rd, rg={a: list(b) for a, b in rg.items() if not a.startswith('_')}, sg=res, lb=lb), None, bx
    return dict(k='O', fail=True, lb=lb), new, bx


# ------------------------------------------------------------------ replay (no LP)
def replay(r):
    fam, k = r['fam'], r['band']; lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    kk = r.get('k')
    if r.get('fail'): return False, -9
    if kk == 'X': return BB.marg_excluded(lo, hi), 1.0
    if kk == 'Xd': return d_outside_R7(lo, hi), 1.0
    bx = L2.Box(lo, hi, band_of(k))
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
    rb = L2.Box(rlo, rhi, band_of(k))
    rgc, _ = ranges(rb, lev, keys_of(fam), rec=r['rd'])
    rg = {}
    for key in keys_of(fam):
        a, b_ = rgc[key]; rl, rh = r['rg'][key]
        if not (rl <= a and rh >= b_): return False, -2
        rg[key] = (rl, rh)
    if fam == 'U' and not derived(rg): return False, -3
    sg = SIGNS if fam == 'U' else [None]
    if len(r['sg']) != len(sg): return False, -5
    worst = 1e9
    for s, rr in zip(sg, r['sg']):
        if not rr['th'] or min(rr['th']) < 0 or sum(rr['th']) < 0.999999: return False, -6
        tot = np.zeros(16)
        for c, th in zip(rr['c'], rr['th']):
            cp, kp = cand_costs(fam, k, bx.Qp, rg, s, c['ch'], c['th']); cb, kb = cand_costs(fam, k, bx.Qb, rg, s, c['ch'], c['th'])
            V = L2.mode1_vertices(bx, cp, kp, cb, kb, lev, c['d'])
            tot = dn(tot + dn(th * V))
        worst = min(worst, float(tot.min()))
    return worst >= 0, worst


# ------------------------------------------------------------------ driver
def split(lo, hi, bx):
    w = hi - lo; nw = w / TW
    vb = getattr(bx, 'vbest', None) if bx is not None else None
    i = int(np.argmax(nw))
    if vb is not None and np.all(np.isfinite(vb)):
        from bk13_lb import VMASK
        infl = np.array([np.mean(np.abs(vb[VMASK[:, q] == 1] - vb[VMASK[:, q] == 0])) for q in range(4)])
        if infl.max() > 0:
            j = int(np.argmax(infl + 1e-12 * w))
            if nw[j] >= nw.max() / 16.0: i = j
    return i


def claim(tag, i):
    try:
        fd = os.open(os.path.join('claims', '%s_%d.claim' % (tag, i)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        return False


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
            while nexti < len(jobs) and not claim(tag, nexti): nexti += 1
            if nexti >= len(jobs): break
            cur = nexti; fam, kb, a, b, l, d = jobs[cur]
            stack = [(np.array(a, float), np.array(b, float), tuple(l), d, None)]
            checkpoint()
        if not stack:
            done.append(cur); fin = cur; cur = None; checkpoint()
            open(os.path.join('claims', '%s_%d.done' % (tag, fin)), 'w').close()
            continue
        fam, kb = jobs[cur][0], jobs[cur][1]
        lo, hi, lev, dep, inh = stack.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        try:
            rec, ninh, bx = process(fam, kb, lo, hi, lev, inh)
        except (ZeroDivisionError, ArithmeticError, ValueError, FloatingPointError) as ex:
            rec, ninh, bx = dict(k='F', lb=-1.0, fail=True, err=str(ex)[:80]), None, None
        if not rec.get('fail'):
            kk = rec['k']; stats['closed'] += 1; stats['kinds'][kk] = stats['kinds'].get(kk, 0) + 1
            if 'lb' in rec: stats['minlb'][kk] = min(stats['minlb'].get(kk, 1e9), rec['lb'])
            out.write(json.dumps(dict(fam=fam, band=kb, lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), **rec)) + '\n')
        else:
            if dep >= MAXDEPTH:
                stats['open'] += 1
                openf.write(json.dumps(dict(fam=fam, band=kb, lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), lb=rec.get('lb'))) + '\n')
            else:
                i = split(lo, hi, bx)
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
                if not ok: bad += 1; print('FAIL', fn, i, kk, r['fam'], r['band'], r['lo'], r['hi'], r['lev'], lb, flush=True)
        print('checked %d kinds %s failures %d min margins %s' % (n, kinds, bad, minlb), flush=True)
    else:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
