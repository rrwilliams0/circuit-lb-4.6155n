"""bk10_run.py : resumable, recorded, margin-first branch and bound for (R-i'-4) on the open boxes of bnb9_RVB.json.
Rules: those of bk10_bnb.py (= bk9_bnb.py + the runtime guard on TAY2 and on LOC-4D) plus TAY3 (bk10_tay3.py).
Order of attempts on a popped box (any order is sound; only the closing rule matters):
  XY, INF-PSD, INF-K, LOC (verbatim bk9 code)  ->  CORNER first if the cached NUMERICAL corner margin is >= 1e-7
  -> TAY2/TAY1 (guarded bk9 taylor, only if the image box is certified PD by pd_box) -> TAY3 (k = 2, then k = 4)
  -> CORNER (if not yet tried) -> bisection (bk9 split_coord).
Every popped box and its outcome is appended to rec_<tag>.jsonl; the open set (heap) is checkpointed atomically to
ckpt_<tag>.json every CKPT seconds and at the end, so a run can be resumed with  python bk10_run.py resume <tag> <maxtime>.
usage:  python bk10_run.py run <tag> <maxtime_s> <worker> <nworkers>  (roots = open boxes of bnb9_RVB.json, worker share)
        python bk10_run.py resume <tag> <maxtime_s>"""
import os, sys, json, time, heapq, math
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import bk10_bnb as BB
import bk10_tay3 as T3
F = BB.F; IA = BB.IA; AP = BB.AP
CKPT = 300.0
STATS = {}; T = {'cheap': 0.0, 'corner': 0.0, 'tay2': 0.0, 'tay3': 0.0, 'key': 0.0}
def stat(k): STATS[k] = STATS.get(k, 0) + 1


def cheap_rules(lo, hi):
    """verbatim the first part of bk9_bnb.try_close (XY, INF-PSD, INF-K, LOC)."""
    if lo[0] > hi[1]: return 'XY'
    thI = [BB.ia(lo[j], hi[j]) for j in range(4)]
    x, y, tu, tv = thI
    S2 = BB.K0 - BB.KAP*(x + y)
    if S2.lo <= 0: return 'NOS2'
    def q_lo(xI, tI):
        q = IA(F(S2.lo))*(IA(2.0)*(1.0 + IA(F(xI.hi)))).inv()
        if xI.lo < 1.0: q = q + (IA(F(tI.lo))*IA(F(tI.lo))*2.0)*(1.0 - IA(F(xI.lo))).inv()
        return IA(F(q.lo))
    if (q_lo(x, tu)*q_lo(y, tv)).lo > 1.0: return 'INF-PSD'
    S = S2.sqrt()
    if ((IA(F(S.lo))*0.5 + F(tu.lo))*(IA(F(S.lo))*0.5 + F(tv.lo))).lo > 1.0: return 'INF-K'
    if np.all(lo >= BB.B0_LO) and np.all(hi <= BB.B0_HI): return 'LOC'
    return None


def base_fn(th0, need):
    """RIGOROUS upper bound of Phi_4(th0) = Phi_F(C(th0)): lem:G(b) at c_dn = lower ends of the outward enclosure of C(th0)
    (bk8_core C_of with the kappa interval), c_dn certified PD by pd_point, value by the checked phi_point / F_point
    (the phi_up).  A NUMERICAL float estimate is used only to give up early (never enters a bound)."""
    CI = BB.C4([IA(float(t)) for t in th0]); cdn = np.array([F(z.lo) for z in CI])
    fm = BB.float_margin(cdn)                 # MT_LO - Phi(cdn) in floats, -1 if not evaluable
    if fm != -1.0 and (BB.MT_LO - fm) > need + 1e-7: return None
    if not AP.pd_point(list(cdn)): return None
    return BB.phi_up(cdn)


def try_close10(lo, hi, mkey):
    r = cheap_rules(lo, hi)
    if r == 'NOS2': return None, None
    if r is not None: return r, None
    tried_corner = False
    thI = [BB.ia(lo[j], hi[j]) for j in range(4)]
    Cb = BB.C4(thI); clo = np.array([F(z.lo) for z in Cb]); chi = np.array([F(z.hi) for z in Cb])
    if 1e-7 <= mkey < 2.0:
        t0 = time.time(); r = BB.corner(clo); T['corner'] += time.time() - t0; tried_corner = True
        if r is not None: return r, None
    pdb = False
    try: pdb = AP.pd_box(clo, chi)
    except (ArithmeticError, ZeroDivisionError): pdb = False
    info = None
    if pdb:
        t0 = time.time(); r = BB.taylor(lo, hi, thI, clo, chi); T['tay2'] += time.time() - t0
        if r is not None: return r, ({'guard': BB.GUARD.get('last')} if r == 'TAY2' else None)
    for k in (2, 4):
        t0 = time.time()
        try: ok, U = T3.tay3(lo, hi, k, base_fn)
        except (ArithmeticError, ZeroDivisionError, ValueError, AssertionError): ok, U = False, None
        T['tay3'] += time.time() - t0
        if ok: return 'TAY3', {'k': k, 'U': U}
        if U is not None and U - BB.MT_LO > 0.5: break          # hopeless at this width (heuristic skip of k = 4)
    if not tried_corner:
        t0 = time.time(); r = BB.corner(clo); T['corner'] += time.time() - t0
        if r is not None: return r, None
    return None, info


def key_of(lo, hi):
    t0 = time.time(); m = BB.margin_est(lo, hi); T['key'] += time.time() - t0
    return -m


def save_ckpt(tag, heap, extra):
    out = dict(tag=tag, open=[(a.tolist(), b.tolist()) for _, _, a, b in heap], stats=STATS, guard=BB.GUARD,
               tay3=T3.STATS3, timers=T, Mt_lo=BB.MT_LO, r0=BB.R0, loc_lambda_max=BB.LMAX, **extra)
    tmp = 'ckpt_%s.json.tmp' % tag
    with open(tmp, 'w') as f: json.dump(out, f)
    os.replace(tmp, 'ckpt_%s.json' % tag)


def run(roots, tag, maxtime, prev_time=0.0, prev_n=0, wmin=1e-7):
    heap = []; cnt = 0
    for lo, hi in roots:
        lo = np.array(lo, float); hi = np.array(hi, float)
        heapq.heappush(heap, (key_of(lo, hi), cnt, lo, hi)); cnt += 1
    rec = open('rec_%s.jsonl' % tag, 'a')
    t0 = time.time(); n = 0; last = t0; lastck = t0
    while heap:
        key, _, lo, hi = heapq.heappop(heap); n += 1
        try: r, info = try_close10(lo, hi, -key)
        except (ArithmeticError, ZeroDivisionError, AssertionError, ValueError) as ex:
            r, info = None, {'err': repr(ex)}; stat('ERR')
        if r is not None:
            stat(r); d = dict(lo=lo.tolist(), hi=hi.tolist(), r=r)
            if info: d.update(info)
            rec.write(json.dumps(d) + '\n')
        else:
            w = hi - lo
            if w.max() < 2*wmin:
                stat('OPEN'); rec.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), r='OPEN')) + '\n')
                heapq.heappush(heap, (1e9, cnt, lo, hi)); cnt += 1     # kept in the open list (never re-popped first)
                if all(k[0] >= 1e9 for k in heap): break
                continue
            j = BB.split_coord(lo, hi); m = 0.5*(lo[j] + hi[j])
            lo2 = lo.copy(); hi2 = hi.copy(); lo2[j] = m; hi2[j] = m
            rec.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), r='SPLIT', j=j, m=m)) + '\n')
            for a, b in ((lo, hi2), (lo2, hi)):
                heapq.heappush(heap, (key_of(a, b), cnt, a, b)); cnt += 1
            stat('SPLIT')
        now = time.time()
        if now - last > 60:
            last = now
            print('[%s] %.0fs boxes %d heap %d stats %s guard nF %.3g lam+ %.3g refused %d/%d/%d tay3 %s timers %s next-key %.3g' % (
                tag, now - t0 + prev_time, n + prev_n, len(heap), STATS, BB.GUARD['max_Mneg_F'], BB.GUARD['max_lam_pos'],
                BB.GUARD['refused_neg'], BB.GUARD['refused_pos'], BB.GUARD['refused_exact'], T3.STATS3,
                {k: round(v) for k, v in T.items()}, -heap[0][0] if heap else 0), flush=True)
        if now - lastck > CKPT:
            lastck = now; rec.flush(); save_ckpt(tag, heap, dict(time=now - t0 + prev_time, boxes=n + prev_n, done=False))
        if now - t0 > maxtime:
            print('[%s] TIME LIMIT; heap %d' % (tag, len(heap)), flush=True); break
    rec.close()
    dt = time.time() - t0
    done = (len(heap) == 0)
    save_ckpt(tag, heap, dict(time=dt + prev_time, boxes=n + prev_n, done=done))
    print('[%s] %s %.0fs (this session %.0fs) boxes %d stats %s open %d' % (tag, 'DONE' if done else 'STOPPED', dt + prev_time, dt,
          n + prev_n, STATS, len(heap)), flush=True)


if __name__ == '__main__':
    BB.setup(1e-4)
    print('LOC guard ||mid||_F = %.4f; XMIN %.12f Mt_lo %.17g' % (BB.GUARD['loc_mid_F'], BB.XMIN, BB.MT_LO), flush=True)
    mode, tag, maxtime = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if mode == 'run':
        wk, nw = int(sys.argv[4]), int(sys.argv[5])
        roots = json.load(open('bnb9_RVB.json'))['open']
        tT = np.array(BB.theta_T4())
        dist = [np.linalg.norm(np.maximum(0, np.maximum(np.array(a) - tT, tT - np.array(b)))) for a, b in roots]
        order = np.argsort(dist, kind='stable')
        mine = [roots[i] for r_, i in enumerate(order) if r_ % nw == wk]
        if len(sys.argv) > 6: mine = mine[:int(sys.argv[6])]
        print('[%s] worker %d of %d: %d roots' % (tag, wk, nw, len(mine)), flush=True)
        run(mine, tag, maxtime)
    elif mode == 'resume':
        ck = json.load(open('ckpt_%s.json' % tag))
        for k, v in ck['stats'].items(): STATS[k] = v
        for k in ('tay2_checked', 'refused_neg', 'refused_pos', 'refused_exact'): BB.GUARD[k] = ck['guard'][k]
        for k in ('max_Mneg_F', 'max_lam_pos', 'max_pos_factor'): BB.GUARD[k] = ck['guard'][k]
        for k, v in ck['tay3'].items(): T3.STATS3[k] = v
        for k, v in ck['timers'].items(): T[k] = v
        print('[%s] resume: %d open boxes, %.0fs and %d boxes so far' % (tag, len(ck['open']), ck['time'], ck['boxes']), flush=True)
        run(ck['open'], tag, maxtime, prev_time=ck['time'], prev_n=ck['boxes'])
