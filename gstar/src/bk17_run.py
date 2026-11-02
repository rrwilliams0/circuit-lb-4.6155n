"""bk17_run.py : resumable depth-first branch and bound for the CKS* certificate, one worker per process.

Usage:  python bk17_run.py <domain Q|G> <tag> <worker k> <workers W> <time budget s>
Domain Q (pm coordinates z = (alpha, lam, dp, dm), p in Q = [0.47, 0.53] x [0, 0.005]; all d of the R7 box):
   levels [0, S1] and [S1, 1] of the off-face mass.  A box-level closes as
     'X'  every d of the box is outside the R7 square (interval test),
     'K'  level [0, S1], box inside D_loc = {|dp| <= WP, |dm| <= WM}: Lemma KINK-K (bk17_kink.kink_check2, repaired
          slope condition), P0 <= P0MAX,
     'E'/'E2'  empty slice-level set,
     'G'  certified relaxation >= tau0 (direct closure; covers Q1, Q2 and the 'D' closure inside D_loc).
Domain G (coordinates per BK17_GPM: pm if '1', else (alpha, lam, d_a, d_b)), p in the window W minus int Q:
     'X'  marginal excluded (V outside R7 or Lemma E0) or (pm) d outside the R7 square,  'E'/'E2',  'G' (>= g(alpha)).
Worker k processes the roots with index = k mod W.  Records: <tag>_w<k>.jsonl ; open boxes: <tag>_w<k>_open.jsonl ;
checkpoint <tag>_w<k>.state.json (queue, stats, byte offsets of the record files, written atomically every 60 s).
Stop: create <tag>.stop (all workers) or <tag>_w<k>.stop.  A restarted worker truncates its record files to the offsets of
the last checkpoint and resumes from its queue, so no record is duplicated or lost.
"""
import os, sys, json, time, math
DOM = sys.argv[1] if len(sys.argv) > 1 else 'Q'
GPM = os.environ.get('BK17_GPM', '1') == '1'
PM = True if DOM == 'Q' else GPM
os.environ['BK16_MODE'] = 'Q2' if PM else 'G'
import numpy as np
import bk17_cfg
from ivarith import IA
import bk16_model as M
import bk16_bnb as BB
import bk17_box as XB

TAU0 = bk17_cfg.TAU0
S1 = float(os.environ.get("BK17_S1", "0.01"))
WP, WM = 0.01, 0.08
P0MAX = 0.02
QA = (0.47, 0.53); QL = (0.0, 0.005)
MAXDEPTH = int(os.environ.get('BK17_MAXDEPTH', '40'))
# per-cell targets of domain Q (Lemma LOC2-K(c)): for p in alpha cell i of Q the direct target is c_{11-i} (>= 0), where
# h_K - P0 mu_0 >= -c_j on cell j (run/loc2c/loc2c.json).  Without the file: tau0 = 1.5e-4 >= every c_j.
_TQF = os.environ.get('BK17_TQ')
TQ = json.load(open(_TQF))['c'] if _TQF else None


def q_cell(lo, hi):
    A = list(np.round(np.linspace(0.47, 0.53, 13), 10)); A[0], A[-1] = 0.47, 0.53      # the alpha grid of roots_Q
    for i in range(12):
        if lo[0] >= A[i] and hi[0] <= A[i + 1]: return i
    return None


def q_target(lo, hi):
    if TQ is None: return TAU0
    i = q_cell(lo, hi)
    return TAU0 if i is None else max(0.0, float(TQ[11 - i]))


def tau_of(al, la):
    q = 1.0 - la; S = q * (2.0 - al) + la
    return 2.5 - (al * q * S.inv()) * 1.5


def d_outside_R7(lo, hi):
    """pm box: True if for every point d_a or d_b is outside [DLO, DHI] (certified)."""
    al = IA(lo[0], hi[0]); la = IA(lo[1], hi[1]); tau = tau_of(al, la)
    dp = IA(lo[2], hi[2]); dm = IA(lo[3], hi[3])
    da = (tau + dp + dm) * 0.5; db = (tau + dp - dm) * 0.5
    for d in (da, db):
        if float(d.hi) < M.DLO or float(d.lo) > M.DHI: return True
    return False


def inside_Dloc_pm(lo, hi):
    return lo[2] >= -WP and hi[2] <= WP and lo[3] >= -WM and hi[3] <= WM and (lo[2] >= 0 or hi[2] <= 0)


def grid(bps):
    return [(bps[i], bps[i + 1]) for i in range(len(bps) - 1)]


def roots_Q():
    A = list(np.round(np.linspace(0.47, 0.53, 13), 10)); A[0], A[-1] = 0.47, 0.53
    L = [0.0, 0.0025, 0.005]
    DP = [-0.55, -0.28, -0.14, -0.07, -0.035, -0.01, -0.005, 0.0, 0.005, 0.01, 0.035, 0.07, 0.14, 0.28, 0.72]
    DMb = [-0.59, -0.3, -0.16, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.16, 0.3, 0.59]
    out = []
    for a in grid(A):
        for l in grid(L):
            for p in grid(DP):
                for m in grid(DMb):
                    out.append(([a[0], l[0], p[0], m[0]], [a[1], l[1], p[1], m[1]]))
    return out


ABP = [0.3, 0.35, 0.4, 0.42, 0.44, 0.46, 0.47, 0.485, 0.5, 0.515, 0.53, 0.54, 0.56, 0.58, 0.6, 0.65, 0.7]
LBP = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
GDP = [-0.79, -0.36, -0.18, -0.09, -0.045, -0.02, -0.01, 0.0, 0.01, 0.02, 0.045, 0.09, 0.18, 0.36, 0.99]   # covers dp in [-0.782, 0.982] (tau on W in [1.6882, 2.2800])
GDM = [0.0, 0.075, 0.15, 0.3, 0.59]          # dm >= 0 suffices by Lemma SYM (a <-> b swap)


def roots_G():
    out = []
    for i in range(len(ABP) - 1):
        for j in range(len(LBP) - 1):
            lo2 = [ABP[i], LBP[j]]; hi2 = [ABP[i + 1], LBP[j + 1]]
            if lo2[0] >= QA[0] and hi2[0] <= QA[1] and hi2[1] <= QL[1]: continue          # inside Q
            if PM:
                for p in grid(GDP):
                    for m in grid(GDM):
                        out.append((lo2 + [p[0], m[0]], hi2 + [p[1], m[1]]))
            else:
                dbp = [M.DLO + (M.DHI - M.DLO) * k / 2 for k in range(3)]; dbp[-1] = M.DHI
                for p in range(2):
                    for q in range(2):
                        out.append((lo2 + [dbp[p], dbp[q]], hi2 + [dbp[p + 1], dbp[q + 1]]))
    return out


LEVELS = [(0.0, S1), (S1, 1.0)] if DOM == 'Q' else [tuple(map(float, x.split(':'))) for x in os.environ.get('BK17_GLEVELS', '0:0.01,0.01:1').split(',')]
TW = np.array([0.005, 0.0025, 0.01, 0.02]) if PM else np.array([0.005, 0.0025, 0.01, 0.01])
KT = np.array([0.00125, 0.000625, 0.0025, 0.005])


def split_choice(lo, hi, dep, bx, kinkzone):
    w = hi - lo
    if kinkzone: return int(np.argmax(w / KT))
    i = _split_choice(lo, hi, dep, bx)
    nw = w / TW
    if nw[i] < nw.max() / 16.0: i = int(np.argmax(nw))           # anti-starvation: never refine one coordinate alone
    return i


def _split_choice(lo, hi, dep, bx):
    w = hi - lo
    vb = getattr(bx, 'vbest', None) if bx is not None else None
    if vb is not None and np.all(np.isfinite(vb)) and dep % 3 != 2:
        from bk13_lb import VMASK
        infl = np.array([np.mean(np.abs(vb[VMASK[:, i] == 1] - vb[VMASK[:, i] == 0])) for i in range(4)])
        if infl.max() > 0: return int(np.argmax(infl + 1e-12 * w))
    return int(np.argmax(w / TW))


def process(lo, hi, lev, dep, inh):
    """returns (record or None, children-inheritance, bx)."""
    if DOM == 'G':
        if BB.marg_excluded(lo, hi): return dict(k='X'), None, None
    if PM and d_outside_R7(lo, hi): return dict(k='Xd'), None, None
    kz = DOM == 'Q' and lev == (0.0, S1) and inside_Dloc_pm(lo, hi)
    why = None
    if kz:
        import bk17_kink as K2
        side = 1 if lo[2] >= 0 else -1
        r = K2.kink_check2(lo, hi, side, S1)
        if r['ok'] and r['P0'] <= P0MAX:
            return dict(k='K', side=side, eta=r['eta'], rd=r['rd'], rg=r['rg'], rcmin=r['rcmin'], P0=r['P0'], slope=r['slope'], lb=min(r['rcmin'], r['slope'])), None, None
        why = r.get('why', '')
        tryD = (why != 'hinge') or max(abs(lo[3]), abs(hi[3])) > 0.0745 or dep >= 12
        if not tryD: return None, inh, None
    tq = q_target(lo, hi) if DOM == 'Q' else TAU0
    rec, bx, new = XB.box_eval(lo, hi, lev, PM, DOM, tq, inh=inh)
    if DOM == 'Q' and rec['k'] == 'G': rec['tq'] = tq
    if rec['k'] in ('E', 'E2', 'G') and rec['lb'] >= 0: return rec, None, bx
    rec_fail = dict(rec); rec_fail['fail'] = True
    return rec_fail, new, bx


def run(tag, wk, W, budget):
    roots = roots_Q() if DOM == 'Q' else roots_G()
    sel = os.environ.get('BK17_SEL')
    if sel:
        a0, a1, l0, l1 = map(float, sel.split(','))
        roots = [r for r in roots if r[0][0] >= a0 and r[1][0] <= a1 and r[0][1] >= l0 and r[1][1] <= l1]
    mine = [r for i, r in enumerate(roots) if i % W == wk]
    base = '%s_w%d' % (tag, wk)
    stf = base + '.state.json'; rf = base + '.jsonl'; of = base + '_open.jsonl'
    if os.path.exists(stf):
        st = json.load(open(stf))
        queue = [(np.array(a), np.array(b), tuple(l), d, None) for a, b, l, d in st['queue']]
        stats = st['stats']
        for fn, off in ((rf, st['roff']), (of, st['ooff'])):
            if os.path.exists(fn):
                with open(fn, 'r+b') as f: f.truncate(off)
        print('resuming', base, len(queue), stats, flush=True)
    else:
        queue = []
        for lo, hi in reversed(mine):
            for lev in reversed(LEVELS):
                queue.append((np.array(lo, float), np.array(hi, float), lev, 0, None))
        stats = dict(processed=0, closed=0, open=0, cpu=0.0, minlb={}, kinds={}, maxdepth=0)
    out = open(rf, 'a'); openf = open(of, 'a')
    t0 = time.time(); c0 = time.process_time(); last = t0; cpu_base = stats['cpu']

    def checkpoint():
        out.flush(); openf.flush()
        stats['cpu'] = cpu_base + time.process_time() - c0
        st = dict(queue=[(a.tolist(), b.tolist(), list(l), d) for a, b, l, d, _ in queue], stats=stats,
                  roff=out.tell(), ooff=openf.tell())
        tmp = stf + '.tmp'; json.dump(st, open(tmp, 'w')); os.replace(tmp, stf)
    while queue:
        if time.time() - t0 > budget or os.path.exists(tag + '.stop') or os.path.exists(base + '.stop'): break
        lo, hi, lev, dep, inh = queue.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        try:
            rec, ninh, bx = process(lo, hi, lev, dep, inh)
        except (ZeroDivisionError, ArithmeticError, ValueError, FloatingPointError) as ex:
            rec, ninh, bx = dict(k='F', lb=-1.0, fail=True, err=str(ex)[:80]), None, None
        if rec is not None and not rec.get('fail'):
            k = rec['k']; stats['closed'] += 1; stats['kinds'][k] = stats['kinds'].get(k, 0) + 1
            if 'lb' in rec: stats['minlb'][k] = min(stats['minlb'].get(k, 1e9), rec['lb'])
            out.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), **rec)) + '\n')
        else:
            if dep >= MAXDEPTH:
                stats['open'] += 1
                openf.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), lb=(rec or {}).get('lb'))) + '\n')
            else:
                kz = DOM == 'Q' and lev == (0.0, S1) and inside_Dloc_pm(lo, hi)
                i = split_choice(lo, hi, dep, bx, kz)
                mid = 0.5 * (lo[i] + hi[i])
                if i == 0 and DOM == 'G':
                    for x in (0.5 - BB.A0G, 0.5 + BB.A0G):
                        if lo[0] < x < hi[0]: mid = x
                h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
                queue.append((l2, hi, lev, dep + 1, ninh)); queue.append((lo, h1, lev, dep + 1, ninh))
        if time.time() - last > 60:
            last = time.time(); checkpoint()
            print('%7.0fs %s queue %d' % (time.time() - t0, json.dumps(stats), len(queue)), flush=True)
    checkpoint()
    out.close(); openf.close()
    print('DONE' if not queue else 'STOPPED', base, json.dumps(stats), 'queue', len(queue), flush=True)


if __name__ == '__main__':
    tag = sys.argv[2]; wk = int(sys.argv[3]); W = int(sys.argv[4]); budget = float(sys.argv[5])
    print('domain', DOM, 'pm', PM, 'levels', LEVELS, 'pilo', bk17_cfg.PILO, 'KILL', M.KILL.hi, 'tau0', TAU0, flush=True)
    run(tag, wk, W, budget)
