"""bk19_grun.py : G* (domain G of bk17_run) with a SHARED job pool, resumable, any number of workers.

Usage:  python bk19_grun.py <tag> <worker k> <time budget s>        (run in the folder that holds <tag>_jobs.json)
Jobs:   <tag>_jobs.json = list of [lo, hi, lev, depth] boxes (built by bk19_mkjobs.py): the pending G boxes and every
        root-level of bk17_run.roots_G() that no the worker had started.  A worker claims job i by creating
        claims/<tag>_<i>.claim exclusively (os.O_EXCL), processes it depth-first to the end, then writes claims/<tag>_<i>.done.
Boxes:  exactly bk17_run.process (same closures 'X', 'Xd', 'E', 'E2', 'G' with the target g(alpha), same split rule,
        same bisection points), with the speed-ups of bk19_fast installed (no change of mathematics).  Depth limit 40 ->
        <tag>_w<k>_open.jsonl.
Files:  records <tag>_w<k>.jsonl, checkpoint <tag>_w<k>.state.json (current job, its stack, finished jobs, stats, byte
        offsets of the record/open files; atomic, every 60 s).  A restarted worker truncates its files to the offsets and
        resumes its current job first.  Stop: <tag>.stop or <tag>_w<k>.stop (checked between boxes).
"""
import os, sys, json, time
os.environ.setdefault('BK17_GPM', '1')
os.environ.setdefault('BK17_GLEVELS', '0:0.01,0.01:1')
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
sys.argv = [sys.argv[0], 'G'] + sys.argv[1:]
import numpy as np
import bk17_run as RUN
import bk16_bnb as BB
import bk19_fast
bk19_fast.install2()
assert RUN.DOM == 'G' and RUN.PM and RUN.LEVELS == [(0.0, 0.01), (0.01, 1.0)]
JOINT = os.environ.get('BK19_JOINT', '0') == '1'
if JOINT:
    import bk19_joint as J
    from ivarith import IA
    J.table()


def process_cap(lo, hi, lev, dep, inh):
    """bk17_run.process, with Lemma CAP (bk19_joint) when BK19_JOINT=1: 'XJ' closure of marginal cells excluded by JOINT,
    otherwise the NS row with c_Theta' = cth_cap (recorded as 'cth' in every closed record of the box)."""
    if not JOINT: return RUN.process(lo, hi, lev, dep, inh)
    i, j = J.root_of(lo, hi)
    if J.excluded(i, j): return dict(k='XJ', cell=[i, j]), None, None
    cth = J.cth_cap(i, j)
    J.M.CTH = IA(cth)
    try:
        rec, ninh, bx = RUN.process(lo, hi, lev, dep, inh)
    finally:
        J.M.CTH = J.CTH0
    if rec is not None and rec.get('k') in ('G', 'E', 'E2'): rec['cth'] = cth
    return rec, ninh, bx


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
        print('resuming', base, 'job', cur, 'stack', len(stack), 'done', len(done), flush=True)
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

    nexti = 0
    stopped = False
    while True:
        if time.time() - t0 > budget or os.path.exists(tag + '.stop') or os.path.exists(base + '.stop'): stopped = True; break
        if cur is None:
            while nexti < len(jobs) and not claim(tag, nexti): nexti += 1
            if nexti >= len(jobs): break
            cur = nexti; a, b, l, d = jobs[cur]
            stack = [(np.array(a, float), np.array(b, float), tuple(l), d, None)]
            checkpoint()
        if not stack:
            done.append(cur); fin = cur; cur = None; checkpoint()
            open(os.path.join('claims', '%s_%d.done' % (tag, fin)), 'w').close()
            continue
        lo, hi, lev, dep, inh = stack.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        try:
            rec, ninh, bx = process_cap(lo, hi, lev, dep, inh)
        except (ZeroDivisionError, ArithmeticError, ValueError, FloatingPointError) as ex:
            rec, ninh, bx = dict(k='F', lb=-1.0, fail=True, err=str(ex)[:80]), None, None
        if rec is not None and not rec.get('fail'):
            k = rec['k']; stats['closed'] += 1; stats['kinds'][k] = stats['kinds'].get(k, 0) + 1
            if 'lb' in rec: stats['minlb'][k] = min(stats['minlb'].get(k, 1e9), rec['lb'])
            out.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), **rec)) + '\n')
        else:
            if dep >= RUN.MAXDEPTH:
                stats['open'] += 1
                openf.write(json.dumps(dict(lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), lb=(rec or {}).get('lb'))) + '\n')
            else:
                i = RUN.split_choice(lo, hi, dep, bx, False)
                mid = 0.5 * (lo[i] + hi[i])
                if i == 0:
                    for x in (0.5 - BB.A0G, 0.5 + BB.A0G):
                        if lo[0] < x < hi[0]: mid = x
                h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
                stack.append((l2, hi, lev, dep + 1, ninh)); stack.append((lo, h1, lev, dep + 1, ninh))
        if time.time() - last > 60:
            last = time.time(); checkpoint()
            print('%7.0fs job %s stack %d done %d %s' % (time.time() - t0, cur, len(stack), len(done), json.dumps(stats)), flush=True)
    checkpoint(); out.close(); openf.close()
    print(('STOPPED' if stopped else 'DONE'), base, 'job', cur, 'done', len(done), json.dumps(stats), flush=True)
    if not stopped: open(base + '.done', 'w').close()


if __name__ == '__main__':
    main(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
