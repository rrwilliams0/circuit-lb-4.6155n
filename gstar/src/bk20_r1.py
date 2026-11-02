r"""bk20_r1.py : the check (exact mirror cells) for G* records.

The G* alpha grid ABP is not mirror-symmetric in doubles (ABP[k] + ABP[16-k] - 1 = +-2^-54 for k in {0,3,4,5,11,12,13,16}), so
the EXACT mirror [1 - hi, 1 - lo] (rationals) of a box of cell i may overhang the table cell u = 15 - i that bk19_joint uses
by a slab of width 2^-54 into a neighbour cell.  Lemma CAP (b) needs, on that slab, a certified lower bound of Emin; valid
are Emin_lo of the neighbour cell(s) met, and an independent program's 1e-12-extended certificate of cell u on that side
(eslv_out.json, key 'u,j,lo|hi', field sliver_min; the check (a)).  So the justified cap value is
    ev = min( T(u, j) [if the mirror meets cell u],  max(min T(neighbours met), sliver_min(u, j, side)) [per overhang side] ),
    need = c_Theta^hi - W_LO * max(ev, 0)     (exact rationals; c_Theta^hi = 0.9595702342051 as in bk16_model)
and a capped record is justified iff its 'cth' >= need.  For 'XJ' records the exclusion is re-checked with the exact mirror:
max(T(i,j),0) + W_LO * max(ev, 0) > B^hi.  Missing table entries count as 0 (no cap).

Usage (any folder):  python bk20_r1.py census <flagged_out.jsonl> <record files...>
                     python bk20_r1.py fix <flagged.jsonl>     (replay every flagged record with cth raised to need, rounded
                                                                 up to a double; ranges re-certified, union with recorded)
Env: BK19_EMIN_A, BK19_EMIN_B (the table), BK20_ESLV (eslv_out.json).
"""
import os, sys, json, math, bisect
from fractions import Fraction as F
from collections import Counter
os.environ.setdefault('BK17_GPM', '1'); os.environ.setdefault('BK17_GLEVELS', '0:0.01,0.01:1')
mode = sys.argv[1]
args = sys.argv[2:]
sys.argv = [sys.argv[0], 'G', 'replay', '1', '0']
import bk19_joint as J
import bk17_run as RUN
T = J.table()
SL = json.load(open(os.environ.get('BK20_ESLV', 'eslv_out.json')))
ABP, LBP = RUN.ABP, RUN.LBP
NC = len(ABP) - 1
CTH = F('0.9595702342051'); W = F('0.97147'); BHI = F(J.B_HI)
assert float(J.CTH0.hi) == 0.9595702342051 and J.W_LO == 0.97147


def tv(c, j): return F(T.get((c, j), 0.0))


def ev_exact(lo0, hi0, i, j):
    """least justified Emin lower bound over the exact mirror of [lo0, hi0] (cell i, row j); also which source decided"""
    u = NC - 1 - i; x0, x1 = 1 - F(hi0), 1 - F(lo0); vals = []; src = 'table'
    a0, a1 = F(ABP[u]), F(ABP[u + 1])
    if x0 < a1 and x1 > a0: vals.append((tv(u, j), 'table'))
    for side, lo_, hi_ in (('lo', x0, a0), ('hi', a1, x1)):
        if lo_ >= hi_: continue
        nb = [tv(c, j) for c in range(NC) if F(ABP[c]) < hi_ and F(ABP[c + 1]) > lo_]
        sl = SL.get('%d,%d,%s' % (u, j, side))
        if not nb: continue   # slab beyond 0.3/0.7 (cells 0/15 only): outside the WIN window, which is certified in doubles
                              # and closed under the mirror, so it holds no marginal of N
        else:
            m = min(nb); s = F(sl['sliver_min']) if (sl and sl['closed_ok']) else None
            vals.append((max(m, s), 'sliver') if (s is not None and s > m) else (m, 'neighbour'))
    v, src = min(vals)
    return v, src


def census(outf, files):
    n = 0; ncap = 0; nxj = 0; bad = Counter(); src = Counter(); slack = None; xjbad = 0
    fo = open(outf, 'w'); fsl = open(outf + '.sliver.jsonl', 'w')   # records justified only via the extended (eslv) table
    for fn in files:
        for l in open(fn):
            n += 1
            if '"cth"' not in l and '"XJ"' not in l: continue
            r = json.loads(l); lo, hi = r['lo'], r['hi']
            i = bisect.bisect_right(ABP, lo[0]) - 1; j = bisect.bisect_right(LBP, lo[1]) - 1
            ev, s = ev_exact(lo[0], hi[0], i, j)
            if r['k'] == 'XJ':
                nxj += 1
                if not (max(tv(i, j), 0) + W * max(ev, 0) > BHI): xjbad += 1; print('XJ NOT JUSTIFIED', fn, lo, hi, flush=True)
                continue
            ncap += 1; need = CTH - W * max(ev, F(0)); d = F(r['cth']) - need
            src[s] += 1
            if s == 'sliver': fsl.write(json.dumps([os.path.basename(fn), lo, hi, r['lev'], r['k'], r['cth']]) + chr(10))
            slack = d if slack is None else min(slack, d)
            if d < 0:
                bad[(i, j, r['k'])] += 1
                r['_need'] = float(need); r['_file'] = os.path.basename(fn); fo.write(json.dumps(r) + '\n')
    fo.close(); fsl.close()
    print('records %d capped %d XJ %d (XJ not justified %d); cap source %s; flagged (cth < need) %d %s; min slack %s'
          % (n, ncap, nxj, xjbad, dict(src), sum(bad.values()), dict(bad), float(slack) if slack is not None else None), flush=True)


def fix(flf):
    import numpy as np
    import bk19_verify as VV
    import bk17_box as XB, bk16_model as M, bk16_lb as B
    from ivarith import dn

    def replay_G(r, pm, dom, tau0):   # bk19_verify.replay_G with ranges = union(recorded, re-certified) (both valid)
        lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
        inh = 'rbox' in r
        rlo, rhi = (np.array(r['rbox'][0]), np.array(r['rbox'][1])) if inh else (lo, hi)
        rlev = tuple(r.get('rlev', lev))
        if not (np.all(rlo <= lo) and np.all(rhi >= hi) and rlev[0] <= lev[0] and rlev[1] >= lev[1]): return False, -4
        rb = XB.make_box(rlo, rhi, pm)
        rg_c, _ = XB.ranges(rb, rlev, pm, rec=r['rd'])
        rg = {}
        for key in M.RANGE_KEYS:
            a, b_ = rg_c[key]; rl, rh = r['rg'][key]; rg[key] = (min(rl, a), max(rh, b_))
        bx = XB.make_box(lo, hi, pm)
        if inh:
            rg['Y'] = (min(rg['Y'][0], 0.0), rg['Y'][1])
            if not np.all(np.asarray(bx.Qb['BC']['pq']) <= np.asarray(rb.Qb['BC']['pq'])): return False, -13
        tp = XB.target_value(dom, bx, bx.Qp, lo, hi, tau0); tb = XB.target_value(dom, bx, bx.Qb, lo, hi, tau0)
        if not r['th'] or min(r['th']) < 0 or sum(r['th']) < 0.999999: return False, -3
        s = np.zeros(16)
        for c, th in zip(r['c'], r['th']):
            cp, kp = M.main_cost(bx.Qp, rg, c['ch']); cb, kb = M.main_cost(bx.Qb, rg, c['ch'])
            Vv = B.mode1_vertices(bx, cp, kp - tp, cb, kb - tb, lev, c['d'])
            s = dn(s + dn(th * Vv))
        return float(s.min()) >= 0, float(s.min())
    XB.replay_G = replay_G
    n = 0; bad = 0; minlb = {}; kinds = Counter()
    for l in open(flf):
        r = json.loads(l); need = F(r.pop('_need')); fn = r.pop('_file')
        lo = r['lo']; i = bisect.bisect_right(ABP, lo[0]) - 1; j = bisect.bisect_right(LBP, lo[1]) - 1
        ev, _ = ev_exact(lo[0], r['hi'][0], i, j); need = CTH - W * max(ev, F(0))
        c = float(need)
        while F(c) < need: c = math.nextafter(c, 2.0)
        r['cth'] = c; n += 1; kinds[r['k']] += 1
        try: ok, lb = VV.replay(r)
        except Exception as ex: ok, lb = False, -10
        minlb[r['k']] = min(minlb.get(r['k'], 1e9), lb)
        if not ok: bad += 1; print('FAIL', fn, r['k'], r['lo'], r['hi'], r['lev'], lb, flush=True)
    print('fix: replayed with the justified cap %d records %s, failures %d, min margins %s' % (n, dict(kinds), bad, minlb), flush=True)


if mode == 'census': census(args[0], args[1:])
elif mode == 'fix': fix(args[0])
