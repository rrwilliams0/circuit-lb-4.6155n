r"""bk23_orun.py : the driver (bk22_orun.py) changed to THEOREM LR's lower-row checks (see the notes):
every check k of a family has the set {E^lin >= e_k} (no lower row for e_k = null) intersected with the full slice-level
set (NS row E^lin <= B, i.e. capc = c_Theta exactly as the FULL band); U checks carry their own L_k and an affine target
t0 + t1 (E^lin - c_k) (c_k = e_k, or 0 for e_k = null), V checks an affine target.  Targets file (the format):
  U = [[e_lo, L, t0, t1], ...], V = [[e_lo, t0, t1], ...].   Everything else (records, replay, ranges, masks) is the.
Original the docstring follows.
bk22_orun.py : the shared-tree PL certificate of O-the-1 (the Theorem PL, obligations (U) and (V)) with
BAND MASKS, on the engine (src\bk19_orun, bk19_obk, bk19_lb2: md5-identical copies, unchanged).

One depth-first branch-and-bound tree per root job (fam, marginal cell x d-box, off-face level).  Every stack item carries
the MASK of E^lin bands still to be certified on it (the root: all 25).  On a box:
  'X' / 'Xd' / 'E' (whole box, full band) / 'E2' (a certified range empty)  -> every band of the mask certified;
  'ER'  certified E^lin range [elo, ehi] of the box's full-band set (two Mode 1 bounds with recorded duals): a band whose
        closed interval [e_k, e_{k+1}] misses [elo, ehi] holds no star of the box -> certified;
  'O'   per band k of the mask: band set certified empty ('E', Farkas dual) or the family bound with target
        t0_k + t1_k (E^lin - e_k) certified ('O', the sign patterns / theta combination / duals), with the certified
        ranges of the box (computed on the FULL-band set) or inherited from an ancestor (rbox), the check applied;
  bands that fail are the children's mask (the box is bisected; children check only those bands).
Per band, the boxes certifying it tile the root (checked by bk22_cover.py through the recorded paths).

Fixes relative to the pilot driver (S\bk21_orun.py), see the report Section 1:
  S1 ranges are always computed on the FULL-band set (the computed them, after a failed inherited check, on the rows of the
     last band checked, and used them for all bands);
  S2 the check on inherited ranges: Ybar's range is [min(Ylo, 0), Yhi] and pq(child) <= pq(parent) is checked;
  F1 band masks (children re-check only the bands that failed; the "band 0" lever, for every band);
  F2 the E^lin range exclusion (the "two LPs per box" lever) when the mask has >= 3 bands;
  F3 roots ordered level [0.01, 1] first (the calibration advice), then the near-tree level [0, 0.01].

Usage:  python bk22_orun.py mkjobs <tag>                       (writes <tag>_jobs.json, 640 roots)
        python bk22_orun.py <tag> <worker k> <budget s>         (pool worker; claims\<tag>_<i>.claim/.done; stop files)
        python bk22_orun.py replay <stride> <offset> <records...>   (no LP: re-certify every record)
Targets: targets.json in the working folder (the format), or BK22_TARGETS.
"""
import os, sys, json, time
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
os.environ['BK16_MODE'] = 'Q2'
os.environ['BK19_TARGETS'] = '__no_bk19_targets__.json'          # bk19_orun's own TG unused (cand_costs replaced)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, 'src'))
import numpy as np
import bk19_orun as R, bk19_obk as O, bk19_lb2 as L2, bk16_model as M, bk16_bnb as BB
from ivarith import IA, dn
from bk13_lb import MU0

TGF = os.environ.get('BK23_TARGETS', 'targets.json')
TG = json.load(open(TGF)) if os.path.exists(TGF) else None
import hashlib
TMD5 = hashlib.md5(open(TGF, 'rb').read()).hexdigest() if TG else None      # note: targets md5 in every log and record
ECAP = 0.0025                                                    # capc = min(c_Theta^hi, mu0 + 0.0025) = c_Theta^hi (NS row)
BANDSF = {f: [(r[0], ECAP) for r in TG[f]] for f in 'UV'} if TG else None
FULL = (None, ECAP)
ALLF = {f: list(range(len(TG[f]))) for f in 'UV'} if TG else None
STAT = dict(t_rng=0.0, t_band=0.0, t_empty=0.0, t_er=0.0)
_uvc = {}


# ------------------------------------------------------------------ bands and targets (the set_band / cand_costs)
def set_band(bx, band):
    e_lo, e_hi = band
    capc = min(float(M.CTH.hi), float((MU0 + IA(e_hi)).hi))
    for Q in (bx.Qb, bx.Qp):
        Q['nshc'] = -Q['nsc'] + IA(capc) * 3.0
        Q['nsl'] = (-Q['nsc'] + (MU0 + IA(e_lo)) * 3.0) if e_lo is not None else None
    bx.band = band; bx.capc = capc; bx._pts = None


def cand_costs(fam, k, Q, rg, sgn, ch, th):
    """cost, const of  [t0_k + t1_k (E^lin - e_k)] - [upper linear form]  on band k (the, unchanged)."""
    elo = BANDSF[fam][k][0] if BANDSF[fam][k][0] is not None else 0.0
    if fam == 'U':
        _, Lk, t0, t1 = TG['U'][k]
        if id(Q) not in _uvc:
            if len(_uvc) > 64: _uvc.clear()
            _uvc[id(Q)] = O.u_vectors(Q)
        cost, const = O.u_cost(Q, rg, ch, sgn, Lk, t0, th, _uvc[id(Q)])
        if t1 != 0.0:
            cost = cost + Q['nsg'] * (IA(t1) / 3.0); const = const + (Q['nsc'] / 3.0 - MU0 - IA(elo)) * IA(t1)
        return cost, const
    _, t0, t1 = TG['V'][k]
    return O.v_cost(Q, rg, ch, t0, t1, elo)


R.cand_costs = cand_costs          # bk19_orun.bound_one / all_bounds look this up at call time


# ------------------------------------------------------------------ the speed-up: memoised DUB (pure function of (Q, rg, ch))
_dubc = {}
FAILSKIP = float(os.environ['BK23_FAILSKIP']) if os.environ.get('BK23_FAILSKIP') else None   # skip threshold (None: the loop)
_dub_orig = O.dub


def _dub_cached(Q, rg, ch):
    """bk19_obk.dub depends only on (Q, rg, ch); it is the same for every check and sign pattern of a box.  The cache keeps
    references to Q and rg (so their ids cannot be reused while cached) and is cleared at the start of every process()/replay()."""
    key = (id(Q), id(rg), json.dumps(ch, sort_keys=True))
    hit = _dubc.get(key)
    if hit is not None and hit[0] is Q and hit[1] is rg: return hit[2]
    out = _dub_orig(Q, rg, ch)
    _dubc[key] = (Q, rg, out)
    return out


if os.environ.get('BK23_NODUBCACHE') != '1':
    O.dub = _dub_cached


# ------------------------------------------------------------------ ranges on a relaxed superset near the NS boundary
RDELTA = float(os.environ.get('BK23_RDELTA', '2e-4'))       # 0: off
BJ_F = 0.0024634531
from scipy.optimize import linprog as _linprog
from bk13_core import A_EQ as _AEQ


def set_rcap(bx, capE):
    """NS row relaxed to E^lin <= capE (capc = max(c_Theta^hi, mu0 + capE rounded up) >= the true cap): a SUPERSET of the
    full set; no lower row.  Ranges certified on it are valid on every check's set."""
    capc = max(float(M.CTH.hi), float((MU0 + IA(capE)).hi))
    for Q in (bx.Qb, bx.Qp):
        Q['nshc'] = -Q['nsc'] + IA(capc) * 3.0
        Q['nsl'] = None
    bx.capc = capc; bx._pts = None


def relaxed_cap(bx, lev):
    """float min of E^lin over the slice-level set (no NS row) at the centre; returns capE = that + RDELTA if it exceeds B,
    else None (the full set is used).  A heuristic choice only: any capE >= B gives valid ranges."""
    if RDELTA <= 0: return None
    Q = bx.Qp; mu = [bx.fl(x) for x in Q['mu']]; b = np.array([mu[0], mu[1], mu[2], bx.dc[0], bx.dc[1]])
    nsg = bx.fl(Q['nsg']); nsc = float(bx.fl(Q['nsc']))
    r = _linprog(nsg, A_ub=np.array([M.OFF, -M.OFF]), b_ub=np.array([lev[1], -lev[0]]), A_eq=_AEQ, b_eq=b,
                 bounds=[(0, None)] * 20, method='highs')
    if r.status != 0: return None
    em = (r.fun + nsc) / 3 - (1 + 2 * 2 ** 0.5) / 4
    return em + RDELTA if em + RDELTA > BJ_F else None


# ------------------------------------------------------------------ value-check candidates with a capped NS multiplier
NSCAPS = [float(c) for c in os.environ.get('BK23_NSCAPS', '3,0.3,30').split(',') if c]
NSALL = os.environ.get('BK23_NSALL') == '1'
QUICKCAPS = os.environ.get('BK23_QUICKCAPS', '1') == '1'          # extension on every box (default: only near the NS boundary)
from bk13_lb import NP as _NP, KV as _KV, AA as _AA, AB as _AB


def float_lp_cap(bx, cost_f, lev, cap):
    """bk19_lb2.float_lp with the NS multiplier (and the lower-row multiplier) bounded by cap instead of 1e4 (candidate only)."""
    Q = bx.Qp; mu = [bx.fl(x) for x in Q['mu']]; da, db = bx.dc
    nsg, nsh, nsl = L2._rows(bx); c = np.asarray(cost_f); slo, shi = lev
    A = np.zeros((_NP, 9))
    for j in range(_NP):
        A[j, int(_KV[j]) + 1] = 1.0; A[j, 3] = _AA[j]; A[j, 4] = _AB[j]; A[j, 5] = -nsg[j]; A[j, 6] = M.OFF[j]; A[j, 7] = -M.OFF[j]; A[j, 8] = nsg[j]
    obj = -np.array([mu[0], mu[1], mu[2], da, db, -nsh, slo, -shi, nsl if nsl is not None else 0.0])
    bnds = [(None, None)] * 3 + [(-1e4, 1e4)] * 2 + [(0, cap), (0, 1e4), (0, 1e4)] + [(0, cap) if nsl is not None else (0, 0)]
    r = _linprog(obj, A_ub=A, b_ub=c, bounds=bnds, method='highs')
    if r.status != 0: return None
    y = r.x
    return (y[3], y[4], max(y[5], 0.0), max(y[6], 0.0), max(y[7], 0.0), max(y[8], 0.0) if nsl is not None else 0.0)


def bound_one_ext(bx, lev, fam, k, rg, sgn, quick):
    """bk19_orun.bound_one (same candidate logic: float LP dual at the centre with least-squares slopes, theta combination,
    robust duals when not quick) with more candidate duals on boxes near the NS boundary (bx.near_ns): the bounded dual LP
    with the NS multiplier capped at each cap in NSCAPS (tried first, in that order), then the engine's (cap 1e4).
    Any dual is a valid weak-duality candidate; the record stores the duals and the replay re-certifies them."""
    vals = []; cands = []
    ths = R.THS if fam == 'U' else (None,)
    keys = [(ch, th) for ch in R.choices(bx, lev, rg) for th in ths]
    _cc = {}

    def costs(i):                                                # lazily: most patterns pass with the first combination
        if i not in _cc:
            ch, th = keys[i]
            cp, kp = cand_costs(fam, k, bx.Qp, rg, sgn, ch, th); cb, kb = cand_costs(fam, k, bx.Qb, rg, sgn, ch, th)
            _cc[i] = (ch, th, cp, kp, cb, kb)
        return _cc[i]
    caps = (NSCAPS + [None]) if (getattr(bx, 'near_ns', False) or NSALL) else [None]
    if quick and QUICKCAPS and len(caps) > 2: caps = [caps[0], None]      # lighter candidate set in quick mode
    for cap in caps:
        for ci in range(len(keys)):
            ch, th, cp, kp, cb, kb = costs(ci)
            if cap is None:
                d0, _ = L2.float_lp(bx, bx.fl(cp), lev)
            else:
                d0 = float_lp_cap(bx, bx.fl(cp), lev, cap)
            if d0 is None: continue
            d0 = [float(x) for x in d0]
            dual = L2.affine_dual(bx, d0, L2.dual_slopes(bx, cp, d0, lev))
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
    if lb < 0 and SBTHR > 0 and float(Vm.min(axis=1).max()) >= -SBTHR and float(np.max((bx.hi - bx.lo) / R.TW)) <= SBW:
        subs = sb_try(bx, lev, fam, k, rg, sgn, cands, Vm)
        if subs is not None:
            return dict(sb=subs, lb=min(e['lb'] for e in subs))
    keep = [i for i in range(len(th_)) if th_[i] > 0]
    bx.vbest = Vm[int(np.argmax(Vm.min(axis=1)))]
    return dict(c=[cands[i] for i in keep], th=[float(th_[i]) for i in keep], lb=lb)


if NSCAPS:
    R.bound_one = bound_one_ext

# ------------------------------------------------------------------ sub-box certificates
SBTHR = float(os.environ.get('BK23_SBTHR', '0.02'))
SBW = float(os.environ.get('BK23_SBW', '1.5'))            # ... and only on boxes with all widths <= SBW x the TW scales         # try sub-boxes only if the best candidate's min >= -SBTHR (0: off)


def sub_boxes(lo, hi):
    """the 16 closed sub-boxes of [lo, hi] cut at the double midpoints 0.5 (lo_i + hi_i) (their union is [lo, hi])."""
    mid = 0.5 * (lo + hi); out = []
    from bk13_lb import VMASK
    for v in VMASK:
        out.append((np.where(v == 1, mid, lo), np.where(v == 1, hi, mid)))
    return out


def sb_rg(rg):
    g = dict(rg); g['Y'] = (min(g['Y'][0], 0.0), g['Y'][1])          # the check on the sub-box's Ybar
    return g


def recentre(spec, zc_old, zc_new):
    """an affine dual component v + g.(z - zc_old) written about zc_new (any dual is a valid candidate; the record
    stores the re-centred numbers, which are what the replay certifies)."""
    if not isinstance(spec, (list, tuple)): return spec
    v, g = spec
    return [float(v + sum(float(gi) * (float(b) - float(a)) for gi, a, b in zip(g, zc_old, zc_new))), [float(x) for x in g]]


SBTOP = int(os.environ.get('BK23_SBTOP', '3'))


def sb_try(bx, lev, fam, k, rg, sgn, cands, Vm):
    """certify one (check k, pattern sgn) on the 16 sub-boxes of bx with the SBTOP best parent candidates (re-centred
    duals), theta-combined per sub-box; returns the list of sub-box entries or None."""
    pq0 = getattr(bx, 'rg_pq', None)
    if pq0 is None: return None
    rgs = sb_rg(rg)
    if fam == 'U' and not R.derived(rgs): return None
    top = list(np.argsort(-Vm.min(axis=1))[:SBTOP])
    if getattr(bx, '_sbx', None) is None:                       # the 16 sub-boxes are built once per box (reused by all patterns)
        bx._sbx = [L2.Box(slo, shi, FULL) for slo, shi in sub_boxes(bx.lo, bx.hi)]
    subs = []
    for sb in bx._sbx:
        if not np.all(np.asarray(sb.Qb['BC']['pq']) <= pq0): return None
        set_band(sb, BANDSF[fam][k])
        vals = []; cs = []
        for i in top:
            c = cands[i]
            d = [recentre(x, bx.zc, sb.zc) for x in c['d']]
            cp, kp = cand_costs(fam, k, sb.Qp, rgs, sgn, c['ch'], c['th']); cb, kb = cand_costs(fam, k, sb.Qb, rgs, sgn, c['ch'], c['th'])
            V = L2.mode1_vertices(sb, cp, kp, cb, kb, lev, d)
            vals.append(V); cs.append(dict(ch=c['ch'], th=c['th'], d=d))
            if float(V.min()) >= 0: break
        Vs = np.where(np.isfinite(np.array(vals)), np.array(vals), -1e9)
        th_, lb = BB.theta_combine(Vs)
        if not lb >= 0: return None
        keep = [i for i in range(len(th_)) if th_[i] > 0]
        subs.append(dict(c=[cs[i] for i in keep], th=[float(th_[i]) for i in keep], lb=lb))
    return subs


def sb_replay(bx, lev, fam, k, rg, sgn, rr, pq0):
    """replay of an 'sb' pattern entry: returns the least sub-box theta-combined vertex value (must be >= 0) or a negative code."""
    rgs = sb_rg(rg)
    if fam == 'U' and not R.derived(rgs): return -3.0
    subs = sub_boxes(bx.lo, bx.hi)
    if len(rr['sb']) != len(subs): return -16.0
    worst = 1e9
    for (slo, shi), ent in zip(subs, rr['sb']):
        if not ent['th'] or min(ent['th']) < 0 or sum(ent['th']) <= 0: return -6.0
        sb = L2.Box(slo, shi, FULL)
        if not np.all(np.asarray(sb.Qb['BC']['pq']) <= pq0): return -13.0
        set_band(sb, BANDSF[fam][k])
        tot = np.zeros(16)
        for c, th in zip(ent['c'], ent['th']):
            if th == 0: continue
            cp, kp = cand_costs(fam, k, sb.Qp, rgs, sgn, c['ch'], c['th']); cb, kb = cand_costs(fam, k, sb.Qb, rgs, sgn, c['ch'], c['th'])
            V = L2.mode1_vertices(sb, cp, kp, cb, kb, lev, c['d'])
            tot = dn(tot + dn(th * V))
        worst = min(worst, float(tot.min()))
        if not worst >= 0: return worst
    return worst



def band_outside(fam, k, elo, ehi):
    """True if the check's closed interval [e_k, ECAP] (e_k null: (-inf, ECAP]) misses [elo, ehi] (strictly)."""
    a, b = BANDSF[fam][k]
    if b < elo: return True
    if a is not None and a > ehi: return True
    return False


# ------------------------------------------------------------------ E^lin range of the full-band set (F2)
def elin_bounds(bx, lev, duals=None):
    """certified [min, max] of 3 (E^lin + mu0) = nsc + nsg.m over the box's full-band slice-level set -> (elo, ehi, d1, d2)."""
    Qp, Qb = bx.Qp, bx.Qb
    out = []; ds = []
    for i, mx in enumerate((False, True)):
        if duals is not None:
            val, d = L2.certified_min(bx, Qp['nsg'], Qb['nsg'], Qp['nsc'], Qb['nsc'], lev, maximize=mx, dual=duals[i])
        else:
            val, d = L2.certified_min(bx, Qp['nsg'], Qb['nsg'], Qp['nsc'], Qb['nsc'], lev, maximize=mx, slopes=True)
            if val is None: return None
            if not np.isfinite(val):
                for aff in (True, False):
                    sp = L2.robust_dual(bx, -Qp['nsg'] if mx else Qp['nsg'], -Qp['nsc'] if mx else Qp['nsc'], lev, affine=aff)
                    if sp is None: continue
                    v2, _ = L2.certified_min(bx, Qp['nsg'], Qb['nsg'], Qp['nsc'], Qb['nsc'], lev, maximize=mx, dual=sp)
                    if np.isfinite(v2): val, d = v2, sp; break
        if val is None or not np.isfinite(val): return None
        out.append(val); ds.append(d)
    elo = float((IA(out[0]) / 3.0 - MU0).lo); ehi = float((IA(out[1]) / 3.0 - MU0).hi)
    return elo, ehi, ds[0], ds[1]


# ------------------------------------------------------------------ per-box evaluation
def bands_check(bx, lev, fam, rg, todo, quick):
    BANDS = BANDSF[fam]
    """returns (passing entries [[k, entry]], failing band list, vbest of the first failing band)."""
    # phase 1: emptiness of every band of the mask (cheap); phase 2: value checks of every active band in band order;
    # after the first failure the remaining bands are checked in quick mode (no robust-dual retries).  Failing bands form
    # the children's mask; passing bands are recorded (any certified bound is valid whatever the mode).
    per = []; act = []; fails = []; vb = None; skipped = False
    for k in todo:
        set_band(bx, BANDS[k])
        t0 = time.process_time(); e = R.emptiness(bx, lev); STAT['t_empty'] += time.process_time() - t0
        if e is not None: per.append([k, dict(k='E', ed=e['ed'], lb=e['lb'])])
        else: act.append(k)
    order = act[::-1] if FAILSKIP is not None else act         # top checks (most often failing) first
    for k in order:
        if FAILSKIP is not None and fails and skipped:              # clearly failing box -> do not evaluate the rest
            fails.append(k); continue
        set_band(bx, BANDS[k])
        t0 = time.process_time()
        bx.vbest = None
        res, lb = R.all_bounds(bx, lev, fam, k, rg, quick=quick or bool(fails))
        STAT['t_band'] += time.process_time() - t0
        if res is None or lb < 0:
            if not fails: vb = getattr(bx, 'vbest', None)
            fails.append(k)
            if FAILSKIP is not None and (res is None or lb < -FAILSKIP): skipped = True
            continue
        per.append([k, dict(k='O', sg=res, lb=lb)])
    set_band(bx, FULL)
    fails = sorted(fails)
    return per, fails, vb


def _rgjson(rg):
    return {a: list(b) for a, b in rg.items() if not a.startswith('_')}


def process(fam, lo, hi, lev, inh, mask):
    """-> (records, failing bands, inheritance for the children, box)."""
    _uvc.clear(); _dubc.clear()
    if BB.marg_excluded(lo, hi): return [dict(k='X', mask=mask)], [], None, None
    if R.d_outside_R7(lo, hi): return [dict(k='Xd', mask=mask)], [], None, None
    bx = L2.Box(lo, hi, FULL)
    e = R.emptiness(bx, lev)
    if e is not None: return [dict(e, mask=mask)], [], None, bx
    recs = []; todo = list(mask)
    if len(todo) >= 3:
        t0 = time.process_time(); er = elin_bounds(bx, lev); STAT['t_er'] += time.process_time() - t0
        if er is not None:
            elo, ehi, d1, d2 = er
            out = [k for k in todo if band_outside(fam, k, elo, ehi)]
            if out:
                recs.append(dict(k='ER', mask=out, erd=[d1, d2], elo=elo, ehi=ehi))
                todo = [k for k in todo if k not in out]
    if not todo: return recs, [], None, bx
    if inh is not None:
        rg0, rd0, rlo, rhi, pq0, rc0 = inh
        bx.near_ns = rc0 is not None
        bx.rg_pq = np.asarray(pq0, float)
        if np.all(np.asarray(bx.Qb['BC']['pq']) <= pq0):
            rg = dict(rg0); rg['Y'] = (min(rg['Y'][0], 0.0), rg['Y'][1])          # the check (fix S2)
            ok = True
            if fam == 'U': ok = R.derived(rg)
            if ok:
                per, fails, vb = bands_check(bx, lev, fam, rg, todo, True)
                if per:
                    recs.append(dict(k='O', mask=[k for k, _ in per], rd=rd0, rg=_rgjson(rg0), rbox=[list(rlo), list(rhi)], bands=per, **({'rcap': rc0} if rc0 is not None else {})))
                todo = fails
                if not todo: return recs, [], None, bx
    set_band(bx, FULL)                                                           # fix S1: ranges on the full-band set
    rc = relaxed_cap(bx, lev)                                                    # or on a relaxed superset
    bx.near_ns = rc is not None
    if rc is not None: set_rcap(bx, rc)
    t0 = time.process_time(); rg, rd = R.ranges(bx, lev, R.keys_of(fam)); STAT['t_rng'] += time.process_time() - t0
    set_band(bx, FULL)
    rcx = {'rcap': rc} if rc is not None else {}
    bx.rg_pq = np.asarray(bx.Qb['BC']['pq'], float).copy()
    if rg is None: return recs, todo, None, bx
    for key, (a, b) in rg.items():
        if a > b:
            recs.append(dict(k='E2', mask=todo, key=key, rd={key: rd[key]}, lb=a - b, **rcx)); return recs, [], None, bx
    if fam == 'U' and not R.derived(rg): return recs, todo, None, bx
    new = (rg, rd, [float(x) for x in lo], [float(x) for x in hi], np.asarray(bx.Qb['BC']['pq'], float).copy(), rc)
    per, fails, vb = bands_check(bx, lev, fam, rg, todo, False)
    if per: recs.append(dict(k='O', mask=[k for k, _ in per], rd=rd, rg=_rgjson(rg), bands=per, **rcx))
    if fails: bx.vbest = vb
    return recs, fails, new, bx


# ------------------------------------------------------------------ replay (no LP)
def replay(r):
    """re-certify one record; returns (ok, margin)."""
    _uvc.clear(); _dubc.clear()
    fam = r['fam']; lo, hi, lev = np.array(r['lo']), np.array(r['hi']), tuple(r['lev'])
    kk = r.get('k'); mask = r.get('mask')
    if r.get('fail') or not mask: return False, -9
    if r.get('tg') != TMD5[:8]: return False, -14                              # record made with these very targets
    BANDS = BANDSF[fam]
    if any((not isinstance(k, int)) or k < 0 or k >= len(BANDS) for k in mask): return False, -9
    if kk == 'X': return BB.marg_excluded(lo, hi), 1.0
    if kk == 'Xd': return R.d_outside_R7(lo, hi), 1.0
    bx = L2.Box(lo, hi, FULL)
    zero = M.tconst(0.0); zv = M.tvecconst(np.zeros(M.NP))
    if kk == 'E':
        V = L2.mode1_vertices(bx, zv, zero, zv, zero, lev, r['ed']); return float(V.min()) > 0, float(V.min())
    if kk == 'E2':
        if 'rcap' in r:
            if not r['rcap'] >= BJ_F: return False, -15
            set_rcap(bx, r['rcap'])
        key = r['key']; d1, d2 = r['rd'][key]; cache = {}
        vp = O.functional_ext(bx.Qp, key, cache); vb = O.functional_ext(bx.Qb, key, cache)
        a, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, dual=d1)
        b_, _ = L2.certified_min(bx, vp, vb, zero, zero, lev, maximize=True, dual=d2)
        return a > b_, a - b_
    if kk == 'ER':
        er = elin_bounds(bx, lev, duals=r['erd'])
        if er is None: return False, -11
        elo, ehi = er[0], er[1]
        if not all(band_outside(fam, k, elo, ehi) for k in mask): return False, -12
        return True, 1.0
    if kk != 'O': return False, -8
    if sorted(k for k, _ in r['bands']) != sorted(mask) or len(set(mask)) != len(mask): return False, -7
    if 'rbox' in r:
        rlo, rhi = np.array(r['rbox'][0]), np.array(r['rbox'][1])
        if not (np.all(rlo <= lo) and np.all(rhi >= hi)): return False, -4
        rb = L2.Box(rlo, rhi, FULL)
        if not np.all(np.asarray(bx.Qb['BC']['pq']) <= np.asarray(rb.Qb['BC']['pq'])): return False, -13
    else:
        rb = bx
    if 'rcap' in r:                                                             # ranges on the relaxed superset
        if not r['rcap'] >= BJ_F: return False, -15
        set_rcap(rb, r['rcap'])
    rgc, _ = R.ranges(rb, lev, R.keys_of(fam), rec=r['rd'])
    if rb is bx: set_band(bx, FULL)
    rg = {}
    for key in R.keys_of(fam):
        a, b_ = rgc[key]; rl, rh = r['rg'][key]
        if not (rl <= a and rh >= b_): return False, -2
        rg[key] = (rl, rh)
    if 'rbox' in r: rg['Y'] = (min(rg['Y'][0], 0.0), rg['Y'][1])              # the check
    if fam == 'U' and not R.derived(rg): return False, -3
    sg = R.SIGNS if fam == 'U' else [None]
    worst = 1e9
    for k, ent in r['bands']:
        set_band(bx, BANDS[k])
        if ent['k'] == 'E':
            V = L2.mode1_vertices(bx, zv, zero, zv, zero, lev, ent['ed'])
            if not float(V.min()) > 0: return False, float(V.min())
            worst = min(worst, float(V.min())); continue
        if ent['k'] != 'O' or len(ent['sg']) != len(sg): return False, -5
        for s, rr in zip(sg, ent['sg']):
            if 'sb' in rr:                                                      # the sub-box certificate
                v = sb_replay(bx, lev, fam, k, rg, s, rr, np.asarray(rb.Qb['BC']['pq'], float))
                set_band(bx, BANDS[k])
                if not v >= 0: return False, v
                worst = min(worst, v); continue
            if not rr['th'] or min(rr['th']) < 0 or sum(rr['th']) <= 0: return False, -6
            tot = np.zeros(16)
            for c, th in zip(rr['c'], rr['th']):
                if th == 0: continue
                cp, kp = cand_costs(fam, k, bx.Qp, rg, s, c['ch'], c['th']); cb, kb = cand_costs(fam, k, bx.Qb, rg, s, c['ch'], c['th'])
                V = L2.mode1_vertices(bx, cp, kp, cb, kb, lev, c['d'])
                tot = dn(tot + dn(th * V))
            if not float(tot.min()) >= 0: return False, float(tot.min())
            worst = min(worst, float(tot.min()))
    return True, worst


# ------------------------------------------------------------------ jobs
def mkjobs(tag):
    """the 640 roots: the omkjobs roots (bk17_run's ABP x LBP grid over the WIN window, d-box dp in [-0.79, 0.99],
    dm in [0, 0.59], levels [0, 0.01] and [0.01, 1]) for U and V, without the band field; level [0.01, 1] first."""
    os.environ['BK17_GPM'] = '1'
    sv = sys.argv; sys.argv = [sys.argv[0], 'G']
    import bk17_run as RUN
    sys.argv = sv
    cells = []
    for i in range(len(RUN.ABP) - 1):
        for j in range(len(RUN.LBP) - 1):
            cells.append(([RUN.ABP[i], RUN.LBP[j], -0.79, 0.0], [RUN.ABP[i + 1], RUN.LBP[j + 1], 0.99, 0.59]))
    cells.sort(key=lambda c: abs(0.5 * (c[0][0] + c[1][0]) - 0.5) + c[0][1])
    jobs = []
    for lev in ([0.01, 1.0], [0.0, 0.01]):
        for lo, hi in cells:
            for f in 'UV':
                jobs.append([f, lo, hi, lev])
    json.dump(dict(jobs=jobs, ABP=list(RUN.ABP), LBP=list(RUN.LBP)), open(tag + '_jobs.json', 'w'))
    print('jobs', len(jobs), 'cells', len(cells))


# ------------------------------------------------------------------ pool worker
SPLITR = float(os.environ.get('BK23_SPLITR', '1'))       # the rule uses 16; the default 1 (widest normalised dimension)


def split_dim(lo, hi, bx):
    """the split rule (bk19_orun.split) with the width ratio SPLITR in place of 16: the influence-chosen dimension is
    used only if its normalised width is >= max / SPLITR.  A heuristic; it has no bearing on soundness."""
    w = hi - lo; nw = w / R.TW
    vb = getattr(bx, 'vbest', None) if bx is not None else None
    i = int(np.argmax(nw))
    if vb is not None and np.all(np.isfinite(vb)):
        from bk13_lb import VMASK
        infl = np.array([np.mean(np.abs(vb[VMASK[:, q] == 1] - vb[VMASK[:, q] == 0])) for q in range(4)])
        if infl.max() > 0:
            j = int(np.argmax(infl + 1e-12 * w))
            if nw[j] >= nw.max() / SPLITR: i = j
    return i


def claim(tag, i):
    try:
        fd = os.open(os.path.join('claims', '%s_%d.claim' % (tag, i)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd); return True
    except FileExistsError:
        return False


def main(tag, wk, budget):
    jobs = json.load(open(tag + '_jobs.json'))['jobs']
    os.makedirs('claims', exist_ok=True)
    base = '%s_w%d' % (tag, wk)
    stf = base + '.state.json'; rf = base + '.jsonl'; of = base + '_open.jsonl'
    if os.path.exists(stf):
        st = json.load(open(stf))
        for fn, off in ((rf, st['roff']), (of, st['ooff'])):
            if os.path.exists(fn):
                with open(fn, 'r+b') as f: f.truncate(off)
        cur = st['cur']; stack = [(np.array(a), np.array(b), tuple(l), d, None, m, p) for a, b, l, d, m, p in st['stack']]
        done = st['done']; stats = st['stats']
    else:
        cur = None; stack = []; done = []
        stats = dict(processed=0, recs=0, open=0, cpu=0.0, kinds={}, maxdepth=0, fails={})
    out = open(rf, 'a'); openf = open(of, 'a')
    t0 = time.time(); c0 = time.process_time(); last = t0; cpu_base = stats['cpu']
    print('worker', base, 'pid', os.getpid(), 'targets', os.path.abspath(TGF), 'md5', TMD5, flush=True)
    if stats.setdefault('tmd5', TMD5) != TMD5: raise SystemExit('targets md5 differs from the checkpoint: ' + stats['tmd5'])

    def checkpoint():
        out.flush(); openf.flush()
        stats['cpu'] = cpu_base + time.process_time() - c0
        st = dict(cur=cur, stack=[(a.tolist(), b.tolist(), list(l), d, m, p) for a, b, l, d, _, m, p in stack], done=done,
                  stats=stats, roff=out.tell(), ooff=openf.tell())
        tmp = stf + '.tmp'; json.dump(st, open(tmp, 'w')); os.replace(tmp, stf)
    order = json.load(open(os.environ["BK22_ORDER"])) if os.environ.get("BK22_ORDER") else list(range(len(jobs)))
    nexti = 0; stopped = False
    while True:
        if time.time() - t0 > budget or os.path.exists(tag + '.stop') or os.path.exists(base + '.stop'): stopped = True; break
        if cur is None:
            while nexti < len(order) and not claim(tag, order[nexti]): nexti += 1
            if nexti >= len(order): break
            cur = order[nexti]; fam, a, b, l = jobs[cur]
            stack = [(np.array(a, float), np.array(b, float), tuple(l), 0, None, list(ALLF[fam]), '')]
            checkpoint()
        if not stack:
            done.append(cur); fin = cur; cur = None; checkpoint()
            open(os.path.join('claims', '%s_%d.done' % (tag, fin)), 'w').close()
            print('%7.0fs job %d DONE' % (time.time() - t0, fin), flush=True)
            continue
        fam = jobs[cur][0]
        lo, hi, lev, dep, inh, mask, path = stack.pop()
        stats['processed'] += 1; stats['maxdepth'] = max(stats['maxdepth'], dep)
        try:
            recs, fails, ninh, bx = process(fam, lo, hi, lev, inh, mask)
        except Exception as ex:                       # no record is written; the box is split (or left open at depth 36)
            recs, fails, ninh, bx = [], list(mask), None, None
            stats['kinds']['exc'] = stats['kinds'].get('exc', 0) + 1
        for rec in recs:
            stats['recs'] += 1; stats['kinds'][rec['k']] = stats['kinds'].get(rec['k'], 0) + 1
            out.write(json.dumps(dict(fam=fam, job=cur, path=path, tg=TMD5[:8], lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), **rec)) + '\n')
        if fails:
            for k in fails: stats['fails'][str(k)] = stats['fails'].get(str(k), 0) + 1
            if dep >= R.MAXDEPTH:
                stats['open'] += 1
                openf.write(json.dumps(dict(fam=fam, job=cur, path=path, lo=lo.tolist(), hi=hi.tolist(), lev=list(lev), mask=fails)) + '\n')
            else:
                i = split_dim(lo, hi, bx)
                mid = 0.5 * (lo[i] + hi[i])
                h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
                stack.append((l2, hi, lev, dep + 1, ninh, fails, path + '%db' % i))
                stack.append((lo, h1, lev, dep + 1, ninh, fails, path + '%da' % i))
        if time.time() - last > 60:
            last = time.time(); checkpoint()
            print('%7.0fs job %s stack %d done %d %s' % (time.time() - t0, cur, len(stack), len(done), json.dumps(stats)), flush=True)
    checkpoint(); out.close(); openf.close()
    print(('STOPPED' if stopped else 'DONE'), base, 'job', cur, 'done', len(done), json.dumps(stats), flush=True)
    if not stopped: open(base + '.done', 'w').close()


if __name__ == '__main__':
    if sys.argv[1] == 'mkjobs':
        mkjobs(sys.argv[2])
    elif sys.argv[1] == 'replay':
        stride, off = int(sys.argv[2]), int(sys.argv[3]); files = sys.argv[4:]
        bad = n = 0; i = -1; kinds = {}; minlb = {}; t0 = time.time()
        print('replay stride', stride, 'offset', off, 'targets', os.path.abspath(TGF), 'md5', TMD5, flush=True)
        for fn in files:
            for l in open(fn):
                i += 1
                if i % stride != off: continue
                if os.path.exists('replay.stop'): break
                r = json.loads(l); n += 1; kk = r.get('k'); kinds[kk] = kinds.get(kk, 0) + 1
                try: ok, lb = replay(r)
                except Exception as ex: ok, lb = False, -10; print('EXC', fn, i, repr(ex)[:200], flush=True)
                minlb[kk] = min(minlb.get(kk, 1e9), lb)
                if not ok: bad += 1; print('FAIL', fn, i, kk, r['fam'], r.get('job'), r.get('path'), r.get('mask'), lb, flush=True)
                if n % 20000 == 0: print('progress', n, 'records', bad, 'failures', '%.0fs' % (time.time() - t0), flush=True)
        print('checked %d kinds %s failures %d min margins %s' % (n, kinds, bad, minlb), flush=True)
    else:
        main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
