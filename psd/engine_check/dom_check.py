"""A-ENG: (1) the 640 roots cover the dm >= 0 half of WIN x (dp, dm)-image of the R7 square, exactly (Fractions);
(2) read-only scan of the record files (snapshot of complete lines): every record box has dm_lo >= 0, kinds, min 'O' lb."""
import os, sys, json, glob
from fractions import Fraction as Fr
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
J = json.load(open(os.path.join(S, 'run', 'T_jobs.json')))
jobs, ABP, LBP = J['jobs'], J['ABP'], J['LBP']
print('jobs', len(jobs), 'ABP', ABP[0], ABP[-1], len(ABP), 'LBP', LBP[0], LBP[-1], len(LBP))
assert ABP[0] == float("0.3") and ABP[-1] == float("0.7") and LBP[0] == 0.0 and LBP[-1] == 0.25   # = bk16_margx window (doubles A0, A1, L1)
assert all(Fr(a) < Fr(b) for a, b in zip(ABP, ABP[1:])) and all(Fr(a) < Fr(b) for a, b in zip(LBP, LBP[1:]))
cells = {}
for f, lo, hi, lev in jobs:
    key = (f, tuple(lev))
    cells.setdefault(key, set()).add((lo[0], hi[0], lo[1], hi[1]))
    assert lo[2] == -0.79 and hi[2] == 0.99 and lo[3] == 0.0 and hi[3] == 0.59, (lo, hi)
want = set((ABP[i], ABP[i + 1], LBP[j], LBP[j + 1]) for i in range(len(ABP) - 1) for j in range(len(LBP) - 1))
for key, cs in cells.items():
    assert cs == want, key
print('families x levels:', sorted(cells), 'each = full ABP x LBP grid (%d cells)' % len(want))
levs = sorted(set(tuple(j[3]) for j in jobs)); print('levels', levs)
# tau = 5/2 - (3/2) m1, m1 = al(1-la)/((1-la)(2-al)+la): increasing in al, decreasing in la on WIN
def m1(al, la): return al * (1 - la) / ((1 - la) * (2 - al) + la)
tmax = Fr(5, 2) - Fr(3, 2) * m1(Fr(ABP[0]), Fr(1, 4)); tmin = Fr(5, 2) - Fr(3, 2) * m1(Fr(ABP[-1]), Fr(0))
DLO, DHI = Fr('0.74898'), Fr('1.33514')     # outer bounds of exp(-+0.28903) (bk16_model)
import math
assert DLO <= Fr(math.exp(-0.28903)) - Fr(1, 10 ** 9) and DHI >= Fr(math.exp(0.28903)) + Fr(1, 10 ** 9)
dpl, dph, dmm = 2 * DLO - tmax, 2 * DHI - tmin, DHI - DLO
print('tau in [%.9f, %.9f]; R7 needs dp in [%.9f, %.9f], |dm| <= %.9f' % (tmin, tmax, dpl, dph, dmm))
assert dpl >= Fr('-0.79') and dph <= Fr('0.99') and dmm <= Fr('0.59')
# monotonicity of m1 (sign of the partial derivatives on WIN), checked symbolically by hand; numeric spot check:
for a in (0.3, 0.5, 0.7):
    for l in (0.0, 0.1, 0.25):
        assert m1(Fr(a), Fr(l)) >= m1(Fr(ABP[0]), Fr(1, 4)) and m1(Fr(a), Fr(l)) <= m1(Fr(ABP[-1]), Fr(0))
print('ROOT DOMAIN OK')
# (2) the records (read-only snapshot)
tot = 0; neg = 0; kinds = {}; minlb = {}; bad = 0; fams = {}
for fn in sorted(glob.glob(os.path.join(S, 'run', 'T_w*.jsonl'))):
    if fn.endswith('_open.jsonl'): continue
    size = os.path.getsize(fn)
    with open(fn, 'rb') as f:
        data = f.read(size)
    for line in data.split(b'\n'):
        if not line.strip(): continue
        try: r = json.loads(line)
        except Exception: bad += 1; continue
        tot += 1
        if r['lo'][3] < 0.0: neg += 1
        k = r.get('k'); kinds[k] = kinds.get(k, 0) + 1; fams[r.get('fam')] = fams.get(r.get('fam'), 0) + 1
        if 'lb' in r and r['lb'] is not None: minlb[k] = min(minlb.get(k, 1e9), r['lb'])
        for kb, e in r.get('bands', []):
            kk = 'band-' + e['k'] + '-' + r['fam']
            minlb[kk] = min(minlb.get(kk, 1e9), e['lb'])
    print(os.path.basename(fn), size, flush=True)
print('BK24 records', tot, 'kinds', kinds, 'fams', fams, 'dm_lo < 0:', neg, 'unparsed (partial) lines', bad, 'min lb', minlb)
