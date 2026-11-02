# A-the own exact cover of pilot-5 roots (independent of bk23_cover.py): per root and per check, the certifying record boxes
# must tile the root box (tree walk over the recorded bisection paths + exact volume sum in Fractions); record fields checked.
import sys, json, glob, hashlib
from fractions import Fraction as F
run = sys.argv[1]; tgf = sys.argv[2]
TG = json.load(open(tgf)); TMD5 = hashlib.md5(open(tgf, 'rb').read()).hexdigest()
NCH = {'U': len(TG['U']), 'V': len(TG['V'])}
jobs = json.load(open(run + '/T_jobs.json'))['jobs']
print('T_jobs md5', hashlib.md5(open(run + '/T_jobs.json', 'rb').read()).hexdigest(), 'targets md5', TMD5)
cert = {}; box = {}; nrec = 0; badfield = 0; badgeo = 0; kinds = {}
def pbox(lo, hi, p):
    lo = list(lo); hi = list(hi)
    for t in range(0, len(p), 2):
        i = int(p[t]); m = 0.5 * (lo[i] + hi[i])
        if p[t + 1] == 'a': hi[i] = m
        elif p[t + 1] == 'b': lo[i] = m
        else: raise ValueError(p)
    return lo, hi
for fn in sorted([g for g in glob.glob(run + '/T_w*.jsonl') if not g.endswith('_open.jsonl')]):
    for l in open(fn):
        r = json.loads(l); nrec += 1; kinds[r['k']] = kinds.get(r['k'], 0) + 1
        jb = r['job']; f, rlo, rhi, lev = jobs[jb]
        if r['fam'] != f or list(r['lev']) != list(lev) or r['tg'] != TMD5[:8] or r.get('fail'): badfield += 1
        if any((not isinstance(k, int)) or k < 0 or k >= NCH[f] for k in r['mask']) or not r['mask']: badfield += 1
        lo, hi = pbox(rlo, rhi, r['path'])
        if lo != r['lo'] or hi != r['hi']: badgeo += 1
        s = cert.setdefault(jb, {}).setdefault(r['path'], [])
        s.extend(r['mask'])
print('records', nrec, kinds, 'bad fields', badfield, 'record box != path box', badgeo)
want = [int(x) for x in sys.argv[3].split(',')]
for jb in want:
    f, rlo, rhi, lev = jobs[jb]; C = cert.get(jb, {})
    dup = sum(len(v) - len(set(v)) for v in C.values())
    kids = {}
    for p in C:
        for t in range(0, len(p), 2): kids.setdefault(p[:t], set()).add(p[t])
    bad_dim = sum(1 for v in kids.values() if len(v) > 1)
    rvol = 1
    for i in range(4): rvol *= F(rhi[i]) - F(rlo[i])
    res = []
    for k in range(NCH[f]):
        gaps = 0; over = 0; vol = F(0); nleaf = 0
        stack = [('', False)]
        while stack:
            p, covered = stack.pop()
            here = k in C.get(p, [])
            if here and covered: over += 1
            if here and not covered:
                lo, hi = pbox(rlo, rhi, p); v = 1
                for i in range(4): v *= F(hi[i]) - F(lo[i])
                vol += v; nleaf += 1
            cov = covered or here
            if p in kids:
                d = next(iter(kids[p])); stack.append((p + d + 'a', cov)); stack.append((p + d + 'b', cov))
            elif not cov: gaps += 1
        res.append((k, nleaf, gaps, over, vol == rvol))
    ok = all(g == 0 and o == 0 and v for _, _, g, o, v in res) and dup == 0 and bad_dim == 0
    print('job %d fam %s lev %s: nodes %d, dup %d, inconsistent split dims %d; per check (k, leaves, gaps, overlaps, exact volume): %s -> %s'
          % (jb, f, lev, len(C), dup, bad_dim, res, 'COVERED' if ok else 'NOT COVERED'))
