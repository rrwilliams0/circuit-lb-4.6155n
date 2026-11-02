"""bk16_kink.py : Lemma KINK-K on Q x D_loc, both sides, by bisection over boxes of (alpha, lam, dp, dm).
Each closed box records its range duals and eta; replay with bk16_replay_kink.py.  CERTIFIED (interval arithmetic)."""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import sys, json, itertools, time
import numpy as np
import bk16_loc as LC
QA = (0.47, 0.53); QL = (0.0, 0.005); WP, WM = 0.01, 0.08
S1 = float(os.environ.get('BK16_S1', '0.005'))


def run(tag, n=(12, 2, 2, 8), maxdepth=int(os.environ.get("BK16_KDEPTH", "5"))):
    out = open(tag + '_kink.jsonl', 'w')
    res = dict(boxes=0, rcmin=1.0, P0=0.0, slope=1.0, fails=0)
    for side in (+1, -1):
        dpr = (0.0, WP) if side > 0 else (-WP, 0.0)
        rng_ = [QA, QL, dpr, (-WM, WM)]
        grids = [np.linspace(r[0], r[1], k + 1) for r, k in zip(rng_, n)]
        for g, r in zip(grids, rng_): g[0], g[-1] = r[0], r[1]
        stack = []
        for idx in itertools.product(*[range(k) for k in n]):
            stack.append((np.array([grids[i][idx[i]] for i in range(4)]), np.array([grids[i][idx[i] + 1] for i in range(4)]), 0))
        while stack:
            lo, hi, dep = stack.pop()
            r = LC.kink_check(lo, hi, side, S1)
            if r['ok']:
                res['boxes'] += 1; res['rcmin'] = min(res['rcmin'], r['rcmin']); res['P0'] = max(res['P0'], r['P0']); res['slope'] = min(res['slope'], r['slope'])
                out.write(json.dumps(dict(side=side, lo=lo.tolist(), hi=hi.tolist(), eta=r['eta'], rd=r['rd'], rcmin=r['rcmin'], P0=r['P0'], slope=r['slope'])) + '\n')
                continue
            if dep >= maxdepth:
                res['fails'] += 1; out.write(json.dumps(dict(side=side, lo=lo.tolist(), hi=hi.tolist(), DELEGATE=1, FAIL=r.get('why', ''), rcmin=r.get('rcmin'), slope=r.get('slope'))) + '\n'); continue
            w = (hi - lo) / np.array([QA[1] - QA[0], QL[1] - QL[0], WP, 2 * WM]); i = int(np.argmax(w)); mid = 0.5 * (lo[i] + hi[i])
            h1 = hi.copy(); h1[i] = mid; l2 = lo.copy(); l2[i] = mid
            stack += [(lo, h1, dep + 1), (l2, hi, dep + 1)]
        print('side', side, res, flush=True)
    out.close(); print('KINK-K', tag, res, 's1', S1, flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
