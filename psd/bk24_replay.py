"""bk24_replay.py : the replay loop of bk23_orun.py's __main__ (same verifier: bk23_orun.replay, unchanged, default
environment, memoised DUB on), with the least margin kept per (family, kind) and a json summary.
Usage (in run/): python ../bk24_replay.py <stride> <offset> <summary.json> <record files...>"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
assert os.environ.get('BK23_NODUBCACHE') != '1'                      # A-the R3
import bk23_orun as D
stride, off, summ = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; files = sys.argv[4:]
bad = n = 0; i = -1; kinds = {}; minlb = {}; arg = {}; fails = []; t0 = time.time()
print('replay stride', stride, 'offset', off, 'targets', os.path.abspath(D.TGF), 'md5', D.TMD5, 'verifier', os.path.abspath(D.__file__), flush=True)
for fn in files:
    for l in open(fn):
        i += 1
        if i % stride != off: continue
        r = json.loads(l); n += 1; kk = (r['fam'], r.get('k')); kinds[str(kk)] = kinds.get(str(kk), 0) + 1
        try: ok, lb = D.replay(r)
        except Exception as ex: ok, lb = False, -10; print('EXC', fn, i, repr(ex)[:200], flush=True)
        if lb < minlb.get(str(kk), 1e9): minlb[str(kk)] = lb; arg[str(kk)] = (fn, i, r.get('job'), r.get('path'))
        if not ok:
            bad += 1; fails.append((fn, i, r.get('job'), r.get('path'))); print('FAIL', fn, i, kk, r.get('job'), r.get('path'), r.get('mask'), lb, flush=True)
        if n % 5000 == 0: print('progress', n, 'records', bad, 'failures', '%.0fs' % (time.time() - t0), flush=True)
res = dict(files=files, stride=stride, offset=off, checked=n, kinds=kinds, failures=bad, fails=fails, minlb=minlb, argmin=arg,
           tmd5=D.TMD5, seconds=time.time() - t0)
json.dump(res, open(summ, 'w'))
print('checked %d kinds %s failures %d min margins %s' % (n, kinds, bad, minlb), flush=True)
