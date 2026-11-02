"""replay10.py : deterministic re-execution of recorded the decisions. Each record's box is re-run through
bk10_run.try_close10 (with its NUMERICAL corner key recomputed); the outcome must equal the record (closure rule, or None
for a SPLIT record).  usage: python replay10.py N seed rec_W0.jsonl rec_W1.jsonl"""
import sys, json, time
import numpy as np
import bk10_run as R
R.BB.setup(1e-4)
recs = []
for f in sys.argv[3:]: recs += [json.loads(l) for l in open(f)]
rng = np.random.default_rng(int(sys.argv[2])); sel = [recs[i] for i in rng.choice(len(recs), int(sys.argv[1]), replace=False)]
same = 0; t0 = time.time(); tally = {}
for r in sel:
    lo = np.array(r['lo']); hi = np.array(r['hi'])
    out, _ = R.try_close10(lo, hi, R.BB.margin_est(lo, hi))
    want = None if r['r'] == 'SPLIT' else r['r']
    ok = (out == want); same += ok; tally[(r['r'], ok)] = tally.get((r['r'], ok), 0) + 1
    if not ok: print('DIFF', r['r'], out, r['lo'], r['hi'])
print('replayed %d records: identical %d (%.0fs) %s' % (len(sel), same, time.time() - t0, tally))
