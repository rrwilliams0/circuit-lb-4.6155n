"""bk19_qrun.py : bk17_run.py (domain Q or G, the per-worker static partition) with bk19_fast's speed-ups.
Usage: python bk19_qrun.py <Q|G> <tag> <worker k> <workers W> <budget s>   (same arguments and files as bk17_run.py)."""
import sys, os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
import bk17_run as RUN
import bk19_fast
bk19_fast.install2()
# Theorem CS-K'' (the notes): per-cell targets T_i with T_i + T_{11-i} >= 0 and T_i >= -H_lo(C_{11-i}).  Cell C_11 uses
# T_11 = -1.5e-4: its partner cell C_0 has H >= 7.8282e-4 (LOC2-K(c) record, cell 0; gap after cell 0: 7.841e-4)
# and every 'G' record of C_0 has target 1.5e-4 (first round).  All other cells keep the targets.
T11 = -1.5e-4
_qt = RUN.q_target
def q_target(lo, hi):
    if RUN.DOM == 'Q' and RUN.q_cell(lo, hi) == 11: return T11
    return _qt(lo, hi)
RUN.q_target = q_target
if __name__ == '__main__':
    tag = sys.argv[2]; wk = int(sys.argv[3]); W = int(sys.argv[4]); budget = float(sys.argv[5])
    print('BK19 wrapper (bk19_fast installed); domain', RUN.DOM, 'levels', RUN.LEVELS, 'KILL', RUN.M.KILL.hi, 'TQ', RUN.TQ is not None, flush=True)
    RUN.run(tag, wk, W, budget)
    import json
    st = json.load(open('%s_w%d.state.json' % (tag, wk)))
    if not st['queue']: open('%s_w%d.done' % (tag, wk), 'w').close()
