r"""bk19_joint.py : the JOINT/Emin cap and the marginal-level exclusion for G* (Lemma CAP of the notes).

Lemma CAP.  Let m be a point of N with u-star m_u (marginal p) and mirror star m^sigma_u (marginal p^sigma).  By Lemma JOINT
 E_u + w E_v <= B with w = 0.9714701 >= W_LO = 0.97147 and B = c_Theta - mu0; E_v = E(m^sigma_u) >= Emin(p^sigma)
and E^lin <= E (Lemma L).  Hence
  (a) Emin(p) + W_LO Emin(p^sigma) <= B: if Emin_lo(P) + W_LO Emin_lo(P^sigma) > B for a marginal cell P, no point of N has its
      u-star marginal in P ('XJ' closure of every box over P);
  (b) E^lin(m_u) <= B - W_LO Emin_lo(P^sigma), i.e. the NS row of the slice LP may use c_Theta' = c_Theta - W_LO Emin_lo(P^sigma)
      (G* then certifies tau_cap(p) := inf F over the capped slice set, and CKS*_cap, tau_cap(p) + tau_cap(p^sigma) >= 0, gives
      (K) on N exactly as CKS* does, since the u-star and the mirror star of every point of N lie in the capped sets).
Emin_lo is the certified table of bk19_emin.py over the root marginal cells of bk17_run (ABP x LBP; the grid is mirror-symmetric:
cell (i, j) <-> (15 - i, j)).
"""
import json, os, bisect
from ivarith import IA, up
import bk16_model as M
from bk13_lb import MU0
import bk17_run as RUN
W_LO = 0.97147
CTH0 = M.CTH
B_HI = float((M.CTH - MU0).hi)
_T = None


def table(files=None):
    global _T
    if _T is not None and files is None: return _T
    files = files or [os.environ.get('BK19_EMIN_A', 'emin_A.json'), os.environ.get('BK19_EMIN_B', 'emin_B.json')]
    T = {}
    for fn in files:
        for r in json.load(open(fn)):
            a0, a1, l0, l1 = r['cell']
            i = RUN.ABP.index(a0); j = RUN.LBP.index(l0)
            assert RUN.ABP[i + 1] == a1 and RUN.LBP[j + 1] == l1
            T[(i, j)] = r['emin_lo']
    _T = T
    return T


def root_of(lo, hi):
    i = bisect.bisect_right(RUN.ABP, lo[0]) - 1; j = bisect.bisect_right(RUN.LBP, lo[1]) - 1
    assert RUN.ABP[i] <= lo[0] and hi[0] <= RUN.ABP[i + 1] and RUN.LBP[j] <= lo[1] and hi[1] <= RUN.LBP[j + 1]
    return i, j


def emins(i, j):
    T = table()
    eu = T.get((i, j), 0.0); ev = T.get((len(RUN.ABP) - 2 - i, j), 0.0)
    return max(eu, 0.0), max(ev, 0.0)


def excluded(i, j):
    eu, ev = emins(i, j)
    lhs = IA(eu) + IA(W_LO) * ev
    return float(lhs.lo) > B_HI


def cth_cap(i, j):
    eu, ev = emins(i, j)
    return float((CTH0 - IA(W_LO) * ev).hi)
