"""bk17_cfg.py : configuration layer over the engine .
Import this module FIRST in every the driver and verifier.

  * kappa from pi_lo = 0.82 (CERTIFIED by an independent program: <folder>, record pi082.json,
    21,924,095 boxes, 0 failures; mpmath re-closure pi_mp082.log, min 0.8200004):
        KILL = 1/(N0^4 * 0.82) = 0.0807729462517..., N0^4 = 4 (1 + 2 sqrt2/3)^2, rounded UP to 0.08077294626.
    Selected by the environment variable BK17_PILO in {'0.815', '0.82'} (default '0.82').
  * tau0 = 1.5e-4 (Theorem CS-K with LOC2-K (b): H >= -1.5e-4), units 4 mu0 s.
"""
import os
for v in ('OMP', 'OPENBLAS', 'MKL'): os.environ[v + '_NUM_THREADS'] = '1'
from ivarith import IA
from bk13_lb import I
import bk16_model as M
PILO = os.environ.get('BK17_PILO', '0.82')
if PILO == '0.82':
    M.KILL = I('0.08077294626')
elif PILO == '0.815':
    M.KILL = I('0.08126848580')
else:
    raise ValueError(PILO)
import bk16_loc as LC
LC.KILL = M.KILL
TAU0 = 1.5e-4
import bk16_bnb as BB
BB.TAU0 = TAU0
