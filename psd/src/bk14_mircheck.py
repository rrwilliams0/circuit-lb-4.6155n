"""check Lemma MIR's frame identities (joint frame v-quantities vs the normalised mirror star)."""
import numpy as np, math
from bk11_core import *
import bk12_star as S
from bk14_sides import *
from bk14_sym import ustar_scal, mirror_star
rng = np.random.default_rng(2); worst = {}
for _ in range(200):
    m = rng.dirichlet(0.4 * np.ones(NT)); mu_, mv_ = S.split(m)
    Vv = vside(mv_); ss = ustar_scal(mirror_star(mv_)); V = Vv['V']
    mv = Vv['m']; b, Q, Q1, ps = Vv['b'], Vv['Q'], Vv['Q1'], Vv['psip']
    c1, cm, c0 = KV == 1, KV == -1, KV == 0; Qh = Q - Q1
    W2 = lambda p, q: p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
    chk = {
      'Xv=-Xs/2': (Vv['Xv'], -ss['X'] / 2), 'psi=V phis': (Vv['psi'], V * ss['phi']), 'Bbar=Abars/V': (np.abs(b[cm]).max(), ss['Abar'] / V),
      'L=Ks': ((mv[c1] * np.abs(b[c1])).sum(), ss['K']), 'B0=A0s/2V': ((mv[c0] * b[c0] ** 2).sum(), ss['A0'] / (2 * V)),
      "L'=2V K's": ((mv[c1] * np.abs(ps[c1] - 1)).sum(), 2 * V * ss['Kp']), 'Psibar=2Phibs': (ps[cm].max(), 2 * ss['Phib']),
      'Psi0=2V Phi0s': ((mv[c0] * ps[c0] ** 2).sum(), 2 * V * ss['Phi0']),
      'w11v=2/V w11s': (max(abs(W2(Q[i], Q[j])) for i in np.where(cm)[0] for j in np.where(cm)[0]), 2 / V * ss['w11']),
      'Rv=2V Rs': (sum(mv[i] * mv[j] * abs(W2(Qh[i], Qh[j])) for i in np.where(c1)[0] for j in np.where(c1)[0]), 2 * V * ss['R']),
      'R0v=R0s': (sum(mv[i] * mv[j] * W2(Q[i], Q[j]) ** 2 for i in np.where(c0)[0] for j in np.where(c0)[0]), ss['R0']),
      'Yv=2Ys': (sum(mv[j] * max(abs(W2(Qh[j], Q[i])) for i in np.where(cm)[0]) for j in np.where(c1)[0]), 2 * ss['Y']),
      'w10v=sqrt2/V w10s': (max(abs(W2(Q[i], Q[j])) for i in np.where(cm)[0] for j in np.where(c0)[0]), math.sqrt(2) / V * ss['w10']),
      'Rm0v=sqrt2 V Rm0s': (sum(mv[i] * mv[j] * abs(W2(Qh[i], Q[j])) for i in np.where(c1)[0] for j in np.where(c0)[0]), math.sqrt(2) * V * ss['Rm0'])}
    for k, (x, y) in chk.items(): worst[k] = max(worst.get(k, 0), abs(x - y) / max(1e-12, abs(y)))
for k, v in worst.items(): print('%-20s max rel err %.1e' % (k, v))
