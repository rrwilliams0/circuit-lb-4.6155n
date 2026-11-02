r"""bk19_fbar.py : interval certificate of Lemma FBAR's constant: on N, |f(pi)| <= 0.48287 for every
u-profile pi, f(pi) = r_a (1 + n_a g_a(pi)) - r_b (1 + n_b g_b(pi)).
Argument (Lemma FBAR, checked PASS in ): E_u <= B (Lemma NS) and E_u >= c_Psi (l_a^2 + l_b^2) (Lemma RAT)
put (l_a, l_b) in the disc of radius R0 = sqrt(B / c_Psi); for fixed (r_a, r_b) f is affine in (n_a, n_b) = (e^{-l_a/2},
e^{-l_b/2}), so |f| is maximal on the image of the boundary circle; f is affine in r_a and r_b separately, so the four corners
of [1/N_max, 1/N0]^2 suffice (r_x in that interval by THIN(c)).  A larger radius only enlarges the disc, so R0 is
rounded UP; 1/N0, 1/N_max are enclosed.  For each of 20 profiles x 4 corners, theta in [0, 2 pi] is bisected until the
interval enclosure of |f| on every theta-interval is <= FB = 0.48287.  mpmath.iv (53 bits, outward)."""
from mpmath import iv, mp
import itertools
iv.prec = 53
FB = iv.mpf('0.48287')
rho = 2 * iv.sqrt(2) / 3
MU0 = (1 + 2 * iv.sqrt(2)) / 4
CTH = iv.mpf('0.9595702342051')            # c_Theta rounded up (bk16_model.CTH)
B = CTH - MU0                               # upper end used
CPSI = iv.mpf('0.029385')                   # Lemma RAT constant (lower bound; checked: own 0.0293883 >= 0.029385)
R0 = iv.sqrt(B / CPSI); R0 = iv.mpf(R0.b)   # rounded up
N0 = iv.sqrt(2 + 2 * rho)
EMAX = iv.mpf('0.9526711763')               # Lemma E0 / THIN(c) upper end of every edge cosine
NMAX = iv.sqrt(2 + 2 * EMAX)
rlo = 1 / NMAX; rhi = 1 / N0
corners = [(rlo.a, rlo.a), (rlo.a, rhi.b), (rhi.b, rlo.a), (rhi.b, rhi.b)]
TWO_PI = 2 * iv.pi
prof = [(k, a, b) for k, a, b in itertools.product((-1, 0, 1), repeat=3) if min(k, a, b) == -1] + [(1, 1, 1)]
g = {-1: iv.sqrt(2), 0: iv.mpf(1), 1: 1 / iv.sqrt(2)}
worst = 0; nint = 0
for (k, ta, tb) in prof:
    for (ra, rb) in corners:
        ra = iv.mpf(ra); rb = iv.mpf(rb)
        stack = [(iv.mpf(0), iv.mpf(TWO_PI.b))]
        while stack:
            t0, t1 = stack.pop()
            th = iv.mpf([t0.a, t1.b])
            la = R0 * iv.cos(th); lb = R0 * iv.sin(th)
            f = ra * (1 + iv.exp(-la / 2) * g[ta]) - rb * (1 + iv.exp(-lb / 2) * g[tb])
            m = max(abs(f.a), abs(f.b))
            if m <= FB.a: nint += 1; worst = max(worst, m); continue
            if t1.b - t0.a < 1e-12: raise SystemExit('FAIL at profile %s corner %s theta %s value %s' % ((k, ta, tb), (ra, rb), th, f))
            mid = iv.mpf((t0.a + t1.b) / 2)
            stack.append((t0, mid)); stack.append((mid, t1))
print('FBAR CERTIFIED: |f| <= %s on all 20 profiles x 4 corners; R0 = %s; %d theta-intervals; max enclosure %.9f' % (FB, R0, nint, float(worst)))
