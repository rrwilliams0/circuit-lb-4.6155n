# exact check: mirror of run cell i inside LOC2-K(c) cell 11-i (alpha coordinates, exact rationals of the doubles)
from fractions import Fraction as F
import numpy as np, json
A = list(np.round(np.linspace(0.47, 0.53, 13), 10)); A[0], A[-1] = 0.47, 0.53
X = [F(float(a)) for a in A]
L = []
for j in range(12):
    a0 = 0.47 + 0.005 * j - 0.5; a1 = a0 + 0.005
    if j == 0: a0 = 0.47 - 0.5
    if j == 11: a1 = 0.53 - 0.5
    L.append((F(a0) + F(1, 2), F(a1) + F(1, 2)))
for j in range(11):
    print('LOC2 cell', j, 'upper', float(L[j][1] - F(1,2)), 'next lower', float(L[j+1][0]-F(1,2)), 'gap' if L[j][1] < L[j+1][0] else 'ok', float(L[j+1][0]-L[j][1]))
for i in range(12):
    lo, hi = 1 - X[i + 1], 1 - X[i]; j = 11 - i
    ok = L[j][0] <= lo and hi <= L[j][1]
    print('run cell', i, '[%r, %r]' % (float(X[i]), float(X[i+1])), 'mirror in LOC2 cell', j, ok, 'excess lo %.3e hi %.3e' % (float(L[j][0] - lo), float(hi - L[j][1])))
print('run cell bounds exact vs decimal:', [float(X[i] - F(47,100) - F(5,1000)*i) for i in range(13)])
