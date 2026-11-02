"""comp_r2.py : (C-LR) by the unchanged S\bk23_comp.py (imported, not modified), with A-the R2's assertion
0 < e < B (exact Fractions) for every finite lower end e of both families, and the max printed exactly as a Fraction.
Usage (in S\run): python ../../comp_r2.py targets.json"""
import sys, os, json, hashlib
sys.path.insert(0, os.path.abspath('..'))
import bk23_comp as C
from fractions import Fraction as Fr
raw = open(sys.argv[1], 'rb').read()
print('targets', sys.argv[1], 'md5', hashlib.md5(raw).hexdigest(), '; bk23_comp.py md5', hashlib.md5(open(C.__file__, 'rb').read()).hexdigest())
U, V = C.load(json.loads(raw))
fin = [(fam, e) for fam, F in (('U', U), ('V', V)) for e, _, _ in F if e is not None]
for fam, e in fin:
    assert Fr(0) < e < C.B, ('R2 violated', fam, e)
print('R2: 0 < e < B holds (exact) for all %d finite lower ends: %s; B = %s' % (len(fin), [float(e) for _, e in fin], C.B))
w, arg = C.compose(U, V)
print('max Phi\'/Phi_0 exact = %s/%s = %.12f' % (w.numerator, w.denominator, float(w)))
print('(C-LR) HOLDS' if w < 1 else '(C-LR) FAILS')
