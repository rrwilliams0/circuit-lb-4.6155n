r"""run_checks.py: runs every computer check of the proof that does not need the large record files (about 10 minutes,
single-threaded), and compares each output with the value stated in proof.md.

  python -B run_checks.py <work folder outside this repository>

Each step runs one script of this repository in a child process (single-threaded: OMP/MKL/OPENBLAS_NUM_THREADS = 1) and
tests its output for the lines that proof.md quotes. Nothing is written inside the repository: the few checkers that write
next to their data run on copies made in the work folder, and every output is kept in <work>/out/.
Steps marked "needs the full records" are skipped unless the regenerated record files are in place (see REGENERATE.md).
The last line is CHECKS PASSED: True/False."""
import os, sys, subprocess, time, shutil, re, json, gzip

R = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
if not W or (W + os.sep).startswith(R + os.sep):
    print(__doc__); sys.exit('give a work folder outside the repository')
OUT = os.path.join(W, 'out'); os.makedirs(OUT, exist_ok=True)
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')


def P(*a):
    return os.path.join(R, *a)


# working copies: verify_cover.py writes open_remaining.json next to the records; the PSD replay and cover read
# targets.json / T_jobs.json from the current folder; the cover of five roots needs T_w99.jsonl unpacked.
shutil.copytree(P('r1', 'records'), os.path.join(W, 'r1_records'), dirs_exist_ok=True)
os.makedirs(os.path.join(W, 'psd_run'), exist_ok=True)
for f in ('targets.json', 'T_jobs.json'):
    shutil.copy(P('psd', 'run', f), os.path.join(W, 'psd_run'))
with gzip.open(P('psd', 'run', 'T_w99.jsonl.gz'), 'rb') as a, open(os.path.join(W, 'psd_run', 'T_w99.jsonl'), 'wb') as b:
    shutil.copyfileobj(a, b)


def has(*subs):
    return lambda t: all(s in t for s in subs)


def same_lines(stored, only=None, strip_times=False):
    def f(t):
        a = [l.rstrip() for l in open(stored, encoding='utf-8', errors='replace').read().splitlines()]
        b = [l.rstrip() for l in t.splitlines()]
        if strip_times:
            a = [re.sub(r' \d+s$', '', l) for l in a]; b = [re.sub(r' \d+s$', '', l) for l in b]
        if only:
            a = [l for l in a if only(l)]; b = [l for l in b if only(l)]
        return len(b) > 0 and a == b
    return f


def json_after(prefix, cond):
    def f(t):
        for l in t.splitlines():
            if l.startswith(prefix):
                d = json.loads(l[len(prefix):].split('{', 1)[1].rsplit('}', 1)[0].join('{}'))
                return cond(d)
        return False
    return f


def job5(l):
    return re.match(r'job (11|45|202|246|252) ', l) is not None


FR = ('4894115117760208191947864438498376734060505474291548201601721665201509080181253344123035807218923803630275967707292032'
      '113238345761372068160856864216488560312317670625/4971788578344406709191304695977174443732270082404450954495883950049377'
      '101271984956864963156303386350245102973490689359962940742592129613716999513651648307260762882048')
MTREE_CMD = ("import sys; sys.path.insert(0, r'%s'); import hloc; M, c = hloc.psi_T_upper(); "
             "print('Phi_F(C_T) <=', repr(M), ' at doubles', c)" % P('mtree', 'upper'))
GENV = {'BK17_GPM': '1', 'BK17_GLEVELS': '0:0.01,0.01:1',
        'BK19_EMIN_A': P('gstar', 'emin', 'emin_A.json'), 'BK19_EMIN_B': P('gstar', 'emin', 'emin_B.json')}

# (name, argv, cwd, extra env, test of the output, the output line to show)
STEPS = [
    # the constant
    ('mtree_upper', [PY, '-B', '-c', MTREE_CMD], W, {}, has('Phi_F(C_T) <= 0.27658360966404055'), 'Phi_F(C_T) <='),
    ('mtree_enclosure', [PY, '-B', P('mtree', 'enclosure', 'bk8_core.py')], W, {},
     has('M_tree in [0.27658360687864314, 0.27658360688603606]'), 'M_tree in'),
    ('constant_exact', [PY, '-B', P('constant', 'constant_check.py')], W, {},
     has('2 kappa_E > M_up: True', '1 + 1/M_up >= 4.6155432392 (10 decimals rounded down): True'), '>= 4.6155432392'),
    ('constant_digits', [PY, '-B', P('constant', 'arith2.py')], W, {},
     has('0.27658360966404055 4.615543239220414825374898366193126551903'), '0.27658360966404055'),
    # Theorem R1
    ('R1_loc4d', [PY, '-B', P('r1', 'trees', 'bk9_bnb.py'), 'check'], W, {},
     has('LOC-4D r0 0.0001', 'lambda_max <= -0.6698'), 'LOC-4D'),
    ('R1_cover', [PY, '-B', 'verify_cover.py', 'rec_W0.jsonl', 'rec_W1.jsonl'], os.path.join(W, 'r1_records'), {},
     has('unresolved boxes: 0', 'records 74096'), 'unresolved'),
    ('R1_replay_50', [PY, '-B', 'replay10.py', '50', '2026', 'rec_W0.jsonl', 'rec_W1.jsonl'], os.path.join(W, 'r1_records'), {},
     has('replayed 50 records: identical 50'), 'replayed'),
    # the thin set N and its constants
    ('N_single_edge', [PY, '-B', P('thin', 'r7', 'need.py')], W, {},
     has('edge thresholds (R = M_tree lower end): uv 0.2051576882', '0.2890285442'), 'edge thresholds'),
    ('N_window', [PY, '-B', P('thin', 'r7', 'need2.py')], W, {}, has('best |log d_v| in N: 0.1206'), 'best |log d_v|'),
    ('N_d3', [PY, '-B', P('thin', 'd3', 'bk12_exact.py')], W, {},
     lambda t: t.count('FEASIBLE') >= 5 and 'INFEASIBLE' not in t, 'alt'),
    ('N_klo', [PY, '-B', P('thin', 'consts', 'bk15_klo_iv.py'), '0.9491'], W, {},
     has('CERTIFIED: k >= [0.9490999', 'closed 67127 boxes, infeasible 95041'), 'CERTIFIED'),
    ('N_consts', [PY, '-B', P('thin', 'consts', 'bk15_const.py')], W, {},
     has('alpha_lo [0.373127819880702', 'F-bound [0.988848244837210'), 'alpha_lo'),
    ('N_box', [PY, '-B', P('thin', 'consts', 'bk15_box_iv.py')], W, {},
     has('h_lo >= [0.78270668658580', 'H_lo >= [0.63459207372126'), 'h_lo'),
    ('N_marginal_window', [PY, '-B', P('loc2k', 'src', 'bk16_margx.py'), '0.3', '0.7', '0.25'], W, {},
     has('excluded boxes 181286 failures 0'), 'excluded boxes'),
    ('N_cross_entries', [PY, '-B', P('thin', 'pi', 'pi_check.py'), '0.82', os.path.join(OUT, 'pi082.json')], W, {},
     has("'processed': 21924095", "'min_bound': 0.8200000063961864"), 'processed'),
    ('N_profile_maxima', [PY, '-B', P('thin', 'sep', 'bk14_supcert.py')], W, {},
     has('pq <= [2.1798183856797051745081469318'), 'pq <='),
    # the kill inequality
    ('K_loc2k_replay', [PY, '-B', P('loc2k', 'src', 'bk16_loc2verify.py'), P('loc2k', 'run', 'loc2_P002', 'loc2K_boxes.jsonl')],
     W, {}, has('replayed 5306 boxes, failures 0', 'psi volume + origin box == K volume: True',
               'single volume == domain volume: True'), 'replayed'),
    ('K_loc2kc_cells', [PY, '-B', P('loc2kc', 'check', 'oloc2c.py'), P('loc2kc', 'run', 'loc2c'), '50'], W, {},
     lambda t: '(2) tiling: cells with a problem 0' in t and t.count(': OK') == 11 and
               all(l.endswith(': OK') for l in t.splitlines() if l.startswith('(3) gap after cell')), '(2) tiling'),
    ('K_emin_2cells', [PY, '-B', P('kill', 'src', 'bk19_emin.py'), 'replay', P('samples', 'emin_2cells.json')], W, {},
     has('leaves 2000 failures 0', 'recorded 4.816757e-04 replayed 4.816757e-04 volume exact',
         'recorded 1.987217e-04 replayed 1.987217e-04 volume exact'), 'leaves'),
    ('K_emin_extended', [PY, '-B', P('gstar', 'emin_ext', 'eslv20.py')], W, {},
     has('cell 12 row 6 side hi: Emin_lo 1.987217211588e-04, 434 edge leaves', 'min elin 2.075457722446e-04, L below table 0',
         'edge leaves tile the face (exact volume): True'), 'cell 12 row 6'),
    ('K_Q_replay_50', [PY, '-B', P('kill', 'src', 'bk19_verify.py'), 'Q', 'replay', '1', '0', P('samples', 'Q_50.jsonl')], W,
     {'BK17_TQ': P('kill', 'run', 'loc2c', 'loc2c.json')}, has('checked 50', 'failures 0'), 'checked'),
    ('K_Gstar_replay_50', [PY, '-B', P('gstar', 'src', 'bk19_verify.py'), 'G', 'replay', '1', '0', P('samples', 'G_50.jsonl')], W,
     GENV, has('checked 50', 'failures 0'), 'checked'),
    ('K_fbar', [PY, '-B', P('kill', 'src', 'bk19_fbar.py')], W, {},
     has('FBAR CERTIFIED: |f| <= [0.48286999999999996591, 0.48287000000000002142]'), 'FBAR'),
    # positive semidefiniteness
    ('PSD_CLR_exact', [PY, '-B', P('psd', 'exact', 'comp_r2.py'), 'targets.json'], P('psd', 'run'), {},
     has("max Phi'/Phi_0 exact = " + FR, '(C-LR) HOLDS', '0 < e < B holds (exact) for all 9'), '(C-LR)'),
    ('PSD_root_domain', [PY, '-B', P('psd', 'engine_check', 'dom_check.py')], W, {}, has('ROOT DOMAIN OK'), 'ROOT DOMAIN'),
    ('PSD_engine_point', [PY, '-B', P('psd', 'engine_check', 'eng_check.py'), 'point', '1', '20'], W, {},
     json_after('POINT seed 1', lambda d: d['minUgap'] > 0 and d['minVgap'] > 0 and d['maxrelU'] < 1e-9 and d['maxrelV'] < 1e-9),
     'POINT'),
    ('PSD_engine_e2e', [PY, '-B', P('psd', 'engine_check', 'eng_check.py'), 'e2e', '1', '20'], W, {},
     json_after('E2E seed 1', lambda d: d['viol1'] == 0 and d['viol2'] == 0 and d['minslack1'] > 0 and d['minslack2'] > 0), 'E2E'),
    ('PSD_cover_5roots', [PY, '-B', P('psd', 'acover_bk24.py'), '.', 'targets.json', '11,45,202,246,252'],
     os.path.join(W, 'psd_run'), {},
     lambda t: t.count('-> COVERED') == 5 and same_lines(P('psd', 'run', 'acover_final.log'), only=job5)(t), 'records'),
    ('PSD_replay_50', [PY, '-B', P('psd', 'bk24_replay.py'), '1', '0', os.path.join(OUT, 'PSD_replay_50.json'),
                       P('samples', 'O_50.jsonl')], os.path.join(W, 'psd_run'), {},
     has('failures 0', "\"('U', 'O')\": 1.4183269003709136e-09", "\"('V', 'O')\": 4.5395387449076436e-10"), 'checked'),
    # the two checks that need the full record files (see REGENERATE.md)
    ('K_Gstar_cover_all', [PY, '-B', P('gstar', 'src', 'bk19_gcover.py'), 'GB1,GB2,GC,GD', 'H1,H2,H3', 'list'],
     os.path.join(W, 'gstar_run'), GENV, has('0 gaps', '0 overlaps', '0 pending'), 'gaps'),
    ('PSD_cover_all', [PY, '-B', P('psd', 'bk23_cover.py'), 'T', 'list'], os.path.join(W, 'psd_run_full'), {},
     has('COVER COMPLETE'), 'COVER'),
]
NEEDS_BIG = {'K_Gstar_cover_all': (P('gstar', 'run'), 'H1_w0.jsonl', 'gstar_run'),
             'PSD_cover_all': (P('psd', 'run'), 'T_w0.jsonl', 'psd_run_full')}

allok = True; T0 = time.time(); summ = []
for name, argv, cwd, extra, check, show in STEPS:
    if name in NEEDS_BIG:
        src, probe, wd = NEEDS_BIG[name]
        if not os.path.exists(os.path.join(src, probe)):
            line = '%-22s SKIPPED (needs the full records at %s)' % (name, os.path.relpath(os.path.join(src, probe), R))
            print(line, flush=True); summ.append(line); continue
        shutil.copytree(src, os.path.join(W, wd), dirs_exist_ok=True)   # the cover checkers write next to the records
    t0 = time.time()
    pr = subprocess.run(argv, cwd=cwd, env=dict(ENV, **extra), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    txt = pr.stdout.decode('utf-8', 'replace')
    open(os.path.join(OUT, name + '.out'), 'w', encoding='utf-8').write(txt)
    try:
        ok = pr.returncode == 0 and bool(check(txt))
    except Exception:
        ok = False
    allok = allok and ok
    key = [l for l in txt.splitlines() if show in l][:1]
    line = '%-22s %6.1f s  %s  %s' % (name, time.time() - t0, 'PASS' if ok else 'FAIL', key[0][:150] if key else '')
    print(line, flush=True); summ.append(line)
line = '%d steps, %.0f s wall; CHECKS PASSED: %s (%d skipped for missing data)' % (
    len(STEPS), time.time() - T0, allok, sum(1 for l in summ if 'SKIPPED' in l))
print(line); summ.append(line)
open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(summ) + '\n')
