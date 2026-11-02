"""regen.py: launcher for the searches that produce the certificate records (see REGENERATE.md).

  python regen.py LIST                                     the stages
  python regen.py STAGE --root DIR [--workers N] [--budget SEC] [--fresh] [--dry-run]

--root must be a COPY of this repository (the searches write records, state files and logs into their own folders).
Each stage runs the original producer commands, single-threaded, one process per worker; all drivers checkpoint and
resume, so a stage can be re-run until every worker reports DONE. --fresh moves the stored records and state files of that
stage aside (into _prev_<stage>/ next to them) so that the search starts from scratch."""
import os, sys, subprocess


def stages(budget):
    G = {'BK17_GPM': '1', 'BK17_GLEVELS': '0:0.01,0.01:1', 'BK19_JOINT': '1',
         'BK19_EMIN_A': '../emin/emin_A.json', 'BK19_EMIN_B': '../emin/emin_B.json'}
    b = lambda d: str(budget or d)
    return {
      'loc4d':   ('Lemma LOC-4D (seconds)', [[('r1/trees', ['bk9_bnb.py', 'check'], {})]]),
      'r1-trees': ('Theorem R1, branch-and-bound trees A, B, then RV (about 5 CPU-h)',
                   [[('r1/trees', ['bk9_bnb.py', 'A', b(5400), '1e-4'], {}), ('r1/trees', ['bk9_bnb.py', 'B', b(5400), '1e-4'], {})],
                    [('r1/trees', ['bk9_bnb.py', 'RV', b(3000), '1e-4', 'bnb9_A.json'], {}),
                     ('r1/trees', ['bk9_bnb.py', 'RV', b(3000), '1e-4', 'bnb9_B.json'], {})]]),
      'r1-records': ('Theorem R1, the 74,096 records (5.4 CPU-h; roots = open boxes of bnb9_RVB.json, supplied)',
                     [[('r1/records', ['bk10_run.py', 'run', 'W%d' % w, b(20000), str(w), '2'], {}) for w in range(2)]]),
      'loc2kc':  ('Lemma LOC2-K(c) (76 s)', [[('loc2kc', ['src/bk17_loc2c.py'], {})]]),
      'emin':    ('the Emin table, parts A and B (2.3 CPU-h)',
                  [[('kill/emin', ['../src/bk19_emin.py', 'run', 'emin_%s.json' % p, '1000', 'cells%s.json' % p], {}) for p in 'AB']]),
      'q':       ('domain Q, runs Q1 and Q1s, 3 workers each (about 30 CPU-h)',
                  [[('kill/run', ['../src/bk17_run.py', 'Q', tag, str(k), '3', b(40000)], {'BK17_TQ': 'loc2c/loc2c.json'})
                    for tag in ('Q1', 'Q1s') for k in range(3)]]),
      'gstar':   ('G*, jobs H1 (17 workers), then H2 (8), H3 (1) (about 180 CPU-h; needs the emin stage first)',
                  [[('gstar/run', ['../src/bk19_grun.py', 'H1', str(k), b(300000)], G) for k in range(17)],
                   [('gstar/run', ['../src/bk19_grun.py', 'H2', str(k), b(300000)], G) for k in range(8)],
                   [('gstar/run', ['../src/bk19_grun.py', 'H3', '0', b(300000)], G)]]),
      'psd':     ('positive semidefiniteness, 640 roots, 13 workers (about 340 CPU-h; the biggest job)',
                  [[('psd/run', ['../bk23_orun.py', 'T', str(k), b(300000)], {'BK22_ORDER': 'order.json'}) for k in range(13)]]),
    }


FRESH = {  # stage: (folder, patterns of the stage's own records / state files)
    'r1-trees': ('r1/trees', ['bnb9_A.json', 'bnb9_B.json', 'bnb9_RVA.json', 'bnb9_RVB.json']),
    'r1-records': ('r1/records', ['rec_W*.jsonl', 'ckpt_W*.json']),
    'emin': ('kill/emin', ['emin_A.json', 'emin_B.json']),
    'q': ('kill/run', ['Q1_w*', 'Q1s_w*']),
    'gstar': ('gstar/run', ['H1_w*', 'H2_w*', 'H3_w*']),
    'psd': ('psd/run', ['T_w*', 'T_open_set.json', 'claims']),
}


def fresh(stage, root):
    import glob, shutil
    if stage not in FRESH: return
    d, pats = FRESH[stage]; base = os.path.join(root, d); prev = os.path.join(base, '_prev_' + stage)
    for pat in pats:
        for f in glob.glob(os.path.join(base, pat)):
            os.makedirs(prev, exist_ok=True); shutil.move(f, os.path.join(prev, os.path.basename(f)))
            print('moved aside', os.path.relpath(f, root), flush=True)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'): print(__doc__); return
    def opt(n, d=None): return a[a.index(n) + 1] if n in a else d
    nw = int(opt('--workers', 0)); budget = int(opt('--budget', 0)) or None
    root = os.path.abspath(opt('--root', '.')); dry = '--dry-run' in a
    st = stages(budget)
    if a[0] == 'LIST':
        for k, (d, _) in st.items(): print('%-11s %s' % (k, d))
        return
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.normcase(root) == os.path.normcase(here) and not dry:
        sys.exit('--root must point at a copy of the repository, not at the repository itself')
    desc, groups = st[a[0]]
    if '--fresh' in a and not dry: fresh(a[0], root)
    env0 = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env0.pop('BK23_NODUBCACHE', None)
    for g in groups:
        procs = []
        for i, (cwd, args, env) in enumerate(g):
            if nw and i >= nw: continue
            cmd = [sys.executable, '-B'] + args
            print('[%s] cd %s && %s %s' % (a[0], cwd, ' '.join('%s=%s' % kv for kv in env.items()), ' '.join(cmd)), flush=True)
            if dry: continue
            tag = '%s_%d' % (a[0], i)
            procs.append(subprocess.Popen(cmd, cwd=os.path.join(root, cwd), env=dict(env0, **env),
                                          stdout=open(os.path.join(root, cwd, tag + '.log'), 'ab'),
                                          stderr=open(os.path.join(root, cwd, tag + '.err'), 'ab')))
        for p in procs: p.wait()
        print('[%s] group finished, return codes %s' % (a[0], [p.returncode for p in procs]), flush=True)


main()
