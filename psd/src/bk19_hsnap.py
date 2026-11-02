r"""bk19_hsnap.py : incremental replay snapshots of G* record files.  Usage (in S\run\rep): python ..\..\src\bk19_hsnap.py <n>
For every ..\H*_w*.jsonl: bytes [off_prev, roff) where off_prev is the end offset of the previous snapshot (manifest
hsnap_manifest.json) and roff the current checkpoint offset; written to Hsnap<n>_<base>.jsonl.  The union of all snapshots is
exactly the closed-record prefix of each file at the last snapshot."""
import json, glob, os, sys
n = int(sys.argv[1]); man = json.load(open('hsnap_manifest.json')) if os.path.exists('hsnap_manifest.json') else {}
tot = 0; files = []
for st_f in sorted(glob.glob('../H*_w*.state.json')):
    base = os.path.basename(st_f)[:-len('.state.json')]
    roff = json.load(open(st_f))['roff']; a = man.get(base, 0)
    if roff <= a: continue
    with open('../' + base + '.jsonl', 'rb') as f: f.seek(a); data = f.read(roff - a)
    out = 'Hsnap%d_%s.jsonl' % (n, base); open(out, 'wb').write(data); files.append(out)
    k = data.count(b'\n'); tot += k; man[base] = roff
json.dump(man, open('hsnap_manifest.json', 'w'), indent=0)
open('hsnap%d_files.txt' % n, 'w').write(' '.join(files))
print('snapshot', n, 'records', tot, 'files', len(files))
