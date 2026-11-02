r"""bk19_qtail.py : extract the domain-Q records written after the final checkpoints (the continuation).
Usage (in S\run\rep):  python ..\..\src\bk19_qtail.py
For each of Q1_w0..w2, Q1s_w0..w2: bytes [roff17, roff19) of S\run\<base>.jsonl, where roff17 is the record offset of
the checkpoint (read from S\run\<base>.state.json, md5-listed in S\md5_bk17_records.txt) and roff19 the
offset of the current the checkpoint.  The first roff17 bytes must be byte-identical to the file .
Writes Qnew_<base>.jsonl and prints the counts."""
import os, json, hashlib
S = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
R17 = os.path.join(S, 'BK17', 'run'); R19 = os.path.join(S, 'BK19', 'run')
tot = 0
for base in ['Q1_w0', 'Q1_w1', 'Q1_w2', 'Q1s_w0', 'Q1s_w1', 'Q1s_w2']:
    st17 = json.load(open(os.path.join(R17, base + '.state.json')))
    st19 = json.load(open(os.path.join(R19, base + '.state.json')))
    a, b = st17['roff'], st19['roff']
    with open(os.path.join(R19, base + '.jsonl'), 'rb') as f: data = f.read()[:b]
    with open(os.path.join(R17, base + '.jsonl'), 'rb') as f: old = f.read()[:a]
    same = hashlib.md5(data[:a]).hexdigest() == hashlib.md5(old).hexdigest()
    new = data[a:]
    n = new.count(b'\n'); tot += n
    open('Qnew_%s.jsonl' % base, 'wb').write(new)
    print(base, 'bk17 bytes', a, 'prefix identical', same, 'new records', n, 'queue now', len(st19['queue']))
print('total new records', tot)
