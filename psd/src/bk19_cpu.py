import json, glob, os, re
S17 = '../../BK17/run/'
q = 0
for b in ['Q1_w0','Q1_w1','Q1_w2','Q1s_w0','Q1s_w1','Q1s_w2']:
    q += json.load(open(b + '.state.json'))['stats']['cpu'] - json.load(open(S17 + b + '.state.json'))['stats']['cpu']
h = sum(json.load(open(f))['stats']['cpu'] for f in glob.glob('H*_w*.state.json'))
em = 0
for f in glob.glob('../emin/emin_*_*.log'):
    for l in open(f):
        m = re.search(r'(\d+)s$', l.strip())
        if m: em += int(m.group(1))
print('Q %.2f h, H %.2f h, Emin table %.2f h' % (q / 3600, h / 3600, em / 3600))
