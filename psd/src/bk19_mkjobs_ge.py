r"""bk19_mkjobs_ge.py : after the run GE stopped, its checkpoint queue (copied to S\run\GE_w0.state.json together with
GE_w0.jsonl) becomes the job list H2_jobs.json (same format as H1)."""
import json
st = json.load(open('GE_w0.state.json'))
jobs = [[a, b, lev, d] for a, b, lev, d in st['queue']]
json.dump(jobs, open('H2_jobs.json', 'w'))
print('GE records offset', st['roff'], 'pending boxes -> H2 jobs', len(jobs))
