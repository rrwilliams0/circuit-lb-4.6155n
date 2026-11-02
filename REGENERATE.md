# Regenerating the certificates

Theorem R1's records (`r1/records/rec_W0.jsonl`, `rec_W1.jsonl`), the Emin tables (`gstar/emin/`), the LOC2-K records
(`loc2k/run/loc2_P002/`), the LOC2-K(c) records (`loc2kc/run/loc2c/`) and the seven roots `psd/run/T_w99.jsonl.gz` are stored.
The records of the kill inequality on N (domain Q: 337,645 leaves; G*: 1,801,388 leaves) and of positive semidefiniteness
(640 roots, 1,495,347 records) are not (47 GB). Every producer is here with its inputs, so they can be made again. A
regeneration gives a *different but equally valid* set of records (the searches use time budgets and a pool of workers,
and an LP solver proposes the dual multipliers); a regenerated certificate is valid if and only if the deterministic
checkers below accept it. The M_tree enclosure, the constants, LOC2-K(c), the Emin table, (C-LR) and the R1 records
reproduce exactly.

## How

```
cp -r <this repository> work/repo        # the searches WRITE into their folders: always work on a copy
cd work/repo
python regen.py LIST
python regen.py STAGE --root . --fresh [--workers N] [--budget SECONDS]
```

`--fresh` moves the stored records and state files of that stage aside; without it a stage resumes. Budgets are per run;
every driver checkpoints, so re-run a stage until each worker reports `DONE` in its log. Run the stages in this order:

| order | stage | what it produces | cost of the original run | needs |
|---|---|---|---|---|
| 1 | `loc4d` | Lemma LOC-4D | seconds | |
| 2 | `r1-trees` | Theorem R1: the branch-and-bound trees A, B, RV | 4.7 CPU-h | |
| 3 | `r1-records` | Theorem R1: the 74,096 records (roots = open boxes of `bnb9_RVB.json`, supplied) | 5.4 CPU-h | stage 2 (or the supplied `bnb9_RVB.json`) |
| 4 | `loc2kc` | Lemma LOC2-K(c) | 76 s | |
| 5 | `emin` | the Emin table, 156 cells, parts A and B | 2.3 CPU-h | |
| 6 | `q` | domain Q, runs Q1 and Q1s, 3 workers each | about 30 CPU-h | `loc2c.json` (stage 4; supplied) |
| 7 | `gstar` | G*: jobs H1 (17 workers), H2 (8), H3 (1) | about 180 CPU-h | the Emin tables (stage 5; supplied in `gstar/emin/`) |
| 8 | `psd` | positive semidefiniteness: 640 roots, 13 workers | about 340 CPU-h | `targets.json`, `T_jobs.json`, `order.json` (supplied) |

Not run by `regen.py` because they are small: LOC2-K (`python -B src/bk16_loc2run.py 0.02 1.5e-4 <eps0>` in `loc2k`,
parameters as in `loc2k/run/loc2_P002/`), Lemma PI (`thin/pi/pi_check.py 0.82 out.json`, 27 s), the profile maxima
(`thin/sep/bk14_supcert.py`), the constant (`constant/constant_check.py`) and (C-LR) (`psd/exact/comp_r2.py`); all of
these are in `run_checks.py`.

The stored G* certificate consists of the runs GB1, GB2, GC, GD (produced with `src/bk17_run.py G`, the earlier form of
the search) and H1, H2, H3 (produced with `src/bk19_grun.py`); stage 7 reproduces the H jobs, whose job lists are in
`gstar/run/`. After a regeneration, give the cover checker the names of the runs that exist.

## Checking regenerated records

1. Exact covers: domain Q — in `kill/run`: `python -B ../src/bk17_coverq.py Q1 Q1s` (expect 0 gaps, 0 overlaps, 0 pending);
   G* — in `gstar/run`: `python -B ../src/bk19_gcover.py <bk17-format runs> <bk19-format runs> list` (expect 0 gaps,
   0 overlaps, 0 pending; it writes the pending boxes next to the records, so run it on the copy);
   PSD — in `psd/run`: `python -B ../bk23_cover.py T list` (expect `COVER COMPLETE`), and `python -B ../acover_bk24.py . targets.json <root list>`.
2. Full replays: `python -B ../src/bk19_verify.py Q replay <stride> <offset> <files>` and `... G replay ...` (about 4 and
   18 CPU-hours), and `python -B ../bk24_replay.py <stride> <offset> <summary.json> <files>` in `psd/run` (about 99 CPU-hours;
   every summary must report `failures 0`).
3. `python -B run_checks.py ../work` then also runs the two steps it skips without the records (they look for
   `gstar/run/H1_w0.jsonl` and `psd/run/T_w0.jsonl`).
