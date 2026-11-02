# Where each step of the paper's appendix is proved, and where each finite claim is checked

Part 1 maps the outline in the paper's appendix "The computer-assisted cutwidth bound" to the sections of `proof.md`.
Part 2 maps every finite claim (`FC-...`) of `proof.md` to the script that checks it and the step of `run_checks.py` that
runs it (`python -B run_checks.py <work folder>`; see `README.md`).

## Part 1: the appendix, section by section

| appendix claim | proof.md |
|---|---|
| **The plan.** Compress a cubic multigraph to a simple 3-regular kernel on h <= 2(M - N) vertices, order it, expand; O(log N) per cut | paper, Lemma lem:order; 2.4, Lemma COMPRESS |
| **Bridges.** Blocks, the counting M - N + 1 = sum over blocks, a centroid block, recursion of depth O(log N) | 2.4: Lemmas ASSEMBLY and CENTROID, Theorem K2C |
| Every bridgeless cubic graph has a perfect matching (Petersen); kernels are bridgeless | 3.4, Fact PETERSEN and Corollary A1 |
| **Step 1.** The median-edge ordering: every cut has at most as many edges as there are open vertices | 3.2 (the mixed order), 3.3 |
| The Gaussian field, the edge vectors and scores; scores are standard Gaussians with correlation <y_e, y_f> | 2.6 |
| Sheppard's formula, "0 is the worst threshold", "more correlation never hurts" | 2.5, Lemma G and Lemma DERIV |
| The star lemma; the edge cosines; kappa_E; the rule bound | 2.8 |
| Concentration over a grid of thresholds; cw(H) <= (kappa_E + xi) h + C_xi | 3.3, 3.4 (Theorem A with the edge-score side rule) |
| **Step 2.** Inserting a matched edge on the better side; the cost F; the side switches once | 2.6, 3.2 |
| The 123 types; the Gram matrix lies in the cone they span up to an error vanishing as R -> infinity | 2.7, 3.1 (the cone with a shell) |
| Theorem A: cw(H) <= (M_123/2 + xi) h + C_xi for every simple bridgeless cubic H | 3.3, 3.4 |
| The rule bound M_123 <= 2 kappa_E | 2.8 |
| **Step 3.** The tree point m_T, the matrix C_T, M_tree as a one-dimensional integral; the certified enclosure | 4.1, 4.2, 4.3 (FC-MT) |
| The tree cut (the gradient at C_T) | 4.4 (FC-AAB) |
| The Hessian has one positive direction; the cross determinant; the family R; one common factor | 5.1, 5.2 |
| The kill as a witness; (R1) and (R2) suffice (Theorem WIT, Theorem TS*) | 5.3, 5.4 |
| (R1): five coordinates, the cut may be taken tight, the local lemma, the branch and bound outside (169,002 boxes) | 6.1-6.5 (FC-LOC4D, FC-R1) |
| (R2) is needed only on the thin set N; its constants | 7.1-7.4 (FC-R7, FC-D3, FC-THIN, FC-WIN, FC-PI) |
| The tree cut is a two-star problem; the kill is separable; reduction to one star | 7.5, 8.1-8.5 (FC-SEP) |
| How a box of star slices is closed: dual multipliers, weak duality, outward-rounded re-check | 8.6 |
| The kill inequality on N: domain Q (337,645 leaves), the Emin table (154,246 leaves), G* (1,801,388 leaves) | 8.7-8.11 (FC-LOC2K, FC-LOC2KC, FC-EMIN, FC-Q, FC-GSTAR) |
| The kill is positive semidefinite on N: the disc criterion, Theorem LR, 640 roots / 1,495,347 records, the exact composition | 9.1-9.8 (FC-FBAR, FC-CLR, FC-LR) |
| Theorem R2, Theorem M_tree (M_123 = M_tree), the (4.6155 - eps) n bound and the arithmetic of the constant | 10.1-10.3 (FC-KAPPA) |
| The limit of the method: no certificate M_123 <= x with x < M_tree exists | 1.2 (a consequence of M_123 = M_tree) |

The appendix writes the cut vector as l, the saddle direction as nu and a type as tau(z); `proof.md` writes a, v and t(z).

## Part 2: the finite claims and their checks

Every script is run by `run_checks.py` (step name in the last column) except the two covers of the large certificates, which
need the regenerated records (`REGENERATE.md`).

| claim | statement | script (folder) | data | step |
|---|---|---|---|---|
| FC-MT | the enclosure of M_tree (4.3) | `mtree/upper/hloc.py` (upper end, F_point at C_T rounded down); `mtree/enclosure/bk8_core.py` (two-sided) | none | mtree_upper, mtree_enclosure |
| FC-KAPPA | 2 kappa_E > M_up; 1 + 1/M_up >= 4.6155432392 (10.3) | `constant/constant_check.py` (exact rationals, 45-digit enclosure of kappa_E); `constant/arith2.py` | none | constant_exact, constant_digits |
| FC-AAB | the gradient at the tree, a_ab and K0 (4.4) | `mtree/enclosure/bk8_core.py` (`consts()`) | none | mtree_enclosure |
| FC-LOC4D | lambda_max <= -0.6698 on the box of radius 1e-4 at the tree (6.3) | `r1/trees/bk9_bnb.py check` | none | R1_loc4d |
| FC-R1 | five branch-and-bound trees, 169,002 boxes, 0 unresolved (6.4) | trees: `r1/trees/bk9_bnb.py` (`bnb9_*.json`); records: `r1/records/bk10_run.py`; cover: `r1/records/verify_cover.py`; replay: `r1/records/replay10.py` | `r1/records/rec_W0.jsonl`, `rec_W1.jsonl` (74,096 records, stored in full) | R1_cover, R1_replay_50 |
| FC-R7 | the single-edge ceilings and the window of N (7.2) | `thin/r7/need.py`, `need2.py` | `need2_best.npy` | N_single_edge, N_window |
| FC-D3 | Lemma D3 on all 123 generators (7.3) | `thin/d3/bk12_exact.py` (exact LPs) | none | N_d3 |
| FC-THIN | k_lo = 0.9491, alpha_lo, the constants, the box minima (7.4) | `thin/consts/bk15_klo_iv.py 0.9491`, `bk15_const.py`, `bk15_box_iv.py` | none | N_klo, N_consts, N_box |
| FC-WIN | every marginal of N lies in the window (7.4) | `loc2k/src/bk16_margx.py 0.3 0.7 0.25` (181,286 boxes) | none | N_marginal_window |
| FC-PI | every cross entry >= 0.82 on N (7.4) | `thin/pi/pi_check.py 0.82 <out.json>` (21,924,095 boxes) | none | N_cross_entries |
| FC-SEP | the profile maxima (8.4) | `thin/sep/bk14_supcert.py` | none | N_profile_maxima |
| FC-LOC2K | Lemma LOC2-K (8.8) | producer `loc2k/src/bk16_loc2run.py`; replay `loc2k/src/bk16_loc2verify.py <records>` (5,306 boxes, exact volumes) | `loc2k/run/loc2_P002/loc2K_boxes.jsonl` | K_loc2k_replay |
| FC-LOC2KC | Lemma LOC2-K(c), its 12 cells and the 11 slivers between them (8.8) | producer `loc2kc/src/bk17_loc2c.py`; check `loc2kc/check/oloc2c.py <cells folder> <n>` (exact tiling of each cell; the slivers certified on extended boxes; part (1) is a comparison of an independent lower bound with the recorded one) | `loc2kc/run/loc2c/loc2c.json`, `loc2c_boxes.jsonl` | K_loc2kc_cells |
| FC-EMIN | the Emin table, 156 cells (8.9); the extended certificate for cell (12, 6), side hi (8.10) | producer and replay `kill/src/bk19_emin.py`; extension `gstar/emin_ext/eslv20.py` (run in `gstar/src`) | `gstar/emin/emin_A.json`, `emin_B.json`; cells in `kill/emin/` | K_emin_2cells, K_emin_extended |
| FC-Q | domain Q, 337,645 leaves (8.10) | producer `kill/src/bk17_run.py Q`; replay `kill/src/bk19_verify.py Q replay`; cover `kill/src/bk17_coverq.py` | records not stored; `samples/Q_50.jsonl` | K_Q_replay_50 |
| FC-GSTAR | G*, 1,801,388 leaves (8.10) | producer `gstar/src/bk19_grun.py`; replay `gstar/src/bk19_verify.py G replay`; cover `gstar/src/bk19_gcover.py`; census of the capped records `gstar/src/bk20_r1.py` | records not stored; `samples/G_50.jsonl`; job lists in `gstar/run/` | K_Gstar_replay_50, K_Gstar_cover_all |
| FC-FBAR | |f| <= 0.48287 on N (9.3) | `kill/src/bk19_fbar.py` | none | K_fbar |
| FC-CLR | the exact composition ratio (C-LR) = 0.984377159374... (9.6) | `psd/exact/comp_r2.py targets.json` (exact rationals) | `psd/run/targets.json` | PSD_CLR_exact |
| (engine) | the U and V upper forms dominate their definitions; the 640 roots cover the domain (9.5, 9.7) | `psd/engine_check/eng_check.py point|e2e|sym|box <seed> <n>`, `dom_check.py` | `psd/run/T_jobs.json` | PSD_root_domain, PSD_engine_point, PSD_engine_e2e |
| FC-LR | (U-LR*), (V-LR) on 640 roots, 1,495,347 records (9.7) | producer `psd/bk23_orun.py`; replay `psd/bk24_replay.py`; covers `psd/bk23_cover.py`, `psd/acover_bk24.py` | records not stored except the seven roots in `psd/run/T_w99.jsonl.gz`; `samples/O_50.jsonl`; `psd/run/targets.json`, `T_jobs.json`, `order.json` | PSD_cover_5roots, PSD_replay_50, PSD_cover_all |

Two facts rest on a single code each: FC-PI at 0.82 (`thin/pi/pi_check.py`) and the extended Emin certificate for cell
(12, 6) (`gstar/emin_ext/eslv20.py`); every other finite claim was re-derived by at least two independent programs during
the project, of which one is kept here.
