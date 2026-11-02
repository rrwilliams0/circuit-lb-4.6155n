# Li's function needs (1 + 1/M_tree - eps) n gates: the complete proof

---

## Abstract

Let f_n be the two-source extractor of X. Li (FOCS 2023), viewed as a Boolean function on n bits, and let s_n be the
least number of gates of a fan-in-two circuit over the full binary basis B2 that computes f_n. We prove

    liminf_{n -> oo} s_n / n  >=  1 + 1/M_tree  >=  4.6155432392,

where M_tree = 0.2765836068823734710... is the expected number of split vertices of one matched edge in a Gaussian
ordering of the 3-regular tree. The proof has three parts. The first is a known chain (the reduction of the paper, reproved here): a circuit with s gates and n inputs yields a read-once program with about 2^w edges, where w is (up to lower
order terms) the cutwidth of a cubic "kernel" graph with at most 2(s - n) + O(log n) vertices, and Li's function needs
2^{n - O(log n)} edges. The second is matched insertion:
every kernel has an ordering whose cuts are bounded by M_123 h/2 + o(h), where M_123 is the supremum of an explicit
four-dimensional Gaussian sign functional over a cone with 123 generators. The third part, which is new, is the exact
value of that supremum: M_123 = M_tree. The tree point is the maximiser.

The proof of M_123 = M_tree is analytic in structure and computer-assisted in three places. The saddle of the functional
at the tree is the cross determinant of the correlation matrix; on the set where that determinant vanishes the saddle is
gone, and a four-parameter branch and bound (169,002 boxes) shows that the tree cut alone pins the functional to M_tree
there. The remaining statement, that every cone point dominates such a matrix, is needed only on a thin set N, where
it splits into a kill inequality and a positive-semidefiniteness condition. Both are reduced by hand to families of star
linear programs and certified by replayed branch and bounds (about 2.29 million records for the kill inequality and
1.5 million for positive semidefiniteness), each with an exact cover. The constant 4.6155432392 is the 10-decimal floor
of 1 + 1/0.27658360966404055, where 0.27658360966404055 is a certified upper bound of M_tree.

---

## How to read this document

Every statement below is either proved in full here by hand, or is a finite claim about explicit numbers that is checked
by computer (marked `FC-...`); the scripts and data that check each finite claim are listed in `PROOFS.md`. Published
theorems used without reproof are cited where used.

**Notation.** Plain-text formulas: `sqrt`, `arccos`, `<x, y>` for inner products, `|x|` for Euclidean norms, `x.y` for the
dot product of coefficient vectors, `1[E]` for the indicator of E. Vectors of the six correlation entries are always
listed in the order (ab, cd, ac, ad, bc, bd). All logarithms are base 2 unless written ln.

**Map.**

```text
Section 1   The result, what is new, the proof in one page, the trust base
Section 2   Preliminaries: circuits, Li's function, programs, kernels (Theorem K2C); Gaussian sign patterns;
            the field and keys; matched insertion; the 123 types and the cone; star lemma, rule bound; Lemma L
Section 3   Theorem A (matched insertion) and Corollary A1: liminf s_n/n >= 1 + 1/M_123
Section 4   The tree point: M_tree in closed form, its certified enclosure, the gradient at the tree
Section 5   The saddle is the cross determinant; the kill; Theorem WIT; Theorem TS*: (R-i') + (R-ii*) => M_123 = M_tree
Section 6   Theorem R1 = (R-i'): a four-parameter certificate
Section 7   Proposition NEED: (R-ii*) is needed only on the thin set N; N is thin; the tree cut is a two-star problem
Section 8   The kill inequality (K) on N: Theorem SEP, Proposition RED, the star certificates (Theorem K-N)
Section 9   The kill is positive semidefinite on N: the disc criterion, Theorem LR, the 640-root certificate
            (Theorem PSD-N)
Section 10  Theorem R2, Theorem M_tree, Theorem 4.6155, and the arithmetic
References
```

---

## 1. Introduction

### 1.1 The result

A *B2 circuit* on input variables x_1, ..., x_n is a directed acyclic graph whose sources are the variables and whose other
vertices, the *gates*, each have two ordered predecessors (possibly equal) and compute an arbitrary Boolean function of
them. One gate is the output. Fan-out is unrestricted. The *size* is the number of gates.

Let f_n : {0,1}^n -> {0,1} be Li's function (Section 2.1) and s_n the least size of a B2 circuit computing f_n.

**Theorem 1 (Theorem 4.6155).** (uses the finite claim FC-MT for the constant,
and those of Theorem M_tree) For
every eps > 0 there is n_0 such that for every n >= n_0, every B2 circuit computing f_n has at least
(1 + 1/M_tree - eps) n gates. In particular

    liminf_{n -> oo} s_n / n  >=  1 + 1/M_tree  >=  1 + 1/0.27658360966404055  =  4.61554323922041...  >=  4.6155432392.

Here M_tree := Phi_F(C_T) is the value of a Gaussian functional at an explicit 4 x 4 matrix (Section 4). Its decimal
expansion is known by quadrature to 30 digits, 0.276583606882373471022763934878 (numerical), which gives
1 + 1/M_tree = 4.6155432755827926... . But only the certified interval

    0.27658360687864314  <=  M_tree  <=  0.27658360966404055                                   (FC-MT)

enters the theorem, and the stated decimal 4.6155432392 is the 10-decimal floor of 1 + 1/(the certified upper end).

In words: Li's function needs about 4.6155 n gates. The theorem is the composition of two statements. The first is a
general reduction (Corollary A1, Section 3): for every family of B2 circuits computing f_n,

    liminf s_n/n >= 1 + 1/M_123,

where M_123 is the largest expected cost of one matched edge over a cone of 123 "local types". The second is new:

**Theorem M_tree.** (uses the finite claims FC-AAB, FC-MT, FC-LOC4D and FC-R1 (Section 6);
FC-R7, FC-D3, FC-THIN, FC-WIN, FC-PI (Section 7); FC-SEP, FC-LOC2K, FC-LOC2KC, FC-EMIN, FC-Q, FC-GSTAR (Section 8); FC-FBAR,
FC-CLR, FC-LR (Section 9)) M_123 = M_tree.

In words: among all local neighbourhoods that a matched edge of a cubic graph can see, the 3-regular tree is the most
expensive one.

### 1.2 What is new

*Li's contribution.* Li's paper [Li] supplies the hard function: an explicit two-source extractor with asymptotically
optimal entropy. We use one consequence of it (Lemma HARD, Section 2.1): f_n has no accepting combinatorial rectangle with
two large sides. Everything else below is independent of [Li].

*The framework.* The reduction from circuits to the cutwidth of cubic graphs (Theorem K2C, Section 2.4) is the reduction of the paper. It gives liminf s_n/n >= 1 + 1/(2 kappa)
whenever every kernel on h vertices has cutwidth at most (kappa + o(1)) h. With the trivial cutwidth bound of Fomin and
Hoie [FH] it gives 4 n. We reprove the reduction in Section 2 so that this document is self-contained, and we do not use [FH].

*Edge scores and matched insertion.* Ordering the edges of a kernel by Gaussian "edge scores" gives
kappa = kappa_E = (3/(2 pi)) arccos((1 + 2 sqrt 2)/4) = 0.1403510, hence 4.5624977 n. Placing each edge of a perfect matching on the better side of the threshold (matched insertion)
gives kappa = M_123/2.

*What this document adds.* The exact value M_123 = M_tree. It is the end of the matched-insertion cone:
since the tree point lies in the cone, no bound below M_tree can ever be proved for M_123, and the constant 1 + 1/M_tree
is the best this method gives.

*Comparison with the literature.* The best published lower bound for an explicit function over B2 is 3.1 n - o(n), by
Li and Yang [LY] (STOC 2022), improving 3n - o(n) and Find, Golovnev, Hirsch and Kulikov's (3 + 1/86) n - o(n) [FGHK].
These proofs use gate elimination. Ours does not: no variable is ever restricted and no gate is ever eliminated. The bound
comes from simulating a circuit by a read-once program (Section 2), and the role played in gate elimination by a potential
function is played here by the cycle rank of the circuit's wire graph. (This comparison is a summary, not a survey.)

### 1.3 The proof in one page

Here is the whole argument, top down. Each step names the section that proves it.

1. *Circuits to kernels* (Section 2). A circuit with s gates that reads b variables has a wire graph of cycle rank
   s - b + 1. Merging vertices of degree at most 2, parallel edges and triangles compresses it to a *kernel* (a simple,
   triangle-free, 2-connected cubic graph) on at most 2(s - b) vertices, at a logarithmic cost in cutwidth. An ordering of
   width w gives an unambiguous read-once program with about 2^w edges, and Li's function needs 2^{n - O(log n)} edges.
   So a kernel bound cw(H) <= (kappa + o(1)) h gives liminf s_n/n >= 1 + 1/(2 kappa) (Theorem K2C).

2. *Matched insertion* (Section 3). A kernel has a perfect matching (Petersen). Give each vertex x a Gaussian field
   vector, each edge a *key* (a unit vector built from the field), and sort the edges by key value, except that each
   matched edge uv enters at the moment when the better side of the threshold switches. The number of split vertices of
   uv at a threshold is a symmetric submodular function F of the signs of its four *boundary keys*. The threshold 0 is the
   worst, and more correlation never hurts (Lemma G). So the expected cost of uv is at most Phi_F(C), where C is the
   4 x 4 correlation matrix of its boundary keys, and C is (up to an error that vanishes with the field radius) C(m) for a
   point m of a cone with 123 generators. Concentration gives cw(H) <= (M_123/2 + o(1)) h, where M_123 = sup_m Phi_F(C(m)).

3. *The tree point* (Section 4). The point m_T of the cone that the 3-regular tree produces has C(m_T) = C_T with pair
   entries (1 + 2 sqrt 2)/4 and cross entries (2 + sqrt 2)/4, and M_tree := Phi_F(C_T) is a one-dimensional integral.

4. *The saddle* (Section 5). The tree cut a.C >= a.C_T (a the gradient of Phi_F at C_T, rescaled) is tight at the tree, but
   the cut alone does not bound Phi_F by M_tree, because Phi_F has one positive Hessian direction at C_T. That direction is
   the gradient of the cross determinant D(C) = C_ac C_bd - C_ad C_bc. So split the problem: (R-i') Phi_F <= M_tree on the
   PSD matrices with D = 0 that satisfy the cut; (R-ii*) every C(m) dominates, entry by entry, a PSD matrix with D = 0 that
   satisfies the cut. Since more correlation never hurts, (R-i') and (R-ii*) give M_123 = M_tree (Theorem TS*).

5. *(R-i')* (Section 6). On D = 0 the four keys share one Gaussian factor. In the resulting chart the cut may be taken
   tight, and (R-i') becomes an inequality in four parameters, with a nondegenerate maximum at the tree. A certified
   branch and bound proves it (Theorem R1).

6. *(R-ii\*) on N only* (Section 7). Where the rule bound RB(m) (an explicit upper bound of f built from the star lemma)
   is below a certified lower bound of M_tree, nothing more is needed. The rest, N, is thin: both stars of m are within 0.0025 of the tree in mean key
   correlation. On N it suffices to show (K): a.C(m) - a.C_T >= delta(m), where delta(m) is the least decrease of one cross
   entry that kills D; and that the matrix after this decrease (the *kill*) is PSD.

7. *(K) on N* (Section 8). An exact identity writes a.C(m) - a.C_T as 4 mu0 (s_u + s_v) plus three squares, where s_u is a
   function of the *u-star* of m alone (its masses on 20 local profiles) and s_v of the v-star alone, and the v-star is a
   mirrored u-star. A matching decomposition bounds delta(m) by a sum of two one-star terms. So (K) reduces to one
   inequality for a single star at two mirror marginals, which star linear programs certify (Theorem K-N).

8. *The kill is PSD on N* (Section 9). An exact criterion turns positive semidefiniteness into a disc condition in two
   alternating cross sums, and each sum is bounded by one-star functionals. A star certificate with "lower rows only"
   bounds these functionals as affine functions of each star's own excess, and an exact composition over the region that
   the two stars' excesses can share proves the disc condition (Theorem PSD-N).

9. *Assembly* (Section 10). (K) and the kill being PSD give (R-ii*) on N (Theorem R2). With Theorem R1 and step 6,
   M_123 = M_tree. With step 2 and FC-MT, Theorem 1.

### 1.4 The trust base

What must a reader believe? Here is the honest list.

*Hand proofs.* Every lemma, theorem and reduction is proved in this document.

*Computations.* The finite claims are of two kinds.

- Small certificates of constants (the enclosure of M_tree, a few interval bounds such as Lemma PI, the local Hessian
  certificates).
- Three large certificates: Theorem R1 (169,002 boxes in five trees), the kill inequality on N (337,645 + 1,801,388
  records plus a 154,246-leaf table) and the kill-PSD condition on N (1,495,347 records on 640 roots). For each, every
  record was replayed by a verifier, exact cover walkers were run over every record, and stratified samples were
  re-closed with independent code.

*Bit-reproducibility.* The certificates are deterministic given their inputs, and the replays reproduce them. But a
re-run from scratch on another machine may branch differently (floating-point library differences can change which
closure rule fires first), so a fresh run is a new certificate, not a byte copy of the old one.

### 1.5 Constants used throughout

| symbol | definition | value | status |
|---|---|---|---|
| rho | 2 sqrt 2/3 | 0.9428090416 | exact |
| mu0 = p | (1 + 2 sqrt 2)/4 | 0.9571067812 | exact |
| s | (2 + sqrt 2)/4 | 0.8535533906 | exact |
| kappa_E | (3/(2 pi)) arccos(mu0) | 0.14035096863582369634... | 200-bit intervals (FC-KAPPA) |
| 2 kappa_E | | 0.28070193727164739269... | FC-KAPPA |
| M_tree | Phi_F(C_T) | 0.276583606882373471022763934878 | numerical value; certified in [Mt_lo, Mt_hi] (FC-MT) |
| Mt_lo | certified lower bound of M_tree | 0.27658360687864314 | FC-MT |
| Mt_hi | certified upper bound of M_tree | 0.27658360966404055 | FC-MT |
| a_ab | ratio of a pair entry to a cross entry of grad Phi_F(C_T) | in [4.478385650326169, 4.478385650327289] | FC-AAB |
| a | (a_ab, a_ab, 1, 1, 1, 1) | | |
| K0 | a.C_T = 2 a_ab p + 4 s | 11.9868001118 | from FC-AAB |
| N0 | sqrt(2 + 2 rho) | 1.9711971193 | exact |

---

## 2. Preliminaries

This section collects everything the proof uses before the cone appears: the hard function, the simulation of circuits by
read-once programs, the reduction to kernels, Gaussian sign patterns, the field and its keys, the matched-edge cost, the
123 types, and the star lemma. Sections 2.1-2.4 reprove the reduction of the paper; Sections 2.5-2.8 give the Gaussian tools. All proofs are given.

### 2.1 Li's function

**Theorem LI.** ([Li], Theorem 7.13, with Definition 1.1 there for uniform polynomial-time computability, taken
with error 1/8 and applied to the uniform distributions on A and B, which are independent sources of min-entropy at least
c log n.) There are a constant c > 1 and a uniformly polynomial-time computable family
f_n : {0,1}^n -> {0,1} such that for all large n and all A, B in {0,1}^n with |A|, |B| >= n^c,

    3/8  <=  (1/(|A| |B|)) sum_{a in A, b in B} f_n(a XOR b)  <=  5/8.

We call f_n *Li's function*. Write c_Li for this c, K := n^{c_Li} and k := ceil(log K).

Split the coordinates {1, ..., n} into disjoint sets U and V. For P in {0,1}^U and Q in {0,1}^V the *rectangle* P x Q is the
set of inputs whose U-part lies in P and whose V-part lies in Q; its *sides* are P and Q. It is *accepting* for f if f = 1
on all of it.

**Lemma HARD.** (uses Theorem LI) For all large n:
(a) f_n has at least 2^{n-2} accepting inputs;
(b) every accepting rectangle of f_n, under every split of the coordinates, has a side of size less than K;
(c) f_n depends on more than n - k of its variables.

PROOF.
```text
<1>1. (a) Take A = B = {0,1}^n. For fixed a, b -> a XOR b is a bijection of {0,1}^n, so the average in Theorem LI is the
      fraction of inputs with f_n = 1. It is at least 3/8 >= 1/4.
<1>2. (b) Suppose P x Q is accepting with |P|, |Q| >= K. Pad P with zeros on V to get A, and Q with zeros on U to get B.
      As a, b range over A, B, a XOR b ranges over P x Q, where f_n = 1. So the average is 1 > 5/8, against Theorem LI.
<1>3. (c) Otherwise f_n ignores some set U of k variables. Let P = {0,1}^U (size 2^k >= K) and Q the set of assignments to
      the other variables on which f_n = 1. Since f_n ignores U, P x Q is accepting, and by (a) |Q| >= 2^{n-2}/2^k >= K
      for large n. This contradicts (b).   QED
```

### 2.2 Read-once programs and the counting lemma

A *read-once program* on x_1, ..., x_n has pairwise disjoint vertex sets V_0, ..., V_l, with V_0 a single *start* vertex
and V_l a single *accept* vertex, and directed edges each going from some V_{i-1} to V_i. The edges from V_{i-1} to V_i form
the i-th *layer*. A layer is either a *query layer*, tagged with a variable x_j, whose edges carry labels 0 or 1, or a
*guess layer*, whose edges carry no labels. Every variable tags exactly one query layer. Parallel edges are allowed and
counted separately; the *size* is the number of edges. A start-to-accept path is *consistent* with an input a if each of
its query edges tagged x_j carries the label a_j. The program *accepts* a if some consistent start-to-accept path exists,
and it *computes* f if it accepts exactly the inputs with f = 1. It is *unambiguous* if every accepted input has exactly
one consistent path. The *past* of V_i is the set of variables tagging layers 1, ..., i; its *future* is the rest.

**Lemma COUNT.** Let f have exactly M >= K > 1 accepting inputs, and let every accepting rectangle of f have a
side of size less than K. Then every read-once program computing f has more than M/K^2 edges.

PROOF IDEA. Charge each accepted input to one edge of its path; no edge is charged K^2 times.

```text
<1>1. For a vertex v in V_i let P_v be the set of assignments to the past of V_i that admit a consistent path from start
      to v, and Q_v the set of assignments to the future that admit a consistent path from v to accept. The two paths
      constrain disjoint variables and concatenate, so P_v x Q_v is an accepting rectangle, and min(|P_v|, |Q_v|) < K.
<1>2. Fix for each accepted input one consistent path. Along it |P_v| goes from 1 < K at start to M >= K at accept (at
      accept the past is everything and P is the set of accepted inputs). Let e = (u, v) be the first edge with
      |P_u| < K <= |P_v|, and charge the input to e. By <1>1, |Q_v| < K.
<1>3. An input charged to e is determined by its part on the past of u (in P_u), its part on the future of v (in Q_v), and,
      if e lies in a query layer, the value of the queried variable, which is the label of e. (In a guess layer the past of
      u and the future of v already exhaust the variables.) So fewer than K^2 inputs are charged to e, and M < |E| K^2.  QED
```

By Lemma HARD (a), (b) and Lemma COUNT, for all large n every read-once program computing f_n has more than

    2^{n-2}/K^2 = 2^{n - 2 c_Li log n - 2}                                                            (2.1)

edges.

### 2.3 Wire graphs and the simulation

A variable or gate is a *signal*. A signal is *read* if it has a directed path to the output. *Pruning* deletes everything
not read; the pruned circuit computes the same function. Assume the circuit has at least one gate.

*The wire graph.* Route every signal with f >= 2 uses through a path of f - 1 *copy vertices*, each with one incoming and
two outgoing wires. The *wire graph* G of a pruned circuit with t gates and b read variables is the undirected multigraph
with a vertex for each gate, read variable and copy vertex, and an edge for each wire. Each of the t + b - 1 non-output
signals has at least one use and there are 2t uses, so there are t + 1 - b copy vertices. Counting edges into gates and
copy vertices,

    N = 2t + 1 vertices,   M = 3t + 1 - b edges,   cycle rank mu(G) := M - N + 1 = t - b + 1.            (2.2)

Gates have degree 3 except the output (degree 2), copy vertices have degree 3, variables degree 1. G is connected (every
vertex reaches the output) and loopless (the circuit is acyclic); a gate reading one signal twice gives a parallel pair.
For an ordering u_1, ..., u_N of the vertices, the *cut* F_i after u_i is the set of edges with exactly one endpoint among
u_1, ..., u_i (F_0 = F_N = empty). The *width* of the ordering is max_i |F_i|, and the *cutwidth* cw(G) is the least width.

**Lemma SIM.** Let G be the wire graph of a pruned circuit with at most s gates on n variables, and let G have an
ordering of width w. Then there is an unambiguous read-once program on x_1, ..., x_n that computes the circuit's function,
with at most 4(2s + 1) 2^w + 2n edges.

PROOF IDEA. Walk along the ordering; the program's state is the vector of bits on the wires that cross the current cut.
Each vertex checks its own wires as it is passed.

```text
<1>1. (checks) Each edge carries one bit. A non-output gate computing g requires z = g(a, b) for the bits a, b on its
      incoming wires (in slot order) and z on its outgoing wire; the output gate requires g(a, b) = 1; a copy vertex
      requires its three bits to agree; a variable allows any bit. For each input there is exactly one bit assignment to
      the edges that puts each read variable's value on its edge and satisfies all non-output and copy checks (evaluate the
      circuit in topological order); it satisfies the output check iff the circuit accepts the input.
<1>2. (the program) V_i is the set of pairs (i, sigma), sigma in {0,1}^{F_i}; V_0 = {start}. To process u_i let B be its edges
      to earlier vertices, O its edges to later vertices, and R = F_{i-1} \ B; then F_i = R u O. For every sigma in
      {0,1}^{F_{i-1}} and omega in {0,1}^O such that sigma|_B u omega satisfies the check of u_i, add an edge from
      (i - 1, sigma) to (i, sigma|_R u omega). If u_i is a gate or a copy vertex this is a guess layer. If u_i is the
      variable x_j, it is a query layer tagged x_j, and the edge carries the bit that sigma|_B u omega puts on u_i's single
      edge. After layer N append one two-edge query layer (labels 0 and 1) for each of the n - b unread variables, ending at
      accept.
<1>3. (correctness) A consistent start-to-accept path is exactly a bit assignment to the edges of G that satisfies every
      check and puts each read variable's value on its edge: every edge is recorded when its first endpoint is processed
      and checked when its second endpoint is processed. By <1>1 there is exactly one such path if the circuit accepts and
      none otherwise. So the program computes the function and is unambiguous.
<1>4. (size) A gate check has 4 satisfying triples, a copy check 2, the output check at most 4; fixing bits only lowers
      these numbers. So each vertex has at most 4 outgoing edges. Each V_i (i < N) has at most 2^w vertices, there are
      N <= 2s + 1 such layers, and the final layers have 2 edges each: at most 4 (2s + 1) 2^w + 2n edges.   QED
```

### 2.4 Kernels, and the reduction from circuits to kernels

Graphs here are finite loopless multigraphs; parallel edges count separately in degrees. A graph is *cubic* if every
degree is 3 and *subcubic* if every degree is at most 3. A graph is *2-connected* if it is connected, has at least three
vertices, and stays connected after deleting any one vertex.

**Definition (kernel).** A *kernel* is a simple, triangle-free, 2-connected cubic graph. We write h for its number of
vertices.

A *bridge* of a connected graph is an edge whose deletion disconnects it; an edge is a bridge iff it lies on no cycle
(two parallel edges form a cycle). *Contracting* a vertex set S replaces S by one vertex, deletes the edges inside S and
redirects the others.

**Lemma COMPRESS.** Let B be a connected bridgeless subcubic graph with N_B vertices. Either B has an ordering of
width at most 6 ceil(log N_B), or there is a kernel H on h <= 2(mu(B) - 1) vertices such that, for every q, an ordering of
H of width at most q yields an ordering of B of width at most q + 6 ceil(log N_B).

PROOF IDEA. Merge vertices until a kernel is left; no merge raises the cycle rank. To expand an ordering, list each merged
group by walking its merge tree, larger subtrees first; then a partial group is a union of few pieces, each with at most
three leaving edges.

```text
<1>1. If N_B = 1 the first alternative holds (width 0). Let N_B >= 2. Apply these rules while one applies, preferring rule 1.
      Rule 1: if adjacent p != q, joined by r >= 1 edges, have deg p <= 2 or deg q <= 2 or r >= 2, contract {p, q}.
      Rule 2: if rule 1 does not apply and a, b, c are pairwise adjacent, contract {a, b, c}.
<1>2. Rule 1 makes a vertex of degree deg p + deg q - 2r <= 3 and lowers M - N by r - 1 >= 0. When rule 2 applies, every
      degree is 3 and the graph is simple, so each of a, b, c has one edge leaving the triangle, the new vertex has degree
      3, and M - N drops by 1. Each rule lowers N, so the process stops. Contraction keeps the graph connected, loopless
      and bridgeless (a path avoiding an edge maps to a walk avoiding it).
<1>3. When it stops, either one vertex is left, or the graph H is cubic (rule 1 with degree <= 2), simple (rule 1 with
      r >= 2) and triangle-free (rule 2). Then h/2 = |E(H)| - |V(H)| <= M_B - N_B = mu(B) - 1. H is 2-connected: if
      deleting v left components C_1, ..., C_m with m >= 2, each C_i would send at least two edges to v (one edge would be a
      bridge), so deg v >= 4.
<1>4. (pieces) Record the contractions as a rooted tree for each current vertex (leaves = original vertices, internal nodes
      with 2 or 3 children). The leaf set of a subtree is a *piece*. A piece was a current vertex of degree <= 3 when it was
      formed, and its set of leaving edges never changes afterwards; so at most 3 edges of B leave a piece.
<1>5. (expanding one piece) List the leaves depth first, visiting children in nonincreasing order of their leaf counts. A
      proper prefix L, with first missing leaf l and root path x_0, ..., x_m = l, is the union over i < m of the pieces of
      the children of x_i that precede x_{i+1}. If x_{i+1} is the j-th child it contributes j - 1 <= 2 pieces, and if j >= 2
      then x_{i+1} has at most half the leaves of x_i. So j >= 2 for at most ceil(log N_B) indices, L is a union of at most
      2 ceil(log N_B) pieces, and at most 6 ceil(log N_B) edges leave it.
<1>6. (expanding an ordering) If one vertex is left, its leaf list is an ordering of B of width <= 6 ceil(log N_B). Otherwise
      list the pieces of the vertices of H in the given order, each expanded by <1>5. A prefix is a prefix of H's ordering
      (whole pieces) plus a proper prefix of one more piece; an edge leaving it is either in the corresponding cut of H or
      leaves the partial piece. So every cut has at most q + 6 ceil(log N_B) edges.   QED
```

The *blocks* of a connected subcubic graph G are the components of G with all bridges deleted (a block may be one vertex).

**Lemma ASSEMBLY.** Let G be connected subcubic, B a block, and F_1, ..., F_m (m >= 0) the components of
G - V(B). (a) B is bridgeless; each F_j is joined to B by exactly one edge e_j, ending at v_j in B; every edge of G lies in
B, in some F_j, or is some e_j; hence mu(G) = mu(B) + sum_j mu(F_j). (b) If B has an ordering of width w and each F_j one
of width w_j, then G has an ordering of width at most w + max_j w_j + 3 (max of the empty set = 0).

```text
<1>1. (a) An edge with both endpoints in V(B) is in B (else it is a bridge whose endpoints are joined by a path in B,
      closing a cycle). An edge with one endpoint in V(B) is a bridge. If two edges xy, x'y' joined F_j to B (x, x' in F_j),
      paths x' -> x in F_j and y -> y' in B would close a cycle through the bridge xy; so exactly one edge joins F_j to B.
      No edge joins F_i to F_j. An edge f of B lies on a cycle of G, which uses no bridge, hence lies in B; so f is not a
      bridge of B. Counting: M - N + 1 = (M_B - N_B + 1) + sum_j (M_j - N_j) + m = mu(B) + sum_j mu(F_j).
<1>2. (b) Walk the ordering of B; on reaching v, first list the orderings of all F_j with v_j = v, then v. Let Q be a prefix
      not containing all of B and u the first vertex of B not in Q. At most w edges of B cross. An F_j with v_j before u
      lies in Q, one with v_j after u lies outside. Of the F_j with v_j = u (listed consecutively) at most one is split,
      in a prefix of its ordering: at most max_j w_j edges. A bridge e_j crosses only if v_j = u: at most 3. If Q contains
      all of B the cut is empty.   QED
```

A block B is a *centroid* of a connected subcubic G with N vertices if every component of G - V(B) has at most N/2
vertices.

**Lemma CENTROID.** Every connected subcubic graph has a centroid block.

```text
<1>1. Start at any block. If the current block B has a component F of G - V(B) with more than N/2 vertices (at most one
      can), let xy be the edge joining F to B with x in F, and move to the block B' containing x. B' is bridgeless and
      connected, so B' lies in V(F). Every path from B into F passes through x, so the component of G - V(B') containing B
      is V(G) \ V(F), of fewer than N/2 vertices. Hence a large component of G - V(B'), if any, lies inside F \ V(B') and
      is smaller than F. The walk stops, at a centroid.   QED
```

**Theorem K2C (kernels to circuits).** (uses Theorem LI) Let kappa > 0, and suppose that for every
eta > 0 there is C_eta >= 0 such that every kernel H on h vertices has cw(H) <= (kappa + eta) h + C_eta. Then
(a) for every eta > 0 there is D_eta such that every connected subcubic graph G with N vertices has an ordering of width at
    most (2 kappa + 2 eta) mu(G) + D_eta ceil(log(N + 1))^2;
(b) liminf_n s_n/n >= 1 + 1/(2 kappa).

In words: a kernel constant kappa gives the circuit constant 1 + 1/(2 kappa). The factor 2 is h <= 2(mu - 1); the 1 is the
n variables, which cost no cycle rank but must all be read.

PROOF IDEA. Order a centroid block through its kernel, order the hanging components recursively, and interleave. Each
component has at most half the vertices, so a cut pays a block's additive error once on each of about log N levels. For
(b), compare Lemma SIM with (2.1).

```text
<1>1. (a) Fix eta, put c = 2 kappa + 2 eta and K_N = 6 ceil(log N) + C_eta + 3.
  <2>1. (blocks) Let B be connected, bridgeless, subcubic, N_B <= N. By Lemma COMPRESS either B has an ordering of width
        <= 6 ceil(log N_B), or its kernel H has h <= 2(mu(B) - 1) and cw(H) <= (kappa + eta) h + C_eta <= c mu(B) + C_eta.
        Either way B has an ordering of width <= c mu(B) + K_N - 3 (mu(B) >= 0).
  <2>2. (recursion) Claim: every connected subcubic G' with N' <= N vertices has an ordering of width at most
        c mu(G') + (floor(log N') + 1) K_N. Induction on N'. Take a centroid block B (Lemma CENTROID) and the components F_j
        of G' - V(B). B is bridgeless (Lemma ASSEMBLY (a)), so it has width <= c mu(B) + K_N - 3. Each F_j has <= N'/2
        vertices, so floor(log |F_j|) <= floor(log N') - 1 and by induction width <= c mu(F_j) + floor(log N') K_N. By
        Lemma ASSEMBLY (b) and mu(G') = mu(B) + sum mu(F_j) >= mu(B) + max mu(F_j), G' has width
        <= c mu(B) + K_N - 3 + c max_j mu(F_j) + floor(log N') K_N + 3 <= c mu(G') + (floor(log N') + 1) K_N.
        (m = 0 gives c mu(B) + K_N - 3; this includes N' = 1.)
  <2>3. With L_N = ceil(log(N + 1)) = floor(log N) + 1 >= ceil(log N), the additive term is <= L_N (6 L_N + C_eta + 3)
        <= D_eta L_N^2 with D_eta = C_eta + 9.
<1>2. (b) Fix eta, c = 2 kappa + 2 eta. Let a circuit with s = s_n gates compute f_n. By Lemma HARD (c) it reads
      b >= n - k + 1 variables (it depends only on read variables), and it has a gate. Its wire graph is connected and
      subcubic with N = 2t + 1 <= 2s + 1 and mu = t - b + 1 <= s - n + k by (2.2). Put l = log(s + n + 2) >= 1. By (a) it has
      an ordering of width w <= c (s - n + k) + D_eta ceil(log(2s + 2))^2 <= c (s - n + k) + 9 D_eta l^2
      (ceil(log(2s + 2)) <= l + 2 <= 3l). By Lemma SIM, f_n has a read-once program with at most
      4(2s + 1) 2^w + 2n <= 2^{w + 3 + l} edges. With (2.1), for large n:
          n - 2 c_Li log n - 2 < c (s - n + k) + (9 D_eta + 4) l^2.
<1>3. l^2 <= 2 log^2(s + 2) + 2 log^2(n + 2), and 2(9 D_eta + 4) log^2(s + 2) <= eta s + C'_eta for all s >= 0. Hence
          (c + eta) s_n > (1 + c) n - c k - 2 c_Li log n - 2 - 2(9 D_eta + 4) log^2(n + 2) - C'_eta.
      Since k = O(log n), liminf s_n/n >= (1 + 2 kappa + 2 eta)/(2 kappa + 3 eta). Let eta -> 0.   QED
```

The order of quantifiers: f_n and c_Li first, then eta, then C_eta, D_eta, C'_eta, and n last. Part (b) needs only the
existence of the orderings.

*Gate elimination and potentials, for comparison.* The published bounds [FGHK], [LY] fix variables or gates and track a
potential (a weighted count of gates and inputs) that must drop at each step. Nothing of that kind happens here. The
circuit is never restricted; the cycle rank s - b + 1 plays the role of the potential, and the read-once program pays one
bit per unit of kernel cutwidth. This is why the bound has the shape 1 + 1/M: the n inputs are free, and each further gate
buys one unit of cycle rank, which buys at most M bits of width.

### 2.5 Gaussian sign patterns

Everything Gaussian in this document rests on three facts about sign patterns: the threshold 0 is the worst, more
correlation never hurts, and two simple patterns have closed forms.

For x in R^k and a real theta let s_theta(x) in {0,1}^k be the pattern with s_theta(x)_i = 1 exactly when x_i <= theta. A
function F : {0,1}^k -> R is *symmetric* if F(1 - s) = F(s). It is *submodular* if every mixed difference

    Delta_ij F(s) = F(s + e_i + e_j) - F(s + e_i) - F(s + e_j) + F(s)      (i != j, s_i = s_j = 0)

is at most 0. A *correlation matrix* is a positive semidefinite (PSD) matrix with unit diagonal; it may be singular. For a
k x k correlation matrix C let Psi_theta(F; C) = E F(s_theta(X)) with X ~ N(0, C).

**Lemma G (sign patterns).** Let F : {0,1}^k -> R be submodular.
(a) If F is also symmetric, then Psi_theta(F; C) <= Psi_0(F; C) for every theta.
(b) If C_0, C_1 are correlation matrices with (C_1)_ij >= (C_0)_ij for all i != j, then Psi_theta(F; C_1) <= Psi_theta(F; C_0)
    for every theta.
(c) With r_ij = C_ij: Pr[X_1, X_2 differ in sign] = arccos(r_12)/pi if k = 2, and
    Pr[X_1, X_2, X_3 are not all of one sign] = (1/(2 pi)) sum_{i<j} arccos(r_ij) if k = 3.

In other words, for these costs the threshold 0 is the worst one, and *more correlation never hurts*. Part (b) is the
single most used fact in this document.

PROOF IDEA. For (a), differentiate in theta; each term of the derivative is a conditional expectation given X_a = theta,
and reflecting the other coordinates through their conditional mean turns submodularity and symmetry into its sign. For
(b), interpolate between C_0 and C_1; the derivative of the Gaussian density is a sum of mixed second derivatives with
weights (C_1 - C_0)_ij >= 0, and integrating F against a mixed second derivative gives a mixed difference of F.

```text
<1>1. (reduction to positive definite C) Each X_i has variance 1, so a.s. no coordinate equals a given number. Let
      W ~ N(0, I) be independent and X^eps = sqrt(1 - eps) X + sqrt(eps) W. Its correlation matrix (1 - eps) C + eps I is
      positive definite, and the order of off-diagonal entries in (b) is preserved. As eps -> 0, X^eps -> X pointwise and
      s_theta is constant near X a.s., so E F(s_theta(X^eps)) -> E F(s_theta(X)) by bounded convergence. So it suffices to
      prove (a) and (b) for positive definite matrices.
<1>2. (a) Evenness: -X has the law of X and s_{-theta}(-X) = 1 - s_theta(X), so symmetry gives Psi_{-theta} = Psi_theta.
      Monotonicity on [0, oo): write F = sum_A c_A prod_{i in A} s_i, so Psi_theta = sum_A c_A Pr[X_i <= theta for i in A].
      Differentiating each orthant probability,
          d Psi_theta/d theta = phi(theta) sum_a E[ d_a F(s_theta(Y)) | X_a = theta ],   d_a F(t) = F(1_a, t) - F(0_a, t),
      where Y is X without coordinate a and phi the standard normal density. Fix theta >= 0 and a. Given X_a = theta, Y is
      Gaussian with mean m = theta (C_aj)_{j != a}, so m_j <= theta. The reflection Y' = 2m - Y has the same conditional law,
      and Y_j > theta implies Y'_j < 2 m_j - theta <= theta; so s_theta(Y') >= 1 - s_theta(Y) coordinatewise. Submodularity
      says d_a F is nonincreasing; symmetry gives d_a F(1 - t) = -d_a F(t). Hence, conditionally on X_a = theta,
          E d_a F(s_theta(Y)) = E d_a F(s_theta(Y')) <= E d_a F(1 - s_theta(Y)) = -E d_a F(s_theta(Y)),
      so each term is <= 0. Psi_theta is even and nonincreasing on [0, oo), hence largest at 0.
<1>3. (b) Let D = C_1 - C_0 (D_ii = 0, D_ij >= 0), C_u = (1 - u) C_0 + u C_1 and p_u the density of N(0, C_u). The Fourier
      transform of p_u is exp(-xi^T C_u xi/2), whose u-derivative is (1/2) sum_{i != j} D_ij (i xi_i)(i xi_j) exp(...); so
          d p_u/du = (1/2) sum_{i != j} D_ij d_i d_j p_u.
      (Differentiation under the integral is justified by dominated convergence: the eigenvalues of C_u lie in
      [lambda_*, k] with lambda_* > 0, which bounds p_u and its first two derivatives by one integrable Gaussian envelope.)
      Hence d/du Psi_theta(F; C_u) = (1/2) sum_{i != j} D_ij I_ij with I_ij = Int F(s_theta(x)) d_i d_j p_u(x) dx.
<1>4. Write x = (x_i, x_j, x'). For fixed x', F(s_theta(x)) is constant on each quadrant of the (x_i, x_j)-plane around
      (theta, theta). The integral of d_i d_j p_u over a quadrant is +p_u(theta, theta, x') for {x_i, x_j <= theta} and
      {x_i, x_j > theta}, and -p_u(theta, theta, x') for the other two. So
          I_ij = Int p_u(theta, theta, x') Delta_ij F(s_theta(x')) dx' <= 0
      for submodular F. So Psi_theta(F; C_u) is nonincreasing in u.
<1>5. (c) For k = 2 and r = r_12 < 1 write (X_1, X_2) = (U, r U + sqrt(1 - r^2) V) with U, V independent N(0, 1). The event
      {X_1 > 0, X_2 > 0} is a sector of opening pi - arccos r of the rotation-invariant plane, so has probability
      (pi - arccos r)/(2 pi); by X -> -X the signs differ with probability 1 - 2 (pi - arccos r)/(2 pi) = arccos(r)/pi.
      (r = 1: X_1 = X_2.) For k = 3: three nonzero numbers have 3 pairs of equal sign if all signs agree and 1 otherwise.
      Taking expectations, 1 + 2q = sum_{i<j} (1 - arccos(r_ij)/pi), q = Pr[all signs agree], so
      1 - q = (1/(2 pi)) sum_{i<j} arccos(r_ij).   QED
```

The proof of (b) gives a formula for the derivative itself. We need it for one function.

**Lemma DERIV (Plackett's partial derivatives).** Let G : {0,1}^4 -> R and E_C G := E G(s_0(X)), X ~ N(0, C). On
positive definite 4 x 4 correlation matrices, C -> E_C G has continuous partial derivatives in each entry r_p (the two
symmetric entries moving together), and for p = {i, j} with complementary pair {k, l},

    d E_C G / d r_p = (1/(2 pi sqrt(1 - r_p^2))) E[ Delta_ij G(1[Y_k <= 0], 1[Y_l <= 0]) ],

where (Y_k, Y_l) is centred Gaussian with unit variances and correlation rho_p(C), the *partial correlation* of X_k and
X_l given X_i = X_j = 0.

```text
<1>1. With E = e_i e_j^T + e_j e_i^T, C + hE is a positive definite correlation matrix for small |h|. Apply <1>3-<1>4 of
      Lemma G with theta = 0, C_0 = C, C_1 = C + hE (that computation holds for every G):
      E_{C+hE} G - E_C G = h Int_0^1 Int p_M(0, 0, x') Delta_ij G(s_0(x')) dx' du, M = C + u h E. Dominated convergence on a
      compact neighbourhood of C gives the derivative as the inner integral at M = C, continuous in C.
<1>2. p_C(0, 0, x') is the density of (X_i, X_j) at (0, 0), 1/(2 pi sqrt(1 - r_p^2)), times the conditional density of
      (X_k, X_l) given X_i = X_j = 0, a centred Gaussian; Delta_ij G depends only on the signs of x_k, x_l, which do not
      change on dividing by the conditional standard deviations.   QED
```

**The matched-edge cost and its derivatives.** The function used throughout, with coordinates ordered (ua, ub, vc, vd)
and abbreviated (a, b, c, d), is

    F(s) = 1[s_a != s_b] + 1[s_c != s_d] + 1[s_a = s_b != s_c = s_d],                                   (2.3)

and Phi_F(C) := Psi_0(F; C) = E F(s_0(X)), X ~ N(0, C). (Section 2.6 says where F comes from.) Its mixed differences are,
by direct evaluation (Lemma MI below gives the table),

    Delta_ab F = -1 - 1[s_c != s_d],   Delta_cd F = -1 - 1[s_a != s_b],   Delta_ij F = -1[s_k != s_l] (ij a cross pair),

where the *pair entries* are ab, cd, the *cross entries* ac, ad, bc, bd, and kl is the complementary pair. So F is
submodular. With Lemma DERIV and Lemma G(c) this gives, on positive definite C,

    dPhi_F/dC_ab = -(3/4 - arcsin(rho_ab)/(2 pi)) / (pi sqrt(1 - C_ab^2)),
    dPhi_F/dC_ac = -(1/4 - arcsin(rho_ac)/(2 pi)) / (pi sqrt(1 - C_ac^2)),                                (2.4)

and likewise for cd and the other cross entries, where rho_p = rho_p(C) is the partial correlation of the complementary
pair given the pair p. In particular every partial derivative of Phi_F is <= 0, and the four cross derivatives are < 0
when |rho_p| < 1. Also, by Lemma G(c) and the symmetry X -> -X,

    Phi_F(C) = (arccos C_ab + arccos C_cd)/pi + 2 Pr[X_a > 0, X_b > 0, X_c < 0, X_d < 0].                  (2.5)

**Lemma PATH.** Let C be a positive definite correlation matrix and C(t) the matrix C with its four cross entries
multiplied by t in [0, 1]. Then C(t) is positive definite and

    Phi_F(C) = (arccos C_ab + arccos C_cd)/pi + 2 (1/4 + arcsin(C_ab)/(2 pi)) (1/4 + arcsin(C_cd)/(2 pi))
               - (1/pi) Int_0^1 sum_x C_x (1/4 - arcsin(rho_x(C(t)))/(2 pi)) / sqrt(1 - t^2 C_x^2) dt,

the sum over the four cross entries x.

```text
<1>1. C(t) = (1 - t) C(0) + t C, and C(0) is block diagonal with positive definite blocks, so C(t) is positive definite.
<1>2. Let P(t) = Pr[X_a, X_b > 0 > X_c, X_d] under C(t). At t = 0 the pairs are independent and Pr[X_a, X_b > 0] =
      (1/2)(1 - arccos(C_ab)/pi) = 1/4 + arcsin(C_ab)/(2 pi); likewise for c, d.
<1>3. Apply Lemma DERIV to the indicator G of the pattern (0, 0, 1, 1). For the cross pair ac, Delta_ac G = -1[s_b = 0,
      s_d = 1], so dP/dC_ac = -(1/(2 pi sqrt(1 - C_ac^2))) Pr[Y_b > 0 >= Y_d], and Pr[Y_b > 0 >= Y_d] =
      arccos(rho_ac)/(2 pi) = 1/4 - arcsin(rho_ac)/(2 pi). The pair entries do not move along the path. Integrate in t
      (dC(t)_x/dt = C_x) and use (2.5).   QED
```

Lemma PATH is the basis of the enclosure codes used by the certificates (Section 4.3): they enclose the integrand
by interval Taylor series on subintervals of [0, 1] and integrate.

### 2.6 The field, the keys, and the cost of a matched edge

Fix an integer R >= 1 (the *field radius*). In a graph H give each vertex x the row a_x in R^{V(H)} with

    a_x(z) = 2^{-d(x,z)/2} if d(x, z) <= R,   a_x(z) = 0 otherwise,

where d is the graph distance. Its entries are nonnegative and a_x(x) = 1. Put â_x = a_x/|a_x|. Let omega = (omega_z) be
independent standard normal variables (the *noise*). For an edge e = xy the *key* is the unit vector

    y_e = (â_x + â_y)/|â_x + â_y|          (|â_x + â_y|^2 = 2 + 2 <â_x, â_y> >= 2),

and the *score* of e is k_e = <y_e, omega>. By the basic fact that (<y_i, omega>)_i is a centred Gaussian vector with
covariances <y_i, y_j>, every score is N(0, 1) and the scores of two edges have correlation <y_e, y_f>.

*Matched edges.* Let M be a matching of a simple cubic graph H, and let uv be in M. Label N(u) = {v, a, b} and
N(v) = {u, c, d}. The *boundary edges* of uv are ua, ub, vc, vd, always in this order; they are four distinct edges and
none is in M. At a threshold theta, the *boundary sides* are s_uv(theta) = (1[k_e < theta])_{e = ua, ub, vc, vd}. If uv is
put on side x in {0, 1}, the number of split vertices among u, v is

    g(x; s) = 1[x, s_1, s_2 not all equal] + 1[x, s_3, s_4 not all equal].

**Lemma MI (the optimal side).** For every s in {0,1}^4: g(1; s) - g(0; s) = n00(s) - n11(s), where n11 (n00) is
the number of the pairs {1, 2}, {3, 4} whose two entries are both 1 (both 0); so x*(s) := 1[n11(s) > n00(s)] is the least
minimiser of g(.; s). Moreover min_x g(x; s) = F(s), with F as in (2.3), and F is symmetric and submodular, with the mixed
differences listed in Section 2.5.

In words: the matched edge goes below exactly when more boundary pairs lie entirely below than entirely above, and then it
splits only F of its two endpoints.

```text
<1>1. For one pair (s_1, s_2), compare x = 1 with x = 0 in 1[x, s_1, s_2 not all equal]: if s_1 = s_2 = 1 the difference is
      -1; if s_1 = s_2 = 0 it is +1; if s_1 != s_2 it is 0. Adding the two pairs gives n00 - n11.
<1>2. min_x g(x; s) = F(s): both pairs split gives 2 for both x; exactly one pair split gives 1 (x equal to the unsplit
      pair's value); all four equal gives 0; s_1 = s_2 != s_3 = s_4 gives 1 for every x.
<1>3. F(1 - s) = F(s) since every term compares coordinates. F is invariant under s_1 <-> s_2, s_3 <-> s_4 and
      (s_1, s_2) <-> (s_3, s_4).
<1>4. Mixed differences. Pair (1, 2), with (s_3, s_4) free: s_3 != s_4 gives 1 - 2 - 2 + 1 = -2; s_3 = s_4 gives -1. Pair
      (1, 3), with (s_2, s_4) free: (0,0) gives 2 - 1 - 1 + 0 = 0; (0,1) gives 1 - 2 - 1 + 1 = -1; (1,0) gives -1; (1,1) gives
      0 - 1 - 1 + 2 = 0. So Delta_13 F = -1[s_2 != s_4]. The other pairs are images under <1>3. All are <= 0.   QED
```

### 2.7 The 123 types and the cone

Order the six labelled vertices of a matched edge as (u, v, a, b, c, d) and its five *internal edges* as uv, ua, ub, vc,
vd. A vector t = (t_u, t_v, t_a, t_b, t_c, t_d) in Z^6 with t_u = 0 is a *generic type* if

    (E)  |t_x - t_y| <= 1 for the five internal edges xy;
    (Ru) min(t_v, t_a, t_b) = -1;
    (Rv) min(t_u, t_c, t_d) = t_v - 1.

The *special types* are the 4 vectors (0, 1, 1, 1, t_c, t_d) with t_c, t_d in {1, 2} and the 4 vectors (0, -1, t_a, t_b, 0, 0)
with t_a, t_b in {0, 1}. T123 is the set of generic and special types.

**Lemma TYPES.** Let uv be an edge of a simple cubic graph H, labelled as above, and z any vertex. Then
t(z) := (d(x, z) - d(u, z))_{x = u, v, a, b, c, d} lies in T123, and |T123| = 123. Every type has |t_x| <= 2.

In words: seen from the edge uv, every vertex of the graph looks like one of 123 distance-offset patterns.

```text
<1>1. t_u = 0, and |d(x,z) - d(y,z)| <= d(x,y) = 1 on each internal edge: (E). So t_v, t_a, t_b in {-1, 0, 1} and
      t_c, t_d in t_v + {-1, 0, 1}; hence |t_x| <= 2.
<1>2. z not in {u, v}: the second vertex of a shortest u-z path is a neighbour of u at distance d(u,z) - 1, so
      min(t_v, t_a, t_b) = -1 by <1>1: (Ru). Likewise at v: min(t_u, t_c, t_d) = (d(v,z) - 1) - d(u,z) = t_v - 1: (Rv).
<1>3. z = u: t = (0, 1, 1, 1, d(c,u), d(d,u)) with d(c,u), d(d,u) in {1, 2} (c != u, c-v-u is a path). z = v:
      t = (0, -1, d(a,v) - 1, d(b,v) - 1, 0, 0) with d(a,v), d(b,v) in {1, 2}. Both are special types.
<1>4. Count of generic types by t_v. t_v = -1: (Ru) automatic, 9 pairs (t_a, t_b); (Rv) asks min(t_c, t_d) = -2 with
      t_c, t_d in {-2, -1, 0}: 5 pairs; 45. t_v = 0: 5 pairs (t_a, t_b) with min = -1, 5 pairs (t_c, t_d) with min = -1; 25.
      t_v = 1: 5 pairs (t_a, t_b); (Rv) min(0, t_c, t_d) = 0 with t_c, t_d in {0, 1, 2} always holds: 9; 45. Total 115.
<1>5. No special type is generic (the z = u types violate (Ru), the z = v types violate (Rv)), and the 8 special types are
      distinct. So |T123| = 123.   QED
```

(Four independent enumerations agree on the 123 types; a breadth-first search over 780,912 neighbourhood profiles in 33
graphs found all 123 realised and no others. Neither fact is used.)

**The cone.** For t in T123 the *generator* is g_t = (2^{-t_x/2})_{x = u, v, a, b, c, d}, a vector in R^6 with positive entries
and g_t(u) = 1. For m in [0, oo)^{T123} \ {0} put

    G(m) = sum_t m_t g_t g_t^T.

For a 6 x 6 PSD matrix G with positive diagonal put Ghat = corr G (Ghat_xy = G_xy/sqrt(G_xx G_yy)), let B be the 4 x 6
incidence matrix of the boundary edges (row ua has ones in columns u and a, and so on), W = B Ghat B^T, and C(G) = corr W.
Put C(m) := C(G(m)) and

    f(m) := Phi_F(C(m)),        M_123 := sup { f(m) : m in [0, oo)^{T123} \ {0} }.

f is invariant under scaling m, so the supremum is over the simplex sum_t m_t = 1, where f is continuous; so it is
attained. (G(m)_uu = sum m_t > 0; every G_xx lies in [G_uu/4, 4 G_uu] because |t_x| <= 2; and W_ee >= 2 because Ghat >= 0
entrywise. So C(m) is continuous on the simplex, and Phi_F is continuous by Lemma A.4 of Section 3.)

Example. One generator: G = g g^T. Then Ghat is the all-ones matrix, W has all entries 4, C is the all-ones matrix, and
f = 0. Mixing types is what produces a cost.

**Rows, keys and the realisation.** Fix m. In R^{T123} with the standard inner product, the *unnormalised rows* are
r_x = (sqrt(m_t) g_t(x))_t, x in {u, v, a, b, c, d}; their Gram matrix is G(m). Put â_x = r_x/|r_x|, so Ghat_xy = <â_x, â_y>
>= 0. For an internal edge e = xy put y_e = (â_x + â_y)/|â_x + â_y| and, with a standard Gaussian vector omega in
R^{T123}, Z_e = <y_e, omega>. The *edge cosines* are e_xy := Ghat_xy for the five internal edges.

**Lemma REAL.** (Z_ua, Z_ub, Z_vc, Z_vd) is a centred Gaussian vector with covariance C(m), Z_uv is jointly
Gaussian with it, every Z_e has variance 1, and f(m) = E F(s_0(Z_ua, Z_ub, Z_vc, Z_vd)).

```text
<1>1. Each y_e is a unit vector, so Z_e ~ N(0, 1) and Cov(Z_e, Z_f) = <y_e, y_f>.
<1>2. For e = xy, f = x'y': <â_x + â_y, â_x' + â_y'> = (B Ghat B^T)_ef = W_ef, so <y_e, y_f> = W_ef/sqrt(W_ee W_ff) = C(m)_ef.
      QED
```

**Lemma SYMC (symmetries of the cone).** The maps sigma_ab (swap t_a, t_b), sigma_cd (swap t_c, t_d) and sigma_uv
(relabel (u', v', a', b', c', d') := (v, u, c, d, a, b) and put t'_{x'} := t_x - t_v) are bijections of T123. Acting on mass
vectors by m'_{sigma_ab(t)} = m_t, m'_{sigma_cd(t)} = m_t and m'_{sigma_uv(t)} = 2^{-t_v} m_t, each preserves f. Under
sigma_ab, C(m') is C(m) with ac <-> bc and ad <-> bd; under sigma_uv, C(m') is C(m) with ab <-> cd and ad <-> bc (ac and
bd fixed), and G(m') = Pi G(m) Pi^T for the permutation Pi = (u v)(a c)(b d).

```text
<1>1. sigma_ab, sigma_cd: conditions (E), (Ru), (Rv) are symmetric in a, b and in c, d; special types go to special types.
      Each is an involution.
<1>2. sigma_uv: (E) is preserved (the internal edges of the new labelling are the old ones, and a common shift does not
      change differences). For generic t: min(t'_{v'}, t'_{a'}, t'_{b'}) = min(t_u, t_c, t_d) - t_v = -1 by the old (Rv);
      min(t'_{u'}, t'_{c'}, t'_{d'}) = min(t_v, t_a, t_b) - t_v = -1 - t_v = t'_{v'} - 1 by the old (Ru). Special types:
      (0, 1, 1, 1, t_c, t_d) -> (0, -1, t_c - 1, t_d - 1, 0, 0) and back. Applying sigma_uv twice is the identity.
<1>3. Under sigma_ab, G(m') = Pi G(m) Pi^T with Pi swapping a, b; the boundary keys ua, ub swap; F is symmetric in
      (s_1, s_2). So f(m') = f(m).
<1>4. Under sigma_uv, g_{t'}(x') = 2^{-(t_x - t_v)/2} = 2^{t_v/2} g_t(x), so m'_{t'} g_{t'} g_{t'}^T = m_t (Pi g_t)(Pi g_t)^T
      and G(m') = Pi G(m) Pi^T. The boundary edges u'a', u'b', v'c', v'd' are the old vc, vd, ua, ub; F is invariant under
      (s_1, s_2) <-> (s_3, s_4). So f(m') = f(m).   QED
```

### 2.8 Edge cosines, the star lemma, the rule bound, and Lemma L

**Lemma EC (edge cosines in ratio form).** For each internal edge xy, with r = G_yy/G_xx,

    e_xy >= (rho/2)(sqrt r + 1/sqrt r) >= rho = 2 sqrt 2/3.

In words: adjacent rows are always at least as correlated as on the tree, and the more their lengths differ, the more
correlated they must be.

```text
<1>1. For each type, |t_x - t_y| <= 1, so g_t(y)/g_t(x) lies in [q, Q] with q = 2^{-1/2}, Q = 2^{1/2}; qQ = 1, q + Q = 3/sqrt 2.
<1>2. For alpha, beta > 0 with beta/alpha in [q, Q]: (beta - q alpha)(Q alpha - beta) >= 0, i.e. beta^2 + alpha^2 <=
      (3/sqrt 2) alpha beta. With alpha = g_t(x), beta = g_t(y), weight m_t and sum: G_yy + G_xx <= (3/sqrt 2) G_xy.
<1>3. Divide by sqrt(G_xx G_yy): sqrt r + 1/sqrt r <= (3/sqrt 2) e_xy, i.e. e_xy >= (sqrt 2/3)(sqrt r + 1/sqrt r); sqrt 2/3 =
      rho/2. AM-GM gives sqrt r + 1/sqrt r >= 2.   QED
```

For x in [-1, 1]^3 put mu(e_1, e_2, e_3) := ((sum_i sqrt((1 + e_i)/2))^2 - 3)/6. It is continuous and increasing in each
argument, and mu(rho, rho, rho) = (9(1 + rho)/2 - 3)/6 = (1 + 3 rho)/4 = mu0 = (1 + 2 sqrt 2)/4.

The *key correlations* at u are k_a := <y_ua, y_uv>, k_b := <y_ub, y_uv> and k_ab := <y_ua, y_ub> = C_ab; at v, k_c, k_d and
k_cd = C_cd likewise. Their mean at w in {u, v} is kbar_w, and S_w is the sum of their three arccos values.

**Lemma STAR (the star lemma).** Let w be u or v, with edge cosines e_1, e_2, e_3 and key correlations k_1, k_2, k_3.
Then (a) each k_i lies in [rho, 1]; (b) kbar_w >= mu(e_1, e_2, e_3) >= mu0; (c) S_w <= 3 arccos(kbar_w); (d) S_w <= 3 arccos(mu0)
= 2 pi kappa_E.

In words: the three keys at a vertex all lean toward that vertex's row, and three vectors leaning toward a common direction
have a small total angle.

```text
<1>1. Let y_1, y_2, y_3 be the keys of the three edges at w and c_i := <y_i, â_w> = (1 + e_i)/sqrt(2 + 2 e_i) =
      sqrt((1 + e_i)/2) >= kappa_0 := sqrt((1 + rho)/2) (Lemma EC).
<1>2. (a) Write y_i = c_i â_w + w_i with w_i orthogonal to â_w, and psi_i = arccos c_i in [0, arccos kappa_0] (within
      [0, pi/2]). Then <y_i, y_j> = c_i c_j + <w_i, w_j> >= cos psi_i cos psi_j - sin psi_i sin psi_j = cos(psi_i + psi_j)
      >= cos(2 arccos kappa_0) = 2 kappa_0^2 - 1 = rho. And <y_i, y_j> <= 1.
<1>3. (b) |y_1 + y_2 + y_3| >= <sum y_i, â_w> = sum c_i >= 0, and |sum y_i|^2 = 3 + 6 kbar_w. So kbar_w >= ((sum c_i)^2 - 3)/6
      = mu(e_1, e_2, e_3) >= mu0 by monotonicity and Lemma EC.
<1>4. (c) arccos is concave on [0, 1] (second derivative -x(1 - x^2)^{-3/2} <= 0), and the k_i lie in [rho, 1]; Jensen.
<1>5. (d) arccos is decreasing and kbar_w >= mu0.   QED
```

**Lemma RB (the rule bound).** For every m, f(m) <= RB(m) := (S_u + S_v)/(2 pi) <= 2 kappa_E.

In words: the matched edge placed by the sign of its own key costs exactly the edge-score price of its endpoints; the
optimal side costs no more.

```text
<1>1. By Lemma MI, F(s) = min_x g(x; s) <= g(X; s) for every X. Take X = 1[Z_uv <= 0] and s = s_0(Z). Pointwise,
      F(s_0(Z)) <= 1[Z_uv, Z_ua, Z_ub not all of one sign] + 1[Z_uv, Z_vc, Z_vd not all of one sign].
<1>2. (Z_uv, Z_ua, Z_ub) is centred Gaussian with unit variances and correlations the key correlations at u (Lemma REAL), so
      by Lemma G(c) the first term has expectation S_u/(2 pi); likewise at v.
<1>3. Take expectations and use Lemma STAR(d).   QED
```

At the tree point (Section 4) every key correlation equals mu0, so RB(m_T) = 2 kappa_E exactly: the whole gain of matched
insertion, 2 kappa_E - M_tree = 0.0041183, comes from choosing the side optimally.

**Diagonal ratios.** The *diagonal vector* of m is d(m) = (d_v, d_a, d_b, d_c, d_d) with d_x = G_xx(m)/G_uu(m) (d_u = 1).
The *slice* of d is S_d = {m >= 0 : sum m = 1, G_xx(m) = d_x for x = v, a, b, c, d}. Since g_t(u) = 1, G_uu(m) = sum m, so
m/(sum m) lies in S_{d(m)}. On a slice, Ghat_xy = G_xy(m)/sqrt(d_x d_y) is LINEAR in m.

**Lemma RANGE.** For every m: d_v, d_a, d_b lie in [1/2, 2]; d_c/d_v and d_d/d_v lie in [1/2, 2].

```text
<1>1. (mediant) If b_t > 0 and a_t/b_t is in [L, U] for all t, and m >= 0, m != 0, then (sum m_t a_t)/(sum m_t b_t) is in [L, U].
<1>2. For x in {v, a, b}, |t_x| <= 1 by (E), so 2^{-t_x} is in [1/2, 2]; apply <1>1 with b_t = 1. For x in {c, d},
      2^{-t_x}/2^{-t_v} is in [1/2, 2]; apply <1>1 with b_t = 2^{-t_v}.   QED
```

**Lemma RC (the rule ceiling over a box of ratios).** For a box Q of diagonal vectors and an internal edge xy
put e_xy(Q) := min over d in Q of (rho/2)(sqrt(d_y/d_x) + sqrt(d_x/d_y)), and

    R(Q) := (3/(2 pi)) [ arccos min(1, mu(e_uv(Q), e_ua(Q), e_ub(Q))) + arccos min(1, mu(e_uv(Q), e_vc(Q), e_vd(Q))) ].

If d(m) lies in Q, then f(m) <= RB(m) <= R(Q).

```text
<1>1. By Lemma EC, e_xy >= (rho/2)(sqrt r + 1/sqrt r) with r = d_y/d_x, which is >= e_xy(Q).
<1>2. By Lemma STAR(b), monotonicity of mu, and kbar_u <= 1: kbar_u >= min(1, mu(e_uv(Q), e_ua(Q), e_ub(Q))).
<1>3. By Lemma STAR(c) and since arccos is decreasing, S_u <= 3 arccos min(1, mu(...)). Likewise at v. Add, divide by
      2 pi, and use Lemma RB.   QED
```

The function r -> (rho/2)(sqrt r + 1/sqrt r) is decreasing on (0, 1], increasing on [1, oo), and equals rho only at r = 1,
so R(Q) < 2 kappa_E exactly when some ratio is bounded away from 1 on all of Q.

**Lemma L (linear lower bounds on a slice).** Put D0 = 2 + 2 rho. For an entry p = (e, f) of C with boundary edges
e = xy, f = zw, put

    N_p = W_ef = Ghat_xz + Ghat_xw + Ghat_yz + Ghat_yw,     D_p = (W_ee + W_ff)/2 = 2 + Ghat_xy + Ghat_zw,
    l_p(m; d) = N_p/D0 - (4/D0^2)(D_p - D0),

with the Ghat entries written as on the slice (Ghat_xy = G_xy(m)/sqrt(d_x d_y), Ghat_xx = 1 = sum m). Then for every m with
diagonal vector d, C_p(m) >= l_p(m/sum m; d), and l_p(.; d) is linear on S_d.

In other words: once the five ratios are fixed, every correlation of the boundary keys is at least a linear function of the
masses.

```text
<1>1. C is invariant under scaling; assume sum m = 1, so m is in S_d and N_p, D_p are linear in m.
<1>2. Every entry of Ghat lies in [0, 1] (generators are positive), so 0 <= N_p <= 4.
<1>3. W_ee = 2 + 2 e_xy >= D0 for every boundary edge (Lemma EC), so D_p >= D0 > 0.
<1>4. sqrt(W_ee W_ff) <= D_p (AM-GM) and N_p >= 0, so C_p >= N_p/D_p.
<1>5. 1/D >= 2/D0 - D/D0^2 for D > 0 (the difference is (D - D0)^2/(D D0^2)); multiply by N_p >= 0:
      N_p/D_p >= N_p/D0 - N_p (D_p - D0)/D0^2 >= N_p/D0 - 4 (D_p - D0)/D0^2 by <1>2 and <1>3.   QED
```

At the tree every edge cosine is rho, so D_p = D0 and Lemma L is exact there. The same argument applies to the key
correlations k_a, k_b at a star (the pair (uv, ua) in place of a boundary pair); this "Lemma L for stars" is used in
Sections 8 and 9.

### 2.9 Kernels of large girth

**Fact GIRTH.** (Erdos and Sachs (1963); or a random cubic graph conditioned on having no
short cycle.) For every g there are 3-connected cubic graphs of girth at least g on every large even number of vertices.
For g >= 4 they are kernels.

This fact is not used in the proof of Theorem 1; it shows that the constant 1 + 1/M_tree is the exact limit of the method. On such a
kernel every ball of radius less than (g - 1)/2 is a tree, so every matched edge sees the tree point.

---

## 3. Theorem A (matched insertion) and Corollary A1

This section proves that every kernel has cutwidth at most (M_123/2 + o(1)) h, and hence liminf s_n/n >= 1 + 1/M_123. The
statement can be made more general: it allows any matching M with mu = |M|/h and gives bags at most
((1 - 2 mu) kappa_E + mu M_123 + xi) h, the unmatched vertices being paid by edge scores. Kernels have perfect matchings,
and for a perfect matching there are no unmatched vertices, so we state and prove only that case. This removes the
edge-score theorem from the proof. (The general case is proved the same way with one more term.)

The plan has three parts. Lemmas A.2-A.4 turn the Gram matrix of a real neighbourhood into a point of the cone, with an
error that vanishes as the field radius R grows. Lemma A.6 builds the order and bounds its bags by local costs,
deterministically. Lemma A.7 bounds the expected local costs, and the proof of Theorem A passes from expectations to one
good noise vector.

### 3.1 The cone with a shell

Fix a simple cubic graph H, the field of Section 2.6 with radius R >= 3, and an edge uv labelled as in Section 2.6. Let G
be the 6 x 6 Gram matrix G_xy = <a_x, a_y>, x, y in (u, v, a, b, c, d).

**Lemma A.2 (the cone with a shell).** G = G_in + G_sh, where

    G_in = sum_{t in T123} m_t g_t g_t^T,   m_t = sum { 2^{-d(u,z)} : t(z) = t, d(u,z) <= R - 2 },   with m_{t(u)} >= 1,

and G_sh is PSD with tr G_sh <= (60/(R - 2)) tr G_in.

In words: up to a shell of relative size 60/(R - 2), the Gram matrix of a real neighbourhood is a point of the cone.

PROOF IDEA. Deep sources z (d(u, z) <= R - 2) are inside every truncation ball, so each contributes a generator times
2^{-d(u,z)}. Shallow sources live on the last four spheres; sphere masses do not increase with distance, so the shell is
at most a constant times one sphere mass, while the deep part has at least R - 2 of them.

```text
<1>1. G = sum_z r_z r_z^T with r_z = (a_x(z))_{x = u, v, a, b, c, d}.
<1>2. If d(u,z) <= R - 2, then d(x,z) <= 2 + (R - 2) = R for each of the six x, so a_x(z) = 2^{-d(u,z)/2} 2^{-t_x(z)/2}, i.e.
      r_z = 2^{-d(u,z)/2} g_{t(z)} with t(z) in T123 (Lemma TYPES). Summing gives G_in; z = u contributes 1 to m_{t(u)}.
<1>3. G_sh := sum over d(u,z) >= R - 1 of r_z r_z^T is PSD. If d(u,z) >= R + 3, all six d(x,z) >= R + 1 and r_z = 0. So only
      R - 1 <= d(u,z) <= R + 2 contributes.
<1>4. Let N_k = #{z : d(u,z) = k} and s_k = N_k 2^{-k}. For k >= 1 every vertex at distance k + 1 has a neighbour at distance
      k, and every vertex at distance k has a neighbour at distance k - 1, hence at most 2 at distance k + 1. So
      N_{k+1} <= 2 N_k and s_{k+1} <= s_k for k >= 1.
<1>5. For z at distance k: d(x,z) >= k - d(x,u) with d(x,u) = 0, 1, 1, 1, <= 2, <= 2, so |r_z|^2 <= 2^{-k}(1 + 3*2 + 2*4) =
      15 * 2^{-k}. Hence tr G_sh <= 15 sum_{k=R-1}^{R+2} s_k <= 60 s_{R-2}.
<1>6. tr G_in >= sum_{d(u,z) <= R-2} a_u(z)^2 = sum_{k=0}^{R-2} s_k >= (R - 2) s_{R-2}. With <1>5 the claim follows.   QED
```

**Lemma A.3 (the shell moves C by O(1/R)).** Let G_in = G(m) be a nonzero cone point and S a PSD 6 x 6 matrix with
tr S <= eps tr G_in, where delta := 96 eps <= 1/12. Then G = G_in + S has positive diagonal, W(G) has positive diagonal, and
|C(G)_ef - C(G_in)_ef| <= 24 delta for all e, f.

PROOF IDEA. Every diagonal entry of G_in is at least 1/96 of the trace, so the shell is a relative perturbation of size
delta of every entry, and the two normalisations amplify it by bounded factors.

```text
<1>1. g_t(x)^2 = 2^{-t_x} is in [1/4, 4] and g_t(u) = 1, so G_in,uu/4 <= G_in,xx <= 4 G_in,uu, tr G_in <= 24 G_in,uu and
      G_in,xx >= tr G_in/96 > 0.
<1>2. |S_xy| <= sqrt(S_xx S_yy) <= tr S <= delta min(G_in,xx, G_in,yy), and 0 <= S_xx <= delta G_in,xx.
<1>3. Ghat(G)_xy = (Ghat(G_in)_xy + e1) phi with |e1| <= delta and phi = ((1 + e2)(1 + e3))^{-1/2}, e2, e3 in [0, delta]; so
      phi is in [1/(1 + delta), 1] and |Ghat(G)_xy - Ghat(G_in)_xy| <= 2 delta.
<1>4. W(G_in)_ee = 2 + 2 Ghat(G_in)_xy >= 2 (Ghat(G_in) >= 0 entrywise). Each entry of W is a sum of 4 entries of Ghat whose
      diagonal entries are exactly 1; so |W(G)_ef - W(G_in)_ef| <= 8 delta <= 12 delta <= 6 delta sqrt(W_ee W_ff).
<1>5. C(G)_ef = (C(G_in)_ef + e1') phi' with |e1'|, |e2'|, |e3'| <= 6 delta <= 1/2, so phi' lies in [1 - 6 delta, 1 + 12 delta]
      and |C(G)_ef - C(G_in)_ef| <= 6 delta (1 + 12 delta) + 12 delta <= 24 delta (as 72 delta^2 <= 6 delta).   QED
```

**Lemma A.4 (Phi_F is uniformly continuous).** Let CC be the set of 4 x 4 correlation matrices. For every
eta > 0 there is d(eta) > 0 with |Phi_F(C) - Phi_F(C')| <= eta whenever C, C' are in CC and max |C_ef - C'_ef| <= d(eta).

```text
<1>1. Phi_F is continuous on CC: if C_n -> C, the characteristic functions exp(-xi^T C_n xi/2) converge pointwise, so
      X_n -> X in distribution (Levy). x -> F(s_0(x)) is bounded and continuous off the hyperplanes {x_i = 0}, which have
      probability 0 under N(0, C) (unit variances). By the portmanteau theorem, Phi_F(C_n) -> Phi_F(C).
<1>2. CC is compact; a continuous function on a compact set is uniformly continuous.   QED
```

For R >= 3 put eta_R := sup { |Phi_F(C) - Phi_F(C')| : C, C' in CC, max |C_ef - C'_ef| <= 24 * 96 * 60/(R - 2) }. By
Lemma A.4, eta_R -> 0 as R -> oo. No explicit modulus is needed: R depends on the target slack alone.

### 3.2 The mixed order

Fix H, a perfect matching M, and real scores k_e for all edges (ties allowed). The *entry time* of an edge is tau_e = k_e if
e is not in M, and for uv in M

    tau_uv = inf { theta : x*(s_uv(theta)) = 1 }.

List the edges as e_1, ..., e_{3h/2} by increasing (tau_e, a fixed index of e). For a vertex w, first(w) and last(w) are the
least and largest positions of its three edges, and the *bags* are X_i = {w : first(w) <= i <= last(w)}. The *cost* at
theta is Cost(theta) = #{w : w has an edge with tau < theta and an edge with tau >= theta}. For a grid
theta_0 < ... < theta_J the *window counts* are W_j = #{e not in M : theta_j <= k_e < theta_{j+1}}, and the *tail counts* are
W_- = #{e not in M : k_e < theta_0} and W_+ = #{e not in M : k_e >= theta_J}.

**Lemma A.6 (the mixed order and its bags; deterministic).**
(a) For uv in M, tau_uv is one of its four boundary scores, and 1[tau_uv < theta] = x*(s_uv(theta)) for every theta.
(b) For every theta, Cost(theta) = sum_{uv in M} F(s_uv(theta)).
(c) The bags X_i form a path decomposition of H.
(d) For every grid, every bag satisfies |X_i| <= 6 W_-, or |X_i| <= 6 W_+, or |X_i| <= Cost(theta_j) + 6 W_j for some j.
(e) Ordering the vertices by the position of their median edge (ties by a fixed numbering) gives every cut at most
    max_i |X_i| + 1 edges.

PROOF IDEA. As theta rises, each boundary side flips once from 0 to 1, so n11 - n00 only grows, and the optimal side flips
once. So the entry time is one of the boundary scores. The bag at a position is bounded by the cost at the grid point
below, plus the endpoints of edges that enter inside the window; a window holds at most three times as many entering edges
as unmatched scores, since an unmatched edge is a boundary edge of at most two matched edges.

```text
<1>1. (a) Each coordinate of s_uv(theta) is nondecreasing in theta and changes only at a boundary score, so n11 - n00 and
      x*(s_uv(theta)) are nondecreasing step functions of theta, constant between consecutive boundary scores (with the
      half-open convention 1[k < theta]). For theta <= the least boundary score, s = 0000 and x* = 0; above the largest,
      s = 1111 and x* = 1. So {theta : x* = 1} = (k_(l), oo) for one boundary score k_(l), and tau_uv = k_(l).
<1>2. (b) A vertex is counted iff its three edges are not all on one side. Every vertex is covered by M. For uv in M, the
      side of uv is 1[tau_uv < theta] = x*(s) by (a), and the boundary edges (unmatched) have sides s = s_uv(theta); so the
      number of counted vertices among u, v is g(x*(s); s) = F(s) (Lemma MI).
<1>3. (c) Every vertex lies in X_{first(w)}; an edge e_i = xy has both endpoints in X_i; the bags containing w are the
      consecutive indices first(w), ..., last(w).
<1>4. (d) Fix i and w in X_i, and t_i = tau_{e_i}. Then w has an edge with tau <= t_i and one with tau >= t_i.
  <2>1. If tau_f lies in a set I of reals, then f is unmatched with k_f in I, or f is matched and tau_f = k_e in I for an
        unmatched boundary edge e of f. An unmatched edge xy is a boundary edge of at most two matched edges. So
        #{f : tau_f in I} <= 3 #{e unmatched : k_e in I}, and these edges have at most 6 #{...} endpoints.
  <2>2. t_i < theta_0: w is an endpoint of an edge with tau < theta_0; |X_i| <= 6 W_-. t_i >= theta_J: |X_i| <= 6 W_+.
  <2>3. theta_j <= t_i < theta_{j+1}: if w has an edge with tau < theta_j it is counted in Cost(theta_j); otherwise its edge
        with tau <= t_i has tau in [theta_j, theta_{j+1}), and at most 6 W_j vertices are of this kind.
<1>5. (e) Fix a nonempty proper prefix P, its last vertex x_0 and theta = med(x_0). The three edges at a vertex have distinct
      positions (H is loopless), one below and one above the median. A crossing edge e = xy (x in P) with position < theta
      is the lowest edge of y and y is in X_theta (first(y) < theta <= med(y) < last(y)); charge it to y. With position
      > theta it is the highest edge of x and x is in X_theta; charge it to x. At most one edge has position theta. No
      vertex is charged twice. So the cut has at most |X_theta| + 1 edges.   QED
```

### 3.3 The expected cost at every threshold

**Lemma A.7.** Let R >= 2 + 12 * 96 * 60 = 69122. For every simple cubic graph H, every perfect matching M and
every real theta, E Cost(theta) <= (h/2) (M_123 + eta_R).

PROOF IDEA. The cost of a matched edge is a symmetric submodular sign functional of four Gaussian scores, so threshold 0 is
the worst (Lemma G(a)); at threshold 0 it is Phi_F of the key covariance, which is C(G) for the real Gram matrix G; and G is
a cone point plus a shell (Lemma A.2), which moves C by little (Lemma A.3) and Phi_F by at most eta_R (Lemma A.4).

```text
<1>1. For uv in M, Z := (k_ua, k_ub, k_vc, k_vd) is centred Gaussian with covariance C(G), G the Gram matrix of Lemma A.2:
      Cov(k_e, k_f) = <y_e, y_f>, and <â_x + â_y, â_x' + â_y'> = (B Ghat B^T)_ef = W_ef; normalise.
<1>2. s_uv(theta) and s_theta(Z) differ only on a null event, so E F(s_uv(theta)) = Psi_theta(F; C(G)) <= Phi_F(C(G)) by
      Lemma G(a) (F symmetric and submodular, Lemma MI).
<1>3. By Lemma A.2, G = G(m) + G_sh with tr G_sh <= eps tr G(m), eps = 60/(R - 2), and delta = 96 eps <= 1/12. By Lemma A.3,
      max |C(G)_ef - C(m)_ef| <= 24 * 96 * 60/(R - 2), so Phi_F(C(G)) <= f(m) + eta_R <= M_123 + eta_R.
<1>4. By Lemma A.6 (b), E Cost(theta) = sum_{uv in M} E F(s_uv(theta)), a sum of h/2 terms.   QED
```

The shell is absorbed here, inside eta_R. So the quantity that must be bounded is the plain cone supremum M_123.

### 3.4 Theorem A and Corollary A1

**Theorem A (matched insertion, perfect matchings).** For every xi in (0, 1] there is h_0(xi) such that every
simple cubic graph H on h >= h_0(xi) vertices with a perfect matching has a path decomposition with every bag of size at
most (M_123/2 + xi) h, and a vertex ordering with every cut of size at most (M_123/2 + xi) h + 1.

PROOF IDEA. The bags are bounded, deterministically, by a cost at a grid point plus six window counts (Lemma A.6 (d)).
Each of these finitely many counts is a sum of local terms, so it concentrates (Chebyshev); its mean is small (Lemma A.7
for costs, the Gaussian density for windows, a tail cut for the extremes). The grid is fixed before the graph, so the union
bound is over a number of events that does not grow with h.

```text
<1>1. (constants, depending on xi only)
  <2>1. Choose an integer R >= 69122 with eta_R <= xi/8 (Lemma A.4).
  <2>2. Delta = xi/100; choose T >= 1 with (3/2)(1 - Phi(T)) <= xi/100 (Phi the standard normal distribution function).
  <2>3. J = ceil(2T/Delta), theta_j = -T + j Delta (j = 0, ..., J). There are L := 2J + 3 counts: Cost(theta_j), W_j, W_-, W_+.
  <2>4. K = 3 * 2^{2R+6}, lambda(h) = xi h/40, h_0 = ceil(1600 L K/xi^2) + 1.
<1>2. Fix H with h >= h_0 and a perfect matching M; build the field with this R; draw omega.
<1>3. (means) E Cost(theta_j) <= (h/2)(M_123 + xi/8) <= (M_123/2) h + xi h/8 (Lemma A.7). E W_j <= (3h/2) Delta/sqrt(2 pi)
      <= 0.006 xi h (each score is N(0, 1)). E W_-, E W_+ <= (3h/2)(1 - Phi(T)) <= xi h/100.
<1>4. (locality) The score of an edge xy depends only on omega on B(x, R + 1). The term F(s_uv(theta)) of uv depends only on
      omega on B(u, R + 2) (the rows of u, v, a, b, c, d, all within distance 2 of u).
<1>5. (variances) Each count X has Var X <= K h. Cost(theta) is a sum of h/2 terms with values in [0, 2] (variance <= 1),
      centred at the first endpoint of each matched edge; two terms are independent when their centres are at distance
      > 2R + 4; a ball of radius 2R + 4 in a cubic graph has fewer than 3 * 2^{2R+4} vertices. So Var Cost(theta) <=
      h * 3 * 2^{2R+4} <= K h. For W_j assign each unmatched edge to one endpoint: W_j = sum_x Y_x, Y_x in [0, 3] (variance
      <= 9/4), dependency radius 2R + 2; Var W_j <= h * 3 * 2^{2R+2} * 9/4 <= K h. Likewise W_-, W_+.
<1>6. (one good omega) By Chebyshev each count deviates from its mean by >= lambda(h) with probability <= K h/lambda(h)^2 =
      1600 K/(xi^2 h); by the union bound over L counts the probability that some count deviates is < 1 for h >= h_0. Fix an
      omega with no deviation.
<1>7. (bags) By Lemma A.6 (d) each bag satisfies |X_i| <= 6 W_- < 6 (xi/100 + xi/40) h = 0.21 xi h (or the same with W_+),
      or |X_i| <= Cost(theta_j) + 6 W_j < (M_123/2) h + (xi/8 + xi/40) h + 6 (0.006 + 0.025) xi h = (M_123/2 + 0.336 xi) h.
      In all cases |X_i| <= (M_123/2 + xi) h.
<1>8. Lemma A.6 (c) and (e) finish.   QED
```

The quantifiers are in order: xi, then R, Delta, T, J, L, K, h_0, then H, M, omega.

**Fact PETERSEN.** (Petersen's theorem; Diestel, Graph Theory, Corollary 2.2.2: "Every bridgeless cubic graph has a
1-factor.") A 1-factor is a perfect matching.

**Corollary A1.** (i) For every xi in (0, 1] there is C_xi such that every kernel H on h vertices has
cw(H) <= (M_123/2 + xi) h + C_xi. (ii) M_123 > 0. (iii) Every family of B2 circuits computing Li's function satisfies

    liminf_{n -> oo} s_n/n >= 1 + 1/M_123.

In words: on kernels every vertex can be matched, so the kernel constant is half the matched-edge price.

```text
<1>1. A kernel H is bridgeless. Suppose uv is a bridge, with components A (containing u) and B of H - uv. u has two
      neighbours in A (H is simple and cubic), so A \ {u} is nonempty; likewise for v. Every path from A \ {u} to B uses uv,
      hence u, so deleting u disconnects H: H is not 2-connected.
<1>2. By Fact PETERSEN, H has a perfect matching.
<1>3. (i) For h >= h_0(xi), Theorem A gives cw(H) <= (M_123/2 + xi) h + 1. For h < h_0 every cut has at most 3h/2 < 3 h_0/2
      edges. Take C_xi = 3 h_0(xi)/2 + 1.
<1>4. (ii) Take m with mass 1 on each of the types A = (0, 1, -1, 1, 2, 2) and B = (0, 1, 1, -1, 2, 2). In R^{A, B} the rows
      are r_u = (1, 1), r_a = (2^{1/2}, 2^{-1/2}), r_b = (2^{-1/2}, 2^{1/2}). â_u + â_a has unequal coordinates and â_u + â_b
      is its mirror image, so y_ua != y_ub and C(m)_ab < 1. Since F(s) >= 1[s_1 != s_2], Lemma G(c) gives
      f(m) >= arccos(C(m)_ab)/pi > 0.
<1>5. (iii) For every eta > 0, (i) with xi = min(eta, 1) is the hypothesis of Theorem K2C with kappa = M_123/2 > 0, and
      Theorem K2C (b) gives liminf s_n/n >= 1 + 1/M_123.   QED
```

So a certified bound M_123 <= x with x > 0 gives the circuit constant 1 + 1/x. The rest of this document proves
M_123 = M_tree and certifies M_tree <= 0.27658360966404055.

*Remark (the edge-score bound is not needed).* Corollary A1 can also be stated as liminf s_n/n >= 1 + 1/min(2 kappa_E, M_123),
using the edge-score bound cw(H) <= (kappa_E + xi) h + C_xi as a second option. Since M_tree < 2 kappa_E (Section 10), the
minimum is M_123 and the edge-score bound plays no role in Theorem 1. The form above avoids it.

---

## 4. The tree point and M_tree

### 4.1 The tree point

The four types that a source vertex can have when the graph is the 3-regular tree are

    A = (0, 1, -1, 1, 2, 2)    (a source beyond a),       B = (0, 1, 1, -1, 2, 2)    (beyond b),
    C = (0, -1, 1, 1, -2, 0)   (beyond c),                D = (0, -1, 1, 1, 0, -2)   (beyond d).

(Type A is one step closer to a than to u, one step farther from v and b, two steps farther from c and d.) The *tree point*
m_T puts mass 1/3 on A and on B and 1/6 on C and on D. The *tree face* is the set of cone points supported on {A, B, C, D};
the *off-face mass* of m is its mass on the other 119 types.

**Lemma TREE.** At m_T all six diagonal entries of G equal 1; Ghat equals rho = 2 sqrt 2/3 on
the five internal edges, 5/6 on the six pairs at distance 2, and 1/sqrt 2 on the four pairs at distance 3; and C_T := C(m_T)
has

    pair entries  p = (1 + 2 sqrt 2)/4 = mu0 = 0.9571068,       cross entries  s = (2 + sqrt 2)/4 = 0.8535534.

Every key correlation at u and at v equals mu0, so S_u = S_v = 3 arccos(mu0) and RB(m_T) = 2 kappa_E.

```text
<1>1. The generators: g_A = (1, 2^{-1/2}, 2^{1/2}, 2^{-1/2}, 1/2, 1/2), g_B = (1, 2^{-1/2}, 2^{-1/2}, 2^{1/2}, 1/2, 1/2),
      g_C = (1, 2^{1/2}, 2^{-1/2}, 2^{-1/2}, 2, 1), g_D = (1, 2^{1/2}, 2^{-1/2}, 2^{-1/2}, 1, 2).
<1>2. Diagonals: G_vv = (2/3)(1/2) + (1/3)(2) = 1; G_aa = (1/3)(2) + (1/3)(1/2) + (1/3)(1/2) = 1; G_cc = (2/3)(1/4) + (1/6)(4)
      + (1/6)(1) = 1; the others by the a <-> b and c <-> d symmetry.
<1>3. Edges: G_uv = (2/3) 2^{-1/2} + (1/3) 2^{1/2} = 2 sqrt 2/3; G_ua = (1/3) 2^{1/2} + (2/3) 2^{-1/2} = 2 sqrt 2/3;
      G_vc = (2/3)(2^{-1/2}/2) + (1/6)(2^{1/2} 2) + (1/6) 2^{1/2} = 2 sqrt 2/3.
<1>4. Distance 2: G_ab = (2/3)(1) + (1/3)(1/2) = 5/6; G_va = (1/3)(1) + (1/3)(1/2) + (1/3)(1) = 5/6; G_uc = (2/3)(1/2) +
      (1/6)(2) + (1/6)(1) = 5/6; G_cd = (2/3)(1/4) + (1/3)(2) = 5/6.
<1>5. Distance 3: G_ac = (1/3)(2^{1/2}/2) + (1/3)(2^{-1/2}/2) + (1/6)(2^{-1/2} 2) + (1/6) 2^{-1/2}
      = (1/3 + 1/6 + 1/3 + 1/6)/sqrt 2 = 1/sqrt 2.
<1>6. W_ee = 2 + 2 rho for each boundary edge. Cross: W_ac = Ghat_uv + Ghat_uc + Ghat_av + Ghat_ac = rho + 5/3 + 1/sqrt 2;
      times 6, (10 + 7 sqrt 2)/(12 + 8 sqrt 2), and (2 + sqrt 2)(12 + 8 sqrt 2) = 40 + 28 sqrt 2 = 4 (10 + 7 sqrt 2); so
      C_ac = (2 + sqrt 2)/4. Pair: W_ab = 1 + 2 rho + 5/6, and (1 + 2 sqrt 2)(12 + 8 sqrt 2) = 44 + 32 sqrt 2 = 4 (11 + 8 sqrt 2),
      so C_ab = (1 + 2 sqrt 2)/4. The key correlation k_a = (Ghat_uu + Ghat_uv + Ghat_ua + Ghat_va)/(2 + 2 rho) has the same
      numerator 1 + 2 rho + 5/6, so equals mu0.   QED
```

### 4.2 M_tree in closed form

**Definition.** M_tree := f(m_T) = Phi_F(C_T).

**Theorem W (the Walsh form of Phi_F).** For every 4 x 4 correlation matrix C, with sigma_ij := (2/pi) arcsin C_ij
and Q(C) := Pr[X_a > 0, X_b > 0, X_c > 0, X_d > 0], X ~ N(0, C),

    Phi_F(C) = 1 - (sigma_ab + sigma_cd)/2 - (sigma_ac + sigma_ad + sigma_bc + sigma_bd)/4 + 2 Q(C).

In words: the cost of a matched edge is an explicit function of the six pairwise correlations plus twice the probability
that all four boundary keys are positive.

```text
<1>1. Put s_i = 1 - 2 1[X_i <= 0] in {-1, +1}; a.s. s_i = sgn X_i.
<1>2. 1[a != b] = (1 - s_a s_b)/2 and 1[a = b != c = d] = (1 + s_a s_b)(1 + s_c s_d)(1 - s_a s_c)/8 (the product is 1 exactly
      when s_a = s_b, s_c = s_d, s_a != s_c). Expanding,
      F = 9/8 - (3/8)(s_a s_b + s_c s_d) - (1/8)(s_a s_c + s_a s_d + s_b s_c + s_b s_d) + (1/8) s_a s_b s_c s_d.
<1>3. E s_i s_j = 1 - 2 arccos(C_ij)/pi = sigma_ij (Lemma G(c)); E s_i = E s_i s_j s_k = 0 (X -> -X).
<1>4. Q(C) = E prod_i (1 + s_i)/2 = (1 + sum_{i<j} sigma_ij + E s_a s_b s_c s_d)/16, so E s_a s_b s_c s_d = 16 Q - 1 - sum sigma.
<1>5. Substitute: constant 9/8 - 1/8 = 1; pair coefficient -3/8 - 1/8 = -1/2; cross coefficient -1/8 - 1/8 = -1/4; Q has 2.
      QED
```

Checks: at C = I, Phi_F = 9/8; at the all-ones matrix, Phi_F = 1 - 1 - 1 + 1 = 0.

**Corollary T (the tree value as a one-dimensional integral).** (p - s)/(1 - s) = 1/sqrt 2 and s/(1 - s) =
(1 + sqrt 2)^2 exactly, and

    M_tree = 1 - (2/pi)(arcsin p + arcsin s) + 2 Int phi(z) Phi_2((1 + sqrt 2) z, (1 + sqrt 2) z; 1/sqrt 2)^2 dz,

where phi is the standard normal density and Phi_2(x, y; r) the bivariate normal distribution function with correlation r.

```text
<1>1. Let Z, U_1, U_2, e_a, e_b, e_c, e_d be independent N(0, 1) and X_i = sqrt(s) Z + sqrt(p - s) U_{g(i)} + sqrt(1 - p) e_i
      with g(a) = g(b) = 1, g(c) = g(d) = 2 (0 < s < p < 1). Then Var X_i = 1, Cov(X_a, X_b) = p, Cov(X_a, X_c) = s: X ~ N(0, C_T).
<1>2. Given Z = z the pairs (X_a, X_b), (X_c, X_d) are independent bivariate normal with means sqrt(s) z, variances 1 - s,
      correlation (p - s)/(1 - s) = ((sqrt 2 - 1)/4)/((2 - sqrt 2)/4) = 1/sqrt 2. And s/(1 - s) = (2 + sqrt 2)/(2 - sqrt 2) =
      3 + 2 sqrt 2 = (1 + sqrt 2)^2.
<1>3. Pr[X_a > 0, X_b > 0 | Z = z] = Phi_2(x, x; 1/sqrt 2) with x = (1 + sqrt 2) z (standardise; W -> -W). So
      Q(C_T) = E Phi_2((1 + sqrt 2) Z, (1 + sqrt 2) Z; 1/sqrt 2)^2. Theorem W with sigma_ab = sigma_cd = (2/pi) arcsin p and
      four cross sigmas (2/pi) arcsin s finishes.   QED
```

### 4.3 The certified enclosure of M_tree

The theorem uses only an interval for M_tree. Both ends come from monotonicity (Lemma G(b)) and an enclosure of
Phi_F at one matrix of doubles.

**Finite claim FC-MT (the enclosure of M_tree).** (computer-checked)

    Mt_lo := 0.27658360687864314  <=  M_tree  <=  Mt_hi := 0.27658360966404055.

- *Lower end*. Let C_up be the 4 x 4 matrix of doubles obtained by rounding each entry of an
  outward enclosure of C_T upward. Then C_up >= C_T entrywise, and C_up is positive definite (it is within a few ulps of
  C_T, whose least eigenvalue is 1 - p = 0.0429; Weyl). The code `phi_point` (an interval Taylor enclosure of a
  Plackett path integral in the manner of Lemma PATH, along a path that scales all six off-diagonal entries) returns an
  enclosure of Phi_F(C_up) whose lower end is Mt_lo.
  By Lemma G(b), M_tree = Phi_F(C_T) >= Phi_F(C_up) >= Mt_lo.
- *Upper end*. Let C_dn be C_T rounded down entrywise to doubles; it is certified positive
  definite. The code `F_point` (the same kind of enclosure along the path of Lemma PATH, which scales only the cross
  entries) returns an enclosure of Phi_F(C_dn) whose upper end is Mt_hi. By Lemma G(b), M_tree <= Phi_F(C_dn) <= Mt_hi.
- *Rounding convention*: every arithmetic operation is round-to-nearest followed by one ulp outward; arcsin enclosures are
  verified against mpmath's interval sine.

*Strictness.* M_tree > Mt_lo. Indeed C_T has irrational entries, so C_up != C_T, and every entry of C_up is at least the
entry of C_T. Both are positive definite, so the segment between them consists of positive definite matrices, along which
Phi_F is nonincreasing in every entry and strictly decreasing in each cross entry (2.4). Since the cross entries of C_up
strictly exceed s (s = (2 + sqrt 2)/4 is irrational), Phi_F(C_T) > Phi_F(C_up) >= Mt_lo. The equality clause of Theorem R1 (Section 6) uses this strict inequality.

*Which digits are certified.* Mt_hi - Mt_lo = 2.79e-9. So the certified statement is 1 + 1/M_tree >= 1 + 1/Mt_hi =
4.6155432392204... . The 30-digit value 4.6155432755827926... of 1 + 1/M_tree is not certified. The 11-digit truncation
"M_tree <= 0.27658360966" is 4.04e-12 below Mt_hi, so as an upper bound it is rounded the
wrong way. Section 10.3 gives the arithmetic.

### 4.4 The gradient at the tree and the tree cut

By the symmetries of C_T (the swaps a <-> b, c <-> d and the pair swap, which fix C_T and F), the gradient of Phi_F at C_T
has equal pair entries and equal cross entries:

    g_T := grad Phi_F(C_T) = -c_T (a_ab, a_ab, 1, 1, 1, 1),       c_T = 0.1456119555 (numerical).

**Finite claim FC-AAB (the gradient ratio).** (computer-checked) The ratio a_ab of a pair entry to a cross
entry of grad Phi_F(C_T) lies in [4.478385650326169, 4.478385650327289]. It is computed by the interval gradient
code `grad_ad` (formulas (2.4), with the partial correlations enclosed) on the one-ulp box around C_T
(bk8_core.consts, which also produces the enclosure of K0 below).

Put

    a := (a_ab, a_ab, 1, 1, 1, 1),      K0 := a.C_T = 2 a_ab p + 4 s = 11.9868001118...

and call the half-space {a.C >= K0} the *tree cut*. Every certificate below carries a_ab and K0 as the intervals of FC-AAB,
and every bound is taken at the unfavourable end, so the statements hold for the exact a_ab.

Why this normal and no other: on the set RC of Section 5 the tangent space at C_T is the orthogonal complement of the saddle
direction v (Lemma R1b), g_T is orthogonal to v, and any half-space {a'.C >= a'.C_T} that excludes every first-order ascent
direction of Phi_F within RC must have a' proportional to -g_T. So the ratio a_ab is forced, and the cut is tight to first
order at C_T.

---

## 5. The saddle, the kill, and the reduction to two obligations

Here is the idea of the whole analytic route. The tree cut a.C >= K0 holds on the cone (we shall prove what we need of it),
and a is the gradient direction of Phi_F at C_T. If Phi_F were concave near C_T, the cut alone would give Phi_F <= M_tree. It
is not: Phi_F has one positive Hessian direction v at C_T. We identify v as the gradient of the cross
determinant D, restrict to the set where D = 0, and lower every cone matrix into that set entry by entry. More correlation
never hurts, so lowering can only raise Phi_F; what must be shown is that the lowered matrix still satisfies the cut.

### 5.1 The saddle is the cross determinant

**Definition.** For a 4 x 4 matrix C put D(C) := C_ac C_bd - C_ad C_bc, the determinant of the *cross block*
K = [[C_ac, C_ad], [C_bc, C_bd]]. Let RC be the set of 4 x 4 correlation matrices with D = 0 (cross block of rank at most
one). C_T lies in RC, since its four cross entries are equal.

**Lemma R1b.**
(i) grad D(C_T) = 2 s v, where v := (ac + bd - ad - bc)/2 = (0, 0, 1, -1, -1, 1)/2 is a unit vector.
(ii) For every m, D(C(m)) = D(W(m))/sqrt(W_aa W_bb W_cc W_dd); so D(C(m)) = 0 iff D(W(m)) = 0.
(iii) Write the cross block of W as W_xz = X_x + Y_z + H_xz (x in {a, b}, z in {c, d}) with X_x := Ghat_uv + Ghat_xv,
      Y_z := Ghat_uz, H_xz := Ghat_xz. Then

      D(W) = (X_a - X_b)(Y_d - Y_c) + (X_a + Y_c) H_bd + (X_b + Y_d) H_ac - (X_a + Y_d) H_bc - (X_b + Y_c) H_ad
             + H_ac H_bd - H_ad H_bc.

(iv) On the tree face, D(C(m)) = 0 whenever m_A = m_B or m_C = m_D. In particular D(C(m)) has zero derivative at m_T along
     each of e_A, e_B, e_C, e_D.

```text
<1>1. (i) dD/dC_ac = C_bd, dD/dC_bd = C_ac, dD/dC_ad = -C_bc, dD/dC_bc = -C_ad, and 0 for ab, cd. At C_T these are s, s, -s,
      -s, so grad D = s (0, 0, 1, -1, -1, 1) = 2 s v.
<1>2. (ii) C_xz = W_xz/sqrt(W_xx W_zz); both products C_ac C_bd and C_ad C_bc have the denominator sqrt(W_aa W_bb W_cc W_dd).
<1>3. (iii) For boundary edges ux and vz, W_xz = Ghat_uv + Ghat_uz + Ghat_xv + Ghat_xz. Expand
      (X_a + Y_c + H_ac)(X_b + Y_d + H_bd) - (X_a + Y_d + H_ad)(X_b + Y_c + H_bc). The H-free part is
      (X_a + Y_c)(X_b + Y_d) - (X_a + Y_d)(X_b + Y_c) = (X_a - X_b)(Y_d - Y_c).
<1>4. (iv) If m_A = m_B (with m on the tree face, or in general if m is fixed by sigma_ab), sigma_ab fixes m and exchanges
      C_ac <-> C_bc and C_ad <-> C_bd (Lemma SYMC), so D -> C_bc C_ad - C_bd C_ac = -D; hence D = 0. Likewise with sigma_cd when
      m_C = m_D. The points m_T + h e_A and m_T + h e_B have m_C = m_D, and m_T + h e_C, m_T + h e_D have m_A = m_B.   QED
```

In words: at the tree, D moves at first order only through the alternating sum of the four distance-3 correlations; the
a/b imbalance X_a - X_b and the c/d imbalance Y_d - Y_c enter only as a product, and each tree type breaks one of the two
symmetries, never both. So on RC,
whose tangent space at C_T is the orthogonal complement of grad D, the saddle is gone. Lemma LOC-4D (Section 6) certifies
this negative definiteness in the form needed.

### 5.2 One common factor

**Lemma R1a.** Let C be in RC with |C_ab| < 1 and |C_cd| < 1, and let P_u, P_v be its two 2 x 2 pair blocks. Then
there are alpha = (alpha_a, alpha_b) and gamma = (gamma_c, gamma_d) with K = alpha gamma^T and P_u - alpha alpha^T,
P_v - gamma gamma^T PSD. Consequently, with Z ~ N(0, 1), eps_u ~ N(0, P_u - alpha alpha^T), eps_v ~ N(0, P_v - gamma gamma^T)
independent, (alpha Z + eps_u, gamma Z + eps_v) ~ N(0, C), and

    Phi_F(C) = arccos(C_ab)/pi + arccos(C_cd)/pi + Int phi(z) [ q_u^+(z) q_v^-(z) + q_u^-(z) q_v^+(z) ] dz,

where q_u^+(z) = Pr[alpha z + eps_u > 0 coordinatewise], q_u^-(z) = Pr[alpha z + eps_u <= 0 coordinatewise], and likewise for v.
Moreover, for alpha', gamma' with K = alpha' gamma'^T, C is PSD iff q_u q_v <= 1, where q_u := alpha'^T P_u^{-1} alpha' and
q_v := gamma'^T P_v^{-1} gamma'.

In words: on RC the four keys share one Gaussian factor, and given it the two pairs are independent.

```text
<1>1. If K = 0 take alpha = gamma = 0. Otherwise K = alpha' gamma'^T with both nonzero; for lambda > 0, alpha = lambda alpha'
      and gamma = gamma'/lambda give the same K.
<1>2. For P positive definite, x != 0 and c >= 0: P - c x x^T is PSD iff c x^T P^{-1} x <= 1 (conjugate by P^{-1/2}: the
      matrix I - c y y^T with y = P^{-1/2} x has eigenvalues 1 - c|y|^2 and 1).
<1>3. P_u, P_v are positive definite. C is PSD iff the Schur complement P_v - K^T P_u^{-1} K = P_v - q_u gamma' gamma'^T is
      PSD, iff q_u q_v <= 1 by <1>2.
<1>4. P_u - lambda^2 alpha' alpha'^T is PSD iff lambda^2 q_u <= 1, and P_v - gamma' gamma'^T/lambda^2 is PSD iff
      q_v <= lambda^2. When q_u q_v <= 1, both hold for lambda^2 in [q_v, 1/q_u].
<1>5. The vector has covariance [[P_u, alpha gamma^T], [gamma alpha^T, P_v]] = C, and given Z = z the pairs are independent.
      By (2.3), Phi_F = Pr[a != b] + Pr[c != d] + Pr[a = b != c = d]; the first two are arccos(C_ab)/pi and arccos(C_cd)/pi
      (Lemma G(c)) and the third is E[q_u^+(Z) q_v^-(Z) + q_u^-(Z) q_v^+(Z)].   QED
```

Corollary T is the case alpha = gamma = (sqrt s, sqrt s), residual correlation 1/sqrt 2.

### 5.3 The kill, and Theorem WIT

Every C(m) has nonnegative entries (Ghat >= 0 entrywise, so W >= 0).

**Definition (the kill).** For m in the cone put C = C(m). If D(C) = 0, the kill is C'(m) := C and delta(m) := 0. If D > 0,
lower the entry ac to C_ad C_bc/C_bd (a decrease of D/C_bd) or the entry bd to C_ad C_bc/C_ac (a decrease of D/C_ac),
whichever decrease is smaller. If D < 0, do the same with the roles of (ac, bd) and (ad, bc) exchanged. So

    delta(m) = |D(C(m))| / max(partner pair),

the *partner pair* being (C_ac, C_bd) if D > 0 and (C_ad, C_bc) if D < 0. The kill C'(m) has D = 0 and C'(m) <= C(m)
entrywise; it lies in RC exactly when it is PSD.

**Theorem WIT.** Let C be a 4 x 4 correlation matrix with nonnegative cross block K and nonnegative pair entries.
(a) Every rank-one 2 x 2 matrix R with R <= K entrywise has sum(R) <= sum(K) - delta(K), where delta(K) = |det K|/max(partner
    pair).
(b) Every PSD C'' in RC with C'' <= C entrywise has a.C'' <= a.C - delta(K).
(c) If the kill C' is PSD, it attains (b): max { a.C'' : C'' in RC, C'' <= C } = a.C - delta(K).

In words: among all rank-one matrices below the cross block, the kill has the largest sum.

PROOF IDEA. Scale the factor so that R = (1, beta)^T (g_c, g_d). The entrywise constraints bound the sum by a function of
beta that is increasing up to the first kink, convex between the kinks, and decreasing after the second; the two kink values
are sum K - D/K_ac and sum K - D/K_bd.

```text
<1>1. (positive factors) Let R = alpha gamma^T with alpha, gamma > 0. Rescale so alpha = (1, beta), beta > 0. R <= K reads
      g_c <= min(K_ac, K_bc/beta) and g_d <= min(K_ad, K_bd/beta), so sum R <= phi(beta) := (1 + beta)(min(K_ac, K_bc/beta) +
      min(K_ad, K_bd/beta)).
<1>2. Suppose D >= 0, so beta_1 := K_bc/K_ac <= beta_2 := K_bd/K_ad (zero entries make a kink 0 or infinite; drop empty pieces).
  <2>1. On (0, beta_1]: phi = (1 + beta)(K_ac + K_ad), increasing; phi(beta_1) = sum K - D/K_ac.
  <2>2. On [beta_1, beta_2]: phi = K_bc + K_ad + K_bc/beta + beta K_ad, convex, so maximal at an end; phi(beta_2) =
        sum K - D/K_bd.
  <2>3. On [beta_2, oo): phi = (K_bc + K_bd)(1 + 1/beta), decreasing.
  <2>4. So max phi = sum K - min(D/K_ac, D/K_bd) = sum K - delta. D < 0 is the same with (ac, bd) and (ad, bc) exchanged.
<1>3. (other sign patterns) Note delta <= min(partner pair) (since |D| <= product of the partners), so sum K - delta is at
      least every row sum, every column sum, and every single entry of K.
  <2>1. A zero entry in alpha or gamma: R has a zero row or column, so sum R is at most one row or column sum of K.
  <2>2. alpha = (1, -beta), beta > 0, gamma >= 0: sum R = (1 - beta)(g_c + g_d) <= g_c + g_d <= K_ac + K_ad. One negative
        entry in gamma: symmetric.
  <2>3. alpha = (1, -beta), gamma = (g_c, -g_d), beta, g_c, g_d > 0: sum R = (1 - beta)(g_c - g_d), which is <= g_c <= K_ac if
        beta <= 1 and <= (beta - 1) g_d <= beta g_d <= K_bd if beta > 1.
  <2>4. R <= 0 entrywise: sum R <= 0. Overall factor signs reduce to these cases. So (a) holds.
<1>4. (b) The cross block R'' of C'' has rank <= 1 and R'' <= K; the pair entries of C'' are <= those of C. Since a >= 0,
      a.C - a.C'' >= sum K - sum R'' >= delta by (a).
<1>5. (c) The PSD kill is in RC, below C, and has a.C' = a.C - delta (the lowered entry is a cross entry, of weight 1).  QED
```

### 5.4 Theorem TS*: two obligations suffice

**Definition.** For m in the cone put

    delta*(m) := a.C(m) - max { a.C'' : C'' in RC, C'' <= C(m) entrywise }.

The maximum exists: the feasible set is compact and nonempty (C(m) with its cross block replaced by 0 is PSD, block diagonal,
and below C(m) since C(m) >= 0). By Theorem WIT (b), delta*(m) >= delta(m), with equality when the kill is PSD.

The two obligations are:

**(R-i').** For every PSD correlation matrix C' with D(C') = 0 and a.C' >= K0: Phi_F(C') <= M_tree.

**(R-ii\*) at m.** a.C(m) - K0 >= delta*(m). Equivalently (by the definition): some PSD C'' in RC with C'' <= C(m)
entrywise has a.C'' >= K0. Such a C'' is a *witness* for m.

**Theorem TS\*.** Assume (R-i'). If (R-ii*) holds at m, then f(m) <= M_tree. If it holds at every m, then
M_123 = M_tree.

```text
<1>1. Let C'' attain the maximum in delta*(m). It is PSD, lies in RC, and a.C'' = a.C(m) - delta*(m) >= K0 by (R-ii*).
<1>2. By (R-i'), Phi_F(C'') <= M_tree.
<1>3. C'' <= C(m) entrywise and both are correlation matrices, so by Lemma G(b), f(m) = Phi_F(C(m)) <= Phi_F(C'') <= M_tree.
<1>4. If this holds for all m, M_123 <= M_tree; and M_123 >= f(m_T) = M_tree.   QED
```

The proof is pointwise; Proposition NEED (Section 7) uses this.

*Remark.* The kill C'(m) need not be PSD: at m = (e_A + e_C)/2 on the tree face, C(m) has rank 2, D = -0.0174786, the kill lowers ad by
0.0192387, and x^T C'(m) x = -0.0086332 for an explicit x (the kill is also non-PSD at about half of random two-generator
points). Theorem TS* never asks for the kill to be PSD: it asks for some witness. We shall need the kill to be PSD only on
the thin set N of Section 7, and we prove it there.

*What remains.* Section 6 proves (R-i') (Theorem R1). Sections 7-9 prove (R-ii*) at every m where it is needed.

---

## 6. Theorem R1: (R-i') holds

**Theorem R1.** (uses the finite claims FC-AAB, FC-MT, FC-LOC4D and FC-R1) Let C' be a 4 x 4
PSD correlation matrix with D(C') = 0 and a.C' >= K0. Then Phi_F(C') <= M_tree, with equality only at C' = C_T. That is,
(R-i') holds.

In words: on the family where the saddle is gone, the tree cut alone pins the cost to the tree value.

PROOF IDEA. Four hand lemmas turn (R-i') into a statement about an explicit smooth function of four real parameters on a
compact set. On RC the cross block is a product alpha gamma^T, so five numbers describe the matrix (Lemma DOM). The cut can
be taken tight, because lowering the common size of the two factors only lowers cross entries (Lemma S-RED). Positive
semidefiniteness is one explicit inequality (Lemma DOM'). Near the tree the reduced function has a nondegenerate maximum,
certified by an interval Hessian (Lemma LOC-4D). The rest of the four-parameter set is covered by a certified branch and
bound.

### 6.1 Five coordinates

Put x_min := (K0 - a_ab - 4)/a_ab = 0.78341 and S_min := sqrt(K0 - 2 a_ab) = 1.74069 (both computed from the intervals of
FC-AAB and rounded in the safe direction).

**Lemma DOM.** Let C' be a PSD correlation matrix with D(C') = 0 and a.C' >= K0. Then
(a) every cross entry of C' is >= 0.0297 and every pair entry is >= x_min;
(b) there are x, y in [x_min, 1], S in [S_min, 2] and t_u, t_v in (-0.3, 0.3) such that C'_ab = x, C'_cd = y and the cross
    block is alpha gamma^T with alpha = (S/2 + t_u, S/2 - t_u), gamma = (S/2 + t_v, S/2 - t_v); that is,
        C'_ac = (S/2 + t_u)(S/2 + t_v),  C'_ad = (S/2 + t_u)(S/2 - t_v),  C'_bc = (S/2 - t_u)(S/2 + t_v),  C'_bd = (S/2 - t_u)(S/2 - t_v);
(c) a.C' = a_ab (x + y) + S^2, so the cut reads a_ab (x + y - 2p) + (S^2 - 4s) >= 0;
(d) Phi_F(C'(x, y, S, t_u, t_v)) is invariant under t_u -> -t_u, under t_v -> -t_v and under (x, t_u) <-> (y, t_v).

We write C'(theta) for the matrix of (b) at theta = (x, y, S, t_u, t_v).

```text
<1>1. (a) Every entry is <= 1. If a cross entry were < 0.0297, then a.C' < 2 a_ab + 3 + 0.0297 = 11.98647 < K0. If a pair
      entry were < x_min, then a.C' < a_ab x_min + a_ab + 4 = K0. (0.0297 is a rounded-down value of K0 - 2 a_ab - 3 = 0.03003.)
<1>2. (b) K has rank <= 1 and positive entries, so K = alpha' gamma'^T with alpha', gamma' entrywise positive (alpha' = the
      first column of K, gamma' = (1, K_ad/K_ac)). Put lambda^2 = (gamma'_c + gamma'_d)/(alpha'_a + alpha'_b), alpha =
      lambda alpha', gamma = gamma'/lambda. Then alpha_a + alpha_b = gamma_c + gamma_d =: S, and t_u := (alpha_a - alpha_b)/2,
      t_v := (gamma_c - gamma_d)/2. The four cross entries sum to (alpha_a + alpha_b)(gamma_c + gamma_d) = S^2.
<1>3. (ranges) x, y <= 1 and S^2 <= 4 (four entries <= 1) with a_ab (x + y) + S^2 >= K0 give x >= x_min, y >= x_min and
      S^2 >= K0 - 2 a_ab. If |t_u| >= 0.3 then max(alpha_a, alpha_b) >= S/2 + 0.3 >= 1.17, so gamma_c, gamma_d <= 1/1.17 < 0.855
      (the products are <= 1), and S = gamma_c + gamma_d < 1.71 < S_min: impossible. Likewise for t_v.
<1>4. (c) a.C' = a_ab (x + y) + (sum of cross entries); and K0 = 2 a_ab p + 4 s.
<1>5. (d) t_u -> -t_u swaps alpha_a and alpha_b, that is, the keys a and b (ac <-> bc, ad <-> bd); F is symmetric in (s_1, s_2).
      Likewise t_v. (x, t_u) <-> (y, t_v) swaps the pairs (ab <-> cd, ad <-> bc); F is symmetric under (s_1, s_2) <-> (s_3, s_4).
      The cut is invariant under all three.   QED
```

### 6.2 The cut may be taken tight

**Lemma DOM'.** Let C = C'(x, y, S, t_u, t_v) with S/2 > |t_u|, |t_v|.
(a) If |x|, |y| < 1: C is PSD iff q_u q_v <= 1, where

    q_u = S^2/(2(1 + x)) + 2 t_u^2/(1 - x),        q_v = S^2/(2(1 + y)) + 2 t_v^2/(1 - y).

(b) If x = 1 and |y| < 1: C is PSD iff t_u = 0 and (S^2/4) q_v <= 1 (symmetrically for y = 1); if x = y = 1, iff
    t_u = t_v = 0 and S^2/4 <= 1.

```text
<1>1. (a) is Lemma R1a's criterion q_u q_v <= 1 with alpha' = alpha, gamma' = gamma. The formula: P_u^{-1} =
      (1 - x^2)^{-1} [[1, -x], [-x, 1]], alpha_a^2 + alpha_b^2 = S^2/2 + 2 t_u^2, alpha_a alpha_b = S^2/4 - t_u^2, so
      alpha^T P_u^{-1} alpha = (S^2 (1 - x)/2 + 2 t_u^2 (1 + x))/(1 - x^2) = S^2/(2(1 + x)) + 2 t_u^2/(1 - x).
<1>2. (b) If x = 1, a PSD C has X_a = X_b a.s., so rows a and b agree: alpha_a gamma = alpha_b gamma, and gamma != 0 forces
      t_u = 0. Conversely with t_u = 0 the rows agree, and C is PSD iff its submatrix on (a, c, d),
      [[1, (S/2) gamma^T], [(S/2) gamma, P_v]], is PSD, iff P_v - (S^2/4) gamma gamma^T is PSD, iff (S^2/4) q_v <= 1 (Lemma R1a
      <1>2). The case x = y = 1 is the same on both sides.   QED
```

**Lemma S-RED.** Let theta = (x, y, S, t_u, t_v) satisfy the conclusions of Lemma DOM with C'(theta) PSD, and put
S_cut := sqrt(K0 - a_ab (x + y)). Then theta' = (x, y, S_cut, t_u, t_v) is again of this kind, the cut is tight at theta', and
Phi_F(C'(theta')) >= Phi_F(C'(theta)).

In words: lowering the total cross correlation to the smallest value the cut allows never lowers the cost.

```text
<1>1. The cut says S^2 >= S_cut^2, so S_cut <= S. Since x + y <= 2, S_cut >= S_min, so S'/2 - 0.3 >= 0.57 > 0 for every S' in
      [S_cut, S]: the factors S'/2 +- t_u, S'/2 +- t_v are positive and nondecreasing in S'.
<1>2. So every cross entry is nondecreasing in S' on [S_cut, S], the pair entries do not change, and C'(theta') <= C'(theta)
      entrywise.
<1>3. PSD: by Lemma DOM' the condition is q_u q_v <= 1 (or its x = 1 form), and q_u, q_v increase with S'. So PSD at S gives
      PSD at S_cut.
<1>4. The cut is tight at theta' by the choice of S_cut. By Lemma G(b), Phi_F(C'(theta)) <= Phi_F(C'(theta')).   QED
```

So (R-i') is equivalent to the four-parameter statement

    Phi_4(theta) := Phi_F(C'(x, y, S_cut(x, y), t_u, t_v)) <= M_tree   for every theta = (x, y, t_u, t_v) in R4,

    R4 := { x_min <= x <= y <= 1, 0 <= t_u, t_v <= 0.3, C'(x, y, S_cut(x, y), t_u, t_v) PSD }

(the symmetries of Lemma DOM (d) give t_u, t_v >= 0 and x <= y; "cross entries <= 1" is implied by PSD). The tree is
theta_T = (p, p, 0, 0), where S_cut = 2 sqrt s.

### 6.3 The local lemma

The map theta -> C(theta) := C'(x, y, S_cut(x, y), t_u, t_v) is smooth where S_cut > 0. Write J4 for its 6 x 4 Jacobian and
Hess C_p for the 4 x 4 Hessians of its entries. In closed form, with S = S_cut, S_x = -a_ab/(2S), S_xx = -a_ab^2/(4 S^3) and,
for the cross entry with signs (s_u, s_v), A = S/2 + s_u t_u, B = S/2 + s_v t_v:

    dC_p/dx = dC_p/dy = (A + B) S_x/2,   dC_p/dt_u = s_u B,   dC_p/dt_v = s_v A;
    d^2C_p/dx^2 = d^2C_p/dx dy = d^2C_p/dy^2 = S_x^2/2 + (A + B) S_xx/2,   d^2C_p/dx dt_u = s_u S_x/2,
    d^2C_p/dx dt_v = s_v S_x/2,   d^2C_p/dt_u dt_v = s_u s_v,   the others 0;

the pair entries x, y have zero Hessians.

**Lemma LOC-4D.** (uses the finite claim FC-LOC4D) Let B_0 := {theta : |theta - theta_T|_inf <= 1e-4}. B_0 lies in the feasible set,
and for every theta in B_0,

    Phi_4(theta) - M_tree <= -0.3349 |theta - theta_T|^2,

with equality only at theta_T.

PROOF IDEA. The cut value a.C(theta) is identically K0 on the family, and the gradient of Phi_F at C_T is a multiple of -a. So
the gradient of Phi_4 at theta_T vanishes, and Phi_4 - M_tree is a pure second-order quantity, whose Hessian is certified
negative definite on B_0.

```text
<1>1. a.C(theta) = a_ab (x + y) + S_cut(x, y)^2 = K0 identically, so a^T J4(theta) = 0 and sum_p a_p Hess C_p(theta) = 0.
<1>2. grad Phi_F(C_T) = -c_T a, so grad Phi_4(theta_T) = J4(theta_T)^T (-c_T a) = 0.
<1>3. B_0 is convex; its image under C is certified positive definite (FC-LOC4D); its points have cross entries within 3e-4
      of s, so they are feasible. Taylor with the Lagrange remainder from theta_T, with d = theta - theta_T:
          Phi_4(theta) - M_tree = (1/2) d^T M4(xi) d,   M4(xi) = J4(xi)^T H(C(xi)) J4(xi) + sum_p g_p(C(xi)) Hess C_p(xi),
      for some xi in B_0, H and g the Hessian and gradient of Phi_F.
<1>4. By FC-LOC4D every symmetric matrix in the interval enclosure of M4(B_0) has largest eigenvalue <= -0.6698, so
      Phi_4(theta) - M_tree <= -(0.6698/2) |d|^2.   QED
```

**Finite claim FC-LOC4D.** (computer-checked: bk9_bnb.py setup, check.log) Over B_0 = {|theta - theta_T|_inf <= 1e-4}: (i) the image box of C is certified
positive definite (pd_box, interval Cholesky); (ii) with H and g enclosed by grad_ad over the image box, and J4,
Hess C_p enclosed by interval evaluation over B_0, every symmetric matrix in the interval matrix M4(B_0) has
lambda_max <= -0.6698. Method: midpoint eigenvalues -8.5878, -8.5183, -2.7821, -2.7821; Frobenius norm of the radius matrix
2.1123; Weyl's inequality. The midpoint eigenvalue step is guarded: ||mid||_F <= 20 (measured 12.72) and
mid - (lambda_max + 1e-12) I is checked negative definite in exact rational arithmetic (the exact test applies to
the float matrix fl(mid - sI), whose diagonal carries one rounding of at most 1.1e-15, irrelevant against -0.6698).

### 6.4 The certificate outside B_0

A *theta-box* is a box [lo, hi] in (x, y, t_u, t_v). A box is *closed* by the first of the following rules that applies.
Every number entering a bound is the appropriate end of an outward-rounded enclosure (operations: round to nearest, then one
ulp outward; arcsin verified). Each rule proves the statement in brackets for every theta of the box that lies in R4.

- **XY**: lo_x > hi_y. [No theta of the box lies in R4.]
- **INF-PSD**: the product of the box lower bounds of q_u and q_v (Lemma DOM'; q_u >= S_lo^2/(2(1 + x_hi)) + 2 t_u,lo^2/(1 - x_lo),
  the second term dropped when x_lo = 1) exceeds 1. [No PSD point; by Lemma DOM' (b) a point with x = 1 and t_u > 0 is
  not PSD either.]
- **INF-K**: the lower end of the largest cross entry exceeds 1. [No correlation matrix.]
- **LOC**: the box lies in B_0. [Lemma LOC-4D.]
- **CORNER**: let lo'' be the entrywise lower corner of the image box, shrunk toward I as (1 - t) lo + t I with t chosen so
  that lo'' is certified positive definite. All entries are positive, so lo'' <= C(theta) entrywise for every theta of the
  box, and by Lemma G(b) Phi_4(theta) <= Phi_F(lo''). Closed if an upper enclosure of Phi_F(lo'') (phi_point or
  F_point) is <= Mt_lo. [Phi_4 <= Mt_lo.]
- **TAY1, TAY2** (only if the image box is certified positive definite): first- and second-order mean-value bounds in C-space
  along the segment from a double matrix c_0 near C(theta_0) (theta_0 the box centre) to C(theta), with interval gradients
  and Hessians (grad_ad), the exact quadratic structure of the chart, and a concave-quadratic maximisation over the box. For
  TAY2 the quadratic form is split M = M_neg + M_pos + E, with M_neg from the negative eigenvalues of mid(M) shifted by
  -1e-9, M_pos bounded by (1/2) sum lambda_k (|V_k|.w)^2 and E bounded entrywise. Closed if the total is <= Mt_lo.
  [Phi_4 <= Mt_lo.]
- **The guard on TAY2** (applied in all replays). The float eigen-decomposition is accepted only
  if (G1) ||M_neg||_F <= 1e4, (G2) lambda+_max x max_ij sum_k |V_ik||V_jk| <= 700, and (G3) M_neg is negative definite by
  exact rational Gaussian elimination. (G1) bounds the LAPACK backward error of the test eigvalsh(M_neg) <= -5e-10; (G2)
  bounds the rounding of M_pos by 9.4e-13 |w|^2, inside TAY2's 1e-12 |w|^2 allowance; (G3) removes any floating-point
  assumption from the concavity step. A refused box goes to the other rules or is split.
- **TAY3** (run W only): exact second-order Taylor about the box centre theta_0. Every one of the k^4 sub-box images (k = 2, then 4)
  is certified positive definite; the Hessian at the intermediate point lies in the hull of interval Hessians over the
  sub-boxes (the (i, j) and (j, i) enclosures intersected); arcsin is verified (degree-25 Taylor enclosure of sin with
  Lagrange remainder); a_ab and K0 are those of FC-AAB. The maximum of Gm.d + (1/2) d^T Mm d over the box is bounded by
  q(dc) + sum_j max_{+-} grad_j q(dc)(+-w_j - dc_j) [+ (1/2) sum_i (Mm_ii - Mq_ii) w_i^2], where Mq = Mm or Mm - sI is
  certified negative definite by exact rational elimination (so q is concave and the gradient inequality holds at any dc).
  The base value: c_dn = the lower ends of an outward enclosure of C(theta_0), certified positive definite, so
  Phi_4(theta_0) <= Phi_F(c_dn) by Lemma G(b), bounded by phi_point or F_point. Closed if the total is <= Mt_lo.
  [Phi_4 <= Mt_lo.]
- Otherwise the box is split in two (runs A, RVA, B, RVB: along the coordinate with the largest Jacobian column norm times
  width; the records of run W store the coordinate j and the cut point).

A few ulp-level steps in TAY1/TAY2 (float sums forming the Taylor totals, the float box centre) are
round-to-nearest rather than outward. Their effect is below 1e-14 and is absorbed because closures are tested against
Mt_lo, which lies below M_tree, while the theorem needs only M_tree. (In INF-PSD the bound is a product of interval
lower ends and has no such step.)

**Finite claim FC-R1 (the five trees).** (computer-checked) R4 lies in the union of the two roots

    Q_A = [x_min, 1] x [x_min, fl(p)+] x [0, 0.3]^2,        Q_B = [x_min, 1] x [fl(p)-, 1] x [0, 0.3]^2

(fl(p)+ and fl(p)- the doubles just above and below p), and every theta-box of the following five trees is resolved: closed
by one of the rules above, or split into two children that are resolved.

| tree | root(s) | boxes | record | closures |
|---|---|---|---|---|
| run A | Q_A | 34,302 | bnb9_A.json (counters, open list) | INF-PSD 1,374, INF-K 17, XY 1,685, LOC 1, CORNER 8,038, TAY2 6,028, TAY1 4 (as replayed under the guard), splits 17,155; 9 open boxes |
| RVA | the 9 open boxes of A | 7,847 | bnb9_RVA.json | closes all 9 |
| run B | Q_B | 29,916 | bnb9_B.json | INF-PSD 3,126, INF-K 2, XY 86, CORNER 9,469, TAY2 2,269, TAY1 1, splits 14,963; 11 open boxes |
| RVB | the 11 open boxes of B | 22,841 | bnb9_RVB.json | leaves 17,326 open boxes (width <= 5.36e-3, within 0.083 of the tree) |
| run W | the 17,326 open boxes of RVB | 74,096 | rec_W0.jsonl, rec_W1.jsonl | CORNER 27,660, TAY3 12,505, INF-PSD 3,650, XY 1,746, TAY2 147, TAY1 2, LOC 1, splits 28,385; 0 open |

Total 169,002 boxes. theta_T lies only in LOC leaves. The smallest TAY3 margin Mt_lo - U is 2.96e-9.

- *Record format* (run W): one JSON line per popped box with its bounds and outcome; a closure record holds the rule (and the
  guard magnitudes for TAY2, or k and the bound U for TAY3); a split record holds the coordinate j and the cut point.
  The json files of runs A, RVA, B and RVB hold counters and open lists; their box-level records were regenerated by
  guarded deterministic replays.
- *What the checkers verify*: verify_cover.py rebuilds the tree of run W from the 17,326 roots (children recomputed as the
  bisection of coordinate j at the recorded point) and must print "unresolved boxes: 0"; replay10.py re-executes N random
  decisions through the closure code and compares; bk10_run.py re-executes everything deterministically (about 5.4 CPU
  hours).

### 6.5 Proof of Theorem R1

```text
<1>1. By Lemma DOM, C' = C'(theta) for some theta in the chart; by Lemma DOM (d) we may take t_u, t_v >= 0 and x <= y.
<1>2. By Lemma S-RED, Phi_F(C'(theta)) <= Phi_4(x, y, t_u, t_v), and (x, y, t_u, t_v) lies in R4 (Lemma DOM').
<1>3. R4 lies in Q_A u Q_B, and every box of the five trees is resolved (FC-R1).
<1>4. Each leaf gives, for every point of its box in R4: XY, INF-PSD, INF-K: no such point; CORNER, TAY1, TAY2, TAY3:
      Phi_4 <= Mt_lo <= M_tree (FC-MT); LOC: Phi_4 - M_tree <= -0.3349 |theta - theta_T|^2 (Lemma LOC-4D). So Phi_4 <= M_tree on
      R4, and Phi_F(C') <= M_tree.
<1>5. Equality. On R4, the CORNER and TAY leaves give Phi_4 <= Mt_lo < M_tree (Section 4.3, strictness), and LOC gives equality
      only at theta_T. So Phi_F(C') = M_tree forces (x, y, t_u, t_v) = theta_T, i.e. x = y = p < 1 and t = 0. Only now use the
      strictness of S-RED, at this point. Let S >= S_cut be the S of C'. With x = y = p < 1, q_u q_v is strictly increasing in
      S' and is <= 1 at S' = S, so it is < 1 for S' in [S_cut, S); there the matrices C'(p, p, S', 0, 0) are positive definite
      (Lemma R1a <1>2-<1>3 with strict inequality). At positive definite matrices every cross-entry partial derivative of
      Phi_F is < 0 (2.4), and each cross entry (S'/2)^2 strictly increases with S'. So if S > S_cut, then
      Phi_F(C') < Phi_F(C'(p, p, S_cut, 0, 0)) = Phi_4(theta_T) <= M_tree. Hence equality forces S = S_cut and
      C' = C'(p, p, 2 sqrt s, 0, 0) = C_T.   QED
```

---

## 7. Where (R-ii*) is needed, and the structure there

By Theorem TS* and Theorem R1, it remains to prove (R-ii*) at every cone point m. This section shows that (R-ii*) is needed
only on a thin set N (Proposition NEED), proves the facts about N that the certificates use, and rewrites the tree cut as
a problem about two "stars" (Theorem TWO-STAR). Section 8 proves the kill inequality on N and Section 9 that the kill is
PSD on N; Lemma KW below turns these two facts into (R-ii*).

### 7.1 The thin set N

The rule bound already gives f(m) < M_tree wherever RB(m) < M_tree. Since M_tree is known only in an interval, we use a
certified lower bound.

**Definition.** Put M_lo := 0.2765836068 (a certified lower bound of M_tree: M_lo < Mt_lo, FC-MT) and

    N := { m in the simplex : RB(m) >= M_lo }.

(Any lower bound of M_tree works; a smaller one only enlarges N. N must be defined with a certified lower
bound, not with M_tree itself.)

**Proposition NEED.** If (R-ii*) holds at every m in N, then M_123 = M_tree.

```text
<1>1. If m is not in N, f(m) <= RB(m) < M_lo <= M_tree (Lemma RB).
<1>2. If m is in N, Theorem TS* at m (its proof is pointwise, and it assumes (R-i'), which is Theorem R1) gives
      f(m) <= M_tree.
<1>3. So M_123 <= M_tree; and M_123 >= f(m_T) = M_tree.   QED
```

Here are the constants that describe N. Put

    Theta := 2 pi M_lo - 3 arccos(mu0),     c_Theta := cos(Theta/3) = 0.9595702342...,
    B := c_Theta - mu0, rounded up = 0.0024634531.

(The certificates carry c_Theta as an interval with upper end 0.9595702342051.)

**Lemma NS (each star is near the tree).** For every m in N, kbar_u <= c_Theta and kbar_v <= c_Theta.

```text
<1>1. m in N means S_u + S_v >= 2 pi M_lo.
<1>2. S_v <= 3 arccos(mu0) (Lemma STAR (d)), so S_u >= Theta.
<1>3. S_u <= 3 arccos(kbar_u) (Lemma STAR (c)), so arccos(kbar_u) >= Theta/3; since 0 < Theta/3 < pi and arccos is
      decreasing, kbar_u <= cos(Theta/3). The same at v.   QED
```

In words: the v-star can contribute at most its tree share to the rule bound, so the u-star must stay within B = 0.0025 of
the tree in mean key correlation, and vice versa. Write E_u := kbar_u - mu0 and E_v := kbar_v - mu0 (the *excesses*; both
are >= 0 by Lemma STAR (b)).

**Lemma JOINT (the two stars share one budget).** Let s_0 := sqrt(1 - mu0^2) and s_c := sqrt(1 - c_Theta^2). For
every m in N,

    E_u + (s_c/s_0) E_v <= c_Theta - mu0,        s_c/s_0 = 0.9714701...,

and symmetrically with u and v exchanged. We use it with w := 0.97147 <= s_c/s_0: E_u + w E_v <= B.

```text
<1>1. On N, arccos(kbar_u) + arccos(kbar_v) >= (S_u + S_v)/3 >= 2 pi M_lo/3 (Lemma STAR (c) at both stars).
<1>2. arccos is concave on [0, 1], so arccos(x) <= arccos(x_0) - (x - x_0)/sqrt(1 - x_0^2) for x, x_0 in [0, 1). Use x_0 = c_Theta
      at u and x_0 = mu0 at v. Since arccos(c_Theta) + arccos(mu0) = Theta/3 + arccos(mu0) = 2 pi M_lo/3, <1>1 gives
      (kbar_u - c_Theta)/s_c + (kbar_v - mu0)/s_0 <= 0. Multiply by s_c and rearrange.   QED
```

In words: if one star spends the whole budget B, the other is forced to the equality case of the star lemma. Two "far"
stars never occur together. Sections 8 and 9 both use this.

### 7.2 The ratio box

Write l_uv := log d_v, l_ua := log d_a, l_ub := log d_b, l_vc := log(d_c/d_v), l_vd := log(d_d/d_v), the five *edge
log-ratios*. They are independent coordinates for the diagonal vector.

**Lemma R7 (the ratio box).** (uses the finite claim FC-R7) Every m in N has

    |l_uv| < 0.20516,      |l_ua|, |l_ub|, |l_vc|, |l_vd| < 0.28903.

```text
<1>1. For each internal edge, the bound of Lemma EC is e_xy(d) := (rho/2)(sqrt r + 1/sqrt r) = rho cosh(l_xy/2) with r the
      ratio along the edge; it is increasing in |l_xy|. In Lemma RC's ceiling for a single point d,
      R(d) = (3/(2 pi)) [arccos min(1, mu(e_uv, e_ua, e_ub)) + arccos min(1, mu(e_uv, e_vc, e_vd))],
      mu is increasing in each argument and arccos is decreasing, so R is nonincreasing in each |l_xy| separately.
<1>2. Hence if |l_ua| >= 0.28903, R(d) <= R at the point with l_ua = 0.28903 and the other four log-ratios 0; by FC-R7 that
      value is < M_lo, so RB(m) <= R(d(m)) < M_lo and m is not in N. Likewise for ub, vc, vd, and for uv with 0.20516.  QED
```

**Finite claim FC-R7.** (computer-checked: 40-digit evaluation, and mpmath.iv interval evaluation at 30 digits)
The single-edge rule ceilings are R = 0.2765835647... at |l| = 0.28903 on one of the edges ua, ub, vc, vd (others 0) and
R = 0.2765835134... at |l_uv| = 0.20516 (others 0), both < M_lo = 0.2765836068. The interval evaluation encloses
rho cosh(l/2), mu, arccos and the final sum outward.

On N, therefore, the u-star's ratios (d_a, d_b) and the *mirror* star's ratios (d_c/d_v, d_d/d_v) lie in the square
|log| < 0.28903 (the *R7 square*), and d_v = V lies in [0.81451, 1.22773] (e^{+-0.20516}, widened outward).

### 7.3 Two linear inequalities on the Gram entries

**Lemma MCC.** Let (x, z) be a pair at distance 2 with middle vertex w (w = u for va, vb, ab; w = v for uc, ud, cd).
For every m, G_xz >= (G_wx + G_wz)/sqrt 2 - G_ww/2, with equality iff every type in the support of m has t_x = t_w + 1 or
t_z = t_w + 1.

```text
<1>1. Fix t and put alpha = g_t(w), beta = g_t(x), gamma = g_t(z). Since |t_x - t_w|, |t_z - t_w| <= 1, beta >= alpha/sqrt 2 and
      gamma >= alpha/sqrt 2. So (beta - alpha/sqrt 2)(gamma - alpha/sqrt 2) >= 0, i.e. beta gamma >= (alpha/sqrt 2)(beta + gamma)
      - alpha^2/2, with equality iff a factor vanishes. Multiply by m_t and sum.   QED
```

At the tree both sides equal 5/6.

**Lemma D3.** (uses the finite claim FC-D3) For each of the four distance-3 pairs (x, z), x in {a, b}, z in {c, d},
every m satisfies

    (sqrt 2/12)(G_uu + G_vv) + (sqrt 2/6)(G_xx + G_zz)  <=  G_xz  <=  (3/sqrt 2)(G_uz + G_vx) - (2 sqrt 2/3)(G_uu + G_vv) - (sqrt 2/3)(G_xx + G_zz).

```text
<1>1. Both sides are linear in G, and G(m) = sum m_t g_t g_t^T with m_t >= 0. So it suffices that each of the 8 inequalities
      holds for every generator, G = g_t g_t^T. That is FC-D3.   QED
```

**Finite claim FC-D3.** (computer-checked: bk12_exact.py) For each of the 123 generators
g_t (entries powers of sqrt 2) and each of the 8 inequalities of Lemma D3, the inequality holds in exact arithmetic in
Q(sqrt 2). All 8 are tight exactly at the four tree types and at (0,-1,1,1,-2,-2), (0,1,-1,-1,2,2), (0,1,1,1,2,2),
(0,-1,1,1,0,0); the smallest positive slack is 1/2 - sqrt 2/3 = 0.0286. (A reader can check this by hand from the type
list in a few hours, or with any computer algebra system in seconds.)

**The key-correlation formula.** (Proof: expand the inner products.) With N_x := |â_u + â_x| = sqrt(2 + 2 e_ux) for x in
{a, b}, N_uv := sqrt(2 + 2 e_uv), and h_va := Ghat_va,

    k_a = <y_ua, y_uv> = (1 + e_uv + e_ua + h_va)/(N_uv N_a),     k_ab = (1 + e_ua + e_ub + h_ab)/(N_a N_b),

and likewise at v.

### 7.4 N is thin: certified constants

**Lemma THIN.** (uses the finite claim FC-THIN) For every m in N and each of the two stars:
(a) k_a + k_b + k_ab <= 3 c_Theta (Lemma NS), and k_a + k_b + k_ab >= 3 mu(e_uv, e_ua, e_ub) (Lemma STAR (b));
(b) N_uv + N_a + N_b <= Sigma* := 2 sqrt(6 c_Theta + 3) = 5.9185881;
(c) every internal edge cosine lies in [rho, e_max], e_max := (Sigma* - 2 N0)^2/2 - 1 = 0.9526712, and every N_x lies in
    [N0, N_max], N_max := Sigma* - 2 N0 = 1.9761939;
(d) every star key correlation (k_a, k_b, k_ab, k_c, k_d, k_cd) is >= k_lo := 0.9491;
(e) every star key correlation is <= k_hi := 3 c_Theta - 2 k_lo = 0.9805107; so 1 - k_ab, 1 - k_cd >= 0.0194893.

```text
<1>1. (b) By (a), mu(e_uv, e_ua, e_ub) <= c_Theta, i.e. (sum_i N_i/2)^2 <= 6 c_Theta + 3, since sqrt((1 + e)/2) = N/2.
<1>2. (c) e >= rho (Lemma EC), so each N_i >= N0, and by (b) each N_i <= Sigma* - 2 N0; and e = N^2/2 - 1.
<1>3. (d) By the key-correlation formula, k_a is increasing in h_va, and Lemma MCC (normalised to G_uu = 1) gives
      h_va >= [(e_uv sqrt(d_v) + e_ua sqrt(d_a))/sqrt 2 - 1/2]/sqrt(d_v d_a). The same form holds for k_ab and, after dividing G
      by G_vv, for k_c, k_d, k_cd. So every key correlation is at least the minimum of
          k(l_1, l_2, e_1, e_2) = (1 + e_1 + e_2 + h)/(N(e_1) N(e_2)),   h = [(e_1 e^{l_1/2} + e_2 e^{l_2/2})/sqrt 2 - 1/2] e^{-(l_1+l_2)/2},
      over |l_i| <= 0.28903, e_i >= rho cosh(l_i/2), N(e_1) + N(e_2) <= Sigma* - N0 (by (b), the third N is >= N0). FC-THIN
      certifies this minimum is >= 0.9491.
<1>4. (e) By (a) and (d), each key correlation is <= 3 c_Theta - 2 k_lo.   QED
```

Let z := (â_u - â_v)/|â_u - â_v| (defined since e_uv <= e_max < 1), and alpha_u := <y_ua + y_ub, z>, alpha_v := <y_vc + y_vd, -z>.
Let k := C_ab, k' := C_cd, sigma := sum of the four cross entries of C, and c := sigma/(2 sqrt((1 + k)(1 + k'))), the cosine
of the angle between the *sum keys* S_u := y_ua + y_ub and S_v := y_vc + y_vd (|S_u|^2 = 2 + 2k).

**Lemma SUMKEY.** (uses the finite claim FC-THIN) For every m in N: alpha_u, alpha_v >= alpha_lo := 0.3731278 and

    c^2 <= c2* := (1 - 2 cos^2 theta*)^2 = 0.8643473,      cos theta* := alpha_lo/sqrt(2 (1 + k_hi)) = 0.1874796.

In words: on N each sum key leans toward its own side of the central edge by a fixed amount, so the two sum keys make an
angle of at least pi - 2 theta*.

```text
<1>1. <y_ua, â_u - â_v> = t_a/N_a with t_a := 1 - e_uv + e_ua - h_va >= rho - 1 > -1 (h_va <= 1, e_uv <= 1). If t_a >= 0 then
      t_a/N_a >= t_a/N_max; if t_a < 0 then t_a/N_a >= t_a/N0 >= t_a/N_max - (1/N0 - 1/N_max). Either way
      t_a/N_a >= t_a/N_max - (1/N0 - 1/N_max). Likewise for b.
<1>2. By the key-correlation formula, h_vx = k_x N_uv N_x - (1 + e_uv + e_ux) <= k_x N_uv N_max - (1 + e_uv + e_ux), and
      k_a + k_b <= 3 c_Theta - k_ab <= 3 c_Theta - k_lo. So h_va + h_vb <= (3 c_Theta - k_lo) N_uv N_max - (2 + 2 e_uv + e_ua + e_ub).
<1>3. t_a + t_b = 2 - 2 e_uv + e_ua + e_ub - (h_va + h_vb) >= 4 + 2 (e_ua + e_ub) - (3 c_Theta - k_lo) N_uv N_max
      >= 4 + 4 rho - (3 c_Theta - k_lo) N_uv N_max.
<1>4. |â_u - â_v| = sqrt(2 - 2 e_uv), so
      alpha_u >= A(e_uv) := { [4 + 4 rho - (3 c_Theta - k_lo) N(e_uv) N_max]/N_max - 2 (1/N0 - 1/N_max) }/sqrt(2 - 2 e_uv),
      and FC-THIN certifies A >= 0.3731278 on [rho, e_max]. The v-star is the mirror image: alpha_v has the same bound.
<1>5. cos angle(S_u, z) = alpha_u/|S_u| >= alpha_lo/sqrt(2 (1 + k_hi)) = cos theta*, so angle(S_u, z) <= theta* < pi/2; likewise
      angle(S_v, -z) <= theta*.
<1>6. Angles on the sphere obey the triangle inequality: pi = angle(z, -z) <= angle(S_u, z) + angle(S_u, S_v) + angle(S_v, -z),
      so angle(S_u, S_v) >= pi - 2 theta*, and c <= cos(pi - 2 theta*) = 1 - 2 cos^2 theta*. Also c > 0 (cross entries > 0), so
      c^2 <= (1 - 2 cos^2 theta*)^2.   QED
```

**Lemma CLO.** (uses the finite claim FC-THIN) On N every cross entry is >= C_lo := 0.8022.

```text
<1>1. C_xz = W_xz/(N_x N_z) with W_xz = e_uv + h_vx + h_uz + Ghat_xz, e_uv >= rho, and N_x N_z <= N_max^2 (Lemma THIN (c)).
<1>2. h_vx, h_uz >= 0.7827: the minimum over the ratio box of the normalised Lemma MCC bound with the cosines at their
      Lemma EC minima (the bound increases with the cosines). Ghat_xz >= 0.6346: the minimum over the box of the normalised
      Lemma D3 lower bound [(sqrt 2/12)(1 + d_v) + (sqrt 2/6)(d_x + d_z)]/sqrt(d_x d_z) (here d_z means G_zz/G_uu). Both by FC-THIN.
<1>3. So C_xz >= (rho + 2 (0.7827) + 0.6346)/N_max^2 = 3.142809/3.905342 = 0.80475 >= 0.8022.   QED
```

**Finite claim FC-THIN.** (computer-checked: bk15_klo_iv.py, bk15_const.py, bk15_box_iv.py: mpmath intervals at 25 digits)
(i) The minimum of k(l_1, l_2, e_1, e_2) over the domain of Lemma THIN <1>3 is >= 0.9491 (67,127 boxes closed
with k >= 0.9491 and 95,041 boxes infeasible; the relaxed
minimum is 0.949306 at l_1 = l_2 = 0.2044). (ii) A(e) >= 0.3731278 on [rho, e_max] (400 subintervals; the minimum is at
e = rho). (iii) The constants Sigma*, e_max, N_max, k_hi, cos theta*, c2*, computed in intervals from c_Theta's upper end.
(iv) The two box minima 0.7827 and 0.6346 of Lemma CLO.

**Lemma E0.** On N, e_0 := Ghat_uv <= e_max = 2 (sqrt(6 c_Theta + 3) - N0)^2 - 1 = 0.9526711763.

```text
<1>1. kbar_u <= c_Theta (Lemma NS) and kbar_u >= mu(e_0, e_ua, e_ub) >= ((s(e_0) + N0)^2 - 3)/6 with s(e) = sqrt((1 + e)/2), since
      e_ua, e_ub >= rho and s(rho) = N0/2.
<1>2. So s(e_0) <= sqrt(6 c_Theta + 3) - N0; square and solve.   QED
```

**The marginal and the window.** For m with sum m = 1, the *t_v-marginal* is mu(m) = (mu_-1, mu_0, mu_1) (the masses of the
classes t_v = -1, 0, 1). It determines V := G_vv = 2 mu_-1 + mu_0 + mu_1/2 and U := G_uv = sqrt 2 mu_-1 + mu_0 + mu_1/sqrt 2,
so e_0 = U/sqrt V depends on the marginal only. We use the coordinates of Lemma COORD (Section 7.5): (alpha, lam) in [0, 1]^2.

**Lemma WIN.** (uses the finite claim FC-WIN) The marginal of every m in N lies in W_win := [0.3, 0.7] x [0, 0.25].

```text
<1>1. On N, V = d_v lies in [0.81451, 1.22773] (Lemma R7) and e_0 <= e_max (Lemma E0). Both are conditions on the marginal.
<1>2. FC-WIN: every (alpha, lam) outside W_win violates one of them.   QED
```

**Finite claim FC-WIN.** (computer-checked: loc2k/src/bk16_margx.py, adaptive interval bisection, 181,286 boxes, 0 failures)
Every box of a cover of [0, 1]^2 minus W_win is shown,
in interval arithmetic, to have V outside [0.81451, 1.22773] or U/sqrt V > e_max on the whole box.

(The window cannot be narrowed much: at lam = 0 the V-constraint alone allows alpha down to 0.3467.)

**Lemma PI (the partner bound).** (uses the finite claim FC-PI) On N every cross entry satisfies C_xz >= pi_lo := 0.82.

```text
<1>1. C_xz = W_xz/(N_x N_z), W_xz = e_uv + h_vx + h_uz + Ghat_xz. Lower bounds in the five ratios: e_uv >= rho cosh(l_uv/2)
      (Lemma EC); h_vx, h_uz >= the normalised Lemma MCC bounds with the Lemma EC bounds inserted, and >= 0; Ghat_xz >= the
      normalised Lemma D3 lower bound.
<1>2. Upper bound of N_x: by Lemma NS and Lemma STAR (b), mu(e_uv, e_ua, e_ub) <= c_Theta, and with the other two cosines at
      their Lemma EC bounds this bounds e_ua, hence N_a; likewise at v.
<1>3. FC-PI: over the ratios in the R7 box intersected with the two star constraints of <1>2, the resulting lower bound
      of min_xz C_xz is >= 0.82.   QED
```

**Finite claim FC-PI.** (computer-checked) The interval branch and bound of Lemma PI closes at pi_lo = 0.82: a
depth-first interval code (outward rounding by nextafter, M_lo = 0.2765836068, c_Theta rounded up), 21,924,095 boxes, 0
failures, with 10,000 leaves re-closed in mpmath.iv at 40 digits (minimum 0.8200004). The weaker pi_lo = 0.815 is certified
twice (bk14_pilo2.py, 56,313,153 boxes processed: 27,789,174 closed leaves plus 367,403 outside N, 0 failures; and a
second code, 4,966,315 boxes). With 0.815 every later constant would change slightly; the certificates of
Section 8 were run at 0.82 (Lemma MONO-kappa, Section 8.5, explains why the local lemma certified at 0.815 stays valid).

### 7.5 The tree cut is a two-star problem

Think of the four boundary keys as unit vectors y_a, y_b, y_c, y_d (Lemma REAL: C(m) is their Gram matrix), and put
Y_u = y_a + y_b, Y_v = y_c + y_d, Y = Y_u + Y_v.

**Lemma GRAM.** a.C(m) = (a_ab - 1)(C_ab + C_cd) + |Y|^2/2 - 2.

PROOF. |Y_u|^2 = 2 + 2 C_ab, |Y_v|^2 = 2 + 2 C_cd, <Y_u, Y_v> = C_ac + C_ad + C_bc + C_bd. Expand |Y|^2.   QED

In words: the tree cut is the pair correlations plus half the squared length of the key sum.

**The star key correlations and slacks.** Put t := k_a + k_b + k_c + k_d = <Y, y_uv> (since <y_x, y_uv> = k_x),
r := (a_ab - 1)/(4 mu0) = 0.90857, and the *star slacks*

    s_u := r k_ab + k_a + k_b - (r + 2) mu0,        s_v := r k_cd + k_c + k_d - (r + 2) mu0.

At the tree every k equals mu0, so s_u = s_v = 0.

**Lemma PROJ2 (the exact two-star identity).** Let P be the orthogonal projector onto span(â_u, â_v), e_0 = Ghat_uv,
p_u := <Y, â_u>, p_v := <Y, â_v>. For every m in the cone with e_0 < 1,

    a.C(m) - K0 = 4 mu0 (s_u + s_v) + |Y - PY|^2/2 + (p_u - p_v)^2/(4 (1 - e_0)) + (t - 4 mu0)^2/2.

(For e_0 = 1 the projection is onto a line and the third term is absent.) In particular a.C(m) - K0 >= 4 mu0 (s_u + s_v).

```text
<1>1. |Y|^2 = |PY|^2 + |Y - PY|^2 (Pythagoras).
<1>2. With Gram matrix [[1, e_0], [e_0, 1]] of (â_u, â_v): |PY|^2 = (p_u^2 + p_v^2 - 2 e_0 p_u p_v)/(1 - e_0^2)
      = (p_u + p_v)^2/(2(1 + e_0)) + (p_u - p_v)^2/(2(1 - e_0)).
<1>3. y_uv = (â_u + â_v)/sqrt(2 + 2 e_0), so t = <Y, y_uv> = (p_u + p_v)/sqrt(2 + 2 e_0), and the first term of <1>2 is
      (p_u + p_v)^2/(2(1 + e_0)) = t^2.
<1>4. So |Y|^2/2 = t^2/2 + (p_u - p_v)^2/(4 (1 - e_0)) + |Y - PY|^2/2.
<1>5. t^2/2 = 4 mu0 t - 8 mu0^2 + (t - 4 mu0)^2/2 (complete the square).
<1>6. By Lemma GRAM, a.C - K0 = (a_ab - 1)(k_ab + k_cd) + 4 mu0 t - 8 mu0^2 - 2 - K0 + [the three squares], using C_ab = k_ab,
      C_cd = k_cd. And (a_ab - 1)(k_ab + k_cd) + 4 mu0 t = 4 mu0 [r (k_ab + k_cd) + k_a + k_b + k_c + k_d].
<1>7. (the tree identity) 8 mu0^2 - 2 mu0 - 2 - 4 s = 0 exactly: 8 mu0^2 = (9 + 4 sqrt 2)/2, 2 mu0 = (1 + 2 sqrt 2)/2,
      4 s = 2 + sqrt 2, and (9 + 4 sqrt 2)/2 - (1 + 2 sqrt 2)/2 - 2 - 2 - sqrt 2 = 0. Hence
      8 mu0^2 + 2 + K0 = 8 mu0^2 + 2 + 2 a_ab mu0 + 4 s = 2 (a_ab - 1) mu0 + 16 mu0^2 = 4 mu0 . 2 (r + 2) mu0.
<1>8. Substituting <1>7 into <1>6, the non-square part is 4 mu0 [r (k_ab + k_cd) + t - 2 (r + 2) mu0] = 4 mu0 (s_u + s_v).  QED
```

**Stars and profiles.** The *u-profile* of a type t is (t_v, t_a, t_b), its *v-profile* (t_v, t_c, t_d). There are 20
u-profiles: the 19 triples in {-1, 0, 1}^3 with minimum -1, and the special (1, 1, 1); by t_v-class, 9 with t_v = -1, 5 with
t_v = 0, 6 with t_v = 1. Likewise 20 v-profiles. The 115 generic types are exactly the pairs (u-profile, v-profile) of
admissible profiles with equal t_v (45 + 25 + 45), and the 8 special types pair a special profile with four others; so
T123 lies in the 133-element fibre product (the 10 missing pairs are a special profile against a profile with a zero entry). For m in the cone, its *u-star* m_u is the vector of masses of the u-profiles (m_u(pi) = sum of m_t over t with
u-profile pi), and its *v-star* m_v likewise. Both have marginal mu(m).

**Theorem TWO-STAR.**
(i) The u-side entries (e_0, e_ua, e_ub, h_ab, h_va, h_vb) depend on m only through m_u, the v-side entries through m_v.
    Hence s_u is a function of m_u alone and s_v of m_v alone.
(ii) (The mirror.) For a v-star m_v let m_v^sigma be the u-star with m_v^sigma(-t_v, t_c - t_v, t_d - t_v) =
     2^{-t_v} m_v(t_v, t_c, t_d). Then s_v(m_v) = s_u(m_v^sigma), and the marginal of m_v^sigma is mu^sigma := (mu_1/2, mu_0, 2 mu_-1).
(iii) Hence, with sigma_u(mu) := inf { s_u(m_u) : m_u >= 0 on the 20 u-profiles with marginal mu },
     a.C(m) - K0 >= 4 mu0 [ sigma_u(mu(m)) + sigma_u(mu(m)^sigma) ].

```text
<1>1. (i) For x, y in {u, v, a, b}, g_t(x) g_t(y) depends on t only through its u-profile, so G_xy(m) = sum_pi m_u(pi) g_pi(x) g_pi(y);
      the u-side Ghat entries are ratios of these. Likewise at v. By the key-correlation formula, k_a, k_b, k_ab are
      functions of the u-side entries.
<1>2. (ii) Relabel u <-> v, a <-> c, b <-> d. For a v-profile pi = (t_v, t_c, t_d), (g_pi(v), g_pi(u), g_pi(c), g_pi(d)) =
      2^{-t_v/2} g_{pi'}(u, v, a, b) with pi' = (-t_v, t_c - t_v, t_d - t_v), which is a u-profile ((E) gives |t_c - t_v| <= 1;
      min(0, t_c, t_d) = t_v - 1 gives min(-t_v, t_c - t_v, t_d - t_v) = -1; the special (-1, 0, 0) goes to (1, 1, 1)). So the Gram
      matrix of (v, u, c, d) under m_v equals the Gram matrix of (u, v, a, b) under m_v^sigma, entry by entry; the v-side
      entries of m_v are the u-side entries of m_v^sigma, and s_v(m_v) = s_u(m_v^sigma). Class t_v = k goes to class -k with
      factor 2^{-k}.
<1>3. (iii) Lemma PROJ2, then (i) and (ii); m_u and m_v^sigma are nonnegative vectors on the u-profiles with marginals
      mu(m) and mu(m)^sigma.   QED
```

In words: the tree cut is at least a fixed multiple of the sum of two star slacks; the two stars see each other only through
the three class masses; and the v-star is the u-star in a mirror. The tree marginal mu_T = (1/3, 0, 2/3) is a fixed point of
the mirror.

**Lemma COORD.** For (alpha, lam) in [0, 1]^2 put

    mu(alpha, lam) := (alpha (1 - lam), lam, 2 (1 - alpha)(1 - lam)) / ((1 - lam)(2 - alpha) + lam).

(a) Every nonzero marginal is a positive multiple of some mu(alpha, lam). (b) mu(alpha, lam)^sigma is a positive multiple of
mu(1 - alpha, lam). (c) mu_T = mu(1/2, 0).

```text
<1>1. Let n := mu_-1 + mu_1/2. (a) If n > 0 take alpha = mu_-1/n and lam = mu_0/(mu_0 + n); then mu(alpha, lam) is a multiple of
      mu/n. If n = 0, mu is a multiple of mu(alpha, 1) for any alpha.
<1>2. (b) The mirror keeps mu_0 and maps n to mu_1/2 + mu_-1 = n, and its first coordinate is mu_1/2 = n - mu_-1; so its alpha
      is 1 - alpha and its lam is lam. (c) mu(1/2, 0) = (1/2, 0, 1)/(3/2).   QED
```

So the mirror reflects alpha about 1/2, and lam (the share of class 0) is shared by the two stars. We identify a marginal
with its (alpha, lam).

**Lemma MIRB (the mirror point).** For m in the cone let m^sigma := sigma_uv(m) (Lemma SYMC). Then
C(m^sigma) is C(m) with ab <-> cd and ad <-> bc; RB(m^sigma) = RB(m), delta(m^sigma) = delta(m), E_u(m^sigma) = E_v(m); the
u-star of m^sigma is (up to normalisation) the mirror m_v^sigma of the v-star of m, and its marginal is mu(m)^sigma. In
particular N is mirror-invariant: m is in N iff m^sigma is, and p is a marginal of N iff p^sigma is.

```text
<1>1. Lemma SYMC gives the action on C. S_u and S_v exchange, so RB is unchanged. D = C_ac C_bd - C_ad C_bc is unchanged
      (ac, bd fixed, ad <-> bc), and the partner pairs are mapped to themselves, so delta is unchanged.
<1>2. G(m^sigma) = Pi G(m) Pi^T with Pi = (u v)(a c)(b d), so the u-side entries of m^sigma are the v-side entries of m, and the
      u-star of m^sigma has masses 2^{-t_v} m_v at the relabelled profiles: this is m_v^sigma (Theorem TWO-STAR (ii)).   QED
```

### 7.6 How (R-ii*) on N will be proved

**The kill inequality.** For m in the cone,

    (K)   a.C(m) - K0 >= delta(m).

**Lemma KW (the kill as witness).** If (K) holds at m and the kill C'(m) is PSD, then (R-ii*) holds at m.

PROOF. The kill is PSD with D = 0, so it is in RC; it is below C(m); and a.C'(m) = a.C(m) - delta(m) >= K0 by (K). So it is a
witness.   QED

So (R-ii*) on N follows from two statements:
- **Theorem K-N** (Section 8): (K) holds at every m in N.
- **Theorem PSD-N** (Section 9): the kill is PSD at every m in N.

Why is (K) hard? Every witness has a.C(m) >= a.C'' >= K0, so (K) contains the plain tree cut a.C(m) >= K0 on N. That
statement resisted every hand proof: a star-by-star chain that projects Y onto the single key y_uv and splits the stars
without the coupling falls short by 6.9e-6. Lemma PROJ2 keeps everything as
explicit squares, and Theorem TWO-STAR keeps the coupling through the marginal. (K)
must be proved with the coupling between the two stars kept.

---

## 8. The kill inequality on N

This section proves

**Theorem K-N.** (uses the finite claims FC-PI, FC-SEP, FC-LOC2K, FC-LOC2KC, FC-Q, FC-EMIN and FC-GSTAR,
besides those of Section 7) For every m in N, a.C(m) - K0 >= delta(m).

The plan has three steps. First (Sections 8.1-8.4), Theorem SEP bounds the kill delta(m) by a sum of two one-star terms,
kappa_S [Delta_u(u-star) + Delta_u(mirror star)]. With Theorem TWO-STAR this reduces (K) to one inequality, CKS*, about a
single star at two mirror marginals (Section 8.5, Proposition RED). Second (Sections 8.6-8.9), CKS* in a slightly weaker
"capped" form is reduced to finitely many statements about linear programs on boxes of "star slices". Third (Sections
8.10-8.11), those statements are the records of two certified branch and bounds.

Why can't (K) be split star by star directly? Because delta depends on how the masses of the two stars are paired within
each t_v-class (through the four distance-3 entries), and the two stars do not determine that pairing. The decomposition of
Section 8.1 isolates the pairing in sums that can be split.

### 8.1 The kill as a determinant of a sum of rank-one matrices

Normalise sum m = 1 (so G_uu = 1) and put n_x := G_xx^{-1/2} (n_u = 1). For a type t put

    P_t := (1 + n_a g_t(a), 1 + n_b g_t(b)),          Q_t := (n_v g_t(v) + n_c g_t(c), n_v g_t(v) + n_d g_t(d)).

P_t depends only on the u-profile of t, Q_t only on its v-profile. The unit row of x has entries sqrt(m_t) g_t(x) n_x, so

    W_xz = <â_u + â_x, â_v + â_z> = sum_t m_t P_t,x Q_t,z,        i.e.  W = sum_t m_t P_t Q_t^T

(a 2 x 2 matrix, rows a, b, columns c, d), and C_xz = W_xz/(N_x N_z) with N_x = |â_u + â_x|, N_z = |â_v + â_z|. Hence

    delta(m) = |D(W)| / (N_a N_b N_c N_d max(partner pair)).                                            (8.1)

For p, q in R^2 write p ^ q := p_1 q_2 - p_2 q_1. For 2 x 2 matrices the *mixed determinant*
[A, B] := A_11 B_22 + A_22 B_11 - A_12 B_21 - A_21 B_12 is bilinear and

    det(A + B) = det A + det B + [A, B],          [p q^T, p' q'^T] = (p ^ p')(q ^ q')                    (8.2)

(both one-line expansions).

### 8.2 The kill decomposition

The tree types put the u-profile pi_0 := (-1, 1, 1) in class t_v = -1 and the v-profile e := (1, 2, 2) in class t_v = 1;
these are the *reference profiles*. Put P_0 := P(pi_0), Q_1 := Q(e), and for a u-profile pi and a v-profile pi'

    a(pi) := P_0 ^ P(pi),     b(pi') := Q_1 ^ Q(pi'),
    nu := (-1, 1)/(P_0,a + P_0,b),     nu' := (-1, 1)/(Q_1,c + Q_1,d)        (so P_0 ^ nu = 1 = Q_1 ^ nu'),
    phi(pi) := P(pi) ^ nu,     psi(pi') := Q(pi') ^ nu'.

So a(pi_0) = 0, b(e) = 0, phi(pi_0) = psi(e) = 1, and phi, psi > 0. With the class aggregates F := sum_{t_v = 1} m_t P_t and
G := sum_{t_v = -1} m_t Q_t, put

    X_u := F ^ P_0,   X_v := Q_1 ^ G,   phi := F ^ nu,   psi := G ^ nu'     (face imbalances and face weights),
    P~_t := P_t - P_0 if t_v = -1, else P_t;      Q~_t := Q_t - Q_1 if t_v = 1, else Q_t,
    J := sum_t m_t a(pi_t) b(pi'_t),   M_u := sum_t m_t (P~_t ^ nu) b(pi'_t),   M_v := sum_t m_t a(pi_t)(Q~_t ^ nu'),
    E := sum_t m_t P~_t Q~_t^T.

**Lemma KD.** D(W) = X_u X_v + (phi + psi) J + X_u M_u - X_v M_v + det E.

In words: the cross determinant is the product of the two face imbalances, plus the face weights times one joint sum J
that vanishes on the tree face, plus mixed terms, plus a determinant that is second order off the face. The only joint sums
are J, M_u, M_v and E; each is a sum over types of a u-profile function times a v-profile function.

```text
<1>1. W = F Q_1^T + P_0 G^T + E: in class 1 write Q_t = Q_1 + Q~_t, in class -1 write P_t = P_0 + P~_t, in class 0 change
      nothing, and collect.
<1>2. det(F Q_1^T + P_0 G^T) = (F ^ P_0)(Q_1 ^ G) = X_u X_v by (8.2), since rank-one matrices have det 0.
<1>3. [F Q_1^T + P_0 G^T, E] = sum_t m_t [(F ^ P~_t)(Q_1 ^ Q~_t) + (P_0 ^ P~_t)(G ^ Q~_t)], and Q_1 ^ Q~_t = b(pi'_t),
      P_0 ^ P~_t = a(pi_t) in every class.
<1>4. (three-term identity) For p, q, r, w in R^2: (p ^ q)(r ^ w) + (q ^ r)(p ^ w) + (r ^ p)(q ^ w) = 0 (alternating and
      trilinear in (p, q, r) in a two-dimensional space). With r = P_0 and w = nu (P_0 ^ nu = 1) this gives
      F ^ P = phi a(P) + X_u (P ^ nu) for every P (a(P) := P_0 ^ P); likewise G ^ Q = psi b(Q) - X_v (Q ^ nu').
<1>5. Substitute <1>4 into <1>3 with P = P~_t, Q = Q~_t: the mixed determinant is (phi + psi) J + X_u M_u - X_v M_v.
<1>6. det W = <1>2 + <1>5 + det E by (8.2).   QED
```

### 8.3 Star scalars and the mirror

For a u-star m (normalised, sum 1) with ratios d_a = G_aa, d_b = G_bb (so n_a, n_b), the vectors P(pi), P_0, the numbers
a(pi), phi(pi) and P^(pi) := P(pi) - P_0 are constants of the star slice. Write U_k for the set of u-profiles of class k.
The *star scalars* of m are:

| scalar | definition |
|---|---|
| phi | sum_{U_1} m(pi) phi(pi) (face weight) |
| X | sum_{U_1} m(pi) P(pi) ^ P_0 (face imbalance) |
| Abar | max_{pi in U_1} |a(pi)| |
| K | sum_{U_-1} m(pi) |a(pi)| (first-order kill mass) |
| A0 | sum_{U_0} m(pi) a(pi)^2 |
| K' | sum_{U_-1} m(pi) |phi(pi) - 1| |
| Phibar, Phi0 | max_{U_1} phi(pi);  sum_{U_0} m(pi) phi(pi)^2 |
| w11, w10 | max |P(pi) ^ P(pi')| over U_1 x U_1, and over U_1 x U_0 |
| Y | sum_{pi in U_-1} m(pi) max_{pi' in U_1} |P(pi') ^ P^(pi)| |
| R | sum over ordered pairs (pi, pi') in U_-1 x U_-1 of m(pi) m(pi') |P^(pi) ^ P^(pi')| |
| R0 | sum over ordered pairs in U_0 x U_0 of m m' (P ^ P')^2 |
| Rm0 | sum over (pi, pi') in U_-1 x U_0 of m(pi) m(pi') |P^(pi) ^ P(pi')| |

For the v-star m_v of m, let V := G_vv = d_v (in the normalisation sum m = 1, V = 2 mu_-1 + mu_0 + mu_1/2 depends only on the
marginal), omega := sqrt V, and let m^sigma be the *mirror star*: the mirror image m_v^sigma of Theorem TWO-STAR (ii),
normalised to total mass 1 (its total before normalising is sum m_v(pi') g_v(pi')^2 = G_vv = V). Its marginal is mu^sigma
(normalised), and V(mu^sigma) = 1/V(mu). Scalars of the u-star carry a subscript u, those of the mirror star a subscript s.

**Lemma MIR.** For a v-profile pi' with image u-profile pi'' and g_v := g_{pi'}(v):

    Q(pi') = (g_v/omega) P^sigma(pi''),      m_v(pi') = V m^sigma(pi'')/g_v^2,

where P^sigma is the P-vector of the mirror star. Consequently X_v = -X_s/2, psi = V phi_s, and every v-side sum or maximum
that enters Lemma JB below is the corresponding mirror-star scalar times a factor depending only on V: max_{V_-1}|b| =
Abar_s/V; sum_{V_1} m_v |b| = K_s; sum_{V_0} m_v b^2 = A0_s/(2V); sum_{V_1} m_v |psi - 1| = 2V K'_s; max_{V_-1} psi = 2 Phibar_s;
sum_{V_0} m_v psi^2 = 2V Phi0_s; and the v-side versions of w11, R, R0, Y, w10, Rm0 equal (2/V) w11_s, 2V R_s, R0_s, 2 Y_s,
(sqrt 2/V) w10_s, sqrt 2 V Rm0_s. (V_k is the set of v-profiles of class k; class k maps to mirror class -k.)

```text
<1>1. (masses) The unnormalised image has mass m_v(pi') 2^{-t_v} = m_v(pi') g_v^2 at pi'', with total V; normalising gives
      m^sigma(pi'') = m_v(pi') g_v^2/V.
<1>2. (vectors) The image's Gram matrix of (u', v', a', b') is that of (v, u, c, d) under m_v (Theorem TWO-STAR (ii)); after
      normalising the masses by 1/V, the mirror star's n-values are n^sigma_{u'} = omega n_v = 1 and n^sigma_{a'} = omega n_c,
      n^sigma_{b'} = omega n_d. Its generator at pi'' has g(a') = g_c/g_v. So P^sigma_a(pi'') = 1 + omega n_c g_c/g_v, and
      Q_c(pi') = n_v g_v + n_c g_c = (g_v/omega)(omega n_v + omega n_c g_c/g_v) = (g_v/omega) P^sigma_a(pi''). Likewise for d.
<1>3. (consequences) The reference e = (1, 2, 2) maps to pi_0 with g_v = 2^{-1/2}, so Q_1 = (2^{-1/2}/omega) P^sigma_0, hence
      b(pi') = (g_v/(sqrt 2 V)) a_s(pi'') and nu' = sqrt 2 omega nu_s. In class 1 (g_v = 2^{-1/2}, image class -1) this gives
      m_v |b| = m^sigma |a_s| summed to K_s; in class -1 (g_v = sqrt 2, image class 1), |b| = |a_s|/V, so max |b| = Abar_s/V;
      in class 0 (g_v = 1), b = a_s/(sqrt 2 V) and m_v = V m^sigma, so sum m_v b^2 = A0_s/(2V). For the face terms,
      G = sum_{V_-1} m_v Q = (omega/sqrt 2) F_s, so X_v = Q_1 ^ G = (1/2) P^sigma_0 ^ F_s = -X_s/2 and psi = G ^ nu' = V phi_s.
      Per profile, psi(pi') = sqrt 2 g_v phi_s(pi''), which is phi_s, 2 phi_s, sqrt 2 phi_s in classes 1, -1, 0; this gives the
      three psi-sums. For the det E scalars substitute Q~ = Q - Q_1 = (2^{-1/2}/omega) P^sigma^ in class 1 and the mass rule
      <1>1 into each double sum; for example sum over V_1^2 of m_v m_v' |Q~ ^ Q~'| = (4 V^2)(1/(2V)) R_s = 2V R_s, and
      sum over V_1 x V_0 of m_v m_v' |Q~ ^ Q'| = (2 V^2)(2^{-1/2}/V) Rm0_s = sqrt 2 V Rm0_s. The others are the same computation.
      QED
```

### 8.4 The joint bound and the split

**Lemma JB.** For every m in the cone, with u the u-star and s the mirror star,

    |D(W)| <= B := |X_u X_s|/2
       + phi_u Abar_u K_s + phi_s Abar_s K_u + V phi_s Abar_u K_s + phi_u Abar_s K_u/V + (phi_u/omega + omega phi_s)(A0_u + A0_s)/(2 sqrt 2)
       + |X_u| Phibar_u K_s + |X_s| Phibar_s K_u + |X_u| Abar_s K'_u/V + V |X_s| Abar_u K'_s
       + |X_u| (Phi0_u + A0_s)/(2 sqrt 2 omega) + omega |X_s| (A0_u + Phi0_s)/(2 sqrt 2)
       + V w11_u R_s + w11_s R_u/V + (R0_u + R0_s)/4 + Y_u^2 + Y_s^2 + sqrt 2 V w10_u Rm0_s + (sqrt 2/V) w10_s Rm0_u.

B is symmetric under (u <-> s, V <-> 1/V). Line by line: the face term, the first-order kill J, the mixed terms, det E.

```text
<1>1. (face) |X_u X_v| = |X_u||X_s|/2 by Lemma MIR.
<1>2. (J) phi + psi = phi_u + V phi_s >= 0, so |(phi + psi) J| <= (phi + psi) sum_t m_t |a(pi_t)||b(pi'_t)|. By class:
      class 1: |a| <= Abar_u and the class-1 type sum of m_t |b| is sum_{V_1} m_v |b| = K_s; class -1: |b| <= Abar_s/V and the
      type sum of m_t |a| is K_u; class 0: b = a_s/(sqrt 2 V) and per type |a||a_s| <= (omega a^2 + a_s^2/omega)/2 (AM-GM), and
      sum m_t a_s^2 = V A0_s, so the class-0 sum is <= (omega A0_u + V A0_s/omega)/(2 sqrt 2 V) = (A0_u + A0_s)/(2 sqrt 2 omega).
      Multiply out with phi_u + V phi_s (V/omega = omega).
<1>3. (mixed) P~_t ^ nu = phi(pi_t) in classes 1, 0 and phi(pi_t) - 1 in class -1 (P_0 ^ nu = 1). So |M_u| <= Phibar_u K_s
      (class 1) + (Abar_s/V) K'_u (class -1) + (Phi0_u + A0_s)/(2 sqrt 2 omega) (class 0, AM-GM with weight omega as in <1>2).
      For |M_v|, symmetrically (class -1: max psi = 2 Phibar_s times K_u; class 1: Abar_u times 2V K'_s; class 0: weight
      1/omega), and |X_v| = |X_s|/2.
<1>4. (det E) Split E = E_1 + E_-1 + E_0 by class; det E = det E_1 + det E_-1 + det E_0 + [E_1, E_-1] + [E_1, E_0] + [E_-1, E_0],
      each a double sum over type pairs of m_t m_t' (P~ ^ P~')(Q~ ^ Q~') (Cauchy-Binet: det of sum_t m_t p_t q_t^T is
      (1/2) sum_{t,t'} m_t m_t' (p_t ^ p_t')(q_t ^ q_t')).
      det E_1 <= (w11_u/2) sum_{V_1^2} m_v m_v' |Q~ ^ Q~'| = V w11_u R_s;  det E_-1 <= w11_s R_u/V (symmetric);
      det E_0 <= (1/2) sqrt(R0_u R0_s) <= (R0_u + R0_s)/4 (Cauchy-Schwarz over type pairs with weights m_t m_t');
      [E_1, E_-1] <= (sum_{U_-1} m max_{U_1} |P ^ P^|)(sum_{V_1} m_v max_{V_-1} |Q~ ^ Q|) = Y_u (2 Y_s) <= Y_u^2 + Y_s^2;
      [E_1, E_0] <= w10_u sum_{V_1 x V_0} m_v m_v' |Q~ ^ Q'| = sqrt 2 V w10_u Rm0_s;  [E_-1, E_0] symmetric.
      (Grouping a type sum by profiles is valid because each summand is nonnegative and depends on the profiles only.)
<1>5. Add <1>1-<1>4 and use Lemma KD.   QED
```

**Lemma HINGE.** If Z >= 0, 0 <= Y <= Ybar and tau >= 0, then Z Y <= tau Y + Ybar (Z - tau)_+.

PROOF. If Z <= tau, Z Y <= tau Y; otherwise Z Y = tau Y + (Z - tau) Y <= tau Y + (Z - tau) Ybar.   QED

In words: a product of a scalar of one star with a scalar of the other is split into a share for each; the threshold tau,
fixed in advance, decides who pays. Where Z is a face imbalance the AM-GM split Z Y <= (lam Z^2 + Y^2/lam)/2 is used instead,
so that the charge is quadratic, matching the quadratic slack.

**Thresholds and sup constants.** For a marginal q let face(q) be the *face star* (class -1 mass on pi_0, class 1 split
equally between (1, -1, 1) and (1, 1, -1), class 0 split equally between (0, 1, -1) and (0, -1, 1)), and with eps = 0.02 put

    tau_U(q) = (1 + eps)[phi Abar](face(q)),  tau_A(q) = (1 + eps) Abar(face(q)),  tau_phi(q) = (1 + eps) phi(face(q)),
    tau_w(q) = (1 + eps) w11(face(q)),  tau_10(q) = (1 + eps) w10(face(q)).

Let a_m1, a_1, a2_0, phm1, ph1, wPh, wPhP0 be upper bounds, over the 0.29-box |log d_a|, |log d_b| <= 0.29 and over the profiles
of the indicated classes, of |a| (class -1), |a| (class 1), a^2 (class 0), |phi - 1| (class -1), phi (class 1), |P^ ^ P^'|
(class -1 pairs) and |P^ ^ P'| (class -1 x class 0). Then for every star with marginal q and ratios in the box:

    K <= Kbar(q) := q_-1 a_m1;   A0 <= A0bar(q) := q_0 a2_0;   phi K/V(q) <= Zphi(q) := q_1 ph1 q_-1 a_m1/V(q);
    |X| K'/V(q) <= ZX(q) := q_1 a_1 q_-1 phm1/V(q);   R/V(q) <= ZR(q) := q_-1^2 wPh/V(q);
    sqrt 2 Rm0/V(q) <= ZRm(q) := sqrt 2 q_-1 q_0 wPhP0/V(q)

(each a class mass times a profile maximum; |X| <= q_1 a_1 since X = -sum_{U_1} m a).

**Finite claim FC-SEP (the profile maxima).** (computer-checked: bk14_supcert.py, mpmath intervals on a 48 x 48 cover of the
box, upper ends rounded up; the functions are bilinear or
linear-fractional in (n_a, n_b), so their maxima over the box are at its four vertices) a_m1 = a_1 <= 1.5035305,
a2_0 <= 2.2606040, phm1 <= 0.4537150, ph1 <= 1.4537150, wPh <= 0.6803371, wPhP0 <= 2.1798184, and w11, w10 <= 3.6574014
(exact vertex values 1.48565716, 2.20717721, 0.44977654, 1.44977654, 0.66821374, 2.15387091, 3.63952807: the certified
values are 0.27-2.4% above them).

**Definition (Delta_u).** For a u-star m at marginal p (q := p^sigma, V := V(p), omega := sqrt V, R22 := 2 sqrt 2, lam := 0.6),
all scalars being those of m:

    Delta_u(m; p) := X^2/4                                                                           [face]
        + tau_U(q) K + tau_A(q) phi K/V + (phi/omega + omega tau_phi(q)) A0/R22                    [J, own share]
        + lam Phibar^2 X^2/2 + K^2/(2 lam) + lam X^2/(2 R22 omega) + omega A0^2/(2 R22 lam)          [mixed, AM-GM]
        + tau_A(q) |X| K'/V + |X| Phi0/(R22 omega)                                                 [mixed, own]
        + tau_w(q) R/V + R0/4 + Y^2 + (sqrt 2/V) tau_10(q) Rm0                                     [det E]
        + Kbar(q) (phi Abar - tau_U(p))_+ + (Zphi(q) + ZX(q)) (Abar - tau_A(p))_+
        + A0bar(q) (phi - tau_phi(p))_+/(R22 omega) + ZR(q) (w11 - tau_w(p))_+ + ZRm(q) (w10 - tau_10(p))_+.   [hinges]

The tau(q) coefficients stand in for the other star's scalars; the hinge lines are what this star pays when its own
coefficients exceed their thresholds. Every term is a product of nonnegative factors, so Delta_u >= 0.

**Lemma SPLIT.** (uses the finite claim FC-SEP) For every m in the cone whose u-star and mirror star have their ratios in the
0.29-box, B <= Delta_u(m_u; mu) + Delta_u(m^sigma; mu^sigma).

```text
<1>1. Face: |X_u X_s|/2 <= X_u^2/4 + X_s^2/4.
<1>2. phi_u Abar_u K_s: HINGE with Z = phi_u Abar_u, tau = tau_U(mu), Y = K_s <= Kbar(mu^sigma). The mirror star gets
      tau_U(mu) K_s (its "tau_U(q) K"); the u-star gets Kbar(q)(phi_u Abar_u - tau_U(mu))_+. Twin: phi_s Abar_s K_u.
<1>3. V phi_s Abar_u K_s: HINGE with Z = Abar_u, tau = tau_A(mu), Y = V phi_s K_s <= Zphi(q) (V(mu) = 1/V(q)). The mirror gets
      tau_A(mu) phi_s K_s/V(q); the u-star gets Zphi(q)(Abar_u - tau_A(mu))_+. Twin: phi_u Abar_s K_u/V.
<1>4. (phi_u/omega + omega phi_s)(A0_u + A0_s)/R22: phi_u A0_u/omega stays with u; phi_u A0_s/omega: HINGE on phi_u (tau_phi(mu),
      Y = A0_s <= A0bar(q)). Twins likewise (omega -> 1/omega).
<1>5. |X_u| Phibar_u K_s <= (lam Phibar_u^2 X_u^2 + K_s^2/lam)/2 and |X_u| A0_s/(R22 omega) <= (lam X_u^2 + A0_s^2/lam)/(2 R22 omega)
      (AM-GM). With the twins, the u-star collects lam Phibar^2 X^2/2 + K^2/(2 lam) + lam X^2/(2 R22 omega) + omega A0^2/(2 R22 lam).
<1>6. |X_u| Abar_s K'_u/V: HINGE with Z = Abar_s (tau_A(q)), Y = |X_u| K'_u/V(mu) <= ZX(mu): the u-star keeps tau_A(q)|X_u| K'_u/V,
      the mirror pays ZX(mu)(Abar_s - tau_A(q))_+. The twin V |X_s| Abar_u K'_s gives the u-star ZX(q)(Abar_u - tau_A(mu))_+.
      |X_u| Phi0_u/(R22 omega) stays.
<1>7. det E: V w11_u R_s: HINGE on w11_u (tau_w(mu), Y = V R_s <= ZR(q)); sqrt 2 V w10_u Rm0_s: HINGE on w10_u (tau_10(mu),
      Y <= ZRm(q)); (R0_u + R0_s)/4 and Y_u^2 + Y_s^2 split as written; twins likewise.
<1>8. Collect. By the symmetry of B under (u <-> s, V <-> 1/V), the shares collected by the mirror star are exactly
      Delta_u(m^sigma; mu^sigma).   QED
```

### 8.5 Theorem SEP and the reduction to one star

Put

    kappa_S := 1/(N0^4 pi_lo) = 0.0807729462517...   (pi_lo = 0.82; N0^4 = 4 (1 + 2 sqrt 2/3)^2).

The certificates carry kappa_S as the interval [0.08077294625999999, 0.08077294626], whose upper end is >= kappa_S; by Lemma
MONO-kappa below, a larger multiplier only weakens the bound, so this is safe.

**Theorem SEP (the kill is separable on N).** (uses the finite claims FC-PI and FC-SEP) For every m in N,

    delta(m) <= kappa_S [ Delta_u(m_u; mu) + Delta_u(m^sigma; mu^sigma) ].

```text
<1>1. By (8.1), delta = |D(W)|/(N_a N_b N_c N_d . max partner).
<1>2. N_x >= N0 for every x (edge cosines >= rho, Lemma EC), and max partner >= min cross entry >= pi_lo (Lemma PI).
<1>3. On N the u-star and the mirror star have their ratios in the R7 square (Lemma R7; the mirror star's ratios are
      d_c/d_v and d_d/d_v), which lies in the 0.29-box.
<1>4. |D(W)| <= B (Lemma JB) <= Delta_u(m_u; mu) + Delta_u(m^sigma; mu^sigma) (Lemma SPLIT, using <1>3).   QED
```

**Lemma MONO-kappa.** Delta_u >= 0, so any statement "4 mu0 s_u - kappa Delta_u >= T" proved at one kappa holds at
every smaller kappa.

**The star domain and the star function.** For a marginal p let S(p) be the set of u-stars m (sum 1) with marginal p, ratios
in the R7 square (|log d_a|, |log d_b| <= 0.28903), and the *NS row* k_a + k_b + k_ab <= 3 c_Theta (Lemma NS). Put

    Net(m) := 4 mu0 s_u(m) - kappa_S Delta_u(m; p),         tau(p) := inf { Net(m) : m in S(p) }

(tau(p) = +oo if S(p) is empty).

**CKS\*.** For every marginal p of N: tau(p) + tau(p^sigma) >= 0.

**Proposition RED.** Theorem SEP and CKS* imply (K) on N.

```text
<1>1. Let m be in N with marginal mu. Its u-star lies in S(mu), and its mirror star m^sigma in S(mu^sigma): the ratios by Lemma R7,
      the NS row by Lemma NS at both stars, and s_v(m_v) = s_u(m^sigma) by Theorem TWO-STAR (ii) (s_u is 0-homogeneous).
<1>2. By Lemma PROJ2 and TWO-STAR, a.C - K0 >= 4 mu0 s_u(m_u) + 4 mu0 s_u(m^sigma).
<1>3. By the definition of tau, 4 mu0 s_u(m_u) >= kappa_S Delta_u(m_u; mu) + tau(mu), and likewise for the mirror star.
<1>4. So a.C - K0 >= kappa_S [Delta_u(m_u; mu) + Delta_u(m^sigma; mu^sigma)] + tau(mu) + tau(mu^sigma) >= delta(m) + 0 by SEP and CKS*.
      QED
```

So (K) on N is one inequality about one function of the marginal and its mirror alpha -> 1 - alpha.

### 8.6 How a box of star slices is closed

A *star slice* is fixed by a marginal p = (alpha, lam) and the two ratios (d_a, d_b) of the u-star. We use the *pm
coordinates*

    dp := d_a + d_b - c_d(p),     dm := d_a - d_b,     c_d(p) := 5/2 - (3/2) mu_-1(p),

because dp = 0 is exactly the set of slices that contain a star supported on the face profiles: every star supported on
{pi_0, (1, -1, 1), (1, 1, -1), (0, 1, -1)} with marginal p has d_a + d_b = mu_-1 + (5/2)(mu_0 + mu_1) = c_d(p). The *off-face mass*
Moff(m) is the mass outside {pi_0, (1, -1, 1), (1, 1, -1), (0, 1, -1), (0, -1, 1)}. A *level* restricts Moff to [0, 0.01]
(near-face stars) or [0.01, 1] (far stars).

**Lemma SYM (the a <-> b swap).** The swap (k, t_a, t_b) -> (k, t_b, t_a) of u-profiles maps S(p, d_a, d_b) onto
S(p, d_b, d_a) and preserves Net, Moff and the levels. So every certificate over u-stars needs only dm >= 0.

PROOF. P = (1 + n_a g_a, 1 + n_b g_b) has its components swapped and P_0 is fixed (pi_0 is symmetric). Every wedge changes sign.
Delta_u uses wedges only through absolute values, squares, X^2 and |X| (X changes sign); phi, Phibar, Phi0, K' are symmetric;
thresholds and sup constants depend on the marginal only; s_u is symmetric (k_a <-> k_b, k_ab fixed); the NS row and the R7
square are symmetric.   QED

**Lemma L for stars.** On a slice, with sum m = 1, G_uu = 1, G_vv = V(p), G_uv = U(p) = sqrt 2 mu_-1 + mu_0 + mu_1/sqrt 2,
e_0 = U/sqrt V (a constant of the slice), the other u-side Ghat entries are linear in m: e_ua = G_ua/sqrt d_a,
h_va = G_va/sqrt(V d_a), h_ab = G_ab/sqrt(d_a d_b). Each star key correlation is at least its Lemma L bound
l = N/D0 - (4/D0^2)(D - D0), where for k_a: N = 1 + e_0 + e_ua + h_va and D = 2 + e_0 + e_ua; for k_ab: N = 1 + e_ua + e_ub + h_ab,
D = 2 + e_ua + e_ub. So s_u(m) >= Lfun(m) := r l_ab + l_a + l_b - (r + 2) mu0, which is linear on the slice, and
E(m) >= E^lin(m) := (l_a + l_b + l_ab)/3 - mu0, also linear. The NS row is enforced in the relaxed form E^lin <= B
(a relaxation, since E^lin <= E).

```text
<1>1. The three star edges have cosines >= rho (Lemma EC applies to every star mass vector: each generator satisfies the
      per-type inequality), so D >= D0; and 0 <= N <= 4 because the Ghat entries lie in [0, 1]. The proof of Lemma L
      (Section 2.8) applies verbatim to each key correlation, since k_a = N/(2 sqrt(AB)) with A + B = D.
<1>2. r > 0, so s_u >= r l_ab + l_a + l_b - (r + 2) mu0.   QED
```

**Lemma REL (the relaxation of the kill).** Fix a box Z of slice coordinates and a level. Suppose valid ranges
(lower and upper bounds on every star of every slice of Z at that level satisfying the relaxed NS row) are known for six
linear functionals: the nonbasic parts (with respect to the basis B+ of Section 8.8) of the face imbalance X, of A0 and of
phi; K; Ybar (Y with the box maxima of its coefficients); and M1 (the off-face class -1 mass). Then on that level

    Net(m) >= 4 mu0 Lfun(m) - kappa_S DUB(m),

where DUB is Delta_u with every nonlinear term replaced by an upper bound that is linear on the slice:

| term of Delta_u | replacement (valid on the level) |
|---|---|
| the X^2 terms | X = X_B + xi; xi^2 <= the secant (x_lo + x_hi) xi - x_lo x_hi on the range, or <= Xi sum_j |xt_j| m_j |
| phi K, phi A0 | phi = phi_B + ft.m; McCormick upper planes for the products (ft.m) K and (ft.m) xi0 |
| K^2, A0^2, Y^2 | secants on the certified ranges (A0 through its basis decomposition) |
| |X| K', |X| Phi0 | |X| <= |X_B| + max |xi| |
| R | R <= Wmax (M1^2 - sum m_i^2) <= Wmax (M1_hi M1 - sum_k th_k (2 h_k m_k - h_k^2)) for any convex weights th |
| R0, Rm0 | R0 <= W00 mu_0^2, Rm0 <= Wm0 mu_0 M1 |
| hinges | the argument bounded above using ft.m <= f_hi and the box maxima of Abar, w11, w10 |

PROOF. Each replacement is one of: a secant of a convex function on an interval containing its argument; a McCormick upper
plane on a product of two quantities with known ranges; the tangent bound -m_i^2 <= -(2 h m_i - h^2); monotonicity of (.)_+;
or a Cauchy-Schwarz-type bound with nonnegative weights (W_ii = 0 because a wedge of a vector with itself vanishes, and
P^(pi_0) = 0). Lfun <= s_u is Lemma L for stars. All coefficients are computed in interval arithmetic.   QED

The ranges are themselves certified per box by small slice LPs (Mode 1 below). *Range inheritance.* A child
box may reuse ranges certified on its parent. One functional, Ybar = sum_j pq_j m_j, has coefficients pq that are box
constants, so a child's Ybar is smaller than its parent's, and the secant (Ylo + Yhi) Ybar - Ylo Yhi of Ybar^2 is valid only if
Ybar >= Ylo. So on inherited records the range [min(Ylo, 0), Yhi] is used and pq(child) <= pq(parent) is checked
componentwise (then 0 <= Ybar_child <= Ybar_parent <= Yhi).

**Lemma MODE1-A (weak duality on a box).** Write the slice constraints as: class sums equal mu_k(p), and
sum_j m_j A_j = d with A_j = (2^{-t_a}, 2^{-t_b}) for profile j. Let c(z), k(z) be the coefficients of the linear lower
objective at slice coordinates z (Lfun minus kappa_S DUB), and let nsg_j, nsc be the coefficients of the relaxed NS row,
OFF_j the off-face indicator. For any multipliers y = (y_a, y_b) and eta_NS, eta_hi, eta_lo >= 0 (which may depend affinely
on z, provided the etas are >= 0 on the whole box), every m in the slice at z, at the level [s_lo, s_hi], with E^lin <= B, has

    c.m + k >= k + y_a d_a + y_b d_b - eta_NS nsh - eta_hi s_hi + eta_lo s_lo
               + sum_k mu_k min_{class k} (c_j - y_a A_aj - y_b A_bj + eta_NS nsg_j + (eta_hi - eta_lo) OFF_j),

where nsh is the right side of the NS row. The right side, as a function of z, is bounded below on the box by its values at
the 16 vertices, via Lemma VERTEX below.

PROOF. c.m = sum_j m_j (c_j - y.A_j) + y.d. Subtracting the nonnegative quantities eta_NS (nsh - nsg.m), eta_hi (s_hi - Moff(m))
and eta_lo (Moff(m) - s_lo) only lowers this; then group by classes (m_j >= 0, class sums mu_k).   QED

**Lemma VERTEX.** Let f be locally Lipschitz on a box Z in R^4, z_c in Z, and [G_i] intervals containing every
Clarke generalised gradient component of f on Z. Then f(z) >= L(z) := f(z_c) + sum_i min(G_i^lo (z_i - z_ci), G_i^hi (z_i - z_ci))
on Z, L is concave, and min_Z L is attained at a vertex. Sums, minima and nonnegative combinations of such minorants are
again concave minorants.

PROOF. Lebourg's mean value theorem: f(z) - f(z_c) lies in <df(xi), z - z_c> for some xi on the segment, with df the Clarke
gradient (for C^1 functions this is the classical mean value theorem); each component of the product is at least the minimum
of the two products. Each min(G^lo t, G^hi t) is concave in t; concave functions on a box attain their minimum at a vertex.  QED

(For the nonsmooth max, |.| and (.)_+ in the costs, the interval gradient is taken as the hull of the active gradients, which
contains the Clarke gradient.) An empty slice-level set is closed the same way by a Farkas-type bound: a dual for the zero
objective whose box bound is > 0.

So a *record* of a closed box stores the box, the level, the candidate duals (affine in z), weights theta of a convex
combination of candidates, the certified ranges and their duals, and the NS constant actually used. A *replay* recomputes the
vertex values from scratch, with outward rounding and no LP solve, and checks the inequality. Floating-point choices
(duals, bases, weights) affect only whether a box closes, never the validity of a closed box.

### 8.7 Away from the tree marginal: the antisymmetric target

With the kill, single stars need not be positive away from the tree marginal: far
stars make single stars negative at marginals away from 1/2. So the two mirror stars are paired by a target that is
antisymmetric in alpha.

**The target.** g(alpha) := -sgn(a) 0.045 max(|a| - 0.015, 0), a := alpha - 1/2. It is exactly antisymmetric:
g(1 - alpha) = -g(alpha). (It was fitted on floating-point relaxations; validity does not depend on the fit.)

**The domain Q.** Q := [0.47, 0.53] x [0, 0.005] in (alpha, lam). It is mirror-symmetric (its endpoints are exactly symmetric in
doubles).

**Proposition PAIR.** If tau(p) >= g(alpha) for every marginal p of N outside Q, and CKS* holds for every marginal of
N in Q, then CKS* holds.

PROOF. p is a marginal of N iff p^sigma is (Lemma MIRB), and Q is mirror-symmetric. For p outside Q,
tau(p) + tau(p^sigma) >= g(alpha) + g(1 - alpha) = 0.   QED

### 8.8 Near the tree marginal: the domain Q

Inside Q a single star is slightly negative. Here the two stars are paired through their exact local structure.

**The face family and the kink basis.** Name the u-profiles pi_0 = (-1, 1, 1), pi_a = (1, -1, 1), pi_b = (1, 1, -1) (the u-profiles
of the tree types), j0 = (0, 1, -1), j0' = (0, -1, 1), jp = (1, -1, -1), jm = (1, 1, 1), and the two bases of the slice LP

    B+ = {pi_0, pi_a, pi_b, j0, jp},        B- = {pi_0, pi_a, pi_b, j0, jm}.

On the *kink line* dp = 0 the basic solution of B+ (and of B-) is the *face star*

    m4(p, t) = mu_-1 e_{pi_0} + mu_0 e_{j0} + (mu_1/2 + t) e_{pi_a} + (mu_1/2 - t) e_{pi_b},      t = t(d) := (dm + (3/2) mu_0)/3.

(Check: its class sums are mu(p), its d_a + d_b is c_d(p), and its d_a - d_b = 3t - (3/2) mu_0.) The range of t(d) on the
local region below is [-0.02667, 0.02838], inside T := [-0.027, 0.029].

**Lemma LREL (the local relaxation).** On the level Moff <= 0.01, use only those replacements of Lemma REL that
vanish on the face support {pi_0, pi_a, pi_b, j0}: absolute-value forms instead of secants, McCormick planes through 0, and
|X| Phi0 <= (gamma X^2 + Phi0^2/gamma)/2 with gamma = 0.05; and suppose every hinge argument of Delta_u is certified <= 0 on the
box and the box maximum ph(jp) of phi over U_1 is certified. Then Net(m) >= c.m + k, a function linear on each slice, and its
value at the basic solution m4(p, t) is the smooth function

    h_K(p, t) := 4 mu0 Lfun(m4) - kappa_S [ cX X^2 + phi A0/(omega R22) + omega tau_phi(q) phi A0/R22 + omega A0^2/(2 R22 lam)
                 + Phi0^2/(2 gamma R22 omega) ](m4),       cX := 1/4 + lam ph(jp)^2/2 + (lam + gamma)/(2 R22 omega).

PROOF. As for Lemma REL: each replacement is a valid upper bound of the corresponding term of Delta_u on the level, and with
the hinges inactive their terms are 0. At m4 the nonbasic masses vanish, so the replaced terms reduce to their basic parts;
K, K', R, Rm0, Y vanish at m4 (class -1 mass only on pi_0, where a = 0, phi = 1, P^ = 0), and R0 = 0 (class 0 mass only on
j0). The remaining terms are the bracket.   QED

**Lemma KINK-K' (one shift per segment).** Let b be a box of slice coordinates in D_loc := {|dp| <= 0.01, |dm| <= 0.08}
on one side of dp = 0, at the level [0, 0.01]. Let B = B+ (side dp >= 0) or B- (side dp <= 0), with dual y_B(z) and reduced
costs rc_j(z) for the lower objective of Lemma LREL, F_B(z) := the value of that objective at the basic solution of B, and a
constant eta_b <= 0 (eta_b = 0 on the side dp <= 0). Put delta_j := (A_aj + A_bj) - base_{k(j)}, base = (1, 5/2, 5/2) for the
classes (-1, 0, 1). Suppose:
(i) rc_j - eta_b delta_j >= 0 on b for every profile j except j0', and rc_{j0'} - eta_b delta_{j0'} >= -P0 with P0 <= 0.02;
(ii) the hinges are inactive on b (as Lemma LREL requires);
(iii) d/d(dp) [F_B + eta_b dp] >= 0 on the *extended box* [alpha] x [lam] x [0, dp_hi] x [dm] (side +), resp. <= 0 on
      [alpha] x [lam] x [dp_lo, 0] x [dm] (side -).
Then every u-star m of a slice in b at level [0, 0.01] has Net(m) >= h_K(p, t(d)) - P0 mu_0(p).

```text
<1>1. (shifted reduced costs) With y' := y_B + eta_b u, u = +1 on both d-rows and -base_k on the class rows,
      c.m + k = F_B(z) + eta_b dp + sum_j (rc_j - eta_b delta_j) m_j for every m in the slice: c.m = y'^T b + sum_j (c_j - y'^T a_j) m_j,
      y'^T b = y_B^T b + eta_b (d_a + d_b - sum_k base_k mu_k) = (F_B - k) + eta_b dp (since sum_k base_k mu_k = c_d(p)), and
      c_j - y'^T a_j = rc_j - eta_b delta_j. (The basic profiles have rc = 0; their delta is 0 except jp, delta = 3/2, whose
      shifted cost -1.5 eta_b is >= 0.)
<1>2. By (i), and since m_{j0'} <= mu_0, c.m + k >= F_B(z) + eta_b dp - P0 mu_0.
<1>3. By (iii), along the segment s -> (p, s, dm) from s = 0 to s = dp, which lies in the extended box, F_B + eta_b s is
      monotone in the right direction; so F_B(p, dp, dm) + eta_b dp >= F_B(p, 0, dm).
<1>4. On dp = 0 the basic solution is m4(p, t(d)), so F_B(p, 0, dm) = h_K(p, t(d)) (Lemma LREL).
<1>5. Net(m) >= c.m + k (Lemma LREL, with (ii)).   QED
```

KINK-K' uses one eta_b along the
whole segment and certifies the slope on the extended box. On the side dp >= 0 the slope of F_B is enclosed by a second-order
interval Taylor form (F_B uses only basis columns, so it needs no ranges).

Put H(p) := min_{t in T} h_K(p, t) - P0 mu_0(p) with P0 = 0.02.

**Lemma LOC2-K (the coupled local inequality).** (uses the finite claim FC-LOC2K) Let Psi_K(a, lam, t, t') := h_K(1/2 + a, lam, t) +
h_K(1/2 - a, lam, t') and S_min := 1.4626499 (a lower bound on Q of the normaliser (1 - lam)(2 - alpha) + lam of Lemma COORD,
so mu_0 <= lam/S_min on Q). Then
(a) Psi_K(a, lam, t, t') - 2 P0 lam/S_min >= 0 on [0, 0.0300001] x [0, 0.005] x T x T, with equality only at 0;
(b) h_K(p, t) - P0 mu_0(p) >= -1.5e-4 on [-0.0300001, 0.0300001] x [0, 0.005] x T (a = alpha - 1/2).
Consequently H(p) + H(p^sigma) >= 0 for every p in Q.

PROOF IDEA. Psi_K(0) = 0 exactly (at the star tree point Lfun = 0, and at t = lam = 0 the kill terms vanish: X(m4) = 0 by the
a <-> b symmetry, A0 = Phi0 = 0 since mu_0 = 0). Its first derivatives in a, t, t' vanish at 0 (the alpha-derivatives of the
two stars cancel; t -> h_K is even at lam = 0 by the a <-> b swap), and its lam-derivative is 0.1786937 > 2 P0/S_min (the kill
takes 34% of the plain cut's lam-slope 0.269437). On the origin box [0, 5e-4]^2 x [-5e-4, 5e-4]^2 an interval Hessian has
lambda_min >= 0.79398 in (a, t, t') with the mixed lam-terms bounded, and a quadratic argument closes it; the rest is covered
by second-order Taylor boxes. For the consequence: for p = (1/2 + a, lam) with a >= 0, p^sigma = (1/2 - a, lam), and both
mu_0(p), mu_0(p^sigma) are <= lam/S_min; take t, t' attaining H.

**Finite claim FC-LOC2K.** (computer-checked: loc2k/src/bk16_loc2.py, loc2k/src/bk16_loc2run.py; record
loc2k/run/loc2_P002/loc2K_boxes.jsonl; replay loc2k/src/bk16_loc2verify.py) The record holds the origin box
(interval Hessian: lambda_min >= 0.79398 on (a, t, t'), |H_{lam,x}| <= 1.80746, |H_{lam,lam}| <= 0.75222, so the
lam-slope G = 0.1786937 - 2 P0/S_min - 0.75222 x 5e-4/2 = 0.151158 > 0 and the quadratic argument needs
lam <= 2 cX G/hX^2 = 0.0735, true on the box), 5,210 coupled second-order Taylor boxes for (a) (minimum margin 1.05e-9) and 96
single boxes for (b) (minimum -1.488e-4 >= -1.5e-4). The replay recomputes every box (0 failures) and checks that the exact
rational volumes of the boxes equal those of the two domains. The certificate was made at pi_lo = 0.815; by Lemma MONO-kappa
(h_K is nonincreasing in kappa) it holds at pi_lo = 0.82.

The single star is negative only near alpha = 1/2, so inside Q we use per-cell targets. Cut the alpha-range of Q into the
twelve cells C_i := [0.47 + 0.005 i, 0.475 + 0.005 i], i = 0, ..., 11; the mirror maps C_i onto C_{11-i}.

**Lemma LOC2-K(c) (per-cell local constants).** (uses the finite claim FC-LOC2KC) For each cell, h_K(p, t) - P0 mu_0(p) >= -c_i on
C_i x [0, 0.005] x T, with c_0 = ... = c_4 = 0, c_5 = 1.99485e-5, c_6 = 4.74759e-5, c_7 = 4.73344e-5, c_8 = 2.93951e-5,
c_9 = c_10 = c_11 = 0; and the same bound -c_j holds on the 11 gaps between consecutive cells as built in doubles.

**Finite claim FC-LOC2KC.** (computer-checked: loc2kc/src/bk17_loc2c.py; record loc2kc/run/loc2c/loc2c_boxes.jsonl, 5,376
second-order Taylor boxes; constants loc2kc/run/loc2c/loc2c.json; at pi_lo = 0.82.) The cells, built in doubles as
a0 = 0.47 + 0.005 i - 0.5, a1 = a0 + 0.005, leave gaps of width
3.47e-18 to 4.34e-18 between consecutive cells, and the mirror of each run cell overhangs its partner cell by such a gap;
h_K - P0 mu_0 >= -c_j is certified on [a1_j - 1e-9, a0_{j+1} + 1e-9] x [0, 0.005] x T for all 11 gaps (Taylor
bound, 112 sub-boxes each, records in loc2kc/check; e.g. after cell 6 the minimum is -4.6016e-5 >= -4.7476e-5). Also
H_lo(C_0) := 7.8282e-4 is the least Taylor bound of cell 0 (7.841e-4 on the sliver above it).

**Theorem CS-K'' (pairing inside Q).** Let T_i (i = 0, ..., 11) be reals with (i) T_i + T_{11-i} >= 0 and
(ii) T_i >= -H_lo(C_{11-i}), where H_lo(C_j) is a certified lower bound of H on C_j x [0, 0.005], slivers included. Suppose every
u-star of S(p), for p in a mirror-symmetric set P inside Q, lies in a record box of kind 'K' (Lemma KINK-K' holds on it),
'G' (the relaxation of Section 8.6 is >= T_i on the box, for p in C_i), 'E', 'E2' or 'Xd' (the box holds no star). Then
tau(p) + tau(p^sigma) >= 0 for every p in P.

```text
<1>1. Stars in 'G' boxes have Net >= T(p); stars in 'K' boxes have Net >= h_K(p, t(d)) - P0 mu_0 >= H(p) (t(d) in T);
      'E', 'E2', 'Xd' boxes hold no star. So tau(p) >= min(T(p), H(p)).
<1>2. For p in C_i, p^sigma lies in C_{11-i} or its sliver. Four cases:
      G and G: T_i + T_{11-i} >= 0 by (i).
      G at p, K at p^sigma: T_i + H(p^sigma) >= T_i + H_lo(C_{11-i}) >= 0 by (ii).
      K at p, G at p^sigma: the same with i and 11 - i exchanged.
      K and K: H(p) + H(p^sigma) >= 0 by Lemma LOC2-K.   QED
```

**The targets used.** H_lo(C_j) = -c_j for j >= 1 (FC-LOC2KC) and H_lo(C_0) = 7.8282e-4. The 'G' records of Q carry targets
T_i = max(0, c_{11-i}), or 1.5e-4 on the first-round records, except T_11 = -1.5e-4. Condition (ii) holds since
1.5e-4 >= every c_j and T_11 = -1.5e-4 >= -7.8282e-4. Condition (i) holds trivially for nonnegative targets, and for the pair
(11, 0) because all 4,488 'G' records of C_0 are first-round records with target 1.5e-4. (The negative target is needed at the
far star near alpha = 0.53, lam = 0, where the relaxation is slightly negative: 47 boxes reached depth 40 with target 0.)

### 8.9 The JOINT cap and the Emin table

Far from the tree marginal, at large lam, the relaxed slice LP admits stars that N itself excludes, because the relaxed NS
row lets both stars sit near the NS boundary, while Lemma JOINT says they cannot. The cap lowers each star's NS row by what
the mirror star is certified to spend.

**Lemma CAP.** For a marginal cell P let Emin_lo(P) be a certified lower bound of E^lin over the u-stars with marginal
in P, ratios in the R7 square and the relaxed NS row. Let w = 0.97147. Then
(a) for the cell P of the marginal of every point of N, max(Emin_lo(P), 0) + w max(Emin_lo(P^sigma), 0) <= B;
(b) the u-star of every point m of N satisfies E^lin(m_u) <= B - w max(Emin_lo(P^sigma), 0), where Emin_lo(P^sigma) is the
    least value over the cells met by the exact mirror of P.

PROOF. Lemma JOINT: E_u + w E_v <= B. E_v = E(m^sigma) for the mirror star, which lies in S(p^sigma) (Proposition RED <1>1),
so E_v >= E^lin(m^sigma) >= Emin_lo(P^sigma); also E_v >= 0 (Lemma STAR). Likewise E_u >= max(Emin_lo(P), 0), and
E^lin(m_u) <= E_u. Rearrange.   QED

**Use.** By (a), a marginal cell with max(Emin_lo(P), 0) + w max(Emin_lo(P^sigma), 0) > B (lower end of the interval sum against
the upper end of B) holds no marginal of N, and every box over it is closed as 'XJ'. By (b), the certificate outside Q need only
bound

    tau_cap(p) := inf of Net over the stars of S(p) whose NS row is lowered to E^lin <= B - w max(Emin_lo(P^sigma), 0);

every record stores the lowered NS constant as 'cth'. **CKS\*_cap** is the statement tau_cap(p) + tau_cap(p^sigma) >= 0 for every
marginal p of N. It gives (K) on N exactly as CKS* does, because Proposition RED applies CKS* only to the u-star and the
mirror star of points of N, and by (b) they lie in the capped sets. On Q we use the uncapped tau, and tau_cap >= tau
everywhere. Proposition PAIR holds verbatim for tau_cap.

**Finite claim FC-EMIN (the Emin table).** (computer-checked: kill/src/bk19_emin.py) The window's root grid has
16 alpha cells and 10 lam rows (row breakpoints 0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25); it is
mirror-symmetric (cell (i, j) <-> (15 - i, j)); 4 of its cells make up Q. For each of the other 156 cells, a best-first branch
and bound over (alpha, lam, dp, dm >= 0) (Lemma SYM) with 1,000 leaves; each leaf is 'Xd' (no d of the box in the R7 square),
'E' (Farkas bound > 0: no star) or 'L' (a certified Mode 1 lower bound of nsc + nsg.m with its dual); Emin_lo(cell) = min over
'L' leaves of lb/3 - mu0, rounded down (1 if there is no 'L' leaf). Files emin_A.json, emin_B.json: 154,246 leaves (E 95,008, L 55,491, Xd 3,747). Checks: the replay re-certifies every leaf
(0 failures, exact volume in every cell, replayed value >= recorded value). Effect: 18 table cells are excluded by (a); the
cap lowers B by 4-28% on the far-star cells at lam < 0.05.

*The exact mirror cells.* The alpha grid is not mirror-symmetric in doubles: for some breakpoints
ABP[k] + ABP[16 - k] - 1 = +-2^-54. So the exact mirror of a box touching such a breakpoint overhangs by 2^-54 into a
neighbouring cell, where Emin_lo may be smaller, and the recorded cap may be too strong there. The
9,292 Emin leaves of the affected edges were re-certified on boxes extended by 1e-12 across the edge (every 'E' and 'Xd' leaf
stays closed; 11 minimiser leaves lose 9e-15 to 1.0e-13), and all 48,057 affected records were replayed with the justified
cap: 0 failures. For the stage-3 records of FC-GSTAR a census against the exact rational mirror found 0 unjustified caps
(minimum slack +9.3e-17); 180 records rely on an extended certificate for cell (12, 6), side "hi"
(434 edge leaves, all closed, minimum E^lin 2.0755e-4 >= Emin_lo(12, 6) =
1.98722e-4). The fallback of replaying those 180 records with the neighbour-cell cap fails for 82 of them, so this extended
certificate is load-bearing.

### 8.10 The records and the two certificates

Every closed box is one record (one JSON line). Its *kind* says how it is closed:

| kind | meaning (for every star of every slice of the box, at the record's level) |
|---|---|
| 'G' | the relaxation of Net (Lemma L for stars, REL, MODE1-A, VERTEX) is >= the target: g(alpha) outside Q, T_i in Q; the NS constant 'cth' is the capped one where recorded |
| 'K' | a kink box of Q: conditions (i)-(iii) of Lemma KINK-K' hold, so Net >= H(p) |
| 'E', 'E2' | the slice-level set is empty on the box (a Farkas-type certificate; 'E2' a second form) |
| 'X' | the marginal box violates Lemma WIN's constraints (the V range of Lemma R7, or Lemma E0) |
| 'Xd' | every d of the box lies outside the R7 square (interval arithmetic) |
| 'XJ' | the marginal cell is excluded by Lemma CAP (a) |

**Finite claim FC-Q (domain Q).** (computer-checked) One branch and bound over Q x (all d of the R7 box) x
both levels, in pm coordinates: root grid 12 alpha cells (the C_i) x 2 lam cells x 14 dp cells over [-0.55, 0.72] (which
contains the R7 square's dp-range [-0.5432, 0.7116] on Q) x 14 dm cells over [-0.59, 0.59] = 4,704 root cells, x 2 levels =
9,408 root-levels. Records: 337,645 leaves = 159,990 (stage 1, cells C_0..C_7) + 177,655 (stage 2, cells C_8..C_11: G 163,919
(target 0 on 137,754, target -1.5e-4 on 26,165), K 8,789, E 4,317, E2 347, Xd 283). In all: 27,231 'K' boxes (min shifted reduced
cost 1.4615e-8, max P0 0.018160, min certified slope 8.56e-6 on side + and <= -0.0970 on side -). Minimum certified margins of
the stage-2 records: G 2.7e-11, K 1.6e-8, E 1.5e-3, E2 2.2e-3.
- *What the checkers verify*: the replay (kill/src/bk19_verify.py, mode Q) recomputes each record's bound from
  its stored candidates with outward rounding and no LP solve and checks it against the record's target (for inherited records
  with the range of Section 8.6 and the pq test; a negative target is accepted only for 'G' records in C_11 and only >= -1.5e-4);
  the cover (bk17_coverq.py, run as `python ../src/bk17_coverq.py Q1 Q1s`) checks exact tiling: 0 gaps, 0 overlaps, 0 pending,
  all 9,408 root-levels closed.

**Finite claim FC-GSTAR (outside Q).** (computer-checked) One branch and bound over
(W_win minus int Q) x (dp in [-0.79, 0.99], dm in [0, 0.59]) x both levels (dp-range covers the R7 square's [-0.782, 0.982] at
every marginal of the window; dm >= 0 by Lemma SYM; |dm| <= 0.58616 on the square), target g(alpha), cap 'cth' from FC-EMIN.
Root grid: the 156 marginal cells x 14 dp cells x 4 dm cells = 8,736 root cells, x 2 levels = 17,472 root-levels. Records:
1,801,388 leaves = 37,093 (stage 1) + 1,136,669 (stage 2) - 1 (an inherited record written before the range of Section 8.6
was enforced, whose bound with that range is -9.35e-5; it is excluded, gstar/run/excluded_records.json, and its box re-done as
job H3) + 627,627 (stage 3). By kind: E 796,534; E2 90,822; G 899,795; Xd 7,214; X 5,231; XJ 1,792. Minimum margins over the replayed
records: G 5.98e-10 (stage 2), 1.89e-9 (stage 3); E 1.86e-6; E2 8.0e-7.
- *What the checkers verify*: the replay (bk19_verify.py, mode G) as for Q, plus the 'cth' check against the Emin table
  (cth >= c_Theta - w Emin_lo(mirror cell), rounded) and the 'XJ' check; the cover (bk19_gcover.py, run as
  `python ../src/bk19_gcover.py GB1,GB2,GC,GD H1,H2,H3 list` with gstar/run/excluded_records.json): 0 gaps, 0 overlaps, 0 stuck, 0
  pending (final gcover_pending.json is the empty list); all 17,472 root-levels closed.

The constant target of the first Q round must be >= 1.5e-4 (the floor
of LOC2-K (b)); it was.

### 8.11 Proof of Theorem K-N

```text
<1>1. Every marginal of N lies in W_win (Lemma WIN). Every u-star and mirror star of a point of N lies in the R7 square and
      satisfies the NS row (Lemmas R7, NS), and by Lemma CAP (b) its capped NS row.
<1>2. (Inside Q.) The Q records cover Q x R7 x both levels (FC-Q), with kinds 'G', 'K', 'E', 'E2', 'Xd' and the targets of
      Section 8.8, which satisfy (i) and (ii) of Theorem CS-K''. Theorem CS-K'' with P = Q (mirror-symmetric) gives
      tau(p) + tau(p^sigma) >= 0 for every p in Q, hence tau_cap(p) + tau_cap(p^sigma) >= 0.
<1>3. (Outside Q.) The G* records cover (W_win minus int Q) x (the dp-range) x (dm >= 0) x both levels (FC-GSTAR); by Lemma
      SYM the half dm >= 0 covers every star. 'G' records give Net >= g(alpha) on the stars of the capped sets; 'E', 'E2', 'X',
      'Xd' hold no star; 'XJ' cells hold no marginal of N (Lemma CAP (a)). So tau_cap(p) >= g(alpha) for every marginal p of N
      outside Q.
<1>4. Proposition PAIR (for tau_cap): CKS*_cap holds for every marginal of N.
<1>5. Proposition RED, with Lemma CAP (b) for the capped sets and Theorem SEP (kappa_S at pi_lo = 0.82): (K) holds at every
      m in N.   QED
```

---

## 9. The kill is positive semidefinite on N

This section proves

**Theorem PSD-N.** (uses the finite claims FC-THIN, FC-FBAR, FC-LR and FC-CLR, besides those of
Section 7) For every m in N, the kill C'(m) is PSD.

Near the tree this is easy (C_T has lambda_min = 1 - p = 0.0429 and the kill moves one entry by O(delta)). Far from the tree
the kill can fail to be PSD (Section 5.4), but those points have small rule bound. The proof goes through an exact criterion
(Lemma ANGLE), which becomes a "disc" condition on two numbers b_1, b_2 (Theorem PSD-PHI). Each b_i is bounded by a sum of two
one-star functionals (Theorem BOUND). A star certificate bounds these functionals as affine functions of each star's own
excess (Theorem LR's obligations (U-LR*), (V-LR)), and an exact finite check over the region that Lemma JOINT allows for the
two excesses proves the disc condition ((C-LR)).

### 9.1 The exact criterion

For the cross block K of C(m) put sigma := the sum of its four entries and the two *alternating sums*

    beta_1 := K_ac + K_ad - K_bc - K_bd = <y_a - y_b, y_c + y_d>,     beta_2 := K_ac - K_ad + K_bc - K_bd = <y_a + y_b, y_c - y_d>,

and the same with primes for the kill's cross block K'. Write k = C_ab, k' = C_cd.

**Lemma ANGLE.** Let |k|, |k'| < 1 and sigma' != 0, and put

    c' = sigma'/(2 sqrt((1 + k)(1 + k'))),   g1' = beta1'/(2 sqrt((1 - k)(1 + k'))),   g2' = beta2'/(2 sqrt((1 + k)(1 - k'))).

Then the kill C' is PSD iff (c'^2 + g1'^2)(c'^2 + g2'^2) <= c'^2.

In words: c' is the cosine between the two sum keys, g1' the cosine between a difference key and the other sum key; the kill
is PSD exactly when these cosines are small together.

```text
<1>1. C' has pair blocks P_u = [[1, k], [k, 1]], P_v = [[1, k'], [k', 1]] (positive definite) and cross block K'. By the Schur
      complement, C' is PSD iff P_v - K'^T P_u^{-1} K' is PSD iff ||P_u^{-1/2} K' P_v^{-1/2}||_op <= 1.
<1>2. With O = [[1, 1], [1, -1]]/sqrt 2: O P_u O = diag(1 + k, 1 - k), O P_v O = diag(1 + k', 1 - k'), and
      O K' O = (1/2)[[sigma', beta2'], [beta1', Alt']], Alt' = K'_ac - K'_ad - K'_bc + K'_bd. K' has rank one (D(K') = 0), so
      Alt' = beta1' beta2'/sigma' and O K' O = (1/(2 sigma')) (sigma', beta1')^T (sigma', beta2').
<1>3. The operator norm of x y^T is |x||y|. So ||P_u^{-1/2} K' P_v^{-1/2}||^2 =
      (1/(4 sigma'^2)) [sigma'^2/(1 + k) + beta1'^2/(1 - k)] [sigma'^2/(1 + k') + beta2'^2/(1 - k')].
<1>4. The first bracket is 4(1 + k')(c'^2 + g1'^2), the second 4(1 + k)(c'^2 + g2'^2), and 4 sigma'^2 = 16(1 + k)(1 + k') c'^2.
      So the norm is <= 1 iff (c'^2 + g1'^2)(c'^2 + g2'^2) <= c'^2.   QED
```

(This is an "iff" for the killed matrix, which has a rank-one cross block; on N, sigma' > 0 since all
cross entries are >= 0.82.)

**Corollary MONO.** With F' := c'^2 + g1'^2 + g2'^2 + g1'^2 g2'^2/c'^2, C' is PSD iff F' <= 1. If c_lo^2 <= c'^2 <= c_hi^2 and
c_lo^2 >= max(g1'^2, g2'^2), then F' <= c_hi^2 + g1'^2 + g2'^2 + g1'^2 g2'^2/c_lo^2. Moreover c'^2 <= c^2, and
|g_i'| <= (|beta_i| + delta)/(2 sqrt((1 - k)(1 + k'))) (resp. with (1 + k)(1 - k') for i = 2).

PROOF. Expand the criterion of Lemma ANGLE and divide by c'^2. The function t -> t + A/t is increasing for t >= sqrt A, with
A = g1'^2 g2'^2 <= c_lo^4. The kill lowers one cross entry by delta > 0, so sigma' = sigma - delta < sigma, and each alternating
sum changes by at most delta.   QED

### 9.2 The disc criterion

**Theorem PSD-PHI (the disc criterion).** (uses the finite claim FC-THIN) Let m be in N and b_i := |beta_i(m)| + delta(m), i = 1, 2.
Put

    den := 4 (1 - k_hi)(1 + k_lo) = 0.1519464,     c2* := 0.8643473,     c2' := 0.61536,
    Disc(b_1, b_2) := b_1^2 + b_2^2 + b_1^2 b_2^2/(den c2'),     Disc_0 := den (1 - c2*) = 0.0206119.

If Disc(b_1, b_2) <= Disc_0, then the kill C'(m) is PSD.

```text
<1>1. delta <= min(b_1, b_2) <= sqrt(Disc_0/2) = 0.10152.
<1>2. By Lemma CLO, sigma' >= 4 C_lo - delta >= 4 (0.8022) - 0.10152, and (1 + k)(1 + k') <= (1 + k_hi)^2 (Lemma THIN (e)); so
      c'^2 >= (3.2088 - 0.10152)^2/(4 (1.9805107)^2) = 0.61538 >= c2'.
<1>3. By Lemma THIN (d), (e): 2 sqrt((1 - k)(1 + k')) >= sqrt(den) (and likewise for the other product), so |g_i'| <= b_i/sqrt(den)
      (Corollary MONO).
<1>4. max g_i'^2 <= Disc_0/den = 0.1357 <= c2', so Corollary MONO applies with c_hi^2 = c2* (Lemma SUMKEY: c'^2 <= c^2 <= c2*) and
      c_lo^2 = c2'.
<1>5. F' <= c2* + (b_1^2 + b_2^2 + b_1^2 b_2^2/(den c2'))/den = c2* + Disc(b_1, b_2)/den <= c2* + Disc_0/den = 1.
<1>6. Lemma ANGLE: C' is PSD.   QED
```

### 9.3 Bounding b_1 and b_2 by one-star functionals

Normalise G_uu = 1. Put r_x := 1/N_x (x = a, b, c, d); on N each lies in [1/N_max, 1/N0] (Lemma THIN (c)); put
r* := (1/N0 + 1/N_max)/2 = 0.506659 and eps_r := (1/N0 - 1/N_max)/2 = 0.000641. Let l_a := log d_a, l_b := log d_b,
l_c := log(d_c/d_v), l_d := log(d_d/d_v), so n_c = n_v e^{-l_c/2}; on N, |l_x| <= 0.28903. For a u-profile pi and a v-profile pi'
put (with P_t, Q_t of Section 8.1)

    f(pi) := r_a P_a(pi) - r_b P_b(pi),      q(pi') := r_c Q_c(pi') + r_d Q_d(pi').

Then the *frame identity* holds: since y_a - y_b = (â_u + â_a) r_a - (â_u + â_b) r_b and y_c + y_d = (â_v + â_c) r_c + (â_v + â_d) r_d,

    beta_1 = sum_t m_t f(pi_t) q(pi'_t)                                                                  (9.1)

(expand the inner product with W = sum_t m_t P_t Q_t^T).

**Lemma MIRB'.** (by Lemma MIRB) beta_2(m) = beta_1(m^sigma), delta(m^sigma) = delta(m), E_u(m^sigma) = E_v(m).

*Reference constants.* For each class c in {1, -1, 0} let (ghat_v(c), ghat(c)) := (2^{-c/2}, s_c) with s_1 = 1, s_-1 = 3,
s_0 = 2.47487, and put

    q*_c := n_v r* (2 ghat_v(c) + ghat(c))                                          (a constant of a u-slice: depends on d_v only),
    q^ref_c := n_v [ ghat_v(c)(r_c + r_d) + (ghat(c)/2)(e^{-l_c/2} r_c + e^{-l_d/2} r_d) ]     (a constant of a v-slice),
    F_c(m_u) := sum_{pi in U_c} m_u(pi) f(pi),      M* := sum_c q*_c F_c,
    R_v(m_v) := fbar sum_{pi'} m_v(pi') |q(pi') - q^ref_{c(pi')}|,      fbar := 0.48287,
    a_c := n_v (2 ghat_v(c) + ghat(c)) eps_r,     b_c := 0.2932 n_v ghat(c)/2,
    L(E) := min(0.57806, sqrt(2 (B - E)/(w c_Psi))),      c_Psi := 0.029385.

(0.57806 = 2 x 0.28903.)

**Lemma FBAR.** (uses the finite claim FC-FBAR) On N, |f(pi)| <= fbar = 0.48287 for every u-profile pi.

```text
<1>1. By Lemma NS, E_u <= B, and by Lemma RAT below, E_u >= c_Psi (l_a^2 + l_b^2); so (l_a, l_b) lies in the disc of radius
      sqrt(B/c_Psi) = 0.28954.
<1>2. For fixed (r_a, r_b), f(pi) is affine in (n_a, n_b) = (e^{-l_a/2}, e^{-l_b/2}); so |f| is a convex function of (n_a, n_b), and
      its maximum over the image of the disc (a compact set) is attained at an extreme point of the convex hull of that image,
      which lies on the image of the circle. f is also affine in each of r_a, r_b, which range over [1/N_max, 1/N0]; so the
      corners suffice.
<1>3. FC-FBAR: on the image of the circle, at the four corners of (r_a, r_b), for every profile, |f| <= 0.48287.   QED
```

**Finite claim FC-FBAR.** (computer-checked: bk19_fbar.py; 2,294 theta-intervals of the circle
parametrisation, largest enclosure 0.482869987) The maximum is attained at the class-1
profile (1, -1, 1) or (1, 1, -1) near (l_a, l_b) = (-0.2676, 0.1105).

**Lemma RAT.** For every m in N: E_u >= c_Psi (l_a^2 + l_b^2) and E_v >= c_Psi (l_c^2 + l_d^2).

```text
<1>1. kbar_v >= mu(e_uv, e_vc, e_vd) (Lemma STAR (b)); mu is increasing; e_uv >= rho, e_vc >= rho cosh(l_c/2), e_vd >= rho cosh(l_d/2)
      (Lemma EC with the edge ratio d_c/d_v). So E_v >= mu(rho, rho cosh(l_c/2), rho cosh(l_d/2)) - mu0.
<1>2. With mu = (S^2 - 3)/6, S = sum_i s_i, s_i = sqrt((1 + e_i)/2): d mu/d e_i = S/(12 s_i) >= 2.9565/(12 x 0.98809) = 0.24934 on
      [rho, e_max]^3 (S >= 3 sqrt((1 + rho)/2) = 2.9568, s_i <= sqrt((1 + e_max)/2) = 0.98809).
<1>3. By the mean value theorem from (rho, rho, rho), E_v >= 0.24934 rho [(cosh(l_c/2) - 1) + (cosh(l_d/2) - 1)]
      >= 0.24934 (rho/8)(l_c^2 + l_d^2) = 0.029385 (l_c^2 + l_d^2), using cosh x - 1 >= x^2/2.
<1>4. The u-star is the same with the edges ua, ub (ratios d_a, d_b, since G_uu = 1).   QED
```

(The arguments of <1>2 lie in [rho, e_max] by Lemma THIN (c); 0.24934 rho/8 = 0.0293851, rounded down.)

**Lemma CUT.** On N, |l_c| + |l_d| <= L(E_u).

PROOF. Lemma JOINT gives E_v <= (B - E_u)/w; with Lemma RAT, l_c^2 + l_d^2 <= (B - E_u)/(w c_Psi), and |l_c| + |l_d| <= sqrt(2(l_c^2 + l_d^2)).
The other branch, 2 x 0.28903, is Lemma R7.   QED

**Lemma REF.** |q^ref_c - q*_c| <= a_c + b_c (|l_c| + |l_d|) for each class c.

PROOF. q^ref_c - q*_c = n_v [ghat_v(c)((r_c - r*) + (r_d - r*)) + (ghat(c)/2)((e^{-l_c/2} r_c - r*) + (e^{-l_d/2} r_d - r*))]. Here
|r_x - r*| <= eps_r, and |e^{-l/2} r - r*| <= |e^{-l/2} - 1| r + |r - r*| <= (|l|/2) e^{|l|/2}/N0 + eps_r <= 0.2932 |l| + eps_r for
|l| <= 0.28903 (e^{0.1445}/(2 N0) = 0.29311). Collect.   QED

**Lemma COUP.** (uses the finite claim FC-FBAR) beta_1 = sum_c q^ref_c F_c(m_u) + Rem with |Rem| <= R_v(m_v), for every coupling of the two
stars.

PROOF. By (9.1), Rem = sum_t m_t f(pi_t)(q(pi'_t) - q^ref_{c(t)}), so |Rem| <= fbar sum_t m_t |q(pi'_t) - q^ref_{c(t)}| (Lemma FBAR);
summing the joint masses over the u-profile within each class leaves the v-star masses: = R_v(m_v).   QED

Define the two one-star functionals (for a u-star s with marginal p, a v-star s', and L >= 0)

    U_L(s) := |M*(s)| + sum_c |F_c(s)| (a_c + b_c L) + kappa_S Delta_u(s; p),
    V(s')  := R_v(s') + kappa_S Delta_u(s'^sigma; p^sigma).

**Theorem BOUND.** For every m in N: b_1 <= U_{L(E_u)}(m_u) + V(m_v), and b_2 <= U_{L(E_v)}((m_v)^sigma) + V((m_u)^sigma).

```text
<1>1. By Lemma COUP, beta_1 = sum_c q*_c F_c + sum_c (q^ref_c - q*_c) F_c + Rem, so
      |beta_1| <= |M*| + sum_c |F_c| |q^ref_c - q*_c| + R_v <= |M*| + sum_c |F_c| (a_c + b_c L(E_u)) + R_v (Lemmas REF, CUT).
<1>2. delta <= kappa_S (Delta_u(m_u) + Delta_u(m^sigma)) (Theorem SEP). Add.
<1>3. The statement for b_2 is <1>1-<1>2 applied to m^sigma (Lemma MIRB'), whose u-star is (m_v)^sigma, whose v-star is (m_u)^sigma,
      and E_u(m^sigma) = E_v(m).   QED
```

*Slice linearity.* R_v is not linear on a v-slice, because q and q^ref contain r_c = 1/N_c and r_d = 1/N_d,
which vary with the masses on a slice. The certificate uses the *corner form*

    R_v^slice := fbar sum_{pi'} m_v(pi') max over (r_c, r_d) in [1/N_max, 1/N0]^2 of |n_v (r_c e^{-l_c/2}(g_c - ghat/2) + r_d e^{-l_d/2}(g_d - ghat/2))|

(the g_v terms cancel inside a class), a slice constant per profile and >= R_v. Everywhere below, V means R_v^slice + kappa_S Delta_u.

The obstacle is now visible. U_L depends on its own star's excess through L, and the disc needs the pair (E_u, E_v), which
Lemma JOINT confines to the triangle E_u + w E_v <= B. A star with large excess has small L (its ratios are pinned) but leaves
little budget to the other star. So the certificate must bound U and V as functions of the excess.

### 9.4 Theorem LR: checks with lower rows only

A star needs a check's lower
end e only to make L legitimate (L is nonincreasing and E_u >= E^lin >= e), and the target needs to be evaluated only at
the star's own E^lin. So no check needs an upper row; the only upper row is the NS row E^lin <= B, which is part of the domain.

For a u-star slice point s let E^lin(s) := (l_a + l_b + l_ab)/3 - mu0 be its Lemma L lower bound of the excess (Section 8.6).

**Checks.** A *U check* is a triple (e, L, ell): a lower end e, either "none" or a number with 0 < e < B; a number
L >= L(max(e, 0)) (e = none read as 0); and an affine function ell(x) = t0 + t1 (x - c) with t1 >= 0 (c = e, or 0 if e = none). A
*V check* is a pair (e, ell) with the same conditions. A check is *active* at x if e = none or e <= x. Fix finitely many U checks
and V checks, each family containing one check with e = none. Let G be the sorted set consisting of 0, B and every finite lower
end. For each interval I = [g_i, g_{i+1}] of G choose one U check p_U(I) and one V check p_V(I) active at g_i, and put
T_U(x) := (ell of p_U(I))(x) for x in I (T_V likewise).

**The three obligations.**
- (U-LR\*) For every U check (e, L, ell) and every u-star slice point s that is THIN (rho <= e_ua, e_ub <= e_max), with marginal in
  W_win, ratios in the R7 square, the NS row, and E^lin(s) >= e (no condition if e = none): U_L(s) <= ell(E^lin(s)).
- (V-LR) For every V check (e, ell) and every star s in mirror coordinates with the same domain and E^lin(s) >= e:
  V(s) <= ell(E^lin(s)).
- (C-LR) For every cell I x J of G x G, every vertex (x, y) of the polygon P := (I x J) intersected with {x + w y <= B} satisfies:
  the four numbers T_U(x), T_V(y), T_U(y), T_V(x) lie in [0, 0.15], and Disc(T_U(x) + T_V(y), T_U(y) + T_V(x)) < Disc_0.

**Theorem LR.** (U-LR*), (V-LR) and (C-LR) imply that the kill is PSD at every m in N.

```text
<1>1. (Where the excesses lie.) Let m be in N with u-star s_u and v-star s_v. Put x := E^lin(s_u), y := E^lin of s_v in mirror
      coordinates, x+ := max(x, 0), y+ := max(y, 0). By Lemma L for stars, x <= E_u and y <= E_u(m^sigma) = E_v. By Lemma STAR,
      E_u, E_v >= 0, so x+ <= E_u and y+ <= E_v. Lemma JOINT gives x+ + w y+ <= E_u + w E_v <= B; and 0 <= x+, y+ <= B.
<1>2. (The U term of s_u.) Let I be an interval of G containing x+, and (e, L, ell) := p_U(I), active at the left end g of I.
  <2>1. s_u is in the check's set: if e = none there is nothing to show; otherwise 0 < e <= g <= x+, so x+ > 0, x = x+ >= e.
  <2>2. L is legitimate: if e = none, L(E_u) <= L(0) <= L; otherwise E_u >= x >= e > 0 gives L(E_u) <= L(e) <= L. U_L is
        nondecreasing in L (its L-coefficient sum_c |F_c| b_c is >= 0), so U_{L(E_u)}(s_u) <= U_L(s_u).
  <2>3. s_u is THIN (Lemma THIN), so by (U-LR*), U_L(s_u) <= ell(x) <= ell(x+) = T_U(x+) (t1 >= 0 and x <= x+).
  The same argument for m^sigma, whose u-star is the mirror of s_v with E^lin = y, gives U_{L(E_v)}(s_v^sigma) <= T_U(y+), with an
  interval J containing y+.
<1>3. (The V terms.) The same with V checks (no L, so only <2>1 and <2>3): V(s_v) <= T_V(y+) and V(s_u^sigma) <= T_V(x+).
<1>4. (Theorem BOUND.) b_1 <= T_U(x+) + T_V(y+) =: A and b_2 <= T_U(y+) + T_V(x+) =: A'.
<1>5. (Composition.) (x+, y+) lies in the polygon P of the cell I x J (by <1>1). On P each of the four maps is affine (one piece
      each), so (x, y) -> (A, A') maps P onto the convex hull of the images of its vertices, which lie in [0, 0.3]^2 by (C-LR).
      Disc is convex on [0, 0.3]^2 (the diagonal Hessian entries are positive and the Hessian determinant is >= 0.58 there),
      so Disc(A, A') <= the maximum over the vertices < Disc_0. Disc is nondecreasing in each argument on the positive quadrant,
      and 0 <= b_1 <= A, 0 <= b_2 <= A', so Disc(b_1, b_2) < Disc_0.
<1>6. Theorem PSD-PHI: the kill is PSD at m.   QED
```

Remarks. (1) The condition e > 0 for finite lower ends matters in <2>1: with e = 0 a star with x < 0 would be charged to a
check whose set does not contain it (the composition code asserts 0 < e < B).
(2) (C-LR) puts each of the four values in
[0, 0.15], so A, A' lie in [0, 0.3]^2, where Disc is still convex. (3) The engine certifies (U-LR) only for
THIN stars, so the theorem is stated with (U-LR*); Theorem LR applies it only to such stars. (4) Nothing depends on how the
targets were chosen; any numbers passing the three obligations will do.

### 9.5 What one record of the engine proves

(U-LR*) and (V-LR) are certified by a branch and bound over boxes Z of (alpha, lam, dp, dm), one off-face level, and the set of
checks of one family, with the machinery of Section 8.6: Lemma L for stars gives E^lin and the slice LP; ranges of the linear
functionals are certified per box; the upper forms of U_L and V are linear on a slice given the ranges (Lemma REL's
replacements, McCormick planes and secants for the products in M*, F_c and Delta_u); and weak duality on the box (Lemma
MODE1-A with a lower row E^lin >= e added, plus Lemma VERTEX) bounds "target minus upper form" from below at the 16 vertices.
For a check (e, ell) the check set at slice point z is

    S_e(z) = { m >= 0 : class sums mu(z), G_aa = d_a, G_bb = d_b, the level rows, E^lin(m) <= B, E^lin(m) >= e }.

**The engine statement.** A U record with box Z, a level and a certified check
(e, L, t0, t1) proves: for every z in Z and every u-star x on the slice of z at that level with E^lin(x) <= 0.9595702342051 - mu0,
E^lin(x) >= e (if e is finite) and rho <= e_ua(x), e_ub(x) <= 0.9526712, U_L(x) <= t0 + t1 (E^lin(x) - c). A V record proves the
same for every mirror u-star with R_v^slice(x) + kappa_S Delta_u(x) in place of U_L. Here kappa_S is carried as 0.08077294626 (an
upper bound) and the constants of U_L and R_v^slice are at least as large as those of Section 9.3 (eps_r, 0.2932, fbar = 0.48287,
the corners 1/N_max, 1/N0 widened outward). The upper forms are bk19_obk.u_cost and v_cost on bk19_lb2's slice
quantities.

One constant of u_cost is rounded
to nearest twice and corrected once, so it can fall short of the exact product by less than 1e-20 (the term is below 4.2e-5).
Only a record whose certified margin is below 1e-20 could be affected; the least margins over all records are 1.4e-9 (U) and
4.5e-10 (V) (FC-LR), so none is.

**Lemma SYM (for U_L, V and E^lin).** Under the a <-> b swap of u-profiles (z = (alpha, lam, dp, dm) ->
(alpha, lam, dp, -dm)), the slice set, the off-face levels, the NS row, every lower row E^lin >= e, THIN and the R7 condition are
preserved; E^lin, U_L (every L), Delta_u, R_v^slice and V are invariant; M*, F_c, X and e_ub - e_ua change sign.

PROOF. On a u-star the swap exchanges the components of P and the ratios d_a, d_b (and r_a, r_b), so f -> -f, F_c -> -F_c,
M* -> -M* (q*_c depends on d_v only); |M*| and |F_c| are invariant, and so is U_L. Delta_u is invariant (Lemma SYM of Section
8.6). k_a and k_b are exchanged and k_ab is fixed, so E^lin, the NS row and the lower rows are invariant, and THIN and the R7
condition are symmetric. For V, the star is a mirror star, whose a, b are the original c, d: the swap exchanges (g_c, l_c) with
(g_d, l_d) inside each profile term of R_v^slice, and the maximum over the square [1/N_max, 1/N0]^2 of (r_c, r_d), which is
symmetric under exchanging its coordinates, is unchanged; ghat depends on the class only. So R_v^slice and V are invariant.
QED

So (U-LR*) and (V-LR) on the roots with dm >= 0 imply them for every star that Theorem LR uses. (The engine's linear forms are
not symmetric themselves, only through the slack of the relaxation; this is why the lemma is stated for the true functionals.)

**The driver's changes, all sound.**
- *Lemma SUPERSET-RANGES.* Let S'(z) contain the full slice-level set (with the NS row) for all z in Z. If ranges are certified on
  S'(z), the upper forms built from them are valid on every check set S_e(z) (a subset of S'(z)); and if some certified range is
  empty, every S_e(z) is empty. (The driver relaxes the NS row to E^lin <= capE with capE >= B when that helps; it is stored as
  'rcap' and re-checked on replay.) PROOF: the forms use the ranges only through "f.m lies in [a_f, b_f]" for the stars being
  bounded.
- *More candidate duals* (the NS multiplier capped at 3, 0.3 and 30). A candidate is only a proposal; weak duality holds for every
  dual with nonnegative inequality multipliers on the box, which the replay checks.
- *Memoised DUB.* The kill bound is a deterministic function of the box and the ranges, cached per box. A
  second cache keyed by id(Q) without keeping Q alive is unsafe when the DUB memo is switched off (by an environment
  variable): then 33 of
  74 tested replays differed. With the default setting no id is reused (0 of 74 differ). So every run and every replay must use
  the default environment; the replay script asserts it.
- *Lemma SUB-BOX.* Cut Z at the midpoints of all four coordinates into 16 closed sub-boxes (they cover Z). If ranges are certified on
  a box Z' containing Z (with Ybar's range widened to [min(Ylo, 0), Yhi] and pq(Z_j) <= pq(Z') checked), and a Mode 1 certificate
  holds on each Z_j, then the statement holds on Z (it is pointwise in z).

### 9.6 The targets and the exact composition

**The targets file.** (final_targets.json.) Ten U checks with lower ends none, 5e-4, 1e-3,
1.4e-3, 1.7e-3, 1.9e-3, 2.1e-3, 2.2e-3, 2.3e-3, 2.4e-3; L_j = Lcut(e_j) (L rounded up); ell_j(x) = 1.08 (K_j + lam_j x) + c_j with
c_j = (1.0, 2.0, 2.5, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5) x 1e-3; and one V check, ell(x) = 1.07 (0.000443 + 23 x) + 0.001. (K_j and
lam_j came from numerical envelopes, sup over u-stars of U_{L(e_j)} - lam_j E^lin with lam_j the chord slope, 43.0 to 44.0. Their
origin does not matter for soundness: the certificate checks the doubles in the file.) The composition must
be evaluated on the exact rational values of the doubles in the file actually used, which every record identifies.

**Finite claim FC-CLR ((C-LR) for final_targets.json).** (computer-checked: exact rational arithmetic on Fraction(float) of the
file's doubles; comp_r2.py, with the assertion 0 < e < B) G has 11 points; 31 cells of
G x G meet the triangle x + w y <= B; at all 124 polygon vertices the four values lie in [0.001474, 0.115220] (so A, A' <= 0.118424);
all 9 finite lower ends satisfy 0 < e < B; every L_j >= L(max(e_j, 0)) exactly; t1 >= 0 for every check; each family has one
check with e = none; and

    max over the 124 vertices of Disc/Disc_0 = 0.984377159374...   (attained at x = 0.0006177, y = 0.0019),

an exact rational with a 166-digit numerator. The constants enter as
exact rationals on the safe side: B = 0.0024634531, w = 0.97147, k_hi := 3 (0.9595702342051) - 2 (0.9491) (the upper end of
c_Theta), den := 4 (1 - k_hi)(1 + 0.9491), c2* = 0.8643473 (rounded up) and c2' = 0.61536 (rounded down), so Disc is
overestimated and Disc_0 underestimated.

So the composition has a margin of 1.56% of Disc_0. This is the thinnest margin of the whole chain at the level of the design.

### 9.7 The 640-root certificate

**Finite claim FC-LR ((U-LR*) and (V-LR) on all roots).** (computer-checked)
- *Domain.* 640 roots = the 160 marginal cells of the window (the grid of FC-EMIN, Q included) x the two off-face levels
  ([0, 0.01], [0.01, 1]) x the two families (U on u-star slices, V on mirror slices), each over the d-box
  [-0.79, 0.99] x [0, 0.59] in (dp, dm) (dm >= 0 by Lemma SYM; the R7 square at every marginal of the window fits inside). A root
  is *done* when every check of its family (10 for U, 1 for V) is certified on the whole root box: per check, the records of the
  root tile the root box.
- *Records.* 1,495,347 records in 14 files: T_w0.jsonl ... T_w12.jsonl and T_w99.jsonl (the 7 roots of a pilot run,
  jobs 11, 45, 202, 246, 252, 396, 473: 12,135 records). Each record stores, among other fields, its root job, family, level,
  an identifier of the targets file ('tg'), the box, and per check either 'O' (a value certificate: candidate duals, weights, ranges with their certificates,
  sub-box entries, 'rcap' where used), 'E' (the check set is empty: a Farkas-type certificate) or 'E2' (a second emptiness
  form), or a split.
- *Replay.* Every record was replayed exactly once in the default environment by bk24_replay.py with the verifier
  bk23_orun.replay (the engine sources psd/src/bk19_orun.py, bk19_obk.py,
  bk19_lb2.py): 35 replay summaries, every one with 0
  failures. Least
  certified margins: 'O' U 1.418327e-9, V 4.539539e-10 (both > 1e-18, as Section 9.5 requires); 'E' 5.362320e-6 (U and V); 'E2'
  1.295747e-3 (U), 2.367091e-3 (V).
- *Cover.* bk23_cover.py: COVER COMPLETE; records 1,495,347; duplicates 0; overlaps 0; record box differs from path box 0; 640 of
  640 roots fully covered; open 0. acover_bk24.py (a second cover code): 640 of 640
  COVERED, 0 bad fields (family, level and targets identifier equal to the root job's), exact volume for every check.

### 9.8 Proof of Theorem PSD-N

```text
<1>1. (U-LR*) and (V-LR) hold for final_targets.json on all 640 roots, for the dm >= 0 half of the d-box (FC-LR).
<1>2. By Lemma SYM they hold for every u-star and mirror star that Theorem LR uses: the roots cover the window x the d-box x both
      levels x both families, and every u-star and mirror star of a point of N lies there (Lemmas WIN, R7, THIN).
<1>3. (C-LR) holds exactly for the same targets file, with 0 < e < B (FC-CLR).
<1>4. Theorem LR: the kill is PSD at every m in N.   QED
```

---

## 10. Theorem R2, Theorem M_tree, and Theorem 4.6155

### 10.1 The assembly

**Theorem R2.** (from Theorems K-N and PSD-N) (R-ii*) holds at every m in N.

PROOF. Let m be in N. By Theorem K-N, (K) holds at m. By Theorem PSD-N, the kill C'(m) is PSD. By Lemma KW, the kill is a
witness, so (R-ii*) holds at m.   QED

**Theorem M_tree.** (from Theorems R1 and R2) M_123 = M_tree.

PROOF. Theorem R1 is (R-i'). Theorem R2 is (R-ii*) on N. By Proposition NEED (which uses (R-i') through Theorem TS*),
M_123 = M_tree.   QED

Unwound at a single cone point m: if RB(m) < M_lo, the rule bound gives f(m) < M_tree. Otherwise m is in N; the kill C'(m) is
PSD (Theorem PSD-N) and still satisfies the tree cut (Theorem K-N), so by Lemma G(b) f(m) <= Phi_F(C'(m)), and by Theorem R1
Phi_F(C'(m)) <= M_tree.

### 10.2 Proof of Theorem 1

**Theorem 1 (Theorem 4.6155).** (uses the finite claims named in Sections 4-9) For every eps > 0 and all large n,
every B2 circuit computing f_n has at least (1 + 1/M_tree - eps) n gates, and

    liminf_{n -> oo} s_n/n  >=  1 + 1/M_tree  >=  1 + 1/Mt_hi  >=  4.6155432392.

PROOF IDEA. Corollary A1 turns the value of M_123 into the circuit constant. Theorem M_tree supplies the value. The rest is
arithmetic, rounded down.

```text
<1>1. By Theorem M_tree, M_123 = M_tree.
<1>2. M_tree > 0 (M_tree >= Mt_lo > 0, FC-MT; or Corollary A1 (ii)). Corollary A1 (iii) gives liminf s_n/n >= 1 + 1/M_tree.
<1>3. t -> 1 + 1/t is decreasing on (0, oo) and M_tree <= Mt_hi = 0.27658360966404055 (FC-MT), so
      1 + 1/M_tree >= 1 + 1/Mt_hi = 4.6155432392204148253748983661931265519028... >= 4.6155432392 (Section 10.3).
<1>4. Fix eps > 0. By <1>2 there is n_0 with s_n/n > 1 + 1/M_tree - eps for all n >= n_0, and every B2 circuit computing f_n
      has at least s_n gates.   QED
```

*The edge-score comparison.* M_tree < 2 kappa_E: indeed 2 kappa_E - Mt_hi = 0.0041183276076068... > 0 (2 kappa_E =
0.28070193727164739269... is known to 200-bit interval precision, FC-KAPPA). So the min in the second form of Corollary A1 (Section 3.4, Remark),
1 + 1/min(2 kappa_E, M_123), is M_123, and the two forms give the same constant.

**Finite claim FC-KAPPA.** (computer-checked: 200-bit interval enclosures and a
verified-cosine enclosure) kappa_E = (3/(2 pi)) arccos((1 + 2 sqrt 2)/4) lies in an interval of width
< 1e-55 around 0.14035096863582369634609711995567682918... (Not an input of Theorem 1 in the form
proved here; see the comparison above.)

### 10.3 The arithmetic

All lines are exact rational arithmetic (Python Fractions) or 50-digit decimal arithmetic (mpmath), and the short ones were
also checked by hand.

```text
1 + 1/0.27658360966404055 = 4.6155432392204148253748983661931265519028455451777...
    as Fractions: 4.6155432392 <= 1 + 1/Fraction('0.27658360966404055') < 4.6155432393   (checked exactly)
1 + 1/0.276583606882373471022763934878 = 4.6155432755827926051806474365644869096...     (quadrature value of M_tree)
1 + 1/0.27658360687864314 = 4.6155432756315560637140731222234000727...                    (from Mt_lo: an upper bound of 1 + 1/M_tree)
1 + 1/0.27658360966 = 63829180483/13829180483 = 4.6155432392732335...                      (the 11-digit truncation; see below)

2 kappa_E - Mt_hi = 0.0041183276076068426921942399113536583601914727...
gain over 23/5: 1 + 1/Mt_hi - 4.6 = 0.0155432392204148...
gain over c_E = 1 + 1/(2 kappa_E) = 4.5624976789250185831386869456813746465050606100431...: 0.0530455602953962...
```

*Which number is rigorous.* A statement "liminf s_n/n >= c" needs c <= 1 + 1/M_tree, so c must come from a certified upper bound
of M_tree, rounded down. From Mt_hi: c = 4.6155432392 (10 decimals).

*The 11-digit truncation.* The upper bound "M_tree <= 0.27658360966" is 4.04e-12 below the computed bound
Mt_hi, so as an upper bound it is rounded the wrong way. Consequences: (a) the 10-decimal floor 4.6155432392 is certified either
way; (b) the exact fraction 63829180483/13829180483 = 1 + 1/0.27658360966 is NOT certified by Mt_hi (it would need
M_tree <= 0.27658360966). *Do not use 4.6155433*: it is 1 + 1/M_tree rounded to nearest, 2.4e-8 above the true value.

---

## References

- [Li] X. Li. Two source extractors for asymptotically optimal entropy, and (many) more. 64th Annual IEEE Symposium on
  Foundations of Computer Science (FOCS 2023), pages 1271-1281. Full version: arXiv:2303.06802v2. (Theorem 7.13 and
  Definition 1.1 are used.)
- [LY] J. Li and T. Yang. 3.1n - o(n) circuit lower bounds for explicit functions. Proceedings of the 54th Annual ACM Symposium
  on Theory of Computing (STOC 2022), pages 1180-1193. doi:10.1145/3519935.3519976.
- [FGHK] M. G. Find, A. Golovnev, E. A. Hirsch and A. S. Kulikov. A better-than-3n lower bound for the circuit complexity of an
  explicit function. 57th Annual IEEE Symposium on Foundations of Computer Science (FOCS 2016).
- [FH] F. V. Fomin and K. Hoie. Pathwidth of cubic graphs and exact algorithms. Information Processing Letters 97(5) (2006),
  191-196. doi:10.1016/j.ipl.2005.10.012. (Mentioned in Section 1.2; not used.)
- [LM] L. Lichev and D. Mitsche. On the minimum bisection of random 3-regular graphs. Electronic Journal of Combinatorics 30(2)
  (2023), P2.40. (a.a.s. 0.103295 n <= bw(G(n, 3)) <= 0.139822 n.)
- [Diestel] R. Diestel. Graph Theory. Corollary 2.2.2 (Petersen's theorem: every bridgeless cubic graph has a 1-factor).
- [ES63] P. Erdos and H. Sachs (1963), on regular graphs of given girth (Fact GIRTH).
- [paper] Explicit functions requiring circuits larger than 4.6n (the paper this repository accompanies): the reduction
  from circuits to the cutwidth of cubic graphs, reproved in Section 2 here.

