# HBS


## Hierarchical Butterfly–Sketch (HBS) Representation

**Rough intuition:** Factor a large dense matrix $M \in \mathbb{R}^{n \times n}$ into three parts simultaneously capturing (1) **global fast-operator structure** (butterfly / FFT-like), (2) reduced **“core” interactions** via a tiny dense sketch, and (3) **local / near-field corrections** via low-rank blocks. Multiplication then becomes a sequence of cheap structured transforms + a tiny dense multiply + a few low-rank updates.

**Concretely propose:**


$$M \approx B_L \, S \, B_R + \sum_{i=1}^{p} U_i V_i^\top \tag{HBS}$$

where:

* $B_L, B_R$ are **butterfly-structured matrices** (log-depth, near-linear nonzeros). They act like fast orthogonal-ish transforms (think FWT/FJLT/butterfly nets). (fast multiply in $O(n \log n)$).
* $S \in \mathbb{R}^{k \times k}$ is a **very small dense core** (with $k \ll n$) that captures the main coupling after projection into the butterfly domain. We obtain $S$ by sketching/projecting rows/cols into $k$-dim subspaces (JL/TensorSketch style).
* $\sum_{i=1}^p U_i V_i^\top$ is a **sum of local low-rank corrections** (blockwise low-rank / hierarchical matrix idea) that fixes near-field interactions that the coarse factor misses. Each $U_i, V_i$ has small column dimension $r$. This is the hierarchical / H-matrix element.

**Key novelty:** use **butterfly transforms** as the global basis (so the dense matrix is “compressible” by fast log-depth transforms) and place the compressed interaction into a learned **tiny core $S$** obtained via randomized projection (**sketch**) — then correct residuals with a **hierarchical low-rank bank**. This mixes three different compression philosophies into a single operator that is still linear and easy to multiply with.
