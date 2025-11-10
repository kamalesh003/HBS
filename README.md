# HBS


## Hierarchical Butterfly–Sketch (HBS) Representation

**Rough intuition:** Factor a large dense matrix $M \in \mathbb{R}^{n \times n}$ into three parts simultaneously capturing (1) **global fast-operator structure** (butterfly / FFT-like), (2) reduced **“core” interactions** via a tiny dense sketch, and (3) **local / near-field corrections** via low-rank blocks. Multiplication then becomes a sequence of cheap structured transforms + a tiny dense multiply + a few low-rank updates.

**Concretely propose:**


<img width="248" height="85" alt="Screenshot 2025-11-10 165510" src="https://github.com/user-attachments/assets/57cc4c0b-339a-4d9b-90ce-199025a54ec6" />


where:

* $B_L, B_R$ are **butterfly-structured matrices** (log-depth, near-linear nonzeros). They act like fast orthogonal-ish transforms (think FWT/FJLT/butterfly nets). (fast multiply in O(n log n).
* $S \in \mathbb{R}^{k \times k}$ is a **very small dense core** (with $k \ll n$) that captures the main coupling after projection into the butterfly domain. We obtain $S$ by sketching/projecting rows/cols into $k$-dim subspaces (JL/TensorSketch style).
* $\sum_{i=1}^p U_i V_i^\top$ is a **sum of local low-rank corrections** (blockwise low-rank / hierarchical matrix idea) that fixes near-field interactions that the coarse factor misses. Each $U_i, V_i$ has small column dimension $r$. This is the hierarchical / H-matrix element.

**Key novelty:** use **butterfly transforms** as the global basis (so the dense matrix is “compressible” by fast log-depth transforms) and place the compressed interaction into a learned **tiny core $S$** obtained via randomized projection (**sketch**) — then correct residuals with a **hierarchical low-rank bank**. This mixes three different compression philosophies into a single operator that is still linear and easy to multiply with.



This is a detailed breakdown of the unique aspects and the operational algorithm of the **Hierarchical Butterfly–Sketch (HBS) Representation**.

Here is the requested text, ready for you to copy:

---

## 🌟 Why Hierarchical Butterfly–Sketch (HBS) is a New Point of View

The HBS decomposition is novel because it combines three distinct compression philosophies into a single, high-speed, multi-resolution operator.

* **Operator-First Decomposition:** Rather than approximating entries or individual weights, we model the **action of the matrix** ($M x$) via transforms $B_R$ and $B_L$ that are cheap to apply (like fast FFT-like layers). That makes multiplication a composition of fast transforms and a tiny dense multiply — a conceptual shift from “sparse weights” to **“fast-change-of-basis + small core.”**
* **Sketchable Core:** The $S$ matrix is learned in a subspace of dimension $k$ (JL/TensorSketch ideas) so the only heavy $O(k^2)$ part is tiny; we use random or learned sketches to find that subspace. This emphasizes **preserving geometry of the action** rather than per-entry fidelity.
* **Hierarchical Low-Rank Residuals:** Errors that are **localized (near-field)** are corrected by a small bank of low-rank blocks ($\sum U_i V_i^\top$) — infusing hierarchical matrix benefits (accuracy where needed) without losing global speed.

---

## 🧮 Math: Multiplication Algorithm $y = M x$ with HBS

Given vector $x \in \mathbb{R}^n$:

1.  **Right Butterfly Transform:**
    $$z = B_R x$$
    — fast $O(n \log n)$ butterfly transform.
2.  **Sketch and Core Multiply:**
    $$w = S z_{[1:k]}$$
    — here we either (a) treat $B_R$ as producing an effective $k$-dimensional compressed vector (if butterfly projects to small subspace), or (b) precompute a linear projection $P$ so $z_k = P z$. Cost $O(k^2)$.
3.  **Left Butterfly Expansion:**
    $$u = B_L ( \text{expand}(w) )$$
    — expand back using $B_L$ (fast $O(n \log n)$).
4.  **Add Corrections:**
    $$y = u + \sum_{i=1}^{p} U_i (V_i^\top x)$$
    — for each low-rank term $U_i V_i^\top$, add $U_i (V_i^\top x)$ — each costs $O(nr)$ with small $r$ and small $p$.

**Total cost** $\approx O(n \log n + k^2 + p\,nr)$. Choose $k, p, r$ to balance accuracy vs cost.

*(If $k$ grows like $\log n$ and $pr$ is small, we get near $O(n \log n)$ complexity.)*
