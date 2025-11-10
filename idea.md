The Hierarchical Butterfly–Sketch (HBS) structure can be adapted into a **Butterfly-Flow** layer by simplifying the overall matrix $M$ to ensure the crucial properties for Normalizing Flows (NFs): **invertibility** and a **tractable log-Jacobian determinant**.

A Normalizing Flow is built by composing a sequence of invertible transformations $f = f_K \circ \dots \circ f_1$ that map a complex data distribution $x$ to a simple base distribution $z$ (e.g., Gaussian). The log-likelihood is calculated via the change of variables formula:
$$\log p_X(x) = \log p_Z(z) + \sum_{i=1}^K \log\left|\det\left(\frac{\partial f_i}{\partial x_{i-1}}\right)\right|$$

The linear layers derived from HBS become the core component $f_i(x) = M x$.

***

## 1. Designing the Butterfly-Flow Linear Layer

The full HBS matrix is $M \approx B_L S B_R^\top + \sum U_i V_i^\top$. For a tractable Normalizing Flow layer, we must simplify this to ensure an easy determinant calculation.

### HBS-Flow Core Design
We design the linear layer **$f(x) = M_{\text{Flow}} x$** based on the most computationally efficient part of HBS:

$$M_{\text{Flow}} = B_L S B_R^\top$$

Where:
* **$B_L, B_R$** are the **Butterfly-Structured Matrices**. To ensure maximum speed and stability, these factors should be initialized to be *orthogonal-ish* or based on fast, structured transforms (like the Fast Wavelet Transform or Discrete Fourier Transform structure).
* **$S$** is the **small dense core**.
* The **Low-Rank Correction Term ($\sum U_i V_i^\top$) is excluded** from the core linear layer. Including this term would make the Jacobian determinant $\log|\det(M)|$ untractable (i.e., too expensive to compute, destroying the $\mathcal{O}(n \log n)$ advantage).

***

## 2. Ensuring Invertibility and Tractable Jacobian

For the $M_{\text{Flow}}$ layer to be a valid NF component, three conditions must be met:

### A. Invertibility
Invertibility requires that $\det(M_{\text{Flow}}) \ne 0$. Since $M_{\text{Flow}}$ is a product of three matrices ($B_L$, $S$, and $B_R^\top$), all three must be invertible:
$$M_{\text{Flow}}^{-1} = (B_R^\top)^{-1} S^{-1} B_L^{-1} = B_R^{-\top} S^{-1} B_L^{-1}$$
* **$B_L, B_R$ Factors:** Butterfly factorization is a product of $\mathcal{O}(\log n)$ sparse, structured matrices. As long as the small $2\times 2$ blocks in each sparse factor are invertible, the overall butterfly matrix is invertible. The inverse is also an $\mathcal{O}(n \log n)$ fast transform.
* **$S$ Core:** Since $S$ is a small $k \times k$ dense matrix ($k \ll n$), its invertibility is maintained by ensuring its weights are not degenerate during training.

### B. Tractable Log-Determinant
The $\log|\det(M_{\text{Flow}})|$ must be cheap to compute, ideally $\mathcal{O}(n \log n)$ or better. Using the property of matrix products:

$$\log|\det(M_{\text{Flow}})| = \log|\det(B_L)| + \log|\det(S)| + \log|\det(B_R^\top)|$$

1.  **$\log|\det(B_L)|$ and $\log|\det(B_R^\top)|$:** Since the butterfly matrices are a product of sparse, log-depth layers, their determinants are the product of the determinants of their small constituent blocks. This can be calculated in **$\mathcal{O}(n \log n)$** time or less.
2.  **$\log|\det(S)|$:** The core $S$ is small ($k \times k$). Its determinant is calculated by a standard LU decomposition in **$\mathcal{O}(k^3)$** time. Since $k$ is small (e.g., $k \approx 64$), this cost is negligible compared to $n$.

The total complexity of both the forward pass and the log-Jacobian calculation is **$\mathcal{O}(n \log n)$**, achieving the efficiency goal.

***

## 3. Building the Butterfly-Flow Architecture

The final generative model will be a deep stack of these layers, interspersed with non-linear operations.

1.  **HBS-Flow Block:** The core linear operation $y = B_L S B_R^\top x$.
2.  **Non-Linearity:** An element-wise non-linear activation function (e.g., ReLU or $\tanh$) is applied to $y$ and scaled (or coupled) in a structured way.
3.  **Full Flow Architecture:** The model is an alternating stack of these blocks:
    $$f(x) = \text{Activation} \circ M_{\text{Flow}, K} \circ \text{Activation} \circ \dots \circ M_{\text{Flow}, 1}(x)$$
    This architecture would allow the butterfly factors to model the **global, fast-operator structures** while the small dense cores $S$ learn the **reduced-dimensional feature couplings**.

***

## 4. End-to-End Training Procedure

The **"Integration in Neural Nets (End-to-End)"** procedure from the R\&D document is directly applicable here.

1.  **Initialization:** Initialize all parameters ($B_L, B_R, S$, and non-linear weights) of the stacked HBS-Flow blocks. The butterfly factors should be initialized to maintain good numerical stability (e.g., near-orthogonal).
2.  **Forward Pass:** Pass the input data $x$ through the stacked blocks to get the latent variable $z$.
    $$z = f(x)$$
3.  **Backpropagation:** Calculate the loss based on the log-likelihood objective:
    $$\mathcal{L} = -\left[\log p_Z(z) + \sum_{i=1}^K \log|\det(M_{\text{Flow}, i})|\right]$$
4.  **Optimization:** Use gradient descent (e.g., Adam) to **backpropagate through all parts**: the butterfly factors' sparse weights, the small dense core $S$, and any non-linear layer parameters. "Because transforms are structured, **training remains scalable**".
5.  **Regularization:** Optionally, regularize the factors of $B_L$ and $B_R$ towards orthogonality to improve flow stability and invertibility over deep stacks.







<img width="678" height="521" alt="Screenshot 2025-11-10 175153" src="https://github.com/user-attachments/assets/255214d2-2ecd-42a3-9d5d-f5bd8a65edfc" />

