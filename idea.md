Universal Butterfly Network (Stacked + Learnable + Stabilized + Low-Rank)

This is the most powerful, most stable, and most universally applicable option.

If your goal is:

✅ approximate any dense matrix
✅ keep speed ~ O(n log n)
✅ minimize sketch dimension k
✅ minimize number of low-rank blocks
✅ avoid instability
✅ avoid hand-tuning patterns
✅ get consistent results across many matrices


Why it  is the Best (Precise Reasons)
# 1. Universal Approximation Capability

Stacking multiple learned butterfly layers + low-rank patches gives theoretical universality:

Can approximate any linear operator to arbitrary precision
using fast routing + flexible correction.

It’s the only option that guarantees coverage of all matrix types.

# 2. Automatic Structure Discovery = Small k

Because the butterfly stack learns a near-optimal basis:

global structure is captured quickly

interactions get concentrated

sketch core S becomes very small

fewer low-rank U,V blocks needed

Result:
You need much smaller k → faster, smaller HBS.

# 3. Stability

It includes:

orthonormal parameterization

layer normalization for butterfly stages

blockwise reconditioning

This eliminates:

exploding gradients

collapsed mixing

unstable transforms

Default butterflies → unstable.
Learned butterflies → also unstable.
Stabilized stacked butterflies → safe.


# 4. Efficient on GPU

It uses:

sparse kernel templates

fused block operations

regular block sizes

multi-stage routing

This makes it friendly to:

CUDA

Triton

XLA

ROCm

Modern butterfly kernels already support this (e.g., ButterflyNet, FNO).


<img width="678" height="521" alt="Screenshot 2025-11-10 175153" src="https://github.com/user-attachments/assets/255214d2-2ecd-42a3-9d5d-f5bd8a65edfc" />

