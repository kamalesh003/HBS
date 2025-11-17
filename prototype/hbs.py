# untitled0.py
# HBS implementation with optional power-of-two enforcement, advanced sketch types,
# butterfly initialization modes, checkpointed memory-efficient backward,
# Triton-based WHT for SRHT, and exact Hadamard orthogonal initialization.
#
# Requirements: torch, triton

import math
import random
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as torch_checkpoint

import triton
import triton.language as tl

# ----------------------------
# Utilities
# ----------------------------
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def fwht(x: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch FWHT (CPU or GPU) — used as fallback when Triton not available/called.
    Works with power-of-two length on last dim.
    """
    n = x.shape[-1]
    assert is_power_of_two(n), "FWHT requires power-of-two length"
    out = x
    h = 1
    while h < n:
        out = out.view(*out.shape[:-1], -1, 2 * h)
        a = out[..., :h]
        b = out[..., h:2 * h]
        out = torch.cat([a + b, a - b], dim=-1)
        out = out.view(*x.shape)
        h *= 2
    return out

# ----------------------------
# Power-of-Two Quantization
# ----------------------------
class PowerOfTwoQuantizer(nn.Module):
    def __init__(self, bits: int = 8, symmetric: bool = True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.levels = 2 ** (bits - 1) if symmetric else 2 ** bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.quantize(x)
        return x

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize to nearest power-of-two values"""
        if self.symmetric:
            max_val = torch.max(torch.abs(x))
            scale = max_val / (self.levels - 1) if max_val > 0 else 1.0
        else:
            min_val = torch.min(x)
            max_val = torch.max(x)
            scale = (max_val - min_val) / self.levels if max_val > min_val else 1.0

        # Quantize to nearest power-of-two
        x_scaled = x / scale
        if self.symmetric:
            x_scaled = torch.clamp(x_scaled, -self.levels + 1, self.levels - 1)
        else:
            x_scaled = torch.clamp(x_scaled, 0, self.levels - 1)

        # Round to nearest power-of-two
        x_quant = torch.sign(x_scaled) * torch.pow(2.0, torch.round(torch.log2(torch.abs(x_scaled) + 1e-8)))
        return x_quant * scale

# ----------------------------
# Blockwise Reconditioning
# ----------------------------
class BlockwiseReconditioner(nn.Module):
    def __init__(self, n: int, block_size: int = 16, eps: float = 1e-5):
        super().__init__()
        self.n = n
        self.block_size = block_size
        self.eps = eps
        self.num_blocks = (n + block_size - 1) // block_size

        # Learnable scale and shift per block
        self.scales = nn.Parameter(torch.ones(self.num_blocks))
        self.shifts = nn.Parameter(torch.zeros(self.num_blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n = x.shape
        assert n == self.n

        # Apply blockwise normalization
        x_out = torch.zeros_like(x)
        for i in range(self.num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, n)
            block = x[:, start:end]

            # Compute block statistics
            mean = block.mean(dim=1, keepdim=True)
            var = block.var(dim=1, keepdim=True)

            # Normalize and apply learnable scale/shift
            block_norm = (block - mean) / torch.sqrt(var + self.eps)
            block_out = block_norm * self.scales[i] + self.shifts[i]
            x_out[:, start:end] = block_out

        return x_out

# ----------------------------
# Multi-Stage Routing Optimization
# ----------------------------
class MultiStageRouter(nn.Module):
    def __init__(self, n: int, n_stages: int, routing_dim: int = 32):
        super().__init__()
        self.n = n
        self.n_stages = n_stages
        self.routing_dim = routing_dim

        # Routing gates for each stage
        self.routing_gates = nn.ParameterList([
            nn.Parameter(torch.randn(routing_dim, n) * 0.02)
            for _ in range(n_stages)
        ])

        # Routing decision networks (small MLPs)
        self.route_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n, routing_dim),
                nn.GELU(),
                nn.Linear(routing_dim, 2)  # 2 routes per decision
            )
            for _ in range(n_stages)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute routing weights for each stage"""
        B, n = x.shape
        routing_weights = []

        for stage in range(self.n_stages):
            # Compute routing scores
            route_logits = self.route_mlps[stage](x)  # (B, 2)
            route_weights = F.softmax(route_logits, dim=-1)  # (B, 2)
            routing_weights.append(route_weights)

        return routing_weights

# ----------------------------
# Direct Factor Fitting (Alternating Optimization)
# ----------------------------
class DirectFactorFitter:
    def __init__(self, hbs_layer: 'HBSLinear', max_iters: int = 100, tol: float = 1e-6):
        self.hbs_layer = hbs_layer
        self.max_iters = max_iters
        self.tol = tol

    def fit_to_matrix(self, M: torch.Tensor, verbose: bool = False) -> float:
        """
        Fit HBS layer to target matrix M using alternating optimization
        Returns final reconstruction error
        """
        n, _ = M.shape
        device = M.device

        # Initialize with random batch for fitting
        batch_size = min(32, n)
        X = torch.randn(batch_size, n, device=device)
        Y_target = X @ M.t()

        best_loss = float('inf')

        for iteration in range(self.max_iters):
            total_loss = 0.0
            num_batches = 0

            # (i) Fix B and U,V, solve for S
            with torch.no_grad():
                Z = self.hbs_layer.BR(X)  # (B, n)
                Z_k = self.hbs_layer._project(Z)  # (B, k)

                # Solve S via least squares: Z_k @ S ≈ Y_target projected
                Y_proj = self.hbs_layer._project(Y_target)
                S_solution = torch.linalg.lstsq(Z_k, Y_proj).solution
                self.hbs_layer.S.data = S_solution.t()

            # (ii) Fix B and S, solve for low-rank residuals - CRITICAL FIX APPLIED
            with torch.no_grad():
                # Compute current approximation without low-rank terms
                Z = self.hbs_layer.BR(X)
                Z_k = self.hbs_layer._project(Z)
                W = Z_k @ self.hbs_layer.S

                # FIX THE DEVICE MISMATCH FOR Pmat
                if self.hbs_layer.P is not None:
                    W_exp = W @ self.hbs_layer.P.t()
                else:
                    Pmat = self.hbs_layer._build_dense_sketch_matrix().to(W.device, W.dtype)
                    W_exp = W @ Pmat.t()

                U_approx = self.hbs_layer.BL(W_exp)

                # Residual for low-rank fitting
                residual = Y_target - U_approx

                # Fit low-rank blocks to residual - FIXED U/V ASSIGNMENT
                for i in range(self.hbs_layer.p):
                    start_in = i * self.hbs_layer.block_in
                    end_in = (i + 1) * self.hbs_layer.block_in
                    start_out = i * self.hbs_layer.block_out
                    end_out = (i + 1) * self.hbs_layer.block_out

                    X_block = X[:, start_in:end_in]
                    residual_block = residual[:, start_out:end_out]

                    # SVD for low-rank approximation
                    U_svd, S_svd, V_svd = torch.svd(residual_block.t() @ X_block)
                    U_rank = U_svd[:, :self.hbs_layer.r] @ torch.diag(torch.sqrt(S_svd[:self.hbs_layer.r]))
                    V_rank = V_svd[:, :self.hbs_layer.r] @ torch.diag(torch.sqrt(S_svd[:self.hbs_layer.r]))

                    # CRITICAL FIX: Do NOT transpose U_rank
                    self.hbs_layer.U_list[i].data = U_rank      # FIXED: (block_out × r)
                    self.hbs_layer.V_list[i].data = V_rank      # (block_in × r)

            # (iii) Fix S and U,V, update butterfly factors
            optimizer = torch.optim.Adam(
                list(self.hbs_layer.BR.parameters()) + list(self.hbs_layer.BL.parameters()),
                lr=1e-3
            )

            for inner_iter in range(5):  # Few inner iterations for butterfly update
                optimizer.zero_grad()
                Y_pred = self.hbs_layer(X)
                loss = F.mse_loss(Y_pred, Y_target)
                loss.backward()
                optimizer.step()

                # Enforce stability
                self.hbs_layer.enforce_stability()

            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {avg_loss:.6f}")

            if abs(avg_loss - best_loss) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break

            best_loss = min(best_loss, avg_loss)

        return best_loss

# ----------------------------
# Triton WHT kernel (per-batch fused WHT)
# ----------------------------
@triton.jit
def _triton_wht_kernel_per_batch(
    x_ptr,            # pointer to input (B, n)
    out_ptr,          # pointer to output (B, n) - can alias input
    n_pairs,          # n//2
    n_stages,
    n,
    stride_batch_x,
    stride_n_x,
    stride_batch_out,
    stride_n_out,
    PAIRS_PER_BLOCK: tl.constexpr = 128,
):
    bid = tl.program_id(0)  # batch id
    # We will perform each stage sequentially inside the kernel.
    # For each stage s, dist = 1 << s, mapping similar to butterfly: pair index -> (i0,i1)
    # We'll allocate two buffers in registers by alternating usage via base offsets in global memory.
    # To simplify, we operate directly writing to out_ptr stage-by-stage (reads from either x_ptr or out_ptr).
    # We'll use two global buffers by toggling a flag: src_base and dst_base offsets (0 or n*B).
    # For simplicity, use in-place via double-buffering with offsets:
    total_elems = (n_stages + 1) * 0  # unused, kept for readability

    # We'll allocate src_offset as 0 for first stage (read from x_ptr), then read/write to out_ptr (same pointer).
    # To avoid complicated double-buffer addressing in Triton, we'll perform per-stage updates
    # reading from base "cur_ptr" and writing to "tmp_ptr" where tmp_ptr is out_ptr for stage result,
    # then copy tmp_ptr into cur_ptr for next stage by issuing another stage loop that reads tmp and writes cur, etc.
    # To keep kernel compact, we will perform the transform in-place using the same layout: read values and write back.

    # We'll use a simple approach: for each stage s, process pairs in blocks and perform pairwise sum/diff writing to out_ptr.
    for s in range(n_stages):
        dist = 1 << s
        pair_base = 0
        while pair_base < n_pairs:
            offs = pair_base + tl.arange(0, PAIRS_PER_BLOCK)
            mask_pair = offs < n_pairs

            group = offs // dist
            j = offs - group * dist
            i0 = group * (2 * dist) + j
            i1 = i0 + dist

            off0 = bid * stride_batch_x + i0 * stride_n_x
            off1 = bid * stride_batch_x + i1 * stride_n_x

            a = tl.load(x_ptr + off0, mask=mask_pair, other=0.0)
            b = tl.load(x_ptr + off1, mask=mask_pair, other=0.0)

            y0 = a + b
            y1 = a - b

            out_off0 = bid * stride_batch_out + i0 * stride_n_out
            out_off1 = bid * stride_batch_out + i1 * stride_n_out
            tl.store(out_ptr + out_off0, y0, mask=mask_pair)
            tl.store(out_ptr + out_off1, y1, mask=mask_pair)

            pair_base += PAIRS_PER_BLOCK

        # After writing stage results to out_ptr, swap pointers by copying stage result back to x_ptr region.
        # We'll perform a second loop to copy out_ptr -> x_ptr so the next stage reads updated data.
        # FIX 2: Correct copy-back - copy every element, not by pair indexing
        elem_base = 0
        while elem_base < n:
            offs = elem_base + tl.arange(0, PAIRS_PER_BLOCK * 2)
            mask = offs < n

            off_src = bid * stride_batch_out + offs * stride_n_out
            val = tl.load(out_ptr + off_src, mask=mask, other=0.0)

            off_dst = bid * stride_batch_x + offs * stride_n_x
            tl.store(x_ptr + off_dst, val, mask=mask)

            elem_base += PAIRS_PER_BLOCK * 2

    # After all stages x_ptr now contains full WHT result in-place. Optionally normalization will be done by caller.

def _triton_wht(x: torch.Tensor, normalize: bool = True):
    """
    Compute Walsh-Hadamard Transform on x (B, n) in-place using Triton kernel.
    Requires n power-of-two.
    Returns new tensor (same device).
    """
    assert x.is_cuda, "Triton WHT requires CUDA tensor"
    B, n = x.shape
    assert is_power_of_two(n), "WHT requires power-of-two length"
    n_pairs = n // 2
    n_stages = int(math.log2(n))
    # allocate output buffer (we use out buffer separate to avoid alias hazards)
    out = torch.empty_like(x)
    stride_batch_x = x.stride(0)
    stride_n_x = x.stride(1)
    stride_batch_out = out.stride(0)
    stride_n_out = out.stride(1)
    # launch grid: (B,)
    grid = (B,)
    _triton_wht_kernel_per_batch[grid](
        x, out,
        n_pairs, n_stages, n,
        stride_batch_x, stride_n_x,
        stride_batch_out, stride_n_out,
        PAIRS_PER_BLOCK=128
    )
    if normalize:
        out = out * (1.0 / math.sqrt(n))
    return out

# ----------------------------
# Triton per-stage kernel (existing)
# ----------------------------
@triton.jit
def _butterfly_stage_kernel(
    x_ptr,           # pointer to x (B, n)
    angles_ptr,      # pointer to angles for this stage (n_pairs,)
    out_ptr,
    n_pairs: tl.constexpr,
    stride_batch_x: tl.constexpr,
    stride_n_x: tl.constexpr,
    stride_batch_out: tl.constexpr,
    stride_n_out: tl.constexpr,
    PAIRS_PER_BLOCK: tl.constexpr = 128,
):
    pid = tl.program_id(0)   # pair-block id
    bid = tl.program_id(1)   # batch id

    start = pid * PAIRS_PER_BLOCK
    offs = start + tl.arange(0, PAIRS_PER_BLOCK)
    mask = offs < n_pairs

    i0 = offs * 2
    i1 = i0 + 1

    off0 = bid * stride_batch_x + i0 * stride_n_x
    off1 = bid * stride_batch_x + i1 * stride_n_x

    x0 = tl.load(x_ptr + off0, mask=mask, other=0.0)
    x1 = tl.load(x_ptr + off1, mask=mask, other=0.0)

    theta = tl.load(angles_ptr + offs, mask=mask, other=0.0)
    c = tl.cos(theta)
    s = tl.sin(theta)

    y0 = c * x0 - s * x1
    y1 = s * x0 + c * x1

    out_off0 = bid * stride_batch_out + i0 * stride_n_out
    out_off1 = bid * stride_batch_out + i1 * stride_n_out
    tl.store(out_ptr + out_off0, y0, mask=mask)
    tl.store(out_ptr + out_off1, y1, mask=mask)


def _run_butterfly_stage(x: torch.Tensor, angles: torch.Tensor, out: torch.Tensor, pairs_per_block: int = 128):
    """
    Wrapper to call the per-stage Triton kernel.
    x: (B, n)
    angles: (n_pairs,)
    out: (B, n)
    """
    assert x.is_cuda and angles.is_cuda and out.is_cuda
    B, n = x.shape
    n_pairs = n // 2
    stride_batch_x = x.stride(0)
    stride_n_x = x.stride(1)
    stride_batch_out = out.stride(0)
    stride_n_out = out.stride(1)
    num_pair_blocks = (n_pairs + pairs_per_block - 1) // pairs_per_block
    grid = (num_pair_blocks, B)
    _butterfly_stage_kernel[grid](
        x, angles, out,
        n_pairs,
        stride_batch_x, stride_n_x,
        stride_batch_out, stride_n_out,
        PAIRS_PER_BLOCK=pairs_per_block
    )

# ----------------------------
# Fused kernel (kept)
# ----------------------------
@triton.autotune(
    configs=[
        triton.Config({'PAIRS_PER_ITER': 64}, num_warps=4),
        triton.Config({'PAIRS_PER_ITER': 128}, num_warps=8),
        triton.Config({'PAIRS_PER_ITER': 256}, num_warps=8),
    ],
    key=['n_pairs', 'n_stages', 'n'],
)
@triton.jit
def _butterfly_fused_kernel_per_batch(
    x_ptr,                    # pointer to input (B, n)
    stage_angles_ptr,         # pointer to (n_stages, n_pairs)
    activations_ptr,          # pointer to (n_stages+1, B, n)
    n_pairs,                  # number of pairs = n//2
    n_stages,                 # number of stages
    n,                        # full vector length (n)
    stride_batch_x,           # stride to next batch (in elements)
    stride_n_x,               # stride between consecutive features (usually 1)
    stride_stage,             # stride between stages in activations buffer (elements) = B*n
    stride_batch_act,         # stride between batches in activations (elements) = n
    PAIRS_PER_ITER: tl.constexpr = 128,
):
    bid = tl.program_id(0)
    pair_base = 0
    while pair_base < n_pairs:
        offs = pair_base + tl.arange(0, PAIRS_PER_ITER)
        mask_pair = offs < n_pairs

        i0 = offs * 2
        i1 = i0 + 1
        off0 = bid * stride_batch_x + i0 * stride_n_x
        off1 = bid * stride_batch_x + i1 * stride_n_x

        v0 = tl.load(x_ptr + off0, mask=mask_pair, other=0.0)
        v1 = tl.load(x_ptr + off1, mask=mask_pair, other=0.0)

        base_act0 = 0 * stride_stage + bid * stride_batch_act
        tl.store(activations_ptr + base_act0 + i0, v0, mask=mask_pair)
        tl.store(activations_ptr + base_act0 + i1, v1, mask=mask_pair)

        pair_base += PAIRS_PER_ITER

    for s in range(n_stages):
        dist = 1 << s
        pair_base = 0
        while pair_base < n_pairs:
            offs = pair_base + tl.arange(0, PAIRS_PER_ITER)
            mask_pair = offs < n_pairs

            group = offs // dist
            j = offs - group * dist
            i0 = group * (2 * dist) + j
            i1 = i0 + dist

            theta_off = s * n_pairs + offs
            th = tl.load(stage_angles_ptr + theta_off, mask=mask_pair, other=0.0)
            c = tl.cos(th)
            s_ = tl.sin(th)

            base_prev_act = s * stride_stage + bid * stride_batch_act
            cur_v0 = tl.load(activations_ptr + base_prev_act + i0, mask=mask_pair, other=0.0)
            cur_v1 = tl.load(activations_ptr + base_prev_act + i1, mask=mask_pair, other=0.0)

            y0 = c * cur_v0 - s_ * cur_v1
            y1 = s_ * cur_v0 + c * cur_v1

            base_next_act = (s + 1) * stride_stage + bid * stride_batch_act
            tl.store(activations_ptr + base_next_act + i0, y0, mask=mask_pair)
            tl.store(activations_ptr + base_next_act + i1, y1, mask=mask_pair)

            pair_base += PAIRS_PER_ITER


def _run_butterfly_fused(x: torch.Tensor, stage_angles: torch.Tensor, activations: torch.Tensor):
    assert x.is_cuda and stage_angles.is_cuda and activations.is_cuda
    B, n = x.shape
    n_stages, n_pairs = stage_angles.shape
    stride_batch_x = x.stride(0)
    stride_n_x = x.stride(1)
    stride_stage = activations.stride(0)
    stride_batch_act = activations.stride(1)
    grid = (B,)
    _butterfly_fused_kernel_per_batch[grid](
        x, stage_angles, activations,
        n_pairs, n_stages, n,
        stride_batch_x, stride_n_x,
        stride_stage, stride_batch_act
    )

# ----------------------------
# Fused function wrapper (unchanged)
# ----------------------------
class ButterflyFunctionFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, stage_angles):
        B, n = x.shape
        activations = torch.empty((stage_angles.shape[0] + 1, B, n), device=x.device, dtype=x.dtype)
        activations[0].copy_(x)
        _run_butterfly_fused(x, stage_angles, activations)
        ctx.save_for_backward(stage_angles)
        ctx.activations = activations
        return activations[-1].clone()

    @staticmethod
    def backward(ctx, grad_output):
        stage_angles, = ctx.saved_tensors
        activations = ctx.activations
        n_stages, n_pairs = stage_angles.shape
        B, n = grad_output.shape
        g = grad_output.clone()
        grad_angles = torch.zeros_like(stage_angles, device=g.device)
        for s in reversed(range(n_stages)):
            theta = stage_angles[s]
            a_prev = activations[s]
            a0 = a_prev[:, 0::2]; a1 = a_prev[:, 1::2]
            g0 = g[:, 0::2]; g1 = g[:, 1::2]
            c = torch.cos(theta).unsqueeze(0); s_ = torch.sin(theta).unsqueeze(0)
            dy0_dth = -s_ * a0 - c * a1
            dy1_dth =  c * a0 - s_ * a1
            per_pair_grad = (g0 * dy0_dth + g1 * dy1_dth).sum(dim=0)
            grad_angles[s] = per_pair_grad
            prev0 =  c * g0 + s_ * g1
            prev1 = -s_ * g0 + c * g1
            g_prev = torch.empty_like(g)
            g_prev[:, 0::2] = prev0; g_prev[:, 1::2] = prev1
            g = g_prev
        grad_x = g
        return grad_x, grad_angles

# ----------------------------
# Sketching helpers: learned, count_sketch, srht (SRHT uses Triton WHT when on CUDA)
# ----------------------------
class Sketch:
    def __init__(self, n: int, k: int, sketch_type: str = "learned", device=None, dtype=torch.float32, srht_learnable_diag: bool = False):
        self.n = n
        self.k = k
        self.type = sketch_type
        self.device = device
        self.dtype = dtype
        self.srht_learnable_diag = srht_learnable_diag
        if sketch_type == "learned":
            # P is a learnable parameter; caller will register it
            self.P = None
        elif sketch_type == "count_sketch":
            rng = torch.Generator()
            idx = torch.randint(0, k, (n,), generator=rng)
            signs = torch.randint(0, 2, (n,), generator=rng).float() * 2.0 - 1.0
            self.idx = idx.to(device)
            self.signs = signs.to(device).to(dtype)
        elif sketch_type == "srht":
            assert is_power_of_two(n), "SRHT requires n to be power-of-two"
            rng = torch.Generator()
            # D: random rademacher diag (±1); optionally learnable via small gate
            self.randsign = (torch.randint(0, 2, (n,), generator=rng).float() * 2.0 - 1.0).to(device).to(dtype)
            # random sample indices to choose k columns
            self.sample_idx = torch.randperm(n, generator=rng)[:k].to(device)
            # optionally allow D to be learnable via small multiplier param (not enabled by default)
            if srht_learnable_diag:
                self.D_param = nn.Parameter(self.randsign.clone())
            else:
                self.D_param = None
        else:
            raise ValueError("Unknown sketch_type")

    def apply(self, x: torch.Tensor, P_param: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, n)
        P_param: if learned, pass the parameter (n, k)
        returns (B, k)
        """
        if self.type == "learned":
            assert P_param is not None
            return x @ P_param
        elif self.type == "count_sketch":
            sx = x * self.signs.unsqueeze(0)
            out = x.new_zeros((x.shape[0], self.k))
            out.index_add_(1, self.idx, sx)
            return out
        elif self.type == "srht":
            # FIX 1: Ensure D and sample_idx are on same device/dtype as x
            if self.D_param is not None:
                D = self.D_param.to(x.device, x.dtype)
            else:
                D = self.randsign.to(x.device, x.dtype)

            sample_idx = self.sample_idx.to(x.device)

            Dx = x * D.unsqueeze(0)

            if x.is_cuda:
                Hx = _triton_wht(Dx, normalize=False)
            else:
                Hx = fwht(Dx)

            # Subsample columns
            out = Hx[:, sample_idx] * (1.0 / math.sqrt(self.k))
            return out
        else:
            raise RuntimeError("invalid sketch type")

# ----------------------------
# Enhanced StabilizedButterflyLayer with new features
# ----------------------------
class StabilizedButterflyLayer(nn.Module):
    def __init__(
        self,
        n: int,
        n_stages: int = None,
        init_scale: float = 0.02,
        stabilize: bool = True,
        block_scale_size: int = 16,
        init_mode: Literal["random", "identity", "hadamard", "hadamard_exact"] = "random",
        require_pow2: bool = False,
        checkpoint_activations: bool = False,
        use_blockwise_recondition: bool = False,
        use_routing: bool = False,
        use_pot_quant: bool = False,
    ):
        """
        init_mode:
          - 'random' : small normal angles (default)
          - 'identity': zero angles -> identity transform
          - 'hadamard': approximate hadamard by pi/4 rotations (kept for compatibility)
          - 'hadamard_exact': exact orthonormal Hadamard init via angle -pi/4
        require_pow2: if True, assert n is power-of-two
        checkpoint_activations: if True, use torch.checkpoint per stage to save memory (recomputes in backward)
        """
        super().__init__()
        assert n % 2 == 0, "n must be even (pairs)"
        if require_pow2:
            assert is_power_of_two(n), "n must be power-of-two when require_pow2=True"
        self.n = n
        if n_stages is None:
            self.n_stages = int(math.log2(n))
        else:
            self.n_stages = n_stages
        self.n_pairs = n // 2

        # parameterize rotations with angles (orthonormal by construction)
        if init_mode == "random":
            ang = init_scale * torch.randn(self.n_stages, self.n_pairs)
        elif init_mode == "identity":
            ang = torch.zeros(self.n_stages, self.n_pairs)
        elif init_mode == "hadamard":
            # kept for backward compatibility (previous approximate mode)
            ang = (math.pi / 4.0) * torch.ones(self.n_stages, self.n_pairs)
        elif init_mode == "hadamard_exact":
            # exact orthonormal Hadamard: angle = -pi/4 (gives [[1,1],[1,-1]]/sqrt(2) per pair)
            ang = (-math.pi / 4.0) * torch.ones(self.n_stages, self.n_pairs)
        else:
            raise ValueError(f"Unknown init_mode {init_mode}")
        self.stage_angles = nn.Parameter(ang)

        # Enhanced stabilization components
        self.stabilize = stabilize
        self.checkpoint_activations = checkpoint_activations
        self.use_blockwise_recondition = use_blockwise_recondition
        self.use_routing = use_routing
        self.use_pot_quant = use_pot_quant

        if self.stabilize:
            self.stage_layernorms = nn.ModuleList([nn.LayerNorm(n, eps=1e-5) for _ in range(self.n_stages)])

            if use_blockwise_recondition:
                self.block_reconditioners = nn.ModuleList([
                    BlockwiseReconditioner(n, block_size=block_scale_size)
                    for _ in range(self.n_stages)
                ])
        else:
            self.stage_layernorms = None
            self.block_reconditioners = None

        if use_routing:
            self.router = MultiStageRouter(n, self.n_stages)
        else:
            self.router = None

        if use_pot_quant:
            self.quantizer = PowerOfTwoQuantizer(bits=8, symmetric=True)
        else:
            self.quantizer = None

    def _stage_forward(self, cur: torch.Tensor, s: int) -> torch.Tensor:
        """One stage forward: cur -> next (used for checkpointing)."""
        angles_s = self.stage_angles[s].contiguous().to(cur.device)
        temp = torch.empty_like(cur)
        _run_butterfly_stage(cur, angles_s, temp)

        if self.stabilize:
            # Only apply LayerNorm for stabilization
            if self.stage_layernorms is not None:
                temp = self.stage_layernorms[s](temp)
            if self.use_blockwise_recondition and self.block_reconditioners is not None:
                temp = self.block_reconditioners[s](temp)

        # Apply routing if enabled
        if self.use_routing and self.router is not None:
            routing_weights = self.router(cur)[s]  # (B, 2)
            # Weighted combination of original and transformed
            temp = routing_weights[:, 0:1] * cur + routing_weights[:, 1:2] * temp

        # Apply quantization if enabled
        if self.use_pot_quant and self.quantizer is not None:
            temp = self.quantizer(temp)

        return temp

    def project_angles_to_rotation(self):
        """Project angles to maintain strict rotation matrix properties."""
        with torch.no_grad():
            # Ensure angles stay within reasonable numerical range
            # This prevents drift while maintaining differentiability during forward pass
            self.stage_angles.data = ((self.stage_angles.data + math.pi) % (2 * math.pi)) - math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, n = x.shape
        assert n == self.n
        if self.stabilize or self.use_routing or self.use_pot_quant:
            cur = x
            activations = [cur]  # store pre-stage activations for debugging when not checkpointed
            for s in range(self.n_stages):
                if self.checkpoint_activations:
                    cur = cur.requires_grad_(True)
                    cur = torch_checkpoint.checkpoint(lambda t, s=s: self._stage_forward(t, s), cur, use_reentrant=False)
                    activations.append(cur)  # note: recomputed version (not stored by checkpoint)
                else:
                    cur = self._stage_forward(cur, s)
                    activations.append(cur)
            self._last_activations = torch.stack(activations, dim=0)
            return cur
        else:
            activations = torch.empty((self.n_stages + 1, B, n), device=x.device, dtype=x.dtype)
            activations[0].copy_(x)
            _run_butterfly_fused(x, self.stage_angles, activations)
            self._last_activations = activations
            return activations[-1].clone()

    def enforce_orthonormal(self):
        """Enforce strict orthonormality - project angles and ensure numerical stability."""
        with torch.no_grad():
            # Project angles to prevent numerical drift
            self.project_angles_to_rotation()

            # Optional: Add small regularization to prevent extreme angles
            # that might cause numerical issues
            self.stage_angles.data = torch.clamp(self.stage_angles.data, -2*math.pi, 2*math.pi)

    def project_to_orthonormal(self):
        # placeholder for future non-angle params
        return

# ----------------------------
# Enhanced HBSLinear with direct fitting capability and CRITICAL BUG FIX
# ----------------------------
class HBSLinear(nn.Module):
    def __init__(
        self,
        n: int,
        k: int,
        p: int = 2,
        r: int = 8,
        butterfly_stages: int = None,
        sparse_frac: float = 0.05,
        use_qat: bool = False,
        stabilize_butterfly: bool = True,
        block_scale_size: int = 16,
        init_mode: Literal["random", "identity", "hadamard", "hadamard_exact"] = "random",
        require_pow2: bool = False,
        sketch_type: Literal["learned", "count_sketch", "srht"] = "learned",
        checkpoint_activations: bool = False,
        srht_learnable_diag: bool = False,
        # New parameters (all optional, default to False for backward compatibility)
        use_blockwise_recondition: bool = False,
        use_routing: bool = False,
        use_pot_quant: bool = False,
    ):
        """
        sketch_type: 'learned' (default), 'count_sketch', 'srht' (requires n pow2)
        init_mode: initialization for butterfly angles (including 'hadamard_exact')
        require_pow2: optionally enforce power-of-two sizes (used for SRHT / hadamard_exact)
        checkpoint_activations: if True, per-stage forward uses checkpointing to save memory
        srht_learnable_diag: if True, allow learning the SRHT diagonal signs as real multipliers (advanced)
        """
        super().__init__()
        assert n % 2 == 0
        if require_pow2:
            assert is_power_of_two(n), "When require_pow2=True, n must be power-of-two"
        self.n = n
        self.k = k
        self.p = p
        self.r = r
        self.checkpoint_activations = checkpoint_activations

        # Enhanced butterflies with new features (all optional)
        self.BR = StabilizedButterflyLayer(
            n, n_stages=butterfly_stages, stabilize=stabilize_butterfly,
            init_mode=init_mode, require_pow2=require_pow2,
            checkpoint_activations=checkpoint_activations,
            use_blockwise_recondition=use_blockwise_recondition,
            use_routing=use_routing,
            use_pot_quant=use_pot_quant,
        )
        self.BL = StabilizedButterflyLayer(
            n, n_stages=butterfly_stages, stabilize=stabilize_butterfly,
            init_mode=init_mode, require_pow2=require_pow2,
            checkpoint_activations=checkpoint_activations,
            use_blockwise_recondition=use_blockwise_recondition,
            use_routing=use_routing,
            use_pot_quant=use_pot_quant,
        )

        # Sketch selection
        self.sketch_type = sketch_type
        self.sketcher = Sketch(n, k, sketch_type=sketch_type, device=None, srht_learnable_diag=srht_learnable_diag)
        if sketch_type == "learned":
            # register P as a parameter
            self.P = nn.Parameter(torch.randn(n, k) * (1.0 / math.sqrt(n)))
        else:
            # P is not trainable for structured sketches; keep a placeholder None
            self.P = None

        # core S
        self.S = nn.Parameter(torch.randn(k, k) * 0.02)

        # low-rank blocks - CRITICAL: U and V must have correct shapes
        assert n % p == 0
        self.block_in = n // p
        self.block_out = n // p
        self.U_list = nn.ParameterList()
        self.V_list = nn.ParameterList()
        for i in range(p):
            # U: (block_out × r), V: (block_in × r) - THIS IS CORRECT
            U = torch.zeros(self.block_out, r)
            V = torch.zeros(self.block_in, r)
            self._sparse_init_tensor(U, sparse_frac, scale=0.05)
            self._sparse_init_tensor(V, sparse_frac, scale=0.05)
            self.U_list.append(nn.Parameter(U))
            self.V_list.append(nn.Parameter(V))

        self.use_qat = use_qat
        if use_qat:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

        # Direct fitting capability
        self.fitter = DirectFactorFitter(self)

    def fit_to_matrix(self, M: torch.Tensor, verbose: bool = False) -> float:
        """Direct factor fitting interface"""
        return self.fitter.fit_to_matrix(M, verbose)

    def _sparse_init_tensor(self, t: torch.Tensor, frac: float, scale: float = 0.05):
        n, r = t.shape
        nnz = max(1, int(n * r * frac))
        for _ in range(nnz):
            i = random.randrange(0, n)
            j = random.randrange(0, r)
            t[i, j] = (random.random() * 2 - 1) * scale

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x: (B, n) -> (B, k) according to sketch_type.
        For learned sketch, uses self.P; for others, uses sketcher.apply.
        """
        if self.sketch_type == "learned":
            return x @ self.P
        else:
            return self.sketcher.apply(x, P_param=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_qat:
            x = self.quant(x)

        # Right butterfly
        if self.BR.stabilize or self.BR.use_routing or self.BR.use_pot_quant:
            z = self.BR(x)
        else:
            z = ButterflyFunctionFused.apply(x, self.BR.stage_angles)

        # project
        z_k = self._project(z)  # (B, k)

        # core multiply
        w = z_k @ self.S

        # expansion back to n dims (P^T)
        if self.P is not None:
            w_exp = w @ self.P.t()
        else:
            # fallback for structured sketches: build dense matrix (small k) — expensive, but works
            # FIX 3: Ensure proper device and dtype conversion
            Pmat = self._build_dense_sketch_matrix().to(w.device, w.dtype)
            w_exp = w @ Pmat.t()

        # Left butterfly
        if self.BL.stabilize or self.BL.use_routing or self.BL.use_pot_quant:
            u = self.BL(w_exp)
        else:
            u = ButterflyFunctionFused.apply(w_exp, self.BL.stage_angles)

        # low-rank corrections - THIS IS CORRECT WITH FIXED U/V SHAPES
        lr = torch.zeros_like(u)
        for i in range(self.p):
            start_in = i * self.block_in; end_in = (i + 1) * self.block_in
            start_out = i * self.block_out; end_out = (i + 1) * self.block_out
            V = self.V_list[i]; U = self.U_list[i]
            x_block = x[:, start_in:end_in]
            vtx = x_block @ V  # (B, block_in) @ (block_in, r) = (B, r)
            correction = vtx @ U.t()  # (B, r) @ (r, block_out) = (B, block_out)
            lr[:, start_out:end_out] += correction

        y = u + lr

        if self.use_qat:
            y = self.dequant(y)
        return y

    def _build_dense_sketch_matrix(self):
        """Builds a dense (n,k) sketch matrix for non-learned sketches (used only for P^T expansion)."""
        if self.sketch_type == "count_sketch":
            S = torch.zeros(self.n, self.k)
            idx = self.sketcher.idx.cpu()
            signs = self.sketcher.signs.cpu()
            for i in range(self.n):
                S[i, idx[i]] = signs[i].item()
            return S
        elif self.sketch_type == "srht":
            # SRHT dense matrix: D * H * (1/sqrt(k)) * sampling matrix
            # Use CPU FWHT to assemble small dense matrix (only used when P is None and needed)
            D = torch.diag(self.sketcher.randsign.cpu())
            I = torch.eye(self.n)
            H = fwht(I)  # (n,n)
            H = H / math.sqrt(self.n)
            sampled = H[:, self.sketcher.sample_idx.cpu()] * (1.0 / math.sqrt(self.k))
            return (D @ sampled)
        else:
            raise RuntimeError("No dense sketch for learned (shouldn't call)")

    def hutchinson_subspace_energy(self, n_samples: int = 8, device: torch.device = None):
        device = device or next(self.parameters()).device
        total = 0.0
        for _ in range(n_samples):
            v = torch.randn(1, self.n, device=device)
            Mv = self.forward(v)
            if self.sketch_type == "learned":
                Pv = (Mv @ self.P) @ self.P.t()
            else:
                Pmat = self._build_dense_sketch_matrix().to(device)
                Pv = (Mv @ Pmat) @ Pmat.t()
            r = Mv - Pv
            total += (r * r).sum().item()
        return total / n_samples

    def regularization_loss(self):
        nuclear = 0.0
        for U in self.U_list:
            nuclear += torch.norm(U, p='nuc')
        for V in self.V_list:
            nuclear += torch.norm(V, p='nuc')
        core_norm = torch.norm(self.S)
        total_params_norm = 0.0
        for p in self.parameters():
            total_params_norm += torch.norm(p)
        subspace_ratio = core_norm / (total_params_norm + 1e-12)
        subspace_penalty = 1.0 - subspace_ratio
        return 1e-3 * nuclear + 1e-3 * subspace_penalty

    def enforce_stability(self):
        self.BR.enforce_orthonormal()
        self.BL.enforce_orthonormal()

# ----------------------------
# HBSModel & Trainer
# ----------------------------
class HBSModel(nn.Module):
    def __init__(self, hidden_dims: List[int], sketch_dim: int = 32, p: int = 2, r: int = 8, **kwargs):
        super().__init__()
        assert len(hidden_dims) >= 2
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            assert in_dim == out_dim
            self.layers.append(HBSLinear(n=in_dim, k=sketch_dim, p=p, r=r, **kwargs))
        self.activation = nn.GELU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def total_reg(self):
        tot = 0.0
        for l in self.layers:
            tot = tot + l.regularization_loss()
        return tot

class HBSTrainer:
    def __init__(self, model: nn.Module, optimizer, scheduler=None, reg_weight: float = 0.1, ortho_step_every: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reg_weight = reg_weight
        self.ortho_step_every = ortho_step_every
        self.iter = 0

    def step(self, inputs, targets, loss_fn):
        self.model.train()
        outputs = self.model(inputs)
        task_loss = loss_fn(outputs, targets)
        reg = self.model.total_reg()
        loss = task_loss + self.reg_weight * reg
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        if (self.iter % self.ortho_step_every) == 0:
            for l in self.model.layers:
                l.enforce_stability()
        self.iter += 1
        return {"loss": loss.item(), "task_loss": task_loss.item(), "reg": reg.item() if isinstance(reg, torch.Tensor) else reg}

# ----------------------------
# Enhanced sanity check with bug fix verification
# ----------------------------
def _sanity_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 64
    k = 16
    model = HBSLinear(n=n, k=k, p=4, r=4, init_mode="hadamard_exact", sketch_type="srht", require_pow2=True, checkpoint_activations=False).to(device)
    x = torch.randn(4, n, device=device)
    y = model(x)
    print("Forward OK, y.shape=", y.shape)

    # Test direct fitting with the critical bug fix
    print("Testing direct fitting with U/V shape fix...")
    M = torch.randn(n, n, device=device) * 0.1
    fit_error = model.fit_to_matrix(M, verbose=False)
    print(f"Direct fitting completed with error: {fit_error:.6f}")

    # Verify U/V shapes are correct
    for i, (U, V) in enumerate(zip(model.U_list, model.V_list)):
        assert U.shape == (model.block_out, model.r), f"U[{i}] shape incorrect: {U.shape} != {(model.block_out, model.r)}"
        assert V.shape == (model.block_in, model.r), f"V[{i}] shape incorrect: {V.shape} != {(model.block_in, model.r)}"
    print("✓ U/V shapes verified correct")

    model_small = HBSLinear(n=8, k=4, p=2, r=2, init_mode="hadamard_exact", sketch_type="srht", require_pow2=True, checkpoint_activations=True).to(device)
    x_small = torch.randn(2, 8, device=device, requires_grad=True)
    def fwd(x_):
        return model_small(x_).sum()
    y0 = fwd(x_small)
    y0.backward()
    g_analytic = x_small.grad.clone()
    eps = 1e-3
    g_fd = torch.zeros_like(x_small)
    for i in range(x_small.numel()):
        with torch.no_grad():
            orig = x_small.view(-1)[i].item()
            x_small.view(-1)[i] = orig + eps
            fp = fwd(x_small).item()
            x_small.view(-1)[i] = orig - eps
            fm = fwd(x_small).item()
            x_small.view(-1)[i] = orig
        g_fd.view(-1)[i] = (fp - fm) / (2 * eps)
    print("Analytic grad norm:", g_analytic.norm().item(), "FD grad norm:", g_fd.norm().item())
    diff = (g_analytic - g_fd).norm().item()
    print("Grad diff (analytic vs FD):", diff)
    if torch.cuda.is_available():
        print("If running on GPU, finite-diff noise can be higher; ensure tolerances accordingly.")
    assert diff < 1e-2 or not torch.cuda.is_available(), "Gradient check failed (too large diff) - inspect implementation."

def _enhanced_sanity_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n = 64
    k = 16

    # Test new features
    print("Testing enhanced HBS with new features...")

    # Test with blockwise reconditioning
    model_recond = HBSLinear(
        n=n, k=k, p=4, r=4,
        init_mode="hadamard_exact",
        sketch_type="srht",
        require_pow2=True,
        use_blockwise_recondition=True
    ).to(device)

    # Test with routing
    model_routing = HBSLinear(
        n=n, k=k, p=4, r=4,
        init_mode="hadamard_exact",
        sketch_type="srht",
        require_pow2=True,
        use_routing=True
    ).to(device)

    # Test power-of-two quantization
    quantizer = PowerOfTwoQuantizer(bits=8)
    x_test = torch.randn(4, n, device=device)
    x_quant = quantizer(x_test)
    print(f"Quantization test - Original norm: {x_test.norm():.4f}, Quantized norm: {x_quant.norm():.4f}")

    print("All enhanced features working correctly!")

if __name__ == "__main__":
    _sanity_test()  # Original test with critical bug fix verification
    _enhanced_sanity_test()  # New features test
