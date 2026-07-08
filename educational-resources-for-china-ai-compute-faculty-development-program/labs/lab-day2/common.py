"""
common.py
---------
Shared Triton matmul kernel and helpers used across Day 2 steps.

This is the reference baseline from Day 1 (04_matmul_compare.py) with all
TODO blocks filled in. Every step in Day 2 either uses this kernel as-is
(00, 03) or defines a small variant of it (01, 02). Keeping one canonical
kernel here lets you compare changes without re-reading the basics.

Nothing in this file is meant to be edited during the lab.
"""
import time
import contextlib
import os
import sys
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


# =============================================================================
# silent_stdio: suppress both stdout and stderr at the file-descriptor level
# =============================================================================
# Why this helper exists:
#
# triton-ascend 3.2 prints a `[WARNING] Please DO NOT tune args [...]` line
# once every time a `triton.Config` object is constructed with GPU-era
# defaults (num_warps, num_stages, Hopper-specific knobs). These defaults
# are set by `triton.Config.__init__` itself -- we cannot stop them from
# being there. On Ascend they are ignored, so the warning is informational
# and identical across all Configs. With a 170-entry autotune space, this
# is 170 copies of the same line and visually overwhelms the actual
# output.
#
# Python's `contextlib.redirect_stdout` is not enough here because the
# warning comes out of the C/C++ side of triton-ascend and writes directly
# to file descriptor 1. We have to redirect at the OS level: dup the real
# fd, point it at /dev/null for the duration of the block, restore on
# exit. Same for fd 2, for completeness.
@contextlib.contextmanager
def silent_stdio():
    """Redirect fds 1 and 2 to /dev/null. Use sparingly; see docstring."""
    # Make sure Python-buffered output is flushed before we swap fds.
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull)
        os.close(saved_out)
        os.close(saved_err)


# =============================================================================
# Baseline kernel (Day 1 reference answer)
# =============================================================================
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute one BLOCK_M x BLOCK_N tile of C = A @ B."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor,
                  BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    """Host-side launcher for matmul_kernel. BLOCK sizes are tunable."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "inner dims must match"
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# =============================================================================
# Benchmark helper (same pattern as Day 1)
# =============================================================================
def bench_ms(fn, *args, warmup=10, iters=50, **kwargs):
    """Return mean time per call in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def make_inputs(M, N, K, dtype=torch.float32, device="npu", seed=0):
    """Deterministic input tensors on NPU."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_cpu = torch.randn(M, K, dtype=dtype, generator=g)
    b_cpu = torch.randn(K, N, dtype=dtype, generator=g)
    return a_cpu.to(device), b_cpu.to(device)


# =============================================================================
# The hardware facts we will discover empirically in Step 01
# (Kept here for reference. Do NOT read these before Step 01.)
# =============================================================================
L0C_CAPACITY_KB = 128     # accumulator buffer; bounds BM * BN * 4
L1_CAPACITY_KB = 512      # operand buffer (cbuf); bounds (BM+BN) * BK * 4 * 2
MULTIBUFFER_FACTOR = 2    # compiler default: --set-workspace-multibuffer=2
L1_BUDGET_KB = L1_CAPACITY_KB // MULTIBUFFER_FACTOR  # effective budget = 256 KB


def is_legal_tile(BM, BN, BK, dtype_bytes=4):
    """Is (BM, BN, BK) within L0C and L1 limits AND 16-aligned?"""
    l0c_kb = BM * BN * dtype_bytes / 1024
    l1_kb = (BM + BN) * BK * dtype_bytes / 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return (l0c_kb <= L0C_CAPACITY_KB) and (l1_kb <= L1_BUDGET_KB) and aligned