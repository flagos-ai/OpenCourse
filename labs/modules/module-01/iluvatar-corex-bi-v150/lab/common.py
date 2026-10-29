"""Shared GEMM, validation, measurement, and resource-probe helpers."""
import statistics
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        remain = K - k_start * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < remain), other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < remain) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_fixed(a, b, block_m=32, block_n=32, block_k=32,
                 num_warps=DEFAULT_NUM_WARPS,
                 num_stages=DEFAULT_NUM_STAGES,
                 out=None):
    """Launch one explicit block/scheduler configuration.

    Passing ``out`` reuses a preallocated output and gives a kernel-focused
    timing path. Leaving it as None measures the end-to-end wrapper path.
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.is_contiguous() and b.is_contiguous()
    # Keep fp16 output while accumulating the K dimension in fp32.
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    assert c.shape == (M, N) and c.dtype == a.dtype and c.device == a.device
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


def make_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    return a, b


def make_reference(a, b, output_dtype=None):
    """Build a correctness-only fp32 reference; never include it in timing."""
    output_dtype = output_dtype or a.dtype
    return (a.float() @ b.float()).to(output_dtype)


def check_close(actual, a, b, expected=None):
    expected = make_reference(a, b, actual.dtype) if expected is None else expected
    assert torch.isfinite(actual).all(), "Triton output contains NaN or Inf"
    diff = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=1e-2)
    return diff


def bench_ms(fn, warmup=5, samples=7, inner=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3 / inner)
    return statistics.median(times)


def tflops(M, N, K, ms):
    return 2.0 * M * N * K / (ms * 1e9)


def tile_footprint_kib(block_m, block_n, block_k,
                       operand_bytes=2, accumulator_bytes=4):
    """Return simple fp16-operand/fp32-accumulator footprint proxies."""
    accumulator = block_m * block_n * accumulator_bytes / 1024
    operands = (block_m + block_n) * block_k * operand_bytes / 1024
    return accumulator, operands


def compact_error(exc, limit=240):
    """Keep the first informative compiler/runtime line without losing context."""
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    preferred = next(
        (line for line in lines
         if any(word in line.lower() for word in
                ("overflow", "resource", "register", "memory", "failed", "error"))),
        lines[0] if lines else repr(exc),
    )
    return preferred[:limit]
