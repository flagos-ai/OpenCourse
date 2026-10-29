"""Task 05 solution: tiled fp16 GEMM with fp32 accumulation."""
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"


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
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k_start * BLOCK_K
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul_triton(a, b, block_m=32, block_n=32, block_k=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.is_contiguous() and b.is_contiguous()
    # tl.store casts the fp32 accumulator to fp16. This matches the vendor
    # operation's input/output dtype, so the timing comparison is meaningful.
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return c


def bench_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    print(f"{'M,K,N':>16} {'torch/ms':>10} {'triton/ms':>11} {'max diff':>12}")
    for M, K, N in [(65, 97, 79), (256, 256, 256), (511, 257, 385)]:
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        reference = a @ b
        vendor = lambda: a @ b
        candidate = lambda: matmul_triton(a, b)
        actual = candidate()
        diff = (actual - reference).abs().max().item()
        torch.testing.assert_close(actual, reference, atol=0.2, rtol=1e-2)
        print(f"{str((M, K, N)):>16} {bench_ms(vendor):10.4f} "
              f"{bench_ms(candidate):11.4f} {diff:12.3e}")


if __name__ == "__main__":
    main()
