"""
04_matmul_compare.py
--------------------
Tile-based matrix multiplication: a Triton kernel vs torch.matmul,
both on the same Ascend NPU.

About the comparison:
- 'torch.matmul' on Ascend dispatches to the CANN library, which targets
  Ascend's Cube Core (the matrix engine) with hand-tuned kernels.
  This is comparable to using cuBLAS on NVIDIA -- a strong baseline.
- The Triton kernel below is a straightforward tiled implementation.

The point is NOT to beat the vendor library. The point is:
  * The same Triton matmul source runs on NVIDIA and on Ascend.
  * For non-standard ops where no library exists, Triton gives you a
    productivity-friendly path to write your own kernel.

This file is also the BASELINE for tomorrow's performance-tuning lab,
where we will progressively optimize this matmul for Ascend.
"""
import time
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


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
    """
    Compute one BLOCK_M x BLOCK_N tile of C = A @ B.

    A is (M, K), B is (K, N), C is (M, N), all row-major float32.
    Each program is identified by a 2D program id (pid_m, pid_n) and
    computes C[pid_m*BLOCK_M : (pid_m+1)*BLOCK_M,
              pid_n*BLOCK_N : (pid_n+1)*BLOCK_N].

    The K dimension is reduced inside the kernel by looping over
    BLOCK_K-sized chunks and accumulating into a register-resident tile.

    TODO ------------------------------------------------------------------
    Two things to fill in:

      (1) Initialize the accumulator tile.
          Shape (BLOCK_M, BLOCK_N), dtype float32, all zeros.
          API: tl.zeros(shape, dtype=tl.float32)

      (2) Inside the K loop, load one (BLOCK_M x BLOCK_K) tile of A and one
          (BLOCK_K x BLOCK_N) tile of B, then accumulate their product.
          - Load with a mask to handle K not divisible by BLOCK_K.
          - Use other=0.0 so masked-out positions contribute 0.0 to the dot.
          - API: tl.dot(a, b) does the BLOCK_M x BLOCK_K @ BLOCK_K x BLOCK_N
            tile-matmul in one instruction.
    -----------------------------------------------------------------------
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row/col offsets for this output tile (and column offsets for K stepping)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointer base for A's first tile and B's first tile.
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # >>> YOUR CODE HERE >>>
    # TODO (1): create the accumulator tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# (2) K loop body
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    # <<< END OF YOUR CODE <<<

    # Store the output tile with a mask for boundary tiles.
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_triton(a: torch.Tensor, b: torch.Tensor,
                  BLOCK_M: int = 64, BLOCK_N: int = 64, BLOCK_K: int = 32) -> torch.Tensor:
    """Host-side launcher. BLOCK sizes are tunable."""
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
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return c


def benchmark(fn, *args, warmup=5, iters=20, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    torch.manual_seed(0)
    print(f"{'Shape (MxKxN)':<22} {'torch.matmul (ms)':<20} {'Triton matmul (ms)':<22} {'Max diff':<12}")
    print("-" * 80)

    for M, K, N in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        a = torch.randn(M, K, dtype=torch.float32, device="npu")
        b = torch.randn(K, N, dtype=torch.float32, device="npu")

        c_torch = a @ b
        c_triton = matmul_triton(a, b)
        max_diff = (c_torch - c_triton).abs().max().item()

        t_torch = benchmark(torch.matmul, a, b)
        t_triton = benchmark(matmul_triton, a, b)

        shape_str = f"{M} x {K} x {N}"
        print(f"{shape_str:<22} {t_torch:<20.4f} {t_triton:<22.4f} {max_diff:<12.2e}")

    print()
    print("Discussion:")
    print(" - torch.matmul on Ascend uses CANN's hand-tuned Cube Core operators")
    print("   (similar role to cuBLAS on NVIDIA). It is a strong baseline.")
    print(" - Our Triton kernel is a straightforward tiled implementation;")
    print("   it does NOT yet exploit several Ascend-specific optimizations.")
    print(" - Tomorrow we will progressively tune this kernel for Ascend.")

    # ----- Try-it-yourself -------------------------------------------------
    # 1. Try (BLOCK_M, BLOCK_N, BLOCK_K) = (128, 128, 32). Faster?
    # 2. Try (32, 32, 32). Slower? Why?
    # 3. Reasoning question: why might increasing BLOCK_K help arithmetic
    #    intensity? Why might it stop helping past some point?
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
