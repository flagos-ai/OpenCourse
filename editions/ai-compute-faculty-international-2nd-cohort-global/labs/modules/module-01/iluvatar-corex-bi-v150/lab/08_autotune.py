"""Step 08 demo: complete the autotune configs, then run all tuning checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import triton
import triton.language as tl
from common import (
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
    bench_ms,
    check_close,
    compact_error,
    make_inputs,
    make_reference,
    matmul_fixed,
    tflops,
    tile_footprint_kib,
)


# >>> YOUR CODE HERE >>>
# TODO: choose a small set of PASS block tiles from 07_block_sweep.py and
# convert them into triton.Config entries.
BLOCK_TILES = [
    # TODO
]
AUTOTUNE_CONFIGS = [
    # TODO
]
# <<< END OF YOUR CODE <<<


@triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_autotuned_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                            stride_am, stride_ak, stride_bk, stride_bn,
                            stride_cm, stride_cn,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
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
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_autotuned(a, b, out=None):
    M, K = a.shape
    _, N = b.shape
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    matmul_autotuned_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c


def best_config_values():
    cfg = matmul_autotuned_kernel.best_config
    tile = tuple(cfg.kwargs[name] for name in ["BLOCK_M", "BLOCK_N", "BLOCK_K"])
    return tile, cfg.num_warps, cfg.num_stages


def run_shape(M, N, K):
    """Autotune blocks, then compare with an explicitly identical fixed launch."""
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    auto_out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    diff = check_close(matmul_autotuned(a, b, auto_out), a, b, expected)
    tile, warps, stages = best_config_values()

    auto_ms = bench_ms(lambda: matmul_autotuned(a, b, auto_out))
    fixed_out = torch.empty_like(auto_out)
    exact_fixed = lambda: matmul_fixed(
        a, b, *tile, num_warps=warps, num_stages=stages, out=fixed_out
    )
    fixed_diff = check_close(exact_fixed(), a, b, expected)
    fixed_ms = bench_ms(exact_fixed)
    delta = fixed_ms / auto_ms if auto_ms else float("inf")

    print(f"shape={(M, N, K)} best block={tile}, warps={warps}, stages={stages}")
    print(f"  autotuned decorated : {auto_ms:.4f} ms, {tflops(M,N,K,auto_ms):.3f} TFLOP/s")
    print(f"  exact-config fixed  : {fixed_ms:.4f} ms, {tflops(M,N,K,fixed_ms):.3f} TFLOP/s")
    print(f"  fixed/auto ratio    : {delta:.3f}x, diffs=({diff:.3e}, {fixed_diff:.3e})")
    return tile


def bk_sweep(M, N, K, bm, bn):
    """Fix BM/BN and expose the K-loop/resource tradeoff."""
    print(f"\nBK sweep at shape={(M, N, K)}, fixed BM/BN={(bm, bn)}")
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    rows = []
    for bk in [16, 32, 64, 128, 256]:
        fn = lambda bk=bk: matmul_fixed(
            a, b, bm, bn, bk,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
            out=out,
        )
        try:
            diff = check_close(fn(), a, b, expected)
            ms = bench_ms(fn)
            _, operands_kib = tile_footprint_kib(bm, bn, bk)
            rows.append((bk, ms))
            print(f"  BK={bk:3d} K-loops={(K+bk-1)//bk:3d} operands~{operands_kib:6.1f} KiB "
                  f"{ms:8.4f} ms {tflops(M,N,K,ms):7.3f} TFLOP/s diff={diff:.3e}")
        except Exception as exc:
            print(f"  BK={bk:3d} FAIL: {compact_error(exc)}")

    # Whole-program approximation: fp16 A/B/C traffic, no cache reuse counted.
    ai = 2 * bm * bn * K / (2 * (bm * K + K * bn + bm * bn))
    print(f"  approximate tile arithmetic intensity = {ai:.2f} FLOPs/byte")
    return rows


def scheduler_sweep(M, N, K, tile):
    """After block tuning, test whether CoreX uses num_warps/num_stages."""
    print(f"\nScheduler sweep at shape={(M, N, K)}, fixed block={tile}")
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    rows = []
    for warps in [1, 2, 4, 8]:
        for stages in [1, 2, 3, 4]:
            fn = lambda warps=warps, stages=stages: matmul_fixed(
                a, b, *tile, num_warps=warps, num_stages=stages, out=out
            )
            try:
                diff = check_close(fn(), a, b, expected)
                ms = bench_ms(fn, warmup=3, samples=5, inner=10)
                rows.append((ms, warps, stages, diff))
                print(f"  warps={warps:2d} stages={stages}  {ms:8.4f} ms  "
                      f"{tflops(M,N,K,ms):7.3f} TFLOP/s diff={diff:.3e}")
            except Exception as exc:
                print(f"  warps={warps:2d} stages={stages}  FAIL: {compact_error(exc)}")
    if rows:
        best = min(rows)
        print(f"  scheduler best: warps={best[1]}, stages={best[2]}, {best[0]:.4f} ms")
    return rows


def main():
    torch.cuda.set_device(0)
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (2048, 512, 1024),
        (512, 2048, 256),
    ]
    winners = [run_shape(*shape) for shape in shapes]
    bm, bn, _ = winners[0]
    bk_sweep(*shapes[0], bm, bn)
    scheduler_sweep(*shapes[0], winners[0])


if __name__ == "__main__":
    main()
