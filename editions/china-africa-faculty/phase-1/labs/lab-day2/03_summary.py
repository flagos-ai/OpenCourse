"""
03_summary.py
-------------
Put every version of the matmul we wrote today on one table, at the
same shape, measured with the same benchmark, so the progress is
visible in one glance.

There is no new code here -- it is just a readout.
"""
import torch
import torch_npu  # noqa: F401

from common import bench_ms, make_inputs, silent_stdio
# Re-use the two kernels from step 02. Importing step 02 runs its module
# body, which constructs ~170 triton.Config objects -- each emits a
# warning. silent_stdio wraps the import to keep the output clean.
from importlib import import_module
with silent_stdio():
    step2 = import_module("02_autotune_roofline")


def main():
    M = N = K = 2048
    a, b = make_inputs(M, N, K)

    print(f"Final comparison at M=N=K={M}, fp32")
    print("=" * 78)

    # 1. Baseline: Day 1's default (64, 64, 32)
    t_baseline = bench_ms(lambda: step2.matmul_fixed(a, b, 64, 64, 32))

    # 2. Hand-picked legal tile: just (128, 128, 64) -- a "reasonable guess"
    #    from step 01's legal space, without actually running autotune.
    #    This isolates "just using the model" from "adding autotune on top".
    t_legal = bench_ms(lambda: step2.matmul_fixed(a, b, 128, 128, 64))

    # 3. Autotuned: step 02 picks the best legal tile
    # (silent_stdio wraps autotune calls to suppress the ~170 benign
    #  triton-ascend warnings about GPU-era Config defaults.)
    with silent_stdio():
        step2.matmul_autotuned(a, b)  # trigger autotune warmup
    best_cfg = step2.matmul_autotuned_kernel.best_config
    bm, bn, bk = best_cfg.kwargs['BLOCK_M'], best_cfg.kwargs['BLOCK_N'], best_cfg.kwargs['BLOCK_K']
    with silent_stdio():
        t_auto = bench_ms(step2.matmul_autotuned, a, b)

    # 4. Vendor: torch.matmul dispatches to CANN's Cube kernels
    t_torch = bench_ms(torch.matmul, a, b)

    print()
    print(f"  {'version':<45} {'ms':>10} {'vs baseline':>13} {'vs vendor':>11}")
    print("  " + "-" * 78)

    def row(name, t):
        print(f"  {name:<45} {t:>10.4f} {t_baseline/t:>12.2f}x {t_torch/t:>10.2f}x")

    row("baseline  (Day 1 default 64x64x32)", t_baseline)
    row("buffer-model tile  (128x128x64, hand)", t_legal)
    row(f"autotuned          ({bm}x{bn}x{bk})", t_auto)
    row("torch.matmul  (CANN hand-tuned)", t_torch)

    # How we used the hardware at each stage
    print()
    print("  What hardware fact did each step use?")
    print("  " + "-" * 78)
    print("  baseline          : none -- GPU-tutorial defaults, ignores Ascend entirely")
    print("  buffer-model tile : L0C = 128 KB, L1 budget = 256 KB (measured in step 01)")
    print("  autotuned         : + asymmetric tiles are faster (measured in step 02a)")
    print("                      + BK >= 32 * multiple saturates roofline (step 02b)")
    print("  torch.matmul      : everything above, plus fp16/bf16 mixed precision,")
    print("                      grouped ordering for L2 reuse, hand-scheduled ILP, ...")

    # The gap that remains
    print()
    print("=" * 78)
    print("What we did NOT do (hooks for Day 3 and beyond)")
    print("=" * 78)
    print()
    print("  1. MIXED PRECISION. We kept fp32 throughout. The Cube unit's")
    print("     fp16 / bf16 throughput is SEVERAL TIMES its fp32 throughput.")
    print("     A fp16 (x fp16 -> fp32 accumulate) matmul routinely beats a")
    print("     fp32 matmul by 3-5x -- this is almost certainly where the")
    print("     remaining gap to a fully hand-tuned CANN kernel lives.")
    print()
    print("  2. GROUPED ORDERING (GROUP_SIZE_M). The Triton GPU tutorial")
    print("     uses a trick where nearby program ids compute a GROUP_SIZE_M")
    print("     x N block of the output, improving L2 cache reuse. On Ascend")
    print("     this affects L1 and global-memory access patterns. Worth")
    print("     measuring, but out of scope today.")
    print()
    print("  3. OPERATOR FUSION. matmul-then-bias-then-gelu is three kernel")
    print("     launches and three full round-trips to global memory. A")
    print("     fused kernel keeps the output in L0C/UB for the epilogue.")
    print("     Triton makes this easy; we just didn't have time today.")
    print()
    print("  4. AUTOMATIC MULTI-BUFFER CONTROL. We observed the compiler")
    print("     doing multibuffer=2 on its own (this is what halved our L1")
    print("     budget in step 01). Triton-Ascend exposes")
    print("     tl.multibuffer(tensor, n) to set this explicitly. Useful in")
    print("     kernels where you want multibuffer=3 or to be turned off.")
    print()
    print("  The point of today was not to beat the vendor -- it was to")
    print("  show that the vendor's gap is EXPLAINABLE and CLOSABLE with")
    print("  measurement-driven reasoning. Every remaining optimization has")
    print("  the same structure: form a hypothesis about the hardware,")
    print("  measure, adjust the code, verify.")


if __name__ == "__main__":
    main()