"""
00_baseline.py
--------------
Re-establish the Day 1 baseline at larger shapes, so all subsequent
optimization steps are measured against a stable, noise-free starting point.

Why this step exists
--------------------
Day 1's 04_matmul_compare.py used shapes (256, 512, 1024)^3. At those
sizes the kernel run time is in the tens of microseconds -- close enough
to launch overhead and sync granularity that any "speedup" we see later
could be in the noise.

For Day 2 we jump to 1024^3 and 2048^3. At 2048^3 the kernel runs for
a few hundred microseconds, and arithmetic intensity is high enough that
the Cube pipeline actually gets to stretch its legs. This is where we
can measure real optimization effects instead of benchmark artefacts.

There is no code for you to write in this step. Just run it and read
the numbers. Keep the output -- every later step compares against it.
"""
import torch
import torch_npu  # noqa: F401

from common import matmul_triton, bench_ms, make_inputs


def main():
    torch.manual_seed(0)

    shapes = [(1024, 1024, 1024), (2048, 2048, 2048)]

    print(f"{'Shape':<18} {'torch (ms)':>12} {'triton (ms)':>14} "
          f"{'ratio':>8} {'max diff':>12}")
    print("-" * 70)

    for M, N, K in shapes:
        a, b = make_inputs(M, N, K)

        # Correctness first -- if this is off, the timing numbers are meaningless.
        c_torch = a @ b
        c_triton = matmul_triton(a, b)  # uses default BM=BN=64, BK=32
        max_diff = (c_torch - c_triton).abs().max().item()

        # Time each
        t_torch = bench_ms(torch.matmul, a, b)
        t_triton = bench_ms(matmul_triton, a, b)

        shape_str = f"{M} x {N} x {K}"
        ratio = t_triton / t_torch
        print(f"{shape_str:<18} {t_torch:>12.4f} {t_triton:>14.4f} "
              f"{ratio:>7.2f}x {max_diff:>12.2e}")

    print()
    print("Reading the numbers")
    print("-" * 70)
    print("  * 'torch' uses CANN's hand-tuned Cube Core matmul -- strong baseline.")
    print("  * 'triton' is the Day 1 kernel with default BLOCK sizes (64, 64, 32).")
    print("  * 'ratio' = triton / torch. 1.0 means we match the vendor.")
    print("  * 'max diff' ~ 1e-4 is normal fp32 rounding over K accumulations.")
    print()
    print("This is our Day 2 starting point. By the end of the lab we will")
    print("close most of the gap -- not by cleverer Triton syntax, but by")
    print("measuring the Ascend Cube architecture and tuning to it.")


if __name__ == "__main__":
    main()
