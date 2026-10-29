"""Step 09: compare the unoptimized baseline with Triton autotune."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from common import (DEFAULT_NUM_STAGES, DEFAULT_NUM_WARPS, bench_ms, check_close,
                    make_inputs, make_reference, matmul_fixed, tflops)

autotune = importlib.import_module("08_autotune")


def main():
    torch.cuda.set_device(0)
    # At 2048^3, GEMM work dominates the fixed autotune-dispatch overhead and
    # makes the benefit of the selected tile/scheduler combination visible.
    M = N = K = 2048
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)

    baseline_out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    auto_out = torch.empty_like(baseline_out)

    # Trigger autotune first, then read the selected launch configuration.
    autotune.matmul_autotuned(a, b, auto_out)
    tile, warps, stages = autotune.best_config_values()

    versions = {
        "baseline 32x32x32": lambda: matmul_fixed(
            a, b, 32, 32, 32,
            num_warps=DEFAULT_NUM_WARPS,
            num_stages=DEFAULT_NUM_STAGES,
            out=baseline_out,
        ),
        "autotuned": lambda: autotune.matmul_autotuned(a, b, auto_out),
    }

    diffs = {name: check_close(fn(), a, b, expected) for name, fn in versions.items()}
    results = [(name, bench_ms(fn)) for name, fn in versions.items()]
    result_map = dict(results)
    baseline = result_map["baseline 32x32x32"]

    print(f"shape={(M, N, K)}, fp16 input/output, fp32 accumulation")
    print("Both Triton paths reuse preallocated outputs.")
    print(f"autotune best: block={tile}, num_warps={warps}, num_stages={stages}")
    print(f"{'version':<24} {'ms':>10} {'TFLOP/s':>10} {'vs base':>10} "
          f"{'max diff':>11}")
    for name, ms in results:
        print(f"{name:<24} {ms:10.4f} {tflops(M,N,K,ms):10.3f} "
              f"{baseline/ms:9.2f}x {diffs[name]:11.3e}")
    print(f"autotune speedup over baseline: {baseline/result_map['autotuned']:.3f}x")


if __name__ == "__main__":
    main()
