"""Step 06: calibrate the unoptimized Triton baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from common import bench_ms, check_close, make_inputs, make_reference, matmul_fixed, tflops


def run_shape(M, N, K):
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    triton_out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    triton_e2e = lambda: matmul_fixed(a, b, 32, 32, 32)
    triton_kernel = lambda: matmul_fixed(a, b, 32, 32, 32, out=triton_out)

    diff = check_close(triton_kernel(), a, b, expected)
    results = [
        ("baseline end-to-end", bench_ms(triton_e2e)),
        ("baseline preallocated", bench_ms(triton_kernel)),
    ]
    print(f"\nshape={M}x{K} @ {K}x{N}, baseline tile=32x32x32, "
          f"fp16 input/output, max_diff={diff:.3e}")
    for name, ms in results:
        print(f"  {name:<22} {ms:9.4f} ms  {tflops(M,N,K,ms):8.3f} TFLOP/s")


def main():
    torch.cuda.set_device(0)
    for shape in [(1024, 1024, 1024), (2048, 2048, 2048)]:
        run_shape(*shape)


if __name__ == "__main__":
    main()
