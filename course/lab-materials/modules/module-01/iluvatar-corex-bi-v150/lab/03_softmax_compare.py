"""Task 03 solution: numerically stable row-wise softmax."""
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"
MAX_BLOCK = 65536


@triton.jit
def softmax_kernel(out_ptr, x_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    tl.store(out_ptr + row * row_stride + cols, numerator / denominator, mask=mask)


def softmax_triton(x):
    assert x.ndim == 2 and x.is_contiguous()
    n_rows, n_cols = x.shape
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"n_cols={n_cols} exceeds the one-program limit")
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), n_cols, BLOCK_SIZE=block)
    return out


def bench_ms(fn, x, warmup=10, iters=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iters


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    print(f"{'shape':>14} {'torch/ms':>10} {'triton/ms':>11} {'max diff':>12}")
    for rows, cols in [(1, 255), (32, 256), (64, 257), (128, 1025)]:
        x = torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)
        actual = softmax_triton(x)
        expected = torch.softmax(x, dim=-1)
        diff = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        print(f"{str((rows, cols)):>14} {bench_ms(lambda z: torch.softmax(z, -1), x):10.4f} "
              f"{bench_ms(softmax_triton, x):11.4f} {diff:12.3e}")


if __name__ == "__main__":
    main()
