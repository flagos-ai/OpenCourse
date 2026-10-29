"""Task 04 solution: LayerNorm forward over the last contiguous dimension."""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEVICE = "cuda:0"
MAX_BLOCK = 65536


@triton.jit
def layernorm_kernel(y_ptr, x_ptr, weight_ptr, bias_ptr, row_stride, n_cols,
                     eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * rstd * weight + bias
    tl.store(y_ptr + row * row_stride + cols, y, mask=mask)


def layernorm_triton(x, weight, bias, eps=1e-5):
    assert x.ndim == 2 and x.is_contiguous()
    rows, cols = x.shape
    assert weight.shape == bias.shape == (cols,)
    block = triton.next_power_of_2(cols)
    if block > MAX_BLOCK:
        raise ValueError(f"hidden size {cols} exceeds the one-program limit")
    out = torch.empty_like(x)
    layernorm_kernel[(rows,)](
        out, x, weight, bias, x.stride(0), cols, eps=eps, BLOCK_SIZE=block
    )
    return out


def bench_ms(fn, warmup=10, iters=50):
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
    print(f"{'shape':>14} {'torch/ms':>10} {'triton/ms':>11} {'max diff':>12}")
    for rows, cols in [(1, 127), (32, 256), (64, 257), (256, 1024)]:
        x = torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)
        weight = torch.randn(cols, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(cols, device=DEVICE, dtype=torch.float32)
        reference = lambda: F.layer_norm(x, (cols,), weight, bias, 1e-5)
        candidate = lambda: layernorm_triton(x, weight, bias)
        expected, actual = reference(), candidate()
        diff = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        print(f"{str((rows, cols)):>14} {bench_ms(reference):10.4f} "
              f"{bench_ms(candidate):11.4f} {diff:12.3e}")


if __name__ == "__main__":
    main()
