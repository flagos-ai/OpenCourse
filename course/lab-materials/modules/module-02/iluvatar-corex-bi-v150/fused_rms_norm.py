import time

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_norm_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    updated_residual_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_offsets = pid * n_cols + cols

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    updated = x + residual
    tl.store(updated_residual_ptr + row_offsets, updated, mask=mask)

    variance = tl.sum(updated * updated, axis=0) / n_cols
    # TODO BEGIN: Compute the reciprocal RMS and apply normalization + weight.

    # TODO END
    tl.store(out_ptr + row_offsets, out, mask=mask)

def naive_add_rms_norm(x, residual, weight, eps=1e-6):
    """
    Compute fused Residual + RMSNorm using native PyTorch.

    This follows exactly the computation order of the Triton kernel.
    """

    # read 2MN elements ; write MN elements
    updated = x + residual

    # save updated residual
    updated_residual = updated

    # read MN elements ; write M elements
    variance = (updated * updated).mean(dim=1)

    # read M elements ; write M elements
    rrms = torch.rsqrt(variance + eps)

    # read MN + M + N elements ; write MN elements
    out = updated * rrms[:, None] * weight

    # in total:
    # read 5MN + 2M + N elements
    # wrote 2MN + 2M elements
    return out, updated_residual

@triton.jit
def _add_kernel(
    x_ptr,
    residual_ptr,
    updated_residual_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_offsets = pid * n_cols + cols

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0)
    tl.store(updated_residual_ptr + row_offsets, x + residual, mask=mask)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_offsets = pid * n_cols + cols

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(variance + eps)
    out = x * rrms * weight
    tl.store(out_ptr + row_offsets, out, mask=mask)


def _check_inputs(x, residual, weight):
    if x.ndim != 2:
        raise ValueError(f"x must be 2D [M, N], got shape={tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(
            f"residual shape must match x, got {tuple(residual.shape)} vs {tuple(x.shape)}"
        )
    if weight.shape != (x.shape[-1],):
        raise ValueError(f"weight must be 1D [N], got shape={tuple(weight.shape)}")
    if not x.is_cuda or not residual.is_cuda or not weight.is_cuda:
        raise ValueError("x, residual, and weight must be CUDA tensors")


def _block_size(n_cols):
    # Compute the block size (a compile-time kernel parameter).
    # Use the smallest power of two that covers an entire row.
    return triton.next_power_of_2(n_cols)

def fused_add_rms_norm(x, residual, weight, eps=1e-6):
    # Allocate device memory for the outputs before launching the kernel.
    out = torch.empty_like(x)
    updated_residual = torch.empty_like(x)
    
    # Get the matrix dimensions.
    n_rows, n_cols = x.shape
    
    # 1. Choose the block size (a compile-time kernel parameter).
    # Each Triton program processes one entire row.
    # Use the smallest power of two that covers n_cols.
    block_size = _block_size(n_cols)

    # 2. Launch the kernel at runtime.
    # The grid has one program per matrix row. Each program performs:
    #   (1) Residual Add
    #   (2) RMS calculation
    #   (3) Normalization
    #   (4) Multiplication by the learnable weight
    _fused_add_rms_norm_kernel[(n_rows,)](
        x,
        residual,
        out,
        updated_residual,
        weight,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )

    return out, updated_residual


def add_rms_norm(x, residual, weight, eps=1e-6):
    _check_inputs(x, residual, weight)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    out = torch.empty_like(x)
    updated_residual = torch.empty_like(x)
    n_rows, n_cols = x.shape
    block_size = _block_size(n_cols)

    _add_kernel[(n_rows,)](
        x,
        residual,
        updated_residual,
        n_cols,
        BLOCK_SIZE=block_size,
    )
    _rms_norm_kernel[(n_rows,)](
        updated_residual,
        out,
        weight,
        n_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return out, updated_residual


def torch_add_rms_norm(x, residual, weight, eps=1e-6):
    """Float32 reference with the same output-storage dtype as the Triton kernel.

    bfloat16 inputs are promoted before the arithmetic, just as the Triton
    kernel promotes each loaded value to ``tl.float32``.  Both results are then
    cast back to the input dtype, matching ``torch.empty_like(x)`` output
    buffers in ``fused_add_rms_norm``.
    """
    x_fp32 = x.float()
    residual_fp32 = residual.float()
    weight_fp32 = weight.float()
    updated_fp32 = x_fp32 + residual_fp32
    rrms = torch.rsqrt(updated_fp32.square().mean(dim=-1, keepdim=True) + eps)
    out_fp32 = updated_fp32 * rrms * weight_fp32
    return out_fp32.to(x.dtype), updated_fp32.to(residual.dtype)


def benchmark(fn, *args, warmup=25, iters=100, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def _validate(dtype, shape=(4096, 4096), eps=1e-6):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn((shape[-1],), dtype=dtype, device="cuda")

    ref_out, ref_residual = torch_add_rms_norm(x, residual, weight, eps)
    out, updated = fused_add_rms_norm(x, residual, weight, eps)

    out_diff = (ref_out.float() - out.float()).abs().max().item()
    residual_diff = (ref_residual.float() - updated.float()).abs().max().item()
    return out_diff, residual_diff


def main():
    print(f"{'dtype':<12} {'shape':<18} {'out diff':<14} {'res diff':<14} {'fused ms':<12}")
    print("-" * 72)
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((512, 256), (1024, 512), (4096, 4096)):
            x = torch.randn(shape, dtype=dtype, device="cuda")
            residual = torch.randn(shape, dtype=dtype, device="cuda")
            weight = torch.randn((shape[-1],), dtype=dtype, device="cuda")
            out_diff, residual_diff = _validate(dtype, shape)
            ms = benchmark(fused_add_rms_norm, x, residual, weight)
            print(
                f"{str(dtype).replace('torch.', ''):<12} "
                f"{shape[0]}x{shape[1]:<12} "
                f"{out_diff:<14.3e} {residual_diff:<14.3e} {ms:<12.4f}"
            )


if __name__ == "__main__":
    main()
