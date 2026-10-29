"""Backward Triton kernel and gradient validation for fused add + RMSNorm.

The forward operator is:
    z = x + residual
    y = z * rsqrt(mean(z * z) + eps) * weight

Both ``y`` and ``z`` are returned.  The latter therefore has its own upstream
gradient in addition to the gradient flowing through RMSNorm.
"""

import argparse

import torch
import triton
import triton.language as tl

from fused_rms_norm import _block_size, _check_inputs, fused_add_rms_norm


@triton.jit
def _fused_add_rms_norm_backward_kernel(
    updated_residual_ptr,
    weight_ptr,
    grad_out_ptr,
    grad_updated_residual_ptr,
    grad_x_ptr,
    grad_residual_ptr,
    grad_weight_ptr,
    n_cols: tl.constexpr,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Let z = x + residual and g = grad_out * weight.  The gradient flowing
    # through RMSNorm is:
    #   grad_z_norm = g * rrms - z * rrms^3 * sum(g * z) / n_cols
    # The second returned forward value contributes grad_updated_residual
    # directly to grad_z.  Because z = x + residual, grad_x == grad_residual.
    #
    # Accumulate grad_weight = grad_out * z * rrms across rows with
    # tl.atomic_add into the float32 grad_weight_ptr buffer.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    z = tl.load(updated_residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad_z_direct = tl.load(
        grad_updated_residual_ptr + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    variance = tl.sum(z * z, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(variance + eps)
    scaled_grad = grad_out * weight
    dot = tl.sum(scaled_grad * z, axis=0)

    # d RMSNorm(z) = rrms * g - z * rrms^3 * sum(g * z) / N.
    grad_z = (
        scaled_grad * rrms
        - z * (rrms * rrms * rrms) * dot / n_cols
        + grad_z_direct
    )
    tl.store(grad_x_ptr + offsets, grad_z, mask=mask)
    tl.store(grad_residual_ptr + offsets, grad_z, mask=mask)

    grad_weight = grad_out * z * rrms
    tl.atomic_add(grad_weight_ptr + cols, grad_weight, mask=mask)


class FusedAddRMSNorm(torch.autograd.Function):
    """Autograd wrapper around the forward and backward Triton kernels."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps=1e-6):
        _check_inputs(x, residual, weight)
        out, updated_residual = fused_add_rms_norm(x, residual, weight, eps)
        ctx.save_for_backward(updated_residual, weight.contiguous())
        ctx.eps = eps
        return out, updated_residual

    @staticmethod
    def backward(ctx, grad_out, grad_updated_residual):
        updated_residual, weight = ctx.saved_tensors
        n_rows, n_cols = updated_residual.shape

        if grad_out is None:
            grad_out = torch.zeros_like(updated_residual)
        if grad_updated_residual is None:
            grad_updated_residual = torch.zeros_like(updated_residual)

        grad_x = torch.empty_like(updated_residual)
        grad_residual = torch.empty_like(updated_residual)
        grad_weight_fp32 = torch.zeros(
            (n_cols,), dtype=torch.float32, device=updated_residual.device
        )

        _fused_add_rms_norm_backward_kernel[(n_rows,)](
            updated_residual,
            weight,
            grad_out.contiguous(),
            grad_updated_residual.contiguous(),
            grad_x,
            grad_residual,
            grad_weight_fp32,
            n_cols,
            ctx.eps,
            BLOCK_SIZE=_block_size(n_cols),
        )
        return grad_x, grad_residual, grad_weight_fp32.to(weight.dtype), None


def fused_add_rms_norm_autograd(x, residual, weight, eps=1e-6):
    """Differentiable fused add + RMSNorm, returning ``(out, updated_residual)``."""
    return FusedAddRMSNorm.apply(x, residual, weight, eps)


def torch_reference(x, residual, weight, eps):
    updated_residual = x + residual
    rrms = torch.rsqrt(updated_residual.square().mean(dim=-1, keepdim=True) + eps)
    return updated_residual * rrms * weight, updated_residual


def _max_abs_error(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def validate_gradients(shape, dtype=torch.float32, eps=1e-6):
    """Compare forward values and all input gradients with PyTorch autograd."""
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    residual = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda", requires_grad=True)
    grad_out = torch.randn_like(x)
    grad_updated = torch.randn_like(x)

    out, updated = fused_add_rms_norm_autograd(x, residual, weight, eps)
    (out * grad_out + updated * grad_updated).sum().backward()
    triton_result = {
        "out": out.detach(),
        "updated": updated.detach(),
        "x_grad": x.grad.detach(),
        "residual_grad": residual.grad.detach(),
        "weight_grad": weight.grad.detach(),
    }

    ref_x = x.detach().clone().requires_grad_(True)
    ref_residual = residual.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    ref_out, ref_updated = torch_reference(ref_x, ref_residual, ref_weight, eps)
    (ref_out * grad_out + ref_updated * grad_updated).sum().backward()

    errors = {
        "out": _max_abs_error(triton_result["out"], ref_out),
        "updated": _max_abs_error(triton_result["updated"], ref_updated),
        "x_grad": _max_abs_error(triton_result["x_grad"], ref_x.grad),
        "residual_grad": _max_abs_error(
            triton_result["residual_grad"], ref_residual.grad
        ),
        "weight_grad": _max_abs_error(triton_result["weight_grad"], ref_weight.grad),
    }
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    errors = validate_gradients((args.m, args.n), dtype, args.eps)
    # bfloat16 gradients are stored back in bfloat16, whose coarse mantissa
    # produces visible absolute rounding error for the row-reduced weight grad.
    tolerance = 2e-5 if dtype == torch.float32 else 2e-1

    print(f"shape=({args.m}, {args.n}) dtype={args.dtype} eps={args.eps}")
    for name, error in errors.items():
        print(f"{name:<14} max_abs_error={error:.3e}")

    if max(errors.values()) <= tolerance:
        print(f"PASS: all forward values and gradients are within {tolerance:.1e}.")
    else:
        raise AssertionError(
            f"Gradient validation failed: max error {max(errors.values()):.3e} "
            f"exceeds tolerance {tolerance:.1e}."
        )


if __name__ == "__main__":
    main()
