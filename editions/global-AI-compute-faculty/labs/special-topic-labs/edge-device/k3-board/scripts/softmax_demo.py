# softmax_demo.py — inference-related Softmax operator demo
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)

    row = tl.load(
        row_start_ptr + col_offsets,
        mask=col_offsets < n_cols,
        other=-float("inf"),
    )
    row_f32 = row.to(tl.float32)
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator

    tl.store(
        output_ptr + row_idx * output_row_stride + col_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=col_offsets < n_cols,
    )


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError("This demo expects a 2D tensor")
    if not x.is_contiguous():
        x = x.contiguous()

    n_rows, n_cols = x.shape
    block_size = triton.next_power_of_2(n_cols)
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
    )
    return out


def run_case(name: str, x: torch.Tensor, atol: float, rtol: float) -> None:
    out_triton = softmax(x)
    if x.dtype == torch.float16:
        out_torch = torch.softmax(x.float(), dim=1).half()
    else:
        out_torch = torch.softmax(x, dim=1)

    torch.testing.assert_close(out_triton, out_torch, atol=atol, rtol=rtol)
    max_error = (out_triton.float() - out_torch.float()).abs().max().item()
    print(f"PASS: {name}, shape={tuple(x.shape)}, dtype={x.dtype}, max_error={max_error}")


if __name__ == "__main__":
    torch.manual_seed(0)
    run_case("fp32 softmax", torch.randn(128, 256, dtype=torch.float32), 1e-5, 1e-5)
    run_case("fp16 softmax", torch.randn(128, 256, dtype=torch.float16) * 5, 1e-3, 1e-3)
