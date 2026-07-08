"""
03_softmax_compare.py
---------------------
Row-wise softmax: a hand-written Triton kernel vs torch_npu's built-in,
both running on the same Ascend NPU.

About the comparison:
- The "torch_npu" column calls torch.softmax, which on Ascend dispatches
  to an optimized CANN library operator hand-tuned by Huawei engineers.
- The "Triton" column is your kernel, ~30 lines of Python.

Whichever wins, the lesson is the same: Triton is a productive way to
write competitive custom kernels for Ascend, in pure Python.
"""
import time
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per row. Reads the row, computes softmax, writes it back.

    The kernel implements the *numerically stable* softmax:

        m = max(row)
        e = exp(row - m)
        out = e / sum(e)

    Subtracting the row-max before the exp prevents overflow for large inputs.

    TODO ------------------------------------------------------------------
    There are TWO things to fill in:

      (1) Load the row from global memory.
          - This program handles row `row_idx = tl.program_id(0)`.
          - The row starts at `input_ptr + row_idx * input_row_stride`.
          - Use `col_offsets = tl.arange(0, BLOCK_SIZE)` for the columns.
          - Mask out-of-bounds columns (col >= n_cols). For a max-reduction
            to ignore them, use `other=-float("inf")` in tl.load.

      (2) Compute the numerically stable softmax over the loaded row.
          - tl.max(row, axis=0)  -> scalar max of the row
          - tl.exp(...)          -> elementwise exponential
          - tl.sum(..., axis=0)  -> scalar sum of the row
    -----------------------------------------------------------------------
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # >>> YOUR CODE HERE >>>
    # TODO (1): pointer arithmetic + masked load
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # (1) load
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # (2) numerically stable softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # <<< END OF YOUR CODE <<<

    # Store the result row.
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Host-side launcher."""
    n_rows, n_cols = x.shape
    # BLOCK_SIZE must cover an entire row (this kernel is one-row-per-program).
    # Round up to the next power of 2 for efficiency.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y, x,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def benchmark(fn, *args, warmup=10, iters=100, **kwargs):
    """Return mean execution time in milliseconds."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main():
    torch.manual_seed(0)
    print(f"{'Shape':<20} {'torch_npu (ms)':<18} {'Triton (ms)':<14} {'Speedup':<10} {'Max diff':<12}")
    print("-" * 75)

    for n_rows, n_cols in [(512, 256), (1024, 512), (2048, 1024), (4096, 2048)]:
        x = torch.randn(n_rows, n_cols, dtype=torch.float32, device="npu")

        # Correctness
        y_torch = torch.softmax(x, dim=-1)
        y_triton = softmax_triton(x)
        max_diff = (y_torch - y_triton).abs().max().item()

        # Timing
        t_torch = benchmark(torch.softmax, x, dim=-1)
        t_triton = benchmark(softmax_triton, x)

        speedup = t_torch / t_triton if t_triton > 0 else float("inf")
        shape_str = f"{n_rows} x {n_cols}"
        print(f"{shape_str:<20} {t_torch:<18.4f} {t_triton:<14.4f} {speedup:<10.2f}x {max_diff:<12.2e}")

    print()
    print("Notes:")
    print(" - 'torch_npu' calls torch.softmax, which dispatches to a CANN library")
    print("   operator hand-tuned by Huawei engineers in C++.")
    print(" - 'Triton' is your ~30-line Python kernel from this file.")
    print(" - Both run on the SAME Ascend NPU; this is a within-NPU comparison.")

    # ----- Try-it-yourself -------------------------------------------------
    # 1. Add a row of shape (8192, 4096). Does the speedup grow with row size?
    # 2. Switch x to dtype=torch.float16 and observe. What changes?
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
