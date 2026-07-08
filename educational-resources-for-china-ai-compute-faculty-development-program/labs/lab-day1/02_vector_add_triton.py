"""
02_vector_add_triton.py
-----------------------
Your first Triton kernel running on Ascend NPU.

This is the canonical "vector add" example from the OpenAI Triton tutorial.
The same source compiles and runs on NVIDIA GPUs (with the CUDA backend)
and on Ascend NPUs (with Triton-Ascend).

You will fill in the body of a Triton kernel. The launcher and verification
code is already written for you.

Reference: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,           # pointer to first input vector
    y_ptr,           # pointer to second input vector
    output_ptr,      # pointer to output vector
    n_elements,      # total number of elements (runtime int)
    BLOCK_SIZE: tl.constexpr,  # number of elements each program handles (compile-time)
):
    """
    Element-wise vector addition: output[i] = x[i] + y[i].

    Each "program" (analogous to a CUDA block) handles BLOCK_SIZE
    contiguous elements of the output.

    TODO ------------------------------------------------------------------
    Fill in the kernel body. You need to:

      (1) Get this program's ID along axis 0 (analog of CUDA blockIdx.x).
          API: tl.program_id(axis=0)

      (2) Compute the vector of element offsets this program is responsible
          for. The first offset is `pid * BLOCK_SIZE`; the next BLOCK_SIZE-1
          offsets are consecutive.
          API: tl.arange(0, BLOCK_SIZE) gives [0, 1, ..., BLOCK_SIZE-1]

      (3) Build a boolean mask that is True for offsets within bounds
          (offset < n_elements). This guards against the last block
          when n_elements is not a multiple of BLOCK_SIZE.

      (4) Load x[offsets] and y[offsets] from global memory, applying the mask.
          API: tl.load(ptr + offsets, mask=mask)

      (5) Compute the sum and store it back to output[offsets].
          API: tl.store(ptr + offsets, value, mask=mask)
    -----------------------------------------------------------------------
    """
    # >>> YOUR CODE HERE >>>
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    # <<< END OF YOUR CODE <<<


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Host-side launcher. Allocates the output and launches the kernel."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    # Launch grid: one program per BLOCK_SIZE elements.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def main():
    torch.manual_seed(0)
    # Deliberately not a multiple of BLOCK_SIZE, to exercise the mask.
    size = 98_432

    x = torch.rand(size, dtype=torch.float32, device="npu")
    y = torch.rand(size, dtype=torch.float32, device="npu")

    out_torch = x + y           # PyTorch reference
    out_triton = add(x, y)      # Your Triton kernel

    max_diff = (out_torch - out_triton).abs().max().item()
    print(f"First 5 (torch):  {out_torch[:5].tolist()}")
    print(f"First 5 (triton): {out_triton[:5].tolist()}")
    print(f"Max absolute difference: {max_diff:.2e}")
    if max_diff < 1e-6:
        print("PASS")
    else:
        print("FAIL  -- check your kernel logic")

    # ----- Try-it-yourself -------------------------------------------------
    # 1. Change BLOCK_SIZE in the launcher from 1024 to 512 and re-run.
    #    Notice the brief pause on the first run -- that is JIT recompilation,
    #    because BLOCK_SIZE is a 'tl.constexpr' (compile-time constant).
    # 2. Run the script twice in a row. The second run is much faster
    #    because compiled kernels are cached in ~/.triton/cache/.
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
