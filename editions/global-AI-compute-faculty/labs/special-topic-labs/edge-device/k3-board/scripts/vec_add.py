# vec_add.py — first spine-triton kernel
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 98432
    block_size = 1024
    x = torch.rand(n)
    y = torch.rand(n)

    out_triton = add(x, y, block_size)
    out_torch = x + y

    torch.testing.assert_close(out_triton, out_torch)
    max_error = (out_triton - out_torch).abs().max().item()
    print(f"PASS: n={n}, BLOCK_SIZE={block_size}, max_error={max_error}")
