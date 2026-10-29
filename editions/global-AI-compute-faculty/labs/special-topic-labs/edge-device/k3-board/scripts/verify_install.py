# verify_install.py — minimal spine-triton installation check
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 1000
    x = torch.rand(n, dtype=torch.float32)
    y = torch.rand(n, dtype=torch.float32)
    out = torch.empty_like(x)

    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK_SIZE=256)
    torch.testing.assert_close(out, x + y)
    print("PASS: spine-triton installation check completed")
