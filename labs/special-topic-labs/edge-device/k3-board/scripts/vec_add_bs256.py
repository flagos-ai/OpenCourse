# vec_add_bs256.py — vector addition with BLOCK_SIZE=256
from vec_add import add
import torch


if __name__ == "__main__":
    torch.manual_seed(0)
    n = 98432
    block_size = 256
    x = torch.rand(n)
    y = torch.rand(n)

    out_triton = add(x, y, block_size)
    out_torch = x + y

    torch.testing.assert_close(out_triton, out_torch)
    max_error = (out_triton - out_torch).abs().max().item()
    print(f"PASS: n={n}, BLOCK_SIZE={block_size}, max_error={max_error}")
