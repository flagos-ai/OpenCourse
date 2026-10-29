"""Task 02 demo: complete the Vector Add kernel body."""
import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # >>> YOUR CODE HERE >>>
    # TODO: use the program id to build offsets, mask the tail, load x/y,
    # add the valid elements, and store the result.
    pass
    # <<< END OF YOUR CODE <<<


def add(x, y, block_size=128):
    assert x.is_contiguous() and y.is_contiguous() and x.shape == y.shape
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), block_size),)](
        x, y, out, x.numel(), BLOCK_SIZE=block_size
    )
    return out


def main():
    torch.cuda.set_device(0)
    for n in [1, 127, 128, 129, 1000, 4097]:
        x = torch.randn(n, device=DEVICE)
        y = torch.randn(n, device=DEVICE)
        actual = add(x, y)
        torch.testing.assert_close(actual, x + y)
        print(f"N={n:4d}: PASS")


if __name__ == "__main__":
    main()
