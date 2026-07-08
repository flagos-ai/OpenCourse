"""
01_torch_npu_hello.py
---------------------
PyTorch "hello world" on the Ascend NPU.

If you have written PyTorch code for an NVIDIA GPU before, the only changes
you need to make for Ascend are:

    import torch_npu               # <-- register NPU backend (do this once)
    x = x.npu()                    # instead of  x.cuda()
    x = torch.randn(.., device='npu')  # instead of  device='cuda'

Everything else -- model definition, training loop, autograd -- is identical.

In this exercise you will port a small CUDA-style snippet to NPU. There is
exactly ONE small thing to fill in.
"""
import time
import torch
import torch_npu  # noqa: F401 -- this import registers 'npu' as a device


def matmul_on_device(M: int, K: int, N: int):
    """
    Run a (M, K) @ (K, N) float32 matmul on the NPU and time it.

    TODO ------------------------------------------------------------------
    Below is a CUDA-style version of this routine. Port it to Ascend NPU.

    Original (CUDA) version:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        A = torch.randn(M, K, dtype=torch.float32, device=device)
        B = torch.randn(K, N, dtype=torch.float32, device=device)
        _ = A @ B                       # warm-up
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            C = A @ B
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) / 50 * 1000
        return C, elapsed_ms

    Your task: change the four 'cuda' references to make this run on NPU.
    Hint: the API is mirror-symmetric -- 'cuda' becomes 'npu' everywhere.
    -----------------------------------------------------------------------
    """
    # >>> YOUR CODE HERE >>>
    device = "npu" if torch.npu.is_available() else "cpu"
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    _ = A @ B                       # warm-up
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        C = A @ B
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    return C, elapsed_ms
    # <<< END OF YOUR CODE <<<


def main():
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"Number of NPU devices: {torch.npu.device_count()}")
    print()

    M = K = N = 1024
    C_npu, elapsed_ms = matmul_on_device(M, K, N)
    print(f"Matmul {M}x{K} @ {K}x{N}: {elapsed_ms:.3f} ms / iter on NPU")

    # Cross-check against CPU
    torch.manual_seed(0)
    A_cpu = torch.randn(M, K, dtype=torch.float32)
    B_cpu = torch.randn(K, N, dtype=torch.float32)
    C_cpu = A_cpu @ B_cpu
    # Note: A and B inside matmul_on_device are different random tensors,
    # so we only sanity-check shape and dtype here, not values.
    assert C_npu.shape == (M, N), f"expected ({M},{N}), got {tuple(C_npu.shape)}"
    assert C_npu.dtype == torch.float32
    print(f"Output shape and dtype correct.")

    # ----- Try-it-yourself -------------------------------------------------
    # 1. Add a bias vector of length N and compute  D = (A @ B) + bias.
    #    Verify the result against a CPU computation.
    # 2. Switch to dtype=torch.float16 and re-time. How much faster is it?
    # -----------------------------------------------------------------------


if __name__ == "__main__":
    main()
