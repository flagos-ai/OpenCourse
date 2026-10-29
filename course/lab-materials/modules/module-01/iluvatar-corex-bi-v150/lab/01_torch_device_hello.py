"""PyTorch on BI-V150: explicit device selection and synchronized timing."""
import time
import torch

DEVICE = "cuda:0"


def matmul_on_device(m=1024, k=1024, n=1024, iters=50):
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    a = torch.randn((m, k), dtype=torch.float32, device=DEVICE)
    b = torch.randn((k, n), dtype=torch.float32, device=DEVICE)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    return c, (time.perf_counter() - start) * 1e3 / iters


def main():
    c, elapsed_ms = matmul_on_device()
    print("device:", c.device)
    print("shape:", tuple(c.shape))
    print(f"steady-state: {elapsed_ms:.3f} ms")
    assert c.device.index == 0


if __name__ == "__main__":
    main()
