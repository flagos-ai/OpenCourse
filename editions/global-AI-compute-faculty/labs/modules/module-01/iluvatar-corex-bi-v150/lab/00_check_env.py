"""Print a reproducible BI-V150/CoreX fingerprint and run two smoke tests."""
import platform
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"


def run_version(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return "command not found"
    except subprocess.TimeoutExpired:
        return "command timed out"


@triton.jit
def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1.0, mask=mask)


def main():
    print("Platform:", platform.platform())
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("ixsmi -L:\n", run_version(["ixsmi", "-L"]))
    assert torch.cuda.is_available(), "CoreX accelerator is not available"
    assert torch.cuda.device_count() >= 1
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    print("Selected:", DEVICE, torch.cuda.get_device_name(0))
    print("Reported warp size:", getattr(properties, "warp_size", "unavailable"))
    print("Total memory (GiB):", f"{properties.total_memory / 2**30:.1f}")
    if "+corex" not in torch.__version__.lower():
        print("WARNING: PyTorch version has no '+corex' marker; verify the platform image.")
    if "+corex" not in triton.__version__.lower():
        print("WARNING: Triton version has no '+corex' marker; verify the platform image.")

    a = torch.randn((512, 512), device=DEVICE)
    b = torch.randn((512, 512), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    print(f"PyTorch matmul first call: {(time.perf_counter() - t0) * 1e3:.3f} ms")
    assert c.shape == (512, 512)

    x = torch.zeros(2049, device=DEVICE)
    y = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 256),)
    add_one_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.ones_like(y))
    print("Minimal Triton kernel: PASS")


if __name__ == "__main__":
    main()
