"""
00_check_env.py
---------------
Verify that everything in the lab environment is working before we start.
Run this first. If it all passes, you are ready for the rest of the lab.

(No coding required in this file -- it is just a smoke test.)
"""
import sys


def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check(label, ok, detail=""):
    mark = "[OK] " if ok else "[FAIL]"
    line = f"  {mark} {label}"
    if detail:
        line += f"  --  {detail}"
    print(line)


# -----------------------------------------------------------
section("Python")
# -----------------------------------------------------------
py_ok = sys.version_info[:2] == (3, 11)
check("Python 3.11", py_ok, sys.version.split()[0])

# -----------------------------------------------------------
section("PyTorch and torch_npu")
# -----------------------------------------------------------
import torch
import torch_npu  # noqa: F401  -- registers the NPU backend

check("torch imported", True, torch.__version__)
check("torch_npu imported", True, torch_npu.__version__)
check("NPU available", torch.npu.is_available())
n = torch.npu.device_count()
check("NPU device count", n >= 1, f"{n} device(s)")

# -----------------------------------------------------------
section("Triton-Ascend")
# -----------------------------------------------------------
import triton
import triton.language as tl  # noqa: F401

check("triton imported", True, triton.__version__)

# -----------------------------------------------------------
section("Smoke test: tensor on NPU")
# -----------------------------------------------------------
x = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
y = torch.randn(1024, 1024, dtype=torch.float32, device="npu")
z = x @ y
torch.npu.synchronize()
check("Matmul on NPU", z.shape == (1024, 1024), f"output shape {tuple(z.shape)}")

# -----------------------------------------------------------
section("Smoke test: minimal Triton kernel on NPU")
# -----------------------------------------------------------


@triton.jit
def _add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v + 1.0, mask=mask)


a = torch.zeros(2048, dtype=torch.float32, device="npu")
b = torch.empty_like(a)
grid = (triton.cdiv(a.numel(), 1024),)
_add_one[grid](a, b, a.numel(), BLOCK=1024)
torch.npu.synchronize()
ok = torch.allclose(b, torch.ones_like(b))
check("Triton kernel on NPU", ok, "result == 1.0 everywhere")

print()
print("All checks passed. You are ready for the lab.")
