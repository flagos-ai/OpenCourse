"""FlagPrism Debugger Iluvatar experiment main script (iluvatar / CoreX, BI-V150).

Two Triton kernels, each marking out the region to capture with collect markers:
- vector_add : adds two arrays element by element, pure memory access, used to
  look at the address summary of load/store.
- rmsnorm    : with sum, square root, and multiply, used to look at the numeric
  summary (mean / min / max / l2_norm).

The collection level is controlled by environment variables
(default level=1, addr_level=1):
    FLAGTREE_DEBUGGER_RECORD_LEVEL  numeric collection level 1 or 2
    FLAGTREE_DEBUGGER_ADDR_LEVEL    address collection level 0 / 1 / 2

The Iluvatar backend is accessed through a CUDA compatibility layer, so tensors
are placed on "cuda" and synchronized with torch.cuda.
"""

import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

import flagtree.debugger as debugger
import flagtree.language as ftl

OUTPUT_DIR = Path("/tmp/flagtree_debugger_tianshu")

# Collection policy: read from environment variables, so you can switch level
# from the command line without editing the script.
RECORD_LEVEL = int(os.environ.get("FLAGTREE_DEBUGGER_RECORD_LEVEL", "1"))
ADDR_LEVEL = int(os.environ.get("FLAGTREE_DEBUGGER_ADDR_LEVEL", "1"))

# The in-kernel collect markers must receive these as constexpr.
RECORD_LEVEL_CE = tl.constexpr(RECORD_LEVEL)
ADDR_LEVEL_CE = tl.constexpr(ADDR_LEVEL)

# 1) Persist the default configuration: output directory, record capacity,
#    whether to write a raw-record sidecar.
debugger.configure(
    output_dir=OUTPUT_DIR,
    record_capacity=4096,
    export_raw_records=False,
)
# 2) Enable the process-level Debugger pipeline. Must be called before
#    compiling the kernel to be debugged.
debugger.activate(level=RECORD_LEVEL, addr_level=ADDR_LEVEL)


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Only Triton ops between the markers are collected.
    ftl.debug_collect_start(level=RECORD_LEVEL_CE, addr_level=ADDR_LEVEL_CE)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
    ftl.debug_collect_end()


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x_off = row * n_cols + cols

    ftl.debug_collect_start(level=RECORD_LEVEL_CE, addr_level=ADDR_LEVEL_CE)
    x = tl.load(x_ptr + x_off, mask=mask, other=0.0)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = x * rstd * w
    tl.store(out_ptr + x_off, out, mask=mask)
    ftl.debug_collect_end()


def run_vector_add(n_elements=4096):
    x = torch.arange(n_elements, dtype=torch.float32, device="cuda")
    y = torch.full_like(x, 2.0)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n_elements, 256), )
    vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=256, num_warps=1)
    torch.cuda.synchronize()
    ok = torch.allclose(out, x + y)
    return ok


def run_rmsnorm(n_rows=4, n_cols=1024, eps=1e-6):
    x = torch.randn(n_rows, n_cols, dtype=torch.float32, device="cuda")
    w = torch.ones(n_cols, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows, )](x, w, out, n_cols, eps, BLOCK_SIZE=block,
                               num_warps=4)
    torch.cuda.synchronize()
    ref = x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    return ok


def main():
    print(f"record_level={RECORD_LEVEL} addr_level={ADDR_LEVEL}")
    add_ok = run_vector_add()
    rms_ok = run_rmsnorm()

    # Take all runs exported this time and print the key fields. take_* clears
    # the internal list.
    runs = debugger.take_exported_runs()
    summary = {
        "vector_add_allclose": bool(add_ok),
        "rmsnorm_allclose": bool(rms_ok),
        "exported_runs": len(runs),
        "reports": [r.get("report_path") for r in runs],
        "json_reports": [r.get("json_report_path") for r in runs],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Debug reports written, directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
