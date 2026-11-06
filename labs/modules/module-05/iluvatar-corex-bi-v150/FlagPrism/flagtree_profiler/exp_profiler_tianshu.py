#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlagTree Profiler experiment script (Iluvatar / Tianshu / CoreX)

Aligned with the actual environment:
  - wheel: flagtree 0.7.1a1+iluvatar3.6 (pip prebuilt, FlagPrism bundled)
  - public API: import flagtree.profiler (start/finalize/scope/cpu_timed_scope/state ...)
  - Iluvatar backend aliases: tianshu / corex / iluvatar (all equivalent)
  - Iluvatar collection goes through ixKN; it must be attached from the process
    start point via the CLI's --ixkn

Experiment contents:
  1. Two Triton kernels (vector addition + RMSNorm) forming a hierarchical call
  2. shadow context + profiler.scope region tagging, injecting a custom metric (bytes)
  3. hook="triton" launch_metadata injects each kernel's flops/bytes/name before launch
  4. cpu_timed_scope compares CPU exclusive time against accelerator time (inclusive)
  5. state annotations append a suffix to the call path for grouping

See the __main__ comments at the end of the file and the README for the two run modes.
"""

import torch
import triton
import triton.language as tl

import flagtree.profiler as profiler


# ----------------------------------------------------------------------------
# launch_metadata: inject metadata (name / FLOPs / bytes) before launching the GPU kernel
# Takes effect together with hook="triton"
# ----------------------------------------------------------------------------
def add_metadata(grid, metadata, args):
    n = args["n_elements"]
    return {
        "name": "vector_add",
        "flops32": float(n),          # 1 fp32 add per element
        "bytes": int(n * 4 * 3),      # read x, read y, write out, 4 bytes each
    }


def rmsnorm_metadata(grid, metadata, args):
    rows = args["n_rows"]
    cols = args["n_cols"]
    return {
        "name": "rmsnorm",
        "flops32": float(rows * cols * 4),   # square + accumulate + multiply + weight multiply, roughly ~4N
        "bytes": int(rows * cols * 4 * 3),
    }


# ----------------------------------------------------------------------------
# Kernel 1: vector addition
# ----------------------------------------------------------------------------
@triton.jit(launch_metadata=add_metadata)
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out


# ----------------------------------------------------------------------------
# Kernel 2: RMSNorm (row-wise normalization)
# ----------------------------------------------------------------------------
@triton.jit(launch_metadata=rmsnorm_metadata)
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_rows, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)                      # compute mean square in fp32 to preserve precision
    mean_sq = tl.sum(xf * xf, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = (xf * rstd).to(x.dtype) * w
    tl.store(out_ptr + row * n_cols + cols, out, mask=mask)


def rmsnorm(x, w, eps=1e-6):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    rmsnorm_kernel[grid](x, w, out, n_rows, n_cols, eps, BLOCK_SIZE=block)
    return out


# ----------------------------------------------------------------------------
# Workload: shadow scopes build the hierarchy + inject metrics
# ----------------------------------------------------------------------------
def run_workload():
    device = "cuda"   # the Iluvatar CoreX driver is compatible with CUDA device semantics
    n = 1 << 20       # 1M elements
    rows, cols = 2048, 1024

    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    m = torch.randn(rows, cols, device=device, dtype=torch.float32)
    w = torch.randn(cols, device=device, dtype=torch.float32)

    # Warm up: avoid polluting the profile with first-time compilation time
    vector_add(x, y)
    rmsnorm(m, w)
    torch.cuda.synchronize()

    out_add = out_norm = None

    # The second scope argument is a custom metric dict; the profiler aggregates it per scope
    with profiler.scope("elementwise", {"bytes": n * 4 * 3}):
        for _ in range(20):
            out_add = vector_add(x, y)

    with profiler.scope("normalization"):
        with profiler.cpu_timed_scope("rmsnorm_host"):   # measure CPU exclusive time
            for _ in range(20):
                out_norm = rmsnorm(m, w)

    # state annotation: append a suffix to the call path to help grouping and aggregation
    with profiler.scope("mixed"):
        with profiler.state("fused_region"):
            with profiler.scope("add_in_mixed"):
                vector_add(x, y)
            with profiler.scope("norm_in_mixed"):
                rmsnorm(m, w)

    torch.cuda.synchronize()
    return out_add, out_norm


def main():
    # WARNING: on Iluvatar it is recommended to run via CLI + --ixkn (see README). CLI mode
    #    auto-handles start/finalize; the start/finalize below only takes effect when run
    #    "directly with python3.12".
    profiler.start(
        name="prof_tianshu",
        context="shadow",
        backend="iluvatar",   # equivalent: tianshu / corex, pick any per your wheel version
        hook="triton",        # enable the launch_metadata callback
    )
    try:
        run_workload()
    finally:
        profiler.finalize()
    print("profile written out: prof_tianshu.hatchet (aggregated)")


if __name__ == "__main__":
    main()
