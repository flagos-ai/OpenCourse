#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runnable examples for the two basic usages described in the docs' "Basic usage" section.

The main experiment script exp_profiler_tianshu.py uses scope tagging (the shadow
usage) and does not demonstrate these two basic usages. This file fills that gap:

  Usage 1: profiler.profile(func, ...)(args)   -- profile an entire function
  Usage 2: start / deactivate / activate / finalize -- manually control which
           part to measure and which part to skip

Run (confirm the kernel runs first; with usage 2 you can just run python directly):
  python3.12 exp_basic_usage.py

For real collection on Iluvatar, wrap it with the CLI the same way:
  flagtree-profiler --ixkn -b iluvatar -c python -k triton -n basic exp_basic_usage.py
"""

import torch
import triton
import triton.language as tl

import flagtree.profiler as profiler


# ----------------------------------------------------------------------------
# A minimal Triton kernel: vector addition. Both usages share it.
# ----------------------------------------------------------------------------
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask)
             + tl.load(y_ptr + offs, mask=mask), mask=mask)


def vector_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK_SIZE=1024)
    return out


# ----------------------------------------------------------------------------
# This is "the function to be profiled". Usage 1 wraps and measures it whole.
# ----------------------------------------------------------------------------
def my_workload(x, y):
    """Runs vector addition 10 times in a row, representing a chunk of computation to measure."""
    out = None
    for _ in range(10):
        out = vector_add(x, y)
    torch.cuda.synchronize()
    return out


# ============================================================================
# Usage 1: profile a function
#   profiler.profile(func, name=..., context=...)  first returns a "wrapped function",
#   then calling it with (args) actually executes and measures it.
# ============================================================================
def demo_profile_a_function():
    print("\n===== Usage 1: profile an entire function =====")
    device = "cuda"
    x = torch.randn(1 << 20, device=device)
    y = torch.randn(1 << 20, device=device)

    # The key line: wrap my_workload to get a callable object,
    # then call it with (x, y) -- this step actually executes and is measured.
    # Note: in this Iluvatar build, the return value here is my_workload's computed
    #       result itself, not a session ID (the session_id behavior described in
    #       the docs does not apply to this backend).
    result = profiler.profile(
        my_workload,              # the function to profile
        name="basic_profile_fn",  # output file name
        context="python",         # use the "call site (file/function/line)" as the tag
    )(x, y)                       # actual execution: pass my_workload's arguments

    # Important: profile() opens a session on its own but does not finalize it
    #       automatically. You must finalize manually, otherwise this session stays
    #       open and the next start() will error out with "overlapping sessions".
    profiler.finalize()

    print(f"Usage 1 done (the return value is the computed result tensor, shape={tuple(result.shape)}), "
          f"results written to basic_profile_fn.*")
    return result


# ============================================================================
# Usage 2: profile a region, and demonstrate "skipping a middle section"
#   start      -> start measuring
#   deactivate -> pause (this section is not counted)
#   activate   -> resume
#   finalize   -> finish and write out
# ============================================================================
def demo_profile_a_region():
    print("\n===== Usage 2: profile a region (with skipping) =====")
    device = "cuda"
    x = torch.randn(1 << 20, device=device)
    y = torch.randn(1 << 20, device=device)

    # Warm up to keep first-time compilation out of the measurement
    vector_add(x, y)
    torch.cuda.synchronize()

    session_id = profiler.start(name="basic_region", context="python")
    print(f"start returned session_id = {session_id}")

    # --- Section A: to be measured ---
    for _ in range(5):
        vector_add(x, y)
    torch.cuda.synchronize()

    # --- Skip section B: these adds should not be counted in the stats ---
    profiler.deactivate(session_id)
    for _ in range(5):
        vector_add(x, y)          # these 5 will not be measured
    torch.cuda.synchronize()
    profiler.activate(session_id)

    # --- Section C: resume and keep measuring ---
    for _ in range(5):
        vector_add(x, y)
    torch.cuda.synchronize()

    # Write out the data and finish. The final stats include only A+C, 10 runs total;
    # the 5 runs in section B are skipped.
    profiler.finalize()
    print("Usage 2 done, results written to basic_region.* (only A+C sections counted)")


if __name__ == "__main__":
    # Note: the Iluvatar backend does not allow two sessions open at the same time
    # (overlapping sessions). So each demo finalizes internally on its own, which
    # lets the two demos be cleanly isolated from each other.
    # If run via the CLI (flagtree-profiler), the CLI takes over start/finalize and
    # only supports a single session -- in that case you usually keep only one usage.
    demo_profile_a_function()
    demo_profile_a_region()
    print("\nBoth basic usages have finished running.")
