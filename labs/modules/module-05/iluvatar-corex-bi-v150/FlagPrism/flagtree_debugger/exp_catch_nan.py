"""Use the FlagPrism Debugger to catch a real numeric anomaly (NaN / Inf).

This kernel deliberately does one "divide by a possibly-zero value" and one
"square root of a negative number". Running it normally, you only see nan/inf
in the output, but you have no idea which statement or which lanes produced
the problem. With the Debugger on, the level 1 summary directly reports the
nan_count / inf_count of each statement's result, pinpointing the faulty
statement at a glance.

Default level=1, addr_level=0 (values only, no addresses, cleanest report).
"""

from pathlib import Path

import torch
import triton
import triton.language as tl

import flagtree.debugger as debugger
import flagtree.language as ftl

OUTPUT_DIR = Path("/tmp/flagtree_debugger_nan")

debugger.configure(output_dir=OUTPUT_DIR, record_capacity=2048,
                   export_raw_records=False)
debugger.activate(level=1, addr_level=0)


@triton.jit
def buggy_kernel(x_ptr, d_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    ftl.debug_collect_start(level=1, addr_level=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    d = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    ratio = x / d          # d contains a 0 -> this produces inf
    root = tl.sqrt(ratio)  # ratio contains negatives -> this produces nan
    tl.store(out_ptr + offsets, root, mask=mask)
    ftl.debug_collect_end()


def main():
    n = 16
    x = torch.linspace(-4, 4, n, dtype=torch.float32, device="cuda")
    d = torch.ones(n, dtype=torch.float32, device="cuda")
    d[3] = 0.0   # create a division by zero
    out = torch.empty_like(x)

    buggy_kernel[(1, )](x, d, out, n, BLOCK_SIZE=16)
    torch.cuda.synchronize()

    print("output =", out.cpu().tolist())
    print("host-side stats: nan =", int(torch.isnan(out).sum()),
          " inf =", int(torch.isinf(out).sum()))

    runs = debugger.take_exported_runs()
    print(f"exported_runs={len(runs)}")
    for r in runs:
        print("report_path =", r.get("report_path"))
    print("\nNow open the .txt report above, find the inf_count of the "
          "ratio = x / d statement,")
    print("then find the nan_count of the root = tl.sqrt(ratio) statement, "
          "and you can pinpoint the source of the anomaly.")


if __name__ == "__main__":
    main()
