"""
01_buffer_model.py
------------------
Build a hardware model of the Ascend Cube compute path by experiment.

Where we start
--------------
A common mental model inherited from GPU thinking is:
    "the chip has some on-chip memory; as long as my tile fits, I'm fine."
That is wrong for Ascend. A single tl.dot on Ascend is not a monolithic
operation -- it is a short pipeline that moves data through several
*physically distinct* buffers:

    Global Memory  --DMA-->  L1 (cbuf)  --MTE-->  L0A, L0B
                                                      |
                                                   Cube Unit
                                                      |
                                                     L0C  --UB-->  Global Memory
                              (A, B live here)     (accumulator)

Each level has its own capacity. If the model is right, we should see
*two independent failure modes*:
    - Making BM * BN too large should overflow L0C (accumulator buffer)
    - Making BK * (BM + BN) too large should overflow L1 (operand buffer)

If, on the other hand, the chip had a single unified on-chip pool, we
would see one combined failure threshold on BM*BN + BK*(BM+BN).

We test which one is true, and quantify the capacities.

What you will do
----------------
Part A: Run a broad tile sweep. Observe that some configs compile and
        run, others fail at compile time. Read one full error message
        to find out what the compiler actually complains about.

Part B: Three targeted probes designed to stress *one* buffer at a
        time. Fill in the legality predicate from what you observed.

Part C: Use the model to produce the legal tile space we will hand to
        autotune in step 02.
"""
import re
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from common import matmul_kernel, make_inputs


# =============================================================================
# Run one (BM, BN, BK) and report the outcome in a compact form
# =============================================================================
def try_config(M, N, K, BM, BN, BK, verbose_error=False):
    """Return (status, buffer_name, required_kb, available_kb).

    status is one of 'OK' or 'FAIL'. If 'FAIL', the other fields describe
    which hardware buffer overflowed and by how much, parsed from the
    compiler error message.
    """
    a, b = make_inputs(M, N, K)
    c = torch.empty((M, N), dtype=torch.float32, device="npu")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    try:
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
        torch.npu.synchronize()
        return "OK", None, None, None
    except Exception as e:
        msg = str(e)
        # The compiler error message looks like:
        #   "error: cc overflow, requires 2097152 bits while 1048576 bits avaliable!"
        # (the typo 'avaliable' is theirs, not ours). We parse it here so
        # students don't have to read 5+ screens of MLIR traceback.
        m = re.search(r"(\w+)\s+overflow,\s*requires\s+(\d+)\s+bits\s+while\s+(\d+)\s+bits", msg)
        if m:
            buf = m.group(1)
            req_kb = int(m.group(2)) / 8 / 1024.0
            av_kb = int(m.group(3)) / 8 / 1024.0
            if verbose_error:
                print("  (first 240 chars of raw error:)")
                print("  " + msg.strip().split("\n")[2][:240])
            return "FAIL", buf, req_kb, av_kb
        return "FAIL", "unknown", None, None


def print_row(BM, BN, BK, status, buf, req_kb, av_kb):
    acc_kb = BM * BN * 4 / 1024
    ab_kb = (BM + BN) * BK * 4 / 1024
    if status == "OK":
        print(f"  {BM:>4} {BN:>4} {BK:>4}  "
              f"BM*BN*4={acc_kb:>6.1f}KB  (BM+BN)*BK*4={ab_kb:>7.1f}KB  "
              f"-> OK")
    else:
        if req_kb is not None:
            print(f"  {BM:>4} {BN:>4} {BK:>4}  "
                  f"BM*BN*4={acc_kb:>6.1f}KB  (BM+BN)*BK*4={ab_kb:>7.1f}KB  "
                  f"-> FAIL: {buf} overflow "
                  f"(req {req_kb:.0f}KB, avail {av_kb:.0f}KB)")
        else:
            print(f"  {BM:>4} {BN:>4} {BK:>4}  "
                  f"-> FAIL: {buf}")


# =============================================================================
# Part A: broad sweep
# =============================================================================
def part_a():
    print("=" * 72)
    print("Part A: which tiles compile?")
    print("=" * 72)
    print()
    print("We try 10 tile configurations at (M=N=K=1024). Some will work,")
    print("some will fail at compile time. We want to see the failure pattern.")
    print()

    configs = [
        ( 64,  64,  32),   # Day 1 default
        (128, 128,  32),
        (128, 128, 128),   # notice: 'total size' (BM*BN + (BM+BN)*BK) is
        (256, 256,  16),   # bigger here than in the row below, yet...
        (256, 256,  32),
        (256, 256, 128),
        ( 64,  64, 512),
        ( 64,  64, 1024),
        (128, 128, 512),
        (512, 512, 128),
    ]

    print(f"  {'BM':>4} {'BN':>4} {'BK':>4}  "
          f"{'(L0C proxy)':<16}  {'(L1 proxy)':<18}  result")
    for BM, BN, BK in configs:
        status, buf, req_kb, av_kb = try_config(1024, 1024, 1024, BM, BN, BK)
        print_row(BM, BN, BK, status, buf, req_kb, av_kb)

    print()
    print("Discussion")
    print("-" * 72)
    print("  1. The failures are NOT explained by 'total tile size'. Look at")
    print("     the (128, 128, 128) row vs the (256, 256, 16) row: the first")
    print("     has a LARGER combined footprint yet passes; the second fails.")
    print("  2. The compiler reports TWO different buffer names: 'cc' and")
    print("     'cbuf'. These are two physically distinct pieces of on-chip")
    print("     storage on the Cube path.")
    print("  3. This tells us the Cube compute path is a pipeline with")
    print("     independent buffers at each stage -- not a single shared")
    print("     pool. Which operands stress which buffer? Part B narrows it.")


# =============================================================================
# Part B: targeted probes -- stress ONE buffer at a time
# =============================================================================
def part_b():
    print()
    print("=" * 72)
    print("Part B: isolating each buffer")
    print("=" * 72)
    print()
    print("Hypothesis:")
    print("  L0C (accumulator buffer) is bounded by BM * BN * 4")
    print("    -> big BM*BN with tiny BK should fail on L0C")
    print("  L1 (cbuf, operand buffer) is bounded by (BM + BN) * BK * 4")
    print("    -> tiny BM*BN with big BK should fail on L1")
    print()

    probes = [
        ("L0C stress 1",  256, 128,  16),   # acc=128KB, operands=24KB
        ("L0C stress 2",  256, 256,  16),   # acc=256KB, operands=32KB
        ("L0C stress 3",  128, 256,  16),   # acc=128KB, operands=24KB
        ("L1  stress 1",   64,  64, 512),   # acc=16KB,  operands=256KB
        ("L1  stress 2",   64,  64, 1024),  # acc=16KB,  operands=512KB
        ("L1  stress 3",  128, 128, 512),   # acc=64KB,  operands=512KB
    ]

    print(f"  {'label':<14}  {'BM':>4} {'BN':>4} {'BK':>4}  result")
    for label, BM, BN, BK in probes:
        status, buf, req_kb, av_kb = try_config(1024, 1024, 1024, BM, BN, BK)
        acc_kb = BM * BN * 4 / 1024
        ab_kb = (BM + BN) * BK * 4 / 1024
        if status == "OK":
            print(f"  {label:<14}  {BM:>4} {BN:>4} {BK:>4}  "
                  f"acc={acc_kb:>5.0f}KB  ops={ab_kb:>5.0f}KB  -> OK")
        else:
            print(f"  {label:<14}  {BM:>4} {BN:>4} {BK:>4}  "
                  f"acc={acc_kb:>5.0f}KB  ops={ab_kb:>5.0f}KB  "
                  f"-> FAIL on {buf}, req {req_kb:.0f}KB avail {av_kb:.0f}KB")

    print()
    print("What the probes tell us")
    print("-" * 72)
    print("  * 'L0C stress' probes that fail do so on 'cc'. Those that pass")
    print("    have BM*BN*4 <= 128 KB. Conclusion: L0C capacity = 128 KB.")
    print()
    print("  * 'L1 stress' probes fail on 'cbuf', and the error says")
    print("    'avail 512 KB'. But look carefully: probe 1 has (BM+BN)*BK*4")
    print("    = 256 KB and PASSES, while probe 2 at 512 KB fails with the")
    print("    message 'req 1024 KB'. The required size is DOUBLE what we")
    print("    computed. That's because the compiler has")
    print("        --enable-auto-multi-buffer=True --set-workspace-multibuffer=2")
    print("    enabled by default: it double-buffers A and B in L1 so that")
    print("    the next K-iteration can be loaded while the current one is")
    print("    computing. Good for us (latency hiding), but it halves our")
    print("    usable L1 budget from 512 KB to 256 KB.")
    print()
    print("  * This double-buffering is Ascend's 'storage-compute parallelism'")
    print("    in action. We didn't ask for it; the compiler does it for us.")


# =============================================================================
# Part C: turn the model into a legality predicate
# =============================================================================
# TODO (student) ------------------------------------------------------------
# Based on Part B, fill in the two thresholds (in KB) used below.
#
# Hardware facts you should have just measured:
#   L0C capacity = ?     KB   (bounds BM * BN * 4)
#   L1  capacity = ?     KB   (bounds (BM+BN) * BK * 4, but the compiler
#                              multi-buffers operands by 2x, so your
#                              EFFECTIVE budget is HALF of this)
#
# Set L0C_MAX_KB and L1_BUDGET_KB to the values you measured.
# ---------------------------------------------------------------------------
# >>> YOUR CODE HERE >>>
L0C_MAX_KB = ...      # hint: one of {96, 128, 192, 256}
L1_BUDGET_KB = ...    # hint: HALF of the L1 capacity you measured
# <<< END OF YOUR CODE <<<


def is_legal(BM, BN, BK, dtype_bytes=4):
    """Return True if (BM, BN, BK) fits in both L0C and the L1 budget."""
    l0c_kb = BM * BN * dtype_bytes / 1024
    l1_kb = (BM + BN) * BK * dtype_bytes / 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return (l0c_kb <= L0C_MAX_KB) and (l1_kb <= L1_BUDGET_KB) and aligned


def part_c():
    print()
    print("=" * 72)
    print("Part C: the legal tile space")
    print("=" * 72)
    print()
    if L0C_MAX_KB is ... or L1_BUDGET_KB is ...:
        print("  (Fill in L0C_MAX_KB and L1_BUDGET_KB first, then re-run.)")
        return

    print(f"  Using L0C_MAX_KB = {L0C_MAX_KB}, L1_BUDGET_KB = {L1_BUDGET_KB}")
    print(f"  Plus 16-alignment on every block dimension.")
    print()

    BMs = [16, 32, 48, 64, 96, 128, 160, 192, 256]
    BNs = [16, 32, 48, 64, 96, 128, 160, 192, 256]
    BKs = [16, 32, 48, 64, 96, 128, 160, 192]

    legal = []
    for BM in BMs:
        for BN in BNs:
            for BK in BKs:
                if is_legal(BM, BN, BK):
                    legal.append((BM, BN, BK))

    total = len(BMs) * len(BNs) * len(BKs)
    print(f"  Searched {total} combinations in the grid, "
          f"{len(legal)} are legal.")
    print()
    print("  First 12 legal tiles (as a sanity check):")
    for BM, BN, BK in legal[:12]:
        print(f"    (BM={BM:>3}, BN={BN:>3}, BK={BK:>3})")
    if len(legal) > 12:
        print(f"    ... and {len(legal) - 12} more.")
    print()
    print("  This legal space is exactly what we will hand to @triton.autotune")
    print("  in step 02. We are not picking sizes 'that look reasonable' --")
    print("  we are picking sizes the hardware PHYSICALLY ALLOWS.")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
