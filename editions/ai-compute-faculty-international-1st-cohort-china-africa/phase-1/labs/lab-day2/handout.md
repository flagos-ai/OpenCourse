# Day 2 Handout: Tuning a Triton Matmul to the Ascend Cube

## Framing

Yesterday we wrote a tiled matrix multiplication kernel in Triton and
saw it run correctly on the Ascend 910C. It was also 3-5x slower than
`torch.matmul`. A reasonable question: *is that gap fundamental, or
is it the result of specific hardware details we haven't engaged
with?*

Today's answer is the second. We are going to build a **model of the
Ascend Cube compute path from measurement** -- not from the
documentation, not from GPU analogy -- and then tune our kernel
against that measured model.

The lab has four steps, but the logic is a single loop repeated:

> **Hypothesis → Experiment → Data → Refined understanding → Code change**

If you walk away with one thing today, let it be this loop. The
specific hardware facts (buffer capacities, alignment preferences) are
useful, but the loop is the transferable skill.

---

## Step 00: Calibrate

### What we assume coming in

Day 1's benchmark used 256, 512, 1024 square matmuls. Kernel times at
those sizes are a few tens of microseconds -- close enough to kernel
launch overhead and `torch.npu.synchronize()` granularity that any
"optimization" we do could be lost in noise.

### Why this step exists

Before we claim a speedup, we need a stable baseline. Moving the test
shape up to 1024^3 and 2048^3 pushes kernel time into the
hundreds-of-microseconds to milliseconds range, where the signal is
much larger than the noise floor.

### What the numbers should tell you

Typical output:

```
Shape                torch (ms)   triton (ms)    ratio     max diff
----------------------------------------------------------------------
1024 x 1024 x 1024       0.039         0.138     3.52x     1.4e-04
2048 x 2048 x 2048       0.220         0.371     1.69x     1.4e-04
```

Two things are worth noting:

1. Even at 1024^3, the gap is about 3.5x -- not the 5x we saw at
   smaller shapes, but still a lot of room.
2. At 2048^3 the gap is already down to 1.69x without any tuning.
   **Larger shapes are more forgiving**. This is because fixed
   overheads (kernel launch, Python-side dispatch, synchronize)
   become a smaller fraction of total time as the useful work grows.

Whatever you see here is your personal Day 2 starting point. Write
down the 2048^3 numbers: they're what you'll compare against in Step
03.

---

## Step 01: Discover the Cube's Buffer Structure

This is the main event of the day. Everything in Step 02 depends on
the model we build here.

### The hypothesis

A common mental model, imported from GPU thinking, is:

> "The chip has some on-chip memory. As long as my tile fits in it,
> the hardware figures out the rest."

This is wrong for Ascend. A single `tl.dot(A_tile, B_tile)` is not a
monolithic operation -- it's a short pipeline that moves data through
several physically distinct buffers:

```
 Global Memory
      │  DMA
      ▼
  L1 (cbuf)          ← A, B tiles buffered here
      │  MTE
      ▼
 L0A, L0B            ← fed into the Cube unit
      │
  Cube Unit (matrix multiply core)
      │
   L0C               ← accumulator lives here
      │
   UB                ← output C tile passes through
      │
 Global Memory
```

Each stage has its own capacity, and the capacities are all
different. If this model is correct, we should see **two independent
failure modes** when we make tiles too large:

- Large `BM * BN` (big accumulator) with small `BK` → fail at L0C
- Small `BM * BN` with large `BK` (big operands) → fail at L1

If instead the chip had one shared pool, we'd see a single combined
threshold: failure would depend on `BM*BN + BK*(BM+BN)` or similar,
and the failures wouldn't partition cleanly by which dimension we
inflated.

### Why this matters

The model determines the shape of our *legal tile space*. If
everything is a single pool, tile selection is one-dimensional. If
there are two independent buffers, tile selection lives on a 2D
constraint surface -- and the optimal tile shape is likely
**rectangular, not square**, because the two constraints aren't
symmetric.

### What the experiment does

Part A runs 10 tile configurations that vary along both axes.
Compile failures are caught and summarized. The key observation is:
some configurations that are "total size small" still fail, and some
with "total size large" still pass. Total size is not the right
predictor.

Part B runs targeted probes designed to stress one buffer at a time.
The compiler errors carry the answer: they name the buffer (`cc` or
`cbuf`) and give the requested and available sizes in bits.

### Reading the data

Typical output snippet:

```
  256  256   16  BM*BN*4= 256.0KB  (BM+BN)*BK*4=   32.0KB
                 -> FAIL: cc overflow (req 256KB, avail 128KB)

  128  128  128  BM*BN*4=  64.0KB  (BM+BN)*BK*4=  256.0KB
                 -> OK

   64   64 1024  BM*BN*4=  16.0KB  (BM+BN)*BK*4=  512.0KB
                 -> FAIL: cbuf overflow (req 1024KB, avail 512KB)
```

Three things jump out:

1. **There really are two error names**: `cc` and `cbuf`. The
   compiler itself is telling us the architecture has multiple
   buffers, and which one we're exceeding.

2. **The `cc` (L0C, accumulator) limit is 128 KB, exactly.** Probes
   at `BM * BN * 4 = 128 KB` pass; probes at `256 KB` fail. No slack,
   no safety margin -- the hardware gives us the full capacity.

3. **The `cbuf` (L1, operand) limit says 512 KB available, but probes
   at 512 KB fail with "req 1024 KB".** That doubling is not a bug.
   The compiler is invoking itself with
   `--enable-auto-multi-buffer=True --set-workspace-multibuffer=2`,
   which allocates *two* copies of each operand tile so the next K
   iteration's data can be DMAed while the current one is computing.
   This is latency hiding. It costs us half our usable L1 budget,
   but it's doing a favour (see Step 02).

### What to write down

- **L0C capacity = 128 KB**, precise. Constrains `BM * BN`.
- **L1 capacity = 512 KB**, but **effective budget = 256 KB** after
  the default multibuffer=2. Constrains `(BM + BN) * BK`.
- **Storage-compute parallelism is on by default.** You don't need
  `tl.multibuffer`; the compiler runs it for you.

### Legality predicate

Combining the two constraints with 16-byte alignment (Ascend Cube
operates on 16x16 fragments):

```python
def is_legal(BM, BN, BK):
    l0c_ok = (BM * BN * 4) <= 128 * 1024   # 128 KB in bytes
    l1_ok  = ((BM + BN) * BK * 4) <= 256 * 1024
    aligned = (BM % 16 == 0) and (BN % 16 == 0) and (BK % 16 == 0)
    return l0c_ok and l1_ok and aligned
```

This predicate is the output of Step 01. Step 02 uses it as the
search space for autotune.

### Reference answer for the fill-in

```python
L0C_MAX_KB = 128
L1_BUDGET_KB = 256   # = 512 KB / 2 (multibuffer=2)
```

---

## Step 02: Find the Best Legal Tile, and Understand Why

### The hypothesis (Part A)

Step 01 told us *which* tiles are legal (~50-80 on a reasonable
search grid). The next question: which legal tile is fastest?

Naive expectation: the biggest legal square tile wins, because it
maximizes arithmetic intensity. (AI = FLOPs per byte = 2·BM·BN·BK /
(4·(BM+BN)·BK) = BM·BN / (2·(BM+BN)), maximized at fixed
`BM·BN` when BM = BN, and improved further by raising `BM·BN` up to
the L0C cap.)

Correction that the experiment will force: "biggest square" is
probably wrong. A and B are loaded with different stride patterns
(row of A vs column of B), so the real cost of growing BM is not the
same as the real cost of growing BN. The best tile is likely
**asymmetric**.

### Why autotune, not manual search

With a few dozen legal configs, hand-timing each is feasible but
miserable, and we'd still miss cross-interactions. `@triton.autotune`
compiles and times each once, caches the result keyed on `(M, N,
K)`, and uses the winner from then on. First call is slow (compile
farm). Subsequent calls at the same shape are free.

One migration note: **don't tune `num_warps` or `num_stages` on
Ascend**. They are GPU-era scheduling knobs. Triton-Ascend ignores
them and prints `[WARNING] Please DO NOT tune args ['num_warps']!`.
The Ascend scheduler is not warp-based. Including them in
`triton.Config` just wastes compile budget.

### What the data looks like

Typical output from Part A, 2048^3:

```
  version                              time (ms)   vs baseline  vs torch
  ---------------------------------------------------------------------
  baseline (BM=64,BN=64,BK=32)           0.9592         1.00x    0.23x
  autotuned (BM=160,BN=192,BK=64)        0.2246         4.27x    0.98x
  torch.matmul (CANN)                    0.2203         4.35x    1.00x
```

Three things:

1. **4.3x speedup** from the tile choice alone, no new kernel code.
2. **Asymmetric winner**: BN > BM. Our symmetry argument was wrong;
   the experiment corrects it.
3. **Parity with `torch.matmul`** in fp32. The vendor's advantage in
   mixed precision is a separate question (Step 03); at fp32, the
   tile choice is essentially everything.

The exact winner will vary slightly run-to-run -- the autotuner
re-picks at each `(M, N, K)`, and for tiles with near-identical
performance (e.g. 160x192x64 vs 128x192x64 vs 192x128x64), which one
wins depends on measurement noise. This is fine.

### Reference answer for the fill-in (Part A)

```python
def build_configs():
    configs = []
    for BM in [32, 64, 96, 128, 160, 192, 256]:
        for BN in [32, 64, 96, 128, 160, 192, 256]:
            for BK in [32, 64, 96, 128]:
                if is_legal(BM, BN, BK):
                    configs.append(
                        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN, 'BLOCK_K': BK})
                    )
    return configs
```

Candidate ranges are a judgement call. The ones above give ~50-80
legal configs after filtering, which is a reasonable compile budget
(~1-3 min first run).

### Part B: why BK matters even though the formula says it shouldn't

Arithmetic intensity for a single tile is

    AI = BM·BN / (2·(BM+BN))    (FLOPs per byte, BK cancels)

BK does not appear. So why does BK sweep produce any change at all?

Fix `(BM, BN)` at the autotuned winner and vary BK:

```
  BK   L1 KB  time (ms)  vs BK=16
  --   -----  ---------  --------
  16    22.0     0.4539    1.00x
  32    44.0     0.2609    1.74x
  48    66.0     0.3067    1.48x
  64    88.0     0.2246    2.02x
  96   132.0     0.2284    1.99x
 128   176.0     0.2272    2.00x
 160   220.0     0.2294    1.98x
```

Two features:

1. **Main trend**: BK=16 → BK=32 cuts time in half, then BK=64
   roughly matches or beats it, and from BK=64 onward the curve
   flattens. The formula says AI doesn't change with BK, and yet
   time changes dramatically from 16 to 32. Why?

   Because AI only counts data *inside one tile*. The real kernel
   has a loop over K that runs `K/BK` iterations. Each iteration
   has per-iteration overhead (the `tl.dot` dispatch, DMA address
   setup, loop bookkeeping). Small BK means many iterations and a
   lot of that overhead; large BK amortizes it.

   Past some BK, the Cube is already saturated -- compute is fully
   occupied and loading more operands per iteration doesn't help
   anymore. That's the **roofline ceiling** you're watching the
   curve flatten against.

2. **Odd-multiple-of-16 bump**: BK=48 reverts. BK=80, BK=112 tend to
   also revert if you extend the sweep. Even multiples of 16 (32,
   64, 128) are steadier performers.

   The documentation says "align BLOCK to 16". The data says "32
   is actually better". Our interpretation: Ascend's DMA bursts and
   L1 bank layout prefer 32-byte (or 64-byte) boundaries, and BK
   being a multiple of 32 keeps every K-loop iteration aligned.
   This is not in the docs; we found it by measurement. **That's
   the point of Step 02**.

### What to write down

- **Tile shape is asymmetric**. BN > BM by a factor of 1.2-1.5 is
  typical at 2048^3 fp32. Trust autotune.
- **BK is the dominant knob**. Once `BM·BN` approaches the L0C cap,
  shifting BK is what moves you along the roofline.
- **Prefer BK as a multiple of 32**, not just 16.

---

## Step 03: Summary and What We Didn't Do

The final table collects everything on one line:

```
  version                                      ms    vs baseline  vs vendor
  --------------------------------------------------------------------------
  baseline (Day 1 default 64x64x32)         0.9592         1.00x     0.23x
  buffer-model tile (128x128x64, hand)      0.2385         4.02x     0.92x
  autotuned (160x192x64)                    0.2246         4.27x     0.98x
  torch.matmul (CANN hand-tuned)            0.2203         4.35x     1.00x
```

The "buffer-model tile" row is interesting: even without running
autotune, just picking *any* tile the model says is legal and big
gets you 92% of the way to the vendor. **Most of the win is in the
model**; autotune gives you the last few percent.

### What we did not do

1. **Mixed precision**. We kept fp32 throughout. Cube fp16/bf16
   throughput is several times fp32. A fp16×fp16→fp32 matmul
   typically runs 3-5x faster than the fp32 version. That is almost
   certainly where the gap to a fully hand-tuned CANN kernel at
   mixed precision lives. Day 3.

2. **Grouped ordering (`GROUP_SIZE_M`)**. The Triton GPU tutorial
   uses a program-id re-ordering trick to improve cache reuse. On
   Ascend this would affect global-memory and L1 traffic patterns.
   Worth measuring; out of scope today.

3. **Operator fusion**. matmul-then-bias-then-activation is three
   kernel launches today; a fused kernel keeps intermediates
   on-chip. Triton makes this easy, and the time saving is usually
   large for small matmuls.

4. **Manual multibuffer control**. The compiler does `multibuffer=2`
   by default. Triton-Ascend exposes `tl.multibuffer(tensor, n)` if
   you want to override. Not needed for this lab.

### The transferable pattern

The thing we did today is applicable beyond matmul. For any kernel
you write on Ascend:

1. **What is the hardware pipeline actually doing?** Draw the path
   the data takes from global memory to result. Name each on-chip
   buffer it touches.
2. **What capacity limits those stages?** Don't trust the docs
   alone. Construct targeted experiments that stress one stage at a
   time. Read the error messages.
3. **What is the legal tile space?** Express the capacity limits as
   a predicate. Alignment requirements go in here too.
4. **What is optimal inside the legal space?** Use autotune rather
   than intuition. The real costs are usually asymmetric.
5. **Sweep the one knob that didn't show up in your model**, and
   look for the mismatch between predicted and measured. That's
   where the next optimization lives.

This is what "understanding the chip" means in practice.
