# Kernel Profiling and Debugging

> Lab name: **Kernel Profiling and Debugging**
> Based on FlagPrism (Iluvatar / CoreX, BI-V150), comprising two sets of labs: Profiler + Debugger.

This document merges two FlagTree labs into one self-contained guide: **Profiler** and
**Debugger**. Both ship inside the same FlagTree wheel, but they solve problems at different
levels. Each part contains a complete lab write-up plus ready-to-run script source code. It
assumes you are encountering these tools for the first time, so every new term is explained the
first time it appears.

> The subdirectories for the two labs sit alongside this file: the Profiler is in
> `flagtree_profiler/`, the Debugger is in `flagtree_debugger/`. This combined guide compiles the
> `README_实验说明.md` files and scripts from both sides into one document; the content is
> identical to the original files in each subdirectory.

## A few shared terms to get straight first

- **GPU kernel**: a small compute program that actually runs on the graphics card. One pass of a
  model launches hundreds or thousands of kernels behind the scenes.
- **Triton**: a tool for writing GPU kernels in Python syntax, simpler than writing CUDA directly.
  All the kernels in this lab are written in Triton.
- **FlagTree / FlagPrism**: FlagTree is a Triton environment that supports a variety of
  domestically produced graphics cards; FlagPrism is a toolset inside it, containing two things —
  the **Debugger** and the **Profiler**.
- **Iluvatar / iluvatar / CoreX / tianshu**: all refer to the same card vendor "Iluvatar Corex"
  and its software stack. Different contexts use different names; whichever you run into, just
  treat it as "Iluvatar". The card model used on this machine is **BI-V150**.
- **Iluvatar goes through a CUDA compatibility layer**: in the scripts, tensors are placed on
  `"cuda"` and synchronized with `torch.cuda.synchronize()` — this is not a mistake; Iluvatar
  reuses CUDA's driver API.

## FlagPrism vs Iluvatar's built-in profiler: positioning

FlagPrism is the cross-vendor tool layer inside FlagTree — you write
`import flagtree.profiler/debugger` once in a unified way and reuse it across Iluvatar, Ascend,
and other domestic cards, seeing reports organized by Triton statement/tag. Iluvatar's built-in
profiler is the low-level hardware probe **ixKN**, which only recognizes Iluvatar cards and reads
hardware counters directly. The two are not in competition: when running on Iluvatar, FlagPrism
does not replace ixKN; instead it **drives ixKN to collect** data, then organizes the raw data
into a portable form that can be compared across cards.

## Profiler vs Debugger in one line

| | Profiler | Debugger |
|---|---|---|
| What it answers | which kernel is slow, how much data was moved, whether the card is fully utilized | whether the value and memory access address of every statement inside a kernel are correct |
| Public import | `import flagtree.profiler` | `import flagtree.debugger` |
| In-kernel annotation | `profiler.scope(...)` / `hook="triton"` | `ftl.debug_collect_start/end` |
| How to launch on Iluvatar | must wrap the script with `flagtree-profiler --ixkn` | just run with `python` directly, no CLI wrapper needed |
| Output | `.hatchet / .timeline.json / .meta.json / .vendor.json` | `.txt / .json / _op_log / _artifacts` |
| Viewing results | `flagtree-profiler-viewer` | read the `.txt` report directly / read run fields |
| How to install | the same FlagTree wheel | the same FlagTree wheel |

---
---

# Part 1: Profiler

This part records one complete profiler lab, split into two major blocks:
**Block 1 "Measure kernel performance"** (install software -> understand the principles -> run the
experiment), and **Block 2 "Interpret the results"** (read the report -> read the real data ->
gap analysis).

## Related files

- `exp_profiler_tianshu.py` — main lab script (two Triton kernels + performance annotation)
- `exp_basic_usage.py` — example script for the two basic usages (profile a function / profile a region)
- `make_gap_charts.py` — script that generates the "expected vs actual" comparison charts

See the source at the end of this part under "Profiler script source code".

## P-1: Install the software

Here we use a prebuilt install package (the jargon is **wheel**), rather than compiling from source
yourself. The package already bundles FlagPrism, so there is no need to install the profiler
separately.

```bash
# Prerequisite: install PyTorch first

# 1. First uninstall the community edition of triton, otherwise it will conflict with the one
#    bundled in FlagTree. Sometimes one uninstall does not remove it cleanly, so repeat this
#    command until it reports there is no triton left to uninstall.
python3 -m pip uninstall -y triton

# 2. Tell pip to download from FlagOS's private package source (this package is not on the public pip index)
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"

# 3. Install the Iluvatar build of FlagTree. The iluvatar3.6 in the version string indicates this is the Iluvatar-specific build.
python3.12 -m pip install flagtree===0.7.1a1+iluvatar3.6 $RES
```

After installation, FlagPrism's files will appear in these two locations (good to know, but you
normally do not touch them):

```
/root/FlagTree/python/flagtree/_flagprism.py        # internal file, responsible for loading components and checking versions; do not use directly
/root/FlagTree/python/test/unit/test_flagprism.py   # test code for the file above
```

Note: **what you actually use in your code is `import flagtree.profiler`**, not the `_flagprism`
above. The leading underscore in `_flagprism` is the Python convention for "internal part, outsiders
keep off".

**Check whether it installed correctly** (if all four commands run without error, it succeeded):

```bash
# Can the profiler be imported?
python3.12 -c "import flagtree.profiler as p; print('profiler OK', p.__file__)"

# Is the low-level collection engine present?
python3.12 -c "import flagtree.profiler._native; print('native OK')"

# Are the command-line tools installed? (should print two paths)
which flagtree-profiler flagtree-profiler-viewer

# Optional: run the bundled official test to confirm FlagPrism is fine
python3.12 -m pytest /root/FlagTree/python/test/unit/test_flagprism.py -v
```

## P-2: Understand how the profiler works

### What question it answers

You have a Triton program running on the card and want to know: **which kernel is slow, how much
time the CPU and GPU each spent, and how much memory movement there is.** The profiler measures
this data for you and organizes it by "code structure".

### A core idea: "tag" regions of code

The profiler does not dump every kernel's data at you all at once; instead it lets you **partition
and tag your code**, then attributes each kernel's elapsed time to the corresponding tag. This way
what you see is not a pile of isolated numbers, but "how much time each functional module spent".
There are two ways to tag:

- **python mode**: automatically uses "which file, which function, and which line this kernel was
  called from" as the tag.
- **shadow mode**: you manually mark off a region as a tag with `with profiler.scope("name"):`.
  This lab uses this one.

For example, the script writes it like this:

```python
with profiler.scope("elementwise"):   # attach a tag called elementwise
    vector_add(x, y)                   # the kernels running inside here are all attributed to this tag
```

### Where the data comes from: the collection backend

The part that actually goes to the card to read performance data is called the **backend**;
different cards use different backends. The backend Iluvatar uses is called **ixKN** (think of it
as Iluvatar's speed probe).

There is an Iluvatar-specific pitfall to remember here: **ixKN must be attached the moment the
program starts**, not turned on halfway through the run. So on Iluvatar you **cannot rely solely on
writing `profiler.start()` inside the script**; instead you use a command-line tool,
`flagtree-profiler --ixkn`, to "wrap" the execution of your script. P-3 below covers this in detail.

### An extra feature: supply "theoretical data" to the kernel

The profiler measures actual elapsed time, but it does not know how many bytes a kernel "in theory"
should move or how many operations it should compute. You can configure a small function with
`hook="triton"` that tells it these theoretical values before each kernel launch (for example,
`bytes` = the number of bytes to move, `flops32` = the number of floating-point operations to
compute). This way the report has both the measured time and the theoretical quantities, making it
easy to judge "whether this kernel is fully utilizing the card".

### The docs also mention two "basic usages"

The above describes the `scope`-based tagging approach. The official docs also mention two more
basic usages, demonstrated in `exp_basic_usage.py`:

- **Usage 1: measure an entire function** — `profiler.profile(func, name=..., context=...)(args)`.
  First wrap the function, then follow with `(args)` to actually execute and measure it.
- **Usage 2: measure a region, and skip the middle** — `start` to begin, `deactivate` to pause,
  `activate` to resume, `finalize` to end. Suited to the scenario of "wanting to measure only a few
  segments".

```bash
python3.12 exp_basic_usage.py
```

Running it produces two result files: `basic_profile_fn.hatchet` (usage 1) and
`basic_region.hatchet` (usage 2). The way to view them is the same as the main experiment: first
`--list` to see the metrics, then use the real metric names:

```bash
flagtree-profiler-viewer --list basic_profile_fn.hatchet
flagtree-profiler-viewer -m flagtree.ir.host_launch_duration_ns,cpu_time/ns \
  basic_region.hatchet --print-sorted
```

Two Iluvatar-specific details:
1. The Iluvatar backend **does not allow two sessions open at the same time**, so in the script,
   usage 1 manually calls `finalize` to wrap up at the end so that usage 2 can `start` again.
2. In this build, `profiler.profile(...)(args)` returns **the function's computed result** (a
   tensor), not the session_id the docs describe.

Usage 2 lets you specifically verify whether "skipping" took effect: the script runs `vector_add`
15 times (segment A 5 + the skipped segment B 5 + segment C 5); if `deactivate` worked, the result
should correspond to only **10 times (A+C)**.

## P-3: Run the experiment (measure the performance data)

The experiment script defines two Triton kernels:
- **`vector_add`**: adds two arrays element by element. It does almost no computation and is all
  data movement — a textbook "memory-bound" case.
- **`rmsnorm`**: a normalization operation commonly used in deep learning, with summation, square
  root, and multiplication, so the compute load is a bit heavier.

Comparing these two makes the difference in elapsed time between different kinds of kernels clear.

There are two ways to run it. **We recommend running the validation script first to confirm the
script is written correctly, then running the formal measurement.**

### Run the validation script: run the script directly

This way does not enable ixKN; it just confirms that the script runs end to end and the kernels
compute correctly.

```bash
python3.12 exp_profiler_tianshu.py
```

On the first run, Triton has to compile the kernels into code the card can execute. This step
**produces no output and may hang for tens of seconds to a minute, which is normal, so be
patient**. When you see "profile written" printed at the end, it succeeded.

### Run the formal measurement script: wrap the script with the command-line tool

As mentioned earlier, Iluvatar's ixKN must be attached from program startup, so you run the script
with `flagtree-profiler --ixkn`.

```bash
# Simplest form
flagtree-profiler --ixkn -b iluvatar -c shadow -k triton -n prof_tianshu \
  exp_profiler_tianshu.py
```

Parameter explanation:
- `--ixkn`: enable Iluvatar's ixKN speed measurement (required on Iluvatar).
- `-b iluvatar`: select the Iluvatar backend. Iluvatar has three names `iluvatar` / `tianshu` /
  `corex`; use whichever you like, they are the same.
- `-c shadow`: use shadow mode for tagging (corresponds to `profiler.scope` in the script).
- `-k triton`: enable the "supply theoretical data to the kernel" feature mentioned earlier.
- `-n prof_tianshu`: give this measurement a name; the result files will be called `prof_tianshu.xxx`.

To measure faster and with more control, you can add the following parameters to limit the
collection scope:

```bash
flagtree-profiler --ixkn -b iluvatar -c shadow -k triton -n prof_tianshu \
  --ixkn-devices 0 \
  --ixkn-section all \
  --ixkn-launch-skip 2 \
  --ixkn-launch-count 20 \
  --ixkn-export-profile prof_tianshu.ixkn \
  exp_profiler_tianshu.py
```

| Parameter | Effect |
|------|------|
| `--ixkn-devices 0` | measure only GPU 0 |
| `--ixkn-section all` | collect all available metrics |
| `--ixkn-launch-skip 2` | skip the first 2 kernel launches (usually warm-up, which you do not want counted) |
| `--ixkn-launch-count 20` | measure at most 20 kernel launches (the fewer, the faster) |
| `--ixkn-export-profile prof_tianshu.ixkn` | save ixKN's raw data to this file |

> Tip: on your first formal measurement, set `--ixkn-launch-count` small (say 4) to confirm the
> whole pipeline runs through, then open it up to collect the full set. This avoids waiting a long
> time only for the result to get stuck at some step.

After a successful run, you get a result file `prof_tianshu.hatchet`; Block 2 below is about
reading it.

## P-4: Open the result file

The measurement result is stored in a format called **hatchet** (`prof_tianshu.hatchet`), which you
view with the companion viewer tool.

```bash
# The viewer depends on the hatchet library, so install it first. Note the package name is llnl-hatchet, not hatchet!
pip install llnl-hatchet
```

**First thing: look at exactly which metrics this result contains.** Do not copy metric names from
online docs, because the metrics Iluvatar can provide differ from other cards.

```bash
flagtree-profiler-viewer --list prof_tianshu.hatchet
```

> The most common pitfall for beginners: writing `-m time/ns` directly errors out with
> `Metric time is not found`. Because this path on Iluvatar **has no metric called `time` at all**.
> You must first use `--list` to see clearly what is available, then pick from that.

After checking `--list`, use real existing metric names to query. The time-related metric here on
Iluvatar is called `flagtree.ir.host_launch_duration_ns`:

```bash
# See each kernel's elapsed time (--print-sorted sorts by elapsed time, largest first)
flagtree-profiler-viewer -m flagtree.ir.host_launch_duration_ns,cpu_time/ns \
  prof_tianshu.hatchet --print-sorted

# See whether the theoretical data injected by the script (byte count, floating-point operation count) was recorded correctly
flagtree-profiler-viewer -m bytes,flops32 prof_tianshu.hatchet --print-sorted

# See only the parts whose names contain rmsnorm
flagtree-profiler-viewer -m flagtree.ir.host_launch_duration_ns \
  prof_tianshu.hatchet -i "rmsnorm"

# View all command options
flagtree-profiler-viewer -h
```

### Three kinds of metrics you will encounter when reading the report

Each metric name may carry a suffix indicating whether it can accumulate up to parent tags:

- `(inc)` or no suffix: can accumulate to the parent tag (for example, time).
- `(exc)`: does not accumulate upward (for example, CPU time, where accumulating would double-count).
- `(pty)`: just a property, does not accumulate.

## P-5: Understanding the results this experiment actually produced (Iluvatar BI-V150, 2026-11-05)

Below is the real data produced and the pitfalls hit, which you can reproduce by following along.

### 1. What all the on-screen prompts mean during measurement

When running method one, the screen scrolls a lot of `Replaying kernel ...`. Line by line:

- `IXKN-CLI is profiling on Iluvatar BI-V150` — ixKN successfully attached to the Iluvatar BI-V150 card.
- The log automatically changed `-b iluvatar` to `--backend tianshu` — confirming that "the three
  names are the same thing".
- At the start there are two kernels with **extremely long names** (`distribution_elementwise_...`);
  those are used by `torch.randn` generating random input data, not the operators we want to
  measure. Their appearance is normal.
- `Replaying kernel "vector_add_kernel"` / `"rmsnorm_kernel"` — our two kernels are being "replayed"
  and measured one by one. To measure accurately, ixKN runs the same kernel several times; this is
  called replay.
- One **warning** is worth remembering:
  `Warning: Kernel Replay may FAIL when HOST MEMORY is used, please use '--replay-mode application'`
  — it means that when measuring via replay, if the program uses a certain special kind of memory,
  the hardware metrics inside the card may not be measurable. It did not error this time, but it is
  exactly the reason we "could not get the GPU's real execution time" later.

### 2. First pitfall hit: the `time` metric does not exist

At first we wrote `-m time/ns` following the generic docs, and it errored immediately:

```
RuntimeError: Metric time is not found. Use the --list flag to list available metrics
```

Using `--list`, the metrics Iluvatar actually has are these:

```
Accumulable metrics:
- bytes / flops32                     # these two are theoretical values injected by the script itself
- flagtree.ir.host_launch_duration_ns # host-side launch duration <- the only thing here that can be treated as "time"
- flagtree.ir.memory_read_bytes / memory_write_bytes / memory_bytes  # memory read/write bytes
- flagtree.internal.nan_count / mean / min / max ...  # these are for the debugger to inspect values, not performance metrics
Non-accumulable metrics:
- cpu_time                            # CPU time
```

**Key takeaway: although `--ixkn-section all` collected a lot of things, what ultimately gets stored
in the file is mainly "program-level" and "host (CPU)-level" data; the card's real internal
execution time was not stored** (that is the result of the warning above). So in this result, the
only things that can be treated as "time" are `host_launch_duration_ns` (the time the CPU takes to
launch a kernel) and `cpu_time`.

### 3. The actual numbers: the two kernels' elapsed times

```bash
flagtree-profiler-viewer -m flagtree.ir.host_launch_duration_ns,cpu_time/ns \
  prof_tianshu.hatchet --print-sorted
```

| kernel | elapsed time (accumulated) | note |
|--------|------------|------|
| `rmsnorm_kernel` | about 153 ms | slower |
| `vector_add_kernel` | about 124 ms | slightly faster |

- rmsnorm is slower than vector_add, as expected (it has to do extra summation and square root).
- Note this is a **sum across multiple accumulations**, and this time vector_add was measured a
  dozen or so times while rmsnorm was measured only once (because `--ixkn-launch-skip 2` happened to
  skip the rmsnorm side), so **per single run rmsnorm is much slower**.
- `cpu_time` shows about 45 seconds, and this number is **too large** because it counts the time
  ixKN spent replaying repeatedly; it is not the program's real CPU time, so do not take it at face
  value.

### 4. Why a large swath of the result is `nan` (this is not an error)

When you query different metrics, you will find that half the nodes always show `nan` (no value).
The reason: **this measurement produced two sets of data, each hanging off a different tree.**

| Data source | Where it hangs | What it contains |
|---------|-------|--------|
| Theoretical values injected by the script (bytes/flops32) | the tag tree you attached (elementwise, etc.) | byte count, floating-point operation count |
| Actual elapsed time measured by ixKN | the kernel tree ixKN built itself | launch duration, instruction-level detail |

When querying `bytes`, the kernel-tree half has no value (nan); when querying elapsed time, the
tag-tree half has no value (nan). **This is normal, not a bug.** The root cause is that Iluvatar's
ixKN builds its own tree by kernel name and does not merge with the tags you attached. To get time
onto the tag tree as well, you have to use method two (run directly with `python`) to generate a
separate result and view it.

### 5. Verifying the injected theoretical data (fully accurate)

```bash
flagtree-profiler-viewer -m bytes,flops32 prof_tianshu.hatchet --print-sorted
```

Comparing the measured values against the formulas in the script, they match exactly:

| Location | bytes measured | formula verification |
|------|-----------|---------|
| vector_add (single) | 12,582,912 | = element count x 4 bytes x 3 reads/writes ✓ |
| rmsnorm (single) | 25,165,824 | = 2048x1024 x 4 x 3 ✓ |
| elementwise (20 accumulations) | 264,241,152 | = kernel injection x 20 + tag injection x 1 ✓ |

This shows the "supply theoretical data to the kernel" feature works perfectly.

## P-6: What this experiment proved, and what it did not achieve

**The successful part:**
- ixKN successfully measured on Iluvatar BI-V150.
- Able to see elapsed time per kernel (rmsnorm 153ms / vector_add 124ms).
- Able to see the breakdown of each instruction inside the kernel (load / sum / sqrt / store).
- The theoretical byte count and operation count injected by the script were recorded with full accuracy.

**What was not achieved:**
- Did not obtain the **card's real internal execution time**, only the time the CPU took to launch
  the kernel (blocked by that memory warning).
- Could not merge the "measured time" onto the tags you attached (Iluvatar's ixKN builds its own
  tree, bypassing the tags).

## P-7: Expected vs actual — gap comparison and charts

Putting "what you ideally should obtain" and "what Iluvatar actually obtains" side by side makes the
gap clear. The charts are generated by the script below:

```bash
# Install the plotting libraries first (only needed the first time)
pip install matplotlib numpy

# Run the script; it generates 4 png charts in the current directory
python make_gap_charts.py
# Generates 4 charts: gap_1_radar.png / gap_2_injected.png / gap_3_duration.png / gap_4_availability.png
```

### Gap overview table

| What you want | Expected | Iluvatar actual | Gap |
|-----------|------|---------|------|
| Measure on an Iluvatar card | works | runs through | no gap |
| Elapsed time per kernel | card's real execution time | only CPU launch time | **gap**: missing the card's internal time |
| Breakdown of instructions inside the kernel | yes | yes | no gap |
| Injected theoretical data | accurate | fully accurate | no gap |
| CPU time | real value | too large (includes replay) | **gap**: the number is inflated |
| Time on tags | yes | no (time is only on the kernel tree) | **gap**: must use method two to fill it in |
| Hardware metrics like bandwidth/occupancy/instruction count | docs have a bunch | Iluvatar has none | **biggest gap**: these are NVIDIA-exclusive |

### What each of the four charts shows

**Chart 1 Capability radar** (`gap_1_radar.png`): the blue line is expected, the orange line is
actual. Where the two overlap, the capability is in place; the two clearly dented items are "CPU
time" (the number is impure) and "time on tags" (basically not achieved).

**Chart 2 Theoretical data comparison** (`gap_2_injected.png`): each group's two bars are the same
height, showing the injected theoretical values are exactly consistent with the formula-computed
ones — this is proof of "zero gap".

**Chart 3 Elapsed time comparison** (`gap_3_duration.png`): rmsnorm 153ms vs vector_add 124ms. But
remember this is "CPU launch time", not card execution time, so it is classified under "gap".

**Chart 4 Metric availability comparison** (`gap_4_availability.png`): the top row is the "ideally
want everything" metrics, the bottom row is what Iluvatar actually has. The left group (card time,
bandwidth, occupancy, instruction count) is all NO on Iluvatar — because these depend on NVIDIA's
dedicated hardware counters, which this Iluvatar path does not provide. The right group (launch
duration, instruction breakdown, memory bytes, CPU time, injected values) is all YES; these are what
Iluvatar can actually give you.

### One-line summary

**The entire core workflow runs through, and the injected theoretical data is fully accurate.** The
gap concentrates on two points: first, time only comes as "CPU launch duration" rather than "card
execution duration"; second, a whole class of NVIDIA-exclusive hardware metrics (bandwidth,
occupancy, etc.) does not exist at all on Iluvatar. The first point might be achievable by changing
`--replay-mode application`; the second is a boundary of hardware capability, so just accept it.

---

## Profiler script source code

### `exp_profiler_tianshu.py` — main lab script

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FlagTree Profiler lab script (Iluvatar / Tianshu / CoreX)

Aligned with the actual environment:
  - wheel: flagtree 0.7.1a1+iluvatar3.6 (pip prebuilt, FlagPrism bundled in)
  - public API: import flagtree.profiler (start/finalize/scope/cpu_timed_scope/state ...)
  - Iluvatar backend aliases: tianshu / corex / iluvatar (all three equivalent)
  - Iluvatar collection goes through ixKN; must be attached from the process start point with the CLI's --ixkn

Experiment content:
  1. Two Triton kernels (vector add + RMSNorm) build a hierarchical call structure
  2. shadow context + profiler.scope for partitioned annotation, injecting a custom metric (bytes)
  3. launch_metadata with hook="triton" injects each kernel's flops/bytes/name before launch
  4. cpu_timed_scope compares CPU exclusive time with the accelerator's inclusive time
  5. state annotation appends a suffix to the call path for categorization

See the __main__ comments at the end of the file and the README for the two run methods.
"""

import torch
import triton
import triton.language as tl

import flagtree.profiler as profiler


# ----------------------------------------------------------------------------
# launch_metadata: inject metadata before launching the GPU kernel (name / FLOPs / bytes)
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
        "flops32": float(rows * cols * 4),   # square + accumulate + mul + mul weight, rough estimate ~4N
        "bytes": int(rows * cols * 4 * 3),
    }


# ----------------------------------------------------------------------------
# Kernel 1: vector add
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
# Workload: shadow scope builds the hierarchy + injects metrics
# ----------------------------------------------------------------------------
def run_workload():
    device = "cuda"   # Iluvatar CoreX driver is compatible with CUDA device semantics
    n = 1 << 20       # 1M elements
    rows, cols = 2048, 1024

    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.randn(n, device=device, dtype=torch.float32)
    m = torch.randn(rows, cols, device=device, dtype=torch.float32)
    w = torch.randn(cols, device=device, dtype=torch.float32)

    # Warm-up: avoid first-compile time polluting the profile
    vector_add(x, y)
    rmsnorm(m, w)
    torch.cuda.synchronize()

    out_add = out_norm = None

    # scope's second argument is a dict of custom metrics; the profiler aggregates by scope
    with profiler.scope("elementwise", {"bytes": n * 4 * 3}):
        for _ in range(20):
            out_add = vector_add(x, y)

    with profiler.scope("normalization"):
        with profiler.cpu_timed_scope("rmsnorm_host"):   # measure CPU exclusive time
            for _ in range(20):
                out_norm = rmsnorm(m, w)

    # state annotation: append a suffix to the call path for easier categorized aggregation
    with profiler.scope("mixed"):
        with profiler.state("fused_region"):
            with profiler.scope("add_in_mixed"):
                vector_add(x, y)
            with profiler.scope("norm_in_mixed"):
                rmsnorm(m, w)

    torch.cuda.synchronize()
    return out_add, out_norm


def main():
    # WARNING: Iluvatar recommends running with the CLI + --ixkn (see README); the CLI mode
    #    auto-handles start/finalize. The start/finalize below only takes effect when
    #    "running directly with python3.12".
    profiler.start(
        name="prof_tianshu",
        context="shadow",
        backend="iluvatar",   # equivalent: tianshu / corex, pick one per your wheel version
        hook="triton",        # enable the launch_metadata callback
    )
    try:
        run_workload()
    finally:
        profiler.finalize()
    print("profile written: prof_tianshu.hatchet (aggregated)")


if __name__ == "__main__":
    main()
```

### `exp_basic_usage.py` — example of the two basic usages

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runnable example of the two basic usages from the docs' Basic usage section.

The main lab script exp_profiler_tianshu.py uses scope annotation (the shadow usage)
and does not demonstrate the two basic usages from the docs. This file fills that gap:

  Usage 1: profiler.profile(func, ...)(args)   — measure an entire function
  Usage 2: start / deactivate / activate / finalize — manually control which segment to measure and which to skip

Run (first confirm the kernel runs; with method two just use python directly):
  python3.12 exp_basic_usage.py

For formal collection on Iluvatar, wrap it with the CLI the same way:
  flagtree-profiler --ixkn -b iluvatar -c python -k triton -n basic exp_basic_usage.py
"""

import torch
import triton
import triton.language as tl

import flagtree.profiler as profiler


# ----------------------------------------------------------------------------
# A very simple Triton kernel: vector add. Both usages share it.
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
# This is "the function to be measured". Usage 1 wraps it whole and measures it once.
# ----------------------------------------------------------------------------
def my_workload(x, y):
    """Runs vector add 10 times inside, representing one segment of computation to measure."""
    out = None
    for _ in range(10):
        out = vector_add(x, y)
    torch.cuda.synchronize()
    return out


# ============================================================================
# Usage 1: profile a function
#   profiler.profile(func, name=..., context=...)  first returns "the wrapped function";
#   following it with (args) is what actually executes and measures it.
# ============================================================================
def demo_profile_a_function():
    print("\n===== Usage 1: profile an entire function =====")
    device = "cuda"
    x = torch.randn(1 << 20, device=device)
    y = torch.randn(1 << 20, device=device)

    # Key line: wrap my_workload to get a callable object,
    # then call it with (x, y) — this step is what actually executes and gets measured.
    # Note: in this Iluvatar build, the return value here is my_workload's computed result itself,
    #       not a session ID (the session_id behavior described in the docs does not apply to this backend).
    result = profiler.profile(
        my_workload,              # the function to measure
        name="basic_profile_fn",  # result file name
        context="python",         # use "call location (file/function/line number)" as the tag
    )(x, y)                       # actually execute: pass my_workload's arguments

    # Key: profile() opens a session of its own but does not auto-wrap-up. You must manually finalize,
    #       otherwise this session stays open and the next start() errors out due to "session overlap".
    profiler.finalize()

    print(f"Usage 1 done (returned is the computed result tensor, shape={tuple(result.shape)}), "
          f"results written to basic_profile_fn.*")
    return result


# ============================================================================
# Usage 2: profile a region, and demonstrate "skipping a middle segment unmeasured"
#   start   -> begin measuring
#   deactivate -> pause (this segment is not counted)
#   activate   -> resume
#   finalize   -> end and write out
# ============================================================================
def demo_profile_a_region():
    print("\n===== Usage 2: profile a region (with skipping) =====")
    device = "cuda"
    x = torch.randn(1 << 20, device=device)
    y = torch.randn(1 << 20, device=device)

    # Warm-up to avoid first-compile time mixing into the measurement
    vector_add(x, y)
    torch.cuda.synchronize()

    session_id = profiler.start(name="basic_region", context="python")
    print(f"start returned session_id = {session_id}")

    # --- Segment A: measure ---
    for _ in range(5):
        vector_add(x, y)
    torch.cuda.synchronize()

    # --- Skip segment B: these few adds should not be counted in the stats ---
    profiler.deactivate(session_id)
    for _ in range(5):
        vector_add(x, y)          # these 5 are not measured
    torch.cuda.synchronize()
    profiler.activate(session_id)

    # --- Segment C: resume and keep measuring ---
    for _ in range(5):
        vector_add(x, y)
    torch.cuda.synchronize()

    # Write out the data and finish. The final stats only have A+C, 10 times total; segment B's 5 are skipped.
    profiler.finalize()
    print("Usage 2 done, results written to basic_region.* (only counted segments A+C)")


if __name__ == "__main__":
    # Note: the Iluvatar backend does not allow two sessions open at the same time (overlapping sessions).
    # So each demo finalizes to wrap up on its own, so the two demos can be cleanly isolated.
    # If run with the CLI (flagtree-profiler), the CLI takes over start/finalize
    # and only supports a single session — in that case you usually keep only one of the usages.
    demo_profile_a_function()
    demo_profile_a_region()
    print("\nBoth basic usages have finished running.")
```

> For the usage of `make_gap_charts.py` (Profiler version) see P-7; the source is at
> `flagtree_profiler/make_gap_charts.py`. Its structure is identical to the Debugger version, with
> the numbers hard-coded to each experiment's real results.

---
---

# Part 2: Debugger

This part records one complete Debugger lab, split into two major blocks:
**Block 1 "Make the kernel internals visible"** (install software -> understand the principles ->
run the experiment), and **Block 2 "Read the report and locate anomalies"** (look at the four kinds
of report files -> read the summary and address summary -> catch a real NaN).

> The Profiler answers "**which kernel is slow**"; the Debugger answers "**whether the value and
> memory access address computed by each statement inside a kernel are correct**". Both ship inside
> the same FlagTree wheel, but they solve problems at different levels.

## A few Debugger-specific terms

- **debugger**: this is not a debugger in the traditional "set breakpoints, single-step" sense. The
  FlagPrism Debugger **records the intermediate values computed by every Triton statement inside the
  kernel, as well as the memory addresses it reads and writes**, then exports a report afterward.
  You use it to answer: does this step's result have NaN/Inf? Is the access address aligned? Is the
  read/write range out of bounds?
- **summary (numeric summary)**: the Debugger does not store all several-thousand numbers of a
  tensor (that would be too large); instead it computes a few statistics: element count, NaN count,
  Inf count, mean, min, max, L2 norm. This is called the summary.
- **artifact (full dump)**: available only at level 2; it dumps **all values** of a statement's
  result and **all memory access addresses** into `.npy` files. Only needed when troubleshooting
  hard problems.

## Related files

- `exp_debugger_tianshu.py` — main lab script (two kernels, vector_add + rmsnorm, with collect markers)
- `exp_catch_nan.py` — use the Debugger to catch a real NaN/Inf, demonstrating its most core use
- `flagprism_report_summary.py` — report summarization + visualization tool, parses the report json into text tables and png charts (see D-7)
- `make_gap_charts.py` — script that generates the "expected vs actual" capability/boundary comparison charts (see D-9)

See the source at the end of this part under "Debugger script source code".

## D-1: Install the software

Exactly the same as the Profiler — FlagPrism's Debugger and Profiler **ship inside the same FlagTree
wheel**; installing FlagTree gives you both tools at once, with no need to install them separately.

```bash
# Prerequisite: install PyTorch first

# 1. First uninstall the community edition of triton, otherwise it will conflict with the one bundled in FlagTree.
python3 -m pip uninstall -y triton

# 2. Tell pip to download from FlagOS's private package source
RES="--index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple"

# 3. Install the Iluvatar build of FlagTree. The iluvatar3.6 in the version string indicates this is the Iluvatar-specific build.
python3.12 -m pip install flagtree===0.7.1a1+iluvatar3.6 $RES
```

> **When do you need to compile it yourself?** If what you got is a core-only build of FlagTree
> (without FlagPrism), or you are developing inside a FlagTree source tree you checked out yourself,
> then you have to compile jointly from source. The Debugger source is located at
> `third_party/FlagPrism/Debugger`, and the main project compiles it into the same wheel via a
> unified switch:
>
> ```bash
> cd /path/to/FlagTree
> git submodule update --init --recursive
>
> export FLAGTREE_BACKEND=iluvatar        # Iluvatar backend (can also write tianshu)
> export FLAGPRISM_BACKEND=tianshu        # compile only the Iluvatar adapter, do not probe Ascend CANN
> export TRITON_BUILD_FLAGPRISM=ON        # the key switch: only ON compiles the Debugger/Profiler in
> export MAX_JOBS=16
>
> python3 -m pip install . --no-build-isolation
> ```
>
> `TRITON_BUILD_FLAGPRISM=ON` compiles the Debugger, Profiler, and their native runtime together
> within the same CMake graph and packs them into the same wheel. Setting it to `OFF` is core-only,
> without these two tools. Release mode has only two forms, "FlagPrism joint build" and "core-only";
> **there is no standalone Debugger-only wheel**.

**Check whether it installed correctly** (success if none of them error):

```bash
# Can the Debugger be imported? (the only public import path is flagtree.debugger)
python3.12 -c "import flagtree.debugger as d; print('debugger OK', d.__file__)"

# Are both the compiler-side and runtime-side native bindings present?
python3.12 -c "import flagtree.debugger as d; print('available =', d.is_available())"

# Is the language extension where the collection markers live present?
python3.12 -c "import flagtree.language as ftl; print('ftl OK')"
```

> `d.is_available()` returning `True` means both the compiler-side and runtime-side native bindings
> are available at the same time. If it is a core-only wheel, the `import flagtree.debugger` step
> will fail outright — that means your package has no Debugger compiled in at all, and you need to
> switch to a FlagPrism joint-build wheel.

## D-2: Understand how the Debugger works

### What question it answers

You wrote a Triton kernel and the result came out wrong (it has NaN, numerical deviation, or a
suspected out-of-bounds memory access), but the kernel runs on the card, `print` cannot reach in,
and you cannot see what each intermediate step computed to. **The Debugger records the result and
memory access address of every statement inside the kernel and exports a report afterward**, so you
can verify line by line.

### A core idea: only collect the segment you mark off

The Debugger does not record the whole kernel, nor does it record Python/PyTorch operations. It only
records **the segment of Triton statements inside `@triton.jit` that you mark off with markers**:

```python
import flagtree.language as ftl

ftl.debug_collect_start(level=1, addr_level=1)   # start collecting here
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
y = tl.abs(x)
z = y + 1.0
tl.store(y_ptr + offsets, z, mask=mask)
ftl.debug_collect_end()                          # end here
```

The one-line division of labor on each side must be clear:

- **`flagtree.debugger` (Python side)**: responsible for configuration, toggling the pipeline, and
  exporting reports. Use `configure()` to set defaults, `activate()` to turn on, `deactivate()` to
  turn off.
- **`ftl.debug_collect_start / end` (kernel side)**: only responsible for **delimiting** the IR
  region to collect inside `@triton.jit`. It does not configure anything; it just "stakes out the
  plot".

`debugger.activate()` must be called **before compiling the kernel you want to debug** — it opens
the process-level collection pipeline.

### Collection level: level and addr_level

How much the Debugger collects is controlled by two knobs. They are set via `activate()`, and can
also be specified individually for a particular segment in `debug_collect_start()`.

| Knob | Value | Meaning |
|------|------|------|
| `level` | 1 | collect only the **numeric summary** (statistics like nan/inf/mean/min/max/l2_norm) |
| `level` | 2 | summary **plus** exporting the full tensor values as `.npy` artifacts |
| `addr_level` | 0 | do not collect memory access addresses |
| `addr_level` | 1 | collect the **address summary** of load/store (first/last address, stride, access pattern, etc.) |
| `addr_level` | 2 | when `level=2` and the backend supports it, additionally export **each lane's full address**; combined with level=1 it still gives only the summary |

Practical advice: **in daily use start with `level=1, addr_level=1`** — the report is small and
enough to locate the vast majority of problems; only go to `level=2` when you need to verify every
value one by one, or suspect a particular lane's address is computed wrong. `level=2` requires that
`output_dir` not be `None`, and only supports numeric and pointer patterns whose shape can be
determined at compile time and that the current backend can legally lower — forcing level 2 on an
unsupported pattern will **error at compile time**, rather than giving you a report that looks
successful but whose data is actually incomplete.

### Config reference (`debugger.configure`)

| Option | Default | Meaning |
|------|--------|------|
| `output_dir` | `/tmp/flagtree_debugger_manual` | report output directory; set `None` to turn off auto-dumping |
| `record_capacity` | `1024` | number of record slots on the device side; must be a positive integer |
| `export_mode` | `POST_KERNEL_EXPORT` | export after the kernel finishes (also accepts `STREAMING_EXPORT`) |
| `export_on_error` | `False` | whether to still try exporting after the kernel errors |
| `export_raw_records` | `False` | whether to additionally write a decoded raw-record sidecar (for protocol/decode debugging) |
| `timeline` | `False` | insert a device timestamp timeline on supported backends (currently not enabled on Iluvatar) |

> `record_capacity` is important: if `overflow_count` is nonzero in the report, it means there were
> not enough slots and the data was truncated, so the report may be incomplete. Just raise the
> capacity and rerun. Level 2 **does not allow** exporting artifacts from an overflowed buffer.

### A few facts about this Iluvatar (CoreX) path

- The Iluvatar backend is accessed through a **CUDA compatibility layer**, so in the scripts tensors
  are placed on `"cuda"` and synchronized with `torch.cuda.synchronize()` — this is not a mistake;
  Iluvatar reuses CUDA's driver API.
- When the Debugger is active, it only appends a hidden control pointer to kernels that **explicitly
  declare `debug_launch_hidden_arg=True`** (for the Iluvatar/Ascend/MUSA/MetaX/NVIDIA class of
  backends). Other backends' kernel launch ABI is unaffected by the global Debugger state.
- Iluvatar's address summary is computed with scalar i64 arithmetic for **provably contiguous
  pointer chains** (of the form `tt.addptr(tt.splat(base), offsets)` plus a prefix mask); when it
  cannot be fully analyzed, it only reports the "provable" portion of the address information and
  does not fabricate.

## D-3: Run the experiment (make the kernel internals visible)

The main script `exp_debugger_tianshu.py` has two kernels:
- **`vector_add`**: adds two arrays element by element, pure memory access. Used to see what the
  **load/store address summary** looks like.
- **`rmsnorm`**: has squared summation, square root, multiplication. Used to see the **numeric
  summary** (mean/min/max/l2_norm).

The script already writes `configure()` + `activate()` at the top of the file, with the collection
level read from environment variables, defaulting to `level=1, addr_level=1`. Just run it:

```bash
python3.12 exp_debugger_tianshu.py
```

On the first run, Triton has to compile the kernels, which **may produce no output and hang for tens
of seconds; this is normal**. When done it prints a block of JSON listing how many runs were
exported this time and the path of each report.

To change the collection level, there is no need to edit the code; just set environment variables:

```bash
# Look at values only, not addresses (the cleanest report)
FLAGTREE_DEBUGGER_RECORD_LEVEL=1 FLAGTREE_DEBUGGER_ADDR_LEVEL=0 \
  python3.12 exp_debugger_tianshu.py

# Full dump: all values of every statement + every lane's address, saved as .npy
FLAGTREE_DEBUGGER_RECORD_LEVEL=2 FLAGTREE_DEBUGGER_ADDR_LEVEL=2 \
  python3.12 exp_debugger_tianshu.py
```

> Tip: first run with the default level 1 and confirm it produces a report, then go to level 2.
> Level 2 produces many more files, so do not turn it on right out of the gate.

## D-4: After collection ends, which files you get

Report file names are composed of "script name_kernel name_timestamp_run number". One collection may
produce these kinds:

| File | Content | When to look at it |
|------|------|-----------|
| `<...>.txt` | **main report**, organized by Triton source statement, human-readable | first choice for daily debugging, start here |
| `<...>.json` | the JSON version of the main report, same structure | when you need to batch-process with a script |
| `<...>_op_log.txt` / `.json` | organized by MLIR `op_id`, **richer static metadata** | when you want IR-level detail, dtype/shape/alignment |
| `<...>_raw_records.txt` | decoded raw records | only generated when `export_raw_records=True`, for protocol debugging |
| `<...>_artifacts/` | level 2's full value/address `.npy` | only generated at level 2 |

**The main report `.txt` is what you will be looking at 90% of the time.** It is organized by every
`tl.load / tl.store / operation` statement you wrote, placing "this statement's result summary" and
"this statement's access address summary" in their respective positions.

### What the main report looks like (using an abs kernel level 2 fragment as an example)

```text
statement: x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
  [result x]:
    summary:
      element_count: [16]
      nan_count    : [0]
      inf_count    : [0]
      min          : [-8.0]
      max          : [7.0]
    full_value_file: op3_inst0_rec10_value.npy        # only present at level 2
    address_summary(load from):
      status            : [complete]
      first_addr        : [0x12c041200000]
      last_addr         : [0x12c04120003c]
      active_lane_count : [16]
      address_span_bytes: [64]

statement: tl.store(y_ptr + offsets, z, mask=mask)
  memory_access:
    address_summary(store to):
      status            : [complete]
      first_addr        : [0x12c041200200]
      active_lane_count : [16]
```

How to read it:

- **The `summary` of `[result x]`** describes the value `x` read in by `tl.load`: 16 elements, no
  NaN/Inf, range [-8, 7]. This is your main basis for verifying "whether this step computed
  correctly".
- **`address_summary(load from)`** describes the range of **source addresses actually read** by
  `x_ptr + offsets` after the mask, not the address of the register `x` itself. `status: complete`
  means the address analysis is complete; `active_lane_count` is the number of lanes actually
  participating in the access after the mask.
- **`address_summary(store to)`** is the same, the target address written by `y_ptr + offsets`.
- `element_count` is the **logical** lane count, not equal to the number of effective access lanes
  after the mask; for the latter see `active_lane_count`.

### Fields available at level 1

**Numeric summary** (for every statement with a result): `element_count`, `nan_count`, `inf_count`,
`zero_count`, `mean`, `min`, `max`, `l2_norm`.

**Address summary** (`addr_level=1`, for every access op): `first_addr`, `last_addr`, `min_addr`,
`max_addr`, `active_lane_count`, `address_span_bytes`, `address_stride_bytes`, `address_pattern`
(`contiguous` / `strided` / `broadcast` / `irregular`, etc.).

### Getting the exported results directly in Python

If you do not want to dig through files, you can also get each run's path and metadata in the script:

```python
runs = debugger.peek_exported_runs()   # take a peek, do not clear
runs = debugger.take_exported_runs()   # return and clear (the main script uses this one)
debugger.clear_exported_runs()         # just clear
```

Each `run` is a dict, with key fields: `report_path`, `json_report_path`, `op_log_report_path`,
`meta` (backend/device info), `runtime_metadata`. A level 2 run additionally has
`full_dump_artifact_dir`, `full_dump_index_path`.

## D-5: Catch a real NaN (the Debugger's most useful scenario)

Looking only at a normal kernel is not intuitive enough. `exp_catch_nan.py` deliberately writes a
buggy kernel (actual code, collected with `level=1, addr_level=0`, looking at values only, not
addresses):

```python
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)   # x = linspace(-4, 4, 16), the first 8 are negative
d = tl.load(d_ptr + offsets, mask=mask, other=0.0)   # d = ones, but d[3] is changed to 0
ratio = x / d          # d[3]=0 -> this produces inf
root  = tl.sqrt(ratio) # ratio has negatives -> this produces nan
```

Without the Debugger on, you can only see on the host side that the output has nan/inf, but **you do
not know which statement it came from or from which step things went bad**. With the Debugger on,
the level 1 summary pinpoints the problem to a statement. Run:

```bash
python3.12 exp_catch_nan.py
```

What the host side actually prints (the first 8 outputs are nan, the last 8 are normal):

```text
output = [nan, nan, nan, nan, nan, nan, nan, nan, 0.5163975954055786, 0.8944271206855774, ...]
host-side stats: nan = 8  inf = 0
report_path = /tmp/flagtree_debugger_nan/exp_catch_nan_buggy_kernel_..._run1.txt
```

Opening the "Triton Statement Records" of that `.txt` report, the actual content is as follows (real
collection results; the `(U64)`/`(F32)` after the values are data type annotations):

```text
statement: d = tl.load(d_ptr + offsets, mask=mask, other=0.0)
  [result d]:
    summary:
      zero_count   : [1 (U64)]      # <- d has 1 zero, the buried divide-by-zero hazard
      min          : [0 (F32)]

statement: ratio = x / d
  [result ratio]:
    summary:
      inf_count    : [1 (U64)]      # <- the divide-by-zero surfaces here: x/0 = inf
      nan_count    : [0 (U64)]
      min          : [-4 (F32)]     # <- ratio has negatives; the next step's sqrt will go wrong

statement: root = tl.sqrt(ratio)
  [result root]:
    summary:
      nan_count    : [8 (U64)]      # <- sqrt of negatives (and that inf) all become nan here
      inf_count    : [0 (U64)]
```

**This is the value of the Debugger:** the host side can only tell you "the final output has 8
nans", while the Debugger unfolds the chain step by step: at the `d` step you can already see
`zero_count=1` (a buried divide-by-zero hazard) -> `x / d` surfaces 1 inf (divide-by-zero), and
`min=-4` indicates ratio contains negatives -> `tl.sqrt` turns those negatives and that inf together
into 8 nans. Which statement and which step the bad values started from is clear at a glance. After
locating it, go back to the source to fix: `d_safe = tl.where(d == 0, 1.0, d)` to prevent
divide-by-zero, `tl.sqrt(tl.maximum(ratio, 0.0))` to prevent sqrt of negatives; patching both places
is needed to be clean.

> To go further and learn "which lanes went bad", change the script to `level=2`, go into
> `_artifacts/` to read the corresponding statement's `*_value.npy`, and use numpy to find the
> indices where `isnan` is true. Level 1 gives "how many bad values there are", level 2 gives
> "exactly which ones". The next step demonstrates precisely this.

## D-6: Go to level 2 — full dump and pinpoint exactly which lanes

The level 1 summary only tells you "this statement has 8 nans", but not **which** lane went bad. To
pin it down to the lane, you have to turn on level 2: beyond the summary, it dumps each statement
result's **full tensor values** (and, when the backend supports it, the **per-lane memory access
addresses**) to disk as `.npy` artifacts, which you then use numpy to pick out the bad indices from.

### How to turn on level 2

The main script `exp_debugger_tianshu.py` reads the collection level from environment variables, so
you can switch directly; the NaN-catching `exp_catch_nan.py` has `level=1` hard-coded, so either
change two lines or turn it on separately following the minimal fragment below:

```python
# Three key points: level=2, give addr_level 2 if you want per-lane addresses, output_dir must be non-None
debugger.configure(output_dir=Path("/tmp/flagtree_debugger_nan_lv2"),
                   record_capacity=2048)
debugger.activate(level=2, addr_level=2)
# The in-kernel marker must be synced to level=2 as well:
#   ftl.debug_collect_start(level=2, addr_level=2)
```

With the main script, one line of environment variables suffices (no code change needed):

```bash
FLAGTREE_DEBUGGER_RECORD_LEVEL=2 FLAGTREE_DEBUGGER_ADDR_LEVEL=2 \
  python3.12 exp_debugger_tianshu.py
```

> Three hard constraints; hitting them errors directly at **compile time** (rather than giving you a
> truncated report):
> 1. At `level=2`, `output_dir` **cannot be `None`**, otherwise there is nowhere to dump.
> 2. Only numeric/pointer patterns whose shape can be determined at compile time and that the
>    current backend can legally lower are supported; forcing level 2 on an unsupported pattern errors.
> 3. If `record_capacity` overflows (`overflow_count` nonzero in the report), level 2 **does not
>    allow** exporting artifacts from the overflowed buffer. Raise the capacity and rerun.

### The extra files from level 2

Compared to level 1, each run has one additional `_artifacts/` directory and an index:

| File/directory | Content |
|-----------|------|
| `<...>_artifacts/` | each statement result's `.npy`, with file names like `op3_inst0_rec10_value.npy` |
| `*_value.npy` | this statement result's **full values** (always present at level 2) |
| `*_memory_address.npy` | this access's **per-lane addresses** (only present when `addr_level=2` and the backend supports that pointer/mask pattern) |
| `full_dump_index` (run field `full_dump_index_path`) | the mapping table from statement -> corresponding `.npy` file |

The main report `.txt` will have an extra line `full_value_file: op3_inst0_rec10_value.npy` for the
corresponding statement, telling you which one to read. You can also get the directory directly in
Python:

```python
runs = debugger.take_exported_runs()
run = runs[0]
print(run["full_dump_artifact_dir"])   # absolute path of _artifacts/
print(run["full_dump_index_path"])     # the index file, which contains the statement->npy mapping
```

### Use numpy to pick out the bad lanes

Taking that NaN example from D-5, after `root = tl.sqrt(ratio)` is dumped to disk:

```python
import numpy as np

root = np.load("/tmp/flagtree_debugger_nan_lv2/..._artifacts/opX_instY_root_value.npy")
print("bad lane indices:", np.where(np.isnan(root))[0])   # -> [0 1 2 3 4 5 6 7]
print("bad value count:", int(np.isnan(root).sum()))      # -> 8, matches level 1's nan_count

# To learn which access address it is, read the corresponding *_memory_address.npy (only when addr_level=2 and the backend supports it)
addr = np.load("/tmp/flagtree_debugger_nan_lv2/..._artifacts/opX_instY_memory_address.npy")
print("source addresses of the first 8 bad lanes:", [hex(int(a)) for a in addr[:8]])
```

This pins down "there are 8 nans" to "lanes 0-7 are bad, and which block of addresses each of them
read" — which exactly matches the first 8 negatives of the input `x = linspace(-4, 4, 16)`. **Level
1 locates to the statement, level 2 locates to the lane and the address.**

> Practical reminder: the number of files level 2 produces grows linearly with the number of
> statements and records, so do not blindly turn it fully on for a large kernel. A common practice
> is to first use level 1 to narrow the suspect statements down to one or two, then turn on level 2
> only for that segment.

## D-7: Use a script to summarize/visualize the report (optional)

Manually leafing through the `.txt` report is fine for single-case troubleshooting; if you want to
**see at a glance** how the values trend across all statements of one run, or to compare across
multiple runs, use `flagprism_report_summary.py`. It parses the report **json** (the structured
version of the main report `.txt`) and outputs two things:

1. **A text summary table** — lists `elem/nan/inf/zero/min/max/mean/l2` per statement, and
   automatically prints conclusion lines "first statement where inf appears" and "first statement
   where nan appears", pointing out the anomaly source in one line.
2. **A `--plot` png chart** — the top half is a bar chart of the nan/inf/zero counts per statement,
   the bottom half is a line chart of min/max/mean/l2_norm as statements progress, making it clear
   at a glance at which step the values blow up. Requires matplotlib (if not installed it is
   automatically skipped and only text is produced).

```bash
# Automatically find the latest main report under /tmp/flagtree_debugger_* and print the text table
python3 flagprism_report_summary.py

# Text + chart (the chart defaults to *_summary.png in the same directory as the report)
python3 flagprism_report_summary.py --plot

# Specify a particular report
python3 flagprism_report_summary.py /tmp/flagtree_debugger_nan/exp_catch_nan_buggy_kernel_xxx_run1.json --plot

# When fields do not match / the table is empty, look at the structure it actually parsed
python3 flagprism_report_summary.py <json path> --dump
```

For that NaN report from D-5, the text table directly gives: `ratio = x / d` is where inf first
appears, `root = tl.sqrt(ratio)` is where nan first appears — consistent with the conclusion you get
from reading the summary manually, but without leafing through line by line.

> The field hierarchy may differ slightly across FlagPrism versions, so the script does adaptive
> probing (it can parse values with type suffixes like `[16 (U64)]`). If it cannot parse, first use
> `--dump` to see the structure, then adjust. The chart titles contain Chinese; if the machine has
> no Chinese font, boxes will be shown, which does not affect the data.

## D-8: What this experiment proved, and the boundaries to be aware of

**What it can do:**
- The numeric summary (nan/inf/mean/min/max/l2_norm) of every statement inside the kernel is visible.
- The load/store address summary (first/last address, stride, access pattern, effective lane count) is visible.
- Level 2 can dump full values and per-lane addresses to disk as `.npy` for one-by-one verification.
- It can pinpoint an "output has NaN" problem to **exactly which Triton statement**.

**Boundaries to know (Iluvatar / current version):**
- **Address analysis is complete only for "provable" pointer chains.** Iluvatar computes the address
  summary with scalar i64 for contiguous pointer chains; for pointer/mask patterns that cannot be
  fully analyzed, it only reports the provable part and does not fabricate lane addresses. This is a
  limitation of the current CoreX TTIR->TTGIR conversion, not a limitation of hardware address bit width.
- **`addr_level=2` (per-lane full addresses) is available only when the backend supports that
  pointer/mask pattern**; `*_memory_address.npy` is generated only when it is supported.
- **The device instruction timeline is currently not enabled on Iluvatar** (`timeline=True` only
  takes effect on NVIDIA/Ascend).
- **Numerical error diagnosis only covers direct fp16/bf16/fp32 conversions** (`arith.truncf/extf`,
  which report precision_conversion). **Automatic determination** of accumulated drift over long
  compute chains, raw-bit/ULP analysis, near-zero divisors, and sqrt of negatives is not yet done —
  so the NaN example above relies on you reading the summary to locate it; the Debugger currently
  does not automatically conclude "divide-by-zero here".
- **The hidden-argument ABI does not yet penetrate arbitrary call graphs**: helpers/callees whose
  signatures cannot be safely rewritten stay metadata-only, to avoid the Debugger changing the
  original kernel's semantics.
- The report can currently compare by program instance and view lane addresses at level 2, but
  **does not yet** directly output warp id, anomalous-lane clustering, or a "suspicious access
  context" conclusion.

### One-line summary

**The Debugger turns the inside of a Triton kernel from a "black box" into "verifiable statement by
statement".** In daily use, `level=1, addr_level=1` to look at the summary and address summary is
enough to locate the vast majority of numeric/access problems; for hard problems, go to `level=2` to
dump full data. The main boundaries of this Iluvatar path are: address analysis is complete only for
provable pointer chains, the instruction timeline is not enabled, and numeric anomalies have to be
located by you reading the summary rather than by automatic determination.

## D-9: Expected vs actual — capability/boundary comparison charts

Putting D-8's "what it can do" and "the boundaries to know" into charts makes it clear at a glance
where this toolset stands on Iluvatar's current version and what still falls short. The numbers in
the charts are all hard-coded to the real results recorded in this experiment's D-5/D-8, generated
with `make_gap_charts.py`:

```bash
pip install matplotlib numpy

# Generate 4 png charts in the current directory
python make_gap_charts.py
# Generates: gap_1_radar.png / gap_2_nan_chain.png / gap_3_levels.png / gap_4_availability.png
```

### What each of the four charts shows

**Chart 1 Capability radar** (`gap_1_radar.png`): the blue line is expected, the orange line is
actual. Where they overlap, the capability is in place (numeric summary, level 2 full dump, and
per-statement NaN localization are all achieved); the dented ones are "address summary" (complete
only for provable pointer chains), "per-lane addresses" (only when the backend supports that
pointer/mask pattern), and "automatic numeric anomaly determination" (currently relies on a human
reading the summary; the tool does not draw conclusions).

**Chart 2 NaN propagation chain** (`gap_2_nan_chain.png`): draws that real report from D-5 as bars —
the `d = load` step has `zero_count=1` burying a divide-by-zero hazard -> `ratio = x/d` surfaces
`inf_count=1` -> `root = sqrt` becomes `nan_count=8` all at once. Where the bad values first appear
and how they accumulate are unfolded level by level.

**Chart 3 level 1 vs level 2** (`gap_3_levels.png`): daily level 1 gives the numeric summary +
address summary, with a small report; level 2 additionally dumps full values and (when the backend
supports it) per-lane addresses to disk as `.npy`, able to pinpoint exactly which lanes are bad, but
with many more files. Run with level 1 first, go to level 2 for hard cases.

**Chart 4 Capability availability heat table** (`gap_4_availability.png`): YES/~/NO lays out in one
row what the Iluvatar CoreX current version actually has. Achieved: numeric summary, address
summary, level 2 dump, localization to statement, fp conversion precision diagnosis; with
reservations (~): per-lane full addresses (depends on backend pointer/mask support); not yet
available: instruction timeline, divide-by-zero/negative-sqrt automatic determination, hidden-argument
ABI penetrating arbitrary call graphs, warp/anomalous-lane clustering.

> The plotting script is `make_gap_charts.py`, where the numbers are hard-coded to this experiment's
> real results. To swap in your own data, open that file, change the corresponding values, and
> rerun. The chart titles contain Chinese; if the machine has no Chinese font, it automatically
> falls back to English labels.
