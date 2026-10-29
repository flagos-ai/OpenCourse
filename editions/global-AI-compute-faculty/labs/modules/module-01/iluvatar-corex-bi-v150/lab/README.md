# BI-V150 Triton Lab Guide

## Introduction and Learning Objectives

This lab introduces AI systems software and heterogeneous computing through
one continuous Triton workflow on the Iluvatar BI-V150:

```text
Environment checks and device execution
        ↓
Vector Add: program / grid / offsets / mask
        ↓
Softmax and LayerNorm: numerically stable reductions
        ↓
Tiled GEMM: 2-D tiles / strides / K-loop / tl.dot
        ↓
Baseline and benchmarking: warmup / synchronization / output reuse
        ↓
Block sweep / BK sweep / scheduler sweep / autotune
        ↓
Correctness and performance summary
```

After completing the lab, you should understand:

- The basic division of work between the CPU host and GPU device.
- Triton programs, grids, offsets, masks, and blocks.
- Basic implementations of Softmax, LayerNorm, and tiled GEMM.
- The GEMM K-loop, fp16 inputs/outputs, and fp32 accumulation.
- Warmup, synchronization, correctness checks, and stable timing.
- The roles of block sweeps, scheduler sweeps, and Triton autotuning.

Most of the code is already provided. Only two core programming tasks remain:

1. Complete `add_kernel` in `02_vector_add_triton.py`.
2. Use the PASS results from `07_block_sweep.py` to complete `BLOCK_TILES` and
   `AUTOTUNE_CONFIGS` in `08_autotune.py`.

The sections you need to fill in are marked as follows:

```python
# >>> YOUR CODE HERE >>>
...
# <<< END OF YOUR CODE <<<
```

Try completing the TODOs and observing the results before consulting the
reference solutions at the end of this guide.

## BI-V150 and the CoreX Software Stack

The BI-V150 is a GPU accelerator designed for AI computing. This lab uses the
following software stack:

```text
+------------------------------------------------------+
| Python application                                   |
| PyTorch operators / Triton kernels                    |
+------------------------------------------------------+
| CoreX-adapted PyTorch and Triton                       |
+------------------------------------------------------+
| CoreX compiler, runtime and optimized libraries       |
+------------------------------------------------------+
| Driver and device management                          |
+------------------------------------------------------+
| BI-V150 GPU: ivcore11 + device memory                  |
+------------------------------------------------------+
```

CoreX provides a CUDA-compatible device interface, so the code uses `cuda:0`
and `torch.cuda.synchronize()`. Interface compatibility does not mean the
underlying hardware is an NVIDIA GPU:

- BI-V150 uses the ivcore11 architecture and its corresponding compiler backend.
- The lab environment reports a warp size of 64.
- NVIDIA GPU binaries cannot run directly on BI-V150.
- Optimal tiles, scheduler parameters, and performance results from other
  platforms cannot be applied directly.
- API compatibility, source portability, and performance portability are
  distinct concerns.

PyTorch provides high-level tensor and operator interfaces. Triton describes
GPU kernels in a Python-like language and offers explicit control over programs,
grids, tiles, offsets, masks, reductions, and compile-time parameters.

## Environment Setup and Execution Order

Use the official BI-V150 platform image, which should include matching CoreX
drivers, runtime, PyTorch, and Triton. Do not run the following commands: upstream
wheels may overwrite the CoreX-adapted builds.

```bash
pip install torch
pip install triton
pip install --upgrade torch triton
```

From the repository root, enter `lab/` and run the scripts in numerical order:

```bash
cd lab
bash setup.sh
python3 00_check_env.py
python3 01_torch_device_hello.py

# Run after completing the Vector Add TODO
python3 02_vector_add_triton.py
python3 03_softmax_compare.py
python3 04_layernorm_compare.py
python3 05_matmul_compare.py

python3 06_baseline.py
python3 07_block_sweep.py

# Complete the autotune TODO using the sweep results before running these scripts
python3 08_autotune.py
python3 09_summary.py
```

`setup.sh` only checks the environment; it does not install dependencies.
`00_check_env.py` prints software versions and device information, then runs
PyTorch/Triton smoke tests. On your first run, you can also save the following
environment information:

```bash
ixsmi -L
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Lab Steps

### 00–01: Environment Checks and Device Execution

- `00_check_env.py`: print an environment fingerprint and run a PyTorch matrix
  multiplication and a minimal Triton kernel.
- `01_torch_device_hello.py`: demonstrate device selection, warmup, and
  synchronized timing.

Before proceeding, confirm that the device is visible, the output is on `cuda:0`,
and the minimal Triton kernel reports `PASS`. The lab uses the CoreX-compatible
interface, so you do not need to import `torch_npu`.

### 02: Vector Add Programming Task

Complete the body of `add_kernel` in `02_vector_add_triton.py`. The `add()` launcher
and test cases are already provided. Implement the following steps:

1. Use the program ID to identify the block handled by the current program.
2. Generate the element offsets for that block.
3. Create a tail mask to protect the final incomplete block.
4. Load `x` and `y` from device memory.
5. Add the values and store the output.

The script tests the following lengths:

```text
1, 127, 128, 129, 1000, 4097
```

Lengths 127, 129, and 4097 exercise the mask for the final block. A correct
implementation prints `PASS` for every test case.

### 03–04: Softmax and LayerNorm

- `03_softmax_compare.py`: implement numerically stable row-wise Softmax.
- `04_layernorm_compare.py`: implement LayerNorm with fp32 reductions.

When reading the code, focus on the following details:

- Softmax subtracts the row maximum to avoid overflow during exponentiation.
- Softmax uses `-inf` as the padding identity so invalid lanes do not affect
  the row maximum.
- LayerNorm zeros invalid centered lanes so they do not contaminate the variance.
- Masks handle column counts that do not fill a complete block.

The scripts report PyTorch/Triton timings and the maximum error. The basic
operator examples focus on understanding the programming model and passing
correctness checks. An educational Triton kernel can be slower than an optimized
framework operator on small shapes.

### 05: Tiled GEMM

`05_matmul_compare.py` provides tiled GEMM with a K-loop. Study how each 2-D program
handles an output tile, and how strides, tail masks, `tl.dot`, and the accumulator
work together.

GEMM uses a consistent data type convention:

```text
A/B input  : fp16
accumulator: fp32
C output   : fp16
```

The script includes correctness checks for shapes that do not divide evenly into
tiles, and reports framework/Triton timings and the maximum error. Framework
matrix multiplication also serves as the correctness reference during tuning.

### 06: Establish the Baseline

`06_baseline.py` uses a fixed `32x32x32` tile and measures both the end-to-end path
and the preallocated path that reuses the output. The final comparison uses
preallocated outputs for both implementations to avoid output allocation costs.

`common.py` provides the shared GEMM kernel, input generation, correctness checks,
timing, TFLOP/s calculations, and approximate tile footprints.

### 07: Block Sweep and Resource Probes

`07_block_sweep.py` has two parts:

- Part A tries larger tiles to explore compilation, execution, and resource limits.
- Part B checks correctness and measures performance for different
  `BLOCK_M/N/K` combinations.

`PASS tiles to consider for Step 08` lists configurations that compile, run, and
pass correctness checks, making them potential autotune candidates. `PASS` does
not mean fastest, and performance rankings can vary across shapes.

Expected failures in resource probes are experimental results rather than a
failure of the entire script. Preserve PASS, FAIL, and resource errors, and use
compiler messages to interpret failures. Footprint values are approximations
that omit backend implementation details; they do not directly establish
physical buffer capacities.

### 08: Autotune Programming Task and Parameter Analysis

Choose a small, representative set of blocks from the sweep results and complete
`BLOCK_TILES` and `AUTOTUNE_CONFIGS` in `08_autotune.py`. Include candidates with:

- Fast symmetric tiles.
- Asymmetric tiles with different M/N orientations.
- Different `BLOCK_K` values.
- Correctness checks that pass on multiple shapes.

To keep the task manageable, retain only a few blocks and use the same scheduler
parameters for every candidate. Validate the full configuration you actually
use: after changing `num_warps` or `num_stages`, check the candidates again.

The first time autotune encounters an `M/N/K` combination, it evaluates the
candidates and caches the fastest configuration. The script also compares the
autotuned path with a fixed launch using the same configuration to help you
observe dispatch overhead. The latter part of the script includes:

- BK sweep: fix BM/BN and observe how `BLOCK_K` affects the K-loop and resource use.
- Scheduler sweep: fix the block and observe how `num_warps` and `num_stages`
  affect performance.

After completing the task, compare your candidates and parameters with the
reference solution at the end of this guide.

### 09: Correctness and Performance Summary

`09_summary.py` uses `2048x2048 @ 2048x2048` and compares only:

- The fixed `32x32x32` Triton baseline.
- The Triton kernel selected by autotune.

Both paths use the same shape, data types, and output reuse strategy. Framework
matrix multiplication is used only to construct the correctness reference; it
is not included in the final performance ranking.

The original lab documentation reports a BI-V150 speedup of approximately `2.9x`
over the fixed `32x32x32` baseline. Absolute timings and the final digits vary
with device load, so a specific speedup must not be used as a pass criterion.
Record the device, software versions, shape, data types, and selected
block/scheduler configuration together with your results.

## Interpreting Results and Important Notes

### Correctness

- Pass correctness checks before comparing performance.
- Check for NaN/Inf, the maximum error, and `assert_close` results.
- Use shapes that do not divide evenly into tiles to validate tail masks.
- The current Softmax example uses fp32 inputs; LayerNorm explicitly uses
  fp32 reductions.
- GEMM uses fp16 inputs/outputs and fp32 accumulation.
- Interpret GEMM `max diff` together with output scale, absolute tolerance,
  and relative tolerance.

### Performance

- Exclude the first JIT compilation/autotune run from steady-state latency.
- The basic examples use warmup, explicit synchronization, and average timings
  across repeated calls.
- The tuning benchmarks use repeated samples and median timings; both the
  baseline and autotuned kernel reuse preallocated outputs.
- Preserve PASS, FAIL, and resource errors instead of recording only the
  fastest result.
- Include the device, software versions, shape, and data types with every
  performance result.
- Do not apply Ascend or NVIDIA resource capacities and optimal configurations
  directly to BI-V150.

## Reference Solutions

Consult these solutions after completing the TODOs and running the tests.

### Vector Add

Insert the following code into the body of `add_kernel` in
`02_vector_add_triton.py`:

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```

### Autotune Configurations

Fill in `08_autotune.py` using the PASS results from `07_block_sweep.py`.
The following configurations are the reference set provided by the original
lab. Select actual candidates based on results from your current device and
software environment:

```python
BLOCK_TILES = [
    (64, 128, 16),
    (64, 128, 32),
    (128, 64, 32),
    (128, 128, 32),
    (128, 128, 64),
    (64, 256, 32),
]

AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=8,
        num_stages=1,
    )
    for bm, bn, bk in BLOCK_TILES
]
```

## References

- [BI-V150 platform documentation](https://moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu)
- [Official Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
