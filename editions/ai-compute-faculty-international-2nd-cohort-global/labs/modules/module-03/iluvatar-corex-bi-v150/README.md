# Triton Optimization Pass Lab (Iluvatar BI-V150)


English | [简体中文](readme_cn.md)

Use the Iluvatar BI-V150 and the platform's preinstalled CoreX, PyTorch, and Triton to explore compiler optimizations in TTIR, TTGIR, and Iluvatar LLIR through blocked GEMM and fused softmax.

The original IR, binaries, and compiler caches included in `artifacts/` and `lab-results/` were generated on an H200.

## 1. Verify the environment

Open the repository directory on the lab platform and use its preinstalled `python3`:

```bash
python3 check_env.py
```

The script checks the active Python interpreter and package paths, `ixsmi`, the assigned logical BI-V150 device, the Triton backend, and the warp size. It also compiles and runs the two existing kernels with numerical checks. These checks cover a matrix shape that is not divisible by the tile dimensions and a softmax column count that is not a power of two. Reference results are computed on the CPU in float64. The script does not install dependencies or change the platform's environment configuration; compiler caches use a temporary directory.

- A successful check returns `0`. Dependency, device, compilation, or numerical failures return `1`.
- If `ixsmi` is not on PATH, the script prints a warning and continues checking the Python GPU runtime.
- `--device-index` is a logical index visible to the current Python process. It is usually `0` in a single-GPU container; do not infer it from a `/dev/iluvatarN` node name.
- `torch.cuda` and `device="cuda"` are compatibility interfaces used by CoreX. Keep these APIs unchanged.
- Use the platform's matching dependencies. There is no need to create a Conda environment or run `pip install -r requirements.txt`.

To check the environment without compiling kernels, or to save a report:

```bash
python3 check_env.py --skip-smoke
python3 check_env.py --device-index 0 --report environment.json
```

A successful `--skip-smoke` run does not validate the kernels. The version constraints in `requirements.txt` are not installation instructions for the course platform. If a package is missing, follow the platform's instructions for loading its matching environment.

## 2. Generate lab artifacts on the V150

Run the following commands in the same Bash terminal at the repository root. Each run uses a new directory and preserves existing files:

```bash
set -o pipefail
mkdir -p lab-results
V150_RUN_DIR="$(mktemp -d "$PWD/lab-results/v150.XXXXXX")"
printf 'Results directory: %s\n' "$V150_RUN_DIR"
python3 check_env.py --device-index 0 --report "$V150_RUN_DIR/environment.json"
```

After the checks pass, run the existing lab script:

```bash
python3 run_lab.py --kernel all --device-index 0 --dump-dir "$V150_RUN_DIR/baseline" 2>&1 | tee "$V150_RUN_DIR/baseline.log"
cat "$V150_RUN_DIR/baseline/matmul/run.json"
cat "$V150_RUN_DIR/baseline/softmax/run.json"
```

Inspect `device.name`, `compiled_metadata`, `artifact_stages`, and the numerical errors in the new records. The lab script records errors; the environment checker additionally provides independent numerical assertions.

In the explanations below, `artifacts/` denotes the output root for the current run. Replace it with `$V150_RUN_DIR/baseline` when locating files. The platform's adapted compiler determines which stages are exported. A typical output tree is:

```text
baseline/
  matmul/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
  softmax/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
```

The backend does not need to export PTX. Follow `Triton Python → TTIR → TTGIR → BI LLIR → device binary`; use `run.json` and the actual directory contents to identify the files. The Iluvatar IR excerpts below are teaching examples. Layouts, operation names, and pipeline structures may vary with the CoreX version and compilation parameters.

## 3. Start with TTIR: what is being computed?

TTIR is the closest representation to Triton Python. It still exposes high-level operations such as memory loads, stores, matrix multiplication, and reductions, before the detailed thread assignment and shared temporary storage are introduced.

### 3.1 Softmax

Start with `artifacts/softmax/00_ttir.mlir` in the current output directory. One program computes one softmax row:

```text
Identify the current row
→ Load 1024 elements
→ Find the row maximum
→ Subtract the maximum from each element
→ Compute exponentials
→ Sum the exponentials
→ Divide
→ Store the row
```

Students should be able to answer:

1. Why are out-of-bounds loads filled with negative infinity? This prevents them from incorrectly becoming the maximum.
2. Why are intermediate results not written back to device memory? Loading, finding the maximum, exponentiation, reduction, division, and storage all happen within one kernel.

### 3.2 Matmul

Next, read `artifacts/matmul/00_ttir.mlir` in the current output directory. Find these five kinds of operations:

| TTIR operation | Meaning |
|---|---|
| `tt.get_program_id` | Identifies the tile of C assigned to the current program |
| `tt.load` | Loads the A and B tiles needed for the current step |
| `scf.for` | Accumulates over multiple steps along K |
| `tt.dot` | Computes the small matrix product for the current step |
| `tt.store` | Writes the completed C tile back to memory |

Some Iluvatar builds produce TTIR with many i64 conversions, range checks, and helper functions, mainly for address computation and safety checks. Start by following the five kinds of operations above.

## 4. TTGIR: who computes, where data lives, and when it moves

TTGIR adds thread assignments and on-chip temporary storage to TTIR. Focus on these questions rather than the spelling of every symbol:

```text
Which threads process a tile?
Where is the data placed after it is loaded from device memory?
When does the next tile start loading?
How is the data finally delivered to the matrix compute units?
```

Different adapted versions may use `#triton_gpu.*` or `#ttg.*` prefixes. Follow the current output.

### 4.1 `#triton_gpu.blocked`: assigning ordinary data to threads

An example layout for ordinary matmul data:

```mlir
#triton_gpu.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [16, 4],
  warpsPerCTA = [2, 2],
  ...
}>
```

Read it as follows:

| Field | Interpretation |
|---|---|
| `sizePerThread=[1,8]` | Each thread processes 1×8 elements |
| `threadsPerWarp=[16,4]` | A 64-thread warp is arranged as 16×4 threads |
| `warpsPerCTA=[2,2]` | Four warps cooperate in a 2×2 arrangement |

This layout is commonly used for loading A/B, generating addresses and masks, and ordinary elementwise operations. Think of it as a map of which elements belong to each thread.

A and B have different access directions in memory, so the generated IR may choose different blocked layouts for them.

### 4.2 `#triton_gpu.iluvatar_mma`: distributing matrix multiplication results

An example layout for matrix multiplication results:

```mlir
#triton_gpu.iluvatar_mma<{
  versionMajor = 1,
  warpsPerCTA = [2, 2],
  instrShape = [16, 16]
}>
```

This layout is designed for the matrix compute units. It distributes the C accumulators across four warps in a form suitable for the BI-V150's matrix operations.

For the lab, explain:

- `warpsPerCTA=[2,2]`: four warps cooperate on one output tile.
- `instrShape=[16,16]`: the larger matrix multiplication is broken into many 16×16 pieces.
- There is no need to interpret the `versionMajor` numbering.

### 4.3 `#triton_gpu.shared`: placing A and B in shared temporary storage

An example shared-memory layout:

```mlir
#triton_gpu.shared<{..., useTcu = true}>
```

This describes how A and B are arranged in temporary storage shared by the CTA. At a teaching level, `useTcu=true` indicates that this arrangement prepares data for the BI-V150 matrix compute units.

For a three-stage pipeline, look for temporary storage types such as:

```mlir
!tt.memdesc<3x64x32xf16, #shared, mutable>
!tt.memdesc<3x32x64xf16, #shared1, mutable>
```

These represent:

```text
Three A tiles, each 64×32
Three B tiles, each 32×64
```

The leading dimension `3` corresponds to `num_stages=3` in this example. Think of three drawers: while one supplies the current computation, the others can receive data for later steps.

### 4.4 `dot_op`: the final preparation before matrix multiplication

The core GEMM steps are:

```text
Load A from shared temporary storage
→ Prepare A in its dot_op layout
Load B from shared temporary storage
→ Prepare B in its dot_op layout
tt.dot(A, B, current accumulator)
→ Updated accumulator
```

Here, `#triton_gpu.dot_op` indicates that A or B has been arranged for the matrix compute units. The backend's additional `useSme` marker indicates that A and B may use different preparation paths; there is no need to speculate about its microarchitectural details.

`tt.dot` remains the key anchor: the high-level matrix multiplication is still visible, but its inputs are now prepared for the hardware.

## 5. Software pipelining: prepare the next tile while computing

The K loop is split into steps, each processing `BLOCK_K=32`. If every step waits for A/B to finish loading before computing, the matrix compute units can spend time idle.

The following three-stage pipeline is an illustration to compare with the actual IR:

```text
Prepare three sets of A/B tiles
→ Compute with the current set
→ Load later A/B tiles at the same time
→ Wait for their data to be ready before consuming it
```

Corresponding TTGIR operations:

| Operation | Interpretation |
|---|---|
| `local_alloc` | Allocates the three temporary slots |
| `async_copy_global_to_local` | Starts loading A/B from device memory ahead of use |
| `async_commit_group` | Commits the batch of transfers |
| `async_wait` | Checks that data has arrived before it is needed |
| `scf.for` | Starts the main loop |
| `local_load` and `tt.dot` | Read the current data and compute |
| The next copy and wait | Prepare future tiles while processing the current tile |
| Final waits and temporary-storage release | Finish outstanding work after the loop |

A useful analogy:

> While solving the current problem, bring the materials for the next two problems to the desk. Once the current problem is finished, there is no need to wait for the next delivery.

Increasing `num_stages` does not guarantee better performance. More slots may reduce waiting, but they also consume more on-chip storage and registers.

## 6. LLIR: identify the BI-V150 matrix compute path

Look for a target marker such as this in the Iluvatar `artifacts/matmul/02_llir.ll`:

```llvm
target triple = "bi-iluvatar-ilurt"
```

This identifies the Iluvatar low-level target. At this stage, focus on the low-level operations and address spaces:

| LLIR clue | Meaning |
|---|---|
| `addrspace(1)` | Ordinary device data addresses, such as A, B, and C |
| `addrspace(3) @global_smem` | Temporary storage shared within the CTA |
| `llvm.bi.sme.load...` | Prepares matrix inputs for the hardware |
| `llvm.bi.sl.waitcnt` | Waits for earlier preparation or transfers |
| `llvm.bi.sl.barrier.alu` | Synchronizes cooperating threads |
| `llvm.bi.matrix.mad.f32x4.f16x4` | Multiplies fp16 inputs and accumulates in fp32; a low-level counterpart of `tt.dot` |

Search the current matmul LLIR for these operations and follow the sequence: input preparation → waiting → synchronization → matrix multiply-accumulate.

Students do not need to read every LLIR line. They should establish this connection:

```text
TTIR tt.dot
→ Prepared A/B, shared temporary storage, and dot_op in TTGIR
→ llvm.bi.matrix.mad in LLIR
```

This connects the high-level matrix multiplication with the device's matrix compute units.

## 7. Low-level softmax implementation

Softmax has no matrix multiplication. Its focus is cooperation among threads to compute maxima and sums.

In this TTGIR example, a CTA has four 64-thread warps. Each warp first processes its own elements, then the partial results from multiple warps are combined.

In LLIR:

| Clue | Interpretation |
|---|---|
| `llvm.bi.slb.shfl.idx.b32.i32` | Exchanges partial results among threads within a warp |
| `addrspace(3) @global_smem` | Stores partial results for exchange between warps |
| `llvm.nvvm.barrier0()` | Waits for cooperating warps before reading shared results |
| `llvm.exp2.f32` | Implements the exponential computation at a lower level |

A classroom explanation:

> A row of 1024 numbers is divided among groups. Each group finds its own maximum, then the four group results are combined through shared storage to obtain the row maximum. The sum follows the same pattern. The complete intermediate arrays never go back to device memory, so the operation remains fused.

## 8. Student tasks

### Task 1: change the tile

After changing `BLOCK_M`, `BLOCK_N`, or `BLOCK_K`, inspect:

- The A, B, and C shapes associated with `tt.dot`.
- `warpsPerCTA` and `instrShape` in `iluvatar_mma`.
- The shapes of the three-stage A/B temporary storage.
- The layout information in `dot_op`.

Explain how a larger tile can increase data reuse while also consuming more temporary storage and registers.

### Task 2: change num_warps

Inspect:

- The module's `num-warps` attribute.
- `threadsPerWarp` and `warpsPerCTA` in `#triton_gpu.blocked`.
- `warpsPerCTA` in `#triton_gpu.iluvatar_mma`.
- Changes to matrix computation and temporary-data preparation in LLIR.

First calculate the number of threads in a CTA:

```text
CTA thread count = num_warps × 64
```

Then explain why more warps can distribute a tile across more threads but also increase resource use, so they do not always improve performance.

### Task 3: change num_stages

Inspect:

- The leading dimension of `local_alloc<STAGES x ...>`.
- How much data is prefetched before the main loop.
- The placement and waiting depth of `async_wait`.
- Changes in benchmark results.

Explain that stages describe how many batches of data are prepared ahead of use, rather than independent execution streams.

### Task 4: compare matmul and softmax

Matmul focuses on delivering A/B to matrix compute units. Softmax focuses on fusion and cooperative maximum and sum reductions. Softmax has no `tt.dot`, so it does not follow the `llvm.bi.matrix.mad` path.

### Task 5: validate the connection to low-level code

For matmul, find `llvm.bi.sme.load`, `llvm.bi.sl.waitcnt`, `llvm.bi.sl.barrier.alu`, and `llvm.bi.matrix.mad`. Explain their roles in input preparation, waiting, synchronization, and matrix multiply-accumulate.

For softmax, find shuffle, shared-memory, barrier, and exponential operations. Explain how they implement combination within a warp, combination across warps, synchronization, and exponentiation.

## 9. Suggested teaching walkthrough

### Part 1: start with what stays the same

Show the Python kernels. The mathematical definitions of matrix multiplication and softmax stay the same. Moving to a different GPU changes thread assignments, temporary storage, and the specialized compute units used at the end of the compilation pipeline.

### Part 2: follow the main computation in TTIR

For matmul, follow A/B loads, the K loop, `tt.dot`, and C stores.

For softmax, follow the row load, maximum, exponential, sum, and store.

If students get distracted by address checks, explain that these validate addresses rather than define the core algorithm.

### Part 3: answer three questions in TTGIR

1. Who computes? In the example, each warp has 64 threads, and four warps cooperate in a 2×2 arrangement.
2. Where is the data? A/B enter shared temporary storage with three buffering slots.
3. When does data move? Later tiles are loaded while the current tile is being computed.

Useful analogies are a shared desk for shared memory, three drawers for three pipeline stages, and a handoff to the matrix compute units for `tt.dot`.

### Part 4: find the endpoint in LLIR

Focus on these clues rather than every line:

```text
Prepare inputs: sme.load
Wait for readiness: waitcnt
Synchronize threads: barrier
Perform matrix multiplication: matrix.mad
```

Connect `matrix.mad` back to TTIR's `tt.dot` to complete the path from Python to hardware-related operations.

### Part 5: finish with softmax

Emphasize fusion: intermediate results stay on chip. Softmax needs threads to cooperate on maxima and sums, but it does not need matrix compute units.

Ask students to summarize:

> Triton Python describes the computation; TTGIR assigns the work and places the data; LLIR turns that arrangement into low-level operations for the BI-V150.

## 10. Common teaching mistakes

- A warp has 64 threads in this environment; do not describe it as 32 threads.
- Do not interpret `iluvatar_mma` numbering as another device's version numbering.
- Extra i64 checks in TTIR are not evidence of an algorithmic optimization.
- `num_stages` describes buffering depth, not multiple independent execution streams.
- Focus LLIR reading on input preparation, waiting, synchronization, and `matrix.mad`.
- Validate the compilation target using the current V150 run's device records, BI LLIR, and final device binary.

## 11. Generate V150 examples with different configurations

Complete the directory setup in Section 2 first, then generate comparisons for pipeline depth, warp count, and tile shape:

```bash
for stages in 1 3 5; do
  python3 run_lab.py --kernel matmul --device-index 0 --num-stages "$stages" --dump-dir "$V150_RUN_DIR/stages_$stages"
done
for warps in 2 4 8; do
  python3 run_lab.py --kernel matmul --device-index 0 --num-warps "$warps" --dump-dir "$V150_RUN_DIR/warps_$warps"
done
python3 run_lab.py --kernel matmul --device-index 0 --block-m 32 --block-n 64 --block-k 32 --dump-dir "$V150_RUN_DIR/tile_32x64x32"
python3 run_lab.py --kernel matmul --device-index 0 --block-m 128 --block-n 64 --block-k 32 --dump-dir "$V150_RUN_DIR/tile_128x64x32"
```

If a configuration exceeds the backend's resource or support limits, keep its error output and complete the lab with the default configuration first.

To benchmark separately and export pass-by-pass logs:

```bash
python3 run_lab.py --kernel matmul --device-index 0 --benchmark --dump-dir "$V150_RUN_DIR/benchmark"
MLIR_ENABLE_DUMP=1 python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/mlir-dump" > "$V150_RUN_DIR/mlir-run.log" 2> "$V150_RUN_DIR/mlir-passes.log"
LLVM_IR_ENABLE_DUMP=1 python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/llvm-dump" > "$V150_RUN_DIR/llvm-run.log" 2> "$V150_RUN_DIR/llvm-passes.log"
```

Dump support depends on the adapted Triton build. `run_mlir_step.py` is a separate pass-replay tool that uses internal APIs, some of which differ from Triton 3.1. It is not a prerequisite for running this lab.

## 12. Optional: Iluvatar performance analysis tools

Exporting IR does not require a profiler. Use `ixsys` for runtime timelines, `ixkn-cli` for kernel performance metrics, and `ixsmi` for device status.

```bash
ixsys --help
ixkn-cli --help
ixkn-cli --list-sets
ixsys -t cuda -o "$V150_RUN_DIR/matmul.ptrace" python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/ixsys"
ixkn-cli --set default -o "$V150_RUN_DIR/matmul.ixkn" python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/ixkn"
```

Tool arguments, Python runtime requirements, and virtualized-device support depend on the installed platform version. Run the two profilers separately and locate `matmul_kernel` in their results. Do not attribute initialization, random-number generation, or reference computation time to that kernel. Use benchmarks without a profiler for timing comparisons.

