# Lab Day 1: Getting Started with Ascend NPU and Triton

**Course:** Software and Hardware Foundations of Intelligent Computing Systems
**Duration:** 2 hours
**Audience:** Faculty members (instructors-in-training)

---

## Learning Goals

By the end of this lab, you will be able to:

1. Explain the Ascend software stack and how it maps to the NVIDIA CUDA stack you may already know.
2. Run a basic PyTorch program on an Ascend NPU using `torch_npu`.
3. Write and execute a Triton kernel on Ascend hardware.
4. Compare the performance of a Triton kernel against the vendor library (`torch_npu`) on the same NPU.
5. Know where to find documentation and examples to continue exploring on your own.

---

## Two Narratives We Will Demonstrate

This lab is built around **two independent claims** about Triton on Ascend:

**Narrative 1 -- Triton has real value on the NPU.**
Inside the same Ascend NPU, a small Python Triton kernel can be a serious
alternative to the vendor's hand-tuned CANN library kernels. Sometimes it
even wins. We show this by benchmarking your Triton kernel against
`torch.softmax` / `torch.matmul`, both running on the NPU.

**Narrative 2 -- Triton source code is portable across NVIDIA and Ascend.**
The same `@triton.jit` kernel runs on both backends, byte-for-byte. We
show this by comparing the source you write today with the OpenAI Triton
tutorial source (and, if available, a screenshot of it running on an
NVIDIA GPU). The portability claim is *source-level*; we are NOT
benchmarking NPU vs GPU here.

> **About today's hardware:** Your container has an Ascend 910C only --
> there is no GPU inside. So when you see "Triton vs torch_npu" benchmarks,
> *both* run on the NPU.

---

## 0. Today's Hardware and Software (5 min)

| Component   | Version          | Analogy in NVIDIA world |
|-------------|------------------|-------------------------|
| Hardware    | Ascend 910C (2 dies) | NVIDIA H-class GPU |
| Driver / Firmware | npu-smi 25.5.0 | NVIDIA driver |
| CANN Toolkit | 8.3.RC2          | CUDA Toolkit + cuDNN + NCCL |
| PyTorch     | 2.8.0 (CPU build) | PyTorch (CUDA build) |
| torch_npu   | 2.8.0            | (built into PyTorch for CUDA) |
| Triton-Ascend | 3.2.0          | OpenAI Triton (CUDA backend) |

**Key idea:** PyTorch on Ascend uses a *CPU-build* of PyTorch plus a
separate `torch_npu` plugin. You will see `torch.__version__ == '2.8.0+cpu'`
-- this is normal.

---

## 1. The Ascend Software Stack (15 min, lecture)

```
+--------------------------------------------------+
|   User code: PyTorch / MindSpore / Triton-Ascend |   <-- you write here
+--------------------------------------------------+
|   CANN: runtime, compiler, opp, hccl, aoe        |   <-- like CUDA Toolkit
+--------------------------------------------------+
|   Driver (HDK) + firmware                        |
+--------------------------------------------------+
|   Ascend 910C NPU (Cube Core + Vector Core + HBM)|
+--------------------------------------------------+
```

| Ascend term       | NVIDIA equivalent      |
|-------------------|------------------------|
| NPU               | GPU                    |
| AI Core (Cube + Vector) | SM (streaming multiprocessor) |
| HBM               | HBM (same!)            |
| CANN              | CUDA Toolkit + libraries |
| Ascend C          | CUDA C++               |
| `torch_npu`       | (built into torch)     |
| HCCL              | NCCL                   |
| `npu-smi`         | `nvidia-smi`           |

**Why Triton matters here.** Writing Ascend C kernels directly is powerful
but has a steep learning curve. Triton-Ascend lets you write a
GPU-style Python DSL and run it on Ascend with **the same source code**
you would use on NVIDIA GPUs. That is the central promise we will
demonstrate today.

---

## 2. Environment Check (10 min, hands-on)

Open a terminal in your container.

**Important:** every time you open a new terminal, run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Get the lab code:

```bash
cd ~
git clone https://gitee.com/<YOUR_USERNAME>/cann-triton-lab-day1.git
cd cann-triton-lab-day1
source setup.sh
```

`setup.sh` sources CANN environment variables and prints version info.
You should see all components listed and `NPU device count: 2`.

Then run the environment check:

```bash
python 00_check_env.py
```

If everything prints OK, you are ready.

> **Discussion question:** the output shows 2 NPU devices, but physically
> this is *one* Ascend 910C card. Why?
>
> *Answer:* 910C is a dual-die package; each die exposes itself as a
> logical NPU.

---

## 3. PyTorch on NPU: Hello World (15 min, hands-on)

Open `01_torch_npu_hello.py` in your editor. Find the TODO block in
`matmul_on_device`. It shows you the CUDA version and asks you to port
four lines to NPU.

After filling it in, run:

```bash
python 01_torch_npu_hello.py
```

You should see a matmul timing and "Output shape and dtype correct."

> **What to notice:**
> - The only difference from CUDA code is `'npu'` instead of `'cuda'`.
> - `import torch_npu` at the top is required *once* -- it registers the
>   NPU backend with PyTorch.

> **Try it yourself:** modify the script to compute `D = (A @ B) + bias`
> where `bias` is a length-N random vector, and verify against CPU.

---

## 4. Your First Triton Kernel on NPU (30 min, hands-on)

Open `02_vector_add_triton.py`. The kernel `add_kernel` has a TODO with
five sub-tasks. Each task corresponds to one line you need to write.
The comments above the TODO list the exact API for each step.

Run:

```bash
python 02_vector_add_triton.py
```

If your kernel is correct you will see `PASS` and `Max absolute difference: 0.00e+00`.

> **What to notice:**
> - The `@triton.jit` decorator and the kernel body are *identical* to
>   what you would write for an NVIDIA GPU.
> - The first run is slow (30s-2min). That is JIT compilation. Triton
>   compiles your Python kernel down to NPU machine code and caches it
>   in `~/.triton/cache/`. The second run is fast.

> **Discussion questions:**
> - What does `tl.program_id(axis=0)` correspond to in CUDA terminology?
>   *(blockIdx.x.)*
> - What happens if you change `BLOCK_SIZE` from 1024 to 512?
>   *(Recompilation, because BLOCK_SIZE is a `tl.constexpr`.)*

> **Portability moment.** The kernel you just wrote is the same one in
> the OpenAI Triton tutorial -- which targets NVIDIA GPUs. Same source,
> different backend. *(See projector for side-by-side screenshot.)*

---

## 5. Triton vs torch_npu: Performance on the Same NPU (40 min, hands-on)

This is the centerpiece. We compare your hand-written Triton kernels
against the optimized CANN library operators that `torch_npu` dispatches
to. **Both run on the same Ascend NPU.**

### 5a. Softmax

Open `03_softmax_compare.py`. Two TODOs in the kernel: load the row, and
implement numerically stable softmax (`max -> subtract -> exp -> sum -> divide`).

Run:

```bash
python 03_softmax_compare.py
```

You will see a table of timings. The `torch_npu` column calls the CANN
library kernel; the `Triton` column calls yours.

> **Discussion:**
> - Why might Triton beat `torch_npu` on softmax for some shapes?
>   *(One pass over memory; no intermediate tensors.)*
> - Why is the comparison fair?
>   *(Same hardware, same dtype, same problem size.)*

### 5b. Matrix Multiplication

Open `04_matmul_compare.py`. Two TODOs: initialize the accumulator tile,
and write the K-loop body that loads tiles of A and B and accumulates
their product.

Run:

```bash
python 04_matmul_compare.py
```

> **Discussion:**
> - Where does Triton's performance come from? *(Tile-based memory
>   access, automatic pipelining inserted by the compiler.)*
> - Where does `torch.matmul` likely win? *(It uses CANN's hand-tuned
>   Cube Core kernels.)*
> - Takeaway: **Triton's value is highest for custom ops** that have no
>   library version. For standard ops the vendor library is hard to beat.

> **This file is also the baseline for tomorrow's lab,** where we will
> progressively optimize this matmul for Ascend (block size sweep,
> autotune, multi-buffering).

---

## 6. Wrap-up and Resources (10 min)

**What we have demonstrated:**
- Inside the NPU: a 30-line Python Triton kernel can compete with the
  vendor library on real workloads.
- Across hardware: the same Triton source runs on both NVIDIA and Ascend.
- The ecosystem is mature enough to support real research and teaching.

**Resources:**
- Triton-Ascend (Gitee): https://gitee.com/ascend/triton-ascend
- Triton-Ascend (GitHub mirror): https://github.com/Ascend/triton-ascend
- CANN docs: https://www.hiascend.com/document
- Ascend community: https://www.hiascend.com/forum
- Today's lab repo (forkable): https://gitee.com/<YOUR_USERNAME>/cann-triton-lab-day1
- OpenAI Triton tutorials (the kernels here are based on these):
  https://triton-lang.org/main/getting-started/tutorials/

**Tomorrow:** progressively tune the matmul kernel for Ascend.

---

## Quick Reference: Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: ... NPU not found` | Forgot `set_env.sh` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `ImportError: torch_npu` | Wrong PyTorch build | Check `pip list \| grep torch` |
| First Triton run hangs | JIT compilation | Wait 30s-2min |
| `Device do not support double dtype` | NPU has no fp64 | Use `dtype=torch.float32` |
| Slow `pip install` | Default pypi.org from China | Use Tsinghua mirror |

---

# Reference Solutions

> If you got everything working without looking at this section,
> congratulations -- skip to the Try-it-yourself prompts in each script.

## Solution: `01_torch_npu_hello.py`

```python
def matmul_on_device(M, K, N):
    device = "npu" if torch.npu.is_available() else "cpu"
    A = torch.randn(M, K, dtype=torch.float32, device=device)
    B = torch.randn(K, N, dtype=torch.float32, device=device)
    _ = A @ B
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        C = A @ B
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) / 50 * 1000
    return C, elapsed_ms
```

The literal change from CUDA: every `cuda` -> `npu`. That is it.

## Solution: `02_vector_add_triton.py`

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Notes:
- `tl.arange(0, BLOCK_SIZE)` produces a **vector** of offsets, not a single
  scalar. The kernel is *block-vectorized* by construction.
- The `mask` is essential. Without it, the last block (when `n_elements`
  is not divisible by `BLOCK_SIZE`) would read/write past the buffer.

## Solution: `03_softmax_compare.py`

```python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride,
                   n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # (1) load
    row_start_ptr = input_ptr + row_idx * input_row_stride
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=mask, other=-float("inf"))

    # (2) numerically stable softmax
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

Notes:
- `other=-float("inf")` matters: it makes masked-out positions invisible
  to `tl.max`. With `other=0` or `other=-1e30` you would get subtly wrong
  results when the row contains negative numbers near the chosen sentinel.
- The whole softmax is fused into one kernel: one read of the row, one
  write of the output. The PyTorch op typically does max -> subtract ->
  exp -> sum -> divide as separate ops, with intermediate tensors.

## Solution: `04_matmul_compare.py`

```python
# (1) accumulator
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# (2) K loop body
for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_remaining = K - k * BLOCK_K
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
    b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

Notes:
- The accumulator stays in registers across the K loop -- it is never
  written back to HBM until the final `tl.store`. This is the key
  performance trick of tile-based matmul.
- `other=0.0` for the masked load lets us pretend the partial K tile is
  full of zeros; zeros do not affect the dot product.
- `tl.dot(a, b)` performs a tile-level matmul. On Ascend, it is mapped
  by the Triton-Ascend compiler to Cube Core instructions when the
  shapes and dtypes are compatible.

## Reasoning answers (for the matmul Try-it-yourself)

**Why might increasing `BLOCK_K` help arithmetic intensity?**
With larger `BLOCK_K`, each iteration of the K loop does
`BLOCK_M * BLOCK_N * BLOCK_K * 2` FLOPs while loading
`(BLOCK_M + BLOCK_N) * BLOCK_K` floats. The ratio of FLOPs to bytes
loaded grows with `BLOCK_K`, so we get better arithmetic intensity and
make better use of the Cube Core.

**Why might it stop helping past some point?**
- Tiles must fit in on-chip storage (Unified Buffer / L1 buffer). Past
  some size, increasing `BLOCK_K` causes spilling.
- Larger tiles also reduce parallelism: fewer programs in the launch
  grid means worse occupancy of AI cores.
- These two effects bound `BLOCK_K` from above, giving a sweet spot
  around 32-64 for fp32 on 910C in this simple kernel.
