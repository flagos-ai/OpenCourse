# CANN-Triton Lab Day 1

Hands-on lab for the international training course
**"Software and Hardware Foundations of Intelligent Computing Systems"**.

This lab introduces the Ascend NPU software stack through PyTorch and Triton.
**You will write code yourself** -- each script has small TODO blocks for you
to fill in. Reference solutions are at the end of `handout.md` if you get stuck.

---

## Quick Start

In your Docker container, every time you open a new terminal:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

Then:

```bash
git clone https://gitee.com/jieran-zhang/lab_day1.git
cd lab_day1
source setup.sh        # also sources set_env.sh and prints versions
python 00_check_env.py # verify everything works
```

If `00_check_env.py` prints "All checks passed", you are ready.

---

## How This Lab Works

Each numbered script has a small section marked with `TODO` and
`>>> YOUR CODE HERE >>>` / `<<< END OF YOUR CODE <<<` markers.

1. Read the comments above the TODO -- they describe what each line should do.
2. Write the missing code.
3. Run the script. The script prints PASS / FAIL or a results table.
4. If you get stuck, check the **Reference Solutions** section in `handout.md`.
5. Then move to the **Try-it-yourself** prompts at the bottom of each script.

---

## Lab Order

| File                       | What you'll do                                          |
|----------------------------|---------------------------------------------------------|
| `00_check_env.py`          | (No coding.) Verify the environment.                    |
| `01_torch_npu_hello.py`    | Port a CUDA-style snippet to NPU (4 lines).             |
| `02_vector_add_triton.py`  | Write a Triton vector-add kernel (5-6 lines).           |
| `03_softmax_compare.py`    | Implement numerically stable softmax in Triton.         |
| `04_matmul_compare.py`     | Fill in the K-loop accumulator of a tiled matmul.       |

For the full handout (lecture notes, exercises, reference solutions), see
[`handout.md`](./handout.md).

---

## Environment

This lab assumes a Docker container with:

| Component       | Version           |
|-----------------|-------------------|
| Hardware        | Ascend 910C (1 card, 2 dies, exposed as 2 NPUs) |
| CPU architecture | aarch64 (e.g. Kunpeng / Phytium) |
| CANN            | 8.3.RC2           |
| PyTorch         | 2.8.0 (CPU build) |
| torch_npu       | 2.8.0             |
| Triton-Ascend   | 3.2.0             |
| Python          | 3.11              |

The BAAI-provided container comes with CANN, PyTorch, and `torch_npu`
already installed. Only `triton-ascend` was added on top (see "Setup
History" below).

---

## Setup History (How This Container Was Configured)

This section records the actual commands used to set up the lab
environment, in case you need to reproduce it elsewhere or troubleshoot.

### Step 0: Reconnaissance (verify what is already installed)

```bash
# CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg

# NPU status and driver version
npu-smi info

# PyTorch and torch_npu versions
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"

# Current pip index
pip config list
```

The container already had:
- CANN 8.3.RC2 at `/usr/local/Ascend/ascend-toolkit/`
- npu-smi 25.5.0, one Ascend 910C card exposed as 2 logical NPUs
- PyTorch 2.8.0 (CPU build) + torch_npu 2.8.0
- `pip` pointing to default `pypi.org`

### Step 1: Verify PyTorch on NPU works

Before installing anything, confirm the existing PyTorch + NPU stack
already runs. This guarantees a working fallback even if `triton-ascend`
installation runs into trouble.

```bash
python -c "
import torch, torch_npu
print('NPU available:', torch.npu.is_available())
print('Device count:', torch.npu.device_count())
x = torch.randn(3, 3).npu()
y = torch.randn(3, 3).npu()
print((x @ y).cpu())
"
```

Expected: `NPU available: True`, `Device count: 2`, and a 3x3 result tensor.

### Step 2: Switch pip to a domestic mirror (required from inside China)

The default `pypi.org` is too slow for a 50 MB wheel from inside China.
Switch to the Tsinghua mirror first:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
```

This writes `/root/.config/pip/pip.conf`. Download speed went from
near-zero to ~7 MB/s after this change.

### Step 3: Install triton-ascend

```bash
pip install triton-ascend
```

This installs `triton-ascend-3.2.0` (wheel:
`triton_ascend-3.2.0-cp311-cp311-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`,
50.7 MB). No source compilation; no LLVM download.

### Step 4: Source CANN environment variables

Every new terminal must source `set_env.sh` before using the NPU.
Sourcing `setup.sh` from this repo also takes care of this.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### Step 5: End-to-end verification

```bash
python -c "import torch, torch_npu, triton; print('all ok')"
```

If this prints `all ok`, the environment is ready. The first Triton
kernel launch (e.g. running `02_vector_add_triton.py`) will JIT-compile
and take 30s - 2 min; subsequent launches use the cache in
`~/.triton/cache/` and are fast.

### From-scratch one-shot script

If you start from a clean container that already has CANN and torch_npu,
the entire setup is just:

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip install triton-ascend
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu, triton; print('all ok')"
```

### Notes on version compatibility

- Triton-Ascend 3.2 officially targets CANN 8.5; this container has
  CANN 8.3.RC2. Tested working in practice for our lab kernels.
- Triton-Ascend 3.2 documentation lists torch_npu 2.6.0; this container
  has torch_npu 2.8.0. Tested working in practice.
- Triton-Ascend requires Python 3.9-3.11. This container has Python 3.11.13.
- The wheel is `aarch64`, matching the ARM-based server (Kunpeng / Phytium
  + Ascend is a common pairing).

---

## Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: ... NPU not found` | Forgot to source `set_env.sh` in this terminal | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Triton's first run takes ~1 minute | JIT compilation | Wait. Subsequent runs use `~/.triton/cache/` |
| `Device do not support double dtype` warning | NPU does not support fp64 | Use `dtype=torch.float32` explicitly |
| `Cannot create tensor with internal format` warning | torch_npu fallback to base format | Harmless; does not affect correctness |
| Slow `pip install` from `pypi.org` | International route | Use Tsinghua mirror (`-i https://pypi.tuna.tsinghua.edu.cn/simple`) |

---

## Resources

- Triton-Ascend (Gitee mirror): https://gitee.com/ascend/triton-ascend
- Triton-Ascend (GitHub mirror): https://github.com/Ascend/triton-ascend
- CANN documentation: https://www.hiascend.com/document
- OpenAI Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

---

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.