# Triton Operator Exercises on K3

This document is a separate guide for the optional Triton experiments under [`scripts/`](scripts/). It does not replace the main `llama.cpp` inference lab.

The goal is to help learners with no operator background build intuition first, then run the Triton scripts.

## 1. What learners should understand first

Do not start with "kernel", "compiler", "IR", or "backend".

Start with this idea:

```text
Input numbers go in.
A fixed rule changes them.
Output numbers come out.
That fixed rule is an operator.
```

Example:

```text
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]

rule: add numbers at the same position

out = [11, 22, 33, 44]
```

Then connect it to PyTorch:

```python
out = x + y
```

In PyTorch, this looks like one line. Under the hood, it is still an operator. Triton lets us write the inside of that operator ourselves.

## 2. Board setup

Run these commands on the K3 board.

> **Important:** The tested board runs Bianbu 4.0.4 with Python 3.14 as the system Python.  
> The SpacemiT PyTorch and Triton packages require **Python 3.12**.  
> `python3.12` is not available via `apt` on this board. We use `uv` to install a standalone CPython 3.12 and create the virtual environment.

### 2.1 Install `uv` (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

Expected output similar to:

```text
uv 0.13.0 (riscv64gc-unknown-linux-gnu)
```

### 2.2 Install system dependencies

The SpacemiT PyTorch wheel depends on `libsleef.so.3` and `libomp.so.5`.  
Install them before running any Python scripts:

```bash
sudo apt update
sudo apt install -y libsleef3 libomp5
sudo ldconfig
```

> **Note:** `ldconfig` may print a warning about `/usr/local/lib/libethercat.so.1` not being a symbolic link. This is harmless and does not affect the Triton lab.

### 2.3 Create a Python 3.12 virtual environment

Use `uv` to create and seed a venv with Python 3.12:

```bash
uv venv ~/.triton-venv --python 3.12 --seed
source ~/.triton-venv/bin/activate
python -V
```

Expected:

```text
Python 3.12.15
```

### 2.4 Install Python packages

```bash
python -m pip install -U pip

python -m pip install PyYAML sympy torch opencv-python pybind11 \
  --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple

python -m pip install triton \
  --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple
```

> **Note:** During `torch` installation you may see a warning about an inconsistent version (`2.8.0+spacemit.2` vs `2.8.0`).  
> The installer will automatically fall back to `torch-2.8.0+spacemit.0`, which works correctly.

### 2.5 Check Triton and the spine backend

```bash
python -c "import triton; print(triton.__version__)"
python -c "import triton.backends.spine_triton; print('spine backend OK')"
```

Expected:

```text
3.6.0
spine backend OK
```

Known working versions from the tested board:

```text
Python 3.12.15
triton 3.6.0+spacemit.b0
torch 2.8.0+spacemit.0
```

## 3. Script location

Clone the current course repository if you have not already:

```bash
git clone https://gitlink.org.cn/flagos-ai/OpenCourse.git
cd OpenCourse/labs/special-topic-labs/edge-device/k3-board
```

The scripts are in:

```bash
scripts/
```

Enter the directory:

```bash
cd scripts
```

Use a dedicated Triton cache while testing:

```bash
export TRITON_CACHE_DIR=/tmp/lab-triton-cache
```

The first run may print C/C++ compiler warnings and take longer because Triton JIT-compiles the kernel. Warnings about `_POSIX_C_SOURCE` or `_XOPEN_SOURCE` being redefined were observed on the tested board and did not prevent the scripts from passing.

## 4. Recommended experiment order

### Experiment 0: Installation check

Run:

```bash
python verify_install.py
```

Expected:

```text
PASS: spine-triton installation check completed
```

Teaching point:

```text
This only checks whether Triton can compile and run a tiny operator on this board.
It is not the main operator lesson.
```

### Experiment 1: First operator, vector addition

Run:

```bash
python vec_add.py
```

Expected:

```text
PASS: n=98432, BLOCK_SIZE=1024, max_error=0.0
```

Teaching point:

```text
This operator reads x and y, adds matching positions, and writes out.
```

Use this simple explanation:

```text
PyTorch: out = x + y
Triton: load x, load y, add them, store out
```

Only after that, introduce these words:

```text
program    = one worker group
BLOCK_SIZE = how many numbers one group handles
mask       = keeps the last group from reading or writing past the real data
```

### Experiment 2: Same operator, different block sizes

Run:

```bash
python vec_add_bs256.py
python vec_add_bs1024.py
python vec_add_bs4096.py
```

Expected:

```text
PASS: n=98432, BLOCK_SIZE=256, max_error=0.0
PASS: n=98432, BLOCK_SIZE=1024, max_error=0.0
PASS: n=98432, BLOCK_SIZE=4096, max_error=0.0
```

Teaching point:

```text
The math is the same, but the work is grouped differently.
```

Comparison:

```text
BLOCK_SIZE=256   more groups, smaller work per group
BLOCK_SIZE=1024  medium grouping
BLOCK_SIZE=4096  fewer groups, larger work per group
```

This is the first performance intuition:

```text
An operator is not only the formula. It also has an implementation strategy.
```

### Experiment 3: Softmax, a real model operator

Run:

```bash
python softmax_demo.py
```

Expected:

```text
PASS: fp32 softmax, shape=(128, 256), dtype=torch.float32, max_error=...
PASS: fp16 softmax, shape=(128, 256), dtype=torch.float16, max_error=...
```

The exact error value may differ, but the script should pass.

Teaching point:

```text
Softmax turns scores into probabilities.
```

Example:

```text
raw scores:
cat: 8
dog: 3
car: 1

after softmax:
cat: high probability
dog: lower probability
car: very low probability
```

Contrast it with vector addition:

```text
vec_add:
  each output depends only on one x value and one y value

softmax:
  each output depends on the whole row, because the row max and row sum are needed
```

This is the key operator comparison:

```text
Different operators have different data dependencies.
```

## 5. Suggested teaching flow

Use this order for beginners:

```text
1. Show input numbers, output numbers, and a fixed rule.
2. Name that fixed rule an operator.
3. Show that PyTorch hides the operator behind one line.
4. Show that Triton lets us write that operator ourselves.
5. Run vector addition.
6. Change block size and compare grouping.
7. Run Softmax and connect it to model probabilities.
8. Return to llama.cpp and explain that full inference is many operators chained together.
```

Avoid these topics in the first pass:

```text
MLIR
LLVM
TCM internals
assembly
full performance tuning
backend implementation details
```

## 6. Troubleshooting

| Symptom | Solution |
|---------|----------|
| `python3.12: command not found` | Use `uv` to create a Python 3.12 venv as shown in section 2.3. |
| `error: externally-managed-environment` when running `pip install` | You are using the system Python. Activate the `uv`-created venv first. |
| `ImportError: libsleef.so.3` | `sudo apt install -y libsleef3 && sudo ldconfig` |
| `ImportError: libomp.so.5` | `sudo apt install -y libomp5 && sudo ldconfig` |
| `torch` cannot be found on the SpacemiT index | Make sure you are using Python 3.12. Python 3.14 is not supported by the SpacemiT torch wheel. |
| Triton cache permission issues | Set `export TRITON_CACHE_DIR=/tmp/lab-triton-cache` before running scripts. |

## 7. Shutdown after experiments

Stop any running program with `Ctrl+C` first.

Then shut down cleanly:

```bash
sync
poweroff
```

Wait until SSH or serial disconnects and the board is stable before removing power.
