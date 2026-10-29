# AI Systems Software Fundamentals and Heterogeneous Computing: BI-V150 Triton Lab

This project provides an introductory Triton lab for the Iluvatar BI-V150 / CoreX
platform. It progresses from device execution, Vector Add, Softmax, LayerNorm,
and tiled GEMM to GEMM benchmarking, block sweeps, scheduler sweeps, and autotuning.

The lab follows a single sequence numbered `00` through `09`, with two programming
tasks: implement the Vector Add kernel and select autotune configurations based
on the sweep results.

## Repository Layout

```text
lab1/
├── README.md                 # Project overview
└── lab/
    ├── README.md             # Complete lab guide and reference solutions
    ├── setup.sh              # Environment checks
    ├── 00_check_env.py
    ├── 01_torch_device_hello.py
    ├── 02_vector_add_triton.py
    ├── 03_softmax_compare.py
    ├── 04_layernorm_compare.py
    ├── 05_matmul_compare.py
    ├── 06_baseline.py
    ├── 07_block_sweep.py
    ├── 08_autotune.py
    ├── 09_summary.py
    ├── common.py             # GEMM, validation, and timing helpers
    └── __init__.py
```

## Getting Started

Use the official BI-V150 platform image with matching CoreX drivers, runtime,
PyTorch, and Triton. Do not install or upgrade upstream `torch` or `triton` packages
with `pip`, as they may overwrite the platform-specific builds.

```bash
cd labs/module-01/lab
bash setup.sh
python3 00_check_env.py
```

See the [complete lab guide](lab/README.md) for the remaining steps, task
instructions, result interpretation, and reference solutions.
