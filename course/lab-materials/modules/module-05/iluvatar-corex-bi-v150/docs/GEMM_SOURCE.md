# Integrated Lab1 GEMM

Source repository: https://gitee.com/sunxt-0719/lab1

Source revision: `6004a08ad6b2faa61b30ca06b7a2a9dfd44bd971`.

The unified Lab1 places its GEMM material in `lab/`, rather than a separate
`lab1-day2` repository. This lab now runs its local implementation directly;
it never dynamically imports an external `common.py`.

| Lab1 material | Integration in this lab |
|---|---|
| `lab/common.py` | The same masked GEMM calculation, fixed launch with preallocated output, and fp32-reference correctness checks in `kernels/gemm_triton.py` |
| `lab/06_baseline.py` | `--impl baseline`, using the configured tile and default four warps / two stages |
| `lab/07_block_sweep.py` | Candidate tiles supplied in `configs/gemm_*.json`; each candidate is checked before tuning |
| `lab/08_autotune.py` | `--impl autotune` uses native `triton.autotune`, keyed on M/N/K; selected tile, warps, and stages are recorded |
| `lab/09_summary.py` | Existing `--impl torch baseline autotune` comparison and `04_summary.py` retain this lab's LoadGen timing and JSON outputs |

The GEMM function bodies were compared structurally with the source kernel.
This integration retains optional CPU/PyTorch fallback and configurable dtype
from lab5. Tensor allocation remains in its SUT classes, so queries reuse output
buffers. The exported profiler kernel name remains `_matmul_kernel`.

Lab1 Step 08 intentionally leaves its autotune candidate lists as student TODOs.
This lab uses its own nonempty JSON candidate lists instead of copying that
unfinished executable script. Optional `num_warps` and `num_stages` in each
candidate become the corresponding `triton.Config` fields. The full resource,
tile, and scheduler sweeps are not added to the default lab5 workflow.

All candidate outputs are checked against an fp32 matmul rounded to output dtype,
with Lab1's fp16 tolerance (`atol=0.2`, `rtol=1e-2`) and a finite-value check.
The winning configuration is checked again. `01_benchmark.py` validates the
selected Triton SUT before warmup/measured queries; these checks and initial
autotuning are outside the reported latency samples. Invalid candidates fail
the run rather than being ranked solely by speed.

Triton SUT metadata contains `source`, `autotune_method`, `validation`, and
`candidate_validation`. Baseline and autotune are local Triton implementations;
`torch` remains the vendor-library comparison. An old external-path environment
variable has no effect on implementation selection.

This change does not alter the Server scheduling, Roofline model, or profiler
diagnosis logic. Their previously identified measurement limitations still apply.
