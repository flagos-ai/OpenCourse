# Fused Add + RMSNorm Lab

This lab follows one operator through the PyTorch stack:

1. Complete the TODO in the fused Triton forward kernel.
2. Review the provided backward kernel and validate its gradients.
3. Review the provided PyTorch dispatcher registration.
4. Verify that torch.compile captures the registered operator without a graph break.
5. Compare the fused implementation with PyTorch, a split Triton baseline, and FlagGems.

All lab files are in the repository root. Complete only the marked TODO in
`fused_rms_norm.py`; the other Python files are already provided. The two-line
solution is in the **Reference Answer** section at the end of this README.

## Files

| File | Purpose |
| --- | --- |
| `fused_rms_norm.py` | Forward TODO, PyTorch reference, split Triton baseline, and forward checks |
| `grad_kernel.py` | Provided backward kernel, autograd wrapper, and gradient checks |
| `custom_op.py` | Provided CUDA, AutogradCUDA, and Meta registrations |
| `compile_demo.py` | Eager and compiled forward/backward checks and Dynamo graph summary |
| `benchmark.py` | Performance comparison with PyTorch, split Triton, and installed FlagGems |

## Operator Contract

Use contiguous tensors on the same CUDA-compatible device, with the same dtype:
`x` and `residual` of shape `[M, N]`, and `weight` of shape `[N]`.
The mean is taken over the N columns of each row:

```text
z = x + residual
rrms = 1 / sqrt(mean(z * z) + eps)
out = z * rrms * weight
return out, z
```

The forward checks cover float32 and bfloat16. The fused kernel converts loaded
values to float32 before addition, reduction, and normalization, then stores
both outputs in the input dtype. The required gradient and compile checks below
use float32.

## Exercise Order

- TODO, `fused_rms_norm.py`: `variance` holds the mean square, not the
  mean-centered variance. Compute `rrms = 1 / sqrt(variance + eps)` and use it
  with `updated` and `weight` to produce
  `out`. Fill only the region marked `TODO BEGIN` and `TODO END`.

The backward kernel, PyTorch operator registration, compile demonstration, and
benchmark are complete and can be run directly after this TODO is filled.

`benchmark.py` is intentionally complete and must use the installed FlagGems
implementation rather than a local fallback. It checks both fused outputs
against the PyTorch reference before comparing performance.

## Validation Commands

Use the prepared lab device with PyTorch, Triton, and FlagGems installed.
From the directory containing this checkout, enter the repository root.
If the checkout has a different name, use that
directory name instead:

```bash
cd completed_lab2

python3 fused_rms_norm.py
python3 grad_kernel.py --m 128 --n 256 --dtype float32
python3 compile_demo.py --m 128 --n 256 --backend eager
python3 benchmark.py --m 4096 --n 4096 --dtype float32
python3 benchmark.py --m 4096 --n 4096 --dtype bfloat16
```

Expected results and targets:

- `fused_rms_norm.py` prints `out diff`, `res diff`, and `fused ms` for
  `(512, 256)`, `(1024, 512)`, and `(4096, 4096)` in both dtypes. Check the
  float32 errors against the target of less than `2e-5`; this script prints
  errors without asserting a tolerance or printing PASS/FAIL.
- `grad_kernel.py` checks both outputs and the gradients of `x`, `residual`,
  and `weight`. For the supplied `(128, 256)` float32 case, all maximum absolute
  errors must be at most `2e-5` to print PASS. An optional bfloat16 run uses
  a separate tolerance of `2e-1`.
- `compile_demo.py` prints eager and compiled errors and a Dynamo graph
  summary. Check for `graphs=1` and `graph_breaks=0`. PASS requires zero graph
  breaks and all output/gradient errors at most `2e-5`; the script does not
  separately assert the graph count.
- `benchmark.py` checks the local fused and FlagGems outputs and verifies that
  both implementations update the supplied buffers in place. Float32 output
  checks use `atol=2e-5, rtol=0`; bfloat16 uses `atol=0.0625, rtol=0.015625`.
  Updated residuals must match exactly. It prints timings, estimated GB/s,
  numerical differences, and the fused/FlagGems throughput ratio. The target
  is at least 90 percent at `(4096, 4096)` for each dtype. Numerical failures or
  a throughput ratio below 90 percent cause a nonzero exit status.

The compile demo defaults to `backend="eager"` with `fullgraph=True`. This checks
Dynamo graph capture and numerical correctness; it does not demonstrate
Inductor code generation or a compilation speedup. Inductor support depends on
the installed software versions and device backend and is not required here.

## Benchmark Conditions

The primary fused comparison uses matching storage and input conditions:

- Both fused implementations update the same preallocated `x` and `residual`
  buffers in place. The benchmark passes these buffers as output pointers to
  the student's existing kernel; no change to the forward exercise is needed.
- Every call, including warmup, starts with identical input values. Two
  `copy_` operations restore the inputs before the timing events, so restoring
  inputs is excluded from the reported latency. Neither fused path allocates
  output tensors in the measured call.
- Calls are interleaved in a reproducible randomized order. CUDA events measure
  device intervals on the same stream, with 25 warmup calls per implementation,
  100 samples per repeat, and 5 repeats by default. The reported latency is the
  median of the repeat means; the range is printed as well. Use `--repeats` to
  change the repeat count.

Input restoration warms the working buffers. These are device event timings
under that shared preparation, not complete application latency or cold-cache
measurements. Device intervals can include idle time while waiting for host
submission; host allocation costs are not measured independently. Estimated
GB/s uses a shared logical byte count, not hardware counters.

The PyTorch stepwise implementation is an allocating baseline. The split
Triton baseline uses the same in-place buffers but launches two kernels. For
bfloat16 it stores the intermediate sum in bfloat16 before normalization,
whereas both fused kernels keep that sum in float32 internally. These baselines
provide context; the 90 percent target applies to the matched fused comparison.

## Reference Answer

Fill the TODO in `_fused_add_rms_norm_kernel` with:

```python
rrms = 1.0 / tl.sqrt(variance + eps)
out = updated * rrms * weight
```

The first line computes the reciprocal RMS scaling factor. The second line
normalizes the updated residual and applies the learnable RMSNorm weight.
