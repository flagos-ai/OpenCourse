# Integrated GEMM implementation

Module 5 includes its GEMM implementation locally in
[`../kernels/gemm_triton.py`](../kernels/gemm_triton.py). No separate lab
repository, sibling checkout, or environment variable is required.

The lab provides two local Triton implementations:

- `baseline`: a fixed configuration using the configured tile, warps, and stages.
- `autotune`: `triton.autotune` candidates from the JSON configuration files.

Each candidate is validated against an fp32 PyTorch reference before it is
ranked. The selected configuration is validated again before benchmark timing.
These correctness checks and any first-time compilation are outside the
reported LoadGen measurements.
