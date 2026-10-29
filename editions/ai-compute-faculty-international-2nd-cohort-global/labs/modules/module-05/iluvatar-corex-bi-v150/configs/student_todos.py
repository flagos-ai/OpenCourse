"""Completed implementation defaults and Roofline parameter exercise.

Offline and Server already compare torch, baseline, and autotune.
Only the Roofline peak verification remains inside YOUR CODE HERE markers.
"""

from __future__ import annotations

# Offline / Server SUT list (used by 01_benchmark.py).
# JSON configs/gemm_*.json should stay in sync as a readable mirror.
OFFLINE_IMPLEMENTATIONS = ["torch", "baseline", "autotune"]
SERVER_IMPLEMENTATIONS = ["torch", "baseline", "autotune"]

# BI-V150 Roofline peaks (must match configs/hardware_bi_v150.json).
#
# >>> YOUR CODE HERE >>>
# TODO: look up BI-V150 FP16 peak (TFLOP/s) and HBM bandwidth (GB/s),
# then set the values below AND copy them into hardware_bi_v150.json.
PEAK_TFLOPS_FP16 = 192.0
HBM_BANDWIDTH_GBPS = 1228.8
# <<< END OF YOUR CODE <<<
