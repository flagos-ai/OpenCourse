"""Roofline model computation and plotting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def gemm_flops(M: int, N: int, K: int) -> int:
    return 2 * M * N * K


def gemm_bytes(M: int, N: int, K: int, dtype_bytes: int) -> int:
    return (M * K + K * N + M * N) * dtype_bytes


def arithmetic_intensity(flops: int, nbytes: int) -> float:
    if nbytes <= 0:
        return float("nan")
    return flops / nbytes


def ridge_point(peak_tflops: float, bandwidth_GBps: float) -> float:
    """Intensity where compute roof meets memory roof (FLOP/Byte)."""
    peak_flops = peak_tflops * 1e12
    bandwidth_Bps = bandwidth_GBps * 1e9
    if bandwidth_Bps <= 0:
        return float("nan")
    return peak_flops / bandwidth_Bps


def attainable_tflops(intensity: float, peak_tflops: float, bandwidth_GBps: float) -> float:
    mem_roof = (bandwidth_GBps * 1e9 * intensity) / 1e12
    return min(peak_tflops, mem_roof)


def diagnose(
    intensity: float,
    measured_tflops: float,
    peak_tflops: float,
    bandwidth_GBps: float,
    efficiency_threshold: float = 0.7,
) -> Dict[str, Any]:
    if not all(math.isfinite(x) and x > 0 for x in (intensity, measured_tflops, peak_tflops, bandwidth_GBps)):
        raise ValueError("Roofline inputs must be finite and positive")
    ridge = ridge_point(peak_tflops, bandwidth_GBps)
    attainable = attainable_tflops(intensity, peak_tflops, bandwidth_GBps)
    bound = "compute-side" if intensity >= ridge else "memory-side"
    below = measured_tflops < attainable * efficiency_threshold
    labels = [bound]
    if below:
        labels.append("below-roofline")
    suffix = (
        ", well below the roof (other overheads or under-utilized hardware)."
        if below
        else "."
    )
    conclusion = (
        f"Model region: {bound}, arithmetic intensity I={intensity:.2f} FLOP/Byte "
        f"vs ridge I_ridge={ridge:.2f} FLOP/Byte; measured {measured_tflops:.3f} TFLOP/s, "
        f"reference roof ~{attainable:.3f} TFLOP/s{suffix} "
        "Actual bottleneck is undetermined; measured counters are required."
    )
    return {
        "bound": bound,  # Compatibility alias for model_region; not an actual bottleneck.
        "model_region": bound,
        "actual_bottleneck": "undetermined",
        "below_roofline": below,
        "labels": labels,
        "ridge_flop_per_byte": ridge,
        "attainable_tflops": attainable,
        "measured_tflops": measured_tflops,
        "arithmetic_intensity": intensity,
        "conclusion": conclusion,
    }


def plot_roofline(
    points: List[Dict[str, Any]],
    peak_tflops: float,
    bandwidth_GBps: float,
    out_path: Path,
    title: str = "Roofline - GEMM on BI-V150",
) -> Optional[Path]:
    """Write PNG if matplotlib is available; otherwise print a warning and return None."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "[warn] matplotlib not installed — skipping PNG. "
            "JSON diagnosis is still written. On FlagOS: pip install matplotlib",
            flush=True,
        )
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ridge = ridge_point(peak_tflops, bandwidth_GBps)

    i_min = 0.1
    i_max = max([p["intensity"] for p in points] + [ridge * 4, 100.0])
    xs = np.logspace(np.log10(i_min), np.log10(i_max), 200)
    ys = [attainable_tflops(float(i), peak_tflops, bandwidth_GBps) for i in xs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(xs, ys, label="Reference roof (verify peak inputs)", linewidth=2)
    ax.axvline(ridge, color="gray", linestyle="--", label=f"ridge={ridge:.1f}")

    for p in points:
        ax.scatter(p["intensity"], p["tflops"], s=80, zorder=5, label=p.get("name", "point"))
        ax.annotate(
            p.get("name", ""),
            (p["intensity"], p["tflops"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (TFLOP/s)")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def dtype_nbytes(dtype: str) -> int:
    key = dtype.lower().replace("torch.", "")
    if key in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if key in ("float32", "fp32"):
        return 4
    if key in ("float64", "fp64"):
        return 8
    raise ValueError(f"unknown dtype: {dtype}")
