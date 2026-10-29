"""Latency / throughput / resource metric helpers."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def summarize_latencies_ms(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    vals = list(latencies_ms)
    return {
        "count": len(vals),
        "mean": float(statistics.fmean(vals)),
        "median": float(statistics.median(vals)),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def effective_tflops(flops: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return float("nan")
    return (flops / (latency_ms * 1e-3)) / 1e12


def samples_per_second(num_queries: int, wall_s: float) -> float:
    if wall_s <= 0:
        return float("nan")
    return num_queries / wall_s


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    if bytes_moved <= 0:
        return float("nan")
    return flops / bytes_moved


def evaluate_sla(latency_summary: Dict[str, float], sla: Optional[Dict[str, float]]) -> Dict[str, Any]:
    if not sla:
        return {"enabled": False}
    p99_limit = sla.get("p99_latency_ms")
    p99 = latency_summary.get("p99", float("nan"))
    if p99_limit is None:
        passed = True
    elif p99 != p99:  # NaN
        passed = False
    else:
        passed = p99 <= float(p99_limit)
    return {
        "enabled": True,
        "p99_latency_ms_limit": p99_limit,
        "p99_latency_ms": p99,
        "passed": bool(passed),
    }
