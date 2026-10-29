"""MLPerf-inspired LoadGen runner producing structured JSON results."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    effective_tflops,
    evaluate_sla,
    samples_per_second,
    summarize_latencies_ms,
)
from .scenarios import generate_offline_schedule, generate_server_schedule
from tools.resource_monitor import ResourceMonitor


def _software_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import triton

        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        info["triton"] = None
    return info


def run_benchmark(
    sut: Any,
    scenario: str,
    num_queries: int,
    warmup: int = 5,
    sync: bool = True,
    repeats: int = 1,
    target_qps: float = 8.0,
    arrival: str = "poisson",
    sla: Optional[Dict[str, float]] = None,
    resource_interval_s: float = 0.2,
    enable_resource_monitor: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute Offline or Server scenario against a SUT with run_once()/synchronize().

    Resource monitoring via ixsmi is off by default: concurrent ixsmi while a Triton
    JIT/CUDA workload runs has been observed to stall on some CoreX stacks.
    """
    if num_queries <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("num_queries/repeats must be positive; warmup must be nonnegative")
    if not sync:
        raise ValueError("sync=true is required for completed-work latency; async submission is not measured")
    scenario_norm = scenario.strip().capitalize()
    if scenario_norm == "Offline":
        schedule = generate_offline_schedule(num_queries)
    elif scenario_norm == "Server":
        schedule = generate_server_schedule(num_queries, target_qps=target_qps, arrival=arrival)
    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    print(f"[loadgen] warmup x{warmup}; repeats={repeats}, queries/round={num_queries}", flush=True)
    for _ in range(warmup):
        sut.run_once()
    sut.synchronize()
    monitor = ResourceMonitor(interval_s=resource_interval_s) if enable_resource_monitor else None
    if monitor is not None:
        monitor.start()
    per_query = []
    rounds = []
    resources = {"sample_count": 0, "enabled": False}
    try:
        for repetition in range(repeats):
            sut.synchronize()
            wall_start = time.perf_counter()
            current = []
            for q in schedule:
                target = wall_start + q.issue_time_s
                if scenario_norm == "Server":
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                sut.synchronize()
                started = time.perf_counter()
                sut.run_once()
                sut.synchronize()
                completed = time.perf_counter()
                service = (completed - started) * 1e3
                queue = max(0.0, started - target) * 1e3
                item = {"query_id": q.query_id, "repetition": repetition,
                        "issue_time_s": q.issue_time_s,
                        "start_time_s": started - wall_start,
                        "completion_time_s": completed - wall_start,
                        "latency_ms": service, "service_latency_ms": service,
                        "queue_latency_ms": queue,
                        "response_latency_ms": service + queue}
                current.append(item)
            wall_s = time.perf_counter() - wall_start
            per_query.extend(current)
            rounds.append({"repetition": repetition, "wall_time_s": wall_s,
                           "service_latency_ms": summarize_latencies_ms([q["service_latency_ms"] for q in current]),
                           "response_latency_ms": summarize_latencies_ms([q["response_latency_ms"] for q in current])})
            print(f"[loadgen] round {repetition + 1}/{repeats}: {len(current)} queries complete", flush=True)
    finally:
        if monitor is not None:
            resources = monitor.stop()

    summaries = {key: summarize_latencies_ms([q[key] for q in per_query])
                 for key in ("service_latency_ms", "queue_latency_ms", "response_latency_ms")}
    median_ms = summaries["service_latency_ms"]["median"]
    flops = int(sut.flops())
    wall_s = sum(r["wall_time_s"] for r in rounds)
    basis = "response_latency_ms" if scenario_norm == "Server" else "service_latency_ms"
    result = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_norm, "software": _software_info(),
        "sut": sut.config() if hasattr(sut, "config") else {},
        "workload": {"num_queries": num_queries, "repeats": repeats,
                     "total_queries": len(per_query), "warmup": warmup, "sync": True,
                     "target_qps": target_qps if scenario_norm == "Server" else None,
                     "arrival": arrival if scenario_norm == "Server" else None,
                     "flops_per_query": flops,
                     "bytes_per_query": int(sut.bytes_moved()) if hasattr(sut, "bytes_moved") else None},
        "latency_ms": summaries["service_latency_ms"],
        **summaries,
        "throughput": {"samples_per_s": samples_per_second(len(per_query), wall_s),
                       "wall_time_s": wall_s, "effective_tflops": effective_tflops(flops, median_ms),
                       "median_latency_ms": median_ms, "timing_basis": "service_latency_ms"},
        "resources": resources,
        "sla": {**evaluate_sla(summaries[basis], sla), "timing_basis": basis},
        "queries": per_query, "repetitions": rounds,
        "warnings": (["Fewer than 100 samples: p99 is exploratory, not a stable tail estimate"]
                     if len(per_query) < 100 else []),
    }
    if extra:
        result.update(extra)
    return result


def save_result(result: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
