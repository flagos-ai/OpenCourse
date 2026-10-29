#!/usr/bin/env python3
"""Exp 2: Roofline plot and bottleneck diagnosis from Exp 1 results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.student_todos import HBM_BANDWIDTH_GBPS, PEAK_TFLOPS_FP16
from tools.evidence import file_sha256, source_fingerprint, sut_signature
from tools.plot_roofline import (
    arithmetic_intensity,
    diagnose,
    dtype_nbytes,
    gemm_bytes,
    gemm_flops,
    plot_roofline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Module-5 Exp2: Roofline analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "benchmark_offline.json",
        help="benchmark JSON from experiment 01",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=None,
        help="hardware peak parameters JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "roofline.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "results" / "roofline.png",
    )
    args = parser.parse_args()

    args.out_json.unlink(missing_ok=True)
    if not args.input.exists():
        print(f"[error] input not found: {args.input}")
        print("        run: python3 01_benchmark.py --scenario offline")
        return 1

    bench = json.loads(args.input.read_text(encoding="utf-8"))
    hardware_path = args.hardware or ROOT / "configs/hardware_bi_v150.json"
    hw = json.loads(hardware_path.read_text(encoding="utf-8"))
    # An explicit --hardware is authoritative; otherwise use the student TODOs.
    peak = float(hw["peak_tflops_fp16"] if args.hardware else PEAK_TFLOPS_FP16 or hw["peak_tflops_fp16"])
    bw = float(hw["hbm_bandwidth_GBps"] if args.hardware else HBM_BANDWIDTH_GBPS or hw["hbm_bandwidth_GBps"])
    if not all(math.isfinite(x) and x > 0 for x in (peak, bw)):
        raise ValueError("Hardware peaks must be finite and positive")
    if bench.get("schema_version") != 2 or bench.get("source_fingerprint") != source_fingerprint(ROOT):
        raise ValueError("Benchmark is legacy or from different code; rerun 01_benchmark.py")
    matches_mirror = peak == hw["peak_tflops_fp16"] and bw == hw["hbm_bandwidth_GBps"]
    verified = bool(hw.get("peaks_verified") and hw.get("source_reference") and matches_mirror)
    parameter_source = str(hardware_path) if args.hardware else "configs/student_todos.py (JSON fallback)"
    print(f"[info] parameter source={parameter_source}; verified={verified}; mirror_match={matches_mirror}")
    print(f"[info] peak_tflops_fp16={peak}  hbm_bandwidth_GBps={bw}", flush=True)

    wl = bench.get("workload", {})
    M, N, K = int(wl["M"]), int(wl["N"]), int(wl["K"])
    dtype = wl.get("dtype", "float16")
    if dtype not in ("float16", "fp16"):
        raise ValueError("This reference roof is FP16-only; use an FP16 benchmark")
    # If CPU fallback promoted to fp32, prefer sut dtype when present.
    nbytes = dtype_nbytes(dtype)
    flops = gemm_flops(M, N, K)

    if not bench.get("comparison"):
        raise ValueError("Benchmark comparison is empty")
    points = []
    diagnoses = []
    for item in bench.get("comparison", []):
        name = item.get("resolved") or item.get("requested") or "unknown"
        sut = item.get("sut") or {}
        if sut.get("device", "").startswith("cpu") or "fallback" in name:
            raise ValueError("GPU Roofline cannot use CPU/fallback results")
        sut_dtype = sut.get("dtype", dtype)
        try:
            nb = dtype_nbytes(sut_dtype)
        except ValueError:
            nb = nbytes
        bytes_moved = gemm_bytes(M, N, K, nb)
        intensity = arithmetic_intensity(flops, bytes_moved)
        tflops = float(item.get("throughput", {}).get("effective_tflops") or float("nan"))
        diag = diagnose(intensity, tflops, peak, bw)
        diag["name"] = name
        diag["sut_signature"] = sut_signature(sut)
        diag["tile"] = sut.get("tile")
        diagnoses.append(diag)
        points.append({"name": name, "intensity": intensity, "tflops": tflops})
        print(f"[{name}] I={intensity:.2f} FLOP/Byte  measured={tflops:.3f} TFLOP/s")
        print(f"         {diag['conclusion']}")

    png = plot_roofline(
        points,
        peak_tflops=peak,
        bandwidth_GBps=bw,
        out_path=args.out_png,
        title=f"Roofline GEMM {M}x{N}x{K} - BI-V150",
    )

    payload = {
        "experiment": "02_roofline",
        "schema_version": 2,
        "benchmark_id": bench["benchmark_id"],
        "benchmark_sha256": file_sha256(args.input),
        "analysis_source_sha256": file_sha256(ROOT / "02_roofline.py") + file_sha256(ROOT / "tools/plot_roofline.py"),
        "input": str(args.input),
        "hardware": {
            "name": hw.get("name"),
            "arch": hw.get("arch"),
            "peak_tflops_fp16": peak,
            "hbm_bandwidth_GBps": bw,
            "source_json": str(hardware_path.resolve()),
            "hardware_sha256": file_sha256(hardware_path),
            "student_todos_sha256": None if args.hardware else file_sha256(ROOT / "configs/student_todos.py"),
            "parameter_source": parameter_source,
            "json_mirror_matches": matches_mirror,
            "peaks_verified": verified,
            "source_reference": hw.get("source_reference", ""),
        },
        "workload": {"M": M, "N": N, "K": K, "dtype": dtype, "flops": flops},
        "points": points,
        "diagnoses": diagnoses,
        "plot": str(png) if png else None,
        "plot_skipped": png is None,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    if png:
        print(f"[done] wrote {png}")
    else:
        print("[done] roofline JSON written without PNG (install matplotlib to plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
