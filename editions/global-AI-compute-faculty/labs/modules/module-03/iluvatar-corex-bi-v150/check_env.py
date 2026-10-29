#!/usr/bin/env python3
"""Verify the preinstalled BI-V150 environment without installing packages."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0, help="Logical GPU index visible to this Python process.")
    parser.add_argument("--report", type=Path, help="Optionally save the check results as JSON.")
    parser.add_argument("--skip-smoke", action="store_true", help="Only check the environment; skip kernel compilation and numerical checks.")
    return parser.parse_args()


def check_environment(device_index: int, skip_smoke: bool = False) -> dict:
    report = {
        "python": sys.executable,
        "python_version": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checks": [],
    }

    def record(name: str, status: str, detail: str) -> None:
        report["checks"].append({"name": name, "status": status, "detail": detail})
        print(f"[{status}] {name}: {detail}", flush=True)

    print(f"Python: {sys.executable}\nVersion: {sys.version.split()[0]}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {report['cuda_visible_devices']!r}", flush=True)
    tool = shutil.which("ixsmi")
    if tool:
        try:
            proc = subprocess.run([tool, "-L"], text=True, capture_output=True, timeout=20, check=False)
            detail = (proc.stdout + proc.stderr).strip()
            record("ixsmi", "PASS" if proc.returncode == 0 else "WARN", detail or f"exit_code={proc.returncode}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            record("ixsmi", "WARN", str(exc))
    else:
        record("ixsmi", "WARN", "Not found on PATH; verify the platform's tool path if needed.")

    modules = {}
    for name in ("torch", "triton", "numpy"):
        try:
            module = importlib.import_module(name)
            modules[name] = module
            record(name, "PASS", f"version={getattr(module, '__version__', 'unknown')}; path={getattr(module, '__file__', 'unknown')}")
        except Exception as exc:
            record(name, "FAIL", f"{type(exc).__name__}: {exc}. Use the platform's matching CoreX environment.")
    if len(modules) != 3:
        report["ok"] = False
        return report

    torch, triton = modules["torch"], modules["triton"]
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        record("GPU runtime", "PASS" if available else "FAIL", f"available={available}; logical_device_count={count}; torch.version.cuda={torch.version.cuda}")
        if not available:
            report["ok"] = False
            return report
        if not 0 <= device_index < count:
            record("device index", "FAIL", f"Requested {device_index}; valid logical indices are 0..{count - 1}.")
            report["ok"] = False
            return report
        torch.cuda.set_device(device_index)
        name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        normalized_name = re.sub(r"[^a-z0-9]", "", name.lower())
        is_v150 = "v150" in normalized_name
        report["device"] = {
            "index": device_index, "name": name,
            "memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
        }
        record("BI-V150 device", "PASS" if is_v150 else "FAIL", f"index={device_index}; name={name}; memory={props.total_memory / 2**30:.1f} GiB")
        if not is_v150:
            report["ok"] = False
            return report
        # Match the existing runner's gate without changing its behavior.
        record("run_lab runtime gate", "PASS" if torch.version.cuda is not None else "FAIL",
               f"torch.version.cuda={torch.version.cuda}; the current runner requires a non-None value.")
        target = triton.runtime.driver.active.get_current_target()
        report["triton_target"] = repr(target)
        warp_size = getattr(target, "warp_size", None)
        record("Triton target", "PASS", repr(target))
        if warp_size is not None:
            record("warp size", "PASS" if warp_size == 64 else "FAIL", str(warp_size))
        else:
            record("warp size", "WARN", "This backend does not expose target.warp_size; inspect generated TTGIR for the thread layout.")
        if any(item["status"] == "FAIL" for item in report["checks"]):
            report["ok"] = False
            return report

        if skip_smoke:
            record("kernel smoke test", "SKIP", "Not requested; kernel compilation and correctness remain unverified.")
        else:
            # Keep the compiler cache out of the source tree and leave the
            # caller's TRITON_CACHE_DIR setting unchanged after verification.
            previous_cache = os.environ.get("TRITON_CACHE_DIR")
            try:
                with tempfile.TemporaryDirectory(prefix="lab3-v150-check-") as cache:
                    os.environ["TRITON_CACHE_DIR"] = cache
                    from kernels import matmul_kernel, softmax_kernel
                    from run_lab import launch_and_get_compiled

                    torch.manual_seed(0)
                    m, n, k = 127, 95, 65
                    a_cpu = torch.randn((m, k), dtype=torch.float32).half()
                    b_cpu = torch.randn((k, n), dtype=torch.float32).half()
                    a, b = a_cpu.to("cuda"), b_cpu.to("cuda")
                    c = torch.full((m, n), float("nan"), device="cuda", dtype=torch.float32)
                    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 64),)
                    compiled = launch_and_get_compiled(matmul_kernel, grid, (a, b, c, m, n, k),
                        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 3})
                    torch.cuda.synchronize()
                    actual = c.cpu().double()
                    expected = a_cpu.double() @ b_cpu.double()
                    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3, equal_nan=False)
                    record("matmul compile and correctness", "PASS", f"shape=127x95x65; max_abs_err={(actual - expected).abs().max().item():.6g}; atol=1e-3; rtol=1e-3")
                    stages = sorted(getattr(compiled, "asm", {}) or {})
                    record("matmul artifact stages", "PASS" if stages else "FAIL", ", ".join(stages) or "compiled.asm is empty")

                    rows, cols = 17, 1003
                    x_cpu = torch.randn((rows, cols), dtype=torch.float32)
                    x = x_cpu.to("cuda")
                    y = torch.full_like(x, float("nan"))
                    launch_and_get_compiled(softmax_kernel, (rows,), (x, y, rows, cols),
                        {"BLOCK_N": triton.next_power_of_2(cols), "num_warps": 4})
                    torch.cuda.synchronize()
                    actual = y.cpu().double()
                    expected = torch.softmax(x_cpu.double(), dim=1)
                    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5, equal_nan=False)
                    record("softmax compile and correctness", "PASS", f"shape=17x1003; max_abs_err={(actual - expected).abs().max().item():.6g}; atol=1e-6; rtol=1e-5")
            finally:
                if previous_cache is None:
                    os.environ.pop("TRITON_CACHE_DIR", None)
                else:
                    os.environ["TRITON_CACHE_DIR"] = previous_cache
    except Exception:
        record("GPU verification", "FAIL", traceback.format_exc())

    report["ok"] = not any(item["status"] == "FAIL" for item in report["checks"])
    return report


def main() -> int:
    args = parse_args()
    report = check_environment(args.device_index, args.skip_smoke)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report: {args.report}", flush=True)
    if report["ok"]:
        print("Environment checks passed." if args.skip_smoke else "Environment and kernel smoke checks passed.")
        return 0
    print("Environment check failed. Review FAIL entries and the platform's CoreX configuration.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
