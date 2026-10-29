#!/usr/bin/env python3
"""Step 00: environment check for Module-5 (with integrated Lab1 GEMM)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Module-5 env check")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow missing CUDA (local smoke); BI-V150 should omit this flag",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings (ixsmi / ixsys / ixkn-cli) as failures",
    )
    args = parser.parse_args()

    checks: List[Dict[str, Any]] = []
    hard_fail = False
    warn_count = 0

    def record(name: str, status: str, detail: str) -> None:
        nonlocal hard_fail, warn_count
        checks.append({"name": name, "status": status, "detail": detail})
        if status == "fail":
            hard_fail = True
            _fail(f"{name}: {detail}")
        elif status == "warn":
            warn_count += 1
            _warn(f"{name}: {detail}")
        else:
            _ok(f"{name}: {detail}")

    # --- Python ---
    record("python", "ok", platform.python_version())

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.iluvatar_device import describe_device_nodes, maybe_clear_bad_cvd_hint

    cvd_hint = maybe_clear_bad_cvd_hint()
    record(
        "iluvatar_nodes",
        "warn" if cvd_hint else "ok",
        cvd_hint
        or f"{describe_device_nodes()}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
    )

    # --- torch ---
    try:
        import torch

        record("torch", "ok", torch.__version__)
    except Exception as exc:
        record("torch", "fail", f"import failed: {exc}")
        torch = None  # type: ignore

    # --- CUDA ---
    cuda_ok = False
    device_name = None
    if torch is not None:
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            torch.cuda.set_device(0)
            device_name = torch.cuda.get_device_name(0)
            record("cuda", "ok", f"available; selected cuda:0 ({device_name})")
        elif args.allow_cpu:
            record("cuda", "warn", "not available; continuing with --allow-cpu")
        else:
            record(
                "cuda",
                "fail",
                "BI-V150/CoreX device is not visible to PyTorch "
                "(pass --allow-cpu for local CPU smoke only)",
            )

    # --- triton ---
    triton_ver = None
    try:
        import triton

        triton_ver = getattr(triton, "__version__", "unknown")
        record("triton", "ok", triton_ver)
    except Exception as exc:
        status = "fail" if (args.strict or not args.allow_cpu) else "warn"
        record("triton", status, f"import failed: {exc}")

    # --- matplotlib (FlagOS images often omit it; needed only for Roofline PNG) ---
    try:
        import matplotlib  # noqa: F401

        record("matplotlib", "ok", getattr(matplotlib, "__version__", "unknown"))
    except Exception as exc:
        record(
            "matplotlib",
            "warn",
            f"import failed: {exc} — Roofline JSON still works; "
            "for PNG: pip install matplotlib",
        )

    # --- CoreX hint: runtime tag, wheel metadata, or install path ---
    try:
        torch_wheel_version = version("torch")
    except PackageNotFoundError:
        torch_wheel_version = ""
    torch_file = ""
    if torch is not None:
        torch_file = str(getattr(torch, "__file__", "") or "")
    ver_blob = " ".join(
        [
            getattr(torch, "__version__", "") if torch is not None else "",
            str(triton_ver or ""),
            torch_file,
            torch_wheel_version,
        ]
    ).lower()
    if "corex" in ver_blob:
        record(
            "corex_wheel",
            "ok",
            "CoreX torch detected via version, wheel metadata, or install path "
            f"(torch={getattr(torch, '__version__', None)}, "
            f"wheel={torch_wheel_version}, file={torch_file})",
        )
    else:
        status = "warn" if args.allow_cpu else "fail"
        record(
            "corex_wheel",
            status,
            "No CoreX marker in runtime version, wheel metadata, or install path. "
            "Avoid PyPI torch; use the FlagOS CoreX image.",
        )

    # In-process tiny alloc with alarm (never fork after CUDA init).
    if torch is not None and cuda_ok and not args.allow_cpu:
        import signal

        def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
            raise TimeoutError("cuda alloc timed out")

        try:
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(45)
            torch.cuda.set_device(0)
            x = torch.empty(1, device="cuda:0", dtype=torch.float32)
            torch.cuda.synchronize()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            record("cuda_alloc", "ok", f"empty(1) on {x.device}")
        except TimeoutError:
            record(
                "cuda_alloc",
                "fail",
                "empty(1) exceeded 45s — try: source env_corex.sh "
                "(sets LD_PRELOAD=libcuda); or retry 00_smoke_cuda.py in a fresh shell",
            )
        except Exception as exc:
            record("cuda_alloc", "fail", str(exc))

    # --- hardware config ---
    hw_path = ROOT / "configs" / "hardware_bi_v150.json"
    if hw_path.is_file():
        try:
            hw = json.loads(hw_path.read_text(encoding="utf-8"))
            record(
                "hardware_config",
                "ok",
                f"{hw_path.name} peak_fp16={hw.get('peak_tflops_fp16')} "
                f"bw={hw.get('hbm_bandwidth_GBps')} GB/s",
            )
        except Exception as exc:
            record("hardware_config", "fail", f"unreadable: {exc}")
    else:
        record("hardware_config", "fail", f"missing {hw_path}")

    # --- local GEMM source ---
    gemm_path = ROOT / "kernels" / "gemm_triton.py"
    record("gemm_source", "ok" if gemm_path.is_file() else "fail",
           f"{gemm_path}; integrated from lab1 (6004a08), no external checkout")

    # --- profiling tools ---
    for tool in ("ixsmi", "ixsys", "ixkn-cli"):
        path = shutil.which(tool)
        if path:
            record(tool, "ok", path)
        else:
            record(tool, "warn", "not in PATH (expected on 天数 machines)")

    print()
    print("Do not pip-install/upgrade torch or triton in the FlagOS BI-V150 environment.")
    print("Use platform +corex wheels from the lab image.")

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "allow_cpu": args.allow_cpu,
        "strict": args.strict,
        "cuda_available": cuda_ok,
        "device_name": device_name,
        "checks": checks,
    }
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "env_check.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")

    if hard_fail:
        return 1
    if args.strict and warn_count:
        _fail(f"--strict: {warn_count} warning(s) treated as failure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
