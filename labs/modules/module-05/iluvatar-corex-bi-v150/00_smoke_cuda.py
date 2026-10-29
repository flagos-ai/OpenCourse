#!/usr/bin/env python3
"""Minimal CUDA smoke test for FlagOS / BI-V150 (run before 01_benchmark)."""

from __future__ import annotations

import os
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.corex_env import needs_reexec_for_corex, prepare_corex_environ
from tools.iluvatar_device import describe_device_nodes, maybe_clear_bad_cvd_hint


def _reexec(env: dict) -> None:
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def main() -> int:
    # LD_PRELOAD must be set before process start — re-exec once if needed.
    if needs_reexec_for_corex():
        env, notes = prepare_corex_environ()
        env["SMOKE_COREX_REEXEC"] = "1"
        print(f"[smoke] re-exec with CoreX env: {'; '.join(notes)}", flush=True)
        _reexec(env)

    print(f"[smoke] device nodes: {describe_device_nodes()}", flush=True)
    print(
        f"[smoke] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
        flush=True,
    )
    hint = maybe_clear_bad_cvd_hint()
    if hint:
        print(f"[smoke] WARN: {hint}", flush=True)

    print(f"[smoke] COREX_HOME={os.environ.get('COREX_HOME', '')}", flush=True)
    print(f"[smoke] LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}", flush=True)

    print("[smoke] import torch ...", flush=True)
    import torch

    ver = torch.__version__
    path = torch.__file__
    try:
        wheel_ver = version("torch")
    except PackageNotFoundError:
        wheel_ver = ""
    print(f"[smoke] torch={ver}", flush=True)
    print(f"[smoke] torch_file={path}", flush=True)
    print(
        f"[smoke] LD_LIBRARY_PATH(head)={os.environ.get('LD_LIBRARY_PATH', '')[:240]}",
        flush=True,
    )

    in_corex_tree = "corex" in path.replace("\\", "/").lower()
    if in_corex_tree or "corex" in ver.lower() or "corex" in wheel_ver.lower():
        print(f"[smoke] CoreX torch OK (path/tag/wheel; wheel={wheel_ver})", flush=True)
    else:
        print(
            "[smoke] WARN: torch not under corex path; prefer FlagOS image wheels",
            flush=True,
        )

    print(f"[smoke] cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd and os.environ.get("SMOKE_CVD_RETRY") != "1":
            print(
                f"[smoke] no device with CUDA_VISIBLE_DEVICES={cvd!r}; "
                "retrying once with CVD unset ...",
                flush=True,
            )
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["SMOKE_CVD_RETRY"] = "1"
            _reexec(env)
        print(
            "[smoke] FAIL: no CUDA device. Try: unset CUDA_VISIBLE_DEVICES && "
            "source env_corex.sh",
            flush=True,
        )
        return 1

    print(f"[smoke] device_count={torch.cuda.device_count()}", flush=True)
    print("[smoke] set_device(0) ...", flush=True)
    torch.cuda.set_device(0)

    t0 = time.perf_counter()
    print(
        "[smoke] torch.zeros(1, device='cuda:0')  "
        "(if stuck >30s: Ctrl+C, then run bandwidthTest / local GEMM baseline)",
        flush=True,
    )
    x = torch.zeros(1, device="cuda:0", dtype=torch.float32)
    print("[smoke] synchronize ...", flush=True)
    torch.cuda.synchronize()
    print(f"[smoke] zeros ok device={x.device}", flush=True)

    print(f"[smoke] name={torch.cuda.get_device_name(0)}", flush=True)
    print("[smoke] matmul 64x64 fp16 ...", flush=True)
    a = torch.randn(64, 64, device="cuda:0", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda:0", dtype=torch.float16)
    c = torch.empty(64, 64, device="cuda:0", dtype=torch.float16)
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    print(f"[smoke] OK  c[0,0]={float(c[0, 0]):.4f}  elapsed={dt:.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
