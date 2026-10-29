#!/usr/bin/env python3
"""Run the Triton optimization-pass lab examples and export compiler IR."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from ir_utils import (
    dump_hint,
    export_compiled_artifacts,
    prepare_output_dir,
    prepare_triton_cache,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Triton teaching kernels and dump TTIR/TTGIR/LLVM/PTX artifacts."
    )
    parser.add_argument(
        "--kernel",
        choices=("matmul", "softmax", "all"),
        default="all",
        help="Which teaching kernel to run.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Directory where IR artifacts will be written.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Reuse the experiment-local Triton cache instead of clearing it.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Logical GPU index to use. On shared BI-V150 machines, pick an assigned device.",
    )
    parser.add_argument(
        "--enable-mlir-dump",
        action="store_true",
        help="Set MLIR_ENABLE_DUMP=1 before importing Triton. This prints pass-by-pass MLIR to stderr.",
    )
    parser.add_argument(
        "--enable-llvm-dump",
        action="store_true",
        help="Set LLVM_IR_ENABLE_DUMP=1 before importing Triton. This prints LLVM pass dumps to stderr.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Also run a short Triton benchmark after correctness checks.",
    )

    matmul = parser.add_argument_group("matmul options")
    matmul.add_argument("--m", type=int, default=512)
    matmul.add_argument("--n", type=int, default=512)
    matmul.add_argument("--k", type=int, default=512)
    matmul.add_argument("--block-m", type=int, default=64)
    matmul.add_argument("--block-n", type=int, default=64)
    matmul.add_argument("--block-k", type=int, default=32)
    matmul.add_argument("--num-warps", type=int, default=4)
    matmul.add_argument("--num-stages", type=int, default=3)
    matmul.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Input dtype for the matmul example.",
    )

    softmax = parser.add_argument_group("softmax options")
    softmax.add_argument("--softmax-rows", type=int, default=128)
    softmax.add_argument("--softmax-cols", type=int, default=1024)
    softmax.add_argument("--softmax-warps", type=int, default=4)

    return parser.parse_args()


def require_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        import triton
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Use the platform's matching "
            "CoreX PyTorch / Triton environment, then run `python3 check_env.py`."
        ) from exc

    if torch.version.cuda is None:
        raise SystemExit(
            "The current PyTorch build reports `torch.version.cuda is None`. "
            "This runner requires the platform's matching GPU-enabled CoreX build. "
            "Run `python3 check_env.py` to inspect the active environment."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This lab can parse --help without a GPU, "
            "but full IR artifact generation needs the CoreX GPU runtime and matching Triton. "
            "Check the device with ixsmi, CUDA_VISIBLE_DEVICES, and the active Python environment."
        )

    from kernels import matmul_kernel, softmax_kernel

    return torch, triton, (matmul_kernel, softmax_kernel)


def configure_debug_env(args: argparse.Namespace) -> None:
    if args.enable_mlir_dump:
        os.environ["MLIR_ENABLE_DUMP"] = "1"
    if args.enable_llvm_dump:
        os.environ["LLVM_IR_ENABLE_DUMP"] = "1"


def device_metadata(torch: Any) -> dict[str, Any]:
    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
        "multi_processor_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
    }


def launch_and_get_compiled(jit_fn: Any, grid: tuple[int, ...], args: tuple[Any, ...], meta: dict[str, Any]) -> Any:
    compiled = None

    if hasattr(jit_fn, "warmup"):
        compiled = jit_fn.warmup(*args, grid=grid, **meta)

    launch_result = jit_fn[grid](*args, **meta)
    if hasattr(launch_result, "asm"):
        compiled = launch_result

    if compiled is None:
        raise RuntimeError("Could not obtain a Triton CompiledKernel for artifact export.")
    return compiled


def run_matmul(args: argparse.Namespace, torch: Any, triton: Any, kernel: Any) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    a = torch.randn((args.m, args.k), device="cuda", dtype=dtype)
    b = torch.randn((args.k, args.n), device="cuda", dtype=dtype)
    c = torch.empty((args.m, args.n), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(args.m, args.block_m) * triton.cdiv(args.n, args.block_n),)
    kernel_args = (a, b, c, args.m, args.n, args.k)
    meta = {
        "BLOCK_M": args.block_m,
        "BLOCK_N": args.block_n,
        "BLOCK_K": args.block_k,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
    }

    compiled = launch_and_get_compiled(kernel, grid, kernel_args, meta)
    torch.cuda.synchronize()

    ref = a.to(torch.float32) @ b.to(torch.float32)
    max_abs_err = torch.max(torch.abs(c - ref)).item()

    bench_ms = None
    if args.benchmark:
        bench_ms = triton.testing.do_bench(lambda: kernel[grid](*kernel_args, **meta))

    out_dir = prepare_output_dir(args.dump_dir / "matmul")
    exported = export_compiled_artifacts(compiled, out_dir)
    write_run_metadata(
        out_dir,
        {
            "kernel": "matmul",
            "shape": {"M": args.m, "N": args.n, "K": args.k},
            "dtype": args.dtype,
            "meta": meta,
            "grid": grid,
            "max_abs_err": max_abs_err,
            "benchmark_ms": bench_ms,
            "artifact_stages": sorted(exported),
            "compiled_metadata": repr(getattr(compiled, "metadata", None)),
            "device": device_metadata(torch),
        },
    )

    return {"kernel": "matmul", "out_dir": out_dir, "max_abs_err": max_abs_err, "benchmark_ms": bench_ms}


def run_softmax(args: argparse.Namespace, torch: Any, triton: Any, kernel: Any) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    n_rows = args.softmax_rows
    n_cols = args.softmax_cols
    block_n = triton.next_power_of_2(n_cols)

    x = torch.randn((n_rows, n_cols), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (n_rows,)
    kernel_args = (x, y, n_rows, n_cols)
    meta = {"BLOCK_N": block_n, "num_warps": args.softmax_warps}

    compiled = launch_and_get_compiled(kernel, grid, kernel_args, meta)
    torch.cuda.synchronize()

    ref = torch.softmax(x, dim=1)
    max_abs_err = torch.max(torch.abs(y - ref)).item()

    bench_ms = None
    if args.benchmark:
        bench_ms = triton.testing.do_bench(lambda: kernel[grid](*kernel_args, **meta))

    out_dir = prepare_output_dir(args.dump_dir / "softmax")
    exported = export_compiled_artifacts(compiled, out_dir)
    write_run_metadata(
        out_dir,
        {
            "kernel": "softmax",
            "shape": {"rows": n_rows, "cols": n_cols},
            "meta": meta,
            "grid": grid,
            "max_abs_err": max_abs_err,
            "benchmark_ms": bench_ms,
            "artifact_stages": sorted(exported),
            "compiled_metadata": repr(getattr(compiled, "metadata", None)),
            "device": device_metadata(torch),
        },
    )

    return {"kernel": "softmax", "out_dir": out_dir, "max_abs_err": max_abs_err, "benchmark_ms": bench_ms}


def main() -> int:
    args = parse_args()
    configure_debug_env(args)
    torch, triton, kernels = require_runtime()
    torch.cuda.set_device(args.device_index)
    prepare_triton_cache(args.dump_dir, clear_cache=not args.keep_cache)
    matmul_kernel, softmax_kernel = kernels

    dev = device_metadata(torch)
    print(f"Device[{dev['index']}]: {dev['name']} capability={dev['capability'][0]}.{dev['capability'][1]} ({dev['multi_processor_count']} compute units)")
    print(f"Triton: {triton.__version__}")
    print(dump_hint())

    results = []
    if args.kernel in ("matmul", "all"):
        results.append(run_matmul(args, torch, triton, matmul_kernel))
    if args.kernel in ("softmax", "all"):
        results.append(run_softmax(args, torch, triton, softmax_kernel))

    for result in results:
        bench = ""
        if result["benchmark_ms"] is not None:
            bench = f", benchmark={result['benchmark_ms']:.3f} ms"
        print(
            f"{result['kernel']}: artifacts={result['out_dir']}, "
            f"max_abs_err={result['max_abs_err']:.6g}{bench}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
